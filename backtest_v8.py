#!/usr/bin/env python3
"""
HUNTER-V4 FULL — XT USDT-M Futures collector + HTF Liquidity/Market Structure backtest.
Self-contained GitHub Actions script.
"""
from __future__ import annotations
import argparse, json, time, math, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import requests

def now_utc():
    return datetime.now(timezone.utc)


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def api_json(session, url, params=None):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"unexpected response type: {type(payload).__name__}")

            rc = payload.get("returnCode")
            if rc not in (None, 0, "0"):
                err = payload.get("error") or payload.get("msgInfo") or payload
                raise RuntimeError(f"XT API error: {err}")

            return payload
        except Exception as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(min(2 * attempt, 5))

    raise RuntimeError(f"API request failed after {RETRIES} attempts: {last}")


def extract_list(payload):
    result = payload.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "list", "data", "symbols"):
            if isinstance(result.get(key), list):
                return result[key]
    raise RuntimeError(f"Could not find symbol list in response: {payload}")


def discover_symbols(session):
    payload = api_json(session, SYMBOL_LIST_URL)
    items = extract_list(payload)

    found = {}
    for item in items:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = (
                item.get("symbol")
                or item.get("s")
                or item.get("name")
                or item.get("pair")
            )
        else:
            continue

        if raw:
            found[str(raw).strip().upper()] = item

    if not found:
        raise RuntimeError("XT symbol/list returned zero usable symbols")

    return found


def resolve_symbol(asset, discovered):
    wanted = asset.upper()

    candidates = [
        f"{wanted}_USDT",
        f"{wanted}/USDT",
        f"{wanted}-USDT",
        wanted,
    ]

    # First exact/canonical matching.
    for candidate in candidates:
        if candidate.upper() in discovered:
            return candidate.upper()

    # Then normalize separators for safety.
    def norm(x):
        return str(x).upper().replace("/", "_").replace("-", "_")

    matches = [
        actual for actual in discovered
        if norm(actual) == f"{wanted}_USDT"
    ]
    if len(matches) == 1:
        return matches[0]

    return None


def kline_rows(payload):
    result = payload.get("result")
    if not isinstance(result, list):
        raise RuntimeError(f"XT Kline result is not a list: {payload}")
    return result


def normalize_rows(rows, symbol):
    records = []

    for row in rows:
        if isinstance(row, dict):
            # Official XT fields.
            keys = ("t", "o", "h", "l", "c", "a")
            if not all(k in row for k in keys):
                raise RuntimeError(f"{symbol}: malformed dict kline: {row}")
            records.append({
                "Timestamp": int(row["t"]),
                "Open": float(row["o"]),
                "High": float(row["h"]),
                "Low": float(row["l"]),
                "Close": float(row["c"]),
                "Volume": float(row["a"]),
                "Turnover": float(row["v"]) if row.get("v") is not None else float("nan"),
            })

        elif isinstance(row, (list, tuple)):
            # Defensive support if XT returns array rows.
            if len(row) < 6:
                raise RuntimeError(f"{symbol}: malformed array kline: {row}")
            records.append({
                "Timestamp": int(row[0]),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
                "Turnover": float(row[6]) if len(row) > 6 else float("nan"),
            })
        else:
            raise RuntimeError(f"{symbol}: unsupported kline row: {row!r}")

    df = pd.DataFrame(records)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    return df


def fetch_batch(session, symbol, start_ms, end_ms):
    # XT Futures Kline API uses the exchange market id in lowercase (e.g. btc_usdt),
    # while symbol discovery may return the same id in uppercase (BTC_USDT).
    # XT's API is case-sensitive here.
    api_symbol = symbol.strip().lower()
    params = {
        "symbol": api_symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }

    payload = api_json(session, KLINE_URL, params)
    return normalize_rows(kline_rows(payload), symbol)


def audit(df, symbol):
    if df.empty:
        raise RuntimeError(f"{symbol}: empty dataset")

    bad = (
        (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["Volume"] < 0)
        | (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    if bad.any():
        raise RuntimeError(f"{symbol}: {int(bad.sum())} invalid OHLCV rows")

    diffs = df["Date"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=15)]
    wrong = diffs[diffs != pd.Timedelta(minutes=15)]

    return {
        "rows": int(len(df)),
        "start": df["Date"].iloc[0].isoformat(),
        "end": df["Date"].iloc[-1].isoformat(),
        "duplicates": int(df["Date"].duplicated().sum()),
        "gaps": int(len(gaps)),
        "max_gap_minutes": int(gaps.max().total_seconds() / 60) if len(gaps) else 15,
        "wrong_intervals": int(len(wrong)),
    }


def download_symbol(session, symbol, start_dt, end_dt):
    # IMPORTANT: XT returns the newest candles first when a wide time range is
    # supplied. Therefore forward pagination (cursor -> last candle) loops back
    # into the same 1500-candle page. We paginate BACKWARD instead:
    # first request = [start, end], then [start, first_timestamp - 1].
    start_ms = to_ms(start_dt)
    cursor_end = to_ms(end_dt)
    all_batches = []
    calls = 0
    interval_ms = 15 * 60 * 1000

    while cursor_end >= start_ms:
        batch = fetch_batch(session, symbol, start_ms, cursor_end)
        calls += 1

        if batch.empty:
            break

        batch = batch.sort_values("Date").reset_index(drop=True)
        first = int(batch["Timestamp"].iloc[0])
        last = int(batch["Timestamp"].iloc[-1])

        if first < start_ms:
            batch = batch[batch["Timestamp"] >= start_ms].copy()
            if batch.empty:
                break
            first = int(batch["Timestamp"].iloc[0])
            last = int(batch["Timestamp"].iloc[-1])

        if last > cursor_end:
            raise RuntimeError(
                f"{symbol}: API returned candle beyond requested end: "
                f"last={last}, cursor_end={cursor_end}"
            )

        all_batches.append(batch)

        print(
            f"  {symbol}: request={calls:02d} "
            f"rows={len(batch):4d} "
            f"{batch['Date'].iloc[0]} -> {batch['Date'].iloc[-1]}"
        )

        # We reached the beginning of the requested historical period.
        if first <= start_ms:
            break

        # Move strictly backward. Using first-1 avoids inclusive-boundary
        # duplication if XT treats endTime as inclusive.
        next_end = first - 1
        if next_end >= cursor_end:
            raise RuntimeError(
                f"{symbol}: backward pagination made no progress: "
                f"first={first}, previous_end={cursor_end}"
            )

        cursor_end = next_end

        if calls > 1000:
            raise RuntimeError(f"{symbol}: pagination safety stop")

        time.sleep(SLEEP)

    if not all_batches:
        raise RuntimeError(f"{symbol}: zero historical candles")

    df = pd.concat(all_batches, ignore_index=True)
    df = (
        df.drop_duplicates(subset=["Date"], keep="last")
          .sort_values("Date")
          .reset_index(drop=True)
    )

    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()

    # Remove current, still-forming 15m candle.
    current = pd.Timestamp.now(tz="UTC")
    if not df.empty and df["Date"].iloc[-1] + pd.Timedelta(minutes=15) > current:
        df = df.iloc[:-1].copy()

    report = audit(df, symbol)
    report["api_requests"] = calls
    return df, report


import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

SYMBOLS = ["BTC","ETH","SOL","SUI","AVAX","NEAR","ADA","BNB","APT","CRV","ONDO","PENDLE","ICP","WIF"]
CLUSTERS = {
    "BTC":"MAJOR","ETH":"MAJOR",
    "SOL":"L1","SUI":"L1","AVAX":"L1","NEAR":"L1","ADA":"L1","BNB":"L1","APT":"L1",
    "CRV":"DEFI","ONDO":"DEFI","PENDLE":"DEFI",
    "ICP":"OTHER","WIF":"MEME",
}
DATA_DIR = Path("data/xt_futures_15m")
OUT_DIR = DATA_DIR / "backtest_v4"
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
MARGIN = 100.0
LEVERAGE = 50.0
RR = 2.0
INITIAL_EQUITY = 1000.0
MAX_POSITIONS = 3
MAX_ONE_PER_CLUSTER = True
MAX_HOLD_15M = 24 * 4  # safety horizon for simulation only; NOT a timeout exit
MIN_STOP_PCT = 0.001
MAX_STOP_PCT = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def load_csv(data_dir: Path, asset: str) -> pd.DataFrame:
    path = data_dir / f"{asset}_USDT_15m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing XT Futures data file: {path}")
    df = pd.read_csv(path)
    if {"Date","Open","High","Low","Close","Volume"}.issubset(df.columns):
        df = df.rename(columns={"Date":"timestamp","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    required = ["timestamp","open","high","low","close","volume"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"{path}: missing columns {set(required)-set(df.columns)}")
    if np.issubdtype(df["timestamp"].dtype, np.number):
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(timestamp=ts).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required[1:])
    # Keep only exact 15m grid. Gaps are allowed, but never bridged.
    bad_grid = df.index.minute % 15 != 0
    if bad_grid.any():
        raise ValueError(f"{asset}: {int(bad_grid.sum())} timestamps are not aligned to 15m")
    return df


def split_segments(df: pd.DataFrame):
    if len(df) < 2:
        return [df]
    delta = df.index.to_series().diff().dt.total_seconds().div(60)
    cuts = np.flatnonzero(delta.to_numpy() > 15.0001)
    starts = [0] + (cuts).tolist()
    ends = (cuts).tolist() + [len(df)]
    return [df.iloc[s:e].copy() for s,e in zip(starts, ends) if e-s >= 20]


def ohlcv_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.resample(rule, label="right", closed="right", origin="epoch").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    )
    counts = df["close"].resample(rule, label="right", closed="right", origin="epoch").count()
    expected = 16 if rule == "4h" else 4
    x = x[counts == expected].dropna()
    return x


def atr(df, n=14):
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low),(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rolling_poc(df: pd.DataFrame, n=48, bins=24):
    """Approximate volume-profile POC from completed prior bars only."""
    vals = np.full(len(df), np.nan)
    tp = ((df.high + df.low + df.close) / 3).to_numpy()
    vol = df.volume.to_numpy()
    for i in range(n, len(df)):
        prices = tp[i-n:i]
        vols = vol[i-n:i]
        lo, hi = float(np.nanmin(prices)), float(np.nanmax(prices))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi <= lo:
            vals[i] = prices[-1]
            continue
        edges = np.linspace(lo, hi, bins+1)
        idx = np.clip(np.digitize(prices, edges)-1, 0, bins-1)
        sums = np.bincount(idx, weights=np.nan_to_num(vols), minlength=bins)
        k = int(np.argmax(sums))
        vals[i] = (edges[k] + edges[k+1]) / 2
    return pd.Series(vals, index=df.index)


def add_4h_context(h4: pd.DataFrame):
    x = h4.copy()
    x["atr"] = atr(x,14)
    # Confirmed pivots: pivot at t becomes usable only 2 completed bars later.
    left = right = 2
    ph = (x.high == x.high.rolling(left+right+1, center=True).max())
    pl = (x.low == x.low.rolling(left+right+1, center=True).min())
    x["pivot_high"] = x.high.where(ph).shift(right)
    x["pivot_low"] = x.low.where(pl).shift(right)
    x["last_ph"] = x["pivot_high"].ffill()
    x["last_pl"] = x["pivot_low"].ffill()
    x["structure_bull"] = x.close > x["last_ph"].shift(1)
    x["structure_bear"] = x.close < x["last_pl"].shift(1)
    # External liquidity levels are all prior-period values; no current-day/week leakage.
    x["pdh"] = x.high.shift(1).resample("1D", label="right", closed="right").transform("max") if False else np.nan
    # Daily/weekly levels are computed directly from timestamps, then shifted one period.
    d = x.resample("1D", label="right", closed="right").agg({"high":"max","low":"min"}).shift(1)
    w = x.resample("1W", label="right", closed="right").agg({"high":"max","low":"min"}).shift(1)
    x["pdh"] = d["high"].reindex(x.index, method="ffill")
    x["pdl"] = d["low"].reindex(x.index, method="ffill")
    x["pwh"] = w["high"].reindex(x.index, method="ffill")
    x["pwl"] = w["low"].reindex(x.index, method="ffill")
    # Premium/discount in latest confirmed dealing range.
    x["range_hi"] = x["last_ph"]
    x["range_lo"] = x["last_pl"]
    mid = (x.range_hi + x.range_lo) / 2
    x["discount"] = x.close < mid
    x["premium"] = x.close > mid
    return x


def add_1h_setup(h1: pd.DataFrame):
    x = h1.copy()
    x["atr"] = atr(x,14)
    x["body"] = (x.close-x.open).abs()
    x["range"] = x.high-x.low
    x["avg_body"] = x.body.shift(1).rolling(20,min_periods=20).mean()
    x["avg_range"] = x.range.shift(1).rolling(20,min_periods=20).mean()
    x["rvol"] = x.volume / x.volume.shift(1).rolling(20,min_periods=20).mean()
    x["prior20_low"] = x.low.shift(1).rolling(20,min_periods=20).min()
    x["prior20_high"] = x.high.shift(1).rolling(20,min_periods=20).max()
    x["prior10_high"] = x.high.shift(1).rolling(10,min_periods=10).max()
    x["prior10_low"] = x.low.shift(1).rolling(10,min_periods=10).min()
    x["poc"] = rolling_poc(x,48,24)
    # FVG: three-candle imbalance, confirmed only after the current 1H close.
    x["bull_fvg_low"] = np.where(x.low > x.high.shift(2), x.high.shift(2), np.nan)
    x["bull_fvg_high"] = np.where(x.low > x.high.shift(2), x.low, np.nan)
    x["bear_fvg_low"] = np.where(x.high < x.low.shift(2), x.high, np.nan)
    x["bear_fvg_high"] = np.where(x.high < x.low.shift(2), x.low.shift(2), np.nan)
    # Sweep + displacement + BOS. All references are prior completed bars.
    x["long_sweep"] = (x.low < x.prior20_low) & (x.close > x.prior20_low)
    x["short_sweep"] = (x.high > x.prior20_high) & (x.close < x.prior20_high)
    x["long_disp"] = (x.close > x.open) & (x.body >= 1.5*x.avg_body) & (x.range >= 1.25*x.avg_range) & (x.rvol >= 1.15)
    x["short_disp"] = (x.close < x.open) & (x.body >= 1.5*x.avg_body) & (x.range >= 1.25*x.avg_range) & (x.rvol >= 1.15)
    x["long_bos"] = x.close > x.prior10_high
    x["short_bos"] = x.close < x.prior10_low
    x["long_core"] = x.long_sweep & x.long_disp & x.long_bos
    x["short_core"] = x.short_sweep & x.short_disp & x.short_bos
    return x


def map_htf_to_15m(base: pd.DataFrame, htf: pd.DataFrame):
    return pd.merge_asof(base.sort_index(), htf.sort_index(), left_index=True, right_index=True, direction="backward", allow_exact_matches=True)


def add_micro_trigger(seg: pd.DataFrame):
    x=seg.copy()
    x["micro_high"] = x.high.shift(1).rolling(3,min_periods=3).max()
    x["micro_low"] = x.low.shift(1).rolling(3,min_periods=3).min()
    x["micro_bull"] = x.close > x.micro_high
    x["micro_bear"] = x.close < x.micro_low
    x["body"]=(x.close-x.open).abs()
    x["avg_body"]=x.body.shift(1).rolling(12,min_periods=12).mean()
    return x


def generate_candidates(asset: str, df: pd.DataFrame):
    candidates=[]
    for seg in split_segments(df):
        if len(seg)<400: continue
        h1=ohlcv_resample(seg,"1h")
        h4=ohlcv_resample(seg,"4h")
        if len(h1)<100 or len(h4)<80: continue
        h4=add_4h_context(h4)
        h1=add_1h_setup(h1)
        # Map completed HTF state to 15m bars.
        base=add_micro_trigger(seg)
        h1cols=[c for c in h1.columns if c not in seg.columns]
        h4cols=[c for c in h4.columns if c not in seg.columns]
        z=map_htf_to_15m(base,h1[h1cols])
        z=map_htf_to_15m(z,h4[h4cols])
        # 1H core timestamps must be available and not older than 12h.
        h1_time=pd.Series(z.index,index=z.index).dt.floor("h")
        # Reconstruct the active 1H close timestamp from merge_asof index via the nearest <= timestamp.
        # Use explicit setup event series to avoid ambiguous column collisions.
        setup_long = h1.index[h1.long_core.fillna(False)]
        setup_short = h1.index[h1.short_core.fillna(False)]
        for side, setup_idx in (("LONG",setup_long),("SHORT",setup_short)):
            for t in setup_idx:
                row=h1.loc[t]
                if not np.isfinite(row.atr) or row.atr<=0: continue
                # 4H regime/context at the setup close.
                h4rows=h4.loc[:t]
                if h4rows.empty: continue
                c4=h4rows.iloc[-1]
                if side=="LONG":
                    if not bool(c4.get("structure_bull",False)): continue
                    if not bool(c4.get("discount",False)): continue
                    fvg_lo=row.bull_fvg_low; fvg_hi=row.bull_fvg_high
                    if not np.isfinite(fvg_lo) or not np.isfinite(fvg_hi): continue
                    if np.isfinite(row.poc) and row.close < row.poc: pass
                    # AVWAP anchored at setup/sweep bar, computed causally on 1H.
                    window=h1.loc[t:].iloc[:13].copy()
                    cum=(window[(window.high+window.low+window.close)/3]*window.volume).cumsum() / window.volume.cumsum()
                    avwap0=float(cum.iloc[0]) if len(cum) else np.nan
                else:
                    if not bool(c4.get("structure_bear",False)): continue
                    if not bool(c4.get("premium",False)): continue
                    fvg_lo=row.bear_fvg_low; fvg_hi=row.bear_fvg_high
                    if not np.isfinite(fvg_lo) or not np.isfinite(fvg_hi): continue
                    avwap0=float((((row.high+row.low+row.close)/3)*row.volume)/row.volume) if row.volume>0 else np.nan
                # Find 15m trigger after the 1H setup close, within 12 hours, only completed bars.
                start=t + pd.Timedelta(minutes=1)
                end=t + pd.Timedelta(hours=12)
                sub=z.loc[(z.index>=start)&(z.index<=end)].copy()
                if sub.empty: continue
                touched=(sub.low <= fvg_hi) & (sub.high >= fvg_lo)
                if side=="LONG":
                    trigger=touched & sub.micro_bull & (sub.close>sub.open)
                else:
                    trigger=touched & sub.micro_bear & (sub.close<sub.open)
                hits=sub.index[trigger.fillna(False)]
                if len(hits)==0: continue
                et=hits[0] + pd.Timedelta(minutes=15)
                if et not in df.index: continue
                entry_raw=float(df.loc[et,"open"])
                entry=entry_raw*(1+SLIPPAGE) if side=="LONG" else entry_raw*(1-SLIPPAGE)
                if side=="LONG":
                    sl=min(float(row.low), float(fvg_lo))
                    if np.isfinite(c4.last_pl): sl=min(sl,float(c4.last_pl))
                    risk=entry-sl
                    if risk<=0: continue
                    if risk/entry<MIN_STOP_PCT or risk/entry>MAX_STOP_PCT: continue
                    tp=entry+RR*risk
                    # Room to external liquidity; don't enter if 2R is already beyond nearest known resistance.
                    resist=[v for v in [c4.pdh,c4.pwh,c4.last_ph] if np.isfinite(v) and v>entry]
                    if resist and min(resist)<tp: continue
                else:
                    sl=max(float(row.high), float(fvg_hi))
                    if np.isfinite(c4.last_ph): sl=max(sl,float(c4.last_ph))
                    risk=sl-entry
                    if risk<=0: continue
                    if risk/entry<MIN_STOP_PCT or risk/entry>MAX_STOP_PCT: continue
                    tp=entry-RR*risk
                    support=[v for v in [c4.pdl,c4.pwl,c4.last_pl] if np.isfinite(v) and v<entry]
                    if support and max(support)>tp: continue
                candidates.append({"asset":asset,"side":side,"signal_time":hits[0],"entry_time":et,"entry":entry,"sl":sl,"tp":tp,"risk":risk,"setup_time":t,"cluster":CLUSTERS[asset],"avwap_anchor":avwap0})
    # de-duplicate per entry time/side, retain earliest setup.
    uniq={}
    for c in candidates:
        k=(c["entry_time"],c["side"])
        if k not in uniq or c["setup_time"]<uniq[k]["setup_time"]: uniq[k]=c
    return sorted(uniq.values(), key=lambda x:x["entry_time"])


def adverse_entry(p,side): return p*(1+SLIPPAGE) if side=="LONG" else p*(1-SLIPPAGE)
def adverse_exit(p,side): return p*(1-SLIPPAGE) if side=="LONG" else p*(1+SLIPPAGE)


def simulate(c, df):
    et=c["entry_time"]
    pos=df.index.get_loc(et)
    for j in range(pos, len(df)):
        bar=df.iloc[j]
        hi=float(bar.high); lo=float(bar.low)
        if c["side"]=="LONG":
            slhit=lo<=c["sl"]; tphit=hi>=c["tp"]
        else:
            slhit=hi>=c["sl"]; tphit=lo<=c["tp"]
        if slhit and tphit:
            exitp=c["sl"]; outcome="LOSS"; reason="BOTH_SAME_BAR_SL_FIRST"
        elif slhit:
            exitp=c["sl"]; outcome="LOSS"; reason="SL"
        elif tphit:
            exitp=c["tp"]; outcome="WIN"; reason="TP"
        else:
            continue
        exitp=adverse_exit(exitp,c["side"])
        notional=MARGIN*LEVERAGE
        gross=(exitp-c["entry"])/c["entry"]*notional if c["side"]=="LONG" else (c["entry"]-exitp)/c["entry"]*notional
        fees=(abs(c["entry"]*notional/c["entry"])+abs(exitp*notional/c["entry"]))*FEE_RATE
        pnl=gross-fees
        return {**c,"exit_time":df.index[j],"exit":exitp,"outcome":outcome,"reason":reason,"pnl":pnl,"bars_held":j-pos+1}
    return None


def portfolio_filter(trades):
    accepted=[]; active=[]
    for t in sorted(trades,key=lambda x:x["entry_time"]):
        active=[a for a in active if a["exit_time"]>t["entry_time"]]
        if any(a["exit_time"]>=t["entry_time"] for a in active): continue
        if len(active)>=MAX_POSITIONS: continue
        if MAX_ONE_PER_CLUSTER and any(a["cluster"]==t["cluster"] for a in active): continue
        accepted.append(t); active.append(t)
    return accepted


def streaks(outcomes):
    out=[]; n=0
    for o in outcomes:
        if o=="LOSS": n+=1
        elif n: out.append(n); n=0
    if n: out.append(n)
    return out


def backtest_main():
    args=parse_args(); data_dir=Path(args.data_dir); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    all_candidates=[]; reports={}
    frames={}
    for asset in SYMBOLS:
        df=load_csv(data_dir,asset); frames[asset]=df
        gaps=int((df.index.to_series().diff().dt.total_seconds().div(900).fillna(1)-1).clip(lower=0).sum())
        reports[asset]={"rows":len(df),"start":df.index.min().isoformat(),"end":df.index.max().isoformat(),"missing_15m_bars":gaps}
        all_candidates.extend(generate_candidates(asset,df))
    raw=len(all_candidates)
    simulated=[]
    for c in all_candidates:
        t=simulate(c,frames[c["asset"]])
        if t is not None: simulated.append(t)
    accepted=portfolio_filter(simulated)
    trades=pd.DataFrame(accepted)
    if trades.empty:
        trades=pd.DataFrame(columns=["asset","side","entry_time","exit_time","entry","exit","sl","tp","outcome","pnl"])
    outcomes=trades.outcome.tolist() if len(trades) else []
    wins=sum(o=="WIN" for o in outcomes); losses=sum(o=="LOSS" for o in outcomes)
    gross_win=float(trades.loc[trades.outcome=="WIN","pnl"].sum()) if len(trades) else 0.0
    gross_loss=abs(float(trades.loc[trades.outcome=="LOSS","pnl"].sum())) if len(trades) else 0.0
    pf=gross_win/gross_loss if gross_loss else math.inf
    equity=INITIAL_EQUITY; peak=equity; maxdd=0.0; curve=[]
    for p in trades.pnl.tolist() if len(trades) else []:
        equity+=float(p); peak=max(peak,equity); maxdd=min(maxdd,equity-peak); curve.append(equity)
    ls=streaks(outcomes)
    start=min((v["start"] for v in reports.values()),default=None); end=max((v["end"] for v in reports.values()),default=None)
    days=(pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/86400 if start and end else 0
    per={}
    for a in SYMBOLS:
        q=trades[trades.asset==a] if len(trades) else trades
        n=len(q); w=int((q.outcome=="WIN").sum()) if n else 0
        per[a]={"trades":n,"win_rate":(w/n*100 if n else 0),"pnl":float(q.pnl.sum()) if n else 0}
    report={"strategy":"HUNTER-V4 HTF Liquidity & Market Structure","raw_candidates":raw,"closed_trades":len(trades),"wins":wins,"losses":losses,"win_rate":(wins/len(trades)*100 if len(trades) else 0),"profit_factor":pf,"net_pnl":float(trades.pnl.sum()) if len(trades) else 0.0,"final_equity":equity,"max_drawdown":maxdd,"max_loss_streak":max(ls) if ls else 0,"loss_streak_list":ls,"trades_per_day":len(trades)/days if days else 0,"open_at_end":0,"data_audit":reports,"per_symbol":per}
    trades.to_csv(out_dir/"TRADES.csv",index=False)
    (out_dir/"BACKTEST_REPORT.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print("="*88); print("HUNTER-V4 — XT USDT-M FUTURES / 15m / 365-DAY BACKTEST"); print("="*88)
    for k,v in [("Raw candidates",raw),("Accepted closed trades",len(trades)),("Wins / Losses",f"{wins} / {losses}"),("Win Rate",f"{report['win_rate']:.2f}%"),("Profit Factor",f"{pf:.3f}"),("Net PnL",f"${report['net_pnl']:.2f}"),("Final equity",f"${equity:.2f}"),("Max Drawdown",f"${maxdd:.2f}"),("Max loss streak",report['max_loss_streak']),("Trades / day",f"{report['trades_per_day']:.2f}")]: print(f"{k:22}: {v}")
    print("Reports written to:",out_dir)


def ensure_xt_data(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    expected=[data_dir / f"{a}_USDT_15m.csv" for a in SYMBOLS]
    if all(p.exists() for p in expected):
        print("XT data: all 14 CSV files already present; skipping download.")
        return
    print("XT data: missing CSVs detected; downloading 365 days from XT Futures...")
    session=requests.Session()
    session.headers.update({"User-Agent":"HUNTER-V4-XT-Futures/1.0"})
    discovered=discover_symbols(session)
    resolved={a:resolve_symbol(a,discovered) for a in SYMBOLS}
    missing=[a for a,v in resolved.items() if v is None]
    if missing:
        raise RuntimeError("Unresolved XT Futures symbols: "+", ".join(missing))
    end_dt=now_utc(); start_dt=end_dt-timedelta(days=365)
    for asset in SYMBOLS:
        actual=resolved[asset]
        df,rep=download_symbol(session,actual,start_dt,end_dt)
        if rep.get("rows",0)<30000:
            raise RuntimeError(f"{asset}: suspicious coverage, only {rep.get('rows')} rows")
        out=data_dir/f"{actual}_15m.csv"
        df.to_csv(out,index=False)
        print(f"{asset}: rows={rep['rows']:,} gaps={rep['gaps']} max_gap={rep['max_gap_minutes']}m")


def full_main():
    args=parse_args()
    data_dir=Path(args.data_dir)
    ensure_xt_data(data_dir)
    backtest_main()

if __name__=="__main__":
    full_main()
