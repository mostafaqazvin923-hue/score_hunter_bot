#!/usr/bin/env python3
"""
HUNTER-V149 — FLOW / POSITIONING PROXY RESEARCH
1D regime + 4H trend context + 1H pullback/reclaim/momentum resumption.
Raw source: XT Futures 15m -> causal 1H/4H/1D.

Purpose
-------
V144-V148 did not produce the required edge. V149 therefore switches to a
different research family: FLOW / POSITIONING PROXY. Because the XT kline
history does not contain historical whale wallets or open-interest snapshots,
this version does NOT pretend to reconstruct true smart-money positions.
Instead it tests a causal market-participant-pressure proxy built from OHLCV:
volume-weighted candle pressure, cumulative flow, price/flow divergence,
absorption, and volatility expansion. Higher timeframes define regime; 1H
provides the trigger.

Signal concept
--------------
LONG (mirror for SHORT):
  1) 1D directional regime is established and EMA50 slope agrees.
  2) 4H trend agrees, with EMA structure and ADX strength.
  3) 1H price sweeps a prior rolling liquidity low and closes back above it.
  4) The same 1H candle shows bullish displacement/body strength.
  5) Volume confirms the event.
  6) Price is not excessively extended from 1H EMA20.

This is a causal liquidity-sweep -> reclaim -> displacement model, not a
future-pivot/retest model. Entry is always the NEXT 1H candle open.

Integrity rules
---------------
- No lookahead / future leak / repainting.
- 15m is only the raw source; signals are 1H.
- 4H and 1D values are right-labeled and forward-filled only after their
  completed candle timestamp.
- No overlapping positions; max 3 portfolio positions.
- No new entry on the same timestamp a prior trade closes.
- Fixed RR 1:2.
- No timeout / max-bars exit.
- Same-candle SL+TP => LOSS.
- Positions still open at dataset end remain OPEN.
- Minimum trade count is used for research ranking, not to manufacture wins.

V148 optimization
------------------
A compact causal grid varies the minimum score and a few structural
thresholds. Features are computed once. The score is never used to alter
outcomes; it only defines which causal setups are admitted.

IMPORTANT: an in-sample winner is NOT considered validated. A qualifying
candidate must be walk-forward/OOS tested before any live consideration.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = ["BTC","ETH","SOL","SUI","AVAX","NEAR","ADA","BNB","APT","CRV","ONDO","PENDLE","ICP","WIF"]
DATA_DIR = Path("data/xt_futures_v141")
OUT_DIR = DATA_DIR / "results"

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0
MAX_OPEN_POSITIONS = 3

BASE = "https://fapi.xt.com"
LIMIT = 1500
INTERVAL = "15m"
DAYS = 365
WARMUP_DAYS = 35
MIN_ROWS = 30000
MAX_REQUEST_RETRIES = 6

S = requests.Session()
S.headers.update({"User-Agent": "HUNTER-backtest/148"})


def resolve_symbols():
    u = f"{BASE}/future/market/v1/public/symbol/list"
    r = S.get(u, timeout=20)
    r.raise_for_status()
    js = r.json()
    data = js.get("result", js)
    rows = data.get("symbols", []) if isinstance(data, dict) else data
    out = {}
    for x in rows:
        if not isinstance(x, dict):
            continue
        raw = str(x.get("symbol", x.get("name", ""))).upper()
        compact = raw.replace("_", "").replace("-", "").replace("/", "").replace(":", "")
        for a in SYMBOLS:
            aliases = {a + "USDT", a + "_USDT", a + "/USDT", a + "/USDT:USDT"}
            norm = {z.replace("_", "").replace("-", "").replace("/", "").replace(":", "") for z in aliases}
            if compact in norm:
                out[a] = x.get("symbol") or x.get("name")
    return out


def _get_json(url, params):
    last = None
    for attempt in range(MAX_REQUEST_RETRIES):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict):
                code = js.get("code")
                return_code = js.get("returnCode")
                if code not in (None, 0, "0", 200, "200"):
                    raise RuntimeError(f"XT API code={code}: {js.get('msg', js.get('message', ''))}")
                if return_code not in (None, 0, "0", 200, "200"):
                    raise RuntimeError(f"XT API returnCode={return_code}: {js.get('msgInfo', js.get('msg', js.get('message', '')))}")
            return js
        except Exception as e:
            last = e
            time.sleep(min(2.0, 0.25 * (2 ** attempt)))
    raise RuntimeError(f"XT request failed after {MAX_REQUEST_RETRIES} retries: {last}")


def _parse_kline(js):
    raw = js.get("result", js) if isinstance(js, dict) else js
    if isinstance(raw, dict):
        arr = raw.get("data", raw.get("rows", raw.get("list", [])))
    else:
        arr = raw
    if not isinstance(arr, list):
        return []
    out = []
    for x in arr:
        if isinstance(x, dict):
            t = x.get("t", x.get("timestamp", x.get("time")))
            o = x.get("o", x.get("open")); h = x.get("h", x.get("high"))
            l = x.get("l", x.get("low")); c = x.get("c", x.get("close"))
            v = x.get("a", x.get("volume", x.get("q", 0)))
        else:
            if len(x) < 6:
                continue
            t, o, h, l, c, v = x[:6]
        try:
            out.append([int(t), float(o), float(h), float(l), float(c), float(v)])
        except (TypeError, ValueError):
            continue
    return out


def fetch_symbol(asset, xt_symbol, days):
    """
    Robust XT 15m collector.

    Important: XT documents both startTime and endTime for /q/kline.  We
    paginate FORWARD with startTime+endTime windows instead of relying on a
    single moving endTime.  This avoids the partial-page behaviour that caused
    the previous run to receive only 15 BTC candles.
    """
    now_ms=int(time.time()*1000)
    start_ms=now_ms-int((days+WARMUP_DAYS)*86400*1000)
    # Never use a still-forming candle.
    interval_ms=15*60*1000
    end_ms=(now_ms//interval_ms)*interval_ms-1

    best=[]
    endpoint=f"{BASE}/future/market/v1/public/q/kline"

    for whole_try in range(5):
        rows=[]
        cursor=start_ms
        guard=0
        failed=False

        while cursor < end_ms and guard < 2000:
            guard += 1
            # 1500 x 15m = 15.625 days. Keep a tiny overlap and dedupe later.
            window_end=min(end_ms, cursor + LIMIT*interval_ms - 1)
            params={
                "symbol":xt_symbol,
                "interval":INTERVAL,
                "startTime":cursor,
                "endTime":window_end,
                "limit":LIMIT,
            }

            batch=[]
            last_err=None
            for attempt in range(6):
                try:
                    batch=_parse_kline(_get_json(endpoint,params))
                    if batch:
                        break
                except Exception as e:
                    last_err=e
                time.sleep(min(2.0,0.35*(attempt+1)))

            if not batch:
                failed=True
                break

            # Keep only valid candles inside the requested window.
            batch=[x for x in batch if start_ms <= x[0] <= end_ms]
            if not batch:
                failed=True
                break

            rows.extend(batch)
            mn=min(x[0] for x in batch)
            mx=max(x[0] for x in batch)

            # Hard progress check. Never loop on the same XT page.
            if mx < cursor:
                failed=True
                break

            # Normal case: move just past the last candle received.
            next_cursor=mx+1
            if next_cursor <= cursor:
                failed=True
                break
            cursor=next_cursor

            # If XT returned fewer than LIMIT rows, the next forward request
            # is still valid; do not treat a short page as a fatal error.
            time.sleep(0.08)

        if rows:
            df=pd.DataFrame(
                rows,
                columns=["timestamp","open","high","low","close","volume"]
            )
            df=df.drop_duplicates("timestamp").sort_values("timestamp")
            df=df[(df.timestamp>=start_ms)&(df.timestamp<=end_ms)]

            if len(df)>len(best):
                best=df

            if len(df)>=MIN_ROWS:
                # Require broad time coverage, not merely row count.
                span=int(df.timestamp.max()-df.timestamp.min())
                required_span=int(days*86400*1000*0.90)
                if span >= required_span:
                    df["timestamp"]=pd.to_datetime(df.timestamp,unit="ms",utc=True)
                    return df.set_index("timestamp").dropna()

        time.sleep(1.0*(whole_try+1))

    got=len(best)
    if got:
        first=pd.to_datetime(int(best.timestamp.min()),unit="ms",utc=True)
        last=pd.to_datetime(int(best.timestamp.max()),unit="ms",utc=True)
        detail=f"; range={first} -> {last}"
    else:
        detail=""
    raise RuntimeError(
        f"Insufficient XT data for {asset}: got {got} rows{detail}; "
        f"refusing to run a partial backtest"
    )

def ensure_data(data_dir,days):
    data_dir.mkdir(parents=True,exist_ok=True); mapping=resolve_symbols()
    print(f"Verified XT symbols: {len(mapping)} / {len(SYMBOLS)}")
    if len(mapping)!=len(SYMBOLS): raise RuntimeError("XT symbol mapping incomplete")
    out={}
    for a in SYMBOLS:
        path=data_dir/f"{a}_USDT_15m.csv"
        try:
            df=fetch_symbol(a,mapping[a],days)
            if len(df)<MIN_ROWS:
                raise RuntimeError(f"Validation failed for {a}: only {len(df)} rows")
            df.to_csv(path)
            out[a]=df
            gaps=int(df.index.to_series().diff().dt.total_seconds().div(900).sub(1).clip(lower=0).sum())
            print(f"{a:<8} rows={len(df)} gaps={gaps}")
        except Exception as e:
            raise RuntimeError(f"XT download failed for {a}: {e}") from e
    return out

def resample(df, rule):
    return df.resample(rule, label="right", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()


def atr14(x):
    tr = pd.concat([(x.high-x.low),
                    (x.high-x.close.shift()).abs(),
                    (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()


def adx14(x):
    up = x.high.diff()
    dn = -x.low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([(x.high-x.low), (x.high-x.close.shift()).abs(), (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()


def build_features(all15):
    """Build causal 1H breakout/volatility features from 15m source."""
    f = {}
    h1 = {a: resample(df, "1h") for a, df in all15.items()}
    h4 = {a: resample(df, "4h") for a, df in all15.items()}
    d1 = {a: resample(df, "1D") for a, df in all15.items()}

    for a in SYMBOLS:
        x = h1[a].copy()
        q = h4[a].copy()
        d = d1[a].copy()

        x["atr"] = atr14(x)
        rng = (x.high - x.low).replace(0, np.nan)
        x["range_atr"] = (x.high - x.low) / x.atr
        x["body_atr"] = (x.close - x.open).abs() / x.atr
        x["close_pos"] = (x.close - x.low) / rng
        x["vol_z"] = (x.volume - x.volume.rolling(48, min_periods=24).mean()) / x.volume.rolling(48, min_periods=24).std()
        x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
        x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
        x["ret24"] = x.close.pct_change(24)
        # Percentile rank of current ATR among roughly 10 days of 1H data.
        x["atr_pct"] = x.atr.rolling(240, min_periods=120).rank(pct=True)

        # 4H context: completed candle only.
        q["ema20"] = q.close.ewm(span=20, adjust=False).mean()
        q["ema50"] = q.close.ewm(span=50, adjust=False).mean()
        q["ema200"] = q.close.ewm(span=200, adjust=False).mean()
        q["adx"] = adx14(q)
        q["ema50_slope"] = q.ema50.pct_change(3)
        q["trend"] = np.where(
            (q.close > q.ema20) & (q.ema20 > q.ema50) & (q.ema50 > q.ema200), 1,
            np.where((q.close < q.ema20) & (q.ema20 < q.ema50) & (q.ema50 < q.ema200), -1, 0)
        )

        # 1D primary regime.
        d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
        d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
        d["ema50_slope"] = d.ema50.pct_change(5)
        d["regime"] = np.where(
            (d.close > d.ema50) & (d.ema50 > d.ema200) & (d.ema50_slope > 0), 1,
            np.where((d.close < d.ema50) & (d.ema50 < d.ema200) & (d.ema50_slope < 0), -1, 0)
        )

        x["h4_trend"] = q.trend.reindex(x.index, method="ffill")
        x["h4_adx"] = q.adx.reindex(x.index, method="ffill")
        x["h4_slope"] = q.ema50_slope.reindex(x.index, method="ffill")
        x["d1_regime"] = d.regime.reindex(x.index, method="ffill")
        x["d1_slope"] = d.ema50_slope.reindex(x.index, method="ffill")
        f[a] = x
    return f


def signal_events(f, p):
    """Causal volatility-compression -> Donchian breakout events."""
    sig = []
    for asset, x in f.items():
        n = int(p["donchian"])
        prior_high = x.high.rolling(n, min_periods=n).max().shift(1)
        prior_low = x.low.rolling(n, min_periods=n).min().shift(1)
        prior_compression = x.atr_pct.shift(1).rolling(
            int(p["compression_bars"]), min_periods=int(p["compression_bars"])
        ).mean()

        for i in range(max(260, n + int(p["compression_bars"]) + 5), len(x)-1):
            r = x.iloc[i]
            prev = x.iloc[i-1]
            vals = [r.atr, r.range_atr, r.body_atr, r.close_pos, r.vol_z,
                    r.atr_pct, r.h4_adx, r.h4_slope, r.d1_regime,
                    r.d1_slope, prior_high.iloc[i], prior_low.iloc[i],
                    prior_compression.iloc[i]]
            if not all(np.isfinite(v) for v in vals):
                continue

            long_ctx = r.d1_regime == 1 and r.h4_trend == 1 and r.h4_slope > 0 and r.h4_adx >= p["adx"]
            short_ctx = r.d1_regime == -1 and r.h4_trend == -1 and r.h4_slope < 0 and r.h4_adx >= p["adx"]
            if not (long_ctx or short_ctx):
                continue

            compressed = prior_compression.iloc[i] <= p["compression_pct"]
            expansion = r.range_atr >= p["expansion_range_atr"] and r.body_atr >= p["expansion_body_atr"]
            volume_confirm = r.vol_z >= p["vol_z"]

            long_break = r.close > prior_high.iloc[i] + p["break_buffer_atr"] * r.atr
            short_break = r.close < prior_low.iloc[i] - p["break_buffer_atr"] * r.atr
            long_candle = r.close_pos >= p["close_pos"] and r.close > r.open
            short_candle = r.close_pos <= (1.0-p["close_pos"]) and r.close < r.open

            # Require the breakout to be the first directional expansion after
            # the compressed state; this avoids chasing an already-expanded bar.
            prior_not_break = prev.close <= prior_high.iloc[i] if np.isfinite(prior_high.iloc[i]) else False
            prior_not_break_s = prev.close >= prior_low.iloc[i] if np.isfinite(prior_low.iloc[i]) else False

            if long_ctx and compressed and expansion and volume_confirm and long_break and long_candle and prior_not_break:
                sig.append({"asset":asset, "signal_ts":x.index[i], "side":"LONG",
                            "family":"VOLATILITY_DONCHIAN_BREAKOUT", "atr":float(r.atr)})
            elif short_ctx and compressed and expansion and volume_confirm and short_break and short_candle and prior_not_break_s:
                sig.append({"asset":asset, "signal_ts":x.index[i], "side":"SHORT",
                            "family":"VOLATILITY_DONCHIAN_BREAKOUT", "atr":float(r.atr)})
    return sig


def backtest(f, sig, stop_atr):
    byts = {}
    for z in sig:
        byts.setdefault(z["signal_ts"], []).append(z)
    all_ts = sorted(set().union(*(x.index for x in f.values())))
    positions = []
    trades = []
    last_close = pd.Timestamp.min.tz_localize("UTC")

    for ts in all_ts:
        for p in positions[:]:
            x = f[p["asset"]]
            if ts not in x.index or ts <= p["entry_ts"]:
                continue
            row = x.loc[ts]
            hit_sl = row.low <= p["sl"] if p["side"] == "LONG" else row.high >= p["sl"]
            hit_tp = row.high >= p["tp"] if p["side"] == "LONG" else row.low <= p["tp"]
            if hit_sl or hit_tp:
                outcome = "LOSS" if hit_sl else "WIN"
                ex = p["sl"] if hit_sl else p["tp"]
                gross = ((ex-p["entry"])/p["entry"] if p["side"] == "LONG" else (p["entry"]-ex)/p["entry"])
                gross *= TRADE_MARGIN * LEVERAGE
                pnl = gross - (TRADE_MARGIN * LEVERAGE * FEE_RATE * 2)
                trades.append({**p, "exit_ts":ts, "outcome":outcome, "pnl":pnl})
                positions.remove(p)
                last_close = ts

        if ts <= last_close or len(positions) >= MAX_OPEN_POSITIONS:
            continue

        used = set()
        for z in byts.get(ts, []):
            if len(positions) >= MAX_OPEN_POSITIONS or z["asset"] in used:
                continue
            if any(p["asset"] == z["asset"] for p in positions):
                continue
            x = f[z["asset"]]
            future = x.index[x.index > ts]
            if len(future) == 0:
                continue
            ets = future[0]
            entry = float(x.loc[ets, "open"]) * (1 + SLIPPAGE if z["side"] == "LONG" else 1 - SLIPPAGE)
            risk = stop_atr * z["atr"]
            sl = entry - risk if z["side"] == "LONG" else entry + risk
            tp = entry + RR * risk if z["side"] == "LONG" else entry - RR * risk
            positions.append({"asset":z["asset"], "family":z["family"], "side":z["side"],
                              "signal_ts":ts, "entry_ts":ets, "entry":entry, "sl":sl, "tp":tp})
            used.add(z["asset"])

    return trades, positions


def metrics(trades):
    wins = sum(z["outcome"] == "WIN" for z in trades)
    gross_w = sum(z["pnl"] for z in trades if z["pnl"] > 0)
    gross_l = -sum(z["pnl"] for z in trades if z["pnl"] < 0)
    eq = INITIAL_EQUITY
    peak = eq
    dd = 0.0
    streak = 0
    mx = 0
    for z in sorted(trades, key=lambda q: q["exit_ts"]):
        eq += z["pnl"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
        streak = streak + 1 if z["outcome"] == "LOSS" else 0
        mx = max(mx, streak)
    first = min((z["exit_ts"] for z in trades), default=None)
    last = max((z["exit_ts"] for z in trades), default=None)
    days = max(1.0, (last-first).total_seconds()/86400) if first is not None else 1.0
    return {"trades":len(trades), "wr":100*wins/len(trades) if trades else 0.0,
            "pf":gross_w/gross_l if gross_l else 0.0, "pnl":sum(z["pnl"] for z in trades),
            "dd":dd, "max_streak":mx, "trades_day":len(trades)/days}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    print("=" * 96)
    print("HUNTER-V150 — VOLATILITY COMPRESSION + DONCHIAN BREAKOUT BASE TEST")
    print("Signal=1H | Context=4H | Regime=1D | Raw source=15m")
    print("Alpha: volatility compression -> range breakout -> expansion confirmation")
    print("Small base test: 5 hand-designed causal configurations; no optimization grid")
    print("=" * 96)

    all15 = ensure_data(Path(args.data_dir), args.days)
    f = build_features(all15)

    configs = [
        {"name":"A_BALANCED_20", "donchian":20, "compression_bars":4, "compression_pct":0.25, "expansion_range_atr":1.20, "expansion_body_atr":0.35, "vol_z":0.50, "break_buffer_atr":0.05, "close_pos":0.70, "adx":16.0, "stop_atr":1.00},
        {"name":"B_WIDE_COMPRESSION_20", "donchian":20, "compression_bars":4, "compression_pct":0.35, "expansion_range_atr":1.20, "expansion_body_atr":0.35, "vol_z":0.50, "break_buffer_atr":0.05, "close_pos":0.70, "adx":16.0, "stop_atr":1.00},
        {"name":"C_LONGER_RANGE_40", "donchian":40, "compression_bars":4, "compression_pct":0.25, "expansion_range_atr":1.20, "expansion_body_atr":0.35, "vol_z":0.50, "break_buffer_atr":0.05, "close_pos":0.70, "adx":16.0, "stop_atr":1.00},
        {"name":"D_STRONG_EXPANSION", "donchian":20, "compression_bars":4, "compression_pct":0.25, "expansion_range_atr":1.40, "expansion_body_atr":0.45, "vol_z":0.75, "break_buffer_atr":0.05, "close_pos":0.70, "adx":20.0, "stop_atr":1.00},
        {"name":"E_WIDE_RANGE_TIGHT_STOP", "donchian":40, "compression_bars":6, "compression_pct":0.35, "expansion_range_atr":1.40, "expansion_body_atr":0.45, "vol_z":0.75, "break_buffer_atr":0.05, "close_pos":0.70, "adx":20.0, "stop_atr":1.25},
    ]

    print(f"Testing {len(configs)} causal base configurations...")
    results = []
    for i, p in enumerate(configs, 1):
        sig = signal_events(f, p)
        trades, openp = backtest(f, sig, p["stop_atr"])
        m = metrics(trades)
        m.update({"open":len(openp), "signals":len(sig)})
        results.append((m, p))
        print(f"  tested {i}/{len(configs)}: {p['name']} | signals={len(sig)} trades={m['trades']}")

    print("\nBASE TEST RESULTS:")
    for i, (m, p) in enumerate(results, 1):
        print(f"{i}. {p['name']:<24} trades={m['trades']:>4} WR={m['wr']:>6.2f}% PF={m['pf']:.3f} "
              f"PnL=${m['pnl']:,.2f} DD=${m['dd']:,.2f} streak={m['max_streak']:>2} "
              f"t/day={m['trades_day']:.2f} open={m['open']} signals={m['signals']}")

    viable = [z for z in results if z[0]["trades"] >= 100 and z[0]["pf"] > 1.0]
    print("\nMEASURABLE_EDGE_PRESENT:", bool(viable))
    if viable:
        print("At least one base configuration has PF>1.0 with >=100 realized trades.")
        print("NEXT: only the strongest base candidate should receive a small, causal refinement test and then OOS validation.")
    else:
        print("No base configuration showed PF>1.0 with >=100 realized trades.")
        print("NEXT: reject this family; do not optimize parameters to force the target.")


if __name__ == "__main__":
    main()
