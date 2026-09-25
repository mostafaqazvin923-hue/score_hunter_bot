#!/usr/bin/env python3
"""
HUNTER-V147 — BALANCED STRUCTURAL MOMENTUM
1D regime + 4H context + 1H liquidity/structure trigger.
Raw source: XT Futures 15m -> causal 1H/4H/1D.

Purpose
-------
V145 showed that parameter tuning did not fix the old regime-switch edge.
V146 then became too restrictive: its AND-gated sweep/reclaim/displacement
logic produced too few trades. V147 keeps the causal MTF architecture but
uses a weighted structural score so valid setups do not need every filter
to fire simultaneously.

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

V147 optimization
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
S.headers.update({"User-Agent": "HUNTER-backtest/146"})


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
    """Fetch completed 15m XT futures candles with forward pagination.

    IMPORTANT: XT's /q/kline endpoint has shown inconsistent behavior when both
    startTime and endTime are supplied for historical windows. The proven-safe
    approach is to paginate using startTime + limit only, then locally trim to
    the requested end. This avoids the recurring "kline batch outside requested
    range" failure seen in GitHub Actions.
    """
    now_ms = int(time.time() * 1000)
    interval_ms = 15 * 60 * 1000
    last_closed_open = (now_ms // interval_ms) * interval_ms - interval_ms
    end_ms = last_closed_open
    start_ms = ((now_ms - int((days + WARMUP_DAYS) * 86400 * 1000)) // interval_ms) * interval_ms

    endpoint = f"{BASE}/future/market/v1/public/q/kline"
    candidates = []
    for sym in (str(xt_symbol).strip(), str(xt_symbol).strip().lower()):
        if sym and sym not in candidates:
            candidates.append(sym)

    last_error = None

    for api_symbol in candidates:
        all_rows = []
        cursor = start_ms
        guard = 0
        failed = False

        while cursor <= end_ms and guard < 1000:
            guard += 1
            params = {
                "symbol": api_symbol,
                "interval": INTERVAL,
                "startTime": int(cursor),
                "limit": LIMIT,
            }

            batch = []
            batch_error = None
            for attempt in range(MAX_REQUEST_RETRIES):
                try:
                    batch = _parse_kline(_get_json(endpoint, params))
                    if batch:
                        break
                    batch_error = RuntimeError(
                        f"empty kline response for {api_symbol} at startTime={cursor}"
                    )
                except Exception as exc:
                    batch_error = exc
                time.sleep(min(2.0, 0.25 * (attempt + 1)))

            if not batch:
                last_error = batch_error
                failed = True
                break

            # XT may return a few candles around the requested boundary. Keep
            # only candles that belong to our requested historical interval.
            batch = [x for x in batch if start_ms <= x[0] <= end_ms]
            if not batch:
                # If the exchange returned only future/out-of-range rows, do not
                # spin forever. Report the actual returned timestamp range.
                raw_min = min(x[0] for x in _parse_kline(_get_json(endpoint, params)))
                raw_max = max(x[0] for x in _parse_kline(_get_json(endpoint, params)))
                last_error = RuntimeError(
                    f"XT returned rows outside requested history for {api_symbol}: "
                    f"requested_start={cursor}, returned={raw_min}->{raw_max}"
                )
                failed = True
                break

            all_rows.extend(batch)
            mx = max(x[0] for x in batch)
            if mx < cursor:
                last_error = RuntimeError(
                    f"XT pagination did not advance for {api_symbol}: {mx} < {cursor}"
                )
                failed = True
                break

            next_cursor = mx + interval_ms
            if next_cursor <= cursor:
                last_error = RuntimeError(
                    f"XT pagination stalled for {api_symbol}: cursor={cursor}, next={next_cursor}"
                )
                failed = True
                break
            cursor = next_cursor
            time.sleep(0.05)

        if failed or not all_rows:
            continue

        df = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        df = df[(df.timestamp >= start_ms) & (df.timestamp <= end_ms)]

        if len(df) >= MIN_ROWS:
            span = int(df.timestamp.max() - df.timestamp.min())
            if span >= int(days * 86400 * 1000 * 0.90):
                df["timestamp"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
                return df.set_index("timestamp").dropna()

        last_error = RuntimeError(
            f"short XT history for {api_symbol}: rows={len(df)}"
        )

    detail = f"; last_error={last_error}" if last_error else ""
    raise RuntimeError(
        f"Insufficient XT data for {asset}: got 0 rows{detail}; refusing partial backtest"
    )


def ensure_data(data_dir, days):
    data_dir.mkdir(parents=True, exist_ok=True)
    mapping = resolve_symbols()
    print(f"Verified XT symbols: {len(mapping)} / {len(SYMBOLS)}")
    if len(mapping) != len(SYMBOLS):
        raise RuntimeError("XT symbol mapping incomplete")
    out = {}
    for a in SYMBOLS:
        path = data_dir / f"{a}_USDT_15m.csv"
        df = fetch_symbol(a, mapping[a], days)
        if len(df) < MIN_ROWS:
            raise RuntimeError(f"Validation failed for {a}: only {len(df)} rows")
        df.to_csv(path)
        out[a] = df
        gaps = int(df.index.to_series().diff().dt.total_seconds().div(900).sub(1).clip(lower=0).sum())
        print(f"{a:<8} rows={len(df)} gaps={gaps}")
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
    """Build all causal features once. No per-parameter recomputation."""
    f = {}
    h1 = {a: resample(df, "1h") for a, df in all15.items()}
    h4 = {a: resample(df, "4h") for a, df in all15.items()}
    d1 = {a: resample(df, "1D") for a, df in all15.items()}

    for a in SYMBOLS:
        x = h1[a].copy()
        q = h4[a].copy()
        d = d1[a].copy()

        # 1H execution/signal features.
        x["atr"] = atr14(x)
        x["body"] = (x.close - x.open).abs()
        x["body_atr"] = x.body / x.atr
        x["range_atr"] = (x.high - x.low) / x.atr
        x["close_pos"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
        x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
        x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
        x["ema20_slope"] = x.ema20.pct_change(3)
        x["ret24"] = x.close.pct_change(24)
        x["ret72"] = x.close.pct_change(72)
        x["vol_z"] = (x.volume - x.volume.rolling(48, min_periods=24).mean()) / x.volume.rolling(48, min_periods=24).std()
        # Liquidity levels use ONLY completed prior bars.
        x["prior_low_24"] = x.low.shift(1).rolling(24, min_periods=24).min()
        x["prior_high_24"] = x.high.shift(1).rolling(24, min_periods=24).max()
        x["prior_low_48"] = x.low.shift(1).rolling(48, min_periods=48).min()
        x["prior_high_48"] = x.high.shift(1).rolling(48, min_periods=48).max()
        x["prior_low_72"] = x.low.shift(1).rolling(72, min_periods=72).min()
        x["prior_high_72"] = x.high.shift(1).rolling(72, min_periods=72).max()
        x["atr_pct"] = x.atr.rolling(240, min_periods=120).rank(pct=True)

        # 4H context.
        q["ema20"] = q.close.ewm(span=20, adjust=False).mean()
        q["ema50"] = q.close.ewm(span=50, adjust=False).mean()
        q["ema200"] = q.close.ewm(span=200, adjust=False).mean()
        q["adx"] = adx14(q)
        q["ema50_slope"] = q.ema50.pct_change(3)
        q["trend"] = np.where((q.close > q.ema20) & (q.ema20 > q.ema50) & (q.ema50 > q.ema200), 1,
                      np.where((q.close < q.ema20) & (q.ema20 < q.ema50) & (q.ema50 < q.ema200), -1, 0))

        # 1D primary regime.
        d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
        d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
        d["ema50_slope"] = d.ema50.pct_change(5)
        d["regime"] = np.where((d.close > d.ema50) & (d.ema50 > d.ema200) & (d.ema50_slope > 0), 1,
                       np.where((d.close < d.ema50) & (d.ema50 < d.ema200) & (d.ema50_slope < 0), -1, 0))

        # Only completed higher-TF candles are available at each 1H timestamp.
        x["h4_trend"] = q.trend.reindex(x.index, method="ffill")
        x["h4_adx"] = q.adx.reindex(x.index, method="ffill")
        x["h4_slope"] = q.ema50_slope.reindex(x.index, method="ffill")
        x["d1_regime"] = d.regime.reindex(x.index, method="ffill")
        f[a] = x
    return f


def add_cross_sectional_rank(f):
    """Causal rank of 24H return; value at t uses only data known at t close."""
    ret = {a: x.close.pct_change(24) for a, x in f.items()}
    times = sorted(set().union(*(x.index for x in f.values())))
    for ts in times:
        vals = [(a, ret[a].get(ts, np.nan)) for a in SYMBOLS]
        vals = [(a, float(v)) for a, v in vals if np.isfinite(v)]
        vals.sort(key=lambda z: z[1])
        n = len(vals)
        if n < 2:
            continue
        for rank, (a, _) in enumerate(vals):
            f[a].loc[ts, "mom_rank"] = rank / (n - 1)
    return f


def signal_events(f, p):
    """Causal weighted structural score on completed 1H candles."""
    sig = []
    for asset, x in f.items():
        for i in range(max(250, p["lookback"]), len(x) - 1):
            r = x.iloc[i]
            keys = ["atr","body_atr","range_atr","close_pos","vol_z","h4_adx",
                    "h4_slope","mom_rank","d1_regime","ema20","ema50",
                    "prior_low_24","prior_high_24","prior_low_48","prior_high_48",
                    "prior_low_72","prior_high_72","ret24","ret72"]
            if not all(np.isfinite(r.get(k, np.nan)) for k in keys):
                continue

            long_score = 0.0
            short_score = 0.0

            # 1D regime: strongest directional context.
            if r.d1_regime == 1: long_score += 2.0
            if r.d1_regime == -1: short_score += 2.0

            # 4H context: trend, slope, and ADX each contribute independently.
            if r.h4_trend == 1: long_score += 1.5
            if r.h4_trend == -1: short_score += 1.5
            if r.h4_slope > 0: long_score += 0.75
            if r.h4_slope < 0: short_score += 0.75
            if r.h4_adx >= p["adx"]:
                if r.h4_trend == 1: long_score += 0.75
                if r.h4_trend == -1: short_score += 0.75

            # 1H momentum/location.
            if r.ret24 > p["mom24"]: long_score += 0.75
            if r.ret24 < -p["mom24"]: short_score += 0.75
            if r.ret72 > p["mom72"]: long_score += 0.75
            if r.ret72 < -p["mom72"]: short_score += 0.75
            if r.close > r.ema20: long_score += 0.50
            if r.close < r.ema20: short_score += 0.50
            if r.close > r.ema50: long_score += 0.50
            if r.close < r.ema50: short_score += 0.50

            # Cross-sectional momentum is confirmation, not a hard gate.
            if r.mom_rank >= p["rank_long"]: long_score += 0.75
            if r.mom_rank <= p["rank_short"]: short_score += 0.75

            # Liquidity / breakout structure. Uses only prior completed bars.
            if p["lookback"] == 24:
                pl, ph = r.prior_low_24, r.prior_high_24
            elif p["lookback"] == 48:
                pl, ph = r.prior_low_48, r.prior_high_48
            else:
                pl, ph = r.prior_low_72, r.prior_high_72

            long_sweep = r.low < pl and r.close > pl
            short_sweep = r.high > ph and r.close < ph
            long_break = r.close > ph
            short_break = r.close < pl
            if long_sweep: long_score += 1.25
            if short_sweep: short_score += 1.25
            if long_break: long_score += 1.00
            if short_break: short_score += 1.00

            # Candle quality: soft confirmation rather than mandatory filters.
            if r.body_atr >= p["body_atr"] and r.close > r.open: long_score += 0.75
            if r.body_atr >= p["body_atr"] and r.close < r.open: short_score += 0.75
            if r.range_atr >= p["range_atr"]:
                if r.close_pos >= 0.60: long_score += 0.50
                if r.close_pos <= 0.40: short_score += 0.50
            if r.vol_z >= p["vol_z"]:
                if r.close > r.open: long_score += 0.50
                if r.close < r.open: short_score += 0.50

            # Avoid chasing extreme extensions.
            long_location_ok = r.close <= r.ema20 + p["max_ext"] * r.atr
            short_location_ok = r.close >= r.ema20 - p["max_ext"] * r.atr

            if long_score >= p["min_score"] and long_location_ok and long_score > short_score:
                sig.append({"asset":asset,"signal_ts":x.index[i],"side":"LONG",
                            "family":"BALANCED_SCORE","atr":float(r.atr),"score":long_score})
            elif short_score >= p["min_score"] and short_location_ok and short_score > long_score:
                sig.append({"asset":asset,"signal_ts":x.index[i],"side":"SHORT",
                            "family":"BALANCED_SCORE","atr":float(r.atr),"score":short_score})
    return sig


def backtest(f, sig):
    byts = {}
    for z in sig:
        byts.setdefault(z["signal_ts"], []).append(z)
    all_ts = sorted(set().union(*(x.index for x in f.values())))
    positions = []
    trades = []
    last_close = pd.Timestamp.min.tz_localize("UTC")

    for ts in all_ts:
        # Exit management is performed on EVERY 1H timestamp.
        for p in positions[:]:
            x = f[p["asset"]]
            if ts not in x.index or ts <= p["entry_ts"]:
                continue
            row = x.loc[ts]
            hit_sl = row.low <= p["sl"] if p["side"] == "LONG" else row.high >= p["sl"]
            hit_tp = row.high >= p["tp"] if p["side"] == "LONG" else row.low <= p["tp"]
            if hit_sl or hit_tp:
                # Conservative rule: if both are touched, SL wins.
                outcome = "LOSS" if hit_sl else "WIN"
                ex = p["sl"] if hit_sl else p["tp"]
                gross = ((ex-p["entry"])/p["entry"] if p["side"] == "LONG" else (p["entry"]-ex)/p["entry"])
                gross *= TRADE_MARGIN * LEVERAGE
                pnl = gross - (TRADE_MARGIN * LEVERAGE * FEE_RATE * 2)
                trades.append({**p, "exit_ts":ts, "outcome":outcome, "pnl":pnl})
                positions.remove(p)
                last_close = ts

        # Never enter on the same timestamp a result becomes known.
        if ts <= last_close or len(positions) >= MAX_OPEN_POSITIONS:
            continue

        used = set()
        for z in sorted(byts.get(ts, []), key=lambda q: abs(q["atr"]), reverse=True):
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
            risk = 1.5 * z["atr"]
            sl = entry - risk if z["side"] == "LONG" else entry + risk
            tp = entry + RR * risk if z["side"] == "LONG" else entry - RR * risk
            positions.append({"asset":z["asset"],"family":z["family"],"side":z["side"],
                              "signal_ts":ts,"entry_ts":ets,"entry":entry,"sl":sl,"tp":tp})
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
    return {
        "trades":len(trades),
        "wr":100*wins/len(trades) if trades else 0.0,
        "pf":gross_w/gross_l if gross_l else 0.0,
        "pnl":sum(z["pnl"] for z in trades),
        "dd":dd,
        "max_streak":mx,
        "trades_day":len(trades)/days,
    }


def score(m):
    # Research ranking only; never alters trade outcomes.
    if m["trades"] < 100:
        return -1e12
    wr_gap = max(0.0, 50.0 - m["wr"])
    streak_gap = max(0, m["max_streak"] - 4)
    dd_penalty = max(0.0, -m["dd"] - 1000.0) * 0.15
    return m["pnl"] + 700*m["pf"] - 180*wr_gap - 180*streak_gap - dd_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    print("=" * 96)
    print("HUNTER-V147 — BALANCED STRUCTURAL MOMENTUM")
    print("Signal=1H | Context=4H | Regime=1D | Raw source=15m")
    print("1D regime + 4H context + weighted 1H structural/momentum score")
    print("=" * 96)

    all15 = ensure_data(Path(args.data_dir), args.days)
    f = add_cross_sectional_rank(build_features(all15))

    # 3 x 2 x 2 x 2 = 24 compact causal configurations.
    grid = []
    for lookback in [24, 48, 72]:
        for min_score in [5.25, 6.00]:
            for adx in [16.0, 20.0]:
                for max_ext in [1.50, 2.00]:
                    grid.append({
                        "lookback":lookback,
                        "min_score":min_score,
                        "adx":adx,
                        "mom24":0.0,
                        "mom72":0.0,
                        "vol_z":-0.75,
                        "body_atr":0.35,
                        "range_atr":0.80,
                        "rank_long":0.55,
                        "rank_short":0.45,
                        "max_ext":max_ext,
                    })

    print(f"Testing {len(grid)} causal C balanced parameter sets...")
    results = []
    for n, p in enumerate(grid, 1):
        sig = signal_events(f, p)
        trades, openp = backtest(f, sig)
        m = metrics(trades)
        m.update({"open":len(openp), "signals":len(sig), "score":score(m)})
        results.append((m, p))
        if n % 12 == 0:
            print(f"  tested {n}/{len(grid)}")

    results.sort(key=lambda z: z[0]["score"], reverse=True)
    print("\nTop 15 candidates:")
    for i, (m, p) in enumerate(results[:15], 1):
        print(f"{i:>2} trades={m['trades']:>4} WR={m['wr']:>6.2f}% PF={m['pf']:.3f} "
              f"PnL=${m['pnl']:,.2f} DD=${m['dd']:,.2f} streak={m['max_streak']:>2} "
              f"t/day={m['trades_day']:.2f} | lb={p['lookback']} score={p['min_score']:.2f} "
              f"adx={p['adx']:.0f} ext={p['max_ext']:.2f}")

    eligible = [z for z in results if z[0]["wr"] >= 50.0 and z[0]["max_streak"] <= 4 and z[0]["trades"] >= 100]
    print("\nELIGIBLE:", bool(eligible))
    if eligible:
        m, p = eligible[0]
        print("Best eligible:", m)
        print("Parameters:", p)
        print("NEXT: validate this parameter set with a strict walk-forward/OOS split before any live use.")
    else:
        print("No tested configuration reached WR>=50% AND max loss streak<=4 with >=100 trades.")
        print("NEXT: do not force the target; inspect the structural family and move to the next research branch.")


if __name__ == "__main__":
    main()
