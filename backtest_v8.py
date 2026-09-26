#!/usr/bin/env python3
"""
HUNTER-V152 — C2 REGIME-CONDITIONED ENTRY ENGINE

Purpose
-------
Causal continuation/reversal redesign of V144 C_REGIME_SWITCH.

Architecture
------------
15m raw XT Futures data -> 1H signal frame + 4H context + 1D regime.

Two independent entry families inside C2:
  1) TREND_CONTINUATION: higher-timeframe alignment + 1H pullback/reclaim
     + momentum resumption.
  2) EXHAUSTION_REVERSAL: higher-timeframe regime + 1H extreme deviation
     + failed continuation + reversal candle.

Hard rules
----------
- 365-day XT 15m history (plus 35-day warmup).
- Fixed $100 margin, 50x leverage, RR 1:2.
- Entry at the NEXT 1H candle open after the signal candle closes.
- No lookahead, no repaint, no future data in signal construction.
- No overlapping positions; max 3 open positions portfolio-wide.
- No timeout, no break-even, no trailing stop.
- Same-candle SL+TP => LOSS.
- Positions still open at dataset end remain OPEN.
- A candle on which any prior position closes cannot also create a new entry.
- Funding is deliberately NOT used: XT public funding history is not 365-day.

This is a research/backtest file, not an order-execution bot.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA", "BNB",
    "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

DATA_DIR = Path("data/xt_futures_v152")
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
S.headers.update({"User-Agent": "HUNTER-V152/1.0"})

# Deliberately small, interpretable causal research grid.
CONFIGS = [
    # name, trend_score, rev_score, trend_pullback_atr, trend_reclaim_atr,
    # trend_mom24, trend_mom72, rev_dev, rev_range_mult, atr_mult
    ("C2_BALANCED", 5.00, 5.00, 0.35, 0.10, 0.0020, 0.0040, 1.75, 0.80, 1.50),
    ("C2_STRICT_TREND", 5.75, 5.00, 0.25, 0.05, 0.0025, 0.0050, 1.75, 0.80, 1.50),
    ("C2_STRICT_REV", 5.00, 5.75, 0.35, 0.10, 0.0020, 0.0040, 2.00, 0.90, 1.50),
    ("C2_TIGHT_BOTH", 5.75, 5.75, 0.25, 0.05, 0.0025, 0.0050, 2.00, 0.90, 1.50),
    ("C2_WIDE_PULLBACK", 5.00, 5.00, 0.50, 0.10, 0.0020, 0.0040, 1.75, 0.80, 1.50),
    ("C2_DEEP_REVERSAL", 5.00, 5.50, 0.35, 0.10, 0.0020, 0.0040, 2.25, 0.90, 1.50),
    ("C2_FAST_MOMENTUM", 5.25, 5.00, 0.35, 0.10, 0.0035, 0.0060, 1.75, 0.80, 1.50),
    ("C2_LOW_RISK_ATR", 5.25, 5.25, 0.35, 0.10, 0.0025, 0.0050, 2.00, 0.90, 1.25),
]


def resolve_symbols():
    url = f"{BASE}/future/market/v1/public/symbol/list"
    r = S.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    data = js.get("result", js)
    rows = data.get("symbols", []) if isinstance(data, dict) else data
    out = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("symbol", item.get("name", ""))).upper()
        compact = raw.replace("_", "").replace("-", "").replace("/", "").replace(":", "")
        for asset in SYMBOLS:
            wanted = {
                asset + "USDT",
                asset + "_USDT",
                asset + "/USDT",
                asset + "/USDT:USDT",
            }
            wanted = {x.replace("_", "").replace("-", "").replace("/", "").replace(":", "") for x in wanted}
            if compact in wanted:
                out[asset] = item.get("symbol") or item.get("name")
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
                if code not in (None, 0, "0", 200, "200"):
                    raise RuntimeError(f"XT API code={code}: {js.get('msg', js.get('message', ''))}")
            return js
        except Exception as exc:
            last = exc
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
    for item in arr:
        if isinstance(item, dict):
            t = item.get("t", item.get("timestamp", item.get("time")))
            o = item.get("o", item.get("open"))
            h = item.get("h", item.get("high"))
            l = item.get("l", item.get("low"))
            c = item.get("c", item.get("close"))
            v = item.get("a", item.get("volume", item.get("q", 0)))
        else:
            if len(item) < 6:
                continue
            t, o, h, l, c, v = item[:6]
        try:
            out.append([int(t), float(o), float(h), float(l), float(c), float(v)])
        except (TypeError, ValueError):
            continue
    return out


def fetch_symbol(asset, xt_symbol, days):
    """Exact V144-style forward XT 15m collector with strict validation."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int((days + WARMUP_DAYS) * 86400 * 1000)
    interval_ms = 15 * 60 * 1000
    end_ms = (now_ms // interval_ms) * interval_ms - 1
    endpoint = f"{BASE}/future/market/v1/public/q/kline"
    best = []

    for whole_try in range(5):
        rows = []
        cursor = start_ms
        guard = 0
        while cursor < end_ms and guard < 2000:
            guard += 1
            window_end = min(end_ms, cursor + LIMIT * interval_ms - 1)
            params = {
                "symbol": xt_symbol,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": window_end,
                "limit": LIMIT,
            }
            batch = []
            for attempt in range(6):
                try:
                    batch = _parse_kline(_get_json(endpoint, params))
                    if batch:
                        break
                except Exception:
                    pass
                time.sleep(min(2.0, 0.35 * (attempt + 1)))
            if not batch:
                break
            batch = [x for x in batch if start_ms <= x[0] <= end_ms]
            if not batch:
                break
            rows.extend(batch)
            mx = max(x[0] for x in batch)
            if mx < cursor:
                break
            next_cursor = mx + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(0.08)

        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates("timestamp").sort_values("timestamp")
            df = df[(df.timestamp >= start_ms) & (df.timestamp <= end_ms)]
            if len(df) > len(best):
                best = df
            if len(df) >= MIN_ROWS:
                span = int(df.timestamp.max() - df.timestamp.min())
                required = int(days * 86400 * 1000 * 0.90)
                if span >= required:
                    df["timestamp"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
                    return df.set_index("timestamp").dropna()
        time.sleep(float(whole_try + 1))

    if best:
        first = pd.to_datetime(int(best.timestamp.min()), unit="ms", utc=True)
        last = pd.to_datetime(int(best.timestamp.max()), unit="ms", utc=True)
        detail = f"; range={first} -> {last}"
    else:
        detail = ""
    raise RuntimeError(f"Insufficient XT data for {asset}: got {len(best)} rows{detail}; refusing partial backtest")


def ensure_data(data_dir, days):
    data_dir.mkdir(parents=True, exist_ok=True)
    mapping = resolve_symbols()
    print(f"Verified XT symbols: {len(mapping)} / {len(SYMBOLS)}")
    if len(mapping) != len(SYMBOLS):
        raise RuntimeError("XT symbol mapping incomplete")
    out = {}
    for asset in SYMBOLS:
        path = data_dir / f"{asset}_USDT_15m.csv"
        df = fetch_symbol(asset, mapping[asset], days)
        if len(df) < MIN_ROWS:
            raise RuntimeError(f"Validation failed for {asset}: only {len(df)} rows")
        df.to_csv(path)
        out[asset] = df
        gaps = int(df.index.to_series().diff().dt.total_seconds().div(900).sub(1).clip(lower=0).sum())
        print(f"{asset:<8} rows={len(df)} gaps={gaps}")
        if gaps != 0:
            raise RuntimeError(f"Unexpected 15m gaps for {asset}: {gaps}")
    return out


def resample(df, rule):
    return df.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()


def atr_wilder(x, n=14):
    tr = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx_wilder(x, n=14):
    up = x.high.diff()
    dn = -x.low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=x.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=x.index)
    tr = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(all15):
    h1 = {a: resample(df, "1h") for a, df in all15.items()}
    h4 = {a: resample(df, "4h") for a, df in all15.items()}
    d1 = {a: resample(df, "1D") for a, df in all15.items()}
    out = {}

    for asset in SYMBOLS:
        x = h1[asset].copy()
        q = h4[asset].copy()
        d = d1[asset].copy()

        x["atr"] = atr_wilder(x, 14)
        x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
        x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
        x["ret24"] = x.close.pct_change(24)
        x["ret72"] = x.close.pct_change(72)
        x["rv24"] = x.close.pct_change().rolling(24).std()
        x["rv120"] = x.close.pct_change().rolling(120).std()
        x["rv_ratio"] = x.rv24 / x.rv120.replace(0, np.nan)
        x["vol_z"] = (x.volume - x.volume.rolling(48).mean()) / x.volume.rolling(48).std()
        x["body_atr"] = (x.close - x.open).abs() / x.atr
        x["range_atr"] = (x.high - x.low) / x.atr
        x["close_pos"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
        x["dev20"] = (x.close - x.ema20) / x.atr
        x["ema20_slope"] = (x.ema20 - x.ema20.shift(6)) / x.atr
        x["break_high_24"] = x.high.rolling(24).max().shift(1)
        x["break_low_24"] = x.low.rolling(24).min().shift(1)

        q["ema50"] = q.close.ewm(span=50, adjust=False).mean()
        q["ema200"] = q.close.ewm(span=200, adjust=False).mean()
        q["adx"] = adx_wilder(q, 14)
        q["slope"] = (q.ema50 - q.ema50.shift(6)) / atr_wilder(q, 14)
        q["trend"] = np.where(
            (q.close > q.ema200) & (q.ema50 > q.ema200), 1,
            np.where((q.close < q.ema200) & (q.ema50 < q.ema200), -1, 0),
        )

        d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
        d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
        d["slope"] = d.ema50.pct_change(5)
        d["regime"] = np.where(
            (d.close > d.ema200) & (d.ema50 > d.ema200), 1,
            np.where((d.close < d.ema200) & (d.ema50 < d.ema200), -1, 0),
        )

        x["h4_trend"] = q.trend.reindex(x.index, method="ffill")
        x["h4_adx"] = q.adx.reindex(x.index, method="ffill")
        x["h4_slope"] = q.slope.reindex(x.index, method="ffill")
        x["d1_regime"] = d.regime.reindex(x.index, method="ffill")
        x["d1_slope"] = d.slope.reindex(x.index, method="ffill")
        out[asset] = x
    return out


def add_cross_sectional_ranks(f):
    all_times = sorted(set().union(*(x.index for x in f.values())))
    for ts in all_times:
        vals = []
        for asset, x in f.items():
            if ts in x.index:
                v = x.loc[ts, "ret24"]
                if np.isfinite(v):
                    vals.append((asset, float(v)))
        vals.sort(key=lambda z: z[1])
        n = len(vals)
        for rank, (asset, _) in enumerate(vals):
            f[asset].loc[ts, "mom_rank"] = rank / (n - 1) if n > 1 else 0.5
    return f


def finite_row(r, keys):
    return all(np.isfinite(r.get(k, np.nan)) for k in keys)


def make_signals(f, cfg):
    (
        name, trend_score_min, rev_score_min, pullback_atr,
        reclaim_atr, mom24_min, mom72_min, rev_dev, rev_range_mult, _atr_mult,
    ) = cfg
    signals = []

    for asset, x in f.items():
        # i-1 is deliberately used for previous-bar confirmation; entry is i+1 open.
        for i in range(250, len(x) - 1):
            ts = x.index[i]
            r = x.iloc[i]
            p = x.iloc[i - 1]

            common = [
                "atr", "ema20", "ema50", "ret24", "ret72", "vol_z",
                "body_atr", "range_atr", "close_pos", "dev20", "ema20_slope",
                "h4_trend", "h4_adx", "h4_slope", "d1_regime", "d1_slope",
                "mom_rank",
            ]
            if not finite_row(r, common) or not finite_row(p, ["close", "ema20", "dev20"]):
                continue

            long_ctx = r.d1_regime == 1 and r.h4_trend == 1 and r.h4_adx >= 18
            short_ctx = r.d1_regime == -1 and r.h4_trend == -1 and r.h4_adx >= 18

            # -----------------------------
            # 1) TREND CONTINUATION
            # -----------------------------
            long_touch = r.low <= r.ema20 + pullback_atr * r.atr and r.low >= r.ema50 - 0.90 * r.atr
            short_touch = r.high >= r.ema20 - pullback_atr * r.atr and r.high <= r.ema50 + 0.90 * r.atr
            long_reclaim = p.close <= p.ema20 + reclaim_atr * p.atr and r.close > r.ema20
            short_reclaim = p.close >= p.ema20 - reclaim_atr * p.atr and r.close < r.ema20
            long_candle = r.close_pos >= 0.62 and r.body_atr >= 0.25 and r.close > r.open
            short_candle = r.close_pos <= 0.38 and r.body_atr >= 0.25 and r.close < r.open
            long_mom = r.ret24 >= mom24_min and r.ret72 >= mom72_min and r.mom_rank >= 0.60
            short_mom = r.ret24 <= -mom24_min and r.ret72 <= -mom72_min and r.mom_rank <= 0.40
            long_flow = r.vol_z >= -0.25
            short_flow = r.vol_z >= -0.25
            long_slope = r.h4_slope > 0 and r.d1_slope > 0
            short_slope = r.h4_slope < 0 and r.d1_slope < 0
            long_expansion = r.rv_ratio >= 0.90
            short_expansion = r.rv_ratio >= 0.90

            long_trend_score = (
                2.0 * long_ctx + 1.0 * long_slope + 1.0 * long_touch +
                1.0 * long_reclaim + 1.0 * long_mom + 0.75 * long_candle +
                0.50 * long_flow + 0.50 * long_expansion
            )
            short_trend_score = (
                2.0 * short_ctx + 1.0 * short_slope + 1.0 * short_touch +
                1.0 * short_reclaim + 1.0 * short_mom + 0.75 * short_candle +
                0.50 * short_flow + 0.50 * short_expansion
            )

            if long_trend_score >= trend_score_min and long_reclaim and long_ctx and long_trend_score > short_trend_score:
                signals.append({
                    "asset": asset, "signal_ts": ts, "family": "TREND_CONTINUATION",
                    "side": "LONG", "score": float(long_trend_score), "atr": float(r.atr),
                })
            elif short_trend_score >= trend_score_min and short_reclaim and short_ctx and short_trend_score > long_trend_score:
                signals.append({
                    "asset": asset, "signal_ts": ts, "family": "TREND_CONTINUATION",
                    "side": "SHORT", "score": float(short_trend_score), "atr": float(r.atr),
                })

            # -----------------------------
            # 2) EXHAUSTION REVERSAL
            # -----------------------------
            # Reversal is not allowed against a strong daily regime. It is a
            # controlled counter-move only when 4H loses momentum and the 1H
            # candle rejects an extreme.
            long_extreme = r.dev20 <= -rev_dev
            short_extreme = r.dev20 >= rev_dev
            long_failed = p.close < p.ema20 and r.close > p.close and r.close > r.open
            short_failed = p.close > p.ema20 and r.close < p.close and r.close < r.open
            long_reject = r.close_pos >= 0.68 and r.range_atr >= rev_range_mult
            short_reject = r.close_pos <= 0.32 and r.range_atr >= rev_range_mult
            weak_h4_long = r.h4_adx < 24 or r.h4_slope >= -0.10
            weak_h4_short = r.h4_adx < 24 or r.h4_slope <= 0.10
            reversal_volume = r.vol_z >= -0.50

            long_rev_score = (
                2.0 * (r.d1_regime == 1) +
                1.0 * (r.h4_trend >= 0) +
                1.25 * long_extreme +
                1.25 * long_failed +
                1.0 * long_reject +
                0.75 * weak_h4_long +
                0.50 * reversal_volume
            )
            short_rev_score = (
                2.0 * (r.d1_regime == -1) +
                1.0 * (r.h4_trend <= 0) +
                1.25 * short_extreme +
                1.25 * short_failed +
                1.0 * short_reject +
                0.75 * weak_h4_short +
                0.50 * reversal_volume
            )

            if long_rev_score >= rev_score_min and long_failed and long_reject and r.d1_regime == 1 and long_rev_score > short_rev_score:
                signals.append({
                    "asset": asset, "signal_ts": ts, "family": "EXHAUSTION_REVERSAL",
                    "side": "LONG", "score": float(long_rev_score), "atr": float(r.atr),
                })
            elif short_rev_score >= rev_score_min and short_failed and short_reject and r.d1_regime == -1 and short_rev_score > long_rev_score:
                signals.append({
                    "asset": asset, "signal_ts": ts, "family": "EXHAUSTION_REVERSAL",
                    "side": "SHORT", "score": float(short_rev_score), "atr": float(r.atr),
                })

    return signals


def execute_backtest(f, signals, atr_mult):
    by_ts = {}
    for s in signals:
        by_ts.setdefault(s["signal_ts"], []).append(s)

    all_ts = sorted(set().union(*(x.index for x in f.values())))
    positions = []
    trades = []
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    last_close_ts = None

    for ts in all_ts:
        closed_any = False

        # Manage exits on EVERY 1H candle, not only signal candles.
        for p in positions[:]:
            x = f[p["asset"]]
            if ts not in x.index or ts < p["entry_ts"]:
                continue
            row = x.loc[ts]
            hit_sl = row.low <= p["sl"] if p["side"] == "LONG" else row.high >= p["sl"]
            hit_tp = row.high >= p["tp"] if p["side"] == "LONG" else row.low <= p["tp"]

            if not (hit_sl or hit_tp):
                continue

            # Conservative same-candle handling: if both levels are touched,
            # LOSS wins regardless of direction or level ordering.
            if hit_sl and hit_tp:
                outcome = "LOSS"
                exit_price = p["sl"]
            elif hit_sl:
                outcome = "LOSS"
                exit_price = p["sl"]
            else:
                outcome = "WIN"
                exit_price = p["tp"]

            price_ret = (
                (exit_price - p["entry"]) / p["entry"]
                if p["side"] == "LONG"
                else (p["entry"] - exit_price) / p["entry"]
            )
            gross = price_ret * TRADE_MARGIN * LEVERAGE
            fees = TRADE_MARGIN * LEVERAGE * FEE_RATE * 2.0
            pnl = gross - fees
            equity += pnl
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

            trades.append({
                **p,
                "exit_ts": ts,
                "outcome": outcome,
                "pnl": pnl,
                "equity": equity,
            })
            positions.remove(p)
            closed_any = True

        if closed_any:
            last_close_ts = ts
            # Explicit rule: no new signal on the same candle where a prior
            # position result became known.
            continue

        if len(positions) >= MAX_OPEN_POSITIONS:
            continue

        candidates = by_ts.get(ts, [])
        if not candidates:
            continue

        # Highest-quality setup first. One position per asset.
        candidates = sorted(candidates, key=lambda z: (z["score"], abs(z["atr"])), reverse=True)

        for s in candidates:
            if len(positions) >= MAX_OPEN_POSITIONS:
                break
            if any(p["asset"] == s["asset"] for p in positions):
                continue
            x = f[s["asset"]]
            future = x.index[x.index > ts]
            if len(future) == 0:
                continue
            entry_ts = future[0]
            entry_open = float(x.loc[entry_ts, "open"])
            entry = entry_open * (1 + SLIPPAGE if s["side"] == "LONG" else 1 - SLIPPAGE)
            risk = atr_mult * s["atr"]
            if not np.isfinite(risk) or risk <= 0 or risk / entry > 0.08:
                continue
            if s["side"] == "LONG":
                sl = entry - risk
                tp = entry + RR * risk
            else:
                sl = entry + risk
                tp = entry - RR * risk

            positions.append({
                "asset": s["asset"],
                "family": s["family"],
                "side": s["side"],
                "signal_ts": ts,
                "entry_ts": entry_ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "score": s["score"],
            })

    return trades, positions, max_dd


def summarize(trades, open_positions, max_dd, days):
    n = len(trades)
    wins = sum(t["outcome"] == "WIN" for t in trades)
    losses = sum(t["outcome"] == "LOSS" for t in trades)
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    eq = INITIAL_EQUITY
    peak = eq
    dd = 0.0
    streak = 0
    max_streak = 0
    for t in sorted(trades, key=lambda z: z["exit_ts"]):
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
        streak = streak + 1 if t["outcome"] == "LOSS" else 0
        max_streak = max(max_streak, streak)
    net = sum(t["pnl"] for t in trades)
    return {
        "trades": n,
        "open": len(open_positions),
        "wins": wins,
        "losses": losses,
        "wr": 100.0 * wins / n if n else 0.0,
        "pf": gross_win / gross_loss if gross_loss else (math.inf if gross_win else 0.0),
        "pnl": net,
        "dd": min(dd, max_dd),
        "streak": max_streak,
        "trades_day": n / days if days else 0.0,
    }


def score_result(s):
    # Research ranking only. It never changes the simulated trades.
    # Reward WR/PF while penalizing large loss clusters and very small samples.
    if s["trades"] < 100:
        return -1e9 + s["trades"]
    score = 0.0
    score += (s["wr"] - 45.0) * 5.0
    score += max(-2.0, min(2.0, s["pf"] - 1.0)) * 25.0
    score -= max(0, s["streak"] - 4) * 7.5
    score += max(-5.0, min(5.0, s["pnl"] / 1000.0)) * 3.0
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    print("=" * 96)
    print("HUNTER-V152 — C2 REGIME-CONDITIONED ENTRY ENGINE")
    print("15m raw -> 1H trigger / 4H context / 1D regime | no funding")
    print("=" * 96)

    all15 = ensure_data(Path(args.data_dir), args.days)
    f = add_cross_sectional_ranks(build_features(all15))

    print(f"\nTesting {len(CONFIGS)} causal C2 configurations...")
    results = []
    for cfg in CONFIGS:
        name = cfg[0]
        signals = make_signals(f, cfg)
        trades, open_positions, max_dd = execute_backtest(f, signals, cfg[-1])
        summary = summarize(trades, open_positions, max_dd, args.days)
        summary["name"] = name
        summary["signals"] = len(signals)
        summary["score"] = score_result(summary)
        summary["eligible"] = (
            summary["trades"] >= 100
            and summary["wr"] >= 50.0
            and summary["streak"] <= 4
            and summary["pf"] > 1.0
            and summary["pnl"] > 0
        )
        trend = sum(t["family"] == "TREND_CONTINUATION" for t in trades)
        rev = sum(t["family"] == "EXHAUSTION_REVERSAL" for t in trades)
        summary["trend_trades"] = trend
        summary["reversal_trades"] = rev
        results.append(summary)
        print(
            f"{name:<22} trades={summary['trades']:4d} WR={summary['wr']:6.2f}% "
            f"PF={summary['pf']:6.3f} PnL=${summary['pnl']:>9,.2f} "
            f"DD=${summary['dd']:>9,.2f} streak={summary['streak']:2d} "
            f"t/day={summary['trades_day']:.2f} trend={trend:3d} rev={rev:3d} "
            f"eligible={summary['eligible']}"
        )

    ranked = sorted(results, key=lambda z: z["score"], reverse=True)
    print("\nTOP CONFIGURATIONS")
    for i, s in enumerate(ranked[:5], 1):
        print(
            f"{i}. {s['name']} | trades={s['trades']} WR={s['wr']:.2f}% "
            f"PF={s['pf']:.3f} PnL=${s['pnl']:,.2f} DD=${s['dd']:,.2f} "
            f"streak={s['streak']} eligible={s['eligible']}"
        )

    eligible = [s for s in results if s["eligible"]]
    print("\n" + "=" * 96)
    if eligible:
        print("MEASURABLE_EDGE_PRESENT: True")
        print("At least one causal configuration met all predeclared target gates.")
    else:
        print("MEASURABLE_EDGE_PRESENT: False")
        print("No tested C2 configuration reached WR>=50%, max loss streak<=4, PF>1, positive PnL and >=100 trades.")
    print("=" * 96)


if __name__ == "__main__":
    main()
