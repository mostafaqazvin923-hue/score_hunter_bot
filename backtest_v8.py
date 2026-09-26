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
        x["ret6"] = x.close.pct_change(6)
        x["ret12"] = x.close.pct_change(12)
        x["ret24"] = x.close.pct_change(24)
        x["ret72"] = x.close.pct_change(72)
        x["rv24"] = x.close.pct_change().rolling(24).std()
        x["rv120"] = x.close.pct_change().rolling(120).std()
        x["rv_ratio"] = x.rv24 / x.rv120.replace(0, np.nan)
        x["vol_z"] = (x.volume - x.volume.rolling(48).mean()) / x.volume.rolling(48).std()
        x["body_atr"] = (x.close - x.open).abs() / x.atr
        x["range_atr"] = (x.high - x.low) / x.atr
        x["close_pos"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
        x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
        x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
        x["ema20_slope"] = (x.ema20 - x.ema20.shift(6)) / x.atr
        # All breakout levels are shifted: the signal candle never sees itself.
        x["high12"] = x.high.rolling(12).max().shift(1)
        x["low12"] = x.low.rolling(12).min().shift(1)
        x["high24"] = x.high.rolling(24).max().shift(1)
        x["low24"] = x.low.rolling(24).min().shift(1)

        q["atr"] = atr_wilder(q, 14)
        q["ema50"] = q.close.ewm(span=50, adjust=False).mean()
        q["ema200"] = q.close.ewm(span=200, adjust=False).mean()
        q["adx"] = adx_wilder(q, 14)
        q["slope"] = (q.ema50 - q.ema50.shift(6)) / q.atr
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
                r = x.loc[ts, "ret24"]
                if np.isfinite(r):
                    vals.append((asset, float(r)))
        vals.sort(key=lambda z: z[1])
        n = len(vals)
        for rank, (asset, _) in enumerate(vals):
            f[asset].loc[ts, "mom_rank"] = rank / (n - 1) if n > 1 else 0.5
    return f


def finite_row(r, keys):
    return all(np.isfinite(r.get(k, np.nan)) for k in keys)


def make_signals(f, cfg):
    (
        name, lookback, rank_long, rank_short, adx_min, rv_min,
        body_min, volz_min, close_pos_long, close_pos_short, atr_mult,
    ) = cfg
    signals = []
    common = [
        "atr", "ret6", "ret12", "ret24", "ret72", "rv_ratio", "vol_z",
        "body_atr", "range_atr", "close_pos", "ema20_slope",
        "h4_trend", "h4_adx", "h4_slope", "d1_regime", "d1_slope",
        "mom_rank",
    ]

    for asset, x in f.items():
        for i in range(250, len(x) - 1):
            ts = x.index[i]
            r = x.iloc[i]
            p = x.iloc[i - 1]
            if not finite_row(r, common):
                continue
            high_key = "high12" if lookback == 12 else "high24"
            low_key = "low12" if lookback == 12 else "low24"
            if not np.isfinite(r.get(high_key, np.nan)) or not np.isfinite(r.get(low_key, np.nan)):
                continue

            long_ctx = (
                r.d1_regime == 1 and r.h4_trend == 1 and
                r.h4_adx >= adx_min and r.h4_slope > 0 and r.d1_slope > 0
            )
            short_ctx = (
                r.d1_regime == -1 and r.h4_trend == -1 and
                r.h4_adx >= adx_min and r.h4_slope < 0 and r.d1_slope < 0
            )

            # Breakout is confirmed only by the signal candle close.
            long_break = r.close > r[high_key] and p.close <= p[high_key]
            short_break = r.close < r[low_key] and p.close >= p[low_key]
            long_quality = (
                r.mom_rank >= rank_long and r.ret6 > 0 and r.ret12 > 0 and
                r.ret24 > 0 and r.ret72 > 0 and r.body_atr >= body_min and
                r.close_pos >= close_pos_long and r.vol_z >= volz_min and
                r.rv_ratio >= rv_min
            )
            short_quality = (
                r.mom_rank <= rank_short and r.ret6 < 0 and r.ret12 < 0 and
                r.ret24 < 0 and r.ret72 < 0 and r.body_atr >= body_min and
                r.close_pos <= close_pos_short and r.vol_z >= volz_min and
                r.rv_ratio >= rv_min
            )

            # A single score is used only to require breadth of confirmation;
            # it is not used to manufacture a target win rate.
            long_score = (
                2.0 * long_ctx + 2.0 * long_break + 1.5 * long_quality +
                0.5 * (r.range_atr >= 1.0) + 0.5 * (r.ema20_slope > 0)
            )
            short_score = (
                2.0 * short_ctx + 2.0 * short_break + 1.5 * short_quality +
                0.5 * (r.range_atr >= 1.0) + 0.5 * (r.ema20_slope < 0)
            )

            if long_ctx and long_break and long_quality and long_score > short_score:
                signals.append({
                    "asset": asset, "signal_ts": ts, "family": "RELATIVE_BREAKOUT",
                    "side": "LONG", "score": float(long_score), "atr": float(r.atr),
                })
            elif short_ctx and short_break and short_quality and short_score > long_score:
                signals.append({
                    "asset": asset, "signal_ts": ts, "family": "RELATIVE_BREAKOUT",
                    "side": "SHORT", "score": float(short_score), "atr": float(r.atr),
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

    for ts in all_ts:
        closed_any = False
        for p in positions[:]:
            x = f[p["asset"]]
            if ts not in x.index or ts < p["entry_ts"]:
                continue
            row = x.loc[ts]
            hit_sl = row.low <= p["sl"] if p["side"] == "LONG" else row.high >= p["sl"]
            hit_tp = row.high >= p["tp"] if p["side"] == "LONG" else row.low <= p["tp"]
            if not (hit_sl or hit_tp):
                continue
            if hit_sl and hit_tp:
                outcome, exit_price = "LOSS", p["sl"]
            elif hit_sl:
                outcome, exit_price = "LOSS", p["sl"]
            else:
                outcome, exit_price = "WIN", p["tp"]

            price_ret = (
                (exit_price - p["entry"]) / p["entry"]
                if p["side"] == "LONG" else
                (p["entry"] - exit_price) / p["entry"]
            )
            gross = price_ret * TRADE_MARGIN * LEVERAGE
            fees = TRADE_MARGIN * LEVERAGE * FEE_RATE * 2.0
            pnl = gross - fees
            equity += pnl
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)
            trades.append({**p, "exit_ts": ts, "outcome": outcome, "pnl": pnl, "equity": equity})
            positions.remove(p)
            closed_any = True

        if closed_any:
            continue
        if len(positions) >= MAX_OPEN_POSITIONS:
            continue
        candidates = sorted(by_ts.get(ts, []), key=lambda z: (z["score"], abs(z["atr"])), reverse=True)
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
                sl, tp = entry - risk, entry + RR * risk
            else:
                sl, tp = entry + risk, entry - RR * risk
            positions.append({
                "asset": s["asset"], "family": s["family"], "side": s["side"],
                "signal_ts": ts, "entry_ts": entry_ts, "entry": entry,
                "sl": sl, "tp": tp, "score": s["score"],
            })

    return trades, positions, max_dd


def summarize(trades, open_positions, max_dd, days):
    n = len(trades)
    wins = sum(t["outcome"] == "WIN" for t in trades)
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    eq = INITIAL_EQUITY
    peak = eq
    dd = 0.0
    streak = max_streak = 0
    for t in sorted(trades, key=lambda z: z["exit_ts"]):
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
        streak = streak + 1 if t["outcome"] == "LOSS" else 0
        max_streak = max(max_streak, streak)
    return {
        "trades": n, "open": len(open_positions), "wins": wins,
        "losses": n - wins, "wr": 100.0 * wins / n if n else 0.0,
        "pf": gross_win / gross_loss if gross_loss else (math.inf if gross_win else 0.0),
        "pnl": sum(t["pnl"] for t in trades), "dd": min(dd, max_dd),
        "streak": max_streak, "trades_day": n / days if days else 0.0,
    }


def score_result(s):
    if s["trades"] < 100:
        return -1e9 + s["trades"]
    return (
        (s["wr"] - 40.0) * 4.0 +
        max(-3.0, min(3.0, s["pf"] - 1.0)) * 30.0 -
        max(0, s["streak"] - 4) * 5.0 +
        max(-5.0, min(5.0, s["pnl"] / 1000.0)) * 2.0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    print("=" * 96)
    print("HUNTER-V153 — VOLATILITY-ADAPTIVE RELATIVE BREAKOUT")
    print("15m raw -> 1H breakout / 4H trend context / 1D regime | no funding")
    print("=" * 96)

    all15 = ensure_data(Path(args.data_dir), args.days)
    f = add_cross_sectional_ranks(build_features(all15))

    configs = [
        ("V153_BALANCED", 12, 0.65, 0.35, 18, 0.90, 0.30, -0.25, 0.62, 0.38, 1.40),
        ("V153_STRICT_LEADERS", 12, 0.75, 0.25, 20, 0.90, 0.35, -0.10, 0.65, 0.35, 1.40),
        ("V153_WIDE_WINDOW", 24, 0.65, 0.35, 18, 0.90, 0.30, -0.25, 0.62, 0.38, 1.40),
        ("V153_HIGH_ADX", 12, 0.65, 0.35, 24, 0.90, 0.30, -0.25, 0.62, 0.38, 1.50),
        ("V153_EXPANSION", 12, 0.70, 0.30, 20, 1.05, 0.35, 0.00, 0.65, 0.35, 1.50),
        ("V153_TIGHT_RISK", 12, 0.70, 0.30, 20, 0.95, 0.35, -0.10, 0.65, 0.35, 1.20),
    ]

    print(f"\nTesting {len(configs)} causal V153 configurations...")
    results = []
    for cfg in configs:
        signals = make_signals(f, cfg)
        trades, open_positions, max_dd = execute_backtest(f, signals, cfg[-1])
        s = summarize(trades, open_positions, max_dd, args.days)
        s["name"] = cfg[0]
        s["signals"] = len(signals)
        s["score"] = score_result(s)
        s["eligible"] = (
            s["trades"] >= 100 and s["wr"] >= 50.0 and s["streak"] <= 4 and
            s["pf"] > 1.0 and s["pnl"] > 0
        )
        results.append(s)
        print(
            f"{cfg[0]:<22} trades={s['trades']:4d} WR={s['wr']:6.2f}% "
            f"PF={s['pf']:6.3f} PnL=${s['pnl']:>9,.2f} DD=${s['dd']:>9,.2f} "
            f"streak={s['streak']:2d} t/day={s['trades_day']:.2f} eligible={s['eligible']}"
        )

    ranked = sorted(results, key=lambda z: z["score"], reverse=True)
    print("\nTOP CONFIGURATIONS")
    for i, s in enumerate(ranked[:5], 1):
        print(
            f"{i}. {s['name']} | trades={s['trades']} WR={s['wr']:.2f}% "
            f"PF={s['pf']:.3f} PnL=${s['pnl']:,.2f} DD=${s['dd']:,.2f} streak={s['streak']}"
        )

    edge = any(s["eligible"] for s in results)
    print("\n" + "=" * 96)
    print(f"MEASURABLE_EDGE_PRESENT: {edge}")
    if not edge:
        print("No tested V153 configuration reached WR>=50%, max loss streak<=4, PF>1, positive PnL and >=100 trades.")
    else:
        print("At least one tested V153 configuration met all hard eligibility conditions.")
    print("=" * 96)


if __name__ == "__main__":
    main()
