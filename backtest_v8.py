#!/usr/bin/env python3
"""
HUNTER-V17-STAGE0
TIME-SERIES MOMENTUM / TREND PERSISTENCE — FAST DISCOVERY

Purpose
-------
This is NOT the final walk-forward strategy. It is a fast, pre-registered
hypothesis test designed to answer one question before spending hours on
optimization:

    Does a simple, economically motivated time-series trend-persistence
    hypothesis produce enough 1:2-RR trades with positive expectancy after
    fees/slippage on real XT USDT-M perpetual data?

Hypothesis
----------
1) The instrument has persistent directional movement over 24h and 72h.
2) Price confirms the direction by breaking the prior completed 24h range.
3) The move is not a low-efficiency/noise move.
4) The higher-timeframe 4H trend agrees with the direction.
5) Entry is at the NEXT completed 1H candle open.
6) Fixed RR = 1:2. No trailing, BE, timeout, pyramiding or overlap.

No parameter optimization is performed. Three fixed stop-distance sensitivity
runs (1.0 / 1.25 / 1.5 ATR) are reported side-by-side and NONE is selected as a
winner. This is a robustness check, not a tuning exercise.

Integrity
---------
- XT futures data only.
- 1H candles are used directly; incomplete latest candle is removed.
- Features use current/past candles only.
- Signal at completed 1H candle t -> entry at t+1 open.
- One global position at a time.
- If SL and TP occur in the same candle, LOSS is assumed conservatively.
- No future symbol selection. Universe is fixed before the test.
- No walk-forward selection in this stage.
- If this stage fails, V17 is rejected before expensive WF.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "sui_usdt", "avax_usdt",
    "near_usdt", "ada_usdt", "bnb_usdt", "apt_usdt", "crv_usdt",
    "ondo_usdt", "pendle_usdt", "icp_usdt", "wif_usdt",
]

FUTURES_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"

LOOKBACK_DAYS = 455
WARMUP_DAYS = 90
INTERVAL_MS = 60 * 60 * 1000
LIMIT = 1500

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE
RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# Pre-registered hypothesis parameters. Do not optimize these in Stage-0.
BREAKOUT_N = 24
RETURN_FAST = 12
RETURN_SLOW = 72
EFF_N = 12
MIN_EFF = 0.35
MIN_RETURN_FAST = 0.0025
MIN_RETURN_SLOW = 0.0050

# Fixed stop sensitivity only. Every run is reported; no winner is selected.
STOP_ATR_VALUES = (1.00, 1.25, 1.50)

CACHE_DIR = Path("data/xt_v17_stage0")
REPORT_DIR = Path("reports/xt_v17_stage0")


@dataclass
class Position:
    symbol: str
    side: int
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry: float
    sl: float
    tp: float
    atr: float


def payload_rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("result", "data", "rows", "list"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return None


def parse_row(row):
    try:
        if isinstance(row, dict):
            ts = row.get("t", row.get("timestamp"))
            o = row.get("o", row.get("open"))
            h = row.get("h", row.get("high"))
            l = row.get("l", row.get("low"))
            c = row.get("c", row.get("close"))
            v = row.get("a", row.get("volume"))
            if None in (ts, o, h, l, c, v):
                return None
            return int(ts), float(o), float(h), float(l), float(c), float(v)
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            return int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
    except Exception:
        return None
    return None


def validate(df, symbol, start_ms, end_ms):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    x = x[(x.timestamp >= pd.to_datetime(start_ms, unit="ms", utc=True)) &
          (x.timestamp <= pd.to_datetime(end_ms, unit="ms", utc=True))].copy()
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if x[cols].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in OHLCV")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol}: non-positive price")
    if (x.volume < 0).any():
        raise RuntimeError(f"{symbol}: negative volume")
    diffs = x.timestamp.diff().dropna().dt.total_seconds().div(3600.0)
    gaps = diffs[diffs > 1.0 + 1e-9]
    if len(gaps):
        raise RuntimeError(f"{symbol}: {len(gaps)} futures 1H gaps; no OHLCV fabrication allowed")
    if len(x) < 9000:
        raise RuntimeError(f"{symbol}: only {len(x)} 1H rows")
    print(f"[VALIDATE] {symbol} rows={len(x)} first={x.timestamp.iloc[0]} last={x.timestamp.iloc[-1]} span={(x.timestamp.iloc[-1]-x.timestamp.iloc[0]).days}d")
    return x.reset_index(drop=True)


def fetch_xt(symbol, refresh=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}.csv"
    now = int(time.time() * 1000)
    start_raw = now - int((LOOKBACK_DAYS + WARMUP_DAYS) * 86400 * 1000)
    start_ms = ((start_raw + INTERVAL_MS - 1) // INTERVAL_MS) * INTERVAL_MS
    end_ms = (now // INTERVAL_MS) * INTERVAL_MS - 1

    if cache.exists() and not refresh:
        return validate(pd.read_csv(cache), symbol, start_ms, end_ms)

    session = requests.Session()
    cursor = start_ms
    all_rows = []
    page = 0
    while cursor <= end_ms:
        page += 1
        window_end = min(end_ms, cursor + LIMIT * INTERVAL_MS - 1)
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": cursor,
            "endTime": window_end,
            "limit": LIMIT,
        }
        payload = None
        err = None
        for attempt in range(4):
            try:
                r = session.get(FUTURES_URL, params=params, timeout=30)
                r.raise_for_status()
                payload = payload_rows(r.json())
                if payload is None:
                    raise RuntimeError("XT response contains no kline list")
                break
            except Exception as exc:
                err = exc
                time.sleep(1.0 + attempt)
        if payload is None:
            raise RuntimeError(f"{symbol}: page {page} failed: {err}")

        parsed = [p for p in (parse_row(r) for r in payload) if p is not None]
        inside = [p for p in parsed if cursor <= p[0] <= window_end]
        if not inside:
            if window_end >= end_ms:
                break
            raise RuntimeError(f"{symbol}: page {page} returned no rows")
        all_rows.extend(inside)
        mx = max(p[0] for p in inside)
        nxt = mx + INTERVAL_MS
        if nxt <= cursor:
            raise RuntimeError(f"{symbol}: pagination stalled")
        cursor = nxt
        if page % 4 == 0:
            print(f"[FETCH] {symbol} page={page} rows={len(all_rows)}")
        if mx >= end_ms:
            break
        time.sleep(0.05)

    x = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    x["timestamp"] = pd.to_datetime(x.pop("ts"), unit="ms", utc=True)
    # Remove the current incomplete 1H candle.
    cutoff = pd.Timestamp.now(tz="UTC").floor("1h")
    x = x[x.timestamp < cutoff]
    x = validate(x, symbol, start_ms, end_ms)
    x.to_csv(cache, index=False)
    return x


def atr(x, n=14):
    prev = x.close.shift(1)
    tr = pd.concat([
        x.high - x.low,
        (x.high - prev).abs(),
        (x.low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def build_features(df):
    x = df.set_index("timestamp").sort_index().copy()
    x["atr"] = atr(x, 14)
    x["ret_fast"] = x.close.pct_change(RETURN_FAST, fill_method=None)
    x["ret_slow"] = x.close.pct_change(RETURN_SLOW, fill_method=None)
    x["range_hi"] = x.high.shift(1).rolling(BREAKOUT_N, min_periods=BREAKOUT_N).max()
    x["range_lo"] = x.low.shift(1).rolling(BREAKOUT_N, min_periods=BREAKOUT_N).min()
    delta = x.close.diff()
    x["eff"] = (x.close - x.close.shift(EFF_N)).abs() / delta.abs().rolling(EFF_N).sum()

    # Completed 4H bars only. The shift prevents using the still-forming 4H bar.
    h4 = pd.DataFrame({
        "open": x.open.resample("4h").first(),
        "high": x.high.resample("4h").max(),
        "low": x.low.resample("4h").min(),
        "close": x.close.resample("4h").last(),
    }).dropna()
    h4["ema20"] = h4.close.ewm(span=20, adjust=False).mean()
    h4["ema50"] = h4.close.ewm(span=50, adjust=False).mean()
    h4["slope"] = h4.ema20.pct_change(3, fill_method=None)
    h4["trend"] = np.where((h4.ema20 > h4.ema50) & (h4.slope > 0), 1,
                           np.where((h4.ema20 < h4.ema50) & (h4.slope < 0), -1, 0))
    h4a = h4.shift(1).reindex(x.index, method="ffill")
    x["h4_trend"] = h4a.trend
    return x


def signal_at(ts, x):
    if ts not in x.index:
        return None
    r = x.loc[ts]
    vals = [r.atr, r.ret_fast, r.ret_slow, r.range_hi, r.range_lo, r.eff, r.h4_trend]
    if not all(np.isfinite(v) for v in vals):
        return None

    # Directional persistence + prior-range breakout.
    long_ok = (
        r.close > r.range_hi and
        r.ret_fast >= MIN_RETURN_FAST and
        r.ret_slow >= MIN_RETURN_SLOW and
        r.eff >= MIN_EFF and
        r.h4_trend == 1
    )
    short_ok = (
        r.close < r.range_lo and
        r.ret_fast <= -MIN_RETURN_FAST and
        r.ret_slow <= -MIN_RETURN_SLOW and
        r.eff >= MIN_EFF and
        r.h4_trend == -1
    )
    if long_ok == short_ok:
        return None
    side = 1 if long_ok else -1
    return {"side": side, "atr": float(r.atr), "ret_fast": float(r.ret_fast),
            "ret_slow": float(r.ret_slow), "eff": float(r.eff),
            "h4_trend": int(r.h4_trend)}


def entry_price(openp, side):
    return openp * (1 + SLIPPAGE * side)


def exit_price(price, side):
    return price * (1 - SLIPPAGE * side)


def pnl_for(side, entry, exitp):
    gross = (exitp - entry) / entry * NOTIONAL * side
    fees = NOTIONAL * FEE_RATE * 2.0
    return gross - fees


def common_times(frames):
    common = None
    for x in frames.values():
        s = set(x.index)
        common = s if common is None else common.intersection(s)
    return pd.DatetimeIndex(sorted(common))


def run(frames, stop_atr):
    times = common_times(frames)
    start = times.min()
    end = times.max() + pd.Timedelta(hours=1)
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    active = None
    trades = []
    last_close_ts = None
    signals = 0

    for ts in times:
        # First resolve an existing position using this completed candle.
        if active is not None:
            x = frames[active.symbol]
            b = x.loc[ts]
            side = active.side
            sl_hit = b.low <= active.sl if side == 1 else b.high >= active.sl
            tp_hit = b.high >= active.tp if side == 1 else b.low <= active.tp
            if sl_hit or tp_hit:
                # Conservative ambiguity rule: SL wins if both are touched.
                win = bool(tp_hit and not sl_hit)
                raw = active.tp if win else active.sl
                xp = exit_price(raw, side)
                pnl = max(-MARGIN, pnl_for(side, active.entry, xp))
                equity = max(0.0, equity + pnl)
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
                trades.append({
                    "symbol": active.symbol, "side": "LONG" if side == 1 else "SHORT",
                    "signal_ts": active.signal_ts, "entry_ts": active.entry_ts,
                    "exit_ts": ts, "entry": active.entry, "exit": xp,
                    "outcome": "WIN" if win else "LOSS", "pnl": pnl,
                    "ret_fast": active.ret_fast, "ret_slow": active.ret_slow,
                    "eff": active.eff, "atr": active.atr,
                })
                last_close_ts = ts
                active = None
                continue

        if active is not None or (last_close_ts is not None and ts <= last_close_ts):
            continue
        if equity < MARGIN:
            continue

        # Evaluate all symbols on the SAME completed timestamp.
        candidates = []
        for sym, x in frames.items():
            s = signal_at(ts, x)
            if s is not None:
                candidates.append((sym, s))
        if not candidates:
            continue
        signals += len(candidates)

        # Deterministic fixed ranking: strongest absolute slow return.
        # This is not an optimized rank floor; it only chooses one trade when
        # multiple symbols signal simultaneously.
        candidates.sort(key=lambda z: (abs(z[1]["ret_slow"]), abs(z[1]["eff"]), z[0]), reverse=True)
        sym, s = candidates[0]
        x = frames[sym]
        pos = x.index.get_loc(ts)
        if pos + 1 >= len(x):
            continue
        entry_ts = x.index[pos + 1]
        if entry_ts > times.max():
            continue
        ep = entry_price(float(x.iloc[pos + 1].open), s["side"])
        dist = s["atr"] * stop_atr
        if not np.isfinite(dist) or dist <= 0:
            continue
        active = Position(
            symbol=sym, side=s["side"], signal_ts=ts, entry_ts=entry_ts,
            entry=ep, sl=ep - dist * s["side"], tp=ep + dist * RR * s["side"], atr=s["atr"],
        )
        # Store extra attributes without changing the dataclass schema.
        active.ret_fast = s["ret_fast"]
        active.ret_slow = s["ret_slow"]
        active.eff = s["eff"]

    vals = pd.Series([t["pnl"] for t in trades], dtype=float)
    wins = int((vals > 0).sum())
    losses = int((vals <= 0).sum())
    gp = float(vals[vals > 0].sum()) if wins else 0.0
    gl = float(-vals[vals <= 0].sum()) if losses else 0.0
    pf = gp / gl if gl else (math.inf if wins else 0.0)
    wr = 100.0 * wins / len(vals) if len(vals) else 0.0
    streak = cur = 0
    for v in vals:
        if v <= 0:
            cur += 1
            streak = max(streak, cur)
        else:
            cur = 0
    return {
        "stop_atr": stop_atr, "signals": signals, "trades": len(vals),
        "wins": wins, "losses": losses, "wr": wr, "pf": pf,
        "pnl": float(vals.sum()) if len(vals) else 0.0, "dd": max_dd,
        "dd_pct": 100 * max_dd / peak if peak > 0 else 100.0,
        "streak": streak, "expectancy": float(vals.mean()) if len(vals) else 0.0,
        "final_equity": equity, "trades_log": trades,
    }


def print_report(r):
    print("\n" + "=" * 72)
    print(f"STAGE-0 STOP = {r['stop_atr']:.2f} ATR | RR = 1:{RR:.0f}")
    print("=" * 72)
    for k, label in [
        ("signals", "Raw candidate signals"), ("trades", "Closed trades"),
        ("wins", "Wins"), ("losses", "Losses"), ("wr", "Win Rate"),
        ("pf", "Profit Factor"), ("pnl", "Net PnL"), ("dd", "Max Drawdown"),
        ("dd_pct", "Max Drawdown %"), ("streak", "Max Loss Streak"),
        ("expectancy", "Expectancy / Trade"), ("final_equity", "Final Equity"),
    ]:
        v = r[k]
        if k in {"pnl", "dd", "expectancy", "final_equity"}:
            print(f"{label:<24}: ${v:,.2f}")
        elif k in {"wr", "dd_pct"}:
            print(f"{label:<24}: {v:.2f}%")
        elif k == "pf":
            print(f"{label:<24}: {v:.4f}")
        else:
            print(f"{label:<24}: {v}")


def save_trades(r):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(r["trades_log"]).to_csv(REPORT_DIR / f"trades_stop_{r['stop_atr']:.2f}.csv", index=False)


def main():
    refresh = os.getenv("XT_REFRESH", "0") == "1"
    print("HUNTER-V17-STAGE0 — TIME-SERIES MOMENTUM / TREND PERSISTENCE")
    print(f"Lookback={LOOKBACK_DAYS}d + warmup={WARMUP_DAYS}d | 1H direct XT futures")
    print(f"RR=1:{RR:.0f} | margin=${MARGIN:.0f} | notional=${NOTIONAL:.0f} | fee={FEE_RATE} | slippage={SLIPPAGE}")
    print(f"Hypothesis: breakout {BREAKOUT_N}h + returns {RETURN_FAST}h/{RETURN_SLOW}h + efficiency {MIN_EFF} + completed 4H trend")
    print("No parameter optimization. Stop sensitivity is reported, not selected.")

    raw = {}
    for sym in SYMBOLS:
        raw[sym] = fetch_xt(sym, refresh=refresh)

    frames = {sym: build_features(df) for sym, df in raw.items()}
    times = common_times(frames)
    if len(times) < 9000:
        raise RuntimeError(f"Common timestamp set too small: {len(times)}")
    print(f"COMMON 1H RANGE: {times[0]} -> {times[-1]} | bars={len(times)}")

    all_results = []
    for stop in STOP_ATR_VALUES:
        r = run(frames, stop)
        print_report(r)
        save_trades(r)
        all_results.append(r)

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "trades_log"} for r in all_results])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "stage0_summary.csv", index=False)

    print("\n================ STAGE-0 DECISION ================")
    print("This stage does NOT select a best stop. It asks whether the fixed hypothesis")
    print("has enough activity and a positive net edge across all pre-registered stops.")
    if (summary.trades >= 60).all() and (summary.pf > 1.0).all() and (summary.expectancy > 0).all():
        print("DISCOVERY_STATUS: PROMISING — eligible for expensive walk-forward research")
    else:
        print("DISCOVERY_STATUS: REJECT — do NOT spend hours on walk-forward yet")
    print("==================================================")


if __name__ == "__main__":
    main()
