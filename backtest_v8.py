#!/usr/bin/env python3
"""
HUNTER-V18-STAGE0
VOLATILITY EXHAUSTION / FAILED-EXTENSION REVERSAL — FAST DISCOVERY

Research hypothesis
-------------------
After an unusually large directional extension, price can mean-revert when:
1) the completed 1H candle is an ATR-normalized volatility expansion,
2) the extension over the prior 6H window is directionally extreme,
3) the expansion candle closes near its extreme (capitulation),
4) the NEXT completed 1H candle fails to continue the extreme and reclaims
   the expansion candle's midpoint,
5) the completed 4H regime is NOT strongly aligned with the impulse direction.

Entry is at the next 1H open after confirmation. Fixed RR=1:2.
No trailing, BE, timeout, pyramiding, overlap, or future-data filters.

This is a Stage-0 discovery test only. Three pre-registered stop distances
are reported side-by-side; none is selected by the script.

Integrity
---------
- XT USDT-M futures 1H direct data.
- Incomplete latest candle removed.
- All features are based only on completed candles.
- Signal candle t -> confirmation candle t+1 -> entry at t+2 open.
- One global position at a time.
- Same-candle SL+TP => LOSS.
- Fixed universe chosen before the test.
- No parameter optimization.
"""

import math
import os
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

# Pre-registered Stage-0 hypothesis parameters.
ATR_PERIOD = 14
IMPULSE_N = 6
MIN_IMPULSE_ATR = 1.50
MIN_EXTENSION_ATR = 1.25
MAX_BODY_FRACTION = 0.35
CAPITULATION_CLOSE = 0.20
RECLAIM_FRACTION = 0.50
MAX_4H_TREND_ATR = 0.35

# Sensitivity only; no winner is selected.
STOP_ATR_VALUES = (1.00, 1.25, 1.50)

CACHE_DIR = Path("data/xt_v18_stage0")
REPORT_DIR = Path("reports/xt_v18_stage0")


@dataclass
class Position:
    symbol: str
    side: int
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry: float
    sl: float
    tp: float


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
    print(
        f"[VALIDATE] {symbol} rows={len(x)} first={x.timestamp.iloc[0]} "
        f"last={x.timestamp.iloc[-1]} span={(x.timestamp.iloc[-1]-x.timestamp.iloc[0]).days}d"
    )
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

    x = pd.DataFrame(
        all_rows,
        columns=["ts", "open", "high", "low", "close", "volume"]
    )
    x["timestamp"] = pd.to_datetime(x.pop("ts"), unit="ms", utc=True)

    # Remove current incomplete 1H candle.
    cutoff = pd.Timestamp.now(tz="UTC").floor("1h")
    x = x[x.timestamp < cutoff]

    x = validate(x, symbol, start_ms, end_ms)
    x.to_csv(cache, index=False)
    return x


def true_range(x):
    prev = x["close"].shift(1)
    return pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev).abs(),
            (x["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def build_features(df):
    x = df.set_index("timestamp").sort_index().copy()
    x["tr"] = true_range(x)
    x["atr"] = x["tr"].rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # Signal candle expansion relative to ATR known before that candle closes.
    x["range"] = x["high"] - x["low"]
    x["range_atr"] = x["range"] / x["atr"].replace(0, np.nan)

    x["body"] = (x["close"] - x["open"]).abs()
    x["body_frac"] = x["body"] / x["range"].replace(0, np.nan)

    # 0 = close at low, 1 = close at high.
    x["close_loc"] = (x["close"] - x["low"]) / x["range"].replace(0, np.nan)

    # Directional extension over the previous completed impulse window.
    x["ret_6"] = x["close"].pct_change(IMPULSE_N, fill_method=None)
    x["prior_hi_6"] = x["high"].shift(1).rolling(IMPULSE_N, min_periods=IMPULSE_N).max()
    x["prior_lo_6"] = x["low"].shift(1).rolling(IMPULSE_N, min_periods=IMPULSE_N).min()

    # 4H features built causally from completed 1H data.
    h4 = (
        x[["open", "high", "low", "close"]]
        .resample("4h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    h4["ema20"] = h4["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    h4["ema50"] = h4["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    h4["atr"] = (
        pd.concat(
            [
                h4["high"] - h4["low"],
                (h4["high"] - h4["close"].shift(1)).abs(),
                (h4["low"] - h4["close"].shift(1)).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14, min_periods=14)
        .mean()
    )
    h4["trend_atr"] = (h4["ema20"] - h4["ema50"]) / h4["atr"].replace(0, np.nan)

    # Shift so an hourly candle can only see the most recently COMPLETED 4H bar.
    h4 = h4[["ema20", "ema50", "atr", "trend_atr"]].shift(1)
    x = x.join(h4.reindex(x.index, method="ffill"), rsuffix="_4h")

    # Confirmation candle must reject continuation and reclaim the impulse midpoint.
    x["impulse_mid"] = (x["high"].shift(1) + x["low"].shift(1)) / 2.0

    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    return x


def candidate(x, i):
    # i is the completed confirmation candle. The signal/extension candle is i-1.
    # Entry occurs at the OPEN of i+1, so no current/future candle is used.
    if i < 101 or i + 1 >= len(x):
        return 0

    conf = x.iloc[i]
    sig = x.iloc[i - 1]

    required = [
        sig["atr"], sig["range_atr"], sig["body_frac"], sig["close_loc"],
        sig["ret_6"], sig["prior_hi_6"], sig["prior_lo_6"],
        conf["trend_atr_4h"], sig["impulse_mid"], conf["close"],
    ]
    if any(pd.isna(v) for v in required):
        return 0

    # Signal candle must be an abnormal extension with capitulation-style shape.
    if sig["range_atr"] < MIN_IMPULSE_ATR:
        return 0
    if sig["body_frac"] > MAX_BODY_FRACTION:
        return 0

    extension_atr = MIN_EXTENSION_ATR * sig["atr"]

    # Long reversal: a downside extension breaks the prior 6H low, closes near
    # its low, and the next completed candle rejects continuation and reclaims
    # the extension candle midpoint.
    down_extension = (
        sig["ret_6"] <= -extension_atr / max(sig["close"], 1e-12)
        and sig["low"] < sig["prior_lo_6"]
        and sig["close_loc"] <= CAPITULATION_CLOSE
    )
    up_extension = (
        sig["ret_6"] >= extension_atr / max(sig["close"], 1e-12)
        and sig["high"] > sig["prior_hi_6"]
        and sig["close_loc"] >= 1.0 - CAPITULATION_CLOSE
    )

    long_confirm = (
        conf["low"] <= sig["low"] * 1.0005
        and conf["close"] > conf["open"]
        and conf["close"] >= sig["impulse_mid"]
        and conf["close"] > sig["close"]
    )
    short_confirm = (
        conf["high"] >= sig["high"] * 0.9995
        and conf["close"] < conf["open"]
        and conf["close"] <= sig["impulse_mid"]
        and conf["close"] < sig["close"]
    )

    # Do not fade a strongly aligned completed 4H trend.
    neutral_4h = abs(conf["trend_atr_4h"]) <= MAX_4H_TREND_ATR
    if not neutral_4h:
        return 0

    long_ok = down_extension and long_confirm
    short_ok = up_extension and short_confirm

    if long_ok and not short_ok:
        return 1
    if short_ok and not long_ok:
        return -1
    return 0


def simulate(features, stop_mult):
    # Merge all symbols into one chronological event stream.
    streams = {s: build_features(df) for s, df in features.items()}
    events = []
    for symbol, x in streams.items():
        for i in range(len(x) - 1):
            sig = candidate(x, i)
            if sig:
                events.append((x.index[i], symbol, i, sig))
    events.sort(key=lambda z: z[0])

    cash = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    position = None
    trades = []
    taken_signals = 0

    for signal_ts, symbol, i, side in events:
        x = streams[symbol]

        # First resolve an existing position through candles strictly after entry.
        if position is not None:
            if signal_ts <= position["entry_ts"]:
                continue

        # Entry occurs at next 1H OPEN after confirmation candle.
        entry_idx = i + 1
        if entry_idx >= len(x):
            continue
        entry_ts = x.index[entry_idx]
        entry_raw = float(x.iloc[entry_idx]["open"])

        # Never enter while another position is active.
        if position is not None:
            continue

        atr_now = float(x.iloc[i]["atr"])
        if not np.isfinite(atr_now) or atr_now <= 0:
            continue

        # Conservative marketable entry/slippage.
        entry = entry_raw * (1.0 + SLIPPAGE if side == 1 else 1.0 - SLIPPAGE)
        stop_dist = stop_mult * atr_now

        if side == 1:
            sl = entry - stop_dist
            tp = entry + RR * stop_dist
        else:
            sl = entry + stop_dist
            tp = entry - RR * stop_dist

        position = {
            "symbol": symbol,
            "side": side,
            "signal_ts": signal_ts,
            "entry_ts": entry_ts,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "stop_dist": stop_dist,
            "entry_idx": entry_idx,
        }
        taken_signals += 1

        # Scan forward until exit. Because only one position is allowed,
        # future signals are ignored while this position is open.
        for j in range(entry_idx, len(x)):
            bar = x.iloc[j]
            ts = x.index[j]

            if side == 1:
                hit_sl = float(bar["low"]) <= sl
                hit_tp = float(bar["high"]) >= tp
            else:
                hit_sl = float(bar["high"]) >= sl
                hit_tp = float(bar["low"]) <= tp

            if not (hit_sl or hit_tp):
                continue

            # Same candle SL+TP => LOSS.
            win = hit_tp and not hit_sl
            exit_px = tp if win else sl

            # Trading costs on both entry and exit.
            gross = (
                NOTIONAL * (exit_px - entry) / entry
                if side == 1
                else NOTIONAL * (entry - exit_px) / entry
            )
            fees = NOTIONAL * FEE_RATE * 2.0
            pnl = gross - fees

            cash += pnl
            peak = max(peak, cash)
            dd = peak - cash
            max_dd = max(max_dd, dd)

            trades.append({
                "signal_ts": signal_ts,
                "entry_ts": entry_ts,
                "exit_ts": ts,
                "symbol": symbol,
                "side": "LONG" if side == 1 else "SHORT",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "win": int(win),
                "pnl": pnl,
            })
            position = None
            break

    # Do not count an unclosed position as a completed trade.
    return pd.DataFrame(trades), taken_signals, max_dd


def summarize(df):
    if df.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "wr": 0.0,
            "pf": 0.0, "pnl": 0.0, "dd": 0.0, "dd_pct": 0.0,
            "streak": 0, "expectancy": 0.0, "final": INITIAL_CAPITAL
        }

    pnl = df["pnl"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl <= 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    equity = INITIAL_CAPITAL + pnl.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    max_dd = float(dd.max()) if len(dd) else 0.0
    max_dd_pct = float((dd / peak.replace(0, np.nan)).max() * 100.0) if len(dd) else 0.0

    streak = 0
    cur = 0
    for v in pnl:
        if v <= 0:
            cur += 1
            streak = max(streak, cur)
        else:
            cur = 0

    return {
        "trades": len(df),
        "wins": wins,
        "losses": losses,
        "wr": 100.0 * wins / len(df),
        "pf": pf,
        "pnl": float(pnl.sum()),
        "dd": max_dd,
        "dd_pct": max_dd_pct,
        "streak": streak,
        "expectancy": float(pnl.mean()),
        "final": float(equity.iloc[-1]),
    }


def print_report(stop_mult, trades, raw_candidates, taken, max_dd):
    s = summarize(trades)
    print("\n" + "=" * 72)
    print(f"STAGE-0 STOP = {stop_mult:.2f} ATR | RR = 1:2")
    print("=" * 72)
    print(f"Raw candidate signals   : {raw_candidates}")
    print(f"Taken entries           : {taken}")
    print(f"Closed trades           : {s['trades']}")
    print(f"Wins                    : {s['wins']}")
    print(f"Losses                  : {s['losses']}")
    print(f"Win Rate                : {s['wr']:.2f}%")
    print(f"Profit Factor           : {s['pf']:.4f}")
    print(f"Net PnL                 : ${s['pnl']:,.2f}")
    print(f"Max Drawdown            : ${s['dd']:,.2f}")
    print(f"Max Drawdown %          : {s['dd_pct']:.2f}%")
    print(f"Max Loss Streak         : {s['streak']}")
    print(f"Expectancy / Trade      : ${s['expectancy']:.2f}")
    print(f"Final Equity            : ${s['final']:,.2f}")
    if not trades.empty:
        side = (
            trades.groupby("side")
            .agg(
                trades=("pnl", "size"),
                wins=("win", "sum"),
                pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        side["wr"] = 100.0 * side["wins"] / side["trades"]
        print("\nBY SIDE")
        print(side.to_string(index=False, formatters={
            "pnl": lambda v: f"{v:.2f}",
            "wr": lambda v: f"{v:.2f}",
        }))


def main():
    print("HUNTER-V18-STAGE0 — VOLATILITY EXHAUSTION / FAILED-EXTENSION REVERSAL")
    print(
        f"Lookback={LOOKBACK_DAYS}d + warmup={WARMUP_DAYS}d | "
        f"1H direct XT futures"
    )
    print(
        f"RR=1:2 | margin=${MARGIN:.0f} | notional=${NOTIONAL:.0f} | "
        f"fee={FEE_RATE} | slippage={SLIPPAGE}"
    )
    print(
        "Hypothesis: abnormal extension + capitulation + next-bar reclaim + "
        "neutral 4H regime"
    )
    print("No parameter optimization. Stop sensitivity is reported, not selected.")

    refresh = os.getenv("XT_REFRESH", "0") == "1"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = {s: fetch_xt(s, refresh=refresh) for s in SYMBOLS}
    common_start = max(df["timestamp"].min() for df in raw.values())
    common_end = min(df["timestamp"].max() for df in raw.values())
    print(f"COMMON 1H RANGE: {common_start} -> {common_end}")

    # Only the common range is used.
    raw = {
        s: df[(df["timestamp"] >= common_start) & (df["timestamp"] <= common_end)].copy()
        for s, df in raw.items()
    }

    results = []
    for stop_mult in STOP_ATR_VALUES:
        trades, taken, max_dd = simulate(raw, stop_mult)
        raw_candidates = 0
        for s, df in raw.items():
            x = build_features(df)
            raw_candidates += sum(candidate(x, i) != 0 for i in range(len(x) - 1))

        print_report(stop_mult, trades, raw_candidates, taken, max_dd)
        out = REPORT_DIR / f"stage0_trades_stop_{stop_mult:.2f}.csv"
        trades.to_csv(out, index=False)
        results.append(summarize(trades))

    # Discovery gate. This is intentionally conservative and does not pick
    # a stop. All pre-registered sensitivities must show some positive edge.
    enough_activity = all(r["trades"] >= 150 for r in results)
    positive_edge = all(r["pf"] > 1.0 and r["expectancy"] > 0 for r in results)
    reasonable_wr = max(r["wr"] for r in results) >= 40.0

    print("\n" + "=" * 72)
    print("STAGE-0 DECISION")
    print("=" * 72)
    print("This stage does NOT select a best stop.")
    print("It asks whether the fixed hypothesis deserves expensive walk-forward.")
    print(f"Activity >=150 trades across all stops : {'PASS' if enough_activity else 'FAIL'}")
    print(f"Positive PF + expectancy across all    : {'PASS' if positive_edge else 'FAIL'}")
    print(f"At least one stop WR >=40%             : {'PASS' if reasonable_wr else 'FAIL'}")

    if enough_activity and positive_edge and reasonable_wr:
        print("DISCOVERY_STATUS: PASS — eligible for robust walk-forward")
    else:
        print("DISCOVERY_STATUS: REJECT — do NOT spend hours on walk-forward yet")
    print("=" * 72)


if __name__ == "__main__":
    main()
