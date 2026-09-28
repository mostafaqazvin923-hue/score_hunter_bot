#!/usr/bin/env python3
"""
HUNTER-V20-STAGE0
VOLUME / ORDER-FLOW IMBALANCE + BREAKOUT ACCEPTANCE

Stage-0 discovery only. No parameter optimization.

Hypothesis
----------
A directional breakout has more information when:
1) the breakout closes beyond a pre-existing range,
2) participation is unusually high relative to its own past,
3) the candle's signed-volume proxy agrees with the breakout direction,
4) the next completed candle accepts the new price area instead of closing
   back inside the old range,
5) the completed 4H regime is directionally aligned.

Entry: next 1H OPEN after the confirmation candle.
RR: 1:2.
No trailing, breakeven, timeout, pyramiding, or global-position lock.

Important:
- "Order flow" here is a causal OHLCV-derived signed-volume proxy, not true
  exchange trade-level bid/ask delta. The script must not imply otherwise.
- One open trade per SYMBOL. Different symbols may be open simultaneously.
- A new entry is forbidden on the same candle on which a previous trade exits.
- Same-candle SL+TP is classified as LOSS.
- All indicators/signals use completed candles only.
- Latest incomplete candle is removed.
- Fixed universe and fixed hypothesis parameters are declared before the run.
- Three stop distances are sensitivity checks; none is selected by the script.

Capital model
-------------
Initial equity = $1,000
Margin/trade = $100
Leverage = 50x
Notional/trade = $5,000
At most 10 simultaneous positions from the stated margin budget.
A new position requires at least $100 of current equity not already reserved
by other open positions. Margin is not deducted from realized PnL; it is a
capital-allocation constraint only.
"""

import os
import time
import hashlib
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
MAX_OPEN_POSITIONS = int(INITIAL_CAPITAL // MARGIN)

# Pre-registered hypothesis parameters.
ATR_PERIOD = 14
RANGE_N = 24
VOLUME_N = 48
FLOW_N = 48
MIN_BREAKOUT_ATR = 0.15
MIN_VOLUME_Z = 1.00
MIN_FLOW_Z = 0.75
MIN_BODY_FRACTION = 0.50
MIN_CLOSE_LOC = 0.75
MIN_ACCEPT_CLOSE_LOC = 0.55
MIN_ACCEPT_VOLUME_RATIO = 0.70
MIN_4H_TREND_ATR = 0.05

# Sensitivity only. The script never selects a winner.
STOP_ATR_VALUES = (1.00, 1.25, 1.50)

CACHE_DIR = Path("data/xt_v20_stage0")
REPORT_DIR = Path("reports/xt_v20_stage0")


@dataclass
class Position:
    symbol: str
    side: int
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry: float
    sl: float
    tp: float
    pnl: float
    win: bool


def payload_rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("result", "data", "rows", "list"):
            if isinstance(obj.get(key), list):
                return obj[key]
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
            return (
                int(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5])
            )
    except Exception:
        return None
    return None


def validate(df, symbol, start_ms, end_ms):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    x = x[
        (x["timestamp"] >= pd.to_datetime(start_ms, unit="ms", utc=True))
        & (x["timestamp"] <= pd.to_datetime(end_ms, unit="ms", utc=True))
    ].copy()

    cols = ["open", "high", "low", "close", "volume"]
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if x[cols].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in OHLCV")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol}: non-positive price")
    if (x["volume"] < 0).any():
        raise RuntimeError(f"{symbol}: negative volume")

    gaps = x["timestamp"].diff().dropna().dt.total_seconds().div(3600.0)
    if (gaps > 1.0 + 1e-9).any():
        count = int((gaps > 1.0 + 1e-9).sum())
        raise RuntimeError(
            f"{symbol}: {count} futures 1H gaps; no OHLCV fabrication allowed"
        )

    minimum_rows = int((LOOKBACK_DAYS + WARMUP_DAYS) * 24 * 0.95)
    if len(x) < minimum_rows:
        raise RuntimeError(f"{symbol}: only {len(x)} 1H rows")

    print(
        f"[VALIDATE] {symbol} rows={len(x)} "
        f"first={x['timestamp'].iloc[0]} last={x['timestamp'].iloc[-1]}"
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
        error = None
        for attempt in range(4):
            try:
                response = session.get(FUTURES_URL, params=params, timeout=30)
                response.raise_for_status()
                payload = payload_rows(response.json())
                if payload is None:
                    raise RuntimeError("XT response contains no kline list")
                break
            except Exception as exc:
                error = exc
                time.sleep(1.0 + attempt)

        if payload is None:
            raise RuntimeError(f"{symbol}: page {page} failed: {error}")

        parsed = [p for p in (parse_row(r) for r in payload) if p is not None]
        inside = [p for p in parsed if cursor <= p[0] <= window_end]

        if not inside:
            if window_end >= end_ms:
                break
            raise RuntimeError(f"{symbol}: page {page} returned no rows")

        all_rows.extend(inside)
        max_ts = max(p[0] for p in inside)
        next_cursor = max_ts + INTERVAL_MS

        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination stalled")

        cursor = next_cursor
        if page % 4 == 0:
            print(f"[FETCH] {symbol} page={page} rows={len(all_rows)}")
        if max_ts >= end_ms:
            break
        time.sleep(0.05)

    x = pd.DataFrame(
        all_rows,
        columns=["ts", "open", "high", "low", "close", "volume"]
    )
    x["timestamp"] = pd.to_datetime(x.pop("ts"), unit="ms", utc=True)

    # Current candle is incomplete and must never enter the test.
    cutoff = pd.Timestamp.now(tz="UTC").floor("1h")
    x = x[x["timestamp"] < cutoff]

    x = validate(x, symbol, start_ms, end_ms)
    x.to_csv(cache, index=False)
    return x


def true_range(x):
    prev_close = x["close"].shift(1)
    return pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def build_features(df):
    x = df.set_index("timestamp").sort_index().copy()

    x["tr"] = true_range(x)
    x["atr"] = x["tr"].rolling(
        ATR_PERIOD, min_periods=ATR_PERIOD
    ).mean()

    x["range"] = (x["high"] - x["low"]).clip(lower=0.0)
    safe_range = x["range"].replace(0, np.nan)
    x["body_frac"] = (x["close"] - x["open"]).abs() / safe_range
    x["close_loc"] = (x["close"] - x["low"]) / safe_range

    # Prior range excludes the current candle by construction.
    x["prior_hi"] = x["high"].shift(1).rolling(
        RANGE_N, min_periods=RANGE_N
    ).max()
    x["prior_lo"] = x["low"].shift(1).rolling(
        RANGE_N, min_periods=RANGE_N
    ).min()

    # Participation statistics are calculated only from candles before the
    # current candle. Current volume/flow is compared against past behavior.
    vol_mean = x["volume"].shift(1).rolling(
        VOLUME_N, min_periods=VOLUME_N
    ).mean()
    vol_std = x["volume"].shift(1).rolling(
        VOLUME_N, min_periods=VOLUME_N
    ).std(ddof=0)
    x["volume_z"] = (x["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # OHLCV signed-flow proxy:
    # volume * normalized candle direction. This is NOT true bid/ask delta.
    x["flow_proxy"] = (
        x["volume"] * (x["close"] - x["open"]) / safe_range
    )
    flow_mean = x["flow_proxy"].shift(1).rolling(
        FLOW_N, min_periods=FLOW_N
    ).mean()
    flow_std = x["flow_proxy"].shift(1).rolling(
        FLOW_N, min_periods=FLOW_N
    ).std(ddof=0)
    x["flow_z"] = (
        (x["flow_proxy"] - flow_mean) / flow_std.replace(0, np.nan)
    )

    # Completed 4H regime. The one-bar shift is critical: at a 1H timestamp,
    # only the already-completed 4H candle may influence the signal.
    h4 = x[["open", "high", "low", "close"]].resample(
        "4h", label="right", closed="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    h4["ema20"] = h4["close"].ewm(
        span=20, adjust=False, min_periods=20
    ).mean()
    h4["ema50"] = h4["close"].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    h4["tr"] = true_range(h4)
    h4["atr"] = h4["tr"].rolling(
        ATR_PERIOD, min_periods=ATR_PERIOD
    ).mean()
    h4["trend_atr"] = (
        (h4["ema20"] - h4["ema50"])
        / h4["atr"].replace(0, np.nan)
    )

    h4 = h4[
        ["ema20", "ema50", "atr", "trend_atr"]
    ].shift(1).rename(columns={
        "ema20": "ema20_4h",
        "ema50": "ema50_4h",
        "atr": "atr_4h",
        "trend_atr": "trend_atr_4h",
    })

    x = x.join(h4.reindex(x.index, method="ffill"))
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    return x


def candidate(x, i):
    """
    i = completed confirmation candle.
    Breakout candle = i-1.
    Entry = i+1 OPEN.

    No feature in this function reads any candle after i.
    """
    if i < 120 or i + 1 >= len(x):
        return 0

    breakout = x.iloc[i - 1]
    confirm = x.iloc[i]

    required = [
        breakout["atr"],
        breakout["prior_hi"],
        breakout["prior_lo"],
        breakout["volume_z"],
        breakout["flow_z"],
        breakout["body_frac"],
        breakout["close_loc"],
        confirm["close"],
        confirm["open"],
        confirm["high"],
        confirm["low"],
        confirm["close_loc"],
        confirm["volume"],
        confirm["atr"],
        confirm["trend_atr_4h"],
    ]
    if any(pd.isna(v) for v in required):
        return 0

    atr = float(breakout["atr"])
    if atr <= 0:
        return 0

    long_breakout = (
        breakout["close"] > breakout["prior_hi"]
        and (breakout["close"] - breakout["prior_hi"]) >= MIN_BREAKOUT_ATR * atr
        and breakout["volume_z"] >= MIN_VOLUME_Z
        and breakout["flow_z"] >= MIN_FLOW_Z
        and breakout["body_frac"] >= MIN_BODY_FRACTION
        and breakout["close_loc"] >= MIN_CLOSE_LOC
    )

    short_breakout = (
        breakout["close"] < breakout["prior_lo"]
        and (breakout["prior_lo"] - breakout["close"]) >= MIN_BREAKOUT_ATR * atr
        and breakout["volume_z"] >= MIN_VOLUME_Z
        and breakout["flow_z"] <= -MIN_FLOW_Z
        and breakout["body_frac"] >= MIN_BODY_FRACTION
        and breakout["close_loc"] <= 1.0 - MIN_CLOSE_LOC
    )

    # Acceptance means the next completed candle does not close back inside
    # the old range. It may test the level, but its close must remain beyond it.
    long_accept = (
        confirm["open"] >= breakout["prior_hi"]
        and confirm["close"] >= breakout["prior_hi"]
        and confirm["close_loc"] >= MIN_ACCEPT_CLOSE_LOC
        and confirm["volume"] >= MIN_ACCEPT_VOLUME_RATIO * breakout["volume"]
    )

    short_accept = (
        confirm["open"] <= breakout["prior_lo"]
        and confirm["close"] <= breakout["prior_lo"]
        and confirm["close_loc"] <= 1.0 - MIN_ACCEPT_CLOSE_LOC
        and confirm["volume"] >= MIN_ACCEPT_VOLUME_RATIO * breakout["volume"]
    )

    long_regime = confirm["trend_atr_4h"] >= MIN_4H_TREND_ATR
    short_regime = confirm["trend_atr_4h"] <= -MIN_4H_TREND_ATR

    if long_breakout and long_accept and long_regime:
        return 1
    if short_breakout and short_accept and short_regime:
        return -1
    return 0


def compute_exit(x, entry_idx, side, entry, sl, tp):
    """Return (exit_ts, pnl, win), or None if no exit before data ends."""
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

        # Conservative rule required by the research protocol.
        win = bool(hit_tp and not hit_sl)
        exit_px = tp if win else sl

        gross = (
            NOTIONAL * (exit_px - entry) / entry
            if side == 1
            else NOTIONAL * (entry - exit_px) / entry
        )
        fees = NOTIONAL * FEE_RATE * 2.0
        pnl = gross - fees
        return ts, float(pnl), win

    return None


def simulate(raw, stop_mult):
    streams = {symbol: build_features(df) for symbol, df in raw.items()}

    events = []
    raw_candidates = 0
    for symbol, x in streams.items():
        for i in range(len(x) - 1):
            side = candidate(x, i)
            if side:
                raw_candidates += 1
                events.append((x.index[i], symbol, i, side))

    events.sort(key=lambda item: (item[0], item[1]))

    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_dd = 0.0
    open_positions = {}
    trades = []
    last_exit_by_symbol = {}

    for signal_ts, symbol, i, side in events:
        x = streams[symbol]
        entry_idx = i + 1
        if entry_idx >= len(x):
            continue

        entry_ts = x.index[entry_idx]

        # Per-symbol overlap lock AND explicit no-same-exit-candle rule.
        previous_exit = last_exit_by_symbol.get(symbol)
        if previous_exit is not None and entry_ts <= previous_exit:
            continue

        # Remove positions whose exit happened strictly before this proposed
        # entry. If exit_ts == entry_ts, keep it active so the same exit candle
        # cannot also be an entry candle.
        stale_symbols = [
            pos_symbol
            for pos_symbol, pos in open_positions.items()
            if pos["exit_ts"] < entry_ts
        ]
        for pos_symbol in stale_symbols:
            del open_positions[pos_symbol]

        active_count = len(open_positions)

        # Stated $100 margin budget allows at most 10 simultaneous positions.
        if active_count >= MAX_OPEN_POSITIONS:
            continue

        # Per-symbol overlap lock.
        if symbol in open_positions:
            continue

        # Reserve $100 margin for every already-open position. This prevents
        # the backtest from opening more positions than the $1,000 capital
        # allocation can support.
        required_margin = (active_count + 1) * MARGIN
        if equity < required_margin:
            continue

        atr = float(x.iloc[i]["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        entry_raw = float(x.iloc[entry_idx]["open"])
        entry = (
            entry_raw * (1.0 + SLIPPAGE)
            if side == 1
            else entry_raw * (1.0 - SLIPPAGE)
        )

        stop_dist = stop_mult * atr
        if side == 1:
            sl = entry - stop_dist
            tp = entry + RR * stop_dist
        else:
            sl = entry + stop_dist
            tp = entry - RR * stop_dist

        result = compute_exit(
            x, entry_idx, side, entry, sl, tp
        )
        if result is None:
            # Open positions are not marked-to-market or counted as trades.
            continue

        exit_ts, pnl, win = result

        position = {
            "symbol": symbol,
            "side": side,
            "signal_ts": signal_ts,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
        }

        open_positions[symbol] = position
        equity += pnl
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, peak_equity - equity)

        trades.append({
            "signal_ts": signal_ts,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "symbol": symbol,
            "side": "LONG" if side == 1 else "SHORT",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "win": int(win),
            "pnl": pnl,
        })

        last_exit_by_symbol[symbol] = exit_ts

        # Keep the position in open_positions until the event stream reaches a
        # candle strictly after exit_ts. This is what makes simultaneous
        # positions across different symbols real in the simulation.

    return pd.DataFrame(trades), raw_candidates


def summarize(trades):
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "pnl": 0.0,
            "dd": 0.0,
            "dd_pct": 0.0,
            "streak": 0,
            "expectancy": 0.0,
            "final": INITIAL_CAPITAL,
        }

    pnl = trades["pnl"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())

    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl <= 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    equity = INITIAL_CAPITAL + pnl.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    max_dd = float(dd.max())
    max_dd_pct = float(
        (dd / peak.replace(0, np.nan)).max() * 100.0
    )

    streak = 0
    current = 0
    for value in pnl:
        if value <= 0:
            current += 1
            streak = max(streak, current)
        else:
            current = 0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "wr": 100.0 * wins / len(trades),
        "pf": pf,
        "pnl": float(pnl.sum()),
        "dd": max_dd,
        "dd_pct": max_dd_pct,
        "streak": streak,
        "expectancy": float(pnl.mean()),
        "final": float(equity.iloc[-1]),
    }


def print_report(stop_mult, trades, raw_candidates):
    s = summarize(trades)

    print("\n" + "=" * 76)
    print(f"STAGE-0 STOP = {stop_mult:.2f} ATR | RR = 1:2")
    print("=" * 76)
    print(f"Raw candidate signals   : {raw_candidates}")
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
        print(
            side.to_string(
                index=False,
                formatters={
                    "pnl": lambda v: f"{v:.2f}",
                    "wr": lambda v: f"{v:.2f}",
                },
            )
        )

        symbol = (
            trades.groupby("symbol")
            .agg(
                trades=("pnl", "size"),
                wins=("win", "sum"),
                pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        symbol["wr"] = 100.0 * symbol["wins"] / symbol["trades"]

        print("\nBY SYMBOL")
        print(
            symbol.sort_values("pnl", ascending=False).to_string(
                index=False,
                formatters={
                    "pnl": lambda v: f"{v:.2f}",
                    "wr": lambda v: f"{v:.2f}",
                },
            )
        )


def main():
    print("HUNTER-V20-STAGE0 — VOLUME / ORDER-FLOW IMBALANCE + BREAKOUT ACCEPTANCE")
    print(
        f"Lookback={LOOKBACK_DAYS}d + warmup={WARMUP_DAYS}d | "
        f"1H direct XT futures"
    )
    print(
        f"RR=1:2 | margin=${MARGIN:.0f} | notional=${NOTIONAL:.0f} | "
        f"fee={FEE_RATE} | slippage={SLIPPAGE}"
    )
    print(
        "OHLCV signed-flow proxy + prior-range breakout + next-bar acceptance "
        "+ completed 4H regime"
    )
    print(
        f"Per-symbol overlap lock | max simultaneous positions={MAX_OPEN_POSITIONS}"
    )
    print("No optimization. Stop sensitivity is reported, not selected.")

    refresh = os.getenv("XT_REFRESH", "0") == "1"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = {symbol: fetch_xt(symbol, refresh=refresh) for symbol in SYMBOLS}

    common_start = max(df["timestamp"].min() for df in raw.values())
    common_end = min(df["timestamp"].max() for df in raw.values())

    print(f"COMMON 1H RANGE: {common_start} -> {common_end}")

    raw = {
        symbol: df[
            (df["timestamp"] >= common_start)
            & (df["timestamp"] <= common_end)
        ].copy()
        for symbol, df in raw.items()
    }

    all_results = []

    for stop_mult in STOP_ATR_VALUES:
        trades, raw_candidates = simulate(raw, stop_mult)

        print_report(
            stop_mult,
            trades,
            raw_candidates,
        )

        output = REPORT_DIR / f"stage0_trades_stop_{stop_mult:.2f}.csv"
        trades.to_csv(output, index=False)

        summary = summarize(trades)
        summary["stop_atr"] = stop_mult
        all_results.append(summary)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        REPORT_DIR / "stage0_summary.csv",
        index=False,
    )

    # Conservative discovery gate. It deliberately does not select the
    # strongest stop. The entire pre-registered sensitivity set must support
    # the existence of an edge before expensive walk-forward is allowed.
    enough_activity = all(r["trades"] >= 150 for r in all_results)
    positive_edge = all(
        r["pf"] > 1.0 and r["expectancy"] > 0
        for r in all_results
    )
    reasonable_wr = max(r["wr"] for r in all_results) >= 40.0
    drawdown_ok = all(r["dd_pct"] <= 50.0 for r in all_results)

    print("\n" + "=" * 76)
    print("STAGE-0 DECISION")
    print("=" * 76)
    print("No stop is selected by this script.")
    print(
        f"Activity >=150 trades across all stops : "
        f"{'PASS' if enough_activity else 'FAIL'}"
    )
    print(
        f"Positive PF + expectancy across all    : "
        f"{'PASS' if positive_edge else 'FAIL'}"
    )
    print(
        f"At least one stop WR >=40%             : "
        f"{'PASS' if reasonable_wr else 'FAIL'}"
    )
    print(
        f"Max DD <=50% across all stops          : "
        f"{'PASS' if drawdown_ok else 'FAIL'}"
    )

    if enough_activity and positive_edge and reasonable_wr and drawdown_ok:
        print("DISCOVERY_STATUS: PASS — eligible for walk-forward")
    else:
        print("DISCOVERY_STATUS: REJECT — do NOT tune or optimize this family")
    print("=" * 76)


if __name__ == "__main__":
    main()
