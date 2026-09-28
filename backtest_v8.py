#!/usr/bin/env python3
"""
HUNTER-V21-STAGE0
FAILED AUCTION / RETURN-TO-VALUE MEAN REVERSION

Research hypothesis
-------------------
When price is operating inside a recent value/range, a one-sided auction that
extends materially outside that range can fail. If the excursion candle closes
back inside the prior value area and the next completed candle confirms the
re-entry, price may mean-revert toward the range midpoint.

This is intentionally different from:
- V17 momentum/trend persistence,
- V18 volatility exhaustion/capitulation,
- V19 structure continuation,
- V20 breakout acceptance.

Entry:
    next 1H OPEN after the completed confirmation candle.

Risk:
    fixed RR = 1:2
    stop sensitivity = 1.00 / 1.25 / 1.50 ATR
    no stop is selected by this script.

Integrity protocol
------------------
- XT USDT-M futures, direct 1H OHLCV.
- Latest incomplete candle removed.
- All indicators are causal.
- Prior range excludes the current candle.
- Signal candle -> confirmation candle -> next-open entry.
- One open position per symbol.
- Different symbols may be open simultaneously.
- Maximum 10 simultaneous $100-margin positions.
- A symbol cannot re-enter on its exit candle.
- Same-candle SL+TP = LOSS.
- No trailing, BE, timeout, pyramiding, or future-data filters.
- Fixed universe and fixed hypothesis parameters.
- No parameter optimization.

Important data limitation
-------------------------
This is not true exchange-level order-flow/auction data. "Failed auction"
is inferred from OHLCV price behavior only.
"""

import os
import time
import hashlib
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
VALUE_N = 24
EXTENSION_ATR = 1.00
MAX_TREND_ATR_4H = 0.75

# The failed-auction candle must show meaningful rejection.
MIN_WICK_FRACTION = 0.25
MIN_CLOSE_BACK_IN_RANGE = True

# Confirmation must move in the reversion direction and close beyond the
# failed-auction candle's midpoint. It must still be a completed candle.
CONFIRM_CLOSE_FRACTION = 0.50

# Sensitivity only. Never selected by the script.
STOP_ATR_VALUES = (1.00, 1.25, 1.50)

CACHE_DIR = Path("data/xt_v21_stage0")
REPORT_DIR = Path("reports/xt_v21_stage0")


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
    x = x.sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last"
    )

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
    bad = gaps > 1.0 + 1e-9
    if bad.any():
        raise RuntimeError(
            f"{symbol}: {int(bad.sum())} futures 1H gaps; "
            "no OHLCV fabrication allowed"
        )

    minimum_rows = int((LOOKBACK_DAYS + WARMUP_DAYS) * 24 * 0.95)
    if len(x) < minimum_rows:
        raise RuntimeError(f"{symbol}: only {len(x)} 1H rows")

    print(
        f"[VALIDATE] {symbol} rows={len(x)} "
        f"first={x['timestamp'].iloc[0]} "
        f"last={x['timestamp'].iloc[-1]}"
    )
    return x.reset_index(drop=True)


def fetch_xt(symbol, refresh=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}.csv"

    now = int(time.time() * 1000)
    start_raw = now - int(
        (LOOKBACK_DAYS + WARMUP_DAYS) * 86400 * 1000
    )
    start_ms = (
        (start_raw + INTERVAL_MS - 1) // INTERVAL_MS
    ) * INTERVAL_MS
    end_ms = (now // INTERVAL_MS) * INTERVAL_MS - 1

    if cache.exists() and not refresh:
        return validate(
            pd.read_csv(cache), symbol, start_ms, end_ms
        )

    session = requests.Session()
    cursor = start_ms
    all_rows = []
    page = 0

    while cursor <= end_ms:
        page += 1
        window_end = min(
            end_ms,
            cursor + LIMIT * INTERVAL_MS - 1
        )

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
                response = session.get(
                    FUTURES_URL,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                payload = payload_rows(response.json())

                if payload is None:
                    raise RuntimeError(
                        "XT response contains no kline list"
                    )
                break
            except Exception as exc:
                error = exc
                time.sleep(1.0 + attempt)

        if payload is None:
            raise RuntimeError(
                f"{symbol}: page {page} failed: {error}"
            )

        parsed = [
            p for p in (parse_row(r) for r in payload)
            if p is not None
        ]

        inside = [
            p for p in parsed
            if cursor <= p[0] <= window_end
        ]

        if not inside:
            if window_end >= end_ms:
                break
            raise RuntimeError(
                f"{symbol}: page {page} returned no rows"
            )

        all_rows.extend(inside)

        max_ts = max(p[0] for p in inside)
        next_cursor = max_ts + INTERVAL_MS

        if next_cursor <= cursor:
            raise RuntimeError(
                f"{symbol}: pagination stalled"
            )

        cursor = next_cursor

        if page % 4 == 0:
            print(
                f"[FETCH] {symbol} page={page} "
                f"rows={len(all_rows)}"
            )

        if max_ts >= end_ms:
            break

        time.sleep(0.05)

    x = pd.DataFrame(
        all_rows,
        columns=[
            "ts", "open", "high", "low", "close", "volume"
        ],
    )
    x["timestamp"] = pd.to_datetime(
        x.pop("ts"),
        unit="ms",
        utc=True,
    )

    # Never include the currently forming 1H candle.
    cutoff = pd.Timestamp.now(
        tz="UTC"
    ).floor("1h")
    x = x[x["timestamp"] < cutoff]

    x = validate(
        x,
        symbol,
        start_ms,
        end_ms,
    )
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
    x = df.set_index(
        "timestamp"
    ).sort_index().copy()

    x["tr"] = true_range(x)
    x["atr"] = x["tr"].rolling(
        ATR_PERIOD,
        min_periods=ATR_PERIOD,
    ).mean()

    x["range"] = (
        x["high"] - x["low"]
    ).clip(lower=0.0)

    safe_range = x["range"].replace(
        0,
        np.nan,
    )

    x["body"] = (
        x["close"] - x["open"]
    ).abs()

    x["upper_wick"] = (
        x["high"]
        - x[["open", "close"]].max(axis=1)
    )

    x["lower_wick"] = (
        x[["open", "close"]].min(axis=1)
        - x["low"]
    )

    x["close_loc"] = (
        (x["close"] - x["low"])
        / safe_range
    )

    # Strictly prior value/range. Current candle is excluded.
    x["value_hi"] = x["high"].shift(1).rolling(
        VALUE_N,
        min_periods=VALUE_N,
    ).max()

    x["value_lo"] = x["low"].shift(1).rolling(
        VALUE_N,
        min_periods=VALUE_N,
    ).min()

    x["value_mid"] = (
        x["value_hi"] + x["value_lo"]
    ) / 2.0

    # Completed 4H regime. Shift by one completed 4H bar before mapping back
    # to 1H timestamps, preventing use of a still-forming 4H candle.
    h4 = x[
        ["open", "high", "low", "close"]
    ].resample(
        "4h",
        label="right",
        closed="right",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    h4["ema20"] = h4["close"].ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    h4["ema50"] = h4["close"].ewm(
        span=50,
        adjust=False,
        min_periods=50,
    ).mean()

    h4["tr"] = true_range(h4)
    h4["atr"] = h4["tr"].rolling(
        ATR_PERIOD,
        min_periods=ATR_PERIOD,
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

    x = x.join(
        h4.reindex(
            x.index,
            method="ffill",
        )
    )

    x.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return x


def candidate(x, i):
    """
    i = completed confirmation candle.
    i-1 = failed-auction candle.
    i+1 open = entry.

    No candle after i is read by this function.
    """
    if i < 120 or i + 1 >= len(x):
        return 0

    failed = x.iloc[i - 1]
    confirm = x.iloc[i]

    required = [
        failed["atr"],
        failed["value_hi"],
        failed["value_lo"],
        failed["value_mid"],
        failed["high"],
        failed["low"],
        failed["open"],
        failed["close"],
        failed["upper_wick"],
        failed["lower_wick"],
        failed["range"],
        confirm["open"],
        confirm["high"],
        confirm["low"],
        confirm["close"],
        confirm["close_loc"],
        confirm["value_hi"],
        confirm["value_lo"],
        confirm["value_mid"],
        confirm["trend_atr_4h"],
    ]

    if any(pd.isna(v) for v in required):
        return 0

    atr = float(failed["atr"])
    if atr <= 0:
        return 0

    value_hi = float(failed["value_hi"])
    value_lo = float(failed["value_lo"])
    value_mid = float(failed["value_mid"])

    # ---------------------------------------------------------------
    # LONG mean reversion:
    # 1) price auctions materially below prior value,
    # 2) the candle rejects the lower excursion,
    # 3) it closes back inside the prior value,
    # 4) next candle confirms upward re-entry.
    # ---------------------------------------------------------------
    long_extension = (
        float(failed["low"])
        <= value_lo - EXTENSION_ATR * atr
    )

    long_rejection = (
        float(failed["lower_wick"])
        / max(float(failed["range"]), 1e-12)
        >= MIN_WICK_FRACTION
        and float(failed["close"])
        > value_lo
        and float(failed["close_loc"]) >= 0.50
    )

    long_confirmation = (
        float(confirm["close"]) > float(confirm["open"])
        and float(confirm["close"]) > float(failed["close"])
        and float(confirm["close"]) > (
            float(failed["low"])
            + CONFIRM_CLOSE_FRACTION
            * float(failed["range"])
        )
        and float(confirm["close"]) > value_lo
    )

    # ---------------------------------------------------------------
    # SHORT mean reversion:
    # symmetric failed auction above prior value.
    # ---------------------------------------------------------------
    short_extension = (
        float(failed["high"])
        >= value_hi + EXTENSION_ATR * atr
    )

    short_rejection = (
        float(failed["upper_wick"])
        / max(float(failed["range"]), 1e-12)
        >= MIN_WICK_FRACTION
        and float(failed["close"])
        < value_hi
        and float(failed["close_loc"]) <= 0.50
    )

    short_confirmation = (
        float(confirm["close"]) < float(confirm["open"])
        and float(confirm["close"]) < float(failed["close"])
        and float(confirm["close"]) < (
            float(failed["high"])
            - CONFIRM_CLOSE_FRACTION
            * float(failed["range"])
        )
        and float(confirm["close"]) < value_hi
    )

    # Mean-reversion hypothesis is specifically for non-trending / bounded
    # conditions. We reject strongly directional completed 4H regimes.
    neutral_4h = (
        abs(float(confirm["trend_atr_4h"]))
        <= MAX_TREND_ATR_4H
    )

    if (
        neutral_4h
        and long_extension
        and long_rejection
        and long_confirmation
    ):
        return 1

    if (
        neutral_4h
        and short_extension
        and short_rejection
        and short_confirmation
    ):
        return -1

    return 0


def compute_exit(
    x,
    entry_idx,
    side,
    entry,
    sl,
    tp,
):
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

        # Conservative same-candle ambiguity rule.
        win = bool(hit_tp and not hit_sl)
        exit_px = tp if win else sl

        gross = (
            NOTIONAL
            * (exit_px - entry)
            / entry
            if side == 1
            else
            NOTIONAL
            * (entry - exit_px)
            / entry
        )

        fees = NOTIONAL * FEE_RATE * 2.0
        pnl = gross - fees

        return ts, float(pnl), win

    return None


def simulate(raw, stop_mult):
    streams = {
        symbol: build_features(df)
        for symbol, df in raw.items()
    }

    events = []
    raw_candidates = 0

    for symbol, x in streams.items():
        for i in range(len(x) - 1):
            side = candidate(x, i)
            if side:
                raw_candidates += 1
                events.append(
                    (x.index[i], symbol, i, side)
                )

    events.sort(
        key=lambda item: (item[0], item[1])
    )

    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_dd = 0.0

    # Position remains active through its exit timestamp so that another
    # event on the same candle cannot open a replacement position.
    open_positions = {}
    last_exit_by_symbol = {}

    trades = []

    for signal_ts, symbol, i, side in events:
        x = streams[symbol]
        entry_idx = i + 1

        if entry_idx >= len(x):
            continue

        entry_ts = x.index[entry_idx]

        # Remove only positions whose exit happened strictly before entry.
        stale = [
            pos_symbol
            for pos_symbol, pos in open_positions.items()
            if pos["exit_ts"] < entry_ts
        ]

        for pos_symbol in stale:
            del open_positions[pos_symbol]

        # Same-symbol overlap lock.
        if symbol in open_positions:
            continue

        # Explicit no-same-exit-candle re-entry rule.
        previous_exit = last_exit_by_symbol.get(symbol)
        if previous_exit is not None and entry_ts <= previous_exit:
            continue

        # Portfolio margin capacity.
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        required_margin = (
            len(open_positions) + 1
        ) * MARGIN

        if equity < required_margin:
            continue

        atr = float(x.iloc[i]["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        entry_raw = float(
            x.iloc[entry_idx]["open"]
        )

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
            x,
            entry_idx,
            side,
            entry,
            sl,
            tp,
        )

        if result is None:
            continue

        exit_ts, pnl, win = result

        open_positions[symbol] = {
            "exit_ts": exit_ts,
            "entry_ts": entry_ts,
        }

        equity += pnl
        peak_equity = max(
            peak_equity,
            equity,
        )
        max_dd = max(
            max_dd,
            peak_equity - equity,
        )

        trades.append({
            "signal_ts": signal_ts,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "symbol": symbol,
            "side": (
                "LONG" if side == 1 else "SHORT"
            ),
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "win": int(win),
            "pnl": pnl,
        })

        last_exit_by_symbol[symbol] = exit_ts

    return (
        pd.DataFrame(trades),
        raw_candidates,
    )


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

    wins = int(
        (pnl > 0).sum()
    )
    losses = int(
        (pnl <= 0).sum()
    )

    gross_win = float(
        pnl[pnl > 0].sum()
    )
    gross_loss = float(
        -pnl[pnl <= 0].sum()
    )

    pf = (
        gross_win / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    equity = (
        INITIAL_CAPITAL
        + pnl.cumsum()
    )

    peak = equity.cummax()
    dd = peak - equity

    max_dd = float(dd.max())
    max_dd_pct = float(
        (
            dd
            / peak.replace(
                0,
                np.nan,
            )
        ).max()
        * 100.0
    )

    streak = 0
    current = 0

    for value in pnl:
        if value <= 0:
            current += 1
            streak = max(
                streak,
                current,
            )
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
        "expectancy": float(
            pnl.mean()
        ),
        "final": float(
            equity.iloc[-1]
        ),
    }


def print_report(
    stop_mult,
    trades,
    raw_candidates,
):
    s = summarize(trades)

    print("\n" + "=" * 76)
    print(
        f"STAGE-0 STOP = {stop_mult:.2f} ATR | "
        "RR = 1:2"
    )
    print("=" * 76)

    print(
        f"Raw candidate signals   : "
        f"{raw_candidates}"
    )
    print(
        f"Closed trades           : "
        f"{s['trades']}"
    )
    print(
        f"Wins                    : "
        f"{s['wins']}"
    )
    print(
        f"Losses                  : "
        f"{s['losses']}"
    )
    print(
        f"Win Rate                : "
        f"{s['wr']:.2f}%"
    )
    print(
        f"Profit Factor           : "
        f"{s['pf']:.4f}"
    )
    print(
        f"Net PnL                 : "
        f"${s['pnl']:,.2f}"
    )
    print(
        f"Max Drawdown            : "
        f"${s['dd']:,.2f}"
    )
    print(
        f"Max Drawdown %          : "
        f"{s['dd_pct']:.2f}%"
    )
    print(
        f"Max Loss Streak         : "
        f"{s['streak']}"
    )
    print(
        f"Expectancy / Trade      : "
        f"${s['expectancy']:.2f}"
    )
    print(
        f"Final Equity            : "
        f"${s['final']:,.2f}"
    )

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

        side["wr"] = (
            100.0
            * side["wins"]
            / side["trades"]
        )

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

        symbol["wr"] = (
            100.0
            * symbol["wins"]
            / symbol["trades"]
        )

        print("\nBY SYMBOL")
        print(
            symbol.sort_values(
                "pnl",
                ascending=False,
            ).to_string(
                index=False,
                formatters={
                    "pnl": lambda v: f"{v:.2f}",
                    "wr": lambda v: f"{v:.2f}",
                },
            )
        )


def main():
    print(
        "HUNTER-V21-STAGE0 — "
        "FAILED AUCTION / RETURN-TO-VALUE MEAN REVERSION"
    )

    print(
        f"Lookback={LOOKBACK_DAYS}d + "
        f"warmup={WARMUP_DAYS}d | "
        "1H direct XT futures"
    )

    print(
        f"RR=1:2 | margin=${MARGIN:.0f} | "
        f"notional=${NOTIONAL:.0f} | "
        f"fee={FEE_RATE} | "
        f"slippage={SLIPPAGE}"
    )

    print(
        "Hypothesis: range extension -> failed "
        "acceptance -> return inside value -> "
        "confirmation"
    )

    print(
        f"Per-symbol overlap lock | "
        f"max simultaneous positions="
        f"{MAX_OPEN_POSITIONS}"
    )

    print(
        "No parameter optimization. "
        "Stop sensitivity is reported, not selected."
    )

    refresh = (
        os.getenv(
            "XT_REFRESH",
            "0",
        ) == "1"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = {
        symbol: fetch_xt(
            symbol,
            refresh=refresh,
        )
        for symbol in SYMBOLS
    }

    common_start = max(
        df["timestamp"].min()
        for df in raw.values()
    )

    common_end = min(
        df["timestamp"].max()
        for df in raw.values()
    )

    print(
        f"COMMON 1H RANGE: "
        f"{common_start} -> {common_end}"
    )

    raw = {
        symbol: df[
            (df["timestamp"] >= common_start)
            & (df["timestamp"] <= common_end)
        ].copy()
        for symbol, df in raw.items()
    }

    results = []

    for stop_mult in STOP_ATR_VALUES:
        trades, raw_candidates = simulate(
            raw,
            stop_mult,
        )

        print_report(
            stop_mult,
            trades,
            raw_candidates,
        )

        trades.to_csv(
            REPORT_DIR
            / f"stage0_trades_stop_{stop_mult:.2f}.csv",
            index=False,
        )

        summary = summarize(trades)
        summary["stop_atr"] = stop_mult
        results.append(summary)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(
        REPORT_DIR / "stage0_summary.csv",
        index=False,
    )

    # Conservative discovery gate. It does not choose a stop.
    enough_activity = all(
        r["trades"] >= 150
        for r in results
    )

    positive_edge = all(
        r["pf"] > 1.0
        and r["expectancy"] > 0
        for r in results
    )

    reasonable_wr = max(
        r["wr"] for r in results
    ) >= 40.0

    drawdown_ok = all(
        r["dd_pct"] <= 50.0
        for r in results
    )

    print("\n" + "=" * 76)
    print("STAGE-0 DECISION")
    print("=" * 76)
    print(
        "No stop is selected by this script."
    )
    print(
        "Activity >=150 trades across all stops : "
        f"{'PASS' if enough_activity else 'FAIL'}"
    )
    print(
        "Positive PF + expectancy across all    : "
        f"{'PASS' if positive_edge else 'FAIL'}"
    )
    print(
        "At least one stop WR >=40%             : "
        f"{'PASS' if reasonable_wr else 'FAIL'}"
    )
    print(
        "Max DD <=50% across all stops          : "
        f"{'PASS' if drawdown_ok else 'FAIL'}"
    )

    if (
        enough_activity
        and positive_edge
        and reasonable_wr
        and drawdown_ok
    ):
        print(
            "DISCOVERY_STATUS: PASS — "
            "eligible for walk-forward"
        )
    else:
        print(
            "DISCOVERY_STATUS: REJECT — "
            "do NOT tune or optimize this family"
        )

    print("=" * 76)


if __name__ == "__main__":
    main()
