# ============================================================
# HUNTER-X V9
# LBank USDT Futures / Swap
# 1H + 4H
# TP = +2%
# SL = -1%
# RR = 1:2
# CLEAN BACKTEST - NO LOOKAHEAD
# ============================================================

import os
import sys
import time
import math
import csv
import subprocess
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ccxt", "pandas", "numpy"]
    )
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT",
    "DOT": "DOT/USDT",
}

TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

TP_PCT = 0.0200       # +2%
SL_PCT = 0.0100       # -1%

RR = TP_PCT / SL_PCT   # 2.0

# Maximum holding time:
# 24 x 1H = 24 hours
MAX_HOLD_BARS = 24

# ------------------------------------------------------------
# COST MODEL
# ------------------------------------------------------------

# Assumption:
# 0.06% taker fee each side
# 0.02% slippage each side
FEE_RATE = 0.0006
SLIPPAGE_RATE = 0.0002

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "365"))

# Warm-up period for indicators
WARMUP_DAYS = 120

# Number of candles fetched per request
FETCH_LIMIT = 1000

OUTPUT_CSV = "hunter_x_v9_lbank_futures_trades.csv"


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 30000,
})


# ============================================================
# UTILITIES
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def ms(dt):
    return int(dt.timestamp() * 1000)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ============================================================
# MARKET RESOLUTION
# ============================================================

def find_futures_market(exchange, base_symbol):
    """
    Find the LBank USDT swap/futures market.

    We DO NOT blindly assume BTC/USDT means futures.
    First load markets and search for swap/futures.
    """

    markets = exchange.load_markets()

    candidates = []

    for symbol, market in markets.items():

        if market.get("base") != base_symbol:
            continue

        if market.get("quote") != "USDT":
            continue

        # Prefer perpetual swap
        if market.get("swap"):
            candidates.append((symbol, market))

    if not candidates:

        # Secondary search for futures
        for symbol, market in markets.items():

            if market.get("base") != base_symbol:
                continue

            if market.get("quote") != "USDT":
                continue

            if market.get("future"):
                candidates.append((symbol, market))

    if not candidates:
        return None, None

    # Prefer linear USDT contract
    candidates.sort(
        key=lambda x: (
            0 if x[1].get("linear") else 1,
            x[0]
        )
    )

    return candidates[0]


# ============================================================
# DOWNLOAD LBank OHLCV
# ============================================================

def fetch_lbank_ohlcv(symbol, timeframe, start_dt, end_dt):

    print(f"  Downloading {symbol} {timeframe} ...")

    all_rows = []

    current_since = ms(start_dt)
    end_ms = ms(end_dt)

    while current_since < end_ms:

        try:

            rows = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=FETCH_LIMIT
            )

        except Exception as e:

            print(
                f"    fetch error: {e}"
            )

            time.sleep(3)
            continue

        if not rows:
            break

        # Keep only requested period
        for row in rows:

            if len(row) < 6:
                continue

            ts = int(row[0])

            if ts >= end_ms:
                continue

            if ts < ms(start_dt):
                continue

            all_rows.append(row)

        last_ts = int(rows[-1][0])

        next_since = last_ts + 1

        if next_since <= current_since:
            break

        current_since = next_since

        if len(rows) < FETCH_LIMIT:
            break

        time.sleep(
            max(
                0.1,
                exchange.rateLimit / 1000
            )
        )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.set_index(
        "timestamp"
    )

    # Remove future/current candle
    now = pd.Timestamp.now(tz="UTC")

    if timeframe == "1h":
        current_bucket = now.floor("1h")
    else:
        current_bucket = now.floor("4h")

    df = df[df.index < current_bucket]

    return df


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    result = 100 - (
        100 / (1 + rs)
    )

    return result


def atr(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def adx(df, period=14):

    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0
    )

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr_value = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_di = (
        100 *
        pd.Series(
            plus_dm,
            index=df.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    minus_di = (
        100 *
        pd.Series(
            minus_dm,
            index=df.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# PREPARE 1H
# ============================================================

def prepare_1h(df):

    x = df.copy()

    x["ema20"] = ema(
        x["close"],
        20
    )

    x["ema50"] = ema(
        x["close"],
        50
    )

    x["ema200"] = ema(
        x["close"],
        200
    )

    x["rsi"] = rsi(
        x["close"],
        14
    )

    x["atr"] = atr(
        x,
        14
    )

    x["adx"] = adx(
        x,
        14
    )

    x["volume_ma20"] = (
        x["volume"]
        .rolling(20)
        .mean()
    )

    x["body"] = (
        x["close"] -
        x["open"]
    ).abs()

    x["range"] = (
        x["high"] -
        x["low"]
    )

    x["body_ratio"] = (
        x["body"] /
        x["range"].replace(
            0,
            np.nan
        )
    )

    x["atr_pct"] = (
        x["atr"] /
        x["close"]
    )

    # Previous candle information
    x["prev_high"] = x["high"].shift(1)
    x["prev_low"] = x["low"].shift(1)
    x["prev_close"] = x["close"].shift(1)

    return x


# ============================================================
# PREPARE 4H
# ============================================================

def prepare_4h(df):

    x = df.copy()

    x["ema20"] = ema(
        x["close"],
        20
    )

    x["ema50"] = ema(
        x["close"],
        50
    )

    x["ema200"] = ema(
        x["close"],
        200
    )

    x["rsi"] = rsi(
        x["close"],
        14
    )

    x["adx"] = adx(
        x,
        14
    )

    return x


# ============================================================
# ATTACH PREVIOUS CLOSED 4H
# ============================================================

def attach_previous_4h(df1, df4):

    h1 = df1.copy()
    h4 = df4.copy()

    # Important:
    #
    # A 1H candle must NEVER use the 4H candle that
    # is still forming.
    #
    # Therefore we shift the 4H data by one completed
    # 4H candle before merging.

    h4_features = h4[
        [
            "close",
            "ema20",
            "ema50",
            "ema200",
            "rsi",
            "adx"
        ]
    ].copy()

    h4_features.columns = [
        "htf_close",
        "htf_ema20",
        "htf_ema50",
        "htf_ema200",
        "htf_rsi",
        "htf_adx"
    ]

    # Shift means only PREVIOUS completed 4H candle
    h4_features = h4_features.shift(1)

    h1["htf_bucket"] = h1.index.floor("4h")

    h4_features["htf_bucket"] = h4_features.index

    h4_features = h4_features.reset_index(
        drop=True
    )

    # Build mapping manually to avoid accidental future merge
    mapping = {}

    for _, row in h4_features.iterrows():

        bucket = row["htf_bucket"]

        mapping[bucket] = row

    values = []

    for idx in h1.index:

        bucket = idx.floor("4h")

        row = mapping.get(
            bucket,
            None
        )

        if row is None:

            values.append(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan
                ]
            )

        else:

            values.append(
                [
                    row["htf_close"],
                    row["htf_ema20"],
                    row["htf_ema50"],
                    row["htf_ema200"],
                    row["htf_rsi"],
                    row["htf_adx"]
                ]
            )

    vals = pd.DataFrame(
        values,
        index=h1.index,
        columns=[
            "htf_close",
            "htf_ema20",
            "htf_ema50",
            "htf_ema200",
            "htf_rsi",
            "htf_adx"
        ]
    )

    h1 = pd.concat(
        [h1, vals],
        axis=1
    )

    return h1


# ============================================================
# 4H REGIME
# ============================================================

def get_regime(row):

    required = [
        "htf_close",
        "htf_ema20",
        "htf_ema50",
        "htf_ema200",
        "htf_rsi",
        "htf_adx"
    ]

    if any(
        pd.isna(row[x])
        for x in required
    ):
        return None

    # Strong bullish regime
    if (
        row["htf_close"] > row["htf_ema20"]
        and
        row["htf_ema20"] > row["htf_ema50"]
        and
        row["htf_ema50"] > row["htf_ema200"]
        and
        row["htf_rsi"] >= 50
        and
        row["htf_adx"] >= 15
    ):
        return "LONG"

    # Strong bearish regime
    if (
        row["htf_close"] < row["htf_ema20"]
        and
        row["htf_ema20"] < row["htf_ema50"]
        and
        row["htf_ema50"] < row["htf_ema200"]
        and
        row["htf_rsi"] <= 50
        and
        row["htf_adx"] >= 15
    ):
        return "SHORT"

    return None


# ============================================================
# 1H MARKET QUALITY
# ============================================================

def market_quality(row):

    required = [
        "ema20",
        "ema50",
        "rsi",
        "adx",
        "volume_ma20",
        "body_ratio",
        "atr_pct"
    ]

    if any(
        pd.isna(row[x])
        for x in required
    ):
        return False

    if row["adx"] < 15:
        return False

    if row["volume"] < row["volume_ma20"] * 0.70:
        return False

    if row["body_ratio"] < 0.30:
        return False

    # Avoid completely dead or extreme volatility
    if row["atr_pct"] < 0.002:
        return False

    if row["atr_pct"] > 0.025:
        return False

    return True


# ============================================================
# PULLBACK SETUP
# ============================================================

def long_pullback(row, prev):

    if not market_quality(row):
        return False

    if row["ema20"] <= row["ema50"]:
        return False

    if row["rsi"] < 48:
        return False

    # Candle tests/reclaims EMA20
    touched_ema20 = (
        row["low"] <=
        row["ema20"] * 1.004
    )

    closed_above = (
        row["close"] >
        row["ema20"]
    )

    bullish = (
        row["close"] >
        row["open"]
    )

    # Previous candle should not already be an enormous move
    previous_extension = (
        abs(prev["close"] - prev["open"])
        /
        max(
            prev["close"],
            1e-12
        )
    )

    if previous_extension > 0.025:
        return False

    return (
        touched_ema20
        and
        closed_above
        and
        bullish
    )


def short_pullback(row, prev):

    if not market_quality(row):
        return False

    if row["ema20"] >= row["ema50"]:
        return False

    if row["rsi"] > 52:
        return False

    touched_ema20 = (
        row["high"] >=
        row["ema20"] * 0.996
    )

    closed_below = (
        row["close"] <
        row["ema20"]
    )

    bearish = (
        row["close"] <
        row["open"]
    )

    previous_extension = (
        abs(prev["close"] - prev["open"])
        /
        max(
            prev["close"],
            1e-12
        )
    )

    if previous_extension > 0.025:
        return False

    return (
        touched_ema20
        and
        closed_below
        and
        bearish
    )


# ============================================================
# MOMENTUM CONTINUATION
# ============================================================

def long_momentum(row, prev):

    if not market_quality(row):
        return False

    if row["ema20"] <= row["ema50"]:
        return False

    if row["rsi"] < 52:
        return False

    if row["close"] <= prev["high"]:
        return False

    if row["close"] <= row["open"]:
        return False

    # Avoid chasing very extended candle
    candle_move = (
        row["close"] -
        row["open"]
    ) / row["open"]

    if candle_move > 0.025:
        return False

    return True


def short_momentum(row, prev):

    if not market_quality(row):
        return False

    if row["ema20"] >= row["ema50"]:
        return False

    if row["rsi"] > 48:
        return False

    if row["close"] >= prev["low"]:
        return False

    if row["close"] >= row["open"]:
        return False

    candle_move = (
        row["open"] -
        row["close"]
    ) / row["open"]

    if candle_move > 0.025:
        return False

    return True


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(df, i):

    if i < 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    regime = get_regime(row)

    if regime is None:
        return None

    if regime == "LONG":

        pb = long_pullback(
            row,
            prev
        )

        mom = long_momentum(
            row,
            prev
        )

        if pb:
            return {
                "direction": "LONG",
                "setup": "PULLBACK"
            }

        if mom:
            return {
                "direction": "LONG",
                "setup": "MOMENTUM"
            }

    if regime == "SHORT":

        pb = short_pullback(
            row,
            prev
        )

        mom = short_momentum(
            row,
            prev
        )

        if pb:
            return {
                "direction": "SHORT",
                "setup": "PULLBACK"
            }

        if mom:
            return {
                "direction": "SHORT",
                "setup": "MOMENTUM"
            }

    return None


# ============================================================
# EXECUTION PRICES
# ============================================================

def execution_entry(raw_price, direction):

    if direction == "LONG":

        return raw_price * (
            1 + SLIPPAGE_RATE
        )

    return raw_price * (
        1 - SLIPPAGE_RATE
    )


def execution_exit(raw_price, direction):

    if direction == "LONG":

        return raw_price * (
            1 - SLIPPAGE_RATE
        )

    return raw_price * (
        1 + SLIPPAGE_RATE
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_idx,
    direction,
    setup
):

    entry_candle = df.iloc[
        entry_idx
    ]

    raw_entry = float(
        entry_candle["open"]
    )

    entry = execution_entry(
        raw_entry,
        direction
    )

    if direction == "LONG":

        tp_price = (
            entry *
            (1 + TP_PCT)
        )

        sl_price = (
            entry *
            (1 - SL_PCT)
        )

    else:

        tp_price = (
            entry *
            (1 - TP_PCT)
        )

        sl_price = (
            entry *
            (1 + SL_PCT)
        )

    last_idx = min(
        len(df) - 1,
        entry_idx + MAX_HOLD_BARS
    )

    for j in range(
        entry_idx,
        last_idx + 1
    ):

        candle = df.iloc[j]

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        if direction == "LONG":

            hit_tp = (
                high >= tp_price
            )

            hit_sl = (
                low <= sl_price
            )

        else:

            hit_tp = (
                low <= tp_price
            )

            hit_sl = (
                high >= sl_price
            )

        # ----------------------------------------------------
        # SAME CANDLE TP + SL
        # ----------------------------------------------------
        #
        # OHLC cannot tell us which happened first.
        # Conservative assumption = LOSS.
        # ----------------------------------------------------

        if hit_tp and hit_sl:

            exit_price = execution_exit(
                sl_price,
                direction
            )

            return make_trade(
                entry_candle,
                candle,
                direction,
                setup,
                entry,
                tp_price,
                sl_price,
                exit_price,
                "BOTH_HIT_SL",
                j - entry_idx
            )

        if hit_tp:

            exit_price = execution_exit(
                tp_price,
                direction
            )

            return make_trade(
                entry_candle,
                candle,
                direction,
                setup,
                entry,
                tp_price,
                sl_price,
                exit_price,
                "WIN",
                j - entry_idx
            )

        if hit_sl:

            exit_price = execution_exit(
                sl_price,
                direction
            )

            return make_trade(
                entry_candle,
                candle,
                direction,
                setup,
                entry,
                tp_price,
                sl_price,
                exit_price,
                "LOSS",
                j - entry_idx
            )

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    exit_candle = df.iloc[
        last_idx
    ]

    raw_exit = float(
        exit_candle["close"]
    )

    exit_price = execution_exit(
        raw_exit,
        direction
    )

    return make_trade(
        entry_candle,
        exit_candle,
        direction,
        setup,
        entry,
        tp_price,
        sl_price,
        exit_price,
        "TIMEOUT",
        last_idx - entry_idx
    )


# ============================================================
# TRADE RECORD
# ============================================================

def make_trade(
    entry_candle,
    exit_candle,
    direction,
    setup,
    entry,
    tp,
    sl,
    exit_price,
    result,
    hold_bars
):

    # Gross percentage
    if direction == "LONG":

        gross_pct = (
            exit_price - entry
        ) / entry

    else:

        gross_pct = (
            entry - exit_price
        ) / entry

    # Fees: entry + exit
    total_fee = (
        2 * FEE_RATE
    )

    net_pct = (
        gross_pct -
        total_fee
    )

    # R based on the intended 1% risk
    r_multiple = (
        net_pct / SL_PCT
    )

    if result == "WIN":

        r_multiple = min(
            r_multiple,
            2.0
        )

    if result in (
        "LOSS",
        "BOTH_HIT_SL"
    ):

        r_multiple = max(
            r_multiple,
            -1.0
        )

    return {
        "entry_time": entry_candle.name,
        "exit_time": exit_candle.name,

        "direction": direction,
        "setup": setup,

        "entry": entry,
        "tp": tp,
        "sl": sl,
        "exit": exit_price,

        "result": result,

        "hold_bars": hold_bars,

        "gross_pct": gross_pct * 100,
        "net_pct": net_pct * 100,

        "r_multiple": r_multiple,
    }


# ============================================================
# SYMBOL BACKTEST
# ============================================================

def backtest_symbol(
    name,
    df
):

    trades = []

    i = 220

    while i < len(df) - 2:

        signal = generate_signal(
            df,
            i
        )

        if signal is None:

            i += 1
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Signal is calculated from CLOSED candle i.
        #
        # Entry is OPEN of candle i+1.
        #
        # This prevents entering using information from the
        # candle's future.
        # ----------------------------------------------------

        entry_idx = i + 1

        trade = simulate_trade(
            df,
            entry_idx,
            signal["direction"],
            signal["setup"]
        )

        trade["symbol"] = name

        # Reorder
        trade = {
            "symbol": trade["symbol"],
            "direction": trade["direction"],
            "setup": trade["setup"],
            "entry_time": trade["entry_time"],
            "exit_time": trade["exit_time"],
            "entry": trade["entry"],
            "tp": trade["tp"],
            "sl": trade["sl"],
            "exit": trade["exit"],
            "result": trade["result"],
            "hold_bars": trade["hold_bars"],
            "gross_pct": trade["gross_pct"],
            "net_pct": trade["net_pct"],
            "r_multiple": trade["r_multiple"],
        }

        trades.append(
            trade
        )

        # No overlapping positions for same symbol
        exit_time = trade[
            "exit_time"
        ]

        exit_positions = df.index[
            df.index <= exit_time
        ]

        if len(exit_positions) > 0:

            last_position = df.index.get_loc(
                exit_positions[-1]
            )

            i = last_position + 1

        else:

            i += 1

    return trades


# ============================================================
# PORTFOLIO LOCK
# ============================================================

def portfolio_lock(
    all_candidates
):

    if not all_candidates:
        return []

    candidates = sorted(
        all_candidates,
        key=lambda x: x["entry_time"]
    )

    accepted = []

    portfolio_free_time = None

    for trade in candidates:

        if portfolio_free_time is None:

            accepted.append(
                trade
            )

            portfolio_free_time = (
                trade["exit_time"]
            )

            continue

        # Only one active trade across the entire portfolio
        if trade["entry_time"] > portfolio_free_time:

            accepted.append(
                trade
            )

            portfolio_free_time = (
                trade["exit_time"]
            )

    return accepted


# ============================================================
# STATISTICS
# ============================================================

def profit_factor(trades):

    profits = sum(
        max(t["r_multiple"], 0)
        for t in trades
    )

    losses = abs(
        sum(
            min(t["r_multiple"], 0)
            for t in trades
        )
    )

    if losses == 0:
        return float("inf")

    return profits / losses


def max_drawdown_r(trades):

    if not trades:
        return 0

    ordered = sorted(
        trades,
        key=lambda x: x["exit_time"]
    )

    equity = 0
    peak = 0
    max_dd = 0

    for t in ordered:

        equity += t["r_multiple"]

        peak = max(
            peak,
            equity
        )

        dd = equity - peak

        max_dd = min(
            max_dd,
            dd
        )

    return max_dd


def summarize(
    trades,
    candidates_count,
    start_date,
    end_date
):

    print()
    print("=" * 78)
    print("HUNTER-X V9 — FINAL BACKTEST")
    print("=" * 78)

    if not trades:

        print("NO TRADES")
        return

    closed = [
        t for t in trades
        if t["result"] != "TIMEOUT"
    ]

    wins = [
        t for t in trades
        if t["result"] == "WIN"
    ]

    losses = [
        t for t in trades
        if t["result"] in (
            "LOSS",
            "BOTH_HIT_SL"
        )
    ]

    timeouts = [
        t for t in trades
        if t["result"] == "TIMEOUT"
    ]

    total = len(trades)

    decisive = len(
        closed
    )

    wr = (
        100 *
        len(wins) /
        decisive
        if decisive
        else 0
    )

    net_r = sum(
        t["r_multiple"]
        for t in trades
    )

    avg_r = (
        net_r / total
        if total
        else 0
    )

    pf = profit_factor(
        trades
    )

    dd = max_drawdown_r(
        trades
    )

    days = max(
        1,
        (
            end_date -
            start_date
        ).total_seconds()
        / 86400
    )

    trades_per_year = (
        total *
        365 /
        days
    )

    print(
        f"Period             : "
        f"{start_date.date()} -> "
        f"{end_date.date()}"
    )

    print(
        f"Candidate signals  : "
        f"{candidates_count}"
    )

    print(
        f"Final trades       : "
        f"{total}"
    )

    print(
        f"Wins               : "
        f"{len(wins)}"
    )

    print(
        f"Losses             : "
        f"{len(losses)}"
    )

    print(
        f"Timeouts           : "
        f"{len(timeouts)}"
    )

    print(
        f"Win Rate           : "
        f"{wr:.2f}%"
    )

    print(
        f"Profit Factor      : "
        f"{pf:.3f}"
    )

    print(
        f"Net R              : "
        f"{net_r:.2f}R"
    )

    print(
        f"Average Trade      : "
        f"{avg_r:.4f}R"
    )

    print(
        f"Max Drawdown       : "
        f"{dd:.2f}R"
    )

    print(
        f"Trades / Year      : "
        f"{trades_per_year:.0f}"
    )

    print("=" * 78)


# ============================================================
# GROUP REPORT
# ============================================================

def grouped_report(
    trades,
    field
):

    groups = sorted(
        set(
            t[field]
            for t in trades
        )
    )

    print()
    print(
        f"--- BY {field.upper()} ---"
    )

    for value in groups:

        x = [
            t for t in trades
            if t[field] == value
        ]

        wins = sum(
            t["result"] == "WIN"
            for t in x
        )

        decisive = sum(
            t["result"] != "TIMEOUT"
            for t in x
        )

        wr = (
            100 * wins / decisive
            if decisive
            else 0
        )

        net_r = sum(
            t["r_multiple"]
            for t in x
        )

        pf = profit_factor(
            x
        )

        print(
            f"{str(value):12s} | "
            f"Trades={len(x):4d} | "
            f"Win={wins:4d} | "
            f"WR={wr:6.2f}% | "
            f"NetR={net_r:8.2f} | "
            f"PF={pf:.3f}"
        )


# ============================================================
# MONTHLY REPORT
# ============================================================

def monthly_report(
    trades
):

    print()
    print("--- MONTHLY REPORT ---")

    months = {}

    for t in trades:

        month = pd.Timestamp(
            t["entry_time"]
        ).strftime(
            "%Y-%m"
        )

        months.setdefault(
            month,
            []
        ).append(t)

    for month in sorted(
        months.keys()
    ):

        x = months[month]

        wins = sum(
            t["result"] == "WIN"
            for t in x
        )

        decisive = sum(
            t["result"] != "TIMEOUT"
            for t in x
        )

        wr = (
            100 * wins / decisive
            if decisive
            else 0
        )

        net_r = sum(
            t["r_multiple"]
            for t in x
        )

        print(
            f"{month} | "
            f"Trades={len(x):4d} | "
            f"WR={wr:6.2f}% | "
            f"NetR={net_r:8.2f}"
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    trades
):

    if not trades:
        return

    fields = [
        "symbol",
        "direction",
        "setup",
        "entry_time",
        "exit_time",
        "entry",
        "tp",
        "sl",
        "exit",
        "result",
        "hold_bars",
        "gross_pct",
        "net_pct",
        "r_multiple"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for trade in trades:

            row = dict(
                trade
            )

            row["entry_time"] = str(
                row["entry_time"]
            )

            row["exit_time"] = str(
                row["exit_time"]
            )

            writer.writerow(
                {
                    k: row.get(k, "")
                    for k in fields
                }
            )


# ============================================================
# ANTI LOOKAHEAD AUDIT
# ============================================================

def anti_lookahead_audit():

    print()
    print("=" * 78)
    print("ANTI-LOOKAHEAD AUDIT")
    print("=" * 78)

    print(
        "1H signal candle        : CLOSED"
    )

    print(
        "Entry                   : NEXT 1H OPEN"
    )

    print(
        "4H data                 : PREVIOUS CLOSED 4H"
    )

    print(
        "Current forming 4H      : NOT USED"
    )

    print(
        "Current forming 1H      : NOT USED"
    )

    print(
        "Future candles in entry : NOT USED"
    )

    print(
        "Same candle TP+SL       : SL"
    )

    print(
        "Overlapping symbol      : NO"
    )

    print(
        "Portfolio simultaneous  : NO"
    )

    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    now = utc_now()

    end_date = now

    start_date = (
        now -
        timedelta(
            days=BACKTEST_DAYS
            + WARMUP_DAYS
        )
    )

    print()
    print("=" * 78)
    print("HUNTER-X V9")
    print("LBank USDT FUTURES / SWAP")
    print("=" * 78)

    print(
        f"Backtest days : "
        f"{BACKTEST_DAYS}"
    )

    print(
        f"TP            : "
        f"{TP_PCT * 100:.2f}%"
    )

    print(
        f"SL            : "
        f"{SL_PCT * 100:.2f}%"
    )

    print(
        f"RR            : "
        f"1:{RR:.1f}"
    )

    print(
        f"Max hold      : "
        f"{MAX_HOLD_BARS} hours"
    )

    print(
        f"Fee/side      : "
        f"{FEE_RATE * 100:.3f}%"
    )

    print(
        f"Slippage/side : "
        f"{SLIPPAGE_RATE * 100:.3f}%"
    )

    print()

    # --------------------------------------------------------
    # LOAD MARKETS
    # --------------------------------------------------------

    print(
        "Loading LBank markets..."
    )

    exchange.load_markets()

    resolved_symbols = {}

    print()
    print("FUTURES MARKET MAPPING")
    print("-" * 78)

    for name in SYMBOLS:

        symbol, market = find_futures_market(
            exchange,
            name
        )

        if symbol is None:

            print(
                f"{name:6s} -> NOT FOUND"
            )

            continue

        resolved_symbols[name] = symbol

        market_type = (
            "SWAP"
            if market.get("swap")
            else "FUTURE"
        )

        print(
            f"{name:6s} -> "
            f"{symbol:25s} "
            f"[{market_type}]"
        )

    if not resolved_symbols:

        raise RuntimeError(
            "No LBank USDT futures markets found."
        )

    # --------------------------------------------------------
    # DOWNLOAD + BACKTEST
    # --------------------------------------------------------

    all_candidates = []

    symbol_stats = {}

    for name, futures_symbol in resolved_symbols.items():

        print()
        print("=" * 78)
        print(
            f"{name} | {futures_symbol}"
        )
        print("=" * 78)

        # 1H
        df1 = fetch_lbank_ohlcv(
            futures_symbol,
            TIMEFRAME_1H,
            start_date,
            end_date
        )

        # 4H
        df4 = fetch_lbank_ohlcv(
            futures_symbol,
            TIMEFRAME_4H,
            start_date,
            end_date
        )

        if df1.empty or df4.empty:

            print(
                f"{name}: insufficient data"
            )

            continue

        print(
            f"1H candles: {len(df1)}"
        )

        print(
            f"4H candles: {len(df4)}"
        )

        # ----------------------------------------------------
        # Indicators
        # ----------------------------------------------------

        df1 = prepare_1h(
            df1
        )

        df4 = prepare_4h(
            df4
        )

        # ----------------------------------------------------
        # Previous closed 4H
        # ----------------------------------------------------

        df1 = attach_previous_4h(
            df1,
            df4
        )

        # Remove rows without enough data
        df1 = df1.dropna(
            subset=[
                "ema200",
                "rsi",
                "adx",
                "htf_close",
                "htf_ema20",
                "htf_ema50",
                "htf_ema200",
                "htf_rsi",
                "htf_adx"
            ]
        )

        if len(df1) < 300:

            print(
                f"{name}: not enough warmup"
            )

            continue

        # ----------------------------------------------------
        # Candidate trades
        # ----------------------------------------------------

        candidates = []

        for i in range(
            220,
            len(df1) - 2
        ):

            signal = generate_signal(
                df1,
                i
            )

            if signal is None:
                continue

            entry_idx = i + 1

            trade = simulate_trade(
                df1,
                entry_idx,
                signal["direction"],
                signal["setup"]
            )

            trade["symbol"] = name

            candidates.append(
                trade
            )

        print(
            f"Candidate trades: "
            f"{len(candidates)}"
        )

        symbol_stats[name] = {
            "candidates": len(candidates)
        }

        all_candidates.extend(
            candidates
        )

    # --------------------------------------------------------
    # PORTFOLIO LOCK
    # --------------------------------------------------------

    final_trades = portfolio_lock(
        all_candidates
    )

    print()
    print("=" * 78)
    print("PORTFOLIO LOCK")
    print("=" * 78)

    print(
        f"All candidates : "
        f"{len(all_candidates)}"
    )

    print(
        f"Accepted trades: "
        f"{len(final_trades)}"
    )

    print(
        f"Rejected by lock: "
        f"{len(all_candidates) - len(final_trades)}"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    final_trades.sort(
        key=lambda x: x["entry_time"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        final_trades
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summarize(
        final_trades,
        len(all_candidates),
        start_date,
        end_date
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    grouped_report(
        final_trades,
        "symbol"
    )

    grouped_report(
        final_trades,
        "direction"
    )

    grouped_report(
        final_trades,
        "setup"
    )

    monthly_report(
        final_trades
    )

    # --------------------------------------------------------
    # Result distribution
    # --------------------------------------------------------

    print()
    print("--- RESULT DISTRIBUTION ---")

    for result in [
        "WIN",
        "LOSS",
        "BOTH_HIT_SL",
        "TIMEOUT"
    ]:

        count = sum(
            t["result"] == result
            for t in final_trades
        )

        print(
            f"{result:15s}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    anti_lookahead_audit()

    print()
    print(
        f"CSV saved as: "
        f"{OUTPUT_CSV}"
    )

    print()
    print("=" * 78)
    print("BACKTEST FINISHED")
    print("=" * 78)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
