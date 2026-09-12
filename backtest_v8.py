import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    print("📦 Installing ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np


# ============================================================
# HUNTER-1% v1 — CLEAN LBank 1 YEAR BACKTEST
#
# TP = +1.00%
# SL = -0.50%
# RR = 1:2
#
# TIMEFRAMES:
# 4H = trend
# 1H = setup / breakout / retest / entry
#
# ANTI LOOK-AHEAD:
# - فقط 4H بسته‌شده قبلی
# - breakout فقط با کندل‌های قبلی
# - retest فقط بعد از breakout
# - ورود در OPEN کندل بعدی
# - TP/SL فقط از کندل ورود به بعد
# - کندل جاری حذف شده
#
# OVERLAP:
# - قفل کامل پوزیشن
# - فقط یک معامله همزمان در کل پورتفولیو
# ============================================================


# ============================================================
# CONFIG
# ============================================================

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
}

TIMEFRAME = "1h"

# ------------------------------------------------------------
# TP / SL
# ------------------------------------------------------------

TP_PCT = 0.0100       # +1%
SL_PCT = 0.0050       # -0.5%

RR = TP_PCT / SL_PCT


# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_LENGTH = 14
ADX_LENGTH = 14
ATR_LENGTH = 14

VOLUME_MA_LENGTH = 20


# ------------------------------------------------------------
# Breakout
# ------------------------------------------------------------

BREAKOUT_LOOKBACK = 20

BREAKOUT_BODY_MIN = 0.55

VOLUME_MULTIPLIER = 1.10


# ------------------------------------------------------------
# Retest
# ------------------------------------------------------------

RETEST_TOLERANCE = 0.0015      # 0.15%

MAX_RETEST_BARS = 6


# ------------------------------------------------------------
# Volatility
# ------------------------------------------------------------

MAX_ATR_PERCENT = 0.0075       # 0.75%


# ------------------------------------------------------------
# Score
# ------------------------------------------------------------

MIN_SCORE = 8


# ------------------------------------------------------------
# Cooldown
# ------------------------------------------------------------

COOLDOWN_BARS = 3


# ------------------------------------------------------------
# Maximum holding time
# ------------------------------------------------------------

MAX_HOLDING_BARS = 12


# ------------------------------------------------------------
# Trading costs
# ------------------------------------------------------------

# LBank Futures taker fee example
TAKER_FEE = 0.0006

# Assumed slippage
SLIPPAGE = 0.0002

ROUND_TRIP_COST = 2 * (TAKER_FEE + SLIPPAGE)


# If TP and SL happen inside the SAME candle,
# assume SL happened first.
CONSERVATIVE_SAME_CANDLE = True


# ============================================================
# LBank CONNECTION
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True
})

exchange.load_markets()


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, length=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(
        ~((avg_loss == 0) & (avg_gain > 0)),
        100
    )

    rsi = rsi.where(
        ~((avg_gain == 0) & (avg_loss > 0)),
        0
    )

    return rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, length=14):

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]

    tr2 = (df["High"] - previous_close).abs()

    tr3 = (df["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    return atr


# ============================================================
# ADX
# ============================================================

def calculate_adx(df, length=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length
        ).mean()
        /
        atr.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length
        ).mean()
        /
        atr.replace(0, np.nan)
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    return adx


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["EMA20"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False,
        min_periods=EMA_FAST
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=EMA_MID,
        adjust=False,
        min_periods=EMA_MID
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False,
        min_periods=EMA_SLOW
    ).mean()

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_LENGTH
    )

    df["ATR"] = calculate_atr(
        df,
        ATR_LENGTH
    )

    df["ADX"] = calculate_adx(
        df,
        ADX_LENGTH
    )

    df["Volume_MA"] = df["Volume"].rolling(
        VOLUME_MA_LENGTH,
        min_periods=VOLUME_MA_LENGTH
    ).mean()

    return df


# ============================================================
# DOWNLOAD LBank DATA
# ============================================================

def download_data(symbol):

    print(f"\n📥 Downloading {symbol} from LBank...")

    now_ms = exchange.milliseconds()

    # 20 extra days for indicator warm-up
    start_date = (
        datetime.now(timezone.utc)
        - timedelta(days=385)
    )

    since = int(
        start_date.timestamp() * 1000
    )

    all_data = []

    while since < now_ms:

        try:

            candles = exchange.fetch_ohlcv(
                SYMBOLS[symbol],
                timeframe="1h",
                since=since,
                limit=1000
            )

        except Exception as e:

            print(
                f"\n⚠️ Error downloading {symbol}: {e}"
            )

            break

        if not candles:
            break

        all_data.extend(candles)

        last_timestamp = candles[-1][0]

        next_since = last_timestamp + 1

        if next_since <= since:
            break

        since = next_since

        print(
            f"   Candles downloaded: "
            f"{len(all_data):,}",
            end="\r"
        )

        if len(candles) < 1000:
            break

    if not all_data:
        return None

    df = pd.DataFrame(
        all_data,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    # Remove duplicates
    df = df.drop_duplicates(
        subset="Timestamp"
    )

    df = df.sort_values(
        "Timestamp"
    )

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
    )

    df = df.set_index("Date")

    # --------------------------------------------------------
    # REMOVE CURRENTLY FORMING CANDLE
    # --------------------------------------------------------

    current_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    df = df[
        df.index < current_hour
    ]

    print(
        f"\n✔️ {symbol}: "
        f"{len(df):,} CLOSED 1H candles"
    )

    return df


# ============================================================
# BUILD 4H
# ============================================================

def build_4h(df1h):

    df4h = df1h.resample(
        "4h",
        label="left",
        closed="left"
    ).agg({

        "Open": "first",

        "High": "max",

        "Low": "min",

        "Close": "last",

        "Volume": "sum"

    }).dropna()

    df4h = add_indicators(
        df4h
    )

    return df4h


# ============================================================
# ATTACH PREVIOUS CLOSED 4H
# ============================================================

def attach_previous_4h(
    df1h,
    df4h
):

    df = df1h.copy()

    # VERY IMPORTANT
    #
    # At 12:00 the 12-16 4H candle has just started.
    # Therefore it CANNOT be used.
    #
    # We always use the PREVIOUS completed 4H candle.

    df["4H_Time"] = (
        df.index.floor("4h")
        - pd.Timedelta(hours=4)
    )

    htf = df4h[
        [
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI",
            "ADX",
            "Close"
        ]
    ].copy()

    htf = htf.rename(
        columns={
            "EMA20": "HTF_EMA20",
            "EMA50": "HTF_EMA50",
            "EMA200": "HTF_EMA200",
            "RSI": "HTF_RSI",
            "ADX": "HTF_ADX",
            "Close": "HTF_Close"
        }
    )

    df = df.join(
        htf,
        on="4H_Time"
    )

    return df


# ============================================================
# PREPARE 1H DATA
# ============================================================

def prepare_1h(df):

    df = add_indicators(df)

    # --------------------------------------------------------
    # CRITICAL:
    #
    # shift(1) means the current candle is NEVER included
    # when calculating breakout levels.
    # --------------------------------------------------------

    df["Previous20High"] = (
        df["High"]
        .shift(1)
        .rolling(
            BREAKOUT_LOOKBACK,
            min_periods=BREAKOUT_LOOKBACK
        )
        .max()
    )

    df["Previous20Low"] = (
        df["Low"]
        .shift(1)
        .rolling(
            BREAKOUT_LOOKBACK,
            min_periods=BREAKOUT_LOOKBACK
        )
        .min()
    )

    df["CandleRange"] = (
        df["High"] -
        df["Low"]
    )

    df["Body"] = (
        df["Close"] -
        df["Open"]
    ).abs()

    df["BodyRatio"] = np.where(
        df["CandleRange"] > 0,

        df["Body"] /
        df["CandleRange"],

        0
    )

    df["ATR_Pct"] = (
        df["ATR"] /
        df["Close"]
    )

    return df


# ============================================================
# LONG SCORE
# ============================================================

def calculate_long_score(row):

    score = 0

    # 4H close above EMA20
    if row["HTF_Close"] > row["HTF_EMA20"]:
        score += 1

    # EMA structure
    if (
        row["HTF_EMA20"] >
        row["HTF_EMA50"] >
        row["HTF_EMA200"]
    ):
        score += 2

    # RSI
    if row["HTF_RSI"] > 52:
        score += 1

    # Strong trend
    if row["HTF_ADX"] > 25:
        score += 1

    # 1H ADX
    if row["ADX"] > 20:
        score += 1

    # Breakout candle
    if row["BodyRatio"] >= BREAKOUT_BODY_MIN:
        score += 2

    # Retest confirmation
    if row["RetestConfirmed"]:
        score += 2

    # Volume
    if (
        row["Volume"] >
        row["Volume_MA"] *
        VOLUME_MULTIPLIER
    ):
        score += 1

    return score


# ============================================================
# SHORT SCORE
# ============================================================

def calculate_short_score(row):

    score = 0

    # 4H close below EMA20
    if row["HTF_Close"] < row["HTF_EMA20"]:
        score += 1

    # EMA structure
    if (
        row["HTF_EMA20"] <
        row["HTF_EMA50"] <
        row["HTF_EMA200"]
    ):
        score += 2

    # RSI
    if row["HTF_RSI"] < 48:
        score += 1

    # Strong trend
    if row["HTF_ADX"] > 25:
        score += 1

    # 1H ADX
    if row["ADX"] > 20:
        score += 1

    # Breakdown candle
    if row["BodyRatio"] >= BREAKOUT_BODY_MIN:
        score += 2

    # Retest
    if row["RetestConfirmed"]:
        score += 2

    # Volume
    if (
        row["Volume"] >
        row["Volume_MA"] *
        VOLUME_MULTIPLIER
    ):
        score += 1

    return score


# ============================================================
# RESOLVE TRADE
# ============================================================

def resolve_trade(
    df,
    entry_index,
    side,
    entry_price,
    sl,
    tp
):

    last_index = min(
        len(df) - 1,
        entry_index + MAX_HOLDING_BARS
    )

    for j in range(
        entry_index,
        last_index + 1
    ):

        candle = df.iloc[j]

        # ====================================================
        # LONG
        # ====================================================

        if side == "LONG":

            hit_sl = (
                candle["Low"] <= sl
            )

            hit_tp = (
                candle["High"] >= tp
            )

            # BOTH TP AND SL IN SAME CANDLE
            if hit_sl and hit_tp:

                if CONSERVATIVE_SAME_CANDLE:

                    exit_price = (
                        sl *
                        (1 - SLIPPAGE)
                    )

                    return (
                        j,
                        exit_price,
                        "LOSS"
                    )

                else:

                    exit_price = (
                        tp *
                        (1 - SLIPPAGE)
                    )

                    return (
                        j,
                        exit_price,
                        "WIN"
                    )

            # SL
            if hit_sl:

                exit_price = (
                    sl *
                    (1 - SLIPPAGE)
                )

                return (
                    j,
                    exit_price,
                    "LOSS"
                )

            # TP
            if hit_tp:

                exit_price = (
                    tp *
                    (1 - SLIPPAGE)
                )

                return (
                    j,
                    exit_price,
                    "WIN"
                )

        # ====================================================
        # SHORT
        # ====================================================

        else:

            hit_sl = (
                candle["High"] >= sl
            )

            hit_tp = (
                candle["Low"] <= tp
            )

            # BOTH TP AND SL
            if hit_sl and hit_tp:

                if CONSERVATIVE_SAME_CANDLE:

                    exit_price = (
                        sl *
                        (1 + SLIPPAGE)
                    )

                    return (
                        j,
                        exit_price,
                        "LOSS"
                    )

                else:

                    exit_price = (
                        tp *
                        (1 + SLIPPAGE)
                    )

                    return (
                        j,
                        exit_price,
                        "WIN"
                    )

            # SL
            if hit_sl:

                exit_price = (
                    sl *
                    (1 + SLIPPAGE)
                )

                return (
                    j,
                    exit_price,
                    "LOSS"
                )

            # TP
            if hit_tp:

                exit_price = (
                    tp *
                    (1 + SLIPPAGE)
                )

                return (
                    j,
                    exit_price,
                    "WIN"
                )

    # ========================================================
    # TIMEOUT
    # ========================================================

    exit_idx = last_index

    exit_price = df.iloc[
        exit_idx
    ]["Close"]

    if side == "LONG":

        pnl = (
            exit_price -
            entry_price
        ) / entry_price

    else:

        pnl = (
            entry_price -
            exit_price
        ) / entry_price

    if pnl > 0:
        outcome = "TIMEOUT_PROFIT"

    elif pnl < 0:
        outcome = "TIMEOUT_LOSS"

    else:
        outcome = "TIMEOUT_BE"

    return (
        exit_idx,
        exit_price,
        outcome
    )


# ============================================================
# BACKTEST SINGLE SYMBOL
# ============================================================

def backtest_symbol(
    symbol,
    df
):

    trades = []

    pending_breakout = None

    i = 0

    while i < len(df) - 1:

        row = df.iloc[i]

        # ====================================================
        # PENDING BREAKOUT -> WAIT FOR RETEST
        # ====================================================

        if pending_breakout is not None:

            bars_waiting = (
                i -
                pending_breakout[
                    "breakout_index"
                ]
            )

            # Retest took too long
            if bars_waiting > MAX_RETEST_BARS:

                pending_breakout = None

                i += 1

                continue

            level = (
                pending_breakout[
                    "level"
                ]
            )

            side = (
                pending_breakout[
                    "side"
                ]
            )

            # =================================================
            # LONG RETEST
            # =================================================

            if side == "LONG":

                touched = (
                    row["Low"]
                    <=
                    level *
                    (
                        1 +
                        RETEST_TOLERANCE
                    )
                )

                confirmed = (
                    touched
                    and
                    row["Close"] > level
                    and
                    row["Close"] > row["Open"]
                )

                if confirmed:

                    row = row.copy()

                    row[
                        "RetestConfirmed"
                    ] = True

                    score = (
                        calculate_long_score(
                            row
                        )
                    )

                    valid = (

                        score >= MIN_SCORE

                        and
                        row["ADX"] > 20

                        and
                        row["Volume"]
                        >
                        row["Volume_MA"]
                        *
                        VOLUME_MULTIPLIER

                        and
                        row["ATR_Pct"]
                        <=
                        MAX_ATR_PERCENT
                    )

                    if valid:

                        entry_index = i + 1

                        if entry_index >= len(df):

                            break

                        entry_candle = (
                            df.iloc[
                                entry_index
                            ]
                        )

                        # ENTRY ONLY ON NEXT CANDLE OPEN
                        raw_entry = (
                            entry_candle["Open"]
                        )

                        entry_price = (
                            raw_entry *
                            (
                                1 +
                                SLIPPAGE
                            )
                        )

                        tp = (
                            entry_price *
                            (
                                1 +
                                TP_PCT
                            )
                        )

                        sl = (
                            entry_price *
                            (
                                1 -
                                SL_PCT
                            )
                        )

                        (
                            exit_index,
                            exit_price,
                            outcome
                        ) = resolve_trade(

                            df,

                            entry_index,

                            "LONG",

                            entry_price,

                            sl,

                            tp
                        )

                        # -------------------------------------
                        # Return
                        # -------------------------------------

                        gross_return = (
                            exit_price -
                            entry_price
                        ) / entry_price

                        net_return = (
                            gross_return -
                            ROUND_TRIP_COST
                        )

                        trades.append({

                            "Symbol": symbol,

                            "Side": "LONG",

                            "BreakoutTime":
                                pending_breakout[
                                    "breakout_time"
                                ],

                            "SignalTime":
                                row.name,

                            "EntryTime":
                                entry_candle.name,

                            "ExitTime":
                                df.iloc[
                                    exit_index
                                ].name,

                            "Entry":
                                entry_price,

                            "Exit":
                                exit_price,

                            "TP":
                                tp,

                            "SL":
                                sl,

                            "Score":
                                score,

                            "Outcome":
                                outcome,

                            "GrossPct":
                                gross_return *
                                100,

                            "NetPct":
                                net_return *
                                100,

                            "BarsHeld":
                                exit_index -
                                entry_index
                        })

                        # =================================================
                        # HARD LOCK
                        #
                        # Nothing can be evaluated while this trade
                        # is open.
                        # =================================================

                        i = (
                            exit_index +
                            COOLDOWN_BARS +
                            1
                        )

                        pending_breakout = None

                        continue

                    pending_breakout = None

                    i += 1

                    continue

            # =================================================
            # SHORT RETEST
            # =================================================

            else:

                touched = (
                    row["High"]
                    >=
                    level *
                    (
                        1 -
                        RETEST_TOLERANCE
                    )
                )

                confirmed = (
                    touched
                    and
                    row["Close"] < level
                    and
                    row["Close"] < row["Open"]
                )

                if confirmed:

                    row = row.copy()

                    row[
                        "RetestConfirmed"
                    ] = True

                    score = (
                        calculate_short_score(
                            row
                        )
                    )

                    valid = (

                        score >= MIN_SCORE

                        and
                        row["ADX"] > 20

                        and
                        row["Volume"]
                        >
                        row["Volume_MA"]
                        *
                        VOLUME_MULTIPLIER

                        and
                        row["ATR_Pct"]
                        <=
                        MAX_ATR_PERCENT
                    )

                    if valid:

                        entry_index = i + 1

                        if entry_index >= len(df):

                            break

                        entry_candle = (
                            df.iloc[
                                entry_index
                            ]
                        )

                        # ENTRY NEXT CANDLE OPEN
                        raw_entry = (
                            entry_candle["Open"]
                        )

                        entry_price = (
                            raw_entry *
                            (
                                1 -
                                SLIPPAGE
                            )
                        )

                        tp = (
                            entry_price *
                            (
                                1 -
                                TP_PCT
                            )
                        )

                        sl = (
                            entry_price *
                            (
                                1 +
                                SL_PCT
                            )
                        )

                        (
                            exit_index,
                            exit_price,
                            outcome
                        ) = resolve_trade(

                            df,

                            entry_index,

                            "SHORT",

                            entry_price,

                            sl,

                            tp
                        )

                        gross_return = (
                            entry_price -
                            exit_price
                        ) / entry_price

                        net_return = (
                            gross_return -
                            ROUND_TRIP_COST
                        )

                        trades.append({

                            "Symbol": symbol,

                            "Side": "SHORT",

                            "BreakoutTime":
                                pending_breakout[
                                    "breakout_time"
                                ],

                            "SignalTime":
                                row.name,

                            "EntryTime":
                                entry_candle.name,

                            "ExitTime":
                                df.iloc[
                                    exit_index
                                ].name,

                            "Entry":
                                entry_price,

                            "Exit":
                                exit_price,

                            "TP":
                                tp,

                            "SL":
                                sl,

                            "Score":
                                score,

                            "Outcome":
                                outcome,

                            "GrossPct":
                                gross_return *
                                100,

                            "NetPct":
                                net_return *
                                100,

                            "BarsHeld":
                                exit_index -
                                entry_index
                        })

                        # HARD OVERLAP LOCK
                        i = (
                            exit_index +
                            COOLDOWN_BARS +
                            1
                        )

                        pending_breakout = None

                        continue

                    pending_breakout = None

                    i += 1

                    continue

            i += 1

            continue

        # ====================================================
        # NO PENDING BREAKOUT
        # SEARCH NEW BREAKOUT
        # ====================================================

        needed = [

            "HTF_EMA20",
            "HTF_EMA50",
            "HTF_EMA200",
            "HTF_RSI",
            "HTF_ADX",
            "HTF_Close",

            "ADX",
            "ATR",
            "Volume_MA",

            "Previous20High",
            "Previous20Low"
        ]

        if row[needed].isna().any():

            i += 1

            continue

        # ====================================================
        # VOLATILITY FILTER
        # ====================================================

        if (
            row["ATR_Pct"]
            >
            MAX_ATR_PERCENT
        ):

            i += 1

            continue

        # ====================================================
        # 4H TREND
        # ====================================================

        bullish_4h = (

            row["HTF_Close"]
            >
            row["HTF_EMA20"]

            and

            row["HTF_EMA20"]
            >
            row["HTF_EMA50"]

            and

            row["HTF_EMA50"]
            >
            row["HTF_EMA200"]

            and

            row["HTF_RSI"]
            >
            52

            and

            row["HTF_ADX"]
            >
            18
        )

        bearish_4h = (

            row["HTF_Close"]
            <
            row["HTF_EMA20"]

            and

            row["HTF_EMA20"]
            <
            row["HTF_EMA50"]

            and

            row["HTF_EMA50"]
            <
            row["HTF_EMA200"]

            and

            row["HTF_RSI"]
            <
            48

            and

            row["HTF_ADX"]
            >
            18
        )

        # ====================================================
        # 1H BREAKOUT
        # ====================================================

        bullish_breakout = (

            row["Close"]
            >
            row["Previous20High"]

            and

            row["Close"]
            >
            row["Open"]

            and

            row["BodyRatio"]
            >=
            BREAKOUT_BODY_MIN

            and

            row["Volume"]
            >
            row["Volume_MA"]
            *
            VOLUME_MULTIPLIER

            and

            row["ADX"]
            >
            20
        )

        bearish_breakdown = (

            row["Close"]
            <
            row["Previous20Low"]

            and

            row["Close"]
            <
            row["Open"]

            and

            row["BodyRatio"]
            >=
            BREAKOUT_BODY_MIN

            and

            row["Volume"]
            >
            row["Volume_MA"]
            *
            VOLUME_MULTIPLIER

            and

            row["ADX"]
            >
            20
        )

        # ====================================================
        # CREATE LONG PENDING SETUP
        # ====================================================

        if (
            bullish_4h
            and
            bullish_breakout
        ):

            pending_breakout = {

                "side":
                    "LONG",

                "level":
                    float(
                        row["Previous20High"]
                    ),

                "breakout_index":
                    i,

                "breakout_time":
                    row.name
            }

        # ====================================================
        # CREATE SHORT PENDING SETUP
        # ====================================================

        elif (
            bearish_4h
            and
            bearish_breakdown
        ):

            pending_breakout = {

                "side":
                    "SHORT",

                "level":
                    float(
                        row["Previous20Low"]
                    ),

                "breakout_index":
                    i,

                "breakout_time":
                    row.name
            }

        i += 1

    return trades


# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 72)
print("🚀 HUNTER-1% v1 — LBank CLEAN BACKTEST")
print("=" * 72)

print(
    f"🎯 TP = +{TP_PCT * 100:.2f}%"
)

print(
    f"🛑 SL = -{SL_PCT * 100:.2f}%"
)

print(
    f"⚖️ RR = 1:{RR:.1f}"
)

print(
    f"💰 Fee/side = {TAKER_FEE * 100:.3f}%"
)

print(
    f"📉 Slippage/side = {SLIPPAGE * 100:.3f}%"
)

print(
    f"⭐ Minimum Score = {MIN_SCORE}/11"
)

print(
    f"📊 Markets = {', '.join(SYMBOLS.keys())}"
)

print(
    "⏱️ Timeframes = 1H + CLOSED previous 4H"
)

print(
    "🔒 Portfolio overlap = LOCKED"
)

print(
    "🚫 Look-ahead = OFF"
)

print("=" * 72)


# ============================================================
# DOWNLOAD ALL DATA
# ============================================================

datasets = {}

for symbol in SYMBOLS:

    df1h = download_data(
        symbol
    )

    if df1h is None:

        continue

    if len(df1h) < 1000:

        print(
            f"❌ Not enough data for {symbol}"
        )

        continue

    # Build 4H
    df4h = build_4h(
        df1h
    )

    # Attach previous CLOSED 4H
    df = attach_previous_4h(
        df1h,
        df4h
    )

    # Indicators + breakout levels
    df = prepare_1h(
        df
    )

    # --------------------------------------------------------
    # Exactly last 365 days
    # --------------------------------------------------------

    end_time = df.index.max()

    start_time = (
        end_time -
        pd.Timedelta(days=365)
    )

    df = df[
        df.index >= start_time
    ].copy()

    datasets[symbol] = df

    print(
        f"✔️ {symbol} ready: "
        f"{len(df):,} 1H candles"
    )


# ============================================================
# RUN INDIVIDUAL TESTS
# ============================================================

print("\n")
print("=" * 72)
print("🧠 RUNNING BACKTEST")
print("=" * 72)


all_trades = []


for symbol, df in datasets.items():

    print(
        f"\n🔸 Testing {symbol}..."
    )

    trades = backtest_symbol(
        symbol,
        df
    )

    print(
        f"   Trades: {len(trades)}"
    )

    all_trades.extend(
        trades
    )


# ============================================================
# IF NOTHING
# ============================================================

if not all_trades:

    print(
        "\n⚠️ NO TRADES FOUND."
    )

    print(
        "Try MIN_SCORE = 7"
    )

    sys.exit(0)


# ============================================================
# PORTFOLIO OVERLAP LOCK
# ============================================================

trades_df = pd.DataFrame(
    all_trades
)

trades_df = trades_df.sort_values(
    "EntryTime"
).reset_index(
    drop=True
)


# ============================================================
# TRUE PORTFOLIO LOCK
#
# Only ONE open position across ALL symbols.
# ============================================================

accepted_trades = []

portfolio_free_time = None


for _, trade in trades_df.iterrows():

    entry_time = (
        trade["EntryTime"]
    )

    exit_time = (
        trade["ExitTime"]
    )

    if (
        portfolio_free_time is None
        or
        entry_time >
        portfolio_free_time
    ):

        accepted_trades.append(
            trade
        )

        portfolio_free_time = (
            exit_time
        )


portfolio = pd.DataFrame(
    accepted_trades
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 72)
print("📊 FINAL PORTFOLIO REPORT")
print("=" * 72)


if portfolio.empty:

    print(
        "⚠️ No trades after portfolio overlap lock."
    )

    sys.exit(0)


total = len(
    portfolio
)


wins = (
    portfolio["Outcome"]
    == "WIN"
).sum()


losses = (
    portfolio["Outcome"]
    == "LOSS"
).sum()


timeout_profit = (
    portfolio["Outcome"]
    ==
    "TIMEOUT_PROFIT"
).sum()


timeout_loss = (
    portfolio["Outcome"]
    ==
    "TIMEOUT_LOSS"
).sum()


timeout_be = (
    portfolio["Outcome"]
    ==
    "TIMEOUT_BE"
).sum()


# ============================================================
# STRICT TP/SL WIN RATE
# ============================================================

decisive = (
    wins +
    losses
)


win_rate = (

    wins /
    decisive *
    100

    if decisive > 0

    else 0
)


# ============================================================
# POSITIVE TRADE RATE
# ============================================================

positive = (
    portfolio["NetPct"] > 0
).sum()


positive_rate = (

    positive /
    total *
    100

    if total > 0

    else 0
)


# ============================================================
# PROFIT
# ============================================================

gross_profit = (
    portfolio[
        portfolio["NetPct"] > 0
    ]["NetPct"].sum()
)


gross_loss = abs(
    portfolio[
        portfolio["NetPct"] < 0
    ]["NetPct"].sum()
)


profit_factor = (

    gross_profit /
    gross_loss

    if gross_loss > 0

    else np.inf
)


net_return = (
    portfolio["NetPct"].sum()
)


average_trade = (
    portfolio["NetPct"].mean()
)


# ============================================================
# MAX DRAWDOWN
# ============================================================

equity = (
    portfolio["NetPct"]
    .cumsum()
)

peak = (
    equity
    .cummax()
)

drawdown = (
    equity -
    peak
)

max_drawdown = (
    drawdown.min()
)


# ============================================================
# LOSING STREAK
# ============================================================

max_losing_streak = 0

current_losing_streak = 0


for outcome in portfolio[
    "Outcome"
]:

    if outcome in [
        "LOSS",
        "TIMEOUT_LOSS"
    ]:

        current_losing_streak += 1

        max_losing_streak = max(
            max_losing_streak,
            current_losing_streak
        )

    else:

        current_losing_streak = 0


# ============================================================
# TRADES / DAY
# ============================================================

first_trade = (
    portfolio["EntryTime"].min()
)

last_trade = (
    portfolio["EntryTime"].max()
)

days = (
    last_trade -
    first_trade
).total_seconds() / 86400


trades_per_day = (

    total /
    days

    if days > 0

    else 0
)


trades_per_year = (
    trades_per_day *
    365
)


# ============================================================
# PRINT
# ============================================================

print(
    f"\n🔸 Total Trades: {total}"
)

print(
    f"🟢 WIN: {wins}"
)

print(
    f"🔴 LOSS: {losses}"
)

print(
    f"🟡 TIMEOUT PROFIT: "
    f"{timeout_profit}"
)

print(
    f"🟠 TIMEOUT LOSS: "
    f"{timeout_loss}"
)

print(
    f"⚪ TIMEOUT BE: "
    f"{timeout_be}"
)

print(
    f"\n🎯 WIN RATE (TP/SL): "
    f"{win_rate:.2f}%"
)

print(
    f"📈 Positive Trade Rate: "
    f"{positive_rate:.2f}%"
)

print(
    f"\n💰 NET RETURN: "
    f"{net_return:.2f}%"
)

print(
    f"📊 Average Trade: "
    f"{average_trade:.4f}%"
)

print(
    f"📊 Profit Factor: "
    f"{profit_factor:.2f}"
)

print(
    f"📉 Max Drawdown: "
    f"{max_drawdown:.2f}%"
)

print(
    f"🔥 Max Losing Streak: "
    f"{max_losing_streak}"
)

print(
    f"📅 Trades / Day: "
    f"{trades_per_day:.2f}"
)

print(
    f"📅 Trades / Year: "
    f"{trades_per_year:.0f}"
)


# ============================================================
# SYMBOL PERFORMANCE
# ============================================================

print("\n")
print("=" * 72)
print("📊 PERFORMANCE BY SYMBOL")
print("=" * 72)


symbol_summary = (
    portfolio
    .groupby("Symbol")
    .agg(

        Trades=(
            "Outcome",
            "size"
        ),

        Wins=(
            "Outcome",
            lambda x:
            (x == "WIN").sum()
        ),

        Losses=(
            "Outcome",
            lambda x:
            (x == "LOSS").sum()
        ),

        NetPct=(
            "NetPct",
            "sum"
        ),

        AverageTrade=(
            "NetPct",
            "mean"
        )
    )
)


symbol_summary["WinRate"] = np.where(

    (
        symbol_summary["Wins"] +
        symbol_summary["Losses"]
    ) > 0,

    symbol_summary["Wins"] /
    (
        symbol_summary["Wins"] +
        symbol_summary["Losses"]
    ) *
    100,

    0
)


print(
    symbol_summary.round(
        3
    ).to_string()
)


# ============================================================
# LONG / SHORT
# ============================================================

print("\n")
print("=" * 72)
print("📊 LONG / SHORT")
print("=" * 72)


side_summary = (
    portfolio
    .groupby("Side")
    .agg(

        Trades=(
            "Outcome",
            "size"
        ),

        Wins=(
            "Outcome",
            lambda x:
            (x == "WIN").sum()
        ),

        Losses=(
            "Outcome",
            lambda x:
            (x == "LOSS").sum()
        ),

        NetPct=(
            "NetPct",
            "sum"
        )
    )
)


side_summary["WinRate"] = np.where(

    (
        side_summary["Wins"] +
        side_summary["Losses"]
    ) > 0,

    side_summary["Wins"] /
    (
        side_summary["Wins"] +
        side_summary["Losses"]
    ) *
    100,

    0
)


print(
    side_summary.round(
        3
    ).to_string()
)


# ============================================================
# SCORE ANALYSIS
# ============================================================

print("\n")
print("=" * 72)
print("⭐ SCORE ANALYSIS")
print("=" * 72)


score_summary = (
    portfolio
    .groupby("Score")
    .agg(

        Trades=(
            "Outcome",
            "size"
        ),

        Wins=(
            "Outcome",
            lambda x:
            (x == "WIN").sum()
        ),

        Losses=(
            "Outcome",
            lambda x:
            (x == "LOSS").sum()
        ),

        NetPct=(
            "NetPct",
            "sum"
        )
    )
)


score_summary["WinRate"] = np.where(

    (
        score_summary["Wins"] +
        score_summary["Losses"]
    ) > 0,

    score_summary["Wins"] /
    (
        score_summary["Wins"] +
        score_summary["Losses"]
    ) *
    100,

    0
)


print(
    score_summary.round(
        3
    ).to_string()
)


# ============================================================
# SAVE TRADE LOG
# ============================================================

output_file = (
    "hunter_1pct_v1_lbank_trades.csv"
)


portfolio.to_csv(
    output_file,
    index=False
)


print("\n")
print("=" * 72)
print(
    f"💾 Trade log saved: "
    f"{output_file}"
)
print("=" * 72)


# ============================================================
# ANTI LOOK-AHEAD AUDIT
# ============================================================

print("\n")
print("=" * 72)
print("🔒 ANTI-LOOKAHEAD AUDIT")
print("=" * 72)

print(
    "✔️ Current unfinished 1H candle excluded"
)

print(
    "✔️ 4H uses ONLY previous CLOSED 4H candle"
)

print(
    "✔️ Breakout level = previous 20 candles"
)

print(
    "✔️ Breakout uses shift(1)"
)

print(
    "✔️ Retest starts AFTER breakout"
)

print(
    "✔️ Entry = NEXT 1H candle OPEN"
)

print(
    "✔️ TP/SL checked AFTER entry only"
)

print(
    "✔️ Same-candle TP + SL = LOSS"
)

print(
    "✔️ No overlapping portfolio positions"
)

print(
    "✔️ Cooldown after completed trade"
)

print(
    "✔️ No future candles used for signals"
)

print("\n✨ BACKTEST COMPLETE.")
