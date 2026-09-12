# ============================================================
# HUNTER-X V2
# TREND PULLBACK + MOMENTUM
# CLEAN LBank 1 YEAR BACKTEST
# ============================================================
#
# TIMEFRAMES:
#   4H = REGIME
#   1H = ENTRY
#
# TP = 2%
# SL = 1%
# RR = 1:2
#
# NO LOOKAHEAD
# NO SAME-CANDLE ENTRY
# NO TRAILING
# NO BREAK EVEN
# NO OVERLAPPING PORTFOLIO TRADES
#
# ============================================================

import os
import sys
import subprocess
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ============================================================
# DEPENDENCIES
# ============================================================

try:
    import ccxt
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "ccxt"
    ])
    import ccxt

try:
    import pandas as pd
    import numpy as np
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "pandas",
        "numpy"
    ])
    import pandas as pd
    import numpy as np


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
    "DOT": "DOT/USDT"
}

TIMEFRAME = "1h"

TEST_DAYS = 365
WARMUP_DAYS = 120


# ============================================================
# TARGET
# ============================================================

TP_PCT = 0.0200       # +2%
SL_PCT = 0.0100       # -1%

RR = TP_PCT / SL_PCT

# Maximum holding time = 24 x 1H
MAX_HOLD_BARS = 24


# ============================================================
# COSTS
# ============================================================

TAKER_FEE = 0.0006

# Slippage applied to execution prices
SLIPPAGE = 0.0002


# ============================================================
# LBANK
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True
})

exchange.load_markets()


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

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
        adjust=False,
        min_periods=period
    ).mean()


def adx(df, period=14):

    high = df["High"]
    low = df["Low"]

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

    atr_value = atr(
        df,
        period
    )

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()
        / atr_value
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()
        / atr_value
    )

    denominator = (
        plus_di + minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# DOWNLOAD LBANK
# ============================================================

def download_lbank(symbol, days):

    print(
        f"\n📥 Downloading {symbol}..."
    )

    end_time = datetime.utcnow()

    start_time = (
        end_time -
        timedelta(days=days)
    )

    since = int(
        start_time.timestamp() * 1000
    )

    all_ohlcv = []

    while True:

        try:

            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe="1h",
                since=since,
                limit=1000
            )

        except Exception as e:

            print(
                f"\n❌ {symbol} download error: {e}"
            )

            break

        if not ohlcv:
            break

        all_ohlcv.extend(
            ohlcv
        )

        print(
            f"\r   Candles: {len(all_ohlcv)}",
            end=""
        )

        last_timestamp = ohlcv[-1][0]

        new_since = (
            last_timestamp + 1
        )

        if new_since <= since:
            break

        since = new_since

        if len(ohlcv) < 1000:
            break

    print()

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
    )

    df = df.drop_duplicates(
        subset=["Timestamp"]
    )

    df = df.sort_values(
        "Timestamp"
    )

    df = df.set_index(
        "Timestamp"
    )

    # --------------------------------------------------------
    # Remove current unfinished candle
    # --------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    current_hour = now.floor("h")

    df = df[
        df.index < current_hour
    ]

    return df


# ============================================================
# PREPARE 4H
# ============================================================

def prepare_4h(df):

    df4 = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    df4 = df4.dropna()

    df4["EMA20"] = ema(
        df4["Close"],
        20
    )

    df4["EMA50"] = ema(
        df4["Close"],
        50
    )

    df4["EMA200"] = ema(
        df4["Close"],
        200
    )

    df4["RSI"] = rsi(
        df4["Close"],
        14
    )

    df4["ADX"] = adx(
        df4,
        14
    )

    df4["ATR"] = atr(
        df4,
        14
    )

    return df4


# ============================================================
# PREPARE 1H
# ============================================================

def prepare_1h(df):

    df = df.copy()

    df["EMA20"] = ema(
        df["Close"],
        20
    )

    df["EMA50"] = ema(
        df["Close"],
        50
    )

    df["RSI"] = rsi(
        df["Close"],
        14
    )

    df["ATR"] = atr(
        df,
        14
    )

    df["ADX"] = adx(
        df,
        14
    )

    df["VOL_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # Previous candle values
    df["PREV_HIGH"] = (
        df["High"].shift(1)
    )

    df["PREV_LOW"] = (
        df["Low"].shift(1)
    )

    df["PREV_CLOSE"] = (
        df["Close"].shift(1)
    )

    # Candle structure
    df["RANGE"] = (
        df["High"] -
        df["Low"]
    )

    df["BODY"] = (
        df["Close"] -
        df["Open"]
    ).abs()

    df["BODY_RATIO"] = (
        df["BODY"] /
        df["RANGE"].replace(
            0,
            np.nan
        )
    )

    # ATR percentage
    df["ATR_PCT"] = (
        df["ATR"] /
        df["Close"]
    )

    return df


# ============================================================
# ATTACH PREVIOUS CLOSED 4H
# ============================================================

def attach_4h(df1, df4):

    htf = df4.copy()

    # --------------------------------------------------------
    # CRITICAL:
    # Shift one completed 4H candle backward.
    #
    # Therefore the current 1H candle can NEVER see
    # the currently forming 4H candle.
    # --------------------------------------------------------

    htf = htf.shift(1)

    htf = htf.rename(
        columns={
            "Close": "HTF_CLOSE",
            "EMA20": "HTF_EMA20",
            "EMA50": "HTF_EMA50",
            "EMA200": "HTF_EMA200",
            "RSI": "HTF_RSI",
            "ADX": "HTF_ADX",
            "ATR": "HTF_ATR"
        }
    )

    htf = htf[
        [
            "HTF_CLOSE",
            "HTF_EMA20",
            "HTF_EMA50",
            "HTF_EMA200",
            "HTF_RSI",
            "HTF_ADX",
            "HTF_ATR"
        ]
    ]

    df = df1.copy()

    df["HTF_BUCKET"] = (
        df.index.floor("4h")
    )

    htf.index.name = (
        "HTF_BUCKET"
    )

    df = df.join(
        htf,
        on="HTF_BUCKET"
    )

    return df


# ============================================================
# SIGNAL ENGINE
# ============================================================

def get_signal(df, i):

    row = df.iloc[i]

    needed = [
        "HTF_CLOSE",
        "HTF_EMA20",
        "HTF_EMA50",
        "HTF_EMA200",
        "HTF_RSI",
        "HTF_ADX",
        "EMA20",
        "EMA50",
        "RSI",
        "ADX",
        "ATR",
        "VOL_MA20",
        "BODY_RATIO",
        "ATR_PCT"
    ]

    for col in needed:

        if pd.isna(row[col]):

            return None

    close = float(
        row["Close"]
    )

    open_price = float(
        row["Open"]
    )

    high = float(
        row["High"]
    )

    low = float(
        row["Low"]
    )

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    rsi_value = float(
        row["RSI"]
    )

    atr_pct = float(
        row["ATR_PCT"]
    )

    # ========================================================
    # VOLATILITY FILTER
    # ========================================================
    #
    # Avoid extremely dead markets
    # and extremely chaotic candles.
    #
    # ========================================================

    volatility_ok = (
        atr_pct >= 0.0025
        and
        atr_pct <= 0.0200
    )

    if not volatility_ok:

        return None


    # ========================================================
    # 4H TREND
    # ========================================================

    long_trend = (
        row["HTF_EMA20"] >
        row["HTF_EMA50"]
        and
        row["HTF_EMA50"] >
        row["HTF_EMA200"]
        and
        row["HTF_CLOSE"] >
        row["HTF_EMA20"]
        and
        row["HTF_RSI"] >= 50
        and
        row["HTF_ADX"] >= 15
    )

    short_trend = (
        row["HTF_EMA20"] <
        row["HTF_EMA50"]
        and
        row["HTF_EMA50"] <
        row["HTF_EMA200"]
        and
        row["HTF_CLOSE"] <
        row["HTF_EMA20"]
        and
        row["HTF_RSI"] <= 50
        and
        row["HTF_ADX"] >= 15
    )


    # ========================================================
    # 1H MOMENTUM
    # ========================================================

    long_momentum = (
        ema20 > ema50
        and
        rsi_value >= 50
        and
        float(row["ADX"]) >= 15
    )

    short_momentum = (
        ema20 < ema50
        and
        rsi_value <= 50
        and
        float(row["ADX"]) >= 15
    )


    # ========================================================
    # VOLUME
    # ========================================================

    volume_ok = (
        row["Volume"] >=
        row["VOL_MA20"] * 0.80
    )


    # ========================================================
    # CANDLE QUALITY
    # ========================================================

    body_ok = (
        row["BODY_RATIO"] >= 0.35
    )


    # ========================================================
    # LONG SETUP
    # ========================================================
    #
    # SETUP A:
    # Pullback -> EMA20/EMA50 -> reclaim
    #
    # SETUP B:
    # Momentum continuation
    #
    # ========================================================

    long_pullback = (
        low <= ema20 * 1.003
        and
        close > ema20
        and
        close > open_price
        and
        rsi_value >= 48
        and
        body_ok
    )

    long_momentum_continuation = (
        close > float(row["PREV_HIGH"])
        and
        close > ema20
        and
        close > open_price
        and
        rsi_value >= 52
        and
        body_ok
    )

    if (
        long_trend
        and
        long_momentum
        and
        volume_ok
        and
        (
            long_pullback
            or
            long_momentum_continuation
        )
    ):

        if long_pullback:

            setup = (
                "LONG_PULLBACK"
            )

        else:

            setup = (
                "LONG_MOMENTUM"
            )

        return setup


    # ========================================================
    # SHORT SETUP
    # ========================================================

    short_pullback = (
        high >= ema20 * 0.997
        and
        close < ema20
        and
        close < open_price
        and
        rsi_value <= 52
        and
        body_ok
    )

    short_momentum_continuation = (
        close < float(row["PREV_LOW"])
        and
        close < ema20
        and
        close < open_price
        and
        rsi_value <= 48
        and
        body_ok
    )

    if (
        short_trend
        and
        short_momentum
        and
        volume_ok
        and
        (
            short_pullback
            or
            short_momentum_continuation
        )
    ):

        if short_pullback:

            setup = (
                "SHORT_PULLBACK"
            )

        else:

            setup = (
                "SHORT_MOMENTUM"
            )

        return setup


    return None


# =================================================
