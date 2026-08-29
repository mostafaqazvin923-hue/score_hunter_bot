# ============================================================
# SCORE HUNTER PRO v9
# ============================================================
# NEW STRATEGY - 1H REGIME + 15M PRECISION ENTRY
#
# NO PANDAS
# NO NUMPY
# CLOSED CANDLE ONLY
#
# 1H:
#   EMA 20 / 50 / 200
#   ADX
#
# 15M:
#   Pullback
#   Breakout
#   RSI
#   ADX
#   ATR
#   Volume
#   Candle confirmation
#
# Risk:
#   ATR based SL
#   TP = configurable R multiple
#
# Backtest:
#   In-sample + OOS
#   Signals/day
#   Win rate
#   Profit factor
#   Net R
#   Max drawdown
#
# Binance Futures public market data
# ============================================================

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

COINS = [
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BTCUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "DOGEUSDT",
]

ENTRY_INTERVAL = "15m"
TREND_INTERVAL = "1h"

# Binance allows large candle batches.
# We download enough data for the requested period.

HISTORY_DAYS = 365
OOS_DAYS = 90

# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------

EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

ADX_MIN = 20.0

# ------------------------------------------------------------
# RSI zones
# ------------------------------------------------------------

LONG_RSI_MIN = 52.0
LONG_RSI_MAX = 70.0

SHORT_RSI_MIN = 30.0
SHORT_RSI_MAX = 48.0

# ------------------------------------------------------------
# Pullback
# ------------------------------------------------------------

PULLBACK_LOOKBACK = 8

# Maximum distance from EMA20 in ATR units.
MAX_EMA_DISTANCE_ATR = 1.25

# Candle is considered a valid touch of EMA20
EMA_TOUCH_ATR = 0.35

# ------------------------------------------------------------
# Breakout
# ------------------------------------------------------------

BREAKOUT_LOOKBACK = 12

# ------------------------------------------------------------
# Volume
# ------------------------------------------------------------

VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.05

# ------------------------------------------------------------
# Risk
# ------------------------------------------------------------

SL_ATR_MULTIPLIER = 1.20

# Main target.
#
# 1.5R is deliberately used here because the requested
# 70-80% win-rate target is easier to evaluate realistically
# than forcing 2R and then starving the strategy.
#
# Change to 2.0 after testing if desired.
#
TP_R_MULTIPLIER = 1.50

# ------------------------------------------------------------
# Trade frequency controls
# ------------------------------------------------------------

COOLDOWN_BARS = 4

# Maximum number of trades per coin per day.
MAX_TRADES_PER_COIN_DAY = 1

# ------------------------------------------------------------
# Fees/slippage
# ------------------------------------------------------------

FEE_PER_SIDE_RISK = 0.0004
SLIPPAGE = 0.0002

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

RESULT_FILE = "backtest_v9_results.json"


# ============================================================
# HTTP
# ============================================================

def get_json(url, params):

    query = urllib.parse.urlencode(params)

    full_url = url + "?" + query

    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "SCORE-HUNTER-PRO-V9"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        raw = response.read()

    return json.loads(
        raw.decode("utf-8")
    )


# ============================================================
# BINANCE KLINES
# ============================================================

def interval_ms(interval):

    if interval == "15m":
        return 15 * 60 * 1000

    if interval == "1h":
        return 60 * 60 * 1000

    raise ValueError(
        f"Unsupported interval: {interval}"
    )


def get_klines(
    symbol,
    interval,
    start_ms,
    end_ms
):

    candles = []

    step = interval_ms(interval)

    current = start_ms

    while current < end_ms:

        data = get_json(
            BASE_URL,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            }
        )

        if not data:
            break

        for row in data:

            candles.append({
                "time": int(row[0]),

                "open": float(row[1]),

                "high": float(row[2]),

                "low": float(row[3]),

                "close": float(row[4]),

                "volume": float(row[5]),
            })

        last_open = int(
            data[-1][0]
        )

        next_time = (
            last_open
            + step
        )

        if next_time <= current:
            break

        current = next_time

        time.sleep(0.10)

        if len(data) < 1000:
            break

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = {}

    for candle in candles:

        unique[
            candle["time"]
        ] = candle

    candles = [
        unique[t]
        for t in sorted(unique)
    ]

    # --------------------------------------------------------
    # Remove currently forming candle
    # --------------------------------------------------------

    now_ms = int(
        datetime.now(
            timezone.utc
        ).timestamp()
        * 1000
    )

    candles = [
        c
        for c in candles
        if (
            c["time"]
            + step
            <= now_ms
        )
    ]

    return candles


# ============================================================
# EMA
# ============================================================

def ema(values, period):

    if len(values) < period:
        return None

    value = (
        sum(
            values[:period]
        )
        / period
    )

    alpha = (
        2.0
        / (period + 1)
    )

    for value_now in values[period:]:

        value = (
            value
            + alpha
            * (
                value_now
                - value
            )
        )

    return value


# ============================================================
# EMA SERIES
# ============================================================

def ema_series(values, period):

    result = [
        None
    ] * len(values)

    if len(values) < period:
        return result

    value = (
        sum(
            values[:period]
        )
        / period
    )

    result[
        period - 1
    ] = value

    alpha = (
        2.0
        / (period + 1)
    )

    for i in range(
        period,
        len(values)
    ):

        value = (
            value
            + alpha
            * (
                values[i]
                - value
            )
        )

        result[i] = value

    return result


# ============================================================
# ATR SERIES
# ============================================================

def atr_series(
    candles,
    period
):

    result = [
        None
    ] * len(candles)

    if len(candles) <= period:
        return result

    tr = [
        None
    ]

    for i in range(
        1,
        len(candles)
    ):

        current = candles[i]

        previous = candles[
            i - 1
        ]

        true_range = max(
            current["high"]
            - current["low"],

            abs(
                current["high"]
                - previous["close"]
            ),

            abs(
                current["low"]
                - previous["close"]
            )
        )

        tr.append(
            true_range
        )

    atr_value = (
        sum(
            tr[1:period + 1]
        )
        / period
    )

    result[
        period
    ] = atr_value

    for i in range(
        period + 1,
        len(candles)
    ):

        atr_value = (
            (
                atr_value
                * (period - 1)
            )
            + tr[i]
        ) / period

        result[i] = atr_value

    return result


# ============================================================
# RSI SERIES
# ============================================================

def rsi_series(
    candles,
    period
):

    result = [
        None
    ] * len(candles)

    if len(candles) <= period:
        return result

    closes = [
        c["close"]
        for c in candles
    ]

    gains = []
    losses = []

    for i in range(
        1,
        len(closes)
    ):

        change = (
            closes[i]
            - closes[i - 1]
        )

        gains.append(
            max(
                change,
                0.0
            )
        )

        losses.append(
            max(
                -change,
                0.0
            )
        )

    avg_gain = (
        sum(
            gains[:period]
        )
        / period
    )

    avg_loss = (
        sum(
            losses[:period]
        )
        / period
    )

    if avg_loss == 0:

        result[
            period
        ] = 100.0

    else:

        rs = (
            avg_gain
            / avg_loss
        )

        result[
            period
        ] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

    for i in range(
        period,
        len(gains)
    ):

        avg_gain = (
            (
                avg_gain
                * (period - 1)
                + gains[i]
            )
            / period
        )

        avg_loss = (
            (
                avg_loss
                * (period - 1)
                + losses[i]
            )
            / period
        )

        if avg_loss == 0:

            result[
                i + 1
            ] = 100.0

        else:

            rs = (
                avg_gain
                / avg_loss
            )

            result[
                i + 1
            ] = (
                100
                - (
                    100
                    / (1 + rs)
                )
            )

    return result


# ============================================================
# ADX SERIES
# ============================================================

def adx_series(
    candles,
    period
):

    n = len(candles)

    result = [
        None
    ] * n

    if n < (
        period * 2
        + 5
    ):
        return result

    tr = [
        None
    ]

    plus_dm = [
        None
    ]

    minus_dm = [
        None
    ]

    for i in range(
        1,
        n
    ):

        current = candles[i]

        previous = candles[
            i - 1
        ]

        true_range = max(
            current["high"]
            - current["low"],

            abs(
                current["high"]
                - previous["close"]
            ),

            abs(
                current["low"]
                - previous["close"]
            )
        )

        up_move = (
            current["high"]
            - previous["high"]
        )

        down_move = (
            previous["low"]
            - current["low"]
        )

        plus = (
            up_move
            if (
                up_move
                > down_move
                and up_move
                > 0
            )
            else 0.0
        )

        minus = (
            down_move
            if (
                down_move
                > up_move
                and down_move
                > 0
            )
            else 0.0
        )

        tr.append(
            true_range
        )

        plus_dm.append(
            plus
        )

        minus_dm.append(
            minus
        )

    atr_value = (
        sum(
            tr[1:period + 1]
        )
        / period
    )

    plus_value = (
        sum(
            plus_dm[1:period + 1]
        )
        / period
    )

    minus_value = (
        sum(
            minus_dm[1:period + 1]
        )
        / period
    )

    dx = [
        None
    ] * n

    for i in range(
        period + 1,
        n
    ):

        atr_value = (
            (
                atr_value
                * (period - 1)
                + tr[i]
            )
            / period
        )

        plus_value = (
            (
                plus_value
                * (period - 1)
                + plus_dm[i]
            )
            / period
        )

        minus_value = (
            (
                minus_value
                * (period - 1)
                + minus_dm[i]
            )
            / period
        )

        if atr_value == 0:
            continue

        plus_di = (
            100
            * plus_value
            / atr_value
        )

        minus_di = (
            100
            * minus_value
            / atr_value
        )

        denominator = (
            plus_di
            + minus_di
        )

        if denominator == 0:
            dx[i] = 0.0
        else:
            dx[i] = (
                100
                * abs(
                    plus_di
                    - minus_di
                )
                / denominator
            )

    dx_values = []

    first_index = None

    for i in range(n):

        if dx[i] is not None:

            dx_values.append(
                dx[i]
            )

            if len(
                dx_values
            ) == period:

                first_index = i
                break

    if first_index is None:
        return result

    adx_value = (
        sum(dx_values)
        / period
    )

    result[
        first_index
    ] = adx_value

    for i in range(
        first_index + 1,
        n
    ):

        if dx[i] is None:
            continue

        adx_value = (
            (
                adx_value
                * (period - 1)
                + dx[i]
            )
            / period
        )

        result[i] = adx_value

    return result


# ============================================================
# AGGREGATE 15M -> 1H
# ============================================================

def aggregate_1h(
    candles_15m
):

    buckets = {}

    for candle in candles_15m:

        timestamp = candle[
            "time"
        ]

        bucket = (
            timestamp
            - (
                timestamp
                % (
                    60
                    * 60
                    * 1000
                )
            )
        )

        buckets.setdefault(
            bucket,
            []
        ).append(
            candle
        )

    result = []

    for timestamp in sorted(
        buckets
    ):

        group = buckets[
            timestamp
        ]

        # Require exactly 4 x 15m candles.
        if len(group) != 4:
            continue

        result.append({

            "time": timestamp,

            "open":
                group[0]["open"],

            "high":
                max(
                    c["high"]
                    for c in group
                ),

            "low":
                min(
                    c["low"]
                    for c in group
                ),

            "close":
                group[-1]["close"],

            "volume":
                sum(
                    c["volume"]
                    for c in group
                ),
        })

    return result


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def candle_confirmation(
    candle,
    direction
):

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    body = abs(
        candle["close"]
        - candle["open"]
    )

    body_ratio = (
        body
        / candle_range
    )

    if body_ratio < BODY_MIN:
        return False

    if direction == "LONG":

        close_location = (
            candle["close"]
            - candle["low"]
        ) / candle_range

        return (
            candle["close"]
            > candle["open"]
            and close_location
            >= 0.65
        )

    if direction == "SHORT":

        close_location = (
            candle["high"]
            - candle["close"]
        ) / candle_range

        return (
            candle["close"]
            < candle["open"]
            and close_location
            >= 0.65
        )

    return False


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

def volume_confirmation(
    candles,
    index
):

    if index < VOLUME_LOOKBACK:
        return False

    average_volume = (
        sum(
            candles[i]["volume"]
            for i in range(
                index
                - VOLUME_LOOKBACK,
                index
            )
        )
        / VOLUME_LOOKBACK
    )

    if average_volume <= 0:
        return False

    return (
        candles[index]["volume"]
        >= (
            average_volume
            * VOLUME_MULTIPLIER
        )
    )


# ============================================================
# 1H TREND
# ============================================================

def get_1h_trend(
    candles_1h
):

    if len(candles_1h) < (
        EMA_TREND
        + 5
    ):
        return None

    closes = [
        c["close"]
        for c in candles_1h
    ]

    ema20_value = ema(
        closes,
        EMA_FAST
    )

    ema50_value = ema(
        closes,
        EMA_SLOW
    )

    ema200_value = ema(
        closes,
        EMA_TREND
    )

    close = closes[-1]

    if (
        close
        > ema200_value
        and ema20_value
        > ema50_value
        and ema50_value
        > ema200_value
    ):

        return "LONG"

    if (
        close
        < ema200_value
        and ema20_value
        < ema50_value
        and ema50_value
        < ema200_value
    ):

        return "SHORT"

    return None


# ============================================================
# TREND SNAPSHOT
# ============================================================

def build_1h_trend_series(
    candles_1h
):

    closes = [
        c["close"]
        for c in candles_1h
    ]

    ema20_values = ema_series(
        closes,
        EMA_FAST
    )

    ema50_values = ema_series(
        closes,
        EMA_SLOW
    )

    ema200_values = ema_series(
        closes,
        EMA_TREND
    )

    trends = [
        None
    ] * len(candles_1h)

    for i in range(
        len(candles_1h)
    ):

        e20 = ema20_values[i]
        e50 = ema50_values[i]
        e200 = ema200_values[i]

        if (
            e20 is None
            or e50 is None
            or e200 is None
        ):
            continue

        close = candles_1h[
            i
        ]["close"]

        if (
            close > e200
            and e20 > e50 > e200
        ):

            trends[i] = "LONG"

        elif (
            close < e200
            and e20 < e50 < e200
        ):

            trends[i] = "SHORT"

    return trends


# ============================================================
# FIND 1H CANDLE AVAILABLE AT 15M CLOSE
# ============================================================

def latest_closed_1h_index(
    candles_1h,
    timestamp
):

    completed = (
        timestamp
        - (
            timestamp
            % (
                60
                * 60
                * 1000
            )
        )
        - (
            60
            * 60
            * 1000
        )
    )

    low = 0
    high = len(
        candles_1h
    ) - 1

    found = None

    while low <= high:

        middle = (
            low + high
        ) // 2

        t = candles_1h[
            middle
        ]["time"]

        if t == completed:

            found = middle
            break

        if t < completed:
            low = middle + 1
        else:
            high = middle - 1

    return found


# ============================================================
# ENTRY SIGNAL
# ============================================================

def find_signal(
    candles,
    index,
    trend
):

    if trend is None:
        return None

    minimum = max(
        EMA_TREND + 10,
        BREAKOUT_LOOKBACK + 5,
        PULLBACK_LOOKBACK + 5,
        VOLUME_LOOKBACK + 5,
        50
    )

    if index < minimum:
        return None

    closes = [
        c["close"]
        for c in candles[:index + 1]
    ]

    ema20_value = ema(
        closes,
        EMA_FAST
    )

    ema50_value = ema(
        closes,
        EMA_SLOW
    )

    ema200_value = ema(
        closes,
        EMA_TREND
    )

    atr_values = atr_series(
        candles[:index + 1],
        ATR_PERIOD
    )

    rsi_values = rsi_series(
        candles[:index + 1],
        RSI_PERIOD
    )

    adx_values = adx_series(
        candles[:index + 1],
        ADX_PERIOD
    )

    atr_value = atr_values[-1]
    rsi_value = rsi_values[-1]
    adx_value = adx_values[-1]

    if (
        atr_value is None
        or rsi_value is None
        or adx_value is None
    ):
        return None

    current = candles[
        index
    ]

    entry = current[
        "close"
    ]

    # ========================================================
    # LONG
    # ========================================================

    if trend == "LONG":

        ema_alignment = (
            entry
            > ema20_value
            > ema50_value
            > ema200_value
        )

        if not ema_alignment:
            return None

        if not (
            LONG_RSI_MIN
            <= rsi_value
            <= LONG_RSI_MAX
        ):
            return None

        if adx_value < ADX_MIN:
            return None

        # ----------------------------------------------------
        # Pullback
        # ----------------------------------------------------

        pullback = False

        for j in range(
            index
            - PULLBACK_LOOKBACK,
            index
        ):

            previous = candles[
                j
            ]

            distance = abs(
                previous["low"]
                - ema20_value
            )

            if (
                previous["low"]
                <= ema20_value
                + atr_value
                * EMA_TOUCH_ATR
                and distance
                <= atr_value
                * 1.50
            ):

                pullback = True
                break

        # ----------------------------------------------------
        # Breakout
        # ----------------------------------------------------

        resistance = max(
            c["high"]
            for c in candles[
                index
                - BREAKOUT_LOOKBACK:
                index
            ]
        )

        breakout = (
            entry
            > resistance
        )

        # ----------------------------------------------------
        # Pullback reclaim
        # ----------------------------------------------------

        reclaim = (
            pullback
            and entry
            > ema20_value
            and current["close"]
            > current["open"]
        )

        valid_setup = (
            breakout
            or reclaim
        )

        if not valid_setup:
            return None

        if not candle_confirmation(
            current,
            "LONG"
        ):
            return None

        if not volume_confirmation(
            candles,
            index
        ):
            return None

        # Don't chase an extreme candle.
        if (
            entry
            - ema20_value
            > atr_value
            * MAX_EMA_DISTANCE_ATR
        ):
            return None

        recent_low = min(
            c["low"]
            for c in candles[
                index - 5:
                index + 1
            ]
        )

        sl = min(
            entry
            - atr_value
            * SL_ATR_MULTIPLIER,

            recent_low
            - atr_value
            * 0.10
        )

        risk = (
            entry
            - sl
        )

        if risk <= 0:
            return None

        tp = (
            entry
            + risk
            * TP_R_MULTIPLIER
        )

        return {
            "direction": "LONG",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "rsi": rsi_value,
            "adx": adx_value,
            "setup": (
                "BREAKOUT"
                if breakout
                else "PULLBACK"
            )
        }

    # ========================================================
    # SHORT
    # ========================================================

    if trend == "SHORT":

        ema_alignment = (
            entry
            < ema20_value
            < ema50_value
            < ema200_value
        )

        if not ema_alignment:
            return None

        if not (
            SHORT_RSI_MIN
            <= rsi_value
            <= SHORT_RSI_MAX
        ):
            return None

        if adx_value < ADX_MIN:
            return None

        # ----------------------------------------------------
        # Pullback
        # ----------------------------------------------------

        pullback = False

        for j in range(
            index
            - PULLBACK_LOOKBACK,
            index
        ):

            previous = candles[
                j
            ]

            distance = abs(
                previous["high"]
                - ema20_value
            )

            if (
                previous["high"]
                >= ema20_value
                - atr_value
                * EMA_TOUCH_ATR
                and distance
                <= atr_value
                * 1.50
            ):

                pullback = True
                break

        # ----------------------------------------------------
        # Breakout
        # ----------------------------------------------------

        support = min(
            c["low"]
            for c in candles[
                index
                - BREAKOUT_LOOKBACK:
                index
            ]
        )

        breakout = (
            entry
            < support
        )

        reclaim = (
            pullback
            and entry
            < ema20_value
            and current["close"]
            < current["open"]
        )

        valid_setup = (
            breakout
            or reclaim
        )

        if not valid_setup:
            return None

        if not candle_confirmation(
            current,
            "SHORT"
        ):
            return None

        if not volume_confirmation(
            candles,
            index
        ):
            return None

        if (
            ema20_value
            - entry
            > atr_value
            * MAX_EMA_DISTANCE_ATR
        ):
            return None

        recent_high = max(
            c["high"]
            for c in candles[
                index - 5:
                index + 1
            ]
        )

        sl = max(
            entry
            + atr_value
            * SL_ATR_MULTIPLIER,

            recent_high
            + atr_value
            * 0.10
        )

        risk = (
            sl
            - entry
        )

        if risk <= 0:
            return None

        tp = (
            entry
            - risk
            * TP_R_MULTIPLIER
        )

        return {
            "direction": "SHORT",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "rsi": rsi_value,
            "adx": adx_value,
            "setup": (
                "BREAKOUT"
                if breakout
                else "PULLBACK"
            )
        }

    return None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    candles,
    entry_index,
    signal
):

    direction = signal[
        "direction"
    ]

    entry = signal[
        "entry"
    ]

    sl = signal[
        "sl"
    ]

    tp = signal[
        "tp"
    ]

    risk = signal[
        "risk"
    ]

    # Entry at NEXT 15M OPEN.
    next_index = (
        entry_index + 1
    )

    if next_index >= len(
        candles
    ):
        return None

    entry_price = candles[
        next_index
    ]["open"]

    # Slippage
    if direction == "LONG":

        entry_price *= (
            1.0
            + SLIPPAGE
        )

        # Recalculate levels from
        # actual executable entry.

        sl = (
            entry_price
            - risk
        )

        tp = (
            entry_price
            + risk
            * TP_R_MULTIPLIER
        )

    else:

        entry_price *= (
            1.0
            - SLIPPAGE
        )

        sl = (
            entry_price
            + risk
        )

        tp = (
            entry_price
            - risk
            * TP_R_MULTIPLIER
        )

    for i in range(
        next_index,
        len(candles)
    ):

        candle = candles[
            i
        ]

        if direction == "LONG":

            hit_sl = (
                candle["low"]
                <= sl
            )

            hit_tp = (
                candle["high"]
                >= tp
            )

            # Conservative:
            # if both occur in same candle,
            # SL wins.

            if hit_sl:

                return {
                    "outcome": "SL",
                    "r": -1.0,
                    "entry_time":
                        candles[
                            next_index
                        ]["time"],
                    "exit_time":
                        candle["time"],
                    "entry":
                        entry_price,
                    "exit": sl,
                    "tp": tp,
                    "sl": sl,
                }

            if hit_tp:

                return {
                    "outcome": "TP",
                    "r":
                        TP_R_MULTIPLIER,
                    "entry_time":
                        candles[
                            next_index
                        ]["time"],
                    "exit_time":
                        candle["time"],
                    "entry":
                        entry_price,
                    "exit": tp,
                    "tp": tp,
                    "sl": sl,
                }

        else:

            hit_sl = (
                candle["high"]
                >= sl
            )

            hit_tp = (
                candle["low"]
                <= tp
            )

            if hit_sl:

                return {
                    "outcome": "SL",
                    "r": -1.0,
                    "entry_time":
                        candles[
                            next_index
                        ]["time"],
                    "exit_time":
                        candle["time"],
                    "entry":
                        entry_price,
                    "exit": sl,
                    "tp": tp,
                    "sl": sl,
                }

            if hit_tp:

                return {
                    "outcome": "TP",
                    "r":
                        TP_R_MULTIPLIER,
                    "entry_time":
                        candles[
                            next_index
                        ]["time"],
                    "exit_time":
                        candle["time"],
                    "entry":
                        entry_price,
                    "exit": tp,
                    "tp": tp,
                    "sl": sl,
                }

    return {
        "outcome": "OPEN",
        "r": 0.0,
        "entry_time":
            candles[
                next_index
            ]["time"],
        "exit_time": None,
        "entry":
            entry_price,
        "exit":
            candles[-1]["close"],
        "tp": tp,
        "sl": sl,
    }


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades
):

    closed = [
        t
        for t in trades
        if t["outcome"]
        in ("TP", "SL")
    ]

    wins = sum(
        1
        for t in closed
        if t["outcome"] == "TP"
    )

    losses = sum(
        1
        for t in closed
        if t["outcome"] == "SL"
    )

    total = (
        wins
        + losses
    )

    win_rate = (
        wins
        / total
        * 100.0
        if total
        else 0.0
    )

    total_r = sum(
        t["r"]
        for t in closed
    )

    average_r = (
        total_r
        / total
        if total
        else 0.0
    )

    gross_profit = sum(
        t["r"]
        for t in closed
        if t["r"] > 0
    )

    gross_loss = abs(
        sum(
            t["r"]
            for t in closed
            if t["r"] < 0
        )
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else None
    )

    # --------------------------------------------------------
    # Equity / Drawdown in R
    # --------------------------------------------------------

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for trade in closed:

        equity += trade["r"]

        peak = max(
            peak,
            equity
        )

        drawdown = (
            peak
            - equity
        )

        max_dd = max(
            max_dd,
            drawdown
        )

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate":
            round(
                win_rate,
                2
            ),
        "profit_factor":
            (
                round(
                    profit_factor,
                    3
                )
                if profit_factor
                is not None
                else None
            ),
        "net_r":
            round(
                total_r,
                3
            ),
        "average_r":
            round(
                average_r,
                4
            ),
        "max_drawdown_r":
            round(
                max_dd,
                3
            ),
        "open_trades":
            len(trades)
            - total,
    }


# ============================================================
# SIGNAL FREQUENCY
# ============================================================

def signals_per_day(
    trades
):

    if not trades:
        return 0.0

    first = min(
        t["entry_time"]
        for t in trades
    )

    last = max(
        t["entry_time"]
        for t in trades
    )

    days = (
        last
        - first
    ) / (
        24
        * 60
        * 60
        * 1000
    )

    days = max(
        days,
        1.0
    )

    return (
        len(trades)
        / days
    )


# ============================================================
# BACKTEST COIN
# ============================================================

def backtest_coin(
    symbol,
    candles
):

    candles_1h = aggregate_1h(
        candles
    )

    trends = build_1h_trend_series(
        candles_1h
    )

    # --------------------------------------------------------
    # OOS timestamp
    # --------------------------------------------------------

    latest_time = candles[
        -1
    ]["time"]

    oos_start = (
        latest_time
        - OOS_DAYS
        * 24
        * 60
        * 60
        * 1000
    )

    warmup_time = (
        oos_start
        - 60
        * 24
        * 60
        * 60
        * 1000
    )

    trades = []

    cooldown_until = -1

    daily_trade_count = {}

    for i in range(
        len(candles)
        - 2
    ):

        candle = candles[
            i
        ]

        timestamp = candle[
            "time"
        ]

        if timestamp < warmup_time:
            continue

        # ----------------------------------------------------
        # Existing trade simulation is handled by jumping
        # after the trade exits.
        # ----------------------------------------------------

        if i < cooldown_until:
            continue

        h1_index = latest_closed_1h_index(
            candles_1h,
            timestamp
        )

        if h1_index is None:
            continue

        trend = trends[
            h1_index
        ]

        signal = find_signal(
            candles,
            i,
            trend
        )

        if signal is None:
            continue

        # ----------------------------------------------------
        # One signal per coin per UTC day.
        # ----------------------------------------------------

        day_key = datetime.fromtimestamp(
            timestamp / 1000,
            tz=timezone.utc
        ).strftime(
            "%Y-%m-%d"
        )

        count = daily_trade_count.get(
            day_key,
            0
        )

        if (
            count
            >= MAX_TRADES_PER_COIN_DAY
        ):
            continue

        trade = simulate_trade(
            candles,
            i,
            signal
        )

        if trade is None:
            continue

        trade[
            "symbol"
        ] = symbol

        trade[
            "direction"
        ] = signal[
            "direction"
        ]

        trade[
            "setup"
        ] = signal[
            "setup"
        ]

        trade[
            "rsi"
        ] = signal[
            "rsi"
        ]

        trade[
            "adx"
        ] = signal[
            "adx"
        ]

        trades.append(
            trade
        )

        daily_trade_count[
            day_key
        ] = count + 1

        # ----------------------------------------------------
        # Jump forward until trade exits.
        # ----------------------------------------------------

        exit_time = trade[
            "exit_time"
        ]

        if exit_time is not None:

            exit_index = i

            while (
                exit_index
                < len(candles)
                and candles[
                    exit_index
                ]["time"]
                <= exit_time
            ):

                exit_index += 1

            cooldown_until = (
                exit_index
                + COOLDOWN_BARS
            )

    # --------------------------------------------------------
    # Split OOS
    # --------------------------------------------------------

    is_trades = []

    oos_trades = []

    for trade in trades:

        if trade[
            "entry_time"
        ] >= oos_start:

            oos_trades.append(
                trade
            )

        else:

            is_trades.append(
                trade
            )

    return {
        "symbol": symbol,

        "all": calculate_stats(
            trades
        ),

        "is": calculate_stats(
            is_trades
        ),

        "oos": calculate_stats(
            oos_trades
        ),

        "signals_per_day_all":
            round(
                signals_per_day(
                    trades
                ),
                3
            ),

        "signals_per_day_oos":
            round(
                signals_per_day(
                    oos_trades
                ),
                3
            ),

        "trade_log": trades,
    }


# ============================================================
# FORMAT
# ============================================================

def print_stats(
    title,
    stats
):

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)

    print(
        f"Trades       : "
        f"{stats['trades']}"
    )

    print(
        f"Wins         : "
        f"{stats['wins']}"
    )

    print(
        f"Losses       : "
        f"{stats['losses']}"
    )

    print(
        f"Win Rate     : "
        f"{stats['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{stats['profit_factor']}"
    )

    print(
        f"Net R        : "
        f"{stats['net_r']:.3f}"
    )

    print(
        f"Average R    : "
        f"{stats['average_r']:.4f}"
    )

    print(
        f"Max Drawdown : "
        f"{stats['max_drawdown_r']:.3f}R"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SCORE HUNTER PRO v9"
    )

    print(
        "NEW STRATEGY"
    )

    print(
        "1H TREND + 15M PRECISION ENTRY"
    )

    print(
        "CLOSED CANDLE ONLY"
    )

    print(
        f"TP = {TP_R_MULTIPLIER}R"
    )

    print(
        f"SL = {SL_ATR_MULTIPLIER} ATR"
    )

    print(
        "NO PANDAS / NO NUMPY"
    )

    print(
        "=" * 60
    )

    now_ms = int(
        datetime.now(
            timezone.utc
        ).timestamp()
        * 1000
    )

    start_ms = (
        now_ms
        - HISTORY_DAYS
        * 24
        * 60
        * 60
        * 1000
        - 70
        * 24
        * 60
        * 60
        * 1000
    )

    end_ms = now_ms

    results = []

    all_trades = []

    for symbol in COINS:

        print(
            f"\nDownloading "
            f"{symbol}..."
        )

        try:

            candles = get_klines(
                symbol,
                ENTRY_INTERVAL,
                start_ms,
                end_ms
            )

            print(
                f"{symbol} "
                f"15M candles: "
                f"{len(candles)}"
            )

            if len(candles) < 500:

                print(
                    f"{symbol}: "
                    f"not enough data"
                )

                continue

            result = backtest_coin(
                symbol,
                candles
            )

            results.append(
                result
            )

            all_trades.extend(
                result[
                    "trade_log"
                ]
            )

            print_stats(
                f"{symbol} ALL",
                result["all"]
            )

            print(
                f"Signals/day: "
                f"{result['signals_per_day_all']}"
            )

            print_stats(
                f"{symbol} OOS",
                result["oos"]
            )

            print(
                f"OOS signals/day: "
                f"{result['signals_per_day_oos']}"
            )

        except Exception as e:

            print(
                f"{symbol} ERROR: "
                f"{type(e).__name__}: "
                f"{e}"
            )

        time.sleep(1)

    # ========================================================
    # COMBINED
    # ========================================================

    all_closed = [
        t
        for t in all_trades
        if t["outcome"]
        in ("TP", "SL")
    ]

    all_stats = calculate_stats(
        all_trades
    )

    # ========================================================
    # OOS COMBINED
    # ========================================================

    latest_time = (
        max(
            t["entry_time"]
            for t in all_trades
        )
        if all_trades
        else now_ms
    )

    oos_start = (
        latest_time
        - OOS_DAYS
        * 24
        * 60
        * 60
        * 1000
    )

    oos_trades = [
        t
        for t in all_trades
        if t["entry_time"]
        >= oos_start
    ]

    is_trades = [
        t
        for t in all_trades
        if t["entry_time"]
        < oos_start
    ]

    is_stats = calculate_stats(
        is_trades
    )

    oos_stats = calculate_stats(
        oos_trades
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_stats(
        "ALL COINS",
        all_stats
    )

    print(
        f"Signals/day: "
        f"{signals_per_day(all_trades):.3f}"
    )

    print_stats(
        "IN SAMPLE",
        is_stats
    )

    print(
        f"IS signals/day: "
        f"{signals_per_day(is_trades):.3f}"
    )

    print_stats(
        "OUT OF SAMPLE",
        oos_stats
    )

    print(
        f"OOS signals/day: "
        f"{signals_per_day(oos_trades):.3f}"
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TARGET CHECK"
    )

    print(
        "=" * 60
    )

    target_win = (
        oos_stats[
            "win_rate"
        ]
        >= 70.0
        and
        oos_stats[
            "win_rate"
        ]
        <= 80.0
    )

    target_frequency = (
        oos_stats[
            "trades"
        ] > 0
        and
        signals_per_day(
            oos_trades
        ) >= 2.0
    )

    print(
        "Win Rate 70-80% : "
        f"{'PASS' if target_win else 'FAIL'}"
    )

    print(
        "Signals >= 2/day : "
        f"{'PASS' if target_frequency else 'FAIL'}"
    )

    print(
        "Both targets    : "
        f"{'PASS' if target_win and target_frequency else 'FAIL'}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {

        "strategy":
            "SCORE HUNTER PRO v9",

        "description":
            "1H trend regime + 15M pullback/breakout precision entry",

        "closed_candle_only":
            True,

        "tp_r":
            TP_R_MULTIPLIER,

        "sl_atr":
            SL_ATR_MULTIPLIER,

        "history_days":
            HISTORY_DAYS,

        "oos_days":
            OOS_DAYS,

        "all":
            all_stats,

        "in_sample":
            is_stats,

        "out_of_sample":
            oos_stats,

        "signals_per_day_all":
            signals_per_day(
                all_trades
            ),

        "signals_per_day_is":
            signals_per_day(
                is_trades
            ),

        "signals_per_day_oos":
            signals_per_day(
                oos_trades
            ),

        "target_win_rate_pass":
            target_win,

        "target_frequency_pass":
            target_frequency,

        "target_both_pass":
            (
                target_win
                and
                target_frequency
            ),

        "per_coin":
            results,
    }

    # Remove trade logs from the
    # main JSON to keep it manageable.

    compact_results = []

    for result in results:

        compact_results.append({
            "symbol":
                result["symbol"],

            "all":
                result["all"],

            "is":
                result["is"],

            "oos":
                result["oos"],

            "signals_per_day_all":
                result[
                    "signals_per_day_all"
                ],

            "signals_per_day_oos":
                result[
                    "signals_per_day_oos"
                ],
        })

    output[
        "per_coin"
    ] = compact_results

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULT_FILE
    )

    print(
        "\nBACKTEST FINISHED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
