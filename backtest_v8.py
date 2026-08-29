import os
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER PRO v10
#
# NEW STRATEGY
#
# 4H  = MACRO TREND
# 1H  = TREND CONFIRMATION
# 15M = PRECISION ENTRY
#
# PULLBACK + MOMENTUM + BREAKOUT
# EMA + RSI + ADX + ATR
# RANGE FILTER
# COOLDOWN
#
# CLOSED CANDLE ONLY
# ENTRY = NEXT 15M OPEN
#
# NO PANDAS
# NO NUMPY
#
# DATA SOURCE = KRAKEN
#
# TP = 1.3R
# SL = ATR / STRUCTURE
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

COINS = [
    "ETH",
    "SOL",
    "XRP",
    "BTC",
    "ADA",
    "LINK",
    "DOGE",
]


KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"


INTERVAL_15M = 15
INTERVAL_1H = 60
INTERVAL_4H = 240


# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------

HISTORY_DAYS = 90

# Extra candles for indicator warmup
WARMUP_DAYS = 40


# ------------------------------------------------------------
# INDICATORS
# ------------------------------------------------------------

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14


# ------------------------------------------------------------
# TREND FILTER
# ------------------------------------------------------------

ADX_MIN = 20.0

RSI_LONG_MIN = 50.0
RSI_LONG_MAX = 72.0

RSI_SHORT_MIN = 28.0
RSI_SHORT_MAX = 50.0


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

PULLBACK_LOOKBACK = 8
BREAKOUT_LOOKBACK = 5

EMA_DISTANCE_ATR = 0.80

MIN_BODY_RATIO = 0.50

MIN_CLOSE_LOCATION = 0.65


# ------------------------------------------------------------
# RISK
# ------------------------------------------------------------

TP_R = 1.30

SL_ATR_MULT = 1.00

STRUCTURE_BUFFER_ATR = 0.10

MIN_RR = 1.20


# ------------------------------------------------------------
# TRADE MANAGEMENT
# ------------------------------------------------------------

COOLDOWN_BARS = 8

MAX_HOLD_BARS = 32


# ------------------------------------------------------------
# COSTS
# ------------------------------------------------------------

FEE_PER_SIDE = 0.0004

SLIPPAGE_PER_SIDE = 0.0002

TOTAL_ENTRY_COST = (
    FEE_PER_SIDE
    + SLIPPAGE_PER_SIDE
)

TOTAL_EXIT_COST = (
    FEE_PER_SIDE
    + SLIPPAGE_PER_SIDE
)


# ------------------------------------------------------------
# ACCOUNT
# ------------------------------------------------------------

INITIAL_BALANCE = 1000.0

RISK_PER_TRADE = 0.01


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

RESULT_FILE = "backtest_v10_results.json"


# ============================================================
# SYMBOL MAP
# ============================================================

KRAKEN_SYMBOLS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}


# ============================================================
# HTTP
# ============================================================

def http_get(url, params):

    query = urllib.parse.urlencode(params)

    full_url = (
        url
        + "?"
        + query
    )

    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent":
                "ScoreHunterBacktest/10.0"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        raw = response.read()

    return json.loads(raw.decode("utf-8"))


# ============================================================
# KRAKEN OHLC
# ============================================================

def get_ohlc(symbol, interval, days):

    pair = KRAKEN_SYMBOLS[symbol]

    interval_seconds = (
        interval * 60
    )

    end_time = int(
        time.time()
    )

    start_time = (
        end_time
        - days * 86400
    )

    candles = []

    cursor = start_time

    safety_counter = 0

    print(
        f"Downloading {symbol} "
        f"{interval}m..."
    )

    while cursor < end_time:

        safety_counter += 1

        if safety_counter > 100:

            raise RuntimeError(
                f"{symbol} {interval}m: "
                f"pagination safety stop"
            )

        data = http_get(
            KRAKEN_URL,
            {
                "pair": pair,
                "interval": interval,
                "since": cursor
            }
        )

        errors = data.get(
            "error",
            []
        )

        if errors:

            raise RuntimeError(
                f"{symbol} Kraken API error: "
                f"{errors}"
            )

        result = data.get(
            "result",
            {}
        )

        pair_key = None

        for key in result:

            if key != "last":

                pair_key = key
                break

        if pair_key is None:

            raise RuntimeError(
                f"{symbol}: "
                f"Kraken returned no pair data"
            )

        rows = result[pair_key]

        if not rows:

            break

        added = 0

        last_timestamp = None

        for row in rows:

            timestamp = int(
                row[0]
            )

            last_timestamp = timestamp

            if (
                timestamp < start_time
                or timestamp >= end_time
            ):
                continue

            candle = {
                "time": timestamp,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[6])
            }

            candles.append(
                candle
            )

            added += 1

        if last_timestamp is None:
            break

        next_cursor = (
            last_timestamp
            + interval_seconds
        )

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        # Kraken rate-limit protection
        time.sleep(0.35)

        if len(rows) < 700:
            break

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    unique = {}

    for candle in candles:

        unique[
            candle["time"]
        ] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x["time"]
    )

    # --------------------------------------------------------
    # Remove currently forming candle
    # --------------------------------------------------------

    now = int(
        time.time()
    )

    candles = [
        c
        for c in candles
        if (
            c["time"]
            + interval_seconds
            <= now
        )
    ]

    if len(candles) < 100:

        raise RuntimeError(
            f"{symbol} {interval}m: "
            f"only {len(candles)} candles received"
        )

    print(
        f"{symbol} {interval}m candles: "
        f"{len(candles)}"
    )

    return candles


# ============================================================
# EMA SERIES
# ============================================================

def ema_series(values, period):

    result = [
        None
        for _ in values
    ]

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

    multiplier = (
        2.0
        / (period + 1)
    )

    for i in range(
        period,
        len(values)
    ):

        value = (
            (
                values[i]
                - value
            )
            * multiplier
            + value
        )

        result[i] = value

    return result


# ============================================================
# RSI SERIES
# ============================================================

def rsi_series(
    candles,
    period=14
):

    n = len(candles)

    result = [
        None
        for _ in range(n)
    ]

    if n <= period:
        return result

    closes = [
        c["close"]
        for c in candles
    ]

    gains = []
    losses = []

    for i in range(
        1,
        n
    ):

        change = (
            closes[i]
            - closes[i - 1]
        )

        gains.append(
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
        )

    avg_gain = (
        sum(gains[:period])
        / period
    )

    avg_loss = (
        sum(losses[:period])
        / period
    )

    if avg_loss == 0:

        result[period] = 100.0

    else:

        rs = (
            avg_gain
            / avg_loss
        )

        result[period] = (
            100
            - 100 / (1 + rs)
        )

    for i in range(
        period + 1,
        n
    ):

        avg_gain = (
            (
                avg_gain
                * (period - 1)
                + gains[i - 1]
            )
            / period
        )

        avg_loss = (
            (
                avg_loss
                * (period - 1)
                + losses[i - 1]
            )
            / period
        )

        if avg_loss == 0:

            result[i] = 100.0

        else:

            rs = (
                avg_gain
                / avg_loss
            )

            result[i] = (
                100
                - 100 / (1 + rs)
            )

    return result


# ============================================================
# ATR SERIES
# ============================================================

def atr_series(
    candles,
    period=14
):

    n = len(candles)

    result = [
        None
        for _ in range(n)
    ]

    if n <= period:
        return result

    tr = [
        None
        for _ in range(n)
    ]

    for i in range(
        1,
        n
    ):

        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = (
            candles[i - 1]["close"]
        )

        tr[i] = max(
            high - low,
            abs(
                high
                - prev_close
            ),
            abs(
                low
                - prev_close
            )
        )

    first = [
        x
        for x in tr[1:period + 1]
        if x is not None
    ]

    if len(first) < period:
        return result

    value = (
        sum(first)
        / period
    )

    result[period] = value

    for i in range(
        period + 1,
        n
    ):

        value = (
            (
                value
                * (period - 1)
                + tr[i]
            )
            / period
        )

        result[i] = value

    return result


# ============================================================
# ADX SERIES
# ============================================================

def adx_series(
    candles,
    period=14
):

    n = len(candles)

    result = [
        None
        for _ in range(n)
    ]

    if n < period * 2 + 5:
        return result

    tr = []
    plus_dm = []
    minus_dm = []

    for i in range(
        1,
        n
    ):

        current = candles[i]
        previous = candles[i - 1]

        high = current["high"]
        low = current["low"]

        prev_high = previous["high"]
        prev_low = previous["low"]
        prev_close = previous["close"]

        true_range = max(
            high - low,
            abs(
                high
                - prev_close
            ),
            abs(
                low
                - prev_close
            )
        )

        up = (
            high
            - prev_high
        )

        down = (
            prev_low
            - low
        )

        if (
            up > down
            and up > 0
        ):
            p = up
        else:
            p = 0.0

        if (
            down > up
            and down > 0
        ):
            m = down
        else:
            m = 0.0

        tr.append(
            true_range
        )

        plus_dm.append(p)
        minus_dm.append(m)

    if len(tr) < period * 2:
        return result

    atr_value = (
        sum(tr[:period])
        / period
    )

    plus_value = (
        sum(plus_dm[:period])
        / period
    )

    minus_value = (
        sum(minus_dm[:period])
        / period
    )

    dx = []

    for i in range(
        period,
        len(tr)
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
            dx.append(0.0)
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

            dx.append(0.0)

        else:

            dx.append(
                100
                * abs(
                    plus_di
                    - minus_di
                )
                / denominator
            )

    if len(dx) < period:
        return result

    adx_value = (
        sum(dx[:period])
        / period
    )

    # First ADX location
    first_index = (
        1
        + period
        + period
        - 1
    )

    if first_index < n:

        result[
            first_index
        ] = adx_value

    for j in range(
        period,
        len(dx)
    ):

        adx_value = (
            (
                adx_value
                * (period - 1)
                + dx[j]
            )
            / period
        )

        index = (
            first_index
            + (j - period + 1)
        )

        if index < n:

            result[index] = (
                adx_value
            )

    return result


# ============================================================
# CANDLE HELPERS
# ============================================================

def bullish_candle(c):

    return c["close"] > c["open"]


def bearish_candle(c):

    return c["close"] < c["open"]


def body_ratio(c):

    candle_range = (
        c["high"]
        - c["low"]
    )

    if candle_range <= 0:
        return 0.0

    return (
        abs(
            c["close"]
            - c["open"]
        )
        / candle_range
    )


def close_location(c):

    candle_range = (
        c["high"]
        - c["low"]
    )

    if candle_range <= 0:
        return 0.5

    return (
        c["close"]
        - c["low"]
    ) / candle_range


# ============================================================
# 4H TREND
# ============================================================

def build_4h_context(candles):

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = ema_series(
        closes,
        EMA_FAST
    )

    ema50 = ema_series(
        closes,
        EMA_MID
    )

    ema200 = ema_series(
        closes,
        EMA_SLOW
    )

    context = []

    for i in range(
        len(candles)
    ):

        direction = None

        if (
            ema20[i] is not None
            and ema50[i] is not None
            and ema200[i] is not None
        ):

            close = closes[i]

            if (
                close > ema200[i]
                and ema20[i] > ema50[i]
                and ema50[i] > ema200[i]
            ):

                direction = "LONG"

            elif (
                close < ema200[i]
                and ema20[i] < ema50[i]
                and ema50[i] < ema200[i]
            ):

                direction = "SHORT"

        context.append(
            {
                "direction": direction,
                "ema20": ema20[i],
                "ema50": ema50[i],
                "ema200": ema200[i]
            }
        )

    return context


# ============================================================
# MAP 4H CONTEXT TO 15M
# ============================================================

def get_context_at_time(
    candles_4h,
    context_4h,
    timestamp
):

    selected = None

    for i in range(
        len(candles_4h)
    ):

        if (
            candles_4h[i]["time"]
            <= timestamp
        ):

            selected = context_4h[i]

        else:

            break

    return selected


# ============================================================
# MAP 1H CONTEXT TO 15M
# ============================================================

def build_1h_context(
    candles_1h
):

    closes = [
        c["close"]
        for c in candles_1h
    ]

    ema20 = ema_series(
        closes,
        EMA_FAST
    )

    ema50 = ema_series(
        closes,
        EMA_MID
    )

    ema200 = ema_series(
        closes,
        EMA_SLOW
    )

    rsi = rsi_series(
        candles_1h,
        RSI_PERIOD
    )

    adx = adx_series(
        candles_1h,
        ADX_PERIOD
    )

    result = []

    for i in range(
        len(candles_1h)
    ):

        direction = None

        if (
            ema20[i] is not None
            and ema50[i] is not None
            and ema200[i] is not None
        ):

            close = closes[i]

            if (
                close > ema200[i]
                and ema20[i] > ema50[i]
                and ema50[i] > ema200[i]
            ):

                direction = "LONG"

            elif (
                close < ema200[i]
                and ema20[i] < ema50[i]
                and ema50[i] < ema200[i]
            ):

                direction = "SHORT"

        result.append(
            {
                "direction": direction,
                "ema20": ema20[i],
                "ema50": ema50[i],
                "ema200": ema200[i],
                "rsi": rsi[i],
                "adx": adx[i]
            }
        )

    return result


def get_1h_context_at_time(
    candles_1h,
    context_1h,
    timestamp
):

    selected = None

    for i in range(
        len(candles_1h)
    ):

        if (
            candles_1h[i]["time"]
            <= timestamp
        ):

            selected = context_1h[i]

        else:

            break

    return selected


# ============================================================
# 15M INDICATORS
# ============================================================

def build_15m_indicators(
    candles
):

    closes = [
        c["close"]
        for c in candles
    ]

    return {
        "ema20": ema_series(
            closes,
            EMA_FAST
        ),
        "ema50": ema_series(
            closes,
            EMA_MID
        ),
        "ema200": ema_series(
            closes,
            EMA_SLOW
        ),
        "rsi": rsi_series(
            candles,
            RSI_PERIOD
        ),
        "atr": atr_series(
            candles,
            ATR_PERIOD
        ),
        "adx": adx_series(
            candles,
            ADX_PERIOD
        )
    }


# ============================================================
# PULLBACK
# ============================================================

def has_pullback(
    candles,
    indicators,
    index,
    direction
):

    atr_value = indicators["atr"][index]

    if atr_value is None:
        return False

    start = max(
        0,
        index
        - PULLBACK_LOOKBACK
    )

    for i in range(
        start,
        index
    ):

        candle = candles[i]

        ema20 = (
            indicators["ema20"][i]
        )

        ema50 = (
            indicators["ema50"][i]
        )

        if (
            ema20 is None
            or ema50 is None
        ):
            continue

        if direction == "LONG":

            if (
                candle["low"]
                <= ema20
                + atr_value
                * EMA_DISTANCE_ATR
            ):

                return True

            if (
                candle["low"]
                <= ema50
                + atr_value
                * 0.50
            ):

                return True

        else:

            if (
                candle["high"]
                >= ema20
                - atr_value
                * EMA_DISTANCE_ATR
            ):

                return True

            if (
                candle["high"]
                >= ema50
                - atr_value
                * 0.50
            ):

                return True

    return False


# ============================================================
# BREAKOUT
# ============================================================

def breakout_confirmation(
    candles,
    index,
    direction
):

    if index < BREAKOUT_LOOKBACK + 1:
        return False

    start = (
        index
        - BREAKOUT_LOOKBACK
    )

    end = index

    previous = candles[
        start:end
    ]

    current = candles[index]

    if direction == "LONG":

        resistance = max(
            c["high"]
            for c in previous
        )

        return (
            current["close"]
            > resistance
        )

    else:

        support = min(
            c["low"]
            for c in previous
        )

        return (
            current["close"]
            < support
        )


# ============================================================
# MOMENTUM
# ============================================================

def momentum_confirmation(
    candles,
    index,
    direction
):

    if index < 3:
        return False

    current = candles[index]
    previous = candles[index - 1]

    if (
        body_ratio(current)
        < MIN_BODY_RATIO
    ):
        return False

    if direction == "LONG":

        return (
            bullish_candle(current)
            and
            current["close"]
            > previous["close"]
            and
            close_location(current)
            >= MIN_CLOSE_LOCATION
        )

    return (
        bearish_candle(current)
        and
        current["close"]
        < previous["close"]
        and
        (
            1.0
            - close_location(current)
        )
        >= MIN_CLOSE_LOCATION
    )


# ============================================================
# RANGE FILTER
# ============================================================

def range_filter(
    candles,
    indicators,
    index
):

    atr_value = (
        indicators["atr"][index]
    )

    ema20 = (
        indicators["ema20"][index]
    )

    ema50 = (
        indicators["ema50"][index]
    )

    if (
        atr_value is None
        or ema20 is None
        or ema50 is None
    ):

        return False

    separation = abs(
        ema20 - ema50
    )

    # Avoid flat EMA markets
    if (
        separation
        < atr_value * 0.12
    ):

        return False

    return True


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(
    candles_15m,
    indicators,
    index,
    direction
):

    candle = candles_15m[index]

    ema20 = indicators["ema20"][index]
    ema50 = indicators["ema50"][index]
    ema200 = indicators["ema200"][index]

    rsi = indicators["rsi"][index]
    atr_value = indicators["atr"][index]
    adx = indicators["adx"][index]

    if any(
        x is None
        for x in [
            ema20,
            ema50,
            ema200,
            rsi,
            atr_value,
            adx
        ]
    ):
        return None

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    if adx < ADX_MIN:
        return None

    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    if not range_filter(
        candles_15m,
        indicators,
        index
    ):
        return None

    # --------------------------------------------------------
    # EMA ALIGNMENT
    # --------------------------------------------------------

    if direction == "LONG":

        if not (
            candle["close"]
            > ema20
            > ema50
            > ema200
        ):
            return None

    else:

        if not (
            candle["close"]
            < ema20
            < ema50
            < ema200
        ):
            return None

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if direction == "LONG":

        if not (
            RSI_LONG_MIN
            <= rsi
            <= RSI_LONG_MAX
        ):
            return None

    else:

        if not (
            RSI_SHORT_MIN
            <= rsi
            <= RSI_SHORT_MAX
        ):
            return None

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    pullback = has_pullback(
        candles_15m,
        indicators,
        index,
        direction
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    breakout = breakout_confirmation(
        candles_15m,
        index,
        direction
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = momentum_confirmation(
        candles_15m,
        index,
        direction
    )

    # Need pullback and confirmation.
    # Breakout OR momentum is enough.
    if not pullback:
        return None

    if not (
        breakout
        or momentum
    ):
        return None

    # --------------------------------------------------------
    # SETUP TYPE
    # --------------------------------------------------------

    if breakout:

        setup = "PULLBACK_BREAKOUT"

    else:

        setup = "PULLBACK_MOMENTUM"

    return {
        "direction": direction,
        "setup": setup,
        "signal_index": index,
        "signal_time": candle["time"],
        "signal_close": candle["close"],
        "atr": atr_value,
        "rsi": rsi,
        "adx": adx
    }


# ============================================================
# RISK
# ============================================================

def calculate_levels(
    candles,
    index,
    direction,
    entry,
    atr_value
):

    start = max(
        0,
        index - 5
    )

    recent = candles[
        start:index + 1
    ]

    recent_low = min(
        c["low"]
        for c in recent
    )

    recent_high = max(
        c["high"]
        for c in recent
    )

    if direction == "LONG":

        atr_sl = (
            entry
            - atr_value
            * SL_ATR_MULT
        )

        structure_sl = (
            recent_low
            - atr_value
            * STRUCTURE_BUFFER_ATR
        )

        sl = min(
            atr_sl,
            structure_sl
        )

        risk = (
            entry
            - sl
        )

        tp = (
            entry
            + risk
            * TP_R
        )

    else:

        atr_sl = (
            entry
            + atr_value
            * SL_ATR_MULT
        )

        structure_sl = (
            recent_high
            + atr_value
            * STRUCTURE_BUFFER_ATR
        )

        sl = max(
            atr_sl,
            structure_sl
        )

        risk = (
            sl
            - entry
        )

        tp = (
            entry
            - risk
            * TP_R
        )

    if risk <= 0:
        return None

    rr = (
        abs(tp - entry)
        / risk
    )

    if rr < MIN_RR:
        return None

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk": risk,
        "rr": rr
    }


# ============================================================
# TRADE RESULT
# ============================================================

def evaluate_trade(
    candles,
    entry_index,
    direction,
    entry,
    sl,
    tp
):

    end = min(
        len(candles),
        entry_index
        + MAX_HOLD_BARS
        + 1
    )

    for i in range(
        entry_index,
        end
    ):

        candle = candles[i]

        high = candle["high"]
        low = candle["low"]

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if direction == "LONG":

            hit_sl = (
                low <= sl
            )

            hit_tp = (
                high >= tp
            )

            # Conservative:
            # if both happen inside same candle,
            # count SL first.
            if hit_sl and hit_tp:

                return {
                    "result": "LOSS",
                    "r": -1.0,
                    "exit_index": i,
                    "exit_price": sl,
                    "reason":
                        "SL_AND_TP_SAME_CANDLE"
                }

            if hit_sl:

                return {
                    "result": "LOSS",
                    "r": -1.0,
                    "exit_index": i,
                    "exit_price": sl,
                    "reason": "SL"
                }

            if hit_tp:

                return {
                    "result": "WIN",
                    "r": TP_R,
                    "exit_index": i,
                    "exit_price": tp,
                    "reason": "TP"
                }

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            hit_sl = (
                high >= sl
            )

            hit_tp = (
                low <= tp
            )

            if hit_sl and hit_tp:

                return {
                    "result": "LOSS",
                    "r": -1.0,
                    "exit_index": i,
                    "exit_price": sl,
                    "reason":
                        "SL_AND_TP_SAME_CANDLE"
                }

            if hit_sl:

                return {
                    "result": "LOSS",
                    "r": -1.0,
                    "exit_index": i,
                    "exit_price": sl,
                    "reason": "SL"
                }

            if hit_tp:

                return {
                    "result": "WIN",
                    "r": TP_R,
                    "exit_index": i,
                    "exit_price": tp,
                    "reason": "TP"
                }

    # --------------------------------------------------------
    # TIME EXIT
    # --------------------------------------------------------

    final_index = end - 1

    final_close = (
        candles[final_index]["close"]
    )

    if direction == "LONG":

        raw_r = (
            final_close - entry
        ) / (
            entry - sl
        )

    else:

        raw_r = (
            entry - final_close
        ) / (
            sl - entry
        )

    return {
        "result":
            "TIME_EXIT",
        "r":
            raw_r,
        "exit_index":
            final_index,
        "exit_price":
            final_close,
        "reason":
            "MAX_HOLD"
    }


# ============================================================
# COST ADJUSTMENT
# ============================================================

def apply_costs(
    r,
    entry,
    exit_price,
    risk
):

    if risk <= 0:
        return r

    total_cost = (
        entry
        * TOTAL_ENTRY_COST
        +
        exit_price
        * TOTAL_EXIT_COST
    )

    cost_r = (
        total_cost
        / risk
    )

    return r - cost_r


# ============================================================
# BACKTEST ONE COIN
# ============================================================

def backtest_coin(
    symbol,
    candles_15m,
    candles_1h,
    candles_4h
):

    indicators = build_15m_indicators(
        candles_15m
    )

    context_1h = build_1h_context(
        candles_1h
    )

    context_4h = build_4h_context(
        candles_4h
    )

    warmup = 500

    trades = []

    balance = INITIAL_BALANCE

    equity_curve = [
        balance
    ]

    next_allowed_index = warmup

    i = warmup

    while (
        i
        < len(candles_15m) - 2
    ):

        candle = candles_15m[i]

        # ----------------------------------------------------
        # 4H CONTEXT
        # ----------------------------------------------------

        ctx4 = get_context_at_time(
            candles_4h,
            context_4h,
            candle["time"]
        )

        if ctx4 is None:

            i += 1
            continue

        macro_direction = (
            ctx4["direction"]
        )

        if macro_direction is None:

            i += 1
            continue

        # ----------------------------------------------------
        # 1H CONTEXT
        # ----------------------------------------------------

        ctx1 = get_1h_context_at_time(
            candles_1h,
            context_1h,
            candle["time"]
        )

        if ctx1 is None:

            i += 1
            continue

        if (
            ctx1["direction"]
            != macro_direction
        ):

            i += 1
            continue

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if i < next_allowed_index:

            i += 1
            continue

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal = generate_signal(
            candles_15m,
            indicators,
            i,
            macro_direction
        )

        if signal is None:

            i += 1
            continue

        # ----------------------------------------------------
        # NEXT CANDLE OPEN
        #
        # This prevents entering at the same candle close.
        # ----------------------------------------------------

        entry_index = i + 1

        entry = (
            candles_15m[
                entry_index
            ]["open"]
        )

        levels = calculate_levels(
            candles_15m,
            i,
            macro_direction,
            entry,
            signal["atr"]
        )

        if levels is None:

            i += 1
            continue

        # ----------------------------------------------------
        # TRADE
        # ----------------------------------------------------

        result = evaluate_trade(
            candles_15m,
            entry_index,
            macro_direction,
            levels["entry"],
            levels["sl"],
            levels["tp"]
        )

        adjusted_r = apply_costs(
            result["r"],
            levels["entry"],
            result["exit_price"],
            levels["risk"]
        )

        # ----------------------------------------------------
        # ACCOUNT
        # ----------------------------------------------------

        risk_money = (
            balance
            * RISK_PER_TRADE
        )

        pnl = (
            risk_money
            * adjusted_r
        )

        balance += pnl

        equity_curve.append(
            balance
        )

        trade = {
            "symbol": symbol,
            "direction":
                macro_direction,
            "setup":
                signal["setup"],
            "signal_time":
                signal["signal_time"],
            "entry_time":
                candles_15m[
                    entry_index
                ]["time"],
            "entry":
                levels["entry"],
            "sl":
                levels["sl"],
            "tp":
                levels["tp"],
            "rr":
                levels["rr"],
            "r":
                adjusted_r,
            "raw_r":
                result["r"],
            "result":
                result["result"],
            "reason":
                result["reason"],
            "exit_price":
                result["exit_price"],
            "rsi":
                signal["rsi"],
            "adx":
                signal["adx"]
        }

        trades.append(
            trade
        )

        # ----------------------------------------------------
        # COOLDOWN AFTER TRADE
        # ----------------------------------------------------

        next_allowed_index = (
            result["exit_index"]
            + COOLDOWN_BARS
        )

        # Jump after trade
        i = max(
            i + 1,
            result["exit_index"]
        )

    return {
        "symbol": symbol,
        "trades": trades,
        "balance": balance,
        "equity_curve":
            equity_curve
    }


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades,
    initial_balance=INITIAL_BALANCE
):

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "net_r": 0.0,
            "average_r": 0.0,
            "max_drawdown_r": 0.0,
            "max_drawdown_percent": 0.0,
            "balance": initial_balance
        }

    wins = [
        t
        for t in trades
        if t["r"] > 0
    ]

    losses = [
        t
        for t in trades
        if t["r"] <= 0
    ]

    gross_profit = sum(
        max(t["r"], 0)
        for t in trades
    )

    gross_loss = abs(
        sum(
            min(t["r"], 0)
            for t in trades
        )
    )

    if gross_loss == 0:

        profit_factor = None

    else:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    net_r = sum(
        t["r"]
        for t in trades
    )

    average_r = (
        net_r
        / len(trades)
    )

    # --------------------------------------------------------
    # Equity in R
    # --------------------------------------------------------

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for trade in trades:

        equity += trade["r"]

        if equity > peak:
            peak = equity

        dd = peak - equity

        if dd > max_dd:
            max_dd = dd

    # Approximate percentage drawdown
    balance = initial_balance

    peak_balance = balance
    max_dd_percent = 0.0

    for trade in trades:

        balance *= (
            1
            + trade["r"]
            * RISK_PER_TRADE
        )

        if balance > peak_balance:
            peak_balance = balance

        if peak_balance > 0:

            dd_percent = (
                (
                    peak_balance
                    - balance
                )
                / peak_balance
                * 100
            )

            if (
                dd_percent
                > max_dd_percent
            ):

                max_dd_percent = (
                    dd_percent
                )

    return {
        "trades":
            len(trades),
        "wins":
            len(wins),
        "losses":
            len(losses),
        "win_rate":
            len(wins)
            / len(trades)
            * 100,
        "profit_factor":
            profit_factor,
        "net_r":
            net_r,
        "average_r":
            average_r,
        "max_drawdown_r":
            max_dd,
        "max_drawdown_percent":
            max_dd_percent,
        "balance":
            balance
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
        t["signal_time"]
        for t in trades
    )

    last = max(
        t["signal_time"]
        for t in trades
    )

    days = (
        last - first
    ) / 86400

    if days <= 0:
        return float(
            len(trades)
        )

    return (
        len(trades)
        / days
    )


# ============================================================
# SPLIT IS / OOS
# ============================================================

def split_trades(
    trades,
    oos_days=30
):

    if not trades:

        return [], []

    latest = max(
        t["signal_time"]
        for t in trades
    )

    cutoff = (
        latest
        - oos_days * 86400
    )

    is_trades = []
    oos_trades = []

    for trade in trades:

        if (
            trade["signal_time"]
            >= cutoff
        ):

            oos_trades.append(
                trade
            )

        else:

            is_trades.append(
                trade
            )

    return (
        is_trades,
        oos_trades
    )


# ============================================================
# PRINT STATS
# ============================================================

def print_stats(
    title,
    stats,
    frequency
):

    print(
        "\n"
        + "=" * 60
    )

    print(title)

    print(
        "=" * 60
    )

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

    if stats["profit_factor"] is None:

        pf = "None"

    else:

        pf = (
            f"{stats['profit_factor']:.2f}"
        )

    print(
        f"Profit Factor: "
        f"{pf}"
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

    print(
        f"Max DD %     : "
        f"{stats['max_drawdown_percent']:.2f}%"
    )

    print(
        f"Balance      : "
        f"${stats['balance']:.2f}"
    )

    print(
        f"Signals/day  : "
        f"{frequency:.3f}"
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
        "SCORE HUNTER PRO v10"
    )

    print(
        "4H TREND + 1H CONFIRMATION + 15M ENTRY"
    )

    print(
        "PULLBACK + MOMENTUM + BREAKOUT"
    )

    print(
        "KRAKEN DATA"
    )

    print(
        "NO PANDAS / NO NUMPY"
    )

    print(
        "CLOSED CANDLE ONLY"
    )

    print(
        "ENTRY = NEXT 15M OPEN"
    )

    print(
        f"TP = {TP_R}R"
    )

    print(
        f"SL = {SL_ATR_MULT} ATR / STRUCTURE"
    )

    print(
        "=" * 60
    )

    all_trades = []

    coin_results = {}

    failed_coins = []

    try:

        for symbol in COINS:

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"PROCESSING {symbol}"
            )

            print(
                "=" * 60
            )

            try:

                # ------------------------------------------------
                # Download enough data
                # ------------------------------------------------

                candles_15m = get_ohlc(
                    symbol,
                    INTERVAL_15M,
                    HISTORY_DAYS
                    + WARMUP_DAYS
                )

                candles_1h = get_ohlc(
                    symbol,
                    INTERVAL_1H,
                    HISTORY_DAYS
                    + WARMUP_DAYS
                )

                candles_4h = get_ohlc(
                    symbol,
                    INTERVAL_4H,
                    HISTORY_DAYS
                    + WARMUP_DAYS
                )

                result = backtest_coin(
                    symbol,
                    candles_15m,
                    candles_1h,
                    candles_4h
                )

                coin_results[
                    symbol
                ] = result

                trades = result[
                    "trades"
                ]

                all_trades.extend(
                    trades
                )

                stats = calculate_stats(
                    trades
                )

                frequency = (
                    signals_per_day(
                        trades
                    )
                )

                print_stats(
                    symbol,
                    stats,
                    frequency
                )

            except Exception as e:

                failed_coins.append(
                    {
                        "symbol": symbol,
                        "error":
                            f"{type(e).__name__}: {e}"
                    }
                )

                print(
                    f"{symbol} FAILED:"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                continue

    except Exception as e:

        print(
            "\nFATAL DATA ERROR:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise

    # ========================================================
    # IMPORTANT:
    # If ALL coins failed, DO NOT create fake 0-trade result.
    # ========================================================

    if (
        len(failed_coins)
        == len(COINS)
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            "BACKTEST ABORTED"
        )

        print(
            "=" * 60
        )

        print(
            "No coin data was successfully downloaded."
        )

        print(
            "The results are NOT valid."
        )

        print(
            "\nErrors:"
        )

        for item in failed_coins:

            print(
                f"- {item['symbol']}: "
                f"{item['error']}"
            )

        raise RuntimeError(
            "All data downloads failed."
        )

    # ========================================================
    # ALL COINS
    # ========================================================

    all_trades.sort(
        key=lambda x:
            x["signal_time"]
    )

    all_stats = calculate_stats(
        all_trades
    )

    all_frequency = (
        signals_per_day(
            all_trades
        )
    )

    print_stats(
        "ALL COINS",
        all_stats,
        all_frequency
    )

    # ========================================================
    # OOS
    # ========================================================

    is_trades, oos_trades = (
        split_trades(
            all_trades,
            30
        )
    )

    is_stats = calculate_stats(
        is_trades
    )

    oos_stats = calculate_stats(
        oos_trades
    )

    is_frequency = (
        signals_per_day(
            is_trades
        )
    )

    oos_frequency = (
        signals_per_day(
            oos_trades
        )
    )

    print_stats(
        "IN SAMPLE",
        is_stats,
        is_frequency
    )

    print_stats(
        "OUT OF SAMPLE - LAST 30 DAYS",
        oos_stats,
        oos_frequency
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

    wr_ok = (
        70
        <= oos_stats["win_rate"]
        <= 80
    )

    frequency_ok = (
        oos_frequency >= 2.0
    )

    print(
        "OOS Win Rate 70-80% : "
        + (
            "PASS"
            if wr_ok
            else "FAIL"
        )
    )

    print(
        "OOS Signals >= 2/day: "
        + (
            "PASS"
            if frequency_ok
            else "FAIL"
        )
    )

    print(
        "Both targets        : "
        + (
            "PASS"
            if (
                wr_ok
                and frequency_ok
            )
            else "FAIL"
        )
    )

    # ========================================================
    # FAILED COINS
    # ========================================================

    if failed_coins:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "FAILED COINS"
        )

        print(
            "=" * 60
        )

        for item in failed_coins:

            print(
                f"{item['symbol']}: "
                f"{item['error']}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    output = {

        "strategy":
            "SCORE HUNTER PRO v10",

        "data_source":
            "Kraken Spot REST",

        "timeframes": {
            "macro": "4H",
            "confirmation": "1H",
            "entry": "15M"
        },

        "settings": {

            "history_days":
                HISTORY_DAYS,

            "warmup_days":
                WARMUP_DAYS,

            "tp_r":
                TP_R,

            "sl_atr":
                SL_ATR_MULT,

            "adx_min":
                ADX_MIN,

            "risk_per_trade":
                RISK_PER_TRADE,

            "fee_per_side":
                FEE_PER_SIDE,

            "slippage_per_side":
                SLIPPAGE_PER_SIDE,

            "cooldown_bars":
                COOLDOWN_BARS,

            "max_hold_bars":
                MAX_HOLD_BARS
        },

        "all_stats":
            all_stats,

        "is_stats":
            is_stats,

        "oos_stats":
            oos_stats,

        "signal_frequency": {

            "all":
                all_frequency,

            "is":
                is_frequency,

            "oos":
                oos_frequency
        },

        "targets": {

            "win_rate_target":
                "70-80%",

            "signals_per_day_target":
                ">=2",

            "win_rate_pass":
                wr_ok,

            "frequency_pass":
                frequency_ok,

            "both_pass":
                (
                    wr_ok
                    and frequency_ok
                )
        },

        "failed_coins":
            failed_coins,

        "coin_stats": {}
    }

    # --------------------------------------------------------
    # Coin statistics
    # --------------------------------------------------------

    for symbol in COINS:

        result = coin_results.get(
            symbol
        )

        if result is None:
            continue

        stats = calculate_stats(
            result["trades"]
        )

        output[
            "coin_stats"
        ][symbol] = stats

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BACKTEST FINISHED"
    )

    print(
        "=" * 60
    )

    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )

    print(
        f"Valid trades: "
        f"{len(all_trades)}"
    )

    if len(all_trades) == 0:

        print(
            "\nWARNING:"
        )

        print(
            "No valid trades were generated."
        )

        print(
            "This is a strategy result, "
            "not a data-download failure."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
