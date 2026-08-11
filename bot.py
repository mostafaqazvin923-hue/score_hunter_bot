import os
import json
import time
import math
import requests


# ============================================================
# SCORE HUNTER PRO v4
#
# SIGNAL ONLY - NO AUTOMATIC ORDER EXECUTION
#
# MARKET MODEL
# ------------------------------------------------------------
# 4H:
#   - Market structure
#   - Swing highs / lows
#   - BOS
#   - CHoCH
#   - Trend alignment
#   - Liquidity sweep
#   - Displacement
#   - Pullback / retest
#   - Price action
#
# 1H:
#   - Trend confirmation
#   - Momentum confirmation
#   - Candle confirmation
#   - Retest confirmation
#
# RISK:
#   - TP = fixed 1%
#   - ATR + structure based SL
#   - SL bounded between 0.40% and 2.50%
#   - Minimum R:R = 1.20
#   - Entry window = 0.30%
#
# PROTECTION:
#   - No signal in mixed/ranging structure
#   - No conflicting BOS
#   - No blocked TP
#   - No duplicate signal
#   - One active signal per coin
#   - Signal expiration
#   - Persistent state
#
# IMPORTANT:
# This bot generates signals only.
# It does NOT place exchange orders.
# ============================================================


# ============================================================
# TELEGRAM
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    "2090120004"
)


# ============================================================
# LBANK
# ============================================================

LBANK_KLINE_URL = (
    "https://api.lbkex.com/v2/kline.do"
)

LBANK_TICKER_URL = (
    "https://api.lbkex.com/v2/ticker.do"
)


# ============================================================
# COINS
# ============================================================

COINS = {
    "BTC": "btc_usdt",
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
}


# ============================================================
# TIMEFRAMES
# ============================================================

MAIN_TIMEFRAME = "hour4"
ENTRY_TIMEFRAME = "hour1"

CANDLE_LIMIT_4H = 500
CANDLE_LIMIT_1H = 500


# ============================================================
# EMA
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200


# ============================================================
# RSI
# ============================================================

RSI_PERIOD = 14


# ============================================================
# ATR
# ============================================================

ATR_PERIOD = 14
ATR_MULTIPLIER = 1.20


# ============================================================
# VOLUME
# ============================================================

VOLUME_LOOKBACK = 20


# ============================================================
# STRUCTURE
# ============================================================

STRUCTURE_LOOKBACK = 100

SWING_LEFT = 2
SWING_RIGHT = 2

BOS_LOOKBACK = 30

# Minimum distance between meaningful swing points.
MIN_SWING_DISTANCE_ATR = 0.20


# ============================================================
# LIQUIDITY
# ============================================================

LIQUIDITY_LOOKBACK = 40

LIQUIDITY_BUFFER_ATR = 0.15


# ============================================================
# PRICE ACTION
# ============================================================

MIN_BODY_RATIO = 0.45

MIN_DISPLACEMENT_ATR = 0.80


# ============================================================
# SIGNAL SCORE
# ============================================================

# Structure itself must be valid.
#
# Additional score:
#
# Trend              1
# BOS/CHoCH          1
# Liquidity sweep    1
# Displacement       1
# Pullback/retest    1
# Candle              1
# Volume              1
#
# Maximum = 7
#

MIN_SCORE_4H = 5

MIN_SCORE_1H = 2


# ============================================================
# ENTRY
# ============================================================

ENTRY_MAX_DISTANCE = 0.003
# 0.30%


# ============================================================
# TP
# ============================================================

TP_PERCENT = 0.01
# 1.00%


# ============================================================
# SL
# ============================================================

MIN_SL_PERCENT = 0.004
# 0.40%

MAX_SL_PERCENT = 0.025
# 2.50%


# ============================================================
# R:R
# ============================================================

MIN_RISK_REWARD = 1.20


# ============================================================
# SIGNAL EXPIRATION
# ============================================================

SIGNAL_EXPIRATION_HOURS = 8


# ============================================================
# STATE
# ============================================================

STATE_FILE = "state.json"


# ============================================================
# REQUEST SETTINGS
# ============================================================

REQUEST_TIMEOUT = 20

MAX_RETRIES = 3


# ============================================================
# UTILITIES
# ============================================================

def safe_float(value, default=None):

    try:
        return float(value)
    except Exception:
        return default


def now_ts():

    return int(time.time())


def percentage_distance(a, b):

    if a == 0:
        return float("inf")

    return abs(a - b) / a


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TOKEN:
        print("Telegram token missing.")
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    try:

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
            },
            timeout=REQUEST_TIMEOUT,
        )

        print(
            "Telegram:",
            response.status_code
        )

        if response.status_code != 200:

            print(
                response.text
            )

            return False

        return True

    except Exception as e:

        print(
            "Telegram error:",
            e
        )

        return False


# ============================================================
# STATE
# ============================================================

def empty_state():

    return {
        "last_signals": {},
        "active_trades": {},
        "completed_trades": [],
    }


def load_state():

    if not os.path.exists(
        STATE_FILE
    ):

        return empty_state()

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            state = json.load(f)

        state.setdefault(
            "last_signals",
            {}
        )

        state.setdefault(
            "active_trades",
            {}
        )

        state.setdefault(
            "completed_trades",
            []
        )

        return state

    except Exception as e:

        print(
            "State load error:",
            e
        )

        return empty_state()


def save_state(state):

    temp_file = (
        STATE_FILE + ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            state,
            f,
            indent=2,
        )

    os.replace(
        temp_file,
        STATE_FILE,
    )


# ============================================================
# HTTP GET WITH RETRY
# ============================================================

def http_get(
    url,
    params,
):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            return response.json()

        except Exception as e:

            last_error = e

            print(
                f"HTTP attempt "
                f"{attempt}/{MAX_RETRIES} "
                f"failed: {e}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    attempt
                )

    raise RuntimeError(
        f"HTTP request failed: "
        f"{last_error}"
    )


# ============================================================
# LBANK KLINES
# ============================================================

def get_candles(
    symbol,
    timeframe,
    limit,
):

    print(
        f"Getting {symbol} "
        f"{timeframe} candles..."
    )

    hours = 1

    if timeframe == "hour4":
        hours = 4

    start_time = (
        now_ts()
        - (
            limit
            * hours
            * 60
            * 60
        )
    )

    result = http_get(
        LBANK_KLINE_URL,
        {
            "symbol": symbol,
            "size": limit,
            "type": timeframe,
            "time": start_time,
        },
    )

    if str(
        result.get("result")
    ).lower() != "true":

        raise RuntimeError(
            f"LBank API error: "
            f"{result}"
        )

    raw_data = result.get(
        "data",
        [],
    )

    if not raw_data:

        raise RuntimeError(
            f"No candle data for "
            f"{symbol}"
        )

    candles = []

    for item in raw_data:

        if len(item) < 6:
            continue

        candles.append(
            {
                "time": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            }
        )

    candles.sort(
        key=lambda x: x["time"]
    )

    print(
        f"{symbol}: received "
        f"{len(candles)} candles"
    )

    return candles


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(symbol):

    result = http_get(
        LBANK_TICKER_URL,
        {
            "symbol": symbol,
        },
    )

    if str(
        result.get("result")
    ).lower() != "true":

        raise RuntimeError(
            f"LBank ticker error: "
            f"{result}"
        )

    data = result.get(
        "data",
        [],
    )

    if not data:

        raise RuntimeError(
            f"No ticker data for "
            f"{symbol}"
        )

    ticker = data[0]

    ticker_data = ticker.get(
        "ticker",
        ticker,
    )

    price = (
        ticker_data.get("latest")
    )

    if price is None:

        price = (
            ticker_data.get("last")
        )

    if price is None:

        raise RuntimeError(
            f"Price not found: "
            f"{result}"
        )

    return float(price)


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(candles):

    if len(candles) < 3:

        raise RuntimeError(
            "Not enough candles."
        )

    # Last candle can still be forming.
    return candles[:-1]


# ============================================================
# EMA
# ============================================================

def ema_series(
    values,
    period,
):

    if len(values) < period:

        return []

    multiplier = (
        2 / (period + 1)
    )

    first = (
        sum(
            values[:period]
        )
        / period
    )

    result = (
        [None] * (period - 1)
    )

    result.append(
        first
    )

    previous = first

    for price in values[period:]:

        current = (
            (
                price
                - previous
            )
            * multiplier
            + previous
        )

        result.append(
            current
        )

        previous = current

    return result


# ============================================================
# RSI
# ============================================================

def rsi_series(
    closes,
    period=14,
):

    if len(closes) <= period:

        return []

    gains = []
    losses = []

    for i in range(
        1,
        len(closes),
    ):

        change = (
            closes[i]
            - closes[i - 1]
        )

        gains.append(
            max(change, 0)
        )

        losses.append(
            max(-change, 0)
        )

    avg_gain = (
        sum(gains[:period])
        / period
    )

    avg_loss = (
        sum(losses[:period])
        / period
    )

    result = (
        [None] * period
    )

    if avg_loss == 0:

        result.append(
            100.0
        )

    else:

        rs = (
            avg_gain
            / avg_loss
        )

        result.append(
            100
            - (
                100
                / (1 + rs)
            )
        )

    for i in range(
        period,
        len(gains),
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

            value = 100.0

        else:

            rs = (
                avg_gain
                / avg_loss
            )

            value = (
                100
                - (
                    100
                    / (1 + rs)
                )
            )

        result.append(
            value
        )

    return result


# ============================================================
# ATR
# ============================================================

def atr_series(
    candles,
    period=14,
):

    if len(candles) <= period:

        return []

    true_ranges = []

    for i in range(
        1,
        len(candles),
    ):

        high = candles[i]["high"]
        low = candles[i]["low"]

        previous_close = (
            candles[i - 1]["close"]
        )

        tr = max(
            high - low,
            abs(
                high
                - previous_close
            ),
            abs(
                low
                - previous_close
            ),
        )

        true_ranges.append(
            tr
        )

    first_atr = (
        sum(
            true_ranges[:period]
        )
        / period
    )

    result = (
        [None] * period
    )

    result.append(
        first_atr
    )

    previous = first_atr

    for tr in true_ranges[period:]:

        current = (
            (
                previous
                * (period - 1)
                + tr
            )
            / period
        )

        result.append(
            current
        )

        previous = current

    return result


# ============================================================
# SWING POINTS
# ============================================================

def find_swing_highs(
    candles,
    left=2,
    right=2,
):

    swings = []

    start = left

    end = (
        len(candles)
        - right
    )

    for i in range(
        start,
        end,
    ):

        high = candles[i]["high"]

        left_values = [
            candles[j]["high"]
            for j in range(
                i - left,
                i,
            )
        ]

        right_values = [
            candles[j]["high"]
            for j in range(
                i + 1,
                i + right + 1,
            )
        ]

        if (
            high >= max(left_values)
            and
            high >= max(right_values)
        ):

            swings.append(
                {
                    "index": i,
                    "price": high,
                    "time":
                        candles[i]["time"],
                }
            )

    return swings


def find_swing_lows(
    candles,
    left=2,
    right=2,
):

    swings = []

    start = left

    end = (
        len(candles)
        - right
    )

    for i in range(
        start,
        end,
    ):

        low = candles[i]["low"]

        left_values = [
            candles[j]["low"]
            for j in range(
                i - left,
                i,
            )
        ]

        right_values = [
            candles[j]["low"]
            for j in range(
                i + 1,
                i + right + 1,
            )
        ]

        if (
            low <= min(left_values)
            and
            low <= min(right_values)
        ):

            swings.append(
                {
                    "index": i,
                    "price": low,
                    "time":
                        candles[i]["time"],
                }
            )

    return swings


# ============================================================
# STRUCTURE CLASSIFICATION
# ============================================================

def classify_structure(
    candles,
):

    if len(candles) < 80:

        return {
            "valid": False,
            "direction": None,
            "reason": "NOT_ENOUGH_DATA",
        }

    swing_highs = (
        find_swing_highs(
            candles,
            SWING_LEFT,
            SWING_RIGHT,
        )
    )

    swing_lows = (
        find_swing_lows(
            candles,
            SWING_LEFT,
            SWING_RIGHT,
        )
    )

    if (
        len(swing_highs) < 4
        or len(swing_lows) < 4
    ):

        return {
            "valid": False,
            "direction": None,
            "reason":
                "NOT_ENOUGH_SWINGS",
        }

    recent_highs = swing_highs[
        -4:
    ]

    recent_lows = swing_lows[
        -4:
    ]

    h1 = recent_highs[-1]["price"]
    h2 = recent_highs[-2]["price"]

    l1 = recent_lows[-1]["price"]
    l2 = recent_lows[-2]["price"]

    higher_high = h1 > h2
    higher_low = l1 > l2

    lower_high = h1 < h2
    lower_low = l1 < l2

    bullish = (
        higher_high
        and higher_low
    )

    bearish = (
        lower_high
        and lower_low
    )

    if bullish:

        direction = "LONG"

    elif bearish:

        direction = "SHORT"

    else:

        return {
            "valid": False,
            "direction": None,
            "reason":
                "STRUCTURE_MIXED_OR_RANGE",
            "swing_highs":
                swing_highs,
            "swing_lows":
                swing_lows,
        }

    return {
        "valid": True,
        "direction": direction,
        "reason": "STRUCTURE_VALID",
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
    }


# ============================================================
# BOS
# ============================================================

def detect_bos(
    candles,
    structure,
):

    if not structure["valid"]:

        return {
            "bullish": False,
            "bearish": False,
        }

    direction = (
        structure["direction"]
    )

    swing_highs = (
        structure["swing_highs"]
    )

    swing_lows = (
        structure["swing_lows"]
    )

    current_close = (
        candles[-1]["close"]
    )

    bullish_bos = False
    bearish_bos = False

    if swing_highs:

        previous_high = (
            swing_highs[-1]["price"]
        )

        bullish_bos = (
            current_close
            > previous_high
        )

    if swing_lows:

        previous_low = (
            swing_lows[-1]["price"]
        )

        bearish_bos = (
            current_close
            < previous_low
        )

    if (
        direction == "LONG"
        and bearish_bos
        and not bullish_bos
    ):

        return {
            "bullish": False,
            "bearish": True,
            "valid": False,
            "reason":
                "BOS_CONFLICTS_WITH_STRUCTURE",
        }

    if (
        direction == "SHORT"
        and bullish_bos
        and not bearish_bos
    ):

        return {
            "bullish": True,
            "bearish": False,
            "valid": False,
            "reason":
                "BOS_CONFLICTS_WITH_STRUCTURE",
        }

    return {
        "bullish": bullish_bos,
        "bearish": bearish_bos,
        "valid": True,
        "reason": "BOS_OK",
    }


# ============================================================
# CHoCH
#
# CHoCH is treated as an early structural transition,
# not as an automatic trade trigger.
# ============================================================

def detect_choch(
    candles,
    structure,
):

    if not structure["valid"]:

        return {
            "bullish": False,
            "bearish": False,
        }

    swing_highs = (
        structure["swing_highs"]
    )

    swing_lows = (
        structure["swing_lows"]
    )

    close = candles[-1]["close"]

    bullish = False
    bearish = False

    if (
        structure["direction"]
        == "LONG"
    ):

        if swing_lows:

            key_low = (
                swing_lows[-1]["price"]
            )

            bullish = (
                close > key_low
            )

    else:

        if swing_highs:

            key_high = (
                swing_highs[-1]["price"]
            )

            bearish = (
                close < key_high
            )

    return {
        "bullish": bullish,
        "bearish": bearish,
    }


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(
    candles,
    direction,
    atr,
):

    if len(candles) < (
        LIQUIDITY_LOOKBACK + 2
    ):

        return False

    current = candles[-1]

    previous = candles[
        -LIQUIDITY_LOOKBACK - 1:
        -1
    ]

    if direction == "LONG":

        liquidity_low = min(
            c["low"]
            for c in previous
        )

        swept = (
            current["low"]
            < (
                liquidity_low
                - (
                    LIQUIDITY_BUFFER_ATR
                    * atr
                )
            )
        )

        reclaimed = (
            current["close"]
            > liquidity_low
        )

        return (
            swept
            and reclaimed
        )

    liquidity_high = max(
        c["high"]
        for c in previous
    )

    swept = (
        current["high"]
        > (
            liquidity_high
            + (
                LIQUIDITY_BUFFER_ATR
                * atr
            )
        )
    )

    rejected = (
        current["close"]
        < liquidity_high
    )

    return (
        swept
        and rejected
    )


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(
    candle,
    atr,
    direction,
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

    large_enough = (
        candle_range
        >= (
            atr
            * MIN_DISPLACEMENT_ATR
        )
    )

    if direction == "LONG":

        directional = (
            candle["close"]
            > candle["open"]
        )

    else:

        directional = (
            candle["close"]
            < candle["open"]
        )

    return (
        large_enough
        and body_ratio >= 0.55
        and directional
    )


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def candle_confirmation(
    candle,
):

    o = candle["open"]
    h = candle["high"]
    l = candle["low"]
    c = candle["close"]

    candle_range = h - l

    if candle_range <= 0:

        return {
            "bullish": False,
            "bearish": False,
            "quality": 0,
        }

    body = abs(
        c - o
    )

    body_ratio = (
        body
        / candle_range
    )

    upper_wick = (
        h - max(o, c)
    )

    lower_wick = (
        min(o, c) - l
    )

    bullish = (
        c > o
        and body_ratio
        >= MIN_BODY_RATIO
        and c
        >= l + (
            candle_range
            * 0.60
        )
    )

    bearish = (
        c < o
        and body_ratio
        >= MIN_BODY_RATIO
        and c
        <= l + (
            candle_range
            * 0.40
        )
    )

    # Additional rejection patterns.

    bullish_rejection = (
        lower_wick
        > body * 1.2
        and c > o
    )

    bearish_rejection = (
        upper_wick
        > body * 1.2
        and c < o
    )

    return {
        "bullish":
            bullish
            or bullish_rejection,

        "bearish":
            bearish
            or bearish_rejection,

        "quality":
            body_ratio,
    }


# ============================================================
# PULLBACK / RETEST
# ============================================================

def detect_pullback(
    candles,
    direction,
    ema20,
    atr,
):

    current = candles[-1]

    close = current["close"]

    high = current["high"]

    low = current["low"]

    if direction == "LONG":

        near_ema = (
            low
            <= ema20
            + (
                atr * 0.25
            )
        )

        reclaimed = (
            close > ema20
        )

        return (
            near_ema
            and reclaimed
        )

    near_ema = (
        high
        >= ema20
        - (
            atr * 0.25
        )
    )

    rejected = (
        close < ema20
    )

    return (
        near_ema
        and rejected
    )


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

def volume_confirmation(
    candles,
):

    if len(candles) < (
        VOLUME_LOOKBACK + 1
    ):

        return False

    current_volume = (
        candles[-1]["volume"]
    )

    average_volume = (
        sum(
            c["volume"]
            for c in candles[
                -VOLUME_LOOKBACK - 1:
                -1
            ]
        )
        / VOLUME_LOOKBACK
    )

    return (
        current_volume
        >= average_volume
    )


# ============================================================
# 4H ANALYSIS
# ============================================================

def analyze_4h(
    candles,
):

    if len(candles) < 250:

        return {
            "valid": False,
            "reason":
                "NOT_ENOUGH_4H_DATA",
        }

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = ema_series(
        closes,
        EMA_FAST,
    )

    ema50 = ema_series(
        closes,
        EMA_MID,
    )

    ema200 = ema_series(
        closes,
        EMA_SLOW,
    )

    rsi_values = rsi_series(
        closes,
        RSI_PERIOD,
    )

    atr_values = atr_series(
        candles,
        ATR_PERIOD,
    )

    i = len(candles) - 1

    e20 = ema20[i]
    e50 = ema50[i]
    e200 = ema200[i]

    rsi = rsi_values[-1]
    previous_rsi = rsi_values[-2]

    atr = atr_values[-1]

    current = candles[-1]

    if any(
        value is None
        for value in [
            e20,
            e50,
            e200,
            rsi,
            previous_rsi,
            atr,
        ]
    ):

        return {
            "valid": False,
            "reason":
                "INDICATOR_NOT_READY",
        }

    structure = (
        classify_structure(
            candles
        )
    )

    if not structure["valid"]:

        return {
            "valid": False,
            "reason":
                structure["reason"],
        }

    direction = (
        structure["direction"]
    )

    # --------------------------------------------------------
    # EMA TREND
    # --------------------------------------------------------

    if direction == "LONG":

        trend_ok = (
            current["close"] > e200
            and e20 > e50
        )

        rsi_ok = (
            rsi > 50
            and rsi < 75
            and rsi >= previous_rsi
        )

    else:

        trend_ok = (
            current["close"] < e200
            and e20 < e50
        )

        rsi_ok = (
            rsi < 50
            and rsi > 25
            and rsi <= previous_rsi
        )

    # --------------------------------------------------------
    # BOS
    # --------------------------------------------------------

    bos = detect_bos(
        candles,
        structure,
    )

    if not bos["valid"]:

        return {
            "valid": False,
            "reason":
                bos["reason"],
        }

    # --------------------------------------------------------
    # CHoCH
    # --------------------------------------------------------

    choch = detect_choch(
        candles,
        structure,
    )

    # CHoCH is supportive, not mandatory.
    choch_ok = (
        (
            direction == "LONG"
            and choch["bullish"]
        )
        or
        (
            direction == "SHORT"
            and choch["bearish"]
        )
    )

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    liquidity = (
        detect_liquidity_sweep(
            candles,
            direction,
            atr,
        )
    )

    # --------------------------------------------------------
    # DISPLACEMENT
    # --------------------------------------------------------

    displacement = (
        detect_displacement(
            current,
            atr,
            direction,
        )
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    pullback = (
        detect_pullback(
            candles,
            direction,
            e20,
            atr,
        )
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    candle = (
        candle_confirmation(
            current
        )
    )

    candle_ok = (
        (
            direction == "LONG"
            and candle["bullish"]
        )
        or
        (
            direction == "SHORT"
            and candle["bearish"]
        )
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ok = (
        volume_confirmation(
            candles
        )
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    volatility_ok = (
        atr / current["close"]
        >= 0.0015
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = (
        int(trend_ok)
        + int(bos["bullish"]
              if direction == "LONG"
              else bos["bearish"])
        + int(liquidity)
        + int(displacement)
        + int(pullback)
        + int(candle_ok)
        + int(volume_ok)
    )

    # CHoCH can replace liquidity/displacement
    # as a supporting confirmation.
    if choch_ok and score < 7:
        score += 1

    # Cap score at 7.
    score = min(
        score,
        7
    )

    # --------------------------------------------------------
    # HARD FILTERS
    # --------------------------------------------------------

    if not trend_ok:

        return {
            "valid": False,
            "reason":
                "TREND_NOT_ALIGNED",
            "score": score,
            "direction": direction,
        }

    if not candle_ok:

        return {
            "valid": False,
            "reason":
                "PRICE_ACTION_NOT_CONFIRMED",
            "score": score,
            "direction": direction,
        }

    if not pullback:

        return {
            "valid": False,
            "reason":
                "NO_PULLBACK_OR_RETEST",
            "score": score,
            "direction": direction,
        }

    if not volatility_ok:

        return {
            "valid": False,
            "reason":
                "VOLATILITY_TOO_LOW",
            "score": score,
            "direction": direction,
        }

    if score < MIN_SCORE_4H:

        return {
            "valid": False,
            "reason":
                f"SCORE_TOO_LOW_{score}_OF_7",
            "score": score,
            "direction": direction,
        }

    # --------------------------------------------------------
    # STRUCTURE LEVELS
    # --------------------------------------------------------

    recent_highs = (
        structure["swing_highs"]
        [-6:]
    )

    recent_lows = (
        structure["swing_lows"]
        [-6:]
    )

    structure_high = max(
        x["price"]
        for x in recent_highs
    )

    structure_low = min(
        x["price"]
        for x in recent_lows
    )

    return {
        "valid": True,

        "direction":
            direction,

        "score":
            score,

        "entry":
            current["close"],

        "atr":
            atr,

        "rsi":
            rsi,

        "ema20":
            e20,

        "ema50":
            e50,

        "ema200":
            e200,

        "bos":
            bos,

        "choch":
            choch_ok,

        "liquidity_sweep":
            liquidity,

        "displacement":
            displacement,

        "pullback":
            pullback,

        "candle":
            candle_ok,

        "volume":
            volume_ok,

        "structure_low":
            structure_low,

        "structure_high":
            structure_high,

        "candle_time":
            current["time"],

        "candles":
            candles,
    }


# ============================================================
# 1H CONFIRMATION
# ============================================================

def analyze_1h(
    candles,
    direction,
):

    if len(candles) < 250:

        return {
            "valid": False,
            "score": 0,
            "reason":
                "NOT_ENOUGH_1H_DATA",
        }

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = ema_series(
        closes,
        EMA_FAST,
    )

    ema50 = ema_series(
        closes,
        EMA_MID,
    )

    ema200 = ema_series(
        closes,
        EMA_SLOW,
    )

    rsi_values = rsi_series(
        closes,
        RSI_PERIOD,
    )

    atr_values = atr_series(
        candles,
        ATR_PERIOD,
    )

    i = len(candles) - 1

    current = candles[i]

    e20 = ema20[i]
    e50 = ema50[i]
    e200 = ema200[i]

    rsi = rsi_values[-1]
    previous_rsi = rsi_values[-2]

    atr = atr_values[-1]

    if any(
        x is None
        for x in [
            e20,
            e50,
            e200,
            rsi,
            previous_rsi,
            atr,
        ]
    ):

        return {
            "valid": False,
            "score": 0,
            "reason":
                "INDICATOR_NOT_READY",
        }

    candle = (
        candle_confirmation(
            current
        )
    )

    score = 0

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if direction == "LONG":

        trend = (
            current["close"] > e200
            and e20 > e50
        )

        momentum = (
            rsi > 50
            and rsi >= previous_rsi
        )

        candle_ok = (
            candle["bullish"]
        )

    else:

        trend = (
            current["close"] < e200
            and e20 < e50
        )

        momentum = (
            rsi < 50
            and rsi <= previous_rsi
        )

        candle_ok = (
            candle["bearish"]
        )

    score += int(trend)
    score += int(momentum)
    score += int(candle_ok)

    # --------------------------------------------------------
    # 1H RETEST
    # --------------------------------------------------------

    if direction == "LONG":

        retest = (
            current["low"]
            <= e20 + atr * 0.30
            and
            current["close"] > e20
        )

    else:

        retest = (
            current["high"]
            >= e20 - atr * 0.30
            and
            current["close"] < e20
        )

    score += int(retest)

    # Maximum score = 4.
    valid = (
        score >= MIN_SCORE_1H
        and trend
        and momentum
    )

    if not valid:

        return {
            "valid": False,
            "score": score,
            "reason":
                "1H_CONFIRMATION_FAILED",
            "rsi": rsi,
        }

    return {
        "valid": True,
        "score": score,
        "rsi": rsi,
        "trend": trend,
        "momentum": momentum,
        "candle": candle_ok,
        "retest": retest,
        "atr": atr,
    }


# ============================================================
# SIGNAL STRENGTH
# ============================================================

def signal_strength(
    score_4h,
    score_1h,
):

    if (
        score_4h >= 7
        and score_1h >= 3
    ):

        return "🔥 VERY STRONG"

    if (
        score_4h >= 6
        and score_1h >= 2
    ):

        return "💪 STRONG"

    return "🟡 VALID"


# ============================================================
# STOP LOSS
# ============================================================

def calculate_stop_loss(
    signal,
):

    entry = signal["entry"]

    atr = signal["atr"]

    direction = (
        signal["direction"]
    )

    structure_low = (
        signal["structure_low"]
    )

    structure_high = (
        signal["structure_high"]
    )

    if direction == "LONG":

        atr_stop = (
            entry
            - (
                ATR_MULTIPLIER
                * atr
            )
        )

        structure_stop = (
            structure_low
            - (
                atr
                * 0.15
            )
        )

        raw_sl = min(
            atr_stop,
            structure_stop,
        )

        risk = (
            entry
            - raw_sl
        )

    else:

        atr_stop = (
            entry
            + (
                ATR_MULTIPLIER
                * atr
            )
        )

        structure_stop = (
            structure_high
            + (
                atr
                * 0.15
            )
        )

        raw_sl = max(
            atr_stop,
            structure_stop,
        )

        risk = (
            raw_sl
            - entry
        )

    min_risk = (
        entry
        * MIN_SL_PERCENT
    )

    max_risk = (
        entry
        * MAX_SL_PERCENT
    )

    # Bound risk.
    risk = max(
        risk,
        min_risk,
    )

    risk = min(
        risk,
        max_risk,
    )

    if direction == "LONG":

        sl = entry - risk

    else:

        sl = entry + risk

    risk_percent = (
        risk
        / entry
        * 100
    )

    return {
        "sl":
            sl,

        "risk":
            risk,

        "risk_percent":
            risk_percent,
    }


# ============================================================
# TAKE PROFIT
# ============================================================

def calculate_tp(
    entry,
    direction,
):

    if direction == "LONG":

        return (
            entry
            * (
                1
                + TP_PERCENT
            )
        )

    return (
        entry
        * (
            1
            - TP_PERCENT
        )
    )


# ============================================================
# TP PATH
# ============================================================

def check_tp_path(
    signal,
    tp,
):

    direction = (
        signal["direction"]
    )

    entry = signal["entry"]

    atr = signal["atr"]

    candles = signal["candles"]

    buffer = (
        atr
        * 0.15
    )

    swing_highs = (
        find_swing_highs(
            candles,
            SWING_LEFT,
            SWING_RIGHT,
        )
    )

    swing_lows = (
        find_swing_lows(
            candles,
            SWING_LEFT,
            SWING_RIGHT,
        )
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == "LONG":

        blocking = []

        for swing in swing_highs:

            level = swing["price"]

            if (
                level > entry
                and
                level
                <= tp + buffer
            ):

                blocking.append(
                    level
                )

        if blocking:

            level = min(
                blocking
            )

            return {
                "valid": False,
                "type":
                    "RESISTANCE",
                "level":
                    level,
            }

        return {
            "valid": True,
            "type": None,
            "level": None,
        }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    blocking = []

    for swing in swing_lows:

        level = swing["price"]

        if (
            level < entry
            and
            level
            >= tp - buffer
        ):

            blocking.append(
                level
            )

    if blocking:

        level = max(
            blocking
        )

        return {
            "valid": False,
            "type":
                "SUPPORT",
            "level":
                level,
        }

    return {
        "valid": True,
        "type": None,
        "level": None,
    }


# ============================================================
# FINAL RISK VALIDATION
# ============================================================

def calculate_risk_levels(
    signal,
):

    entry = signal["entry"]

    direction = (
        signal["direction"]
    )

    stop = (
        calculate_stop_loss(
            signal
        )
    )

    sl = stop["sl"]

    risk_percent = (
        stop["risk_percent"]
    )

    tp = calculate_tp(
        entry,
        direction,
    )

    risk = abs(
        entry - sl
    )

    reward = abs(
        tp - entry
    )

    if risk <= 0:

        return {
            "valid": False,
            "reason":
                "INVALID_SL",
        }

    rr = (
        reward
        / risk
    )

    if rr < MIN_RISK_REWARD:

        return {
            "valid": False,
            "reason":
                f"RR_TOO_LOW_1:{rr:.2f}",
        }

    path = (
        check_tp_path(
            signal,
            tp,
        )
    )

    if not path["valid"]:

        return {
            "valid": False,

            "reason":
                (
                    f"TP_BLOCKED_BY_"
                    f"{path['type']}_"
                    f"{path['level']:.8f}"
                ),

            "blocking_level":
                path["level"],

            "blocking_type":
                path["type"],
        }

    return {
        "valid": True,

        "entry":
            entry,

        "sl":
            sl,

        "tp":
            tp,

        "risk_percent":
            risk_percent,

        "rr":
            rr,

        "blocking_level":
            None,

        "blocking_type":
            None,
    }


# ============================================================
# ENTRY WINDOW
# ============================================================

def entry_is_valid(
    entry,
    current_price,
):

    distance = (
        percentage_distance(
            entry,
            current_price,
        )
    )

    return (
        distance
        <= ENTRY_MAX_DISTANCE
    )


# ============================================================
# EXPIRATION
# ============================================================

def signal_is_expired(
    candle_time,
):

    age = (
        now_ts()
        - candle_time
    )

    return (
        age
        > SIGNAL_EXPIRATION_HOURS
        * 3600
    )


# ============================================================
# ACTIVE TRADES
# ============================================================

def check_active_trades(
    state,
):

    active = state.get(
        "active_trades",
        {},
    )

    if not active:

        return

    finished = []

    for trade_id, trade in list(
        active.items()
    ):

        try:

            current_price = (
                get_current_price(
                    trade[
                        "lbank_symbol"
                    ]
                )
            )

        except Exception as e:

            print(
                f"{trade['symbol']}: "
                f"active price error: "
                f"{e}"
            )

            continue

        direction = (
            trade["direction"]
        )

        entry = trade["entry"]

        tp = trade["tp"]

        sl = trade["sl"]

        result = None

        if direction == "LONG":

            if current_price >= tp:

                result = "TP"

            elif current_price <= sl:

                result = "SL"

        else:

            if current_price <= tp:

                result = "TP"

            elif current_price >= sl:

                result = "SL"

        if result is None:

            continue

        if result == "TP":

            pnl_percent = (
                abs(tp - entry)
                / entry
                * 100
            )

        else:

            pnl_percent = -(
                abs(sl - entry)
                / entry
                * 100
            )

        trade["result"] = result

        trade["exit_price"] = (
            current_price
        )

        trade["pnl_percent"] = (
            pnl_percent
        )

        trade["closed_at"] = (
            now_ts()
        )

        state.setdefault(
            "completed_trades",
            [],
        ).append(
            trade
        )

        finished.append(
            trade_id
        )

        emoji = (
            "✅"
            if result == "TP"
            else "❌"
        )

        message = (
            f"{emoji} SCORE HUNTER PRO v4\n\n"
            f"TRADE CLOSED\n\n"
            f"💰 {trade['symbol']}USDT\n"
            f"📊 {direction}\n\n"
            f"💵 Entry: {entry:.8f}\n"
            f"🏁 Exit: {current_price:.8f}\n\n"
            f"📌 Result: {result}\n"
            f"📈 P/L: {pnl_percent:+.2f}%\n\n"
            f"🎯 TP: 1.00%\n"
            f"🏦 LBank"
        )

        send_telegram(
            message
        )

    for trade_id in finished:

        del active[
            trade_id
        ]

    state[
        "active_trades"
    ] = active


# ============================================================
# STATISTICS
# ============================================================

def get_statistics(
    state,
):

    trades = state.get(
        "completed_trades",
        [],
    )

    if not trades:

        return None

    total = len(
        trades
    )

    wins = sum(
        1
        for trade in trades
        if trade.get("result")
        == "TP"
    )

    losses = sum(
        1
        for trade in trades
        if trade.get("result")
        == "SL"
    )

    decided = (
        wins + losses
    )

    win_rate = (
        wins
        / decided
        * 100
        if decided
        else 0
    )

    total_pnl = sum(
        trade.get(
            "pnl_percent",
            0,
        )
        for trade in trades
    )

    return {
        "total":
            total,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            win_rate,

        "pnl":
            total_pnl,
    }


def send_statistics_if_needed(
    state,
):

    stats = get_statistics(
        state
    )

    if not stats:

        return

    if (
        stats["total"] == 0
        or
        stats["total"] % 10 != 0
    ):

        return

    message = (
        "📊 SCORE HUNTER PRO v4\n\n"
        "STATISTICS\n\n"
        f"📌 Closed: "
        f"{stats['total']}\n"
        f"✅ TP: "
        f"{stats['wins']}\n"
        f"❌ SL: "
        f"{stats['losses']}\n"
        f"🎯 Win rate: "
        f"{stats['win_rate']:.1f}%\n"
        f"📈 P/L: "
        f"{stats['pnl']:+.2f}%\n\n"
        "🎯 TP target: 1.00%\n"
        "📊 4H + 1H\n"
        "🏦 LBank"
    )

    send_telegram(
        message
    )


# ============================================================
# DIAGNOSTIC REPORT
# ============================================================

def print_analysis_report(
    symbol,
    analysis,
):

    if not analysis.get(
        "valid"
    ):

        print(
            f"{symbol}: "
            f"❌ "
            f"{analysis.get('reason')}"
        )

        if "score" in analysis:

            print(
                f"{symbol}: "
                f"diagnostic score = "
                f"{analysis['score']}/7"
            )

        return

    print(
        f"{symbol}: "
        f"✅ 4H structure valid"
    )

    print(
        f"{symbol}: "
        f"direction="
        f"{analysis['direction']}"
    )

    print(
        f"{symbol}: "
        f"score="
        f"{analysis['score']}/7"
    )

    print(
        f"{symbol}: "
        f"BOS="
        f"{analysis['bos']['reason']}"
    )

    print(
        f"{symbol}: "
        f"CHoCH="
        f"{analysis['choch']}"
    )

    print(
        f"{symbol}: "
        f"liquidity="
        f"{analysis['liquidity_sweep']}"
    )

    print(
        f"{symbol}: "
        f"displacement="
        f"{analysis['displacement']}"
    )

    print(
        f"{symbol}: "
        f"pullback="
        f"{analysis['pullback']}"
    )

    print(
        f"{symbol}: "
        f"candle="
        f"{analysis['candle']}"
    )

    print(
        f"{symbol}: "
        f"volume="
        f"{analysis['volume']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🟢 SCORE HUNTER PRO v4"
    )

    print(
        "🧠 MARKET STRUCTURE + "
        "PRICE ACTION + LIQUIDITY"
    )

    print(
        "📊 Main structure: 4H"
    )

    print(
        "📱 Entry confirmation: 1H"
    )

    print(
        "🔎 BOS enabled"
    )

    print(
        "🔄 CHoCH enabled"
    )

    print(
        "💧 Liquidity sweep enabled"
    )

    print(
        "↩️ Pullback / retest enabled"
    )

    print(
        "🕯 Price action enabled"
    )

    print(
        f"🎯 Fixed TP: "
        f"{TP_PERCENT * 100:.2f}%"
    )

    print(
        f"🔒 Entry window: "
        f"{ENTRY_MAX_DISTANCE * 100:.2f}%"
    )

    print(
        f"🛑 ATR multiplier: "
        f"{ATR_MULTIPLIER}"
    )

    print(
        f"⚖️ Minimum R:R: "
        f"1:{MIN_RISK_REWARD}"
    )

    print(
        f"⏳ Expiration: "
        f"{SIGNAL_EXPIRATION_HOURS}h"
    )

    state = load_state()

    # --------------------------------------------------------
    # ACTIVE TRADES
    # --------------------------------------------------------

    check_active_trades(
        state
    )

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    for symbol, lbank_symbol in (
        COINS.items()
    ):

        print(
            f"\n========== "
            f"{symbol} =========="
        )

        try:

            # =================================================
            # 4H
            # =================================================

            candles_4h = (
                get_candles(
                    lbank_symbol,
                    MAIN_TIMEFRAME,
                    CANDLE_LIMIT_4H,
                )
            )

            closed_4h = (
                get_closed_candles(
                    candles_4h
                )
            )

            latest_4h = (
                closed_4h[-1]
            )

            print(
                f"{symbol}: "
                f"latest closed 4H: "
                f"{latest_4h['time']}"
            )

            analysis_4h = (
                analyze_4h(
                    closed_4h
                )
            )

            print_analysis_report(
                symbol,
                analysis_4h,
            )

            if not analysis_4h.get(
                "valid"
            ):

                continue

            # =================================================
            # EXPIRATION
            # =================================================

            candle_time = (
                analysis_4h[
                    "candle_time"
                ]
            )

            if signal_is_expired(
                candle_time
            ):

                print(
                    f"{symbol}: "
                    f"❌ SIGNAL_EXPIRED"
                )

                continue

            # =================================================
            # 1H
            # =================================================

            candles_1h = (
                get_candles(
                    lbank_symbol,
                    ENTRY_TIMEFRAME,
                    CANDLE_LIMIT_1H,
                )
            )

            closed_1h = (
                get_closed_candles(
                    candles_1h
                )
            )

            confirmation = (
                analyze_1h(
                    closed_1h,
                    analysis_4h[
                        "direction"
                    ],
                )
            )

            print(
                f"{symbol}: "
                f"1H score="
                f"{confirmation['score']}/4"
            )

            if not confirmation.get(
                "valid"
            ):

                print(
                    f"{symbol}: "
                    f"❌ "
                    f"{confirmation.get('reason')}"
                )

                continue

            # =================================================
            # CURRENT PRICE
            # =================================================

            current_price = (
                get_current_price(
                    lbank_symbol
                )
            )

            entry = (
                analysis_4h[
                    "entry"
                ]
            )

            distance = (
                percentage_distance(
                    entry,
                    current_price,
                )
                * 100
            )

            print(
                f"{symbol}: "
                f"direction="
                f"{analysis_4h['direction']} "
                f"entry="
                f"{entry:.8f} "
                f"current="
                f"{current_price:.8f} "
                f"distance="
                f"{distance:.3f}%"
            )

            # =================================================
            # ENTRY WINDOW
            # =================================================

            if not entry_is_valid(
                entry,
                current_price,
            ):

                print(
                    f"{symbol}: "
                    f"❌ ENTRY_TOO_FAR"
                )

                continue

            # =================================================
            # DUPLICATE
            # =================================================

            direction = (
                analysis_4h[
                    "direction"
                ]
            )

            signal_id = (
                f"{symbol}_"
                f"{candle_time}_"
                f"{direction}"
            )

            previous_id = (
                state
                .get(
                    "last_signals",
                    {},
                )
                .get(symbol)
            )

            if (
                previous_id
                == signal_id
            ):

                print(
                    f"{symbol}: "
                    f"❌ DUPLICATE_SIGNAL"
                )

                continue

            # =================================================
            # ACTIVE TRADE
            # =================================================

            existing_trade = None

            for trade in (
                state
                .get(
                    "active_trades",
                    {},
                )
                .values()
            ):

                if (
                    trade["symbol"]
                    == symbol
                ):

                    existing_trade = (
                        trade
                    )

                    break

            if existing_trade:

                print(
                    f"{symbol}: "
                    f"❌ ACTIVE_TRADE_EXISTS"
                )

                continue

            # =================================================
            # RISK
            # =================================================

            signal = {
                **analysis_4h
            }

            levels = (
                calculate_risk_levels(
                    signal
                )
            )

            if not levels.get(
                "valid"
            ):

                print(
                    f"{symbol}: "
                    f"❌ "
                    f"{levels.get('reason')}"
                )

                if levels.get(
                    "blocking_level"
                ):

                    print(
                        f"{symbol}: "
                        f"blocking level="
                        f"{levels['blocking_level']}"
                    )

                continue

            sl = levels["sl"]
            tp = levels["tp"]
            rr = levels["rr"]
            risk_percent = (
                levels[
                    "risk_percent"
                ]
            )

            # =================================================
            # STRENGTH
            # =================================================

            score_4h = (
                analysis_4h[
                    "score"
                ]
            )

            score_1h = (
                confirmation[
                    "score"
                ]
            )

            strength = (
                signal_strength(
                    score_4h,
                    score_1h,
                )
            )

            # =================================================
            # FINAL SIGNAL
            # =================================================

            print(
                "\n"
                f"🚨 {symbol}: "
                f"FINAL VALID SIGNAL"
            )

            print(
                f"Direction: "
                f"{direction}"
            )

            print(
                f"4H Score: "
                f"{score_4h}/7"
            )

            print(
                f"1H Score: "
                f"{score_1h}/4"
            )

            print(
                f"Entry: "
                f"{entry:.8f}"
            )

            print(
                f"TP: "
                f"{tp:.8f}"
            )

            print(
                f"SL: "
                f"{sl:.8f}"
            )

            print(
                f"Risk: "
                f"{risk_percent:.2f}%"
            )

            print(
                f"R:R: "
                f"1:{rr:.2f}"
            )

            # =================================================
            # SAVE TRADE
            # =================================================

            trade = {

                "trade_id":
                    signal_id,

                "symbol":
                    symbol,

                "lbank_symbol":
                    lbank_symbol,

                "direction":
                    direction,

                "score_4h":
                    score_4h,

                "score_1h":
                    score_1h,

                "strength":
                    strength,

                "entry":
                    entry,

                "current_price_at_signal":
                    current_price,

                "tp":
                    tp,

                "sl":
                    sl,

                "risk_percent":
                    risk_percent,

                "rr":
                    rr,

                "candle_time":
                    candle_time,

                "created_at":
                    now_ts(),

                "status":
                    "ACTIVE",
            }

            state.setdefault(
                "active_trades",
                {},
            )[signal_id] = trade

            state.setdefault(
                "last_signals",
                {},
            )[symbol] = signal_id

            save_state(
                state
            )

            # =================================================
            # TELEGRAM
            # =================================================

            if direction == "LONG":

                direction_text = (
                    "🟢 LONG"
                )

                tp_move = (
                    tp - entry
                ) / entry * 100

                sl_move = (
                    sl - entry
                ) / entry * 100

            else:

                direction_text = (
                    "🔴 SHORT"
                )

                tp_move = (
                    entry - tp
                ) / entry * 100

                sl_move = (
                    entry - sl
                ) / entry * 100

            message = (

                "🚨 SCORE HUNTER PRO v4 🚨\n\n"

                f"💰 {symbol}USDT\n"

                f"📊 {direction_text}\n"

                f"{strength}\n\n"

                f"⭐ 4H Score: "
                f"{score_4h}/7\n"

                f"📱 1H Confirm: "
                f"{score_1h}/4\n\n"

                f"💵 Entry: "
                f"{entry:.8f}\n"

                f"📍 Current: "
                f"{current_price:.8f}\n"

                f"📏 Distance: "
                f"{distance:.2f}%\n\n"

                f"🎯 TP: "
                f"{tp:.8f} "
                f"(+{tp_move:.2f}%)\n"

                f"🛑 SL: "
                f"{sl:.8f} "
                f"({sl_move:+.2f}%)\n\n"

                f"⚖️ R:R: "
                f"1:{rr:.2f}\n"

                f"📐 Risk: "
                f"{risk_percent:.2f}%\n\n"

                f"🔎 BOS: "
                f"{'YES' if "
                f"(analysis_4h['bos']['bullish'] "
                f"or analysis_4h['bos']['bearish']) "
                f"else 'NO'}\n"

                f"🔄 CHoCH: "
                f"{'YES' if analysis_4h['choch'] else 'NO'}\n"

                f"💧 Liquidity sweep: "
                f"{'YES' if analysis_4h['liquidity_sweep'] else 'NO'}\n"

                f"↩️ Retest: "
                f"{'YES' if analysis_4h['pullback'] else 'NO'}\n"

                f"🕯 Price action: "
                f"{'YES' if analysis_4h['candle'] else 'NO'}\n"

                f"🧱 TP Path: CLEAR\n\n"

                "📊 Structure: 4H\n"

                "📱 Confirmation: 1H\n"

                "🎯 TP target: 1.00%\n"

                "🔒 Entry window: ±0.30%\n"

                "🏦 LBank\n\n"

                "⚠️ SIGNAL ONLY — "
                "MANAGE RISK."
            )

            sent = send_telegram(
                message
            )

            if sent:

                print(
                    f"{symbol}: "
                    f"📨 SIGNAL SENT"
                )

            else:

                print(
                    f"{symbol}: "
                    f"⚠️ TELEGRAM FAILED"
                )

        except Exception as e:

            print(
                f"{symbol}: "
                f"❌ ERROR: {e}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    save_state(
        state
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    send_statistics_if_needed(
        state
    )

    print(
        "\n"
        "✅ SCORE HUNTER PRO v4 "
        "scan completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
