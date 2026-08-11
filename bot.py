import os
import json
import time
import math
import requests


# ============================================================
# SCORE HUNTER PRO v3
#
# MARKET STRUCTURE + PRICE ACTION + LIQUIDITY
#
# 4H = MAIN STRUCTURE
# 1H = ENTRY CONFIRMATION
#
# CORE:
#   - Swing High / Swing Low
#   - BOS
#   - CHoCH
#   - Liquidity Sweep
#   - Pullback / Retest
#   - Price Action candle confirmation
#   - EMA filter
#   - RSI filter
#   - Volume filter
#   - ATR risk model
#   - Fixed TP 1%
#
# IMPORTANT:
# This bot generates SIGNALS only.
# It does NOT place exchange orders.
# ============================================================


# ============================================================
# TELEGRAM
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    "2090120004",
)


# ============================================================
# LBANK API
# ============================================================

LBANK_KLINE_URL = (
    "https://api.lbkex.com/v2/kline.do"
)

LBANK_PRICE_URL = (
    "https://api.lbkex.com/v2/supplement/"
    "ticker/price.do"
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
# MARKET STRUCTURE
# ============================================================

PIVOT_LEFT = 2
PIVOT_RIGHT = 2

STRUCTURE_LOOKBACK = 100

EVENT_LOOKBACK_4H = 12
EVENT_LOOKBACK_1H = 10

PULLBACK_LOOKBACK = 8


# ============================================================
# INDICATORS
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14

ATR_PERIOD = 14

VOLUME_LOOKBACK = 20


# ============================================================
# SIGNAL THRESHOLDS
# ============================================================

REQUIRED_SCORE_4H = 6
REQUIRED_SCORE_1H = 3


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
# STOP LOSS
# ============================================================

ATR_MULTIPLIER = 1.20

MIN_SL_PERCENT = 0.004
# 0.40%

MAX_SL_PERCENT = 0.025
# 2.50%


# ============================================================
# RISK / REWARD
# ============================================================

MIN_RISK_REWARD = 1.20


# ============================================================
# STRUCTURE / LIQUIDITY
# ============================================================

SWEEP_ATR_TOLERANCE = 0.20

PULLBACK_ATR_TOLERANCE = 0.35

LEVEL_BUFFER_ATR = 0.15


# ============================================================
# SIGNAL EXPIRATION
# ============================================================

SIGNAL_EXPIRATION_HOURS = 8


# ============================================================
# STATE
# ============================================================

STATE_FILE = "state.json"


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "ScoreHunterPro/3.0",
        "Accept":
            "application/json",
    }
)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    try:

        response = SESSION.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
            },
            timeout=20,
        )

        print(
            "Telegram:",
            response.status_code,
        )

        if response.status_code != 200:
            print(response.text)

        return response.ok

    except Exception as e:

        print(
            "Telegram error:",
            e,
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

    if not os.path.exists(STATE_FILE):
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
            {},
        )

        state.setdefault(
            "active_trades",
            {},
        )

        state.setdefault(
            "completed_trades",
            [],
        )

        return state

    except Exception as e:

        print(
            "State load error:",
            e,
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
# TIMEFRAME SECONDS
# ============================================================

def timeframe_seconds(timeframe):

    mapping = {
        "minute1": 60,
        "minute5": 300,
        "minute15": 900,
        "minute30": 1800,
        "hour1": 3600,
        "hour4": 14400,
        "hour8": 28800,
        "hour12": 43200,
        "day1": 86400,
        "week1": 604800,
    }

    if timeframe not in mapping:

        raise ValueError(
            f"Unsupported timeframe: "
            f"{timeframe}"
        )

    return mapping[timeframe]


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

    interval = (
        timeframe_seconds(
            timeframe
        )
    )

    now = int(time.time())

    start_time = (
        now
        - (
            limit
            * interval
        )
    )

    response = SESSION.get(
        LBANK_KLINE_URL,
        params={
            "symbol": symbol,
            "size": limit,
            "type": timeframe,
            "time": start_time,
        },
        timeout=20,
    )

    print(
        "LBank:",
        response.status_code,
    )

    response.raise_for_status()

    result = response.json()

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

        try:

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

        except Exception:
            continue

    candles.sort(
        key=lambda x: x["time"]
    )

    print(
        f"{symbol}: received "
        f"{len(candles)} candles"
    )

    return candles


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(
    candles,
    timeframe,
):

    if len(candles) < 3:

        raise RuntimeError(
            "Not enough candles."
        )

    interval = (
        timeframe_seconds(
            timeframe
        )
    )

    now = int(time.time())

    closed = []

    for candle in candles:

        candle_end = (
            candle["time"]
            + interval
        )

        if candle_end <= now:

            closed.append(candle)

    if len(closed) < 3:

        raise RuntimeError(
            "Could not determine "
            "closed candles."
        )

    return closed


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(symbol):

    response = SESSION.get(
        LBANK_PRICE_URL,
        params={
            "symbol": symbol,
        },
        timeout=20,
    )

    response.raise_for_status()

    result = response.json()

    if str(
        result.get("result")
    ).lower() != "true":

        raise RuntimeError(
            f"LBank price error: "
            f"{result}"
        )

    data = result.get(
        "data",
        [],
    )

    if not data:

        raise RuntimeError(
            f"No price data: "
            f"{result}"
        )

    price = data[0].get(
        "price"
    )

    if price is None:

        raise RuntimeError(
            f"Price not found: "
            f"{result}"
        )

    return float(price)


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
        2.0
        / (
            period + 1
        )
    )

    first = (
        sum(
            values[:period]
        )
        / period
    )

    result = (
        [None]
        * (
            period - 1
        )
    )

    result.append(first)

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
            max(
                change,
                0,
            )
        )

        losses.append(
            max(
                -change,
                0,
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

    result = (
        [None]
        * period
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
                / (
                    1 + rs
                )
            )
        )

    for i in range(
        period,
        len(gains),
    ):

        avg_gain = (
            (
                avg_gain
                * (
                    period - 1
                )
            )
            + gains[i]
        ) / period

        avg_loss = (
            (
                avg_loss
                * (
                    period - 1
                )
            )
            + losses[i]
        ) / period

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
                    / (
                        1 + rs
                    )
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
            candles[
                i - 1
            ]["close"]
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

        true_ranges.append(tr)

    if len(true_ranges) < period:
        return []

    first_atr = (
        sum(
            true_ranges[
                :period
            ]
        )
        / period
    )

    result = (
        [None]
        * period
    )

    result.append(
        first_atr
    )

    previous = first_atr

    for tr in true_ranges[
        period:
    ]:

        current = (
            (
                previous
                * (
                    period - 1
                )
            )
            + tr
        ) / period

        result.append(
            current
        )

        previous = current

    return result


# ============================================================
# AVERAGE VOLUME
# ============================================================

def average_volume(
    candles,
    lookback=20,
):

    if len(candles) < lookback:
        return 0.0

    return (
        sum(
            c["volume"]
            for c in candles[
                -lookback:
            ]
        )
        / lookback
    )


# ============================================================
# SWING HIGHS
# ============================================================

def find_swing_highs(
    candles,
    left=PIVOT_LEFT,
    right=PIVOT_RIGHT,
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

        is_high = True

        for j in range(
            i - left,
            i,
        ):

            if candles[j]["high"] > high:

                is_high = False
                break

        if not is_high:
            continue

        for j in range(
            i + 1,
            i + right + 1,
        ):

            if candles[j]["high"] > high:

                is_high = False
                break

        if is_high:

            swings.append(
                {
                    "index": i,
                    "price": high,
                    "time":
                        candles[i][
                            "time"
                        ],
                }
            )

    return swings


# ============================================================
# SWING LOWS
# ============================================================

def find_swing_lows(
    candles,
    left=PIVOT_LEFT,
    right=PIVOT_RIGHT,
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

        is_low = True

        for j in range(
            i - left,
            i,
        ):

            if candles[j]["low"] < low:

                is_low = False
                break

        if not is_low:
            continue

        for j in range(
            i + 1,
            i + right + 1,
        ):

            if candles[j]["low"] < low:

                is_low = False
                break

        if is_low:

            swings.append(
                {
                    "index": i,
                    "price": low,
                    "time":
                        candles[i][
                            "time"
                        ],
                }
            )

    return swings


# ============================================================
# MARKET STRUCTURE
# ============================================================

def analyze_structure(
    candles,
):

    if len(candles) < 30:

        return {
            "bias": None,
            "swing_highs": [],
            "swing_lows": [],
            "reason":
                "NOT_ENOUGH_DATA",
        }

    working = candles[
        -STRUCTURE_LOOKBACK:
    ]

    highs = find_swing_highs(
        working
    )

    lows = find_swing_lows(
        working
    )

    if (
        len(highs) < 2
        or len(lows) < 2
    ):

        return {
            "bias": None,
            "swing_highs": highs,
            "swing_lows": lows,
            "reason":
                "NOT_ENOUGH_SWINGS",
        }

    h1 = highs[-2]["price"]
    h2 = highs[-1]["price"]

    l1 = lows[-2]["price"]
    l2 = lows[-1]["price"]

    bullish_structure = (
        h2 > h1
        and l2 > l1
    )

    bearish_structure = (
        h2 < h1
        and l2 < l1
    )

    if bullish_structure:

        bias = "LONG"

    elif bearish_structure:

        bias = "SHORT"

    else:

        bias = None

    return {
        "bias": bias,
        "swing_highs": highs,
        "swing_lows": lows,
        "reason":
            (
                "BULLISH_STRUCTURE"
                if bias == "LONG"
                else
                "BEARISH_STRUCTURE"
                if bias == "SHORT"
                else
                "RANGE_OR_MIXED"
            ),
    }


# ============================================================
# BOS / CHOCH
# ============================================================

def detect_structure_break(
    candles,
    structure,
):

    highs = structure[
        "swing_highs"
    ]

    lows = structure[
        "swing_lows"
    ]

    if not highs or not lows:

        return {
            "bos_long": False,
            "bos_short": False,
            "choch_long": False,
            "choch_short": False,
            "level": None,
            "index": None,
        }

    last_close = candles[-1]["close"]

    previous_bias = None

    # Determine older structural bias
    if len(highs) >= 3 and len(lows) >= 3:

        old_h1 = highs[-3]["price"]
        old_h2 = highs[-2]["price"]

        old_l1 = lows[-3]["price"]
        old_l2 = lows[-2]["price"]

        if (
            old_h2 > old_h1
            and old_l2 > old_l1
        ):

            previous_bias = "LONG"

        elif (
            old_h2 < old_h1
            and old_l2 < old_l1
        ):

            previous_bias = "SHORT"

    latest_high = highs[-1]["price"]
    latest_low = lows[-1]["price"]

    bos_long = (
        last_close > latest_high
    )

    bos_short = (
        last_close < latest_low
    )

    choch_long = (
        bos_long
        and previous_bias == "SHORT"
    )

    choch_short = (
        bos_short
        and previous_bias == "LONG"
    )

    if bos_long:

        level = latest_high

    elif bos_short:

        level = latest_low

    else:

        level = None

    return {
        "bos_long": bos_long,
        "bos_short": bos_short,
        "choch_long": choch_long,
        "choch_short": choch_short,
        "level": level,
        "index":
            len(candles) - 1
            if level is not None
            else None,
    }


# ============================================================
# RECENT BOS
# ============================================================

def detect_recent_bos(
    candles,
):

    if len(candles) < 20:

        return {
            "direction": None,
            "level": None,
            "index": None,
            "type": None,
        }

    start = max(
        10,
        len(candles)
        - EVENT_LOOKBACK_4H
        - 10,
    )

    for i in range(
        start,
        len(candles),
    ):

        previous = candles[
            :i
        ]

        highs = find_swing_highs(
            previous
        )

        lows = find_swing_lows(
            previous
        )

        if highs:

            resistance = (
                highs[-1]["price"]
            )

            if (
                candles[i]["close"]
                > resistance
            ):

                return {
                    "direction":
                        "LONG",
                    "level":
                        resistance,
                    "index": i,
                    "type":
                        "BOS",
                }

        if lows:

            support = (
                lows[-1]["price"]
            )

            if (
                candles[i]["close"]
                < support
            ):

                return {
                    "direction":
                        "SHORT",
                    "level":
                        support,
                    "index": i,
                    "type":
                        "BOS",
                }

    return {
        "direction": None,
        "level": None,
        "index": None,
        "type": None,
    }


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(
    candles,
    direction,
    atr,
):

    if len(candles) < 10:

        return {
            "valid": False,
            "level": None,
            "index": None,
        }

    start = max(
        5,
        len(candles)
        - EVENT_LOOKBACK_4H,
    )

    tolerance = (
        atr
        * SWEEP_ATR_TOLERANCE
    )

    if direction == "LONG":

        # Bullish setup:
        # sweep below prior swing low
        # then close back above it.

        lows = find_swing_lows(
            candles[:start]
        )

        if not lows:
            return {
                "valid": False,
                "level": None,
                "index": None,
            }

        target = lows[-1]["price"]

        for i in range(
            start,
            len(candles),
        ):

            candle = candles[i]

            swept = (
                candle["low"]
                <= target
                + tolerance
            )

            reclaimed = (
                candle["close"]
                > target
            )

            if swept and reclaimed:

                return {
                    "valid": True,
                    "level": target,
                    "index": i,
                }

    else:

        highs = find_swing_highs(
            candles[:start]
        )

        if not highs:
            return {
                "valid": False,
                "level": None,
                "index": None,
            }

        target = highs[-1]["price"]

        for i in range(
            start,
            len(candles),
        ):

            candle = candles[i]

            swept = (
                candle["high"]
                >= target
                - tolerance
            )

            rejected = (
                candle["close"]
                < target
            )

            if swept and rejected:

                return {
                    "valid": True,
                    "level": target,
                    "index": i,
                }

    return {
        "valid": False,
        "level": None,
        "index": None,
    }


# ============================================================
# CANDLE PATTERNS
# ============================================================

def bullish_engulfing(
    previous,
    current,
):

    return (
        previous["close"]
        < previous["open"]
        and current["close"]
        > current["open"]
        and current["close"]
        >= previous["open"]
        and current["open"]
        <= previous["close"]
    )


def bearish_engulfing(
    previous,
    current,
):

    return (
        previous["close"]
        > previous["open"]
        and current["close"]
        < current["open"]
        and current["open"]
        >= previous["close"]
        and current["close"]
        <= previous["open"]
    )


def bullish_pinbar(
    candle,
):

    body = abs(
        candle["close"]
        - candle["open"]
    )

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    lower_wick = (
        min(
            candle["open"],
            candle["close"],
        )
        - candle["low"]
    )

    upper_wick = (
        candle["high"]
        - max(
            candle["open"],
            candle["close"],
        )
    )

    return (
        lower_wick
        >= body * 1.5
        and lower_wick
        > upper_wick
        and candle["close"]
        >= (
            candle["low"]
            + candle_range * 0.55
        )
    )


def bearish_pinbar(
    candle,
):

    body = abs(
        candle["close"]
        - candle["open"]
    )

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    upper_wick = (
        candle["high"]
        - max(
            candle["open"],
            candle["close"],
        )
    )

    lower_wick = (
        min(
            candle["open"],
            candle["close"],
        )
        - candle["low"]
    )

    return (
        upper_wick
        >= body * 1.5
        and upper_wick
        > lower_wick
        and candle["close"]
        <= (
            candle["low"]
            + candle_range * 0.45
        )
    )


def strong_bull_candle(
    candle,
):

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    close_location = (
        candle["close"]
        - candle["low"]
    ) / candle_range

    return (
        candle["close"]
        > candle["open"]
        and close_location >= 0.65
    )


def strong_bear_candle(
    candle,
):

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    close_location = (
        candle["high"]
        - candle["close"]
    ) / candle_range

    return (
        candle["close"]
        < candle["open"]
        and close_location >= 0.65
    )


def candle_confirmation(
    candles,
    direction,
):

    if len(candles) < 2:
        return False

    previous = candles[-2]
    current = candles[-1]

    if direction == "LONG":

        return (
            bullish_engulfing(
                previous,
                current,
            )
            or bullish_pinbar(
                current
            )
            or strong_bull_candle(
                current
            )
        )

    return (
        bearish_engulfing(
            previous,
            current,
        )
        or bearish_pinbar(
            current
        )
        or strong_bear_candle(
            current
        )
    )


# ============================================================
# PULLBACK / RETEST
# ============================================================

def detect_pullback(
    candles,
    direction,
    broken_level,
    atr,
):

    if broken_level is None:

        return {
            "valid": False,
            "index": None,
            "level": None,
        }

    tolerance = (
        atr
        * PULLBACK_ATR_TOLERANCE
    )

    start = max(
        0,
        len(candles)
        - PULLBACK_LOOKBACK,
    )

    if direction == "LONG":

        for i in range(
            start,
            len(candles),
        ):

            candle = candles[i]

            touched = (
                candle["low"]
                <= broken_level
                + tolerance
            )

            held = (
                candle["close"]
                >= broken_level
            )

            if touched and held:

                return {
                    "valid": True,
                    "index": i,
                    "level":
                        broken_level,
                }

    else:

        for i in range(
            start,
            len(candles),
        ):

            candle = candles[i]

            touched = (
                candle["high"]
                >= broken_level
                - tolerance
            )

            held = (
                candle["close"]
                <= broken_level
            )

            if touched and held:

                return {
                    "valid": True,
                    "index": i,
                    "level":
                        broken_level,
                }

    return {
        "valid": False,
        "index": None,
        "level": None,
    }


# ============================================================
# STRUCTURE LEVELS
# ============================================================

def resistance_levels(
    candles,
):

    swings = find_swing_highs(
        candles
    )

    return [
        x["price"]
        for x in swings
    ]


def support_levels(
    candles,
):

    swings = find_swing_lows(
        candles
    )

    return [
        x["price"]
        for x in swings
    ]


# ============================================================
# TP PATH
# ============================================================

def check_tp_path(
    candles,
    entry,
    tp,
    direction,
    atr,
):

    buffer = (
        atr
        * LEVEL_BUFFER_ATR
    )

    if direction == "LONG":

        levels = resistance_levels(
            candles
        )

        blocking = [
            level
            for level in levels
            if (
                level > entry
                and level
                <= tp + buffer
            )
        ]

        if blocking:

            nearest = min(
                blocking
            )

            return {
                "valid": False,
                "type":
                    "RESISTANCE",
                "level": nearest,
            }

    else:

        levels = support_levels(
            candles
        )

        blocking = [
            level
            for level in levels
            if (
                level < entry
                and level
                >= tp - buffer
            )
        ]

        if blocking:

            nearest = max(
                blocking
            )

            return {
                "valid": False,
                "type":
                    "SUPPORT",
                "level": nearest,
            }

    return {
        "valid": True,
        "type": None,
        "level": None,
    }


# ============================================================
# STOP LOSS
# ============================================================

def calculate_stop_loss(
    signal,
):

    direction = (
        signal["direction"]
    )

    entry = signal["entry"]

    atr = signal["atr"]

    candles = signal[
        "candles"
    ]

    pullback_index = (
        signal.get(
            "pullback_index"
        )
    )

    sweep_index = (
        signal.get(
            "sweep_index"
        )
    )

    if direction == "LONG":

        atr_sl = (
            entry
            - ATR_MULTIPLIER
            * atr
        )

        candidates = [
            atr_sl
        ]

        if (
            pullback_index
            is not None
        ):

            start = max(
                0,
                pullback_index
                - 3,
            )

            pullback_low = min(
                c["low"]
                for c in candles[
                    start:
                    pullback_index + 1
                ]
            )

            candidates.append(
                pullback_low
                - (
                    LEVEL_BUFFER_ATR
                    * atr
                )
            )

        if (
            sweep_index
            is not None
        ):

            sweep_low = (
                candles[
                    sweep_index
                ]["low"]
            )

            candidates.append(
                sweep_low
                - (
                    LEVEL_BUFFER_ATR
                    * atr
                )
            )

        raw_sl = min(
            candidates
        )

        risk = (
            entry
            - raw_sl
        )

    else:

        atr_sl = (
            entry
            + ATR_MULTIPLIER
            * atr
        )

        candidates = [
            atr_sl
        ]

        if (
            pullback_index
            is not None
        ):

            start = max(
                0,
                pullback_index
                - 3,
            )

            pullback_high = max(
                c["high"]
                for c in candles[
                    start:
                    pullback_index + 1
                ]
            )

            candidates.append(
                pullback_high
                + (
                    LEVEL_BUFFER_ATR
                    * atr
                )
            )

        if (
            sweep_index
            is not None
        ):

            sweep_high = (
                candles[
                    sweep_index
                ]["high"]
            )

            candidates.append(
                sweep_high
                + (
                    LEVEL_BUFFER_ATR
                    * atr
                )
            )

        raw_sl = max(
            candidates
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

    risk = max(
        risk,
        min_risk,
    )

    risk = min(
        risk,
        max_risk,
    )

    if direction == "LONG":

        sl = (
            entry
            - risk
        )

    else:

        sl = (
            entry
            + risk
        )

    risk_percent = (
        abs(
            entry - sl
        )
        / entry
        * 100
    )

    return {
        "sl": sl,
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
# RISK LEVELS
# ============================================================

def calculate_risk_levels(
    signal,
):

    entry = signal["entry"]

    direction = (
        signal["direction"]
    )

    candles = signal[
        "candles"
    ]

    atr = signal["atr"]

    stop_data = (
        calculate_stop_loss(
            signal
        )
    )

    sl = stop_data["sl"]

    risk_percent = (
        stop_data[
            "risk_percent"
        ]
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
                "INVALID_RISK",
        }

    rr = (
        reward / risk
    )

    if rr < MIN_RISK_REWARD:

        return {
            "valid": False,
            "reason":
                (
                    f"R:R_TOO_LOW "
                    f"1:{rr:.2f}"
                ),
        }

    path = check_tp_path(
        candles,
        entry,
        tp,
        direction,
        atr,
    )

    if not path["valid"]:

        return {
            "valid": False,
            "reason":
                (
                    f"{path['type']}_BLOCKS_TP"
                ),
            "blocking_level":
                path["level"],
            "blocking_type":
                path["type"],
        }

    return {
        "valid": True,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_percent":
            risk_percent,
        "rr": rr,
        "blocking_level":
            None,
        "blocking_type":
            None,
    }


# ============================================================
# 4H SIGNAL
# ============================================================

def calculate_4h_signal(
    candles,
):

    if len(candles) < 230:

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

    current = candles[i]

    close = current["close"]

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
            "reason":
                "INDICATOR_NOT_READY",
        }

    structure = (
        analyze_structure(
            candles
        )
    )

    bias = structure[
        "bias"
    ]

    if bias is None:

        return {
            "valid": False,
            "reason":
                "STRUCTURE_MIXED_OR_RANGE",
        }

    # --------------------------------------------------------
    # EMA FILTER
    # --------------------------------------------------------

    ema_long = (
        close > e200
        and e20 > e50
    )

    ema_short = (
        close < e200
        and e20 < e50
    )

    # --------------------------------------------------------
    # RSI FILTER
    # --------------------------------------------------------

    rsi_long = (
        rsi > 50
        and rsi < 75
        and rsi >= previous_rsi
    )

    rsi_short = (
        rsi < 50
        and rsi > 25
        and rsi <= previous_rsi
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    vol_ma = average_volume(
        candles,
        VOLUME_LOOKBACK,
    )

    volume_ok = (
        current["volume"]
        >= vol_ma * 0.85
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    volatility_ok = (
        atr / close
        >= 0.0015
    )

    # --------------------------------------------------------
    # RECENT BOS
    # --------------------------------------------------------

    bos = detect_recent_bos(
        candles
    )

    if bos["direction"] is None:

        return {
            "valid": False,
            "reason":
                "NO_RECENT_BOS",
        }

    if bos["direction"] != bias:

        return {
            "valid": False,
            "reason":
                "BOS_CONFLICTS_WITH_STRUCTURE",
        }

    broken_level = bos[
        "level"
    ]

    # --------------------------------------------------------
    # LIQUIDITY SWEEP
    # --------------------------------------------------------

    sweep = (
        detect_liquidity_sweep(
            candles,
            bias,
            atr,
        )
    )

    # Sweep is valuable but not mandatory.
    sweep_score = int(
        sweep["valid"]
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    pullback = (
        detect_pullback(
            candles,
            bias,
            broken_level,
            atr,
        )
    )

    if not pullback["valid"]:

        return {
            "valid": False,
            "reason":
                "NO_PULLBACK_RETEST",
        }

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    candle_ok = (
        candle_confirmation(
            candles,
            bias,
        )
    )

    if not candle_ok:

        return {
            "valid": False,
            "reason":
                "NO_PRICE_ACTION_CONFIRMATION",
        }

    # --------------------------------------------------------
    # SCORE
    #
    # Max:
    # structure 2
    # BOS 2
    # sweep 1
    # pullback 1
    # candle 1
    # EMA 1
    # RSI 1
    # volume 1
    # volatility 1
    #
    # total = 11
    # --------------------------------------------------------

    score = 0

    score += 2

    score += 2

    score += sweep_score

    score += int(
        pullback["valid"]
    )

    score += int(
        candle_ok
    )

    if bias == "LONG":

        score += int(
            ema_long
        )

        score += int(
            rsi_long
        )

    else:

        score += int(
            ema_short
        )

        score += int(
            rsi_short
        )

    score += int(
        volume_ok
    )

    score += int(
        volatility_ok
    )

    if score < REQUIRED_SCORE_4H:

        return {
            "valid": False,
            "reason":
                (
                    f"SCORE_TOO_LOW "
                    f"{score}/11"
                ),
        }

    return {
        "valid": True,
        "direction": bias,
        "score": score,
        "entry": close,
        "atr": atr,
        "broken_level":
            broken_level,
        "bos_type":
            bos["type"],
        "sweep_index":
            sweep["index"],
        "pullback_index":
            pullback["index"],
        "candle_time":
            current["time"],
        "candles":
            candles,
        "rsi":
            rsi,
        "ema20":
            e20,
        "ema50":
            e50,
        "ema200":
            e200,
        "volume_ok":
            volume_ok,
        "sweep":
            sweep["valid"],
    }


# ============================================================
# 1H CONFIRMATION
# ============================================================

def calculate_1h_confirmation(
    candles,
    direction,
):

    if len(candles) < 230:

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

    close = candles[i]["close"]

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

    structure = (
        analyze_structure(
            candles
        )
    )

    score = 0

    reasons = []

    # --------------------------------------------------------
    # 1H STRUCTURE
    # --------------------------------------------------------

    if (
        structure["bias"]
        == direction
    ):

        score += 1

    else:

        reasons.append(
            "1H_STRUCTURE_CONFLICT"
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if direction == "LONG":

        if (
            close > e200
            and e20 > e50
        ):

            score += 1

        else:

            reasons.append(
                "1H_EMA_CONFLICT"
            )

    else:

        if (
            close < e200
            and e20 < e50
        ):

            score += 1

        else:

            reasons.append(
                "1H_EMA_CONFLICT"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if direction == "LONG":

        if (
            rsi > 50
            and rsi >= previous_rsi
        ):

            score += 1

        else:

            reasons.append(
                "1H_RSI_CONFLICT"
            )

    else:

        if (
            rsi < 50
            and rsi <= previous_rsi
        ):

            score += 1

        else:

            reasons.append(
                "1H_RSI_CONFLICT"
            )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    candle_ok = (
        candle_confirmation(
            candles,
            direction,
        )
    )

    if candle_ok:

        score += 1

    else:

        reasons.append(
            "1H_CANDLE_NOT_CONFIRMED"
        )

    # --------------------------------------------------------
    # 1H PULLBACK
    # --------------------------------------------------------

    recent_ema_touch = False

    start = max(
        0,
        len(candles)
        - 5,
    )

    for candle in candles[
        start:
    ]:

        if direction == "LONG":

            if (
                candle["low"]
                <= e20
                and candle["close"]
                >= e20
            ):

                recent_ema_touch = True
                break

        else:

            if (
                candle["high"]
                >= e20
                and candle["close"]
                <= e20
            ):

                recent_ema_touch = True
                break

    if recent_ema_touch:

        score += 1

    else:

        reasons.append(
            "1H_NO_PULLBACK"
        )

    valid = (
        score
        >= REQUIRED_SCORE_1H
    )

    if valid:

        reason = "1H_CONFIRMED"

    else:

        reason = (
            "1H_SCORE_LOW:"
            + ",".join(reasons)
        )

    return {
        "valid": valid,
        "score": score,
        "rsi": rsi,
        "atr": atr,
        "reason": reason,
    }


# ============================================================
# ENTRY WINDOW
# ============================================================

def entry_is_valid(
    entry,
    current_price,
):

    distance = (
        abs(
            current_price
            - entry
        )
        / entry
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

    age_seconds = (
        int(time.time())
        - candle_time
    )

    return (
        age_seconds
        > (
            SIGNAL_EXPIRATION_HOURS
            * 3600
        )
    )


# ============================================================
# SIGNAL STRENGTH
# ============================================================

def signal_strength(
    score_4h,
    score_1h,
):

    if (
        score_4h >= 10
        and score_1h >= 4
    ):

        return "🔥 VERY STRONG"

    if (
        score_4h >= 8
        and score_1h >= 4
    ):

        return "💪 STRONG"

    if (
        score_4h >= 7
        and score_1h >= 3
    ):

        return "🟢 GOOD"

    return "🟡 NORMAL"


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
                f"price check error: "
                f"{e}"
            )

            continue

        direction = trade[
            "direction"
        ]

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
                abs(
                    tp - entry
                )
                / entry
                * 100
            )

        else:

            pnl_percent = -(
                abs(
                    sl - entry
                )
                / entry
                * 100
            )

        trade[
            "result"
        ] = result

        trade[
            "exit_price"
        ] = current_price

        trade[
            "pnl_percent"
        ] = pnl_percent

        trade[
            "closed_at"
        ] = int(time.time())

        state.setdefault(
            "completed_trades",
            [],
        ).append(trade)

        finished.append(
            trade_id
        )

        emoji = (
            "✅"
            if result == "TP"
            else "❌"
        )

        message = (
            f"{emoji} SCORE HUNTER PRO v3\n\n"
            f"TRADE CLOSED\n\n"
            f"💰 {trade['symbol']}USDT\n"
            f"📊 {direction}\n"
            f"⭐ 4H Score: "
            f"{trade['score_4h']}/11\n"
            f"📱 1H Score: "
            f"{trade['score_1h']}/5\n"
            f"📌 Result: {result}\n\n"
            f"💵 Entry: "
            f"{entry:.8f}\n"
            f"🏁 Exit: "
            f"{current_price:.8f}\n"
            f"📈 P/L: "
            f"{pnl_percent:+.2f}%\n\n"
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

    total = len(trades)

    wins = sum(
        1
        for t in trades
        if t.get("result")
        == "TP"
    )

    losses = sum(
        1
        for t in trades
        if t.get("result")
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
        t.get(
            "pnl_percent",
            0,
        )
        for t in trades
    )

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
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
        stats["total"] % 10
        != 0
    ):
        return

    message = (
        "📊 SCORE HUNTER PRO v3\n\n"
        "STATISTICS\n\n"
        f"📌 Closed trades: "
        f"{stats['total']}\n"
        f"✅ TP: "
        f"{stats['wins']}\n"
        f"❌ SL: "
        f"{stats['losses']}\n"
        f"🎯 Win rate: "
        f"{stats['win_rate']:.1f}%\n"
        f"📈 Total P/L: "
        f"{stats['pnl']:+.2f}%\n\n"
        f"🎯 TP target: 1.00%\n"
        f"⏱ 4H + 1H\n"
        f"🏦 LBank"
    )

    send_telegram(
        message
    )


# ============================================================
# PRINT REJECTION
# ============================================================

def print_rejection(
    symbol,
    reason,
):

    print(
        f"{symbol}: ❌ {reason}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🟢 SCORE HUNTER PRO v3"
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
        "🕯 Price action confirmation enabled"
    )

    print(
        "🎯 Fixed TP: 1.00%"
    )

    print(
        "🔒 Entry window: "
        f"{ENTRY_MAX_DISTANCE * 100:.2f}%"
    )

    print(
        "🛑 ATR multiplier: "
        f"{ATR_MULTIPLIER}"
    )

    print(
        "⚖️ Minimum R:R: "
        f"1:{MIN_RISK_REWARD}"
    )

    print(
        "⏳ Expiration: "
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
            # 4H DATA
            # =================================================

            candles_4h_raw = (
                get_candles(
                    lbank_symbol,
                    MAIN_TIMEFRAME,
                    CANDLE_LIMIT_4H,
                )
            )

            candles_4h = (
                get_closed_candles(
                    candles_4h_raw,
                    MAIN_TIMEFRAME,
                )
            )

            latest_4h = (
                candles_4h[-1]
            )

            print(
                f"{symbol}: "
                f"latest closed 4H: "
                f"{latest_4h['time']}"
            )

            # =================================================
            # 4H SIGNAL
            # =================================================

            signal = (
                calculate_4h_signal(
                    candles_4h
                )
            )

            if not signal["valid"]:

                print_rejection(
                    symbol,
                    signal["reason"],
                )

                continue

            direction = signal[
                "direction"
            ]

            score_4h = signal[
                "score"
            ]

            entry = signal[
                "entry"
            ]

            candle_time = signal[
                "candle_time"
            ]

            # =================================================
            # EXPIRATION
            # =================================================

            if signal_is_expired(
                candle_time
            ):

                print_rejection(
                    symbol,
                    "4H_SIGNAL_EXPIRED",
                )

                continue

            # =================================================
            # 1H DATA
            # =================================================

            candles_1h_raw = (
                get_candles(
                    lbank_symbol,
                    ENTRY_TIMEFRAME,
                    CANDLE_LIMIT_1H,
                )
            )

            candles_1h = (
                get_closed_candles(
                    candles_1h_raw,
                    ENTRY_TIMEFRAME,
                )
            )

            # =================================================
            # 1H CONFIRMATION
            # =================================================

            confirmation = (
                calculate_1h_confirmation(
                    candles_1h,
                    direction,
                )
            )

            score_1h = confirmation[
                "score"
            ]

            if not confirmation[
                "valid"
            ]:

                print_rejection(
                    symbol,
                    confirmation[
                        "reason"
                    ],
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

            distance = (
                abs(
                    current_price
                    - entry
                )
                / entry
                * 100
            )

            print(
                f"{symbol}: "
                f"{direction} "
                f"4H={score_4h}/11 "
                f"1H={score_1h}/5 "
                f"entry={entry:.8f} "
                f"current={current_price:.8f} "
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

                print_rejection(
                    symbol,
                    (
                        f"ENTRY_TOO_FAR "
                        f"{distance:.3f}%"
                    ),
                )

                continue

            # =================================================
            # DUPLICATE
            # =================================================

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

                print_rejection(
                    symbol,
                    "SIGNAL_ALREADY_SENT",
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
                    trade[
                        "symbol"
                    ]
                    == symbol
                ):

                    existing_trade = (
                        trade
                    )

                    break

            if existing_trade:

                print_rejection(
                    symbol,
                    "ACTIVE_TRADE_EXISTS",
                )

                continue

            # =================================================
            # RISK
            # =================================================

            levels = (
                calculate_risk_levels(
                    signal
                )
            )

            if not levels[
                "valid"
            ]:

                print_rejection(
                    symbol,
                    levels[
                        "reason"
                    ],
                )

                continue

            sl = levels["sl"]
            tp = levels["tp"]

            risk_percent = (
                levels[
                    "risk_percent"
                ]
            )

            rr = levels["rr"]

            # =================================================
            # STRENGTH
            # =================================================

            strength = (
                signal_strength(
                    score_4h,
                    score_1h,
                )
            )

            print(
                f"{symbol}: "
                f"🔥 FINAL VALID SIGNAL"
            )

            print(
                f"Direction: "
                f"{direction}"
            )

            print(
                f"4H Score: "
                f"{score_4h}/11"
            )

            print(
                f"1H Score: "
                f"{score_1h}/5"
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

                "tp_percent":
                    TP_PERCENT * 100,

                "risk_percent":
                    risk_percent,

                "rr":
                    rr,

                "broken_level":
                    signal[
                        "broken_level"
                    ],

                "bos_type":
                    signal[
                        "bos_type"
                    ],

                "liquidity_sweep":
                    signal[
                        "sweep"
                    ],

                "candle_time":
                    candle_time,

                "created_at":
                    int(
                        time.time()
                    ),

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

                tp_percent = (
                    tp - entry
                ) / entry * 100

                sl_percent = (
                    sl - entry
                ) / entry * 100

            else:

                direction_text = (
                    "🔴 SHORT"
                )

                tp_percent = (
                    entry - tp
                ) / entry * 100

                sl_percent = (
                    entry - sl
                ) / entry * 100

            message = (

                "🚨 SCORE HUNTER PRO v3 🚨\n\n"

                f"💰 {symbol}USDT\n"

                f"📊 {direction_text}\n"

                f"{strength}\n\n"

                f"⭐ 4H Score: "
                f"{score_4h}/11\n"

                f"📱 1H Score: "
                f"{score_1h}/5\n\n"

                f"🧱 Structure: "
                f"{signal['direction']}\n"

                f"🔎 BOS: "
                f"{signal['bos_type']}\n"

                f"💧 Liquidity Sweep: "
                f"{'YES' if signal['sweep'] else 'NO'}\n"

                f"↩️ Pullback: YES\n"

                f"🕯 Price Action: YES\n\n"

                f"💵 Entry: "
                f"{entry:.8f}\n"

                f"📍 Current: "
                f"{current_price:.8f}\n"

                f"📏 Distance: "
                f"{distance:.2f}%\n\n"

                f"🎯 TP: "
                f"{tp:.8f} "
                f"(+{tp_percent:.2f}%)\n"

                f"🛑 SL: "
                f"{sl:.8f} "
                f"({sl_percent:+.2f}%)\n\n"

                f"⚖️ R:R: "
                f"1:{rr:.2f}\n"

                f"📐 Risk: "
                f"{risk_percent:.2f}%\n\n"

                "🧱 TP Path: CLEAR\n"

                "📊 Main Structure: 4H\n"

                "📱 Entry Confirmation: 1H\n"

                "🎯 TP Target: 1.00%\n"

                "🔒 Entry Window: ±0.30%\n"

                "⏳ Expiration: 8h\n"

                "🏦 Data: LBank\n\n"

                "⚠️ Risk management required."
            )

            sent = send_telegram(
                message
            )

            if sent:

                print(
                    f"{symbol}: "
                    f"🚨 {direction} "
                    f"SIGNAL SENT"
                )

            else:

                print(
                    f"{symbol}: "
                    f"Telegram failed"
                )

        except Exception as e:

            print(
                f"{symbol}: "
                f"💥 ERROR: {e}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    save_state(
        state
    )

    # ========================================================
    # STATS
    # ========================================================

    send_statistics_if_needed(
        state
    )

    print(
        "\n✅ SCORE HUNTER PRO v3 "
        "scan completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
