import os
import json
import time
import requests


# ============================================================
# SCORE HUNTER PRO - LBANK
#
# MULTI-TIMEFRAME:
# 4H = MAIN MARKET STRUCTURE
# 1H = ENTRY CONFIRMATION
#
# SIGNAL MODEL:
# 1. 4H Trend
# 2. 4H EMA structure
# 3. 4H RSI
# 4. 4H Volume
# 5. 4H Breakout
# 6. 4H Pullback
# 7. 4H Candle confirmation
# 8. 4H Volatility
# 9. 1H Trend confirmation
# 10. 1H RSI confirmation
# 11. 1H Candle confirmation
# 12. Support / Resistance path
#
# RISK:
# - Fixed TP = 1%
# - ATR + structure SL
# - Minimum SL = 0.40%
# - Maximum SL = 2.50%
# - Minimum R:R = 1.20
# - Entry window = 0.30%
#
# PROTECTION:
# - Duplicate signal protection
# - One active trade per coin
# - Signal expiration
# - TP / SL tracking
# - Trade statistics
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
# LBANK API
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

CANDLE_LIMIT_4H = 250
CANDLE_LIMIT_1H = 250


# ============================================================
# SIGNAL SETTINGS
# ============================================================

# Minimum 4H score
REQUIRED_SCORE_4H = 5

# Entry confirmation score on 1H
REQUIRED_SCORE_1H = 2


# Current market price must remain close
# to the signal candle close.
ENTRY_MAX_DISTANCE = 0.003
# 0.30%


# A signal becomes too old after this time.
SIGNAL_EXPIRATION_HOURS = 8


# ============================================================
# TAKE PROFIT
# ============================================================

TP_PERCENT = 0.01
# Exactly 1% price movement


# ============================================================
# STOP LOSS
# ============================================================

ATR_PERIOD = 14

ATR_MULTIPLIER = 1.2

MIN_SL_PERCENT = 0.004
# 0.40%

MAX_SL_PERCENT = 0.025
# 2.50%


# ============================================================
# RISK / REWARD
# ============================================================

MIN_RISK_REWARD = 1.20


# ============================================================
# MARKET STRUCTURE
# ============================================================

BREAKOUT_LOOKBACK_4H = 6

STRUCTURE_LOOKBACK_4H = 50

PIVOT_LEFT = 2

PIVOT_RIGHT = 2

LEVEL_BUFFER_ATR = 0.15


# ============================================================
# 1H CONFIRMATION
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14

VOLUME_LOOKBACK = 20


# ============================================================
# STATE
# ============================================================

STATE_FILE = "state.json"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    response = requests.post(
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

    now = int(time.time())

    hours = 1

    if timeframe == "hour4":
        hours = 4

    start_time = (
        now
        - (
            limit
            * hours
            * 60
            * 60
        )
    )

    response = requests.get(
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
            f"LBank API error: {result}"
        )

    raw_data = result.get(
        "data",
        [],
    )

    if not raw_data:

        raise RuntimeError(
            f"No candle data for {symbol}"
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

    response = requests.get(
        LBANK_TICKER_URL,
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
            f"LBank ticker error: {result}"
        )

    data = result.get(
        "data",
        [],
    )

    if not data:

        raise RuntimeError(
            f"No ticker data for {symbol}"
        )

    ticker = data[0]

    ticker_data = ticker.get(
        "ticker",
        ticker,
    )

    price = ticker_data.get(
        "latest"
    )

    if price is None:

        price = ticker_data.get(
            "last"
        )

    if price is None:

        raise RuntimeError(
            f"Could not find price: "
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

    # Ignore currently forming candle.
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
        sum(values[:period])
        / period
    )

    result = (
        [None] * (period - 1)
    )

    result.append(first)

    previous = first

    for price in values[period:]:

        current = (
            (price - previous)
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
            )
            + gains[i]
        ) / period

        avg_loss = (
            (
                avg_loss
                * (period - 1)
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
                    / (1 + rs)
                )
            )

        result.append(value)

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

        true_ranges.append(tr)

    if len(true_ranges) < period:
        return []

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
            )
            + tr
        ) / period

        result.append(
            current
        )

        previous = current

    return result


# ============================================================
# PIVOT RESISTANCE
# ============================================================

def find_resistance_levels(
    candles,
    entry,
):

    start = max(
        PIVOT_LEFT,
        len(candles)
        - STRUCTURE_LOOKBACK_4H,
    )

    end = (
        len(candles)
        - PIVOT_RIGHT
    )

    levels = []

    for i in range(
        start,
        end,
    ):

        high = candles[i]["high"]

        left_highs = [
            candles[j]["high"]
            for j in range(
                i - PIVOT_LEFT,
                i,
            )
        ]

        right_highs = [
            candles[j]["high"]
            for j in range(
                i + 1,
                i + PIVOT_RIGHT + 1,
            )
        ]

        if (
            left_highs
            and right_highs
            and high >= max(
                left_highs
            )
            and high >= max(
                right_highs
            )
            and high > entry
        ):

            levels.append(high)

    return sorted(
        set(levels)
    )


# ============================================================
# PIVOT SUPPORT
# ============================================================

def find_support_levels(
    candles,
    entry,
):

    start = max(
        PIVOT_LEFT,
        len(candles)
        - STRUCTURE_LOOKBACK_4H,
    )

    end = (
        len(candles)
        - PIVOT_RIGHT
    )

    levels = []

    for i in range(
        start,
        end,
    ):

        low = candles[i]["low"]

        left_lows = [
            candles[j]["low"]
            for j in range(
                i - PIVOT_LEFT,
                i,
            )
        ]

        right_lows = [
            candles[j]["low"]
            for j in range(
                i + 1,
                i + PIVOT_RIGHT + 1,
            )
        ]

        if (
            left_lows
            and right_lows
            and low <= min(
                left_lows
            )
            and low <= min(
                right_lows
            )
            and low < entry
        ):

            levels.append(low)

    return sorted(
        set(levels),
        reverse=True,
    )


# ============================================================
# CANDLE QUALITY
# ============================================================

def candle_confirmation(candle):

    open_price = candle["open"]
    close = candle["close"]
    high = candle["high"]
    low = candle["low"]

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        return (
            False,
            False,
        )

    bull_ratio = (
        close - open_price
    ) / candle_range

    bear_ratio = (
        open_price - close
    ) / candle_range

    bull = (
        close > open_price
        and bull_ratio >= 0.40
    )

    bear = (
        close < open_price
        and bear_ratio >= 0.40
    )

    return (
        bull,
        bear,
    )


# ============================================================
# 4H SIGNAL
# ============================================================

def calculate_4h_signal(
    candles,
):

    if len(candles) < 210:

        return None

    closes = [
        c["close"]
        for c in candles
    ]

    volumes = [
        c["volume"]
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

    low = current["low"]

    high = current["high"]

    e20 = ema20[i]

    e50 = ema50[i]

    e200 = ema200[i]

    rsi = rsi_values[-1]

    previous_rsi = (
        rsi_values[-2]
    )

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

        return None

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    long_trend = (
        close > e200
        and e20 > e50
    )

    short_trend = (
        close < e200
        and e20 < e50
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    long_rsi = (
        rsi > 50
        and rsi < 72
        and rsi > previous_rsi
    )

    short_rsi = (
        rsi < 50
        and rsi > 28
        and rsi < previous_rsi
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ma = (
        sum(
            volumes[-VOLUME_LOOKBACK:]
        )
        / VOLUME_LOOKBACK
    )

    volume_ok = (
        current["volume"]
        >= volume_ma
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    recent_high = max(
        c["high"]
        for c in candles[
            -BREAKOUT_LOOKBACK_4H - 1:
            -1
        ]
    )

    recent_low = min(
        c["low"]
        for c in candles[
            -BREAKOUT_LOOKBACK_4H - 1:
            -1
        ]
    )

    bull_break = (
        close > recent_high
    )

    bear_break = (
        close < recent_low
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    long_pullback = (
        low <= e20
        and close > e20
    )

    short_pullback = (
        high >= e20
        and close < e20
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    bull_candle, bear_candle = (
        candle_confirmation(
            current
        )
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    volatility_ok = (
        atr / close >= 0.002
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    long_score = (
        int(long_trend)
        + int(long_rsi)
        + int(volume_ok)
        + int(bull_break)
        + int(long_pullback)
        + int(bull_candle)
        + int(volatility_ok)
    )

    short_score = (
        int(short_trend)
        + int(short_rsi)
        + int(volume_ok)
        + int(bear_break)
        + int(short_pullback)
        + int(bear_candle)
        + int(volatility_ok)
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if (
        long_score >= REQUIRED_SCORE_4H
        and long_trend
        and long_pullback
        and bull_candle
    ):

        return {
            "direction": "LONG",
            "score": long_score,
            "entry": close,
            "atr": atr,
            "structure_low": recent_low,
            "structure_high": recent_high,
            "candle_time": current["time"],
            "candles": candles,
        }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if (
        short_score >= REQUIRED_SCORE_4H
        and short_trend
        and short_pullback
        and bear_candle
    ):

        return {
            "direction": "SHORT",
            "score": short_score,
            "entry": close,
            "atr": atr,
            "structure_low": recent_low,
            "structure_high": recent_high,
            "candle_time": current["time"],
            "candles": candles,
        }

    return None


# ============================================================
# 1H CONFIRMATION
# ============================================================

def calculate_1h_confirmation(
    candles,
    direction,
):

    if len(candles) < 210:

        return {
            "valid": False,
            "score": 0,
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

    i = len(candles) - 1

    current = candles[i]

    close = current["close"]

    e20 = ema20[i]

    e50 = ema50[i]

    e200 = ema200[i]

    rsi = rsi_values[-1]

    previous_rsi = (
        rsi_values[-2]
    )

    bull_candle, bear_candle = (
        candle_confirmation(
            current
        )
    )

    score = 0

    if direction == "LONG":

        if (
            close > e200
            and e20 > e50
        ):
            score += 1

        if (
            rsi > 50
            and rsi > previous_rsi
        ):
            score += 1

        if bull_candle:
            score += 1

    else:

        if (
            close < e200
            and e20 < e50
        ):
            score += 1

        if (
            rsi < 50
            and rsi < previous_rsi
        ):
            score += 1

        if bear_candle:
            score += 1

    return {
        "valid": (
            score >= REQUIRED_SCORE_1H
        ),
        "score": score,
        "rsi": rsi,
    }


# ============================================================
# SIGNAL STRENGTH
# ============================================================

def signal_strength(
    score_4h,
    score_1h,
):

    total = (
        score_4h
        + score_1h
    )

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

    return "🟡 NORMAL"


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

    structure_low = (
        signal["structure_low"]
    )

    structure_high = (
        signal["structure_high"]
    )

    if direction == "LONG":

        atr_sl = (
            entry
            - (
                ATR_MULTIPLIER
                * atr
            )
        )

        structure_sl = (
            structure_low
            - (
                0.15
                * atr
            )
        )

        sl = min(
            atr_sl,
            structure_sl,
        )

        risk = (
            entry - sl
        )

    else:

        atr_sl = (
            entry
            + (
                ATR_MULTIPLIER
                * atr
            )
        )

        structure_sl = (
            structure_high
            + (
                0.15
                * atr
            )
        )

        sl = max(
            atr_sl,
            structure_sl,
        )

        risk = (
            sl - entry
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

        sl = entry - risk

    else:

        sl = entry + risk

    risk_percent = (
        abs(entry - sl)
        / entry
        * 100
    )

    return {
        "sl": sl,
        "risk_percent": risk_percent,
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
        * LEVEL_BUFFER_ATR
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == "LONG":

        resistance_levels = (
            find_resistance_levels(
                candles,
                entry,
            )
        )

        blocking = []

        for level in (
            resistance_levels
        ):

            if (
                level > entry
                and level
                <= tp + buffer
            ):

                blocking.append(
                    level
                )

        if blocking:

            nearest = min(
                blocking
            )

            return {
                "valid": False,
                "type": "RESISTANCE",
                "level": nearest,
            }

        return {
            "valid": True,
            "type": None,
            "level": None,
        }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    support_levels = (
        find_support_levels(
            candles,
            entry,
        )
    )

    blocking = []

    for level in (
        support_levels
    ):

        if (
            level < entry
            and level
            >= tp - buffer
        ):

            blocking.append(
                level
            )

    if blocking:

        nearest = max(
            blocking
        )

        return {
            "valid": False,
            "type": "SUPPORT",
            "level": nearest,
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
            "reason": "Invalid risk",
        }

    rr = (
        reward / risk
    )

    # --------------------------------------------------------
    # R:R FILTER
    # --------------------------------------------------------

    if rr < MIN_RISK_REWARD:

        return {
            "valid": False,
            "reason": (
                f"R:R too low "
                f"1:{rr:.2f}"
            ),
        }

    # --------------------------------------------------------
    # TP PATH FILTER
    # --------------------------------------------------------

    path = check_tp_path(
        signal,
        tp,
    )

    if not path["valid"]:

        return {
            "valid": False,
            "reason": (
                f"{path['type']} "
                f"at "
                f"{path['level']:.8f} "
                f"blocks TP"
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
# SIGNAL EXPIRATION
# ============================================================

def signal_is_expired(
    candle_time,
):

    age_seconds = (
        int(time.time())
        - candle_time
    )

    age_hours = (
        age_seconds
        / 3600
    )

    return (
        age_hours
        > SIGNAL_EXPIRATION_HOURS
    )


# ============================================================
# CHECK ACTIVE TRADES
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

        symbol = (
            trade["symbol"]
        )

        direction = (
            trade["direction"]
        )

        entry = trade["entry"]

        tp = trade["tp"]

        sl = trade["sl"]

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
                f"{symbol}: "
                f"price check error: "
                f"{e}"
            )

            continue

        result = None

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if direction == "LONG":

            if current_price >= tp:

                result = "TP"

            elif current_price <= sl:

                result = "SL"

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            if current_price <= tp:

                result = "TP"

            elif current_price >= sl:

                result = "SL"

        if result is None:
            continue

        # ----------------------------------------------------
        # PNL
        # ----------------------------------------------------

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

        trade["result"] = result

        trade["exit_price"] = (
            current_price
        )

        trade["pnl_percent"] = (
            pnl_percent
        )

        trade["closed_at"] = (
            int(time.time())
        )

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
            f"{emoji} "
            f"SCORE HUNTER PRO\n\n"
            f"TRADE CLOSED\n\n"
            f"💰 {symbol}USDT\n"
            f"📊 {direction}\n"
            f"⭐ 4H Score: "
            f"{trade['score_4h']}/7\n"
            f"📱 1H Score: "
            f"{trade['score_1h']}/3\n"
            f"📌 Result: "
            f"{result}\n\n"
            f"💵 Entry: "
            f"{entry:.8f}\n"
            f"🏁 Exit: "
            f"{current_price:.8f}\n"
            f"📈 P/L: "
            f"{pnl_percent:+.2f}%\n\n"
            f"🎯 TP: 1.00%\n"
            f"⏱ 4H + 1H\n"
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
        wins
        + losses
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


# ============================================================
# STATISTICS TELEGRAM
# ============================================================

def send_statistics_if_needed(
    state,
):

    stats = get_statistics(
        state
    )

    if not stats:
        return

    # Every 10 completed trades
    if (
        stats["total"] % 10
        != 0
    ):
        return

    message = (
        "📊 SCORE HUNTER PRO\n\n"
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
# MAIN
# ============================================================

def main():

    print(
        "🟢 SCORE HUNTER PRO"
    )

    print(
        "🛡 Multi-timeframe "
        "system enabled"
    )

    print(
        "📊 Main structure: 4H"
    )

    print(
        "📱 Entry confirmation: 1H"
    )

    print(
        "🎯 Fixed TP: 1.00%"
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
        f"⏳ Signal expiration: "
        f"{SIGNAL_EXPIRATION_HOURS}h"
    )

    state = load_state()

    # ========================================================
    # CHECK ACTIVE TRADES FIRST
    # ========================================================

    check_active_trades(
        state
    )

    # ========================================================
    # SCAN COINS
    # ========================================================

    for symbol, lbank_symbol in (
        COINS.items()
    ):

        print(
            f"\n========== "
            f"{symbol} =========="
        )

        try:

            # ------------------------------------------------
            # 4H
            # ------------------------------------------------

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

            if len(
                closed_4h
            ) < 210:

                print(
                    f"{symbol}: "
                    f"not enough 4H data"
                )

                continue

            latest_4h = (
                closed_4h[-1]
            )

            print(
                f"{symbol}: "
                f"latest closed 4H: "
                f"{latest_4h['time']}"
            )

            # ------------------------------------------------
            # 4H SIGNAL
            # ------------------------------------------------

            signal = (
                calculate_4h_signal(
                    closed_4h
                )
            )

            if signal is None:

                print(
                    f"{symbol}: "
                    f"no valid 4H setup"
                )

                continue

            direction = (
                signal["direction"]
            )

            score_4h = (
                signal["score"]
            )

            entry = (
                signal["entry"]
            )

            candle_time = (
                signal["candle_time"]
            )

            # ------------------------------------------------
            # EXPIRATION
            # ------------------------------------------------

            if signal_is_expired(
                candle_time
            ):

                print(
                    f"{symbol}: "
                    f"4H signal expired"
                )

                continue

            # ------------------------------------------------
            # 1H
            # ------------------------------------------------

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
                calculate_1h_confirmation(
                    closed_1h,
                    direction,
                )
            )

            score_1h = (
                confirmation["score"]
            )

            if not confirmation[
                "valid"
            ]:

                print(
                    f"{symbol}: "
                    f"1H confirmation "
                    f"failed "
                    f"({score_1h}/3)"
                )

                continue

            # ------------------------------------------------
            # CURRENT PRICE
            # ------------------------------------------------

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
                f"4H={score_4h}/7 "
                f"1H={score_1h}/3 "
                f"entry={entry} "
                f"current={current_price} "
                f"distance="
                f"{distance:.3f}%"
            )

            # ------------------------------------------------
            # ENTRY WINDOW
            # ------------------------------------------------

            if not entry_is_valid(
                entry,
                current_price,
            ):

                print(
                    f"{symbol}: "
                    f"entry too far"
                )

                continue

            # ------------------------------------------------
            # DUPLICATE PROTECTION
            # ------------------------------------------------

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
                    f"signal already sent"
                )

                continue

            # ------------------------------------------------
            # ACTIVE TRADE PROTECTION
            # ------------------------------------------------

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
                    f"active trade exists"
                )

                continue

            # ------------------------------------------------
            # RISK LEVELS
            # ------------------------------------------------

            levels = (
                calculate_risk_levels(
                    signal
                )
            )

            if not levels[
                "valid"
            ]:

                print(
                    f"{symbol}: "
                    f"risk filter "
                    f"rejected: "
                    f"{levels['reason']}"
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

            # ------------------------------------------------
            # FINAL INFORMATION
            # ------------------------------------------------

            strength = (
                signal_strength(
                    score_4h,
                    score_1h,
                )
            )

            print(
                f"{symbol}: "
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
                f"{score_1h}/3"
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
                f"R:R: "
                f"1:{rr:.2f}"
            )

            # ------------------------------------------------
            # SAVE ACTIVE TRADE
            # ------------------------------------------------

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

                "candle_time":
                    candle_time,

                "created_at":
                    int(time.time()),

                "status":
                    "ACTIVE",
            }

            state.setdefault(
                "active_trades",
                {},
            )[
                signal_id
            ] = trade

            state.setdefault(
                "last_signals",
                {},
            )[symbol] = (
                signal_id
            )

            save_state(
                state
            )

            # ------------------------------------------------
            # TELEGRAM MESSAGE
            # ------------------------------------------------

            if direction == "LONG":

                direction_text = (
                    "🟢 LONG"
                )

                tp_percent = (
                    (
                        tp - entry
                    )
                    / entry
                    * 100
                )

                sl_percent = (
                    (
                        sl - entry
                    )
                    / entry
                    * 100
                )

            else:

                direction_text = (
                    "🔴 SHORT"
                )

                tp_percent = (
                    (
                        entry - tp
                    )
                    / entry
                    * 100
                )

                sl_percent = (
                    (
                        entry - sl
                    )
                    / entry
                    * 100
                )

            message = (

                "🚨 SCORE HUNTER PRO 🚨\n\n"

                f"💰 {symbol}USDT\n"

                f"📊 {direction_text}\n"

                f"{strength}\n\n"

                f"⭐ 4H Score: "
                f"{score_4h}/7\n"

                f"📱 1H Confirm: "
                f"{score_1h}/3\n\n"

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

                "🔎 Structure: 4H\n"

                "📱 Entry confirmation: 1H\n"

                "🎯 TP target: 1.00%\n"

                "🔒 Entry window: ±0.30%\n"

                "⏱ Timeframe: 4H + 1H\n"

                "🏦 Data: LBank\n\n"

                "⚠️ Manage risk."
            )

            sent = send_telegram(
                message
            )

            if sent:

                print(
                    f"{symbol}: "
                    f"{direction} "
                    f"SENT"
                )

            else:

                print(
                    f"{symbol}: "
                    f"Telegram failed"
                )

        except Exception as e:

            print(
                f"{symbol}: "
                f"ERROR: {e}"
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
        "\n✅ SCORE HUNTER PRO "
        "scan completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
