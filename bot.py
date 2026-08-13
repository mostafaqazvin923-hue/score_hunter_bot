import os
import json
import requests
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER PRO
# BINANCE USDⓈ-M FUTURES
#
# COINS:
# ETH / SOL / XRP / APT
#
# TIMEFRAME:
# 4H
#
# PINE LOGIC:
# EMA20 / EMA50 / EMA200
# RSI14
# Volume SMA20
# Highest/Lowest 6 previous candles
# EMA20 Pullback
# Candle Confirmation
# ATR14
#
# MIN SCORE: 5/7
# TP: 1%
# SL: 0.5%
#
# CLOSED CANDLE ONLY
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
# BINANCE FUTURES ENDPOINTS
#
# We try multiple official Futures REST hosts.
# This is useful when one Binance gateway returns 451.
# ============================================================

BINANCE_FUTURES_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]


KLINE_PATH = "/fapi/v1/klines"
TIME_PATH = "/fapi/v1/time"


# ============================================================
# REQUEST SETTINGS
# ============================================================

REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# ============================================================
# SYMBOLS
# ============================================================

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "APT": "APTUSDT",
}


# ============================================================
# SETTINGS
# ============================================================

INTERVAL = "4h"

CANDLE_LIMIT = 1000

REQUIRED_SCORE = 5

TP_PERCENT = 1.0
SL_PERCENT = 0.50

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
        timeout=REQUEST_TIMEOUT,
    )

    print(
        "Telegram:",
        response.status_code
    )

    print(
        response.text
    )

    response.raise_for_status()


# ============================================================
# STATE
# ============================================================

def load_state():

    if not os.path.exists(
        STATE_FILE
    ):
        return {}

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            "State load error:",
            e
        )

        return {}


def save_state(state):

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )


# ============================================================
# BINANCE REQUEST HELPER
# ============================================================

def binance_get(path, params=None):

    last_error = None

    for base in BINANCE_FUTURES_BASES:

        url = (
            base
            +
            path
        )

        print(
            f"Trying Binance endpoint: "
            f"{base}"
        )

        try:

            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            print(
                f"HTTP {response.status_code}"
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                return response.json()

            # ------------------------------------------------
            # GEO / ACCESS BLOCK
            # ------------------------------------------------

            if response.status_code == 451:

                print(
                    f"451 from {base} "
                    f"→ trying next endpoint..."
                )

                last_error = RuntimeError(
                    f"HTTP 451 from {base}"
                )

                continue

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code in (
                418,
                429,
            ):

                print(
                    f"Rate limit "
                    f"from {base}"
                )

                last_error = RuntimeError(
                    f"HTTP {response.status_code}"
                )

                continue

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            try:

                error_body = (
                    response.json()
                )

            except Exception:

                error_body = (
                    response.text
                )

            last_error = RuntimeError(
                f"Binance HTTP "
                f"{response.status_code}: "
                f"{error_body}"
            )

        except Exception as e:

            print(
                f"Connection error "
                f"on {base}: {e}"
            )

            last_error = e

    raise RuntimeError(
        "ALL BINANCE FUTURES "
        "ENDPOINTS FAILED.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# BINANCE SERVER TIME
# ============================================================

def get_binance_server_time():

    data = binance_get(
        TIME_PATH
    )

    return int(
        data["serverTime"]
    )


# ============================================================
# GET BINANCE FUTURES KLINES
# ============================================================

def get_4h_candles(symbol):

    print(
        f"Getting {symbol} "
        f"4H candles from "
        f"Binance USDⓈ-M Futures..."
    )

    raw_data = binance_get(
        KLINE_PATH,
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": CANDLE_LIMIT,
        }
    )

    if not isinstance(
        raw_data,
        list
    ):

        raise RuntimeError(
            f"Unexpected Binance response: "
            f"{raw_data}"
        )

    candles = []

    for item in raw_data:

        if len(item) < 7:
            continue

        candles.append(
            {
                "time": int(item[0]),

                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),

                "close_time": int(item[6]),
            }
        )

    candles.sort(
        key=lambda x: x["time"]
    )

    if not candles:

        raise RuntimeError(
            f"No candles returned "
            f"for {symbol}"
        )

    print(
        f"{symbol}: received "
        f"{len(candles)} candles"
    )

    return candles


# ============================================================
# CLOSED CANDLES ONLY
# ============================================================

def get_closed_candles(candles):

    if len(candles) < 3:

        raise RuntimeError(
            "Not enough candles."
        )

    server_time = (
        get_binance_server_time()
    )

    closed = []

    for candle in candles:

        if (
            candle["close_time"]
            <
            server_time
        ):

            closed.append(
                candle
            )

    if len(closed) < 210:

        raise RuntimeError(
            "Not enough CLOSED "
            "candles."
        )

    return closed


# ============================================================
# EMA
# ============================================================

def ema_series(
    values,
    period
):

    if len(values) < period:

        return []

    alpha = (
        2.0 /
        (period + 1.0)
    )

    first = (
        sum(values[:period])
        /
        period
    )

    result = [None] * (
        period - 1
    )

    result.append(
        first
    )

    previous = first

    for price in values[period:]:

        current = (
            alpha * price
            +
            (1.0 - alpha)
            * previous
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
    period=14
):

    if len(closes) <= period:

        return []

    gains = []
    losses = []

    for i in range(
        1,
        len(closes)
    ):

        change = (
            closes[i]
            -
            closes[i - 1]
        )

        gains.append(
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
        )

    if len(gains) < period:

        return []

    avg_gain = (
        sum(gains[:period])
        /
        period
    )

    avg_loss = (
        sum(losses[:period])
        /
        period
    )

    result = [None] * period

    if avg_loss == 0:

        result.append(
            100.0
        )

    else:

        rs = (
            avg_gain
            /
            avg_loss
        )

        result.append(
            100.0
            -
            (
                100.0
                /
                (1.0 + rs)
            )
        )

    for i in range(
        period,
        len(gains)
    ):

        avg_gain = (
            (
                avg_gain
                *
                (period - 1)
            )
            +
            gains[i]
        )
        / period

        avg_loss = (
            (
                avg_loss
                *
                (period - 1)
            )
            +
            losses[i]
        )
        / period

        if avg_loss == 0:

            value = 100.0

        else:

            rs = (
                avg_gain
                /
                avg_loss
            )

            value = (
                100.0
                -
                (
                    100.0
                    /
                    (1.0 + rs)
                )
            )

        result.append(
            value
        )

    return result


# ============================================================
# SMA
# ============================================================

def sma_series(
    values,
    period
):

    if len(values) < period:

        return []

    result = [None] * (
        period - 1
    )

    running_sum = sum(
        values[:period]
    )

    result.append(
        running_sum / period
    )

    for i in range(
        period,
        len(values)
    ):

        running_sum += (
            values[i]
        )

        running_sum -= (
            values[
                i - period
            ]
        )

        result.append(
            running_sum / period
        )

    return result


# ============================================================
# ATR / RMA
# ============================================================

def atr_series(
    candles,
    period=14
):

    if len(candles) <= period:

        return []

    true_ranges = []

    for i in range(
        1,
        len(candles)
    ):

        high = (
            candles[i]["high"]
        )

        low = (
            candles[i]["low"]
        )

        previous_close = (
            candles[i - 1]["close"]
        )

        tr = max(
            high - low,
            abs(
                high
                -
                previous_close
            ),
            abs(
                low
                -
                previous_close
            ),
        )

        true_ranges.append(
            tr
        )

    if len(true_ranges) < period:

        return []

    first_atr = (
        sum(
            true_ranges[:period]
        )
        /
        period
    )

    result = [None] * period

    result.append(
        first_atr
    )

    previous = first_atr

    for tr in true_ranges[period:]:

        current = (
            (
                previous
                *
                (period - 1)
            )
            +
            tr
        ) / period

        result.append(
            current
        )

        previous = current

    return result


# ============================================================
# SIGNAL
# ============================================================

def calculate_signal(
    candles
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
        20
    )

    ema50 = ema_series(
        closes,
        50
    )

    ema200 = ema_series(
        closes,
        200
    )

    rsi_values = rsi_series(
        closes,
        14
    )

    volume_ma = sma_series(
        volumes,
        20
    )

    atr_values = atr_series(
        candles,
        14
    )

    i = len(candles) - 1

    current = candles[i]

    open_price = (
        current["open"]
    )

    high = (
        current["high"]
    )

    low = (
        current["low"]
    )

    close = (
        current["close"]
    )

    e20 = ema20[i]
    e50 = ema50[i]
    e200 = ema200[i]

    rsi = rsi_values[i]
    previous_rsi = (
        rsi_values[i - 1]
    )

    vol_ma = volume_ma[i]

    atr = atr_values[i]

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or rsi is None
        or previous_rsi is None
        or vol_ma is None
        or atr is None
    ):

        return None

    # ========================================================
    # 1 TREND
    # ========================================================

    long_trend = (
        close > e200
        and
        e20 > e50
    )

    short_trend = (
        close < e200
        and
        e20 < e50
    )

    # ========================================================
    # 2 RSI
    # ========================================================

    long_rsi = (
        rsi > 50
        and
        rsi < 72
        and
        rsi > previous_rsi
    )

    short_rsi = (
        rsi < 50
        and
        rsi > 28
        and
        rsi < previous_rsi
    )

    # ========================================================
    # 3 VOLUME
    # ========================================================

    volume_ok = (
        current["volume"]
        >=
        vol_ma
    )

    # ========================================================
    # 4 MARKET STRUCTURE
    #
    # Pine:
    #
    # ta.highest(high, 6)[1]
    #
    # Previous 6 candles only.
    # ========================================================

    previous_6 = candles[
        -7:-1
    ]

    recent_high = max(
        c["high"]
        for c in previous_6
    )

    recent_low = min(
        c["low"]
        for c in previous_6
    )

    bull_break = (
        close > recent_high
    )

    bear_break = (
        close < recent_low
    )

    # ========================================================
    # 5 EMA PULLBACK
    # ========================================================

    long_pullback = (
        low <= e20
        and
        close > e20
    )

    short_pullback = (
        high >= e20
        and
        close < e20
    )

    # ========================================================
    # 6 CANDLE
    # ========================================================

    candle_range = (
        high - low
    )

    if candle_range > 0:

        bull_ratio = (
            close - open_price
        ) / candle_range

        bear_ratio = (
            open_price - close
        ) / candle_range

    else:

        bull_ratio = 0.0
        bear_ratio = 0.0

    bull_candle = (
        close > open_price
        and
        candle_range > 0
        and
        bull_ratio >= 0.40
    )

    bear_candle = (
        close < open_price
        and
        candle_range > 0
        and
        bear_ratio >= 0.40
    )

    # ========================================================
    # 7 VOLATILITY
    # ========================================================

    volatility_ok = (
        atr / close >= 0.002
    )

    # ========================================================
    # SCORE
    # ========================================================

    long_score = (
        int(long_trend)
        +
        int(long_rsi)
        +
        int(volume_ok)
        +
        int(bull_break)
        +
        int(long_pullback)
        +
        int(bull_candle)
        +
        int(volatility_ok)
    )

    short_score = (
        int(short_trend)
        +
        int(short_rsi)
        +
        int(volume_ok)
        +
        int(bear_break)
        +
        int(short_pullback)
        +
        int(bear_candle)
        +
        int(volatility_ok)
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print(
        f"Close       : {close}"
    )

    print(
        f"EMA20       : {e20}"
    )

    print(
        f"EMA50       : {e50}"
    )

    print(
        f"EMA200      : {e200}"
    )

    print(
        f"RSI         : {rsi:.6f}"
    )

    print(
        f"Previous RSI: {previous_rsi:.6f}"
    )

    print(
        f"Volume      : "
        f"{current['volume']}"
    )

    print(
        f"Volume MA20 : "
        f"{vol_ma}"
    )

    print(
        f"Recent High : "
        f"{recent_high}"
    )

    print(
        f"Recent Low  : "
        f"{recent_low}"
    )

    print(
        f"ATR         : "
        f"{atr}"
    )

    print(
        f"ATR/Close   : "
        f"{atr / close:.8f}"
    )

    print(
        f"LONG SCORE  : "
        f"{long_score}/7"
    )

    print(
        f"SHORT SCORE : "
        f"{short_score}/7"
    )

    # ========================================================
    # FINAL
    # ========================================================

    if long_score >= REQUIRED_SCORE:

        return {
            "direction": "LONG",
            "score": long_score,
            "entry": close,
            "candle_time": current["time"],
        }

    if short_score >= REQUIRED_SCORE:

        return {
            "direction": "SHORT",
            "score": short_score,
            "entry": close,
            "candle_time": current["time"],
        }

    return None


# ============================================================
# PRICE FORMAT
# ============================================================

def format_price(price):

    if price >= 1000:
        return f"{price:.2f}"

    if price >= 100:
        return f"{price:.3f}"

    if price >= 10:
        return f"{price:.4f}"

    if price >= 1:
        return f"{price:.5f}"

    if price >= 0.1:
        return f"{price:.6f}"

    return f"{price:.8f}"


# ============================================================
# TIME FORMAT
# ============================================================

def format_candle_time(
    timestamp_ms
):

    dt = datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc
    )

    return dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "🟢 SCORE HUNTER PRO\n"
        "🏦 BINANCE USDⓈ-M FUTURES\n"
        "🔒 CLOSED CANDLE MODE\n"
        "⏱ TIMEFRAME: 4H\n"
        "⭐ MIN SCORE: 5/7\n"
        "💰 ETH / SOL / XRP / APT\n"
    )

    state = load_state()

    for symbol, binance_symbol in COINS.items():

        print(
            "\n"
            "================================"
        )

        print(
            f"========== {symbol} =========="
        )

        print(
            "================================"
        )

        try:

            # ------------------------------------------------
            # GET CANDLES
            # ------------------------------------------------

            candles = get_4h_candles(
                binance_symbol
            )

            # ------------------------------------------------
            # CLOSED ONLY
            # ------------------------------------------------

            closed_candles = (
                get_closed_candles(
                    candles
                )
            )

            latest = (
                closed_candles[-1]
            )

            print(
                f"{symbol}: latest "
                f"CLOSED 4H candle:"
            )

            print(
                format_candle_time(
                    latest["time"]
                )
            )

            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            signal = calculate_signal(
                closed_candles
            )

            if signal is None:

                print(
                    f"{symbol}: NO SIGNAL"
                )

                continue

            direction = (
                signal["direction"]
            )

            score = (
                signal["score"]
            )

            entry = (
                signal["entry"]
            )

            candle_time = (
                signal["candle_time"]
            )

            # ------------------------------------------------
            # DUPLICATE PROTECTION
            # ------------------------------------------------

            signal_id = (
                f"{candle_time}_"
                f"{direction}"
            )

            previous_signal = (
                state
                .get(symbol, {})
                .get("last_signal")
            )

            if (
                previous_signal
                ==
                signal_id
            ):

                print(
                    f"{symbol}: "
                    f"signal already sent"
                )

                continue

            # ------------------------------------------------
            # TP / SL
            # ------------------------------------------------

            if direction == "LONG":

                tp = (
                    entry
                    *
                    (
                        1
                        +
                        TP_PERCENT / 100
                    )
                )

                sl = (
                    entry
                    *
                    (
                        1
                        -
                        SL_PERCENT / 100
                    )
                )

                direction_text = (
                    "🟢 LONG"
                )

                tp_text = (
                    f"+{TP_PERCENT:.1f}%"
                )

                sl_text = (
                    f"-{SL_PERCENT:.1f}%"
                )

            else:

                tp = (
                    entry
                    *
                    (
                        1
                        -
                        TP_PERCENT / 100
                    )
                )

                sl = (
                    entry
                    *
                    (
                        1
                        +
                        SL_PERCENT / 100
                    )
                )

                direction_text = (
                    "🔴 SHORT"
                )

                tp_text = (
                    f"-{TP_PERCENT:.1f}%"
                )

                sl_text = (
                    f"+{SL_PERCENT:.1f}%"
                )

            # ------------------------------------------------
            # TELEGRAM
            # ------------------------------------------------

            message = (
                "🚨 SCORE HUNTER PRO 🚨\n\n"

                f"💰 {symbol}USDT\n"

                f"📊 {direction_text}\n"

                f"⭐ Score: {score}/7\n"

                f"💵 Entry: "
                f"{format_price(entry)}\n"

                f"🎯 TP: "
                f"{format_price(tp)} "
                f"({tp_text})\n"

                f"🛑 SL: "
                f"{format_price(sl)} "
                f"({sl_text})\n\n"

                "⏱ Timeframe: 4H\n"

                "🏦 Data: Binance "
                "USDⓈ-M Futures\n"

                "🔒 CLOSED CANDLE\n"

                f"🕐 Candle: "
                f"{format_candle_time(candle_time)}\n\n"

                "📐 Strategy: "
                "7-Factor Score Hunter\n"

                "⚠️ Manage risk."
            )

            send_telegram(
                message
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            state[symbol] = {

                "last_signal":
                    signal_id,

                "last_signal_time":
                    candle_time,

                "direction":
                    direction,

                "score":
                    score,

                "entry":
                    entry,

                "tp":
                    tp,

                "sl":
                    sl,
            }

            save_state(
                state
            )

            print(
                f"{symbol}: "
                f"{direction} "
                f"Score {score}/7 "
                f"SENT"
            )

        except Exception as e:

            print(
                f"{symbol}: ERROR: {e}"
            )

    save_state(
        state
    )

    print(
        "\n"
        "✅ Binance Futures "
        "4H scan completed."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
