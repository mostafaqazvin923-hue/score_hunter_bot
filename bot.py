import os
import sys
import json
import time
import subprocess


# ============================================================
# AUTO INSTALL REQUIRED PACKAGES
# ============================================================

def ensure_package(package_name, import_name=None):

    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        return

    except ImportError:
        print(
            f"Installing missing package: {package_name}"
        )

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package_name,
            ]
        )


ensure_package(
    "requests",
    "requests"
)

ensure_package(
    "websocket-client",
    "websocket"
)


import requests
import websocket


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
# CLOSED CANDLE ONLY
#
# STRATEGY:
# EXACT 7 FACTOR PINE LOGIC
#
# MIN SCORE:
# 5/7
#
# TP:
# +1%
#
# SL:
# 0.5%
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
# BINANCE
# ============================================================

REST_URL = (
    "https://fapi.binance.com"
)

WEBSOCKET_URL = (
    "wss://fstream.binance.com/ws"
)


# ============================================================
# COINS
# ============================================================

COINS = {
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
    "APT": "aptusdt",
}


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "4h"

CANDLE_LIMIT = 500

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
        timeout=20,
    )

    print(
        "Telegram:",
        response.status_code
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
            "State error:",
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
# BINANCE REST
#
# HISTORY ONLY
#
# If REST is blocked, the error is clearly displayed.
# ============================================================

def get_history(symbol):

    print(
        f"Getting {symbol.upper()} "
        f"4H history..."
    )

    url = (
        REST_URL
        +
        "/fapi/v1/klines"
    )

    response = requests.get(
        url,
        params={
            "symbol": symbol.upper(),
            "interval": TIMEFRAME,
            "limit": CANDLE_LIMIT,
        },
        headers={
            "User-Agent":
                "Mozilla/5.0"
        },
        timeout=20,
    )

    print(
        "Binance REST:",
        response.status_code
    )

    if response.status_code != 200:

        raise RuntimeError(
            "Binance REST history unavailable: "
            f"HTTP {response.status_code}\n"
            f"{response.text[:300]}"
        )

    data = response.json()

    candles = []

    for item in data:

        candles.append({
            "time": int(item[0]),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "close_time": int(item[6]),
        })

    candles.sort(
        key=lambda x: x["time"]
    )

    return candles


# ============================================================
# CLOSED CANDLES
# ============================================================

def remove_forming_candle(
    candles
):

    if len(candles) < 3:

        raise RuntimeError(
            "Not enough candles."
        )

    now_ms = (
        int(time.time())
        * 1000
    )

    closed = [
        c
        for c in candles
        if c["close_time"]
        <= now_ms
    ]

    if len(closed) < 210:

        raise RuntimeError(
            "Not enough closed candles."
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

    multiplier = (
        2.0
        /
        (period + 1.0)
    )

    first = (
        sum(values[:period])
        /
        period
    )

    result = (
        [None]
        *
        (period - 1)
    )

    result.append(first)

    previous = first

    for price in values[period:]:

        current = (
            (
                price
                -
                previous
            )
            *
            multiplier
            +
            previous
        )

        result.append(
            current
        )

        previous = current

    return result


# ============================================================
# RSI
# TradingView Wilder RMA
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

    result = (
        [None]
        *
        period
    )

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
        ) / period

        avg_loss = (
            (
                avg_loss
                *
                (period - 1)
            )
            +
            losses[i]
        ) / period

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

    result = (
        [None]
        *
        (period - 1)
    )

    for i in range(
        period - 1,
        len(values)
    ):

        window = values[
            i - period + 1:
            i + 1
        ]

        result.append(
            sum(window)
            /
            period
        )

    return result


# ============================================================
# ATR
# TradingView Wilder RMA
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

        high = candles[i]["high"]

        low = candles[i]["low"]

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

        true_ranges.append(tr)

    if len(true_ranges) < period:

        return []

    first_atr = (
        sum(
            true_ranges[:period]
        )
        /
        period
    )

    result = (
        [None]
        *
        period
    )

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
#
# SAME LOGIC AS THE PINE CODE
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

    rsi = rsi_series(
        closes,
        14
    )

    volume_ma = sma_series(
        volumes,
        20
    )

    atr = atr_series(
        candles,
        14
    )

    i = len(candles) - 1

    current = candles[i]

    open_price = current["open"]

    high = current["high"]

    low = current["low"]

    close = current["close"]

    e20 = ema20[i]

    e50 = ema50[i]

    e200 = ema200[i]

    current_rsi = rsi[i]

    previous_rsi = rsi[i - 1]

    current_volume_ma = (
        volume_ma[i]
    )

    current_atr = atr[i]

    if any(
        x is None
        for x in [
            e20,
            e50,
            e200,
            current_rsi,
            previous_rsi,
            current_volume_ma,
            current_atr,
        ]
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
        current_rsi > 50
        and
        current_rsi < 72
        and
        current_rsi > previous_rsi
    )

    short_rsi = (
        current_rsi < 50
        and
        current_rsi > 28
        and
        current_rsi < previous_rsi
    )

    # ========================================================
    # 3 VOLUME
    # ========================================================

    volume_ok = (
        current["volume"]
        >=
        current_volume_ma
    )

    # ========================================================
    # 4 MARKET STRUCTURE
    #
    # Pine:
    #
    # ta.highest(high, 6)[1]
    #
    # Previous 6 candles
    # ========================================================

    previous_six = (
        candles[-7:-1]
    )

    recent_high = max(
        c["high"]
        for c in previous_six
    )

    recent_low = min(
        c["low"]
        for c in previous_six
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
            close
            -
            open_price
        ) / candle_range

        bear_ratio = (
            open_price
            -
            close
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
        current_atr
        /
        close
        >=
        0.002
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

    print(
        f"Close={close}"
    )

    print(
        f"EMA20={e20:.8f} "
        f"EMA50={e50:.8f} "
        f"EMA200={e200:.8f}"
    )

    print(
        f"RSI={current_rsi:.4f} "
        f"PrevRSI={previous_rsi:.4f}"
    )

    print(
        f"VolumeOK={volume_ok}"
    )

    print(
        f"BullBreak={bull_break} "
        f"BearBreak={bear_break}"
    )

    print(
        f"LongPullback={long_pullback} "
        f"ShortPullback={short_pullback}"
    )

    print(
        f"BullCandle={bull_candle} "
        f"BearCandle={bear_candle}"
    )

    print(
        f"VolatilityOK={volatility_ok}"
    )

    print(
        f"LONG SCORE = {long_score}/7"
    )

    print(
        f"SHORT SCORE = {short_score}/7"
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

def format_price(
    price
):

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
# WEBSOCKET TEST
# ============================================================

def websocket_test():

    symbol = "ethusdt"

    url = (
        f"{WEBSOCKET_URL}/"
        f"{symbol}@kline_4h"
    )

    print(
        "\nTesting Binance Futures "
        "WebSocket..."
    )

    print(
        url
    )

    ws = None

    try:

        ws = websocket.create_connection(
            url,
            timeout=15,
            enableTrace=False,
        )

        print(
            "WebSocket CONNECTED ✅"
        )

        message = ws.recv()

        data = json.loads(
            message
        )

        if "k" in data:

            print(
                "Binance Futures "
                "Kline received ✅"
            )

        else:

            print(
                "WebSocket response:",
                data
            )

        return True

    except Exception as e:

        print(
            "WebSocket ERROR:",
            repr(e)
        )

        return False

    finally:

        if ws is not None:

            try:
                ws.close()
            except Exception:
                pass


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

    # --------------------------------------------------------
    # TEST WEBSOCKET
    # --------------------------------------------------------

    ws_ok = websocket_test()

    if not ws_ok:

        print(
            "\n❌ Binance Futures "
            "WebSocket is not accessible "
            "from this environment."
        )

        print(
            "Stopping before generating "
            "any false signal."
        )

        return

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    state = load_state()

    # --------------------------------------------------------
    # SCAN COINS
    # --------------------------------------------------------

    for name, symbol in COINS.items():

        print(
            "\n================================"
        )

        print(
            f"========== {name} =========="
        )

        print(
            "================================"
        )

        try:

            # HISTORY
            candles = get_history(
                symbol
            )

            # CLOSED ONLY
            closed = (
                remove_forming_candle(
                    candles
                )
            )

            print(
                f"{name}: "
                f"{len(closed)} closed "
                f"candles"
            )

            # SIGNAL
            signal = calculate_signal(
                closed
            )

            if signal is None:

                print(
                    f"{name}: NO SIGNAL"
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
            # DUPLICATE
            # ------------------------------------------------

            signal_id = (
                f"{candle_time}_"
                f"{direction}"
            )

            old_signal = (
                state
                .get(name, {})
                .get("last_signal")
            )

            if old_signal == signal_id:

                print(
                    f"{name}: "
                    "SIGNAL ALREADY SENT"
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
                        TP_PERCENT
                        /
                        100
                    )
                )

                sl = (
                    entry
                    *
                    (
                        1
                        -
                        SL_PERCENT
                        /
                        100
                    )
                )

                emoji = "🟢 LONG"

            else:

                tp = (
                    entry
                    *
                    (
                        1
                        -
                        TP_PERCENT
                        /
                        100
                    )
                )

                sl = (
                    entry
                    *
                    (
                        1
                        +
                        SL_PERCENT
                        /
                        100
                    )
                )

                emoji = "🔴 SHORT"

            # ------------------------------------------------
            # MESSAGE
            # ------------------------------------------------

            message = (
                "🚨 SCORE HUNTER PRO 🚨\n\n"
                f"💰 {name}USDT\n"
                f"📊 {emoji}\n"
                f"⭐ Score: {score}/7\n"
                f"💵 Entry: "
                f"{format_price(entry)}\n"
                f"🎯 TP: "
                f"{format_price(tp)} "
                f"(1%)\n"
                f"🛑 SL: "
                f"{format_price(sl)} "
                f"(0.5%)\n\n"
                "⏱ Timeframe: 4H\n"
                "🏦 Data: Binance "
                "USDⓈ-M Futures\n"
                "🔒 CLOSED CANDLE\n"
                "📐 7-Factor Score Hunter"
            )

            send_telegram(
                message
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            state[name] = {
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
                f"{name}: "
                f"{direction} "
                f"{score}/7 SENT ✅"
            )

        except Exception as e:

            print(
                f"{name}: ERROR:"
            )

            print(
                repr(e)
            )

    save_state(
        state
    )

    print(
        "\n"
        "✅ SCORE HUNTER PRO "
        "SCAN COMPLETED"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
