import os
import json
import time
import requests

# ============================================================
# SCORE HUNTER 4H
# DATA SOURCE: LBANK
# COINS: BTC / ETH / SOL / XRP
# STRATEGY: 7 FACTORS - MIN SCORE 5/7
# TP: 1% | SL: 0.5%
# CLOSED CANDLE ONLY
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "2090120004")

LBANK_KLINE_URL = "https://api.lbkex.com/v2/kline.do"

COINS = {
    "BTC": "btc_usdt",
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
}

TIMEFRAME = "hour4"
CANDLE_LIMIT = 250

REQUIRED_SCORE = 5

TP_PERCENT = 1.0
SL_PERCENT = 0.50

STATE_FILE = "state.json"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message,
        },
        timeout=20,
    )

    print("Telegram:", response.status_code)
    print(response.text)


# ============================================================
# STATE
# ============================================================

def load_state():

    if not os.path.exists(STATE_FILE):
        return {}

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print("State load error:", e)
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
# LBANK 4H DATA
# ============================================================

def get_4h_candles(symbol):

    print(
        f"Getting {symbol} 4H candles from LBank..."
    )

    now = int(time.time())

    start_time = (
        now
        - (CANDLE_LIMIT * 4 * 60 * 60)
    )

    response = requests.get(
        LBANK_KLINE_URL,
        params={
            "symbol": symbol,
            "size": CANDLE_LIMIT,
            "type": TIMEFRAME,
            "time": start_time,
        },
        timeout=20,
    )

    print("LBank:", response.status_code)

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
        []
    )

    if not raw_data:

        raise RuntimeError(
            f"No candle data returned for {symbol}"
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
# CLOSED CANDLES ONLY
# ============================================================

def get_closed_candles(candles):

    if len(candles) < 3:

        raise RuntimeError(
            "Not enough candles."
        )

    # آخرین کندل ممکن است هنوز در حال تشکیل باشد.
    # بنابراین برای سیگنال استفاده نمی‌شود.
    return candles[:-1]


# ============================================================
# EMA
# ============================================================

def ema_series(values, period):

    if len(values) < period:
        return []

    multiplier = 2 / (period + 1)

    first = (
        sum(values[:period])
        / period
    )

    result = [None] * (
        period - 1
    )

    result.append(first)

    previous = first

    for price in values[period:]:

        current = (
            (price - previous)
            * multiplier
            + previous
        )

        result.append(current)

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

    result = [None] * period

    if avg_loss == 0:

        result.append(100.0)

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
        len(gains)
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

    result = [None] * period

    result.append(first_atr)

    previous = first_atr

    for tr in true_ranges[period:]:

        current = (
            (
                previous
                * (period - 1)
            )
            + tr
        ) / period

        result.append(current)

        previous = current

    return result


# ============================================================
# SIGNAL
# ============================================================

def calculate_signal(candles):

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

    atr_values = atr_series(
        candles,
        14
    )

    i = len(candles) - 1

    current = candles[i]

    close = current["close"]
    open_price = current["open"]
    high = current["high"]
    low = current["low"]
    volume = current["volume"]

    e20 = ema20[i]
    e50 = ema50[i]
    e200 = ema200[i]

    rsi = rsi_values[-1]
    previous_rsi = rsi_values[-2]

    atr = atr_values[-1]

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or rsi is None
        or previous_rsi is None
        or atr is None
    ):

        return None

    # ========================================================
    # 1. TREND
    # ========================================================

    long_trend = (
        close > e200
        and e20 > e50
    )

    short_trend = (
        close < e200
        and e20 < e50
    )

    # ========================================================
    # 2. RSI
    # ========================================================

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

    # ========================================================
    # 3. VOLUME
    # ========================================================

    volume_ma = (
        sum(volumes[-20:])
        / 20
    )

    volume_ok = (
        volume >= volume_ma
    )

    # ========================================================
    # 4. MARKET STRUCTURE
    # ========================================================

    recent_high = max(
        c["high"]
        for c in candles[-7:-1]
    )

    recent_low = min(
        c["low"]
        for c in candles[-7:-1]
    )

    bull_break = (
        close > recent_high
    )

    bear_break = (
        close < recent_low
    )

    # ========================================================
    # 5. EMA PULLBACK
    # ========================================================

    long_pullback = (
        low <= e20
        and close > e20
    )

    short_pullback = (
        high >= e20
        and close < e20
    )

    # ========================================================
    # 6. CANDLE CONFIRMATION
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

        bull_ratio = 0
        bear_ratio = 0

    bull_candle = (
        close > open_price
        and candle_range > 0
        and bull_ratio >= 0.40
    )

    bear_candle = (
        close < open_price
        and candle_range > 0
        and bear_ratio >= 0.40
    )

    # ========================================================
    # 7. VOLATILITY
    # ========================================================

    volatility_ok = (
        atr / close >= 0.002
    )

    # ========================================================
    # SCORE
    # ========================================================

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

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    if long_score >= REQUIRED_SCORE:

        return {
            "direction": "LONG",
            "score": long_score,
            "entry": close,
        }

    if short_score >= REQUIRED_SCORE:

        return {
            "direction": "SHORT",
            "score": short_score,
            "entry": close,
        }

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🟢 SCORE HUNTER 4H - LBANK SCANNING"
    )

    print(
        "🕯 CLOSED CANDLE ONLY - NO MID-CANDLE SIGNAL"
    )

    print(
        "📊 Coins: BTC / ETH / SOL / XRP"
    )

    print(
        "⏱ Timeframe: 4H"
    )

    state = load_state()

    for symbol, lbank_symbol in COINS.items():

        print(
            f"\n========== {symbol} =========="
        )

        try:

            candles = get_4h_candles(
                lbank_symbol
            )

            # =================================================
            # فقط کندل‌های کاملاً بسته‌شده
            # =================================================

            closed_candles = (
                get_closed_candles(
                    candles
                )
            )

            latest = (
                closed_candles[-1]
            )

            print(
                f"{symbol}: latest CLOSED "
                f"4H candle: "
                f"{latest['time']}"
            )

            signal = calculate_signal(
                closed_candles
            )

            if signal is None:

                print(
                    f"{symbol}: no signal"
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

            # =================================================
            # DUPLICATE PROTECTION
            # =================================================

            signal_id = (
                f"{latest['time']}_{direction}"
            )

            previous_signal = (
                state
                .get(symbol, {})
                .get("last_signal")
            )

            if previous_signal == signal_id:

                print(
                    f"{symbol}: signal "
                    f"already sent"
                )

                continue

            # =================================================
            # TP / SL
            # =================================================

            if direction == "LONG":

                tp = (
                    entry
                    * (
                        1
                        + TP_PERCENT
                        / 100
                    )
                )

                sl = (
                    entry
                    * (
                        1
                        - SL_PERCENT
                        / 100
                    )
                )

                direction_text = (
                    "🟢 LONG"
                )

                tp_text = (
                    f"+{TP_PERCENT:.0f}%"
                )

                sl_text = (
                    f"-{SL_PERCENT:.1f}%"
                )

            else:

                tp = (
                    entry
                    * (
                        1
                        - TP_PERCENT
                        / 100
                    )
                )

                sl = (
                    entry
                    * (
                        1
                        + SL_PERCENT
                        / 100
                    )
                )

                direction_text = (
                    "🔴 SHORT"
                )

                tp_text = (
                    f"-{TP_PERCENT:.0f}%"
                )

                sl_text = (
                    f"+{SL_PERCENT:.1f}%"
                )

            # =================================================
            # TELEGRAM
            # =================================================

            message = (
                "🚨 SCORE HUNTER 4H 🚨\n\n"
                f"💰 {symbol}USDT\n"
                f"📊 {direction_text}\n"
                f"⭐ Score: {score}/7\n"
                f"💵 Entry: {entry:.8f}\n"
                f"🎯 TP: {tp:.8f} "
                f"({tp_text})\n"
                f"🛑 SL: {sl:.8f} "
                f"({sl_text})\n\n"
                "⏱ Timeframe: 4H\n"
                "🕯 Signal: CLOSED CANDLE\n"
                "🏦 Data: LBank\n"
                "⚠️ Manage risk."
            )

            send_telegram(message)

            # =================================================
            # SAVE
            # =================================================

            state[symbol] = {
                "last_signal": signal_id,
                "last_signal_time": latest["time"],
                "direction": direction,
                "score": score,
                "entry": entry,
                "tp": tp,
                "sl": sl,
            }

            save_state(state)

            print(
                f"{symbol}: "
                f"{direction} "
                f"Score {score}/7 SENT"
            )

        except Exception as e:

            print(
                f"{symbol}: ERROR: {e}"
            )

    save_state(state)

    print(
        "\n✅ LBank 4H scan completed."
    )


if __name__ == "__main__":
    main()
