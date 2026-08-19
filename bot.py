import os
import json
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
STATE_FILE = "state.json"

INTERVAL = 240
CANDLE_SECONDS = INTERVAL * 60

# SIGNAL SETTINGS
REQUIRED_SCORE = 5

# ATR-BASED RISK MANAGEMENT
SL_ATR_MULTIPLIER = 1.0
TP_ATR_MULTIPLIER = 2.0

# SMART TP / MARKET-SPACE FILTER
TP_SPACE_LOOKBACK = 20
TP_SPACE_BUFFER_PERCENT = 0.10
MIN_B_SETUP_ATR = 1.20

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}


def utc_text(ts):
    return datetime.fromtimestamp(
        ts,
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except Exception as e:
        print(
            f"⚠️ state.json could not be read: {e}"
        )
        return {}


def save_state(state):
    tmp_file = STATE_FILE + ".tmp"

    with open(
        tmp_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            state,
            f,
            indent=2
        )

    os.replace(
        tmp_file,
        STATE_FILE
    )


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

    print(response.text)

    response.raise_for_status()


def get_4h_data(symbol):
    """
    دریافت کندل‌های 4H از Kraken.

    فقط کندل‌هایی که کاملاً بسته شده‌اند
    وارد استراتژی می‌شوند.

    کندل جاری هیچ‌وقت بررسی نمی‌شود.
    """

    print(
        f"\nGetting {symbol} 4H candles..."
    )

    response = requests.get(
        KRAKEN_URL,
        params={
            "pair": COINS[symbol],
            "interval": INTERVAL,
        },
        timeout=20,
    )

    print(
        f"{symbol} Kraken:",
        response.status_code
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            f"{symbol} Kraken API error: "
            f"{payload['error']}"
        )

    result = payload.get(
        "result",
        {}
    )

    pair_key = next(
        (
            key
            for key in result
            if key != "last"
        ),
        None
    )

    if pair_key is None:
        raise RuntimeError(
            f"{symbol}: no candle data returned"
        )

    raw_rows = result.get(
        pair_key,
        []
    )

    if not raw_rows:
        raise RuntimeError(
            f"{symbol}: empty OHLC response"
        )

    candles = []

    for row in raw_rows:

        if len(row) < 7:
            continue

        try:
            candles.append(
                {
                    "time": int(float(row[0])),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[6]),
                }
            )

        except (
            TypeError,
            ValueError
        ):
            continue

    if not candles:
        raise RuntimeError(
            f"{symbol}: no valid candles parsed"
        )

    # مرتب‌سازی بر اساس زمان
    candles.sort(
        key=lambda x: x["time"]
    )

    # حذف timestampهای تکراری
    unique = {}

    for candle in candles:
        unique[candle["time"]] = candle

    candles = [
        unique[t]
        for t in sorted(unique)
    ]

    now_ts = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )

    print(
        f"{symbol}: current UTC: "
        f"{utc_text(now_ts)}"
    )

    print(
        f"{symbol}: Kraken newest row UTC: "
        f"{utc_text(candles[-1]['time'])}"
    )

    # فقط کندل‌هایی که 4 ساعتشان کامل شده
    closed_candles = [
        candle
        for candle in candles
        if (
            candle["time"]
            + CANDLE_SECONDS
            <= now_ts
        )
    ]

    if not closed_candles:
        raise RuntimeError(
            f"{symbol}: no fully closed "
            f"4H candle available"
        )

    latest_closed = (
        closed_candles[-1]
    )

    print(
        f"{symbol}: newest CLOSED "
        f"4H candle UTC: "
        f"{utc_text(latest_closed['time'])}"
    )

    if len(closed_candles) < 210:
        raise RuntimeError(
            f"{symbol}: only "
            f"{len(closed_candles)} "
            f"closed candles available"
        )

    print(
        f"{symbol}: "
        f"{len(closed_candles)} "
        f"closed 4H candles"
    )

    return closed_candles


def ema(values, period):

    if len(values) < period:
        return None

    value = sum(
        values[:period]
    ) / period

    multiplier = (
        2.0 / (period + 1)
    )

    for price in values[period:]:

        value = (
            (price - value)
            * multiplier
            + value
        )

    return value


def sma(values, period):

    if len(values) < period:
        return None

    return (
        sum(values[-period:])
        / period
    )


def rsi(values, period=14):

    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(
        1,
        len(values)
    ):

        change = (
            values[i]
            - values[i - 1]
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
        return 100.0

    rs = (
        avg_gain
        / avg_loss
    )

    return (
        100.0
        - (
            100.0
            / (1.0 + rs)
        )
    )


def atr(
    highs,
    lows,
    closes,
    period=14
):

    if len(closes) <= period:
        return None

    true_ranges = []

    for i in range(
        1,
        len(closes)
    ):

        true_ranges.append(
            max(
                highs[i]
                - lows[i],

                abs(
                    highs[i]
                    - closes[i - 1]
                ),

                abs(
                    lows[i]
                    - closes[i - 1]
                ),
            )
        )

    if len(true_ranges) < period:
        return None

    return (
        sum(
            true_ranges[-period:]
        )
        / period
    )


def get_risk_levels(
    direction,
    entry,
    current_atr
):

    sl_distance = (
        current_atr
        * SL_ATR_MULTIPLIER
    )

    tp_distance = (
        current_atr
        * TP_ATR_MULTIPLIER
    )

    if direction == "LONG":

        tp = (
            entry
            + tp_distance
        )

        sl = (
            entry
            - sl_distance
        )

    else:

        tp = (
            entry
            - tp_distance
        )

        sl = (
            entry
            + sl_distance
        )

    return (
        tp,
        sl,
        tp_distance,
        sl_distance
    )


def get_smart_tp(
    candles,
    direction,
    entry,
    current_atr
):

    """
    A:
        Full 2 ATR TP has enough space.

    B:
        Structure blocks 2 ATR,
        but at least 1.2 ATR
        usable space exists.

    C:
        Less than 1.2 ATR space.
        No signal.
    """

    if len(candles) < (
        TP_SPACE_LOOKBACK + 2
    ):
        return None

    lookback = candles[
        -(TP_SPACE_LOOKBACK + 1):-1
    ]

    (
        full_tp,
        sl,
        tp_distance,
        sl_distance
    ) = get_risk_levels(
        direction,
        entry,
        current_atr
    )

    min_distance = (
        current_atr
        * MIN_B_SETUP_ATR
    )

    buffer = (
        entry
        * (
            TP_SPACE_BUFFER_PERCENT
            / 100.0
        )
    )

    # LONG
    if direction == "LONG":

        resistance = max(
            c["high"]
            for c in lookback
        )

        # A setup
        if resistance >= (
            full_tp + buffer
        ):

            return (
                full_tp,
                sl,
                tp_distance,
                sl_distance,
                "A",
                resistance
            )

        # B setup
        smart_tp = (
            resistance
            - buffer
        )

        smart_distance = (
            smart_tp
            - entry
        )

        if (
            smart_distance
            >= min_distance
        ):

            return (
                smart_tp,
                sl,
                smart_distance,
                sl_distance,
                "B",
                resistance
            )

        return None

    # SHORT

    support = min(
        c["low"]
        for c in lookback
    )

    # A setup
    if support <= (
        full_tp - buffer
    ):

        return (
            full_tp,
            sl,
            tp_distance,
            sl_distance,
            "A",
            support
        )

    # B setup
    smart_tp = (
        support
        + buffer
    )

    smart_distance = (
        entry
        - smart_tp
    )

    if (
        smart_distance
        >= min_distance
    ):

        return (
            smart_tp,
            sl,
            smart_distance,
            sl_distance,
            "B",
            support
        )

    return None


def calculate_signal(
    candles,
    symbol
):

    if len(candles) < 210:
        return None

    opens = [
        c["open"]
        for c in candles
    ]

    highs = [
        c["high"]
        for c in candles
    ]

    lows = [
        c["low"]
        for c in candles
    ]

    closes = [
        c["close"]
        for c in candles
    ]

    volumes = [
        c["volume"]
        for c in candles
    ]

    # آخرین کندل کاملاً بسته‌شده
    latest = candles[-1]

    open_price = latest["open"]
    high = latest["high"]
    low = latest["low"]
    close = latest["close"]
    volume = latest["volume"]

    ema20 = ema(
        closes,
        20
    )

    ema50 = ema(
        closes,
        50
    )

    ema200 = ema(
        closes,
        200
    )

    current_rsi = rsi(
        closes,
        14
    )

    previous_rsi = rsi(
        closes[:-1],
        14
    )

    volume_ma = sma(
        volumes,
        20
    )

    current_atr = atr(
        highs,
        lows,
        closes,
        14
    )

    if any(
        x is None
        for x in [
            ema20,
            ema50,
            ema200,
            current_rsi,
            previous_rsi,
            volume_ma,
            current_atr
        ]
    ):

        print(
            f"{symbol}: "
            "indicator calculation failed"
        )

        return None

    # 1 - TREND

    long_trend = (
        close > ema200
        and ema20 > ema50
    )

    short_trend = (
        close < ema200
        and ema20 < ema50
    )

    # 2 - RSI

    long_rsi = (
        current_rsi > 50
        and current_rsi < 72
        and current_rsi > previous_rsi
    )

    short_rsi = (
        current_rsi < 50
        and current_rsi > 28
        and current_rsi < previous_rsi
    )

    # 3 - VOLUME

    volume_ok = (
        volume >= volume_ma
    )

    # 4 - BREAKOUT

    recent_high = max(
        highs[-7:-1]
    )

    recent_low = min(
        lows[-7:-1]
    )

    bull_break = (
        close > recent_high
    )

    bear_break = (
        close < recent_low
    )

    # 5 - PULLBACK

    long_pullback = (
        low <= ema20
        and close > ema20
    )

    short_pullback = (
        high >= ema20
        and close < ema20
    )

    # 6 - CANDLE QUALITY

    candle_range = (
        high - low
    )

    if candle_range > 0:

        bull_body_ratio = (
            close
            - open_price
        ) / candle_range

        bear_body_ratio = (
            open_price
            - close
        ) / candle_range

    else:

        bull_body_ratio = 0.0
        bear_body_ratio = 0.0

    bull_candle = (
        close > open_price
        and candle_range > 0
        and bull_body_ratio >= 0.40
    )

    bear_candle = (
        close < open_price
        and candle_range > 0
        and bear_body_ratio >= 0.40
    )

    # 7 - VOLATILITY

    volatility_ok = (
        current_atr / close
    ) >= 0.002

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

    atr_percent = (
        current_atr
        / close
    ) * 100.0

    long_levels = get_smart_tp(
        candles,
        "LONG",
        close,
        current_atr
    )

    short_levels = get_smart_tp(
        candles,
        "SHORT",
        close,
        current_atr
    )

    print(
        f"\n===== {symbol} 4H ====="
    )

    print(
        f"Price: {close:.8f}"
    )

    print(
        f"RSI: {current_rsi:.2f}"
    )

    print(
        f"EMA20: {ema20:.8f}"
    )

    print(
        f"EMA50: {ema50:.8f}"
    )

    print(
        f"EMA200: {ema200:.8f}"
    )

    print(
        f"ATR: {current_atr:.8f} "
        f"({atr_percent:.3f}%)"
    )

    print(
        f"Volume OK: {volume_ok}"
    )

    print(
        f"LONG SCORE: "
        f"{long_score}/7"
    )

    print(
        f"SHORT SCORE: "
        f"{short_score}/7"
    )

    if long_levels:

        print(
            f"LONG TP/SL: "
            f"{long_levels[0]:.8f} / "
            f"{long_levels[1]:.8f} | "
            f"Quality: {long_levels[4]} | "
            f"Resistance: "
            f"{long_levels[5]:.8f}"
        )

    else:

        print(
            "LONG TP/SL: REJECTED "
            "(< 1.2 ATR usable room)"
        )

    if short_levels:

        print(
            f"SHORT TP/SL: "
            f"{short_levels[0]:.8f} / "
            f"{short_levels[1]:.8f} | "
            f"Quality: {short_levels[4]} | "
            f"Support: "
            f"{short_levels[5]:.8f}"
        )

    else:

        print(
            "SHORT TP/SL: REJECTED "
            "(< 1.2 ATR usable room)"
        )

    print(
        "===================="
    )

    # LONG

    if (
        long_score >= REQUIRED_SCORE
        and long_levels
    ):

        return {
            "direction": "LONG",
            "score": long_score,
            "price": close,
            "atr": current_atr,
            "tp": long_levels[0],
            "sl": long_levels[1],
            "tp_distance": long_levels[2],
            "sl_distance": long_levels[3],
            "quality": long_levels[4],
            "structure": long_levels[5],
        }

    # SHORT

    if (
        short_score >= REQUIRED_SCORE
        and short_levels
    ):

        return {
            "direction": "SHORT",
            "score": short_score,
            "price": close,
            "atr": current_atr,
            "tp": short_levels[0],
            "sl": short_levels[1],
            "tp_distance": short_levels[2],
            "sl_distance": short_levels[3],
            "quality": short_levels[4],
            "structure": short_levels[5],
        }

    if (
        long_score
        >= REQUIRED_SCORE
    ):

        print(
            f"{symbol}: LONG rejected - "
            "less than 1.2 ATR "
            "usable TP space"
        )

    if (
        short_score
        >= REQUIRED_SCORE
    ):

        print(
            f"{symbol}: SHORT rejected - "
            "less than 1.2 ATR "
            "usable TP space"
        )

    return None


def create_message(
    symbol,
    signal
):

    direction = signal[
        "direction"
    ]

    score = signal[
        "score"
    ]

    entry = signal[
        "price"
    ]

    atr_value = signal[
        "atr"
    ]

    tp = signal[
        "tp"
    ]

    sl = signal[
        "sl"
    ]

    quality = signal[
        "quality"
    ]

    structure = signal[
        "structure"
    ]

    tp_distance = signal[
        "tp_distance"
    ]

    sl_distance = signal[
        "sl_distance"
    ]

    tp_percent = (
        tp_distance
        / entry
    ) * 100.0

    sl_percent = (
        sl_distance
        / entry
    ) * 100.0

    if direction == "LONG":

        direction_text = (
            "📊 🟢 LONG"
        )

        tp_text = (
            f"🎯 TP: "
            f"{tp:.8f} "
            f"(+{tp_percent:.2f}%)"
        )

        sl_text = (
            f"🛑 SL: "
            f"{sl:.8f} "
            f"(-{sl_percent:.2f}%)"
        )

        structure_text = (
            f"📈 Resistance: "
            f"{structure:.8f}"
        )

    else:

        direction_text = (
            "📊 🔴 SHORT"
        )

        tp_text = (
            f"🎯 TP: "
            f"{tp:.8f} "
            f"(-{tp_percent:.2f}%)"
        )

        sl_text = (
            f"🛑 SL: "
            f"{sl:.8f} "
            f"(+{sl_percent:.2f}%)"
        )

        structure_text = (
            f"📉 Support: "
            f"{structure:.8f}"
        )

    return (
        "🚨 SCORE HUNTER 4H 🚨\n\n"
        f"💰 {symbol}USDT\n"
        f"{direction_text}\n"
        f"⭐ Score: {score}/7\n"
        f"🏆 Setup Quality: {quality}\n"
        f"💵 Entry: {entry:.8f}\n"
        f"{tp_text}\n"
        f"{sl_text}\n"
        f"{structure_text}\n\n"
        "⏱ Timeframe: 4H\n"
        "🕯 Closed candle confirmation\n"
        f"📐 ATR: {atr_value:.8f}\n"
        "🧠 Smart TP: "
        "ATR + Market Structure\n"
        "🔴 C setup disabled\n"
        "⚠️ Manage risk."
    )


def main():

    print(
        "🟢 SCORE HUNTER "
        "4H MULTI-COIN SCANNING"
    )

    print(
        "🕯 CLOSED CANDLE ONLY - "
        "NO MID-CANDLE SIGNAL"
    )

    print(
        "🚫 PRE-SIGNAL: DISABLED"
    )

    print(
        "♾️ DAILY SIGNAL LIMIT: DISABLED"
    )

    print(
        "📊 Coins: "
        + " / ".join(
            COINS.keys()
        )
    )

    print(
        "⏱ Timeframe: 4H"
    )

    print(
        f"⭐ Minimum Score: "
        f"{REQUIRED_SCORE}/7"
    )

    print(
        f"🎯 TP: "
        f"{TP_ATR_MULTIPLIER} ATR"
    )

    print(
        f"🛑 SL: "
        f"{SL_ATR_MULTIPLIER} ATR"
    )

    print(
        "⚖️ Risk/Reward: "
        "1:2 base | "
        "B setup = structure-based TP"
    )

    print(
        "🧠 SMART TP: ON | "
        "2 ATR base + "
        "support/resistance"
    )

    print(
        f"🟡 B SETUP: minimum usable "
        f"TP space = "
        f"{MIN_B_SETUP_ATR} ATR | "
        f"buffer="
        f"{TP_SPACE_BUFFER_PERCENT}%"
    )

    print(
        "🔴 C SETUP: "
        "DISABLED (NO SIGNAL)"
    )

    state = load_state()

    for symbol in COINS:

        print(
            f"\n\n========== "
            f"{symbol} =========="
        )

        try:

            candles = get_4h_data(
                symbol
            )

            latest_candle_time = (
                candles[-1]["time"]
            )

            coin_state = state.get(
                symbol,
                {}
            )

            previous_candle_time = (
                coin_state.get(
                    "last_checked_candle"
                )
            )

            print(
                f"{symbol} latest CLOSED "
                f"4H candle: "
                f"{latest_candle_time} "
                f"("
                f"{utc_text(latest_candle_time)}"
                f")"
            )

            # اگر همین کندل قبلاً بررسی شده
            # دوباره سیگنال نده.
            if (
                previous_candle_time
                == latest_candle_time
            ):

                print(
                    f"{symbol}: "
                    "No new 4H candle."
                )

                continue

            # ثبت کندل قبل از محاسبه
            # برای جلوگیری از اجرای تکراری
            coin_state[
                "last_checked_candle"
            ] = latest_candle_time

            coin_state[
                "last_checked_candle_utc"
            ] = utc_text(
                latest_candle_time
            )

            state[symbol] = coin_state

            save_state(state)

            signal = calculate_signal(
                candles,
                symbol
            )

            if signal is None:

                print(
                    f"{symbol}: "
                    "No valid signal."
                )

                continue

            message = create_message(
                symbol,
                signal
            )

            send_telegram(
                message
            )

            coin_state[
                "last_signal"
            ] = signal

            coin_state[
                "signal_candle"
            ] = latest_candle_time

            coin_state[
                "signal_candle_utc"
            ] = utc_text(
                latest_candle_time
            )

            coin_state[
                "signal_time"
            ] = int(
                datetime.now(
                    timezone.utc
                ).timestamp()
            )

            state[symbol] = (
                coin_state
            )

            save_state(state)

            print(
                f"🚨 {symbol}: "
                "SIGNAL SENT"
            )

        except Exception as e:

            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: {e}"
            )

            continue

    save_state(state)

    print(
        "\n✅ ALL COINS SCANNED"
    )


if __name__ == "__main__":
    main()
