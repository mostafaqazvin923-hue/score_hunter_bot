import os
import json
import requests
from datetime import datetime, timezone

# ============================================================
# SCORE HUNTER PRO v2
# TRUE 4H STRUCTURE + TRUE 1H ENTRY CONFIRMATION
#
# Signal rules:
# - 4H defines the directional bias.
# - 1H must provide the actual entry confirmation.
# - Only CLOSED candles are used.
# - Bot scans on every NEW CLOSED 1H candle.
# - No daily signal limit.
# - TP = 1%, SL = 0.5%.
# - TP-space filter rejects trades when the nearest
#   opposing level is too close to the 1% target.
# - No exchange orders are placed; Telegram only.
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
STATE_FILE = "state.json"

# Timeframes
TF_4H = 240
TF_1H = 60

# Coins
COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}

# Indicators
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_LOOKBACK = 20

# 4H structure
STRUCTURE_LOOKBACK = 20
BOS_LOOKBACK = 10
SWEEP_LOOKBACK = 10

# 1H confirmation
RETEST_LOOKBACK = 8
RETEST_TOLERANCE_PERCENT = 0.20
MIN_1H_SCORE = 4

# 4H quality
MIN_4H_SCORE = 4
VOLUME_MULTIPLIER = 1.20
RANGE_EMA_GAP_PERCENT = 0.15

# Risk / target
TP_PERCENT = 1.0
SL_PERCENT = 0.50
TP_SPACE_LOOKBACK = 30
TP_SPACE_BUFFER_PERCENT = 0.10

# State
# A signal can occur once per CLOSED 1H candle per coin.
# There is NO daily limit.
STATE_FILE = "state.json"


# ============================================================
# STATE
# ============================================================

def empty_state():
    return {
        "last_processed_1h": {},
        "last_signals": {},
    }


def load_state():
    if not os.path.exists(STATE_FILE):
        return empty_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        state.setdefault("last_processed_1h", {})
        state.setdefault("last_signals", {})
        return state

    except Exception as exc:
        print("State load error:", exc)
        return empty_state()


def save_state(state):
    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    os.replace(tmp, STATE_FILE)


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

    response.raise_for_status()


# ============================================================
# KRAKEN
# ============================================================

def get_ohlc(symbol, interval):
    label = "4H" if interval == TF_4H else "1H"
    print(f"Getting {symbol} {label} candles...")

    response = requests.get(
        KRAKEN_URL,
        params={
            "pair": COINS[symbol],
            "interval": interval,
        },
        timeout=20,
    )

    print(f"{symbol} Kraken {label}:", response.status_code)
    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            f"{symbol} Kraken {label} API error: "
            f"{payload['error']}"
        )

    result = payload.get("result", {})

    pair_key = next(
        (key for key in result if key != "last"),
        None,
    )

    if pair_key is None:
        raise RuntimeError(
            f"{symbol}: no {label} candle data"
        )

    candles = []

    for row in result[pair_key]:
        if len(row) < 7:
            continue

        candles.append({
            "time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6]),
        })

    candles.sort(key=lambda x: x["time"])

    # Kraken can return the currently forming candle.
    # Remove it. Everything below uses CLOSED candles only.
    if len(candles) > 1:
        candles = candles[:-1]

    if len(candles) < 210:
        raise RuntimeError(
            f"{symbol}: only {len(candles)} closed {label} candles"
        )

    print(
        f"{symbol}: {len(candles)} closed {label} candles"
    )

    return candles


# ============================================================
# INDICATORS
# ============================================================

def ema_series(values, period):
    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1)

    value = sum(values[:period]) / period

    result = [None] * (period - 1)
    result.append(value)

    for price in values[period:]:
        value = (
            (price - value) * multiplier
            + value
        )
        result.append(value)

    return result


def rsi_series(values, period=14):
    if len(values) <= period:
        return []

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    result = [None] * period

    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(
            100.0 - (100.0 / (1.0 + rs))
        )

    for i in range(period, len(gains)):
        avg_gain = (
            avg_gain * (period - 1)
            + gains[i]
        ) / period

        avg_loss = (
            avg_loss * (period - 1)
            + losses[i]
        ) / period

        if avg_loss == 0:
            value = 100.0
        else:
            rs = avg_gain / avg_loss
            value = (
                100.0
                - 100.0 / (1.0 + rs)
            )

        result.append(value)

    return result


def atr_series(candles, period=14):
    if len(candles) <= period:
        return []

    true_ranges = []

    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        previous_close = candles[i - 1]["close"]

        true_ranges.append(
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        )

    first = (
        sum(true_ranges[:period])
        / period
    )

    result = [None] * period
    result.append(first)

    previous = first

    for tr in true_ranges[period:]:
        previous = (
            previous * (period - 1)
            + tr
        ) / period

        result.append(previous)

    return result


def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


# ============================================================
# 4H STRUCTURE
# ============================================================

def get_4h_context(candles):
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    ema20 = ema_series(closes, EMA_FAST)[-1]
    ema50 = ema_series(closes, EMA_MID)[-1]
    ema200 = ema_series(closes, EMA_SLOW)[-1]
    rsi_values = rsi_series(closes, RSI_PERIOD)
    atr_values = atr_series(candles, ATR_PERIOD)

    rsi_now = rsi_values[-1]
    rsi_prev = rsi_values[-2]
    atr_now = atr_values[-1]

    close = closes[-1]

    volume_ma = sum(
        volumes[-(VOLUME_LOOKBACK + 1):-1]
    ) / VOLUME_LOOKBACK

    volume_ok = (
        volumes[-1]
        >= volume_ma * VOLUME_MULTIPLIER
    )

    # Directional structure:
    # current 20-bar range versus previous 20-bar range.
    recent = candles[-STRUCTURE_LOOKBACK:]
    previous = candles[
        -(STRUCTURE_LOOKBACK * 2):-STRUCTURE_LOOKBACK
    ]

    recent_high = max(c["high"] for c in recent)
    recent_low = min(c["low"] for c in recent)

    previous_high = max(
        c["high"] for c in previous
    )
    previous_low = min(
        c["low"] for c in previous
    )

    long_structure = (
        close > ema200
        and ema20 > ema50
        and recent_high >= previous_high
        and recent_low >= previous_low
    )

    short_structure = (
        close < ema200
        and ema20 < ema50
        and recent_high <= previous_high
        and recent_low <= previous_low
    )

    if long_structure and not short_structure:
        structure = "LONG"
    elif short_structure and not long_structure:
        structure = "SHORT"
    else:
        structure = "NONE"

    # 4H BOS must happen on the latest CLOSED 4H candle.
    bos_long = (
        close
        > max(
            c["high"]
            for c in candles[-(BOS_LOOKBACK + 1):-1]
        )
        and close > candles[-1]["open"]
    )

    bos_short = (
        close
        < min(
            c["low"]
            for c in candles[-(BOS_LOOKBACK + 1):-1]
        )
        and close < candles[-1]["open"]
    )

    # 4H liquidity sweep.
    sweep_window = candles[
        -(SWEEP_LOOKBACK + 1):-1
    ]

    sweep_long_level = min(
        c["low"] for c in sweep_window
    )

    sweep_short_level = max(
        c["high"] for c in sweep_window
    )

    sweep_long = (
        candles[-1]["low"] < sweep_long_level
        and close > sweep_long_level
    )

    sweep_short = (
        candles[-1]["high"] > sweep_short_level
        and close < sweep_short_level
    )

    volatility_ok = (
        atr_now / close >= 0.002
    )

    range_market = (
        abs(ema20 - ema50) / close * 100
        < RANGE_EMA_GAP_PERCENT
        and
        abs(ema50 - ema200) / close * 100
        < RANGE_EMA_GAP_PERCENT
    )

    long_rsi = (
        rsi_now > 50
        and rsi_now < 72
        and rsi_now > rsi_prev
    )

    short_rsi = (
        rsi_now < 50
        and rsi_now > 28
        and rsi_now < rsi_prev
    )

    long_score = (
        int(long_structure)
        + int(long_rsi)
        + int(volume_ok)
        + int(bos_long)
        + int(sweep_long)
        + int(volatility_ok)
    )

    short_score = (
        int(short_structure)
        + int(short_rsi)
        + int(volume_ok)
        + int(bos_short)
        + int(sweep_short)
        + int(volatility_ok)
    )

    return {
        "structure": structure,
        "rsi": rsi_now,
        "atr": atr_now,
        "volume_ok": volume_ok,
        "range_market": range_market,
        "long_score": long_score,
        "short_score": short_score,
        "long_bos": bos_long,
        "short_bos": bos_short,
        "long_sweep": sweep_long,
        "short_sweep": sweep_short,
        "long_rsi": long_rsi,
        "short_rsi": short_rsi,
        "volatility_ok": volatility_ok,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
    }


# ============================================================
# 1H ENTRY CONFIRMATION
# ============================================================

def one_hour_confirmation(candles, direction):
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    ema20 = ema_series(closes, EMA_FAST)[-1]
    ema50 = ema_series(closes, EMA_MID)[-1]
    rsi_values = rsi_series(closes, RSI_PERIOD)
    atr_values = atr_series(candles, ATR_PERIOD)

    rsi_now = rsi_values[-1]
    rsi_prev = rsi_values[-2]
    atr_now = atr_values[-1]

    current = candles[-1]
    previous = candles[-2]

    candle_range = current["high"] - current["low"]

    if candle_range <= 0:
        candle_body_ratio = 0.0
    else:
        candle_body_ratio = (
            abs(current["close"] - current["open"])
            / candle_range
        )

    bullish_candle = (
        current["close"] > current["open"]
        and candle_body_ratio >= 0.50
    )

    bearish_candle = (
        current["close"] < current["open"]
        and candle_body_ratio >= 0.50
    )

    volume_ma = sum(
        volumes[-21:-1]
    ) / 20

    volume_ok = (
        current["volume"]
        >= volume_ma * 1.10
    )

    # 1H BOS
    prior_window = candles[-(BOS_LOOKBACK + 1):-1]

    bull_bos = (
        current["close"]
        > max(c["high"] for c in prior_window)
        and current["close"] > current["open"]
    )

    bear_bos = (
        current["close"]
        < min(c["low"] for c in prior_window)
        and current["close"] < current["open"]
    )

    # 1H liquidity sweep + reclaim.
    sweep_window = candles[
        -(SWEEP_LOOKBACK + 1):-1
    ]

    sweep_low = min(
        c["low"] for c in sweep_window
    )

    sweep_high = max(
        c["high"] for c in sweep_window
    )

    bull_sweep = (
        current["low"] < sweep_low
        and current["close"] > sweep_low
    )

    bear_sweep = (
        current["high"] > sweep_high
        and current["close"] < sweep_high
    )

    # Retest:
    # For LONG, the current candle must retest the latest
    # broken resistance and close back above it.
    # For SHORT, the inverse.
    retest_window = candles[-(RETEST_LOOKBACK + 1):-1]

    if direction == "LONG":
        level = max(
            c["high"] for c in retest_window
        )
        tolerance = (
            level
            * RETEST_TOLERANCE_PERCENT
            / 100.0
        )

        retest = (
            current["low"] <= level + tolerance
            and current["close"] > level
        )

        trend = (
            current["close"] > ema20
            and ema20 > ema50
        )

        rsi_ok = (
            rsi_now > 50
            and rsi_now > rsi_prev
            and rsi_now < 75
        )

        candle_ok = bullish_candle

        event = (
            bull_bos
            or bull_sweep
            or retest
        )

    else:
        level = min(
            c["low"] for c in retest_window
        )
        tolerance = (
            level
            * RETEST_TOLERANCE_PERCENT
            / 100.0
        )

        retest = (
            current["high"] >= level - tolerance
            and current["close"] < level
        )

        trend = (
            current["close"] < ema20
            and ema20 < ema50
        )

        rsi_ok = (
            rsi_now < 50
            and rsi_now < rsi_prev
            and rsi_now > 25
        )

        candle_ok = bearish_candle

        event = (
            bear_bos
            or bear_sweep
            or retest
        )

    # Score is 6 components.
    score = (
        int(trend)
        + int(rsi_ok)
        + int(volume_ok)
        + int(event)
        + int(retest)
        + int(candle_ok)
    )

    return {
        "score": score,
        "trend": trend,
        "rsi_ok": rsi_ok,
        "volume_ok": volume_ok,
        "event": event,
        "retest": retest,
        "candle": candle_ok,
        "bull_bos": bull_bos,
        "bear_bos": bear_bos,
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "atr": atr_now,
        "previous_close": previous["close"],
    }


# ============================================================
# TP SPACE
# ============================================================

def nearest_opposing_level(candles, direction, entry):
    """
    Finds the NEAREST historical obstacle in the TP path.

    LONG:
      nearest high above entry.

    SHORT:
      nearest low below entry.

    This is intentionally different from using max(high) or
    min(low), because one distant extreme should not hide a
    nearby resistance/support level.
    """
    window = candles[
        -(TP_SPACE_LOOKBACK + 1):-1
    ]

    if direction == "LONG":
        levels = [
            c["high"]
            for c in window
            if c["high"] > entry
        ]

        return min(levels) if levels else None

    levels = [
        c["low"]
        for c in window
        if c["low"] < entry
    ]

    return max(levels) if levels else None


def has_tp_space(candles, direction, entry):
    tp = (
        entry * (1 + TP_PERCENT / 100.0)
        if direction == "LONG"
        else entry * (1 - TP_PERCENT / 100.0)
    )

    buffer = TP_SPACE_BUFFER_PERCENT / 100.0

    obstacle = nearest_opposing_level(
        candles,
        direction,
        entry,
    )

    if obstacle is None:
        return True, None

    if direction == "LONG":
        allowed = obstacle >= tp * (1 + buffer)
    else:
        allowed = obstacle <= tp * (1 - buffer)

    return allowed, obstacle


# ============================================================
# FINAL SIGNAL
# ============================================================

def calculate_signal(candles_4h, candles_1h, symbol):
    context = get_4h_context(candles_4h)

    print(
        f"\n===== {symbol} TRUE 4H/1H ====="
    )

    print(
        f"4H Structure: {context['structure']}"
    )

    print(
        f"4H RSI: {context['rsi']:.2f}"
    )

    print(
        f"4H Volume >= "
        f"{VOLUME_MULTIPLIER}x MA: "
        f"{context['volume_ok']}"
    )

    print(
        f"4H Range market: "
        f"{context['range_market']}"
    )

    print(
        f"4H LONG score: "
        f"{context['long_score']}/6 "
        f"| SHORT: "
        f"{context['short_score']}/6"
    )

    if context["range_market"]:
        print(
            f"{symbol}: rejected - 4H market range"
        )
        return None

    candidates = []

    for direction in ("LONG", "SHORT"):
        if direction == "LONG":
            four_score = context["long_score"]
            four_bos = context["long_bos"]
            four_sweep = context["long_sweep"]
        else:
            four_score = context["short_score"]
            four_bos = context["short_bos"]
            four_sweep = context["short_sweep"]

        # Direction MUST come from 4H.
        if context["structure"] != direction:
            print(
                f"{symbol} {direction}: "
                f"rejected - 4H structure mismatch"
            )
            continue

        if four_score < MIN_4H_SCORE:
            print(
                f"{symbol} {direction}: "
                f"rejected - 4H quality "
                f"{four_score}/{MIN_4H_SCORE}"
            )
            continue

        one = one_hour_confirmation(
            candles_1h,
            direction,
        )

        print(
            f"1H {direction}: "
            f"score={one['score']}/6 "
            f"trend={int(one['trend'])} "
            f"rsi={int(one['rsi_ok'])} "
            f"volume={int(one['volume_ok'])} "
            f"event={int(one['event'])} "
            f"retest={int(one['retest'])} "
            f"candle={int(one['candle'])}"
        )

        # 1H is a REAL entry gate, not merely a score bonus.
        if one["score"] < MIN_1H_SCORE:
            print(
                f"{symbol} {direction}: "
                f"rejected - weak 1H confirmation"
            )
            continue

        # Require at least one concrete 1H event.
        if not one["event"]:
            print(
                f"{symbol} {direction}: "
                f"rejected - no 1H BOS/sweep/retest event"
            )
            continue

        entry = candles_1h[-1]["close"]

        tp_ok, obstacle = has_tp_space(
            candles_1h,
            direction,
            entry,
        )

        if not tp_ok:
            print(
                f"{symbol} {direction}: "
                f"rejected - nearest opposing level "
                f"{obstacle:.8f} blocks 1% TP"
            )
            continue

        total_score = (
            four_score
            + one["score"]
        )

        candidates.append({
            "direction": direction,
            "score_4h": four_score,
            "score_1h": one["score"],
            "score": total_score,
            "price": entry,
            "tp_obstacle": obstacle,
            "four_bos": four_bos,
            "four_sweep": four_sweep,
        })

    if not candidates:
        print(
            f"{symbol}: No valid 4H/1H signal."
        )
        return None

    return max(
        candidates,
        key=lambda x: (
            x["score"],
            x["score_1h"],
        ),
    )


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def create_message(symbol, signal):
    direction = signal["direction"]
    entry = signal["price"]

    tp = (
        entry * (1 + TP_PERCENT / 100.0)
        if direction == "LONG"
        else entry * (1 - TP_PERCENT / 100.0)
    )

    sl = (
        entry * (1 - SL_PERCENT / 100.0)
        if direction == "LONG"
        else entry * (1 + SL_PERCENT / 100.0)
    )

    icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"

    return (
        "🚨 SCORE HUNTER PRO v2 🚨\n\n"
        f"💰 {symbol}USDT\n"
        f"📊 {icon}\n"
        f"⭐ 4H Score: {signal['score_4h']}/6\n"
        f"⭐ 1H Score: {signal['score_1h']}/6\n"
        f"⭐ Combined: {signal['score']}/12\n"
        f"💵 Entry: {entry:.8f}\n"
        f"🎯 TP: {tp:.8f} "
        f"({'+' if direction == 'LONG' else '-'}{TP_PERCENT:.1f}%)\n"
        f"🛑 SL: {sl:.8f} "
        f"({'-' if direction == 'LONG' else '+'}{SL_PERCENT:.1f}%)\n\n"
        "📊 4H structure + 1H entry confirmation\n"
        "🕯 Closed candle only\n"
        "📐 TP-space filter passed\n"
        "♾️ Daily signal limit disabled\n"
        "⚠️ Signal/research bot — manage risk."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "🟢 SCORE HUNTER PRO v2 "
        "TRUE 4H/1H"
    )

    print(
        "🔎 4H STRUCTURE + 1H BOS/SWEEP/RETEST"
    )

    print(
        f"⭐ Minimum 4H quality: "
        f"{MIN_4H_SCORE}/6"
    )

    print(
        f"⭐ Minimum 1H confirmation: "
        f"{MIN_1H_SCORE}/6"
    )

    print(
        "🕯 CLOSED CANDLE ONLY - "
        "NO MID-CANDLE SIGNAL"
    )

    print(
        "♾️ DAILY SIGNAL LIMIT: DISABLED"
    )

    print(
        "📊 Coins: "
        + " / ".join(COINS.keys())
    )

    print("⏱ Main structure: 4H")
    print("⏱ Entry confirmation: 1H")
    print(f"🎯 TP: {TP_PERCENT}%")
    print(f"🛑 SL: {SL_PERCENT}%")
    print(
        "📐 TP SPACE FILTER: ON | "
        f"lookback={TP_SPACE_LOOKBACK} | "
        f"buffer={TP_SPACE_BUFFER_PERCENT}%"
    )

    state = load_state()

    for symbol in COINS:
        print(
            f"\n\n========== {symbol} =========="
        )

        try:
            candles_4h = get_ohlc(
                symbol,
                TF_4H,
            )

            candles_1h = get_ohlc(
                symbol,
                TF_1H,
            )

            latest_1h = candles_1h[-1]["time"]

            previous_1h = state[
                "last_processed_1h"
            ].get(symbol)

            print(
                f"{symbol} latest CLOSED 1H: "
                f"{latest_1h}"
            )

            # Process each closed 1H candle only once.
            if previous_1h == latest_1h:
                print(
                    f"{symbol}: "
                    "No new 1H candle."
                )
                continue

            # Mark processed before calculations so a rerun
            # cannot accidentally duplicate the same signal.
            state[
                "last_processed_1h"
            ][symbol] = latest_1h

            signal = calculate_signal(
                candles_4h,
                candles_1h,
                symbol,
            )

            if signal is None:
                save_state(state)
                continue

            signal_id = (
                f"{latest_1h}_"
                f"{signal['direction']}"
            )

            if state["last_signals"].get(symbol) == signal_id:
                print(
                    f"{symbol}: duplicate signal blocked"
                )
                save_state(state)
                continue

            message = create_message(
                symbol,
                signal,
            )

            send_telegram(message)

            state["last_signals"][symbol] = signal_id

            save_state(state)

            print(
                f"🚨 {symbol}: "
                f"{signal['direction']} SIGNAL SENT"
            )

        except Exception as exc:
            print(
                f"❌ {symbol} ERROR: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

    save_state(state)

    print(
        "\n✅ SCORE HUNTER PRO v2 SCAN COMPLETED"
    )


if __name__ == "__main__":
    main()
