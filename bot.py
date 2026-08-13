import os
import json
import time
import requests
from statistics import mean

# ============================================================
# SCORE HUNTER PRO FINAL
# 4H MARKET STRUCTURE + PRICE ACTION + LIQUIDITY
# 1H ENTRY CONFIRMATION
#
# IMPORTANT:
# This is a SIGNAL/RESEARCH BOT, not an automatic trading bot.
# It does not place exchange orders.
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "2090120004")

KLINE_URL = "https://api.lbkex.com/v2/kline.do"
TICKER_URL = "https://api.lbkex.com/v2/kline.do"

COINS = {
    "BTC": "btc_usdt",
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
}

MAIN_TF = "hour4"
ENTRY_TF = "hour1"

CANDLE_LIMIT = 500

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_LOOKBACK = 20

SWING_LEFT = 3
SWING_RIGHT = 3
STRUCTURE_LOOKBACK = 120
BOS_LOOKBACK = 30
LIQUIDITY_LOOKBACK = 40

# Signal quality
MIN_4H_SCORE = 4
MIN_1H_SCORE = 1

# Entry/risk
TP_PERCENT = 0.0100
ENTRY_WINDOW = 0.0100  # absolute safety ceiling: 1.00%
ATR_MULTIPLIER = 1.20
MIN_SL_PERCENT = 0.0040
MAX_SL_PERCENT = 0.0250
MIN_RR = 1.20

# Signal age
EXPIRATION_HOURS = 12  # informational/trade-management only; never blocks fresh signal creation

# Structure buffers
ATR_LEVEL_BUFFER = 0.15

STATE_FILE = "state.json"


# ============================================================
# TELEGRAM
# ============================================================

def latest_closed(candles):
    """Return the last fully closed candle from a raw LBank series."""
    closed = closed_candles(candles)
    if not closed:
        raise RuntimeError("No closed candle available")
    return closed[-1]



def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message},
            timeout=20,
        )
        print("Telegram:", response.status_code)
        if response.status_code != 200:
            print(response.text)
        return response.ok
    except Exception as exc:
        print("Telegram error:", exc)
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
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        state.setdefault("last_signals", {})
        state.setdefault("active_trades", {})
        state.setdefault("completed_trades", [])
        return state

    except Exception as exc:
        print("State load error:", exc)
        return empty_state()


def save_state(state):
    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)

    os.replace(tmp, STATE_FILE)


# ============================================================
# LBANK DATA
# ============================================================

def get_candles(symbol, timeframe, limit=CANDLE_LIMIT):
    print(f"Getting {symbol} {timeframe} candles...")

    hours = 4 if timeframe == "hour4" else 1
    now = int(time.time())
    start_time = now - limit * hours * 3600

    response = requests.get(
        KLINE_URL,
        params={
            "symbol": symbol,
            "size": limit,
            "type": timeframe,
            "time": start_time,
        },
        timeout=20,
    )

    print("LBank:", response.status_code)
    response.raise_for_status()

    result = response.json()

    if str(result.get("result")).lower() != "true":
        raise RuntimeError(f"LBank kline error: {result}")

    raw = result.get("data", [])
    if not raw:
        raise RuntimeError(f"No candle data for {symbol}")

    candles = []

    for row in raw:
        if len(row) < 6:
            continue

        candles.append(
            {
                "time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )

    candles.sort(key=lambda x: x["time"])

    print(f"{symbol}: received {len(candles)} candles")
    return candles


def get_current_price(symbol):
    response = requests.get(
        TICKER_URL,
        params={"symbol": symbol},
        timeout=20,
    )
    response.raise_for_status()

    result = response.json()

    if str(result.get("result")).lower() != "true":
        raise RuntimeError(f"LBank ticker error: {result}")

    data = result.get("data", [])
    if not data:
        raise RuntimeError(f"No ticker data for {symbol}")

    ticker = data[0]
    ticker_data = ticker.get("ticker", ticker)

    price = ticker_data.get("latest")
    if price is None:
        price = ticker_data.get("last")

    if price is None:
        raise RuntimeError(f"Price not found: {result}")

    return float(price)


def closed_candles(candles):
    """
    Return only fully closed candles.
    The last LBank candle is treated as forming and is NEVER
    used by the signal engine.
    """
    if len(candles) < 5:
        raise RuntimeError("Not enough candles")

    closed = candles[:-1]
    if not closed:
        raise RuntimeError("No closed candles available")

    return closed


# ============================================================
# INDICATORS
# ============================================================

def adaptive_entry_window(entry_price, atr_value, direction):
    """
    Adaptive entry distance. The current market is allowed to move away from
    the original setup, but the allowed distance is capped by both ATR and
    the global safety ceiling.
    """
    if entry_price <= 0 or atr_value <= 0:
        return ENTRY_WINDOW

    atr_pct = atr_value / entry_price
    # 0.50 ATR, capped at the configured safety ceiling.
    return min(ENTRY_WINDOW, max(0.0030, atr_pct * 0.50))



def ema(values, period):
    if len(values) < period:
        return [None] * len(values)

    result = [None] * len(values)
    current = sum(values[:period]) / period
    result[period - 1] = current

    alpha = 2.0 / (period + 1.0)

    for i in range(period, len(values)):
        current = alpha * values[i] + (1.0 - alpha) * current
        result[i] = current

    return result


def rsi(values, period=14):
    result = [None] * len(values)

    if len(values) <= period:
        return result

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    idx = period
    result[idx] = 100.0 if avg_loss == 0 else (
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    )

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        if avg_loss == 0:
            value = 100.0
        else:
            rs = avg_gain / avg_loss
            value = 100.0 - 100.0 / (1.0 + rs)

        idx = i + 1
        result[idx] = value

    return result


def atr(candles, period=14):
    result = [None] * len(candles)

    if len(candles) <= period:
        return result

    trs = []

    for i in range(1, len(candles)):
        c = candles[i]
        prev = candles[i - 1]["close"]

        tr = max(
            c["high"] - c["low"],
            abs(c["high"] - prev),
            abs(c["low"] - prev),
        )
        trs.append(tr)

    value = sum(trs[:period]) / period
    result[period] = value

    for i in range(period, len(trs)):
        value = ((value * (period - 1)) + trs[i]) / period
        result[i + 1] = value

    return result


# ============================================================
# CANDLE / PRICE ACTION
# ============================================================

def candle_features(candle):
    o = candle["open"]
    h = candle["high"]
    l = candle["low"]
    c = candle["close"]

    rng = h - l

    if rng <= 0:
        return {
            "bull": False,
            "bear": False,
            "body_ratio": 0.0,
            "upper_wick": 0.0,
            "lower_wick": 0.0,
        }

    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l

    return {
        "bull": c > o and body / rng >= 0.45,
        "bear": c < o and body / rng >= 0.45,
        "body_ratio": body / rng,
        "upper_wick": upper / rng,
        "lower_wick": lower / rng,
    }


def bullish_rejection(candle):
    f = candle_features(candle)
    return f["bull"] or (
        f["lower_wick"] >= 0.45 and f["lower_wick"] > f["upper_wick"] * 1.3
    )


def bearish_rejection(candle):
    f = candle_features(candle)
    return f["bear"] or (
        f["upper_wick"] >= 0.45 and f["upper_wick"] > f["lower_wick"] * 1.3
    )


def bullish_engulfing(previous, current):
    return (
        previous["close"] < previous["open"]
        and current["close"] > current["open"]
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
    )


def bearish_engulfing(previous, current):
    return (
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    )


# ============================================================
# SWINGS / STRUCTURE
# ============================================================

def swing_highs(candles):
    points = []

    start = SWING_LEFT
    end = len(candles) - SWING_RIGHT

    for i in range(start, end):
        high = candles[i]["high"]

        left = [candles[j]["high"] for j in range(i - SWING_LEFT, i)]
        right = [
            candles[j]["high"]
            for j in range(i + 1, i + SWING_RIGHT + 1)
        ]

        if high >= max(left) and high >= max(right):
            points.append((i, high))

    return points


def swing_lows(candles):
    points = []

    start = SWING_LEFT
    end = len(candles) - SWING_RIGHT

    for i in range(start, end):
        low = candles[i]["low"]

        left = [candles[j]["low"] for j in range(i - SWING_LEFT, i)]
        right = [
            candles[j]["low"]
            for j in range(i + 1, i + SWING_RIGHT + 1)
        ]

        if low <= min(left) and low <= min(right):
            points.append((i, low))

    return points


def structure_state(candles):
    highs = swing_highs(candles)
    lows = swing_lows(candles)

    if len(highs) < 3 or len(lows) < 3:
        return {
            "state": "UNKNOWN",
            "last_high": None,
            "last_low": None,
            "prev_high": None,
            "prev_low": None,
            "highs": highs,
            "lows": lows,
        }

    recent_highs = highs[-3:]
    recent_lows = lows[-3:]

    h1 = recent_highs[-1][1]
    h0 = recent_highs[-2][1]

    l1 = recent_lows[-1][1]
    l0 = recent_lows[-2][1]

    if h1 > h0 and l1 > l0:
        state = "BULLISH"
    elif h1 < h0 and l1 < l0:
        state = "BEARISH"
    else:
        state = "RANGE"

    return {
        "state": state,
        "last_high": h1,
        "last_low": l1,
        "prev_high": h0,
        "prev_low": l0,
        "highs": highs,
        "lows": lows,
    }


def detect_bos(candles, structure):
    if len(candles) < 10:
        return {"bull": False, "bear": False}

    close = candles[-1]["close"]

    last_high = structure["last_high"]
    last_low = structure["last_low"]

    return {
        "bull": last_high is not None and close > last_high,
        "bear": last_low is not None and close < last_low,
    }


def detect_choch(candles, structure):
    if len(candles) < 10:
        return {"bull": False, "bear": False}

    close = candles[-1]["close"]

    return {
        "bull": (
            structure["state"] == "BEARISH"
            and structure["last_high"] is not None
            and close > structure["last_high"]
        ),
        "bear": (
            structure["state"] == "BULLISH"
            and structure["last_low"] is not None
            and close < structure["last_low"]
        ),
    }


# ============================================================
# LIQUIDITY
# ============================================================

def detect_liquidity_sweep(candles, structure):
    if len(candles) < 5:
        return {"bull": False, "bear": False}

    c = candles[-1]

    last_high = structure["last_high"]
    last_low = structure["last_low"]

    bull = False
    bear = False

    if last_low is not None:
        bull = (
            c["low"] < last_low
            and c["close"] > last_low
        )

    if last_high is not None:
        bear = (
            c["high"] > last_high
            and c["close"] < last_high
        )

    return {"bull": bull, "bear": bear}


# ============================================================
# PULLBACK / RETEST
# ============================================================

def pullback_confirmation(candles, direction, ema20_value):
    if len(candles) < 3:
        return False

    c = candles[-1]
    previous = candles[-2]

    if direction == "LONG":
        return (
            c["close"] > ema20_value
            and c["low"] <= ema20_value * 1.003
            and c["close"] >= previous["close"]
        )

    return (
        c["close"] < ema20_value
        and c["high"] >= ema20_value * 0.997
        and c["close"] <= previous["close"]
    )


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def resistance_levels(candles, entry):
    highs = swing_highs(candles)
    levels = [
        value
        for _, value in highs
        if value > entry
    ]
    return sorted(set(levels))


def support_levels(candles, entry):
    lows = swing_lows(candles)
    levels = [
        value
        for _, value in lows
        if value < entry
    ]
    return sorted(set(levels), reverse=True)


# ============================================================
# 4H ANALYSIS
# ============================================================

def analyze_4h(candles):

    if len(candles) < 250:
        return None, "NOT_ENOUGH_DATA"

    closes = [c["close"] for c in candles]

    e20 = ema(closes, EMA_FAST)
    e50 = ema(closes, EMA_MID)
    e200 = ema(closes, EMA_SLOW)
    rsi_values = rsi(closes, RSI_PERIOD)
    atr_values = atr(candles, ATR_PERIOD)

    i = len(candles) - 1
    current = candles[i]
    previous = candles[i - 1]

    if any(
        x is None
        for x in [e20[i], e50[i], e200[i], rsi_values[i], atr_values[i]]
    ):
        return None, "INDICATOR_ERROR"

    structure = structure_state(candles)

    if structure["state"] == "UNKNOWN":
        return None, "STRUCTURE_UNKNOWN"

    bos = detect_bos(candles, structure)
    choch = detect_choch(candles, structure)
    sweep = detect_liquidity_sweep(candles, structure)

    current_rsi = rsi_values[i]
    previous_rsi = rsi_values[i - 1]
    current_atr = atr_values[i]

    volume_average = mean(
        c["volume"]
        for c in candles[-VOLUME_LOOKBACK:]
    )

    volume_ok = current["volume"] >= volume_average

    features = candle_features(current)

    bull_engulf = bullish_engulfing(previous, current)
    bear_engulf = bearish_engulfing(previous, current)

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # Trend / EMA
    if current["close"] > e200[i]:
        long_score += 1
        long_reasons.append("price>EMA200")

    if current["close"] < e200[i]:
        short_score += 1
        short_reasons.append("price<EMA200")

    if e20[i] > e50[i]:
        long_score += 1
        long_reasons.append("EMA20>EMA50")

    if e20[i] < e50[i]:
        short_score += 1
        short_reasons.append("EMA20<EMA50")

    # RSI momentum
    if 50 < current_rsi < 72 and current_rsi > previous_rsi:
        long_score += 1
        long_reasons.append("RSI bullish")

    if 28 < current_rsi < 50 and current_rsi < previous_rsi:
        short_score += 1
        short_reasons.append("RSI bearish")

    # Volume
    if volume_ok:
        long_score += 1
        short_score += 1
        long_reasons.append("volume")
        short_reasons.append("volume")

    # Structure / BOS / CHoCH / sweep
    if bos["bull"]:
        long_score += 2
        long_reasons.append("BOS bullish")

    if bos["bear"]:
        short_score += 2
        short_reasons.append("BOS bearish")

    if choch["bull"]:
        long_score += 2
        long_reasons.append("CHoCH bullish")

    if choch["bear"]:
        short_score += 2
        short_reasons.append("CHoCH bearish")

    if sweep["bull"]:
        long_score += 2
        long_reasons.append("liquidity sweep")

    if sweep["bear"]:
        short_score += 2
        short_reasons.append("liquidity sweep")

    # Price action
    if bullish_rejection(current) or bull_engulf:
        long_score += 1
        long_reasons.append("bullish PA")

    if bearish_rejection(current) or bear_engulf:
        short_score += 1
        short_reasons.append("bearish PA")

    # Volatility
    if current_atr / current["close"] >= 0.0015:
        long_score += 1
        short_score += 1

    # Pullback
    long_pullback = pullback_confirmation(candles, "LONG", e20[i])
    short_pullback = pullback_confirmation(candles, "SHORT", e20[i])

    if long_pullback:
        long_score += 1
        long_reasons.append("pullback")

    if short_pullback:
        short_score += 1
        short_reasons.append("pullback")

    # --------------------------------------------------------
    # Direction selection
    # --------------------------------------------------------

    if long_score >= MIN_4H_SCORE and long_score > short_score:
        direction = "LONG"
    elif short_score >= MIN_4H_SCORE and short_score > long_score:
        direction = "SHORT"
    else:
        return None, f"NO_DIRECTION | close={current["close"]:.8f} EMA20={e20[i]:.8f} EMA50={e50[i]:.8f} EMA200={e200[i]:.8f} RSI={current_rsi:.2f}"

    # A range is not automatically rejected if momentum has
    # a clean directional break/sweep. This prevents the
    # over-filtering seen in FINAL.
    if structure["state"] == "RANGE":
        directional_event = (
            bos["bull"]
            or bos["bear"]
            or choch["bull"]
            or choch["bear"]
            or sweep["bull"]
            or sweep["bear"]
        )
        if not directional_event:
            return None, "STRUCTURE_RANGE_NO_EVENT"

    # Prevent contradictory structure.
    if direction == "LONG":
        if bos["bear"] and not sweep["bull"] and not choch["bull"]:
            return None, "BEARISH_STRUCTURE_CONFLICT"
    else:
        if bos["bull"] and not sweep["bear"] and not choch["bear"]:
            return None, "BULLISH_STRUCTURE_CONFLICT"

    return {
        "direction": direction,
        "score": long_score if direction == "LONG" else short_score,
        "long_score": long_score,
        "short_score": short_score,
        "entry": current["close"],
        "atr": current_atr,
        "candle_time": current["time"],
        "structure": structure,
        "bos": bos,
        "choch": choch,
        "sweep": sweep,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
        "candles": candles,
        "ema20": e20[i],
        "ema50": e50[i],
        "ema200": e200[i],
        "rsi": current_rsi,
    }, "VALID"


# ============================================================
# 1H CONFIRMATION
# ============================================================

def analyze_1h(candles, direction):

    if len(candles) < 250:
        return {"valid": False, "score": 0, "reason": "NOT_ENOUGH_DATA"}

    closes = [c["close"] for c in candles]

    e20 = ema(closes, EMA_FAST)
    e50 = ema(closes, EMA_MID)
    e200 = ema(closes, EMA_SLOW)
    rsi_values = rsi(closes, RSI_PERIOD)

    i = len(candles) - 1
    c = candles[i]
    p = candles[i - 1]

    score = 0
    reasons = []

    if direction == "LONG":
        if c["close"] > e200[i] and e20[i] > e50[i]:
            score += 1
            reasons.append("trend")

        if rsi_values[i] > 50 and rsi_values[i] > rsi_values[i - 1]:
            score += 1
            reasons.append("RSI")

        if bullish_rejection(c) or bullish_engulfing(p, c):
            score += 1
            reasons.append("price action")

        # Retest of fast EMA
        if c["low"] <= e20[i] * 1.003 and c["close"] > e20[i]:
            score += 1
            reasons.append("EMA retest")

    else:
        if c["close"] < e200[i] and e20[i] < e50[i]:
            score += 1
            reasons.append("trend")

        if rsi_values[i] < 50 and rsi_values[i] < rsi_values[i - 1]:
            score += 1
            reasons.append("RSI")

        if bearish_rejection(c) or bearish_engulfing(p, c):
            score += 1
            reasons.append("price action")

        if c["high"] >= e20[i] * 0.997 and c["close"] < e20[i]:
            score += 1
            reasons.append("EMA retest")

    return {
        "valid": score >= MIN_1H_SCORE,
        "score": score,
        "reasons": reasons,
        "rsi": rsi_values[i],
    }


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_sl(signal):
    entry = signal["entry"]
    atr_value = signal["atr"]
    direction = signal["direction"]
    structure = signal["structure"]

    if direction == "LONG":
        atr_stop = entry - ATR_MULTIPLIER * atr_value

        structural_candidates = []
        if structure["last_low"] is not None:
            structural_candidates.append(
                structure["last_low"] - ATR_LEVEL_BUFFER * atr_value
            )

        if structural_candidates:
            structural_stop = min(structural_candidates)
            stop = min(atr_stop, structural_stop)
        else:
            stop = atr_stop

        risk = entry - stop

    else:
        atr_stop = entry + ATR_MULTIPLIER * atr_value

        structural_candidates = []
        if structure["last_high"] is not None:
            structural_candidates.append(
                structure["last_high"] + ATR_LEVEL_BUFFER * atr_value
            )

        if structural_candidates:
            structural_stop = max(structural_candidates)
            stop = max(atr_stop, structural_stop)
        else:
            stop = atr_stop

        risk = stop - entry

    min_risk = entry * MIN_SL_PERCENT
    max_risk = entry * MAX_SL_PERCENT

    risk = max(risk, min_risk)
    risk = min(risk, max_risk)

    stop = entry - risk if direction == "LONG" else entry + risk

    return stop, risk / entry


def calculate_tp(entry, direction):
    if direction == "LONG":
        return entry * (1.0 + TP_PERCENT)
    return entry * (1.0 - TP_PERCENT)


def tp_path_clear(signal, tp):
    entry = signal["entry"]
    atr_value = signal["atr"]
    candles = signal["candles"]
    buffer = atr_value * ATR_LEVEL_BUFFER

    if signal["direction"] == "LONG":
        levels = resistance_levels(candles, entry)

        blockers = [
            level for level in levels
            if level <= tp + buffer
        ]

        if blockers:
            return False, "RESISTANCE", min(blockers)

    else:
        levels = support_levels(candles, entry)

        blockers = [
            level for level in levels
            if level >= tp - buffer
        ]

        if blockers:
            return False, "SUPPORT", max(blockers)

    return True, None, None


def calculate_levels(signal):
    entry = signal["entry"]
    direction = signal["direction"]

    sl, risk_ratio = calculate_sl(signal)
    tp = calculate_tp(entry, direction)

    reward = abs(tp - entry)
    risk = abs(entry - sl)

    if risk <= 0:
        return {"valid": False, "reason": "INVALID_RISK"}

    # With a fixed TP, RR imposes a hard maximum acceptable risk.
    max_risk_for_rr = reward / MIN_RR

    if risk > max_risk_for_rr:
        return {
            "valid": False,
            "reason": f"RR_TOO_LOW_1:{reward / risk:.2f}"
        }

    rr = reward / risk

    if rr < MIN_RR:
        return {
            "valid": False,
            "reason": f"RR_TOO_LOW_1:{rr:.2f}",
        }

    clear, blocker_type, blocker = tp_path_clear(signal, tp)

    if not clear:
        return {
            "valid": False,
            "reason": f"TP_BLOCKED_{blocker_type}_{blocker:.8f}",
        }

    return {
        "valid": True,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_percent": risk_ratio * 100.0,
        "rr": rr,
    }


# ============================================================
# SIGNAL UTILITIES
# ============================================================

def signal_expired(candle_time):
    age = (int(time.time()) - candle_time) / 3600.0
    return age > EXPIRATION_HOURS


def entry_valid(entry, current, atr_value=None):
    """
    Validate whether the current price is still reasonably close to the
    setup entry. The window adapts to volatility but is never wider than
    ENTRY_WINDOW.
    """
    if entry <= 0 or current <= 0:
        return False, 0.0, ENTRY_WINDOW

    distance = abs(current - entry) / entry

    if atr_value is not None and atr_value > 0:
        atr_window = (atr_value / entry) * 0.50
        allowed = min(ENTRY_WINDOW, max(0.0030, atr_window))
    else:
        allowed = ENTRY_WINDOW

    return distance <= allowed, distance, allowed


def signal_strength(score4, score1):
    if score4 >= 10 and score1 >= 3:
        return "🔥 VERY STRONG"
    if score4 >= 8 and score1 >= 3:
        return "💪 STRONG"
    return "🟡 VALID"


def has_active_coin(state, symbol):
    for trade in state.get("active_trades", {}).values():
        if trade.get("symbol") == symbol:
            return True
    return False


# ============================================================
# ACTIVE TRADE TRACKING
# ============================================================

def check_active_trades(state):
    active = state.get("active_trades", {})
    completed = state.setdefault("completed_trades", [])

    for trade_id, trade in list(active.items()):
        try:
            current = get_current_price(trade["lbank_symbol"])
        except Exception as exc:
            print(f"{trade['symbol']}: active price error: {exc}")
            continue

        direction = trade["direction"]
        entry = trade["entry"]
        tp = trade["tp"]
        sl = trade["sl"]

        result = None

        # Conservative ordering:
        # if a single polling price crosses both levels,
        # do not invent which level was hit.
        if direction == "LONG":
            if current >= tp:
                result = "TP"
            elif current <= sl:
                result = "SL"
        else:
            if current <= tp:
                result = "TP"
            elif current >= sl:
                result = "SL"

        if result is None:
            continue

        if result == "TP":
            pnl = abs(tp - entry) / entry * 100.0
        else:
            pnl = -abs(sl - entry) / entry * 100.0

        trade["result"] = result
        trade["exit_price"] = current
        trade["pnl_percent"] = pnl
        trade["closed_at"] = int(time.time())

        completed.append(trade)

        emoji = "✅" if result == "TP" else "❌"

        message = (
            f"{emoji} SCORE HUNTER PRO FINAL\n\n"
            f"TRADE CLOSED\n\n"
            f"💰 {trade['symbol']}USDT\n"
            f"📊 {direction}\n"
            f"📌 Result: {result}\n\n"
            f"💵 Entry: {entry:.8f}\n"
            f"🏁 Exit: {current:.8f}\n"
            f"📈 P/L: {pnl:+.2f}%\n"
            f"🎯 TP: 1.00%\n"
            f"🏦 LBank"
        )

        send_telegram(message)
        del active[trade_id]

    state["active_trades"] = active


# ============================================================
# STATISTICS
# ============================================================

def statistics(state):
    trades = state.get("completed_trades", [])

    if not trades:
        return None

    wins = sum(t.get("result") == "TP" for t in trades)
    losses = sum(t.get("result") == "SL" for t in trades)

    decided = wins + losses
    win_rate = wins / decided * 100.0 if decided else 0.0

    pnl = sum(t.get("pnl_percent", 0.0) for t in trades)

    return {
        "total": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl": pnl,
    }


# ============================================================
# MAIN SCAN
# ============================================================

def main():
    print("🟢 SCORE HUNTER PRO FINAL")
    print("🔒 CLOSED CANDLE MODE: forming candle is ignored")
    print("🧠 MARKET STRUCTURE + PRICE ACTION + LIQUIDITY")
    print("📊 Main structure: 4H")
    print("📱 Entry confirmation: 1H")
    print("🔎 BOS / CHoCH enabled")
    print("💧 Liquidity sweep enabled")
    print("↩️ Pullback / retest enabled")
    print("🕯 Price action confirmation enabled")
    print(f"🎯 Fixed TP: {TP_PERCENT * 100:.2f}%")
    print(f"🔒 Entry window: {ENTRY_WINDOW * 100:.2f}%")
    print(f"🛑 ATR multiplier: {ATR_MULTIPLIER}")
    print(f"⚖️ Minimum R:R: 1:{MIN_RR}")
    print("⏳ Signal age gate: OFF (latest closed candle + duplicate protection)")

    state = load_state()

    check_active_trades(state)

    for symbol, lbank_symbol in COINS.items():
        print(f"\n========== {symbol} ==========")

        try:
            candles4 = closed_candles(
                get_candles(lbank_symbol, MAIN_TF)
            )

            latest = candles4[-1]
            print(
                f"{symbol}: latest closed 4H: "
                f"{latest['time']}"
            )

            signal, status = analyze_4h(candles4)

            if signal is None:
                print(f"{symbol}: ❌ {status}")
                continue

            direction = signal["direction"]
            score4 = signal["score"]

            candles1 = closed_candles(
                get_candles(lbank_symbol, ENTRY_TF)
            )

            confirmation = analyze_1h(candles1, direction)

            if not confirmation["valid"]:
                print(
                    f"{symbol}: ❌ 1H_CONFIRMATION_FAILED "
                    f"({confirmation['score']}) "
                    f"reasons={','.join(confirmation.get('reasons', [])) or 'none'} "
                    f"RSI={confirmation.get('rsi', 0):.2f}"
                )
                continue

            score1 = confirmation["score"]

            current = get_current_price(lbank_symbol)
            entry = signal["entry"]

            distance = abs(current - entry) / entry

            print(
                f"{symbol}: {direction} "
                f"4H={score4} "
                f"1H={score1} "
                f"entry={entry:.8f} "
                f"current={current:.8f} "
                f"distance={distance * 100:.3f}%"
            )

            entry_ok, distance, allowed_window = entry_valid(
                entry, current, signal.get("atr")
            )

            if not entry_ok:
                print(
                    f"{symbol}: ❌ ENTRY_TOO_FAR "
                    f"distance={distance * 100:.3f}% "
                    f"allowed={allowed_window * 100:.3f}%"
                )
                continue

            # The setup is confirmed on the latest closed candles, but the
            # executable entry is the current market price. This prevents
            # stale TP/SL levels when price has moved inside the entry window.
            signal = dict(signal)
            signal["entry"] = current
            entry = current

            if has_active_coin(state, symbol):
                print(f"{symbol}: ❌ ACTIVE_TRADE_EXISTS")
                continue

            signal_id = (
                f"{symbol}_{signal['candle_time']}_{direction}"
            )

            if state["last_signals"].get(symbol) == signal_id:
                print(f"{symbol}: ❌ DUPLICATE_SIGNAL")
                continue

            levels = calculate_levels(signal)

            if not levels["valid"]:
                print(
                    f"{symbol}: ❌ RISK_FILTER "
                    f"{levels['reason']}"
                )
                continue

            tp = levels["tp"]
            sl = levels["sl"]
            rr = levels["rr"]
            risk_percent = levels["risk_percent"]

            strength = signal_strength(score4, score1)

            trade = {
                "trade_id": signal_id,
                "symbol": symbol,
                "lbank_symbol": lbank_symbol,
                "direction": direction,
                "score_4h": score4,
                "score_1h": score1,
                "strength": strength,
                "entry": entry,
                "current_price_at_signal": current,
                "tp": tp,
                "sl": sl,
                "risk_percent": risk_percent,
                "rr": rr,
                "candle_time": signal["candle_time"],
                "created_at": int(time.time()),
                "status": "ACTIVE",
                "structure": signal["structure"]["state"],
                "bos": signal["bos"],
                "choch": signal["choch"],
                "liquidity_sweep": signal["sweep"],
                "confirmation_reasons": confirmation["reasons"],
            }

            state["active_trades"][signal_id] = trade
            state["last_signals"][symbol] = signal_id
            save_state(state)

            if direction == "LONG":
                direction_text = "🟢 LONG"
                tp_move = (tp - entry) / entry * 100.0
                sl_move = (sl - entry) / entry * 100.0
            else:
                direction_text = "🔴 SHORT"
                tp_move = (entry - tp) / entry * 100.0
                sl_move = (entry - sl) / entry * 100.0

            message = (
                "🚨 SCORE HUNTER PRO FINAL 🚨\n\n"
                f"💰 {symbol}USDT\n"
                f"📊 {direction_text}\n"
                f"{strength}\n\n"
                f"⭐ 4H Score: {score4}\n"
                f"📱 1H Confirm: {score1}\n"
                f"🧱 Structure: {signal['structure']['state']}\n"
                f"🔎 BOS: "
                f"{'YES' if (signal['bos']['bull'] or signal['bos']['bear']) else 'NO'}\n"
                f"🔄 CHoCH: "
                f"{'YES' if (signal['choch']['bull'] or signal['choch']['bear']) else 'NO'}\n"
                f"💧 Liquidity sweep: "
                f"{'YES' if (signal['sweep']['bull'] or signal['sweep']['bear']) else 'NO'}\n\n"
                f"💵 Entry: {entry:.8f}\n"
                f"📍 Current: {current:.8f}\n"
                f"📏 Distance: {distance * 100:.2f}%\n\n"
                f"🎯 TP: {tp:.8f} ({tp_move:+.2f}%)\n"
                f"🛑 SL: {sl:.8f} ({sl_move:+.2f}%)\n"
                f"⚖️ R:R: 1:{rr:.2f}\n"
                f"📐 Risk: {risk_percent:.2f}%\n\n"
                "🧱 TP Path: CLEAR\n"
                "📊 Structure: 4H\n"
                "📱 Confirmation: 1H\n"
                "🎯 TP target: 1.00%\n"
                "🔒 Entry window: ±0.30%\n"
                "🏦 Data: LBank\n\n"
                "⚠️ Signal only. Manage leverage and risk."
            )

            if send_telegram(message):
                print(f"{symbol}: ✅ SIGNAL SENT")
            else:
                print(f"{symbol}: ⚠️ TELEGRAM FAILED")

        except Exception as exc:
            print(f"{symbol}: ❌ ERROR: {exc}")

    save_state(state)

    stats = statistics(state)
    if stats and stats["total"] % 10 == 0:
        message = (
            "📊 SCORE HUNTER PRO FINAL\n\n"
            "STATISTICS\n\n"
            f"📌 Closed trades: {stats['total']}\n"
            f"✅ TP: {stats['wins']}\n"
            f"❌ SL: {stats['losses']}\n"
            f"🎯 Win rate: {stats['win_rate']:.1f}%\n"
            f"📈 Total P/L: {stats['pnl']:+.2f}%"
        )
        send_telegram(message)

    print("\n✅ SCORE HUNTER PRO FINAL scan completed.")


if __name__ == "__main__":
    main()
