import os
import time
import math
import requests
from datetime import datetime, timezone

# ============================================================
# SCORE HUNTER PRO - BINANCE FUTURES
# Exact Python recreation of the supplied TradingView Pine logic
#
# Market: Binance USDⓈ-M Futures
# Symbols: ETHUSDT, SOLUSDT, XRPUSDT, APTUSDT
# Timeframe: 4H
# Signal: CLOSED CANDLE ONLY
# Minimum score: 5/7
# ============================================================

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"

SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "APTUSDT"]
INTERVAL = "4h"
REQUIRED_SCORE = 5

# Optional Telegram configuration.
# Set these environment variables before running:
# TELEGRAM_BOT_TOKEN=...
# TELEGRAM_CHAT_ID=...
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

POLL_SECONDS = 30
REQUEST_TIMEOUT = 15
KLINE_LIMIT = 300

# Keep the last processed CLOSED candle per symbol.
last_processed_close_time = {symbol: None for symbol in SYMBOLS}


# ============================================================
# HTTP
# ============================================================

session = requests.Session()
session.headers.update({"User-Agent": "ScoreHunterPro/1.0"})


def get_klines(symbol, limit=KLINE_LIMIT):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": limit,
    }

    response = session.get(
        BINANCE_KLINES_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 220:
        raise RuntimeError(f"{symbol}: insufficient Binance Futures kline data")

    return data


# ============================================================
# EXACT-STYLE INDICATOR HELPERS
# ============================================================

def sma(values, length):
    out = [None] * len(values)

    if len(values) < length:
        return out

    running = sum(values[:length])
    out[length - 1] = running / length

    for i in range(length, len(values)):
        running += values[i] - values[i - length]
        out[i] = running / length

    return out


def ema(values, length):
    """
    TradingView ta.ema-style recursive EMA.
    Seed = SMA(length), then:
        EMA = alpha * source + (1-alpha) * EMA[1]
    """
    out = [None] * len(values)

    if len(values) < length:
        return out

    seed = sum(values[:length]) / length
    out[length - 1] = seed

    alpha = 2.0 / (length + 1.0)

    for i in range(length, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]

    return out


def rma(values, length):
    """
    TradingView ta.rma-style Wilder moving average.
    Seed = SMA(length), then:
        RMA = (previous_RMA*(length-1) + source) / length
    """
    out = [None] * len(values)

    if len(values) < length:
        return out

    seed = sum(values[:length]) / length
    out[length - 1] = seed

    alpha = 1.0 / length

    for i in range(length, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]

    return out


def pine_rsi(closes, length=14):
    """
    Recreation of TradingView ta.rsi(close, 14):
      up   = RMA(max(change, 0), length)
      down = RMA(max(-change, 0), length)
      RSI  = 100 - 100/(1 + up/down)

    The first change is zero because Pine's change() is na and
    the RMA warm-up is represented consistently here.
    """
    if len(closes) < length + 1:
        return [None] * len(closes)

    changes = [0.0]
    for i in range(1, len(closes)):
        changes.append(closes[i] - closes[i - 1])

    gains = [max(x, 0.0) for x in changes]
    losses = [max(-x, 0.0) for x in changes]

    avg_gain = rma(gains, length)
    avg_loss = rma(losses, length)

    out = [None] * len(closes)

    for i in range(len(closes)):
        if avg_gain[i] is None or avg_loss[i] is None:
            continue

        if avg_loss[i] == 0:
            # TradingView RSI becomes 100 when average loss is zero
            out[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


def true_ranges(highs, lows, closes):
    tr = [None] * len(closes)

    for i in range(len(closes)):
        if i == 0:
            # Pine ta.tr(true) uses high-low when there is no previous close.
            tr[i] = highs[i] - lows[i]
        else:
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

    return tr


def pine_atr(highs, lows, closes, length=14):
    tr = true_ranges(highs, lows, closes)
    return rma(tr, length)


def highest_previous(highs, length=6):
    """
    Exact conceptual equivalent of:
        ta.highest(high, 6)[1]

    At candle i, this returns the highest high among candles
    i-6 ... i-1, excluding the current candle.
    """
    out = [None] * len(highs)

    for i in range(length, len(highs)):
        out[i] = max(highs[i - length:i])

    return out


def lowest_previous(lows, length=6):
    """
    Exact conceptual equivalent of:
        ta.lowest(low, 6)[1]
    """
    out = [None] * len(lows)

    for i in range(length, len(lows)):
        out[i] = min(lows[i - length:i])

    return out


# ============================================================
# DATA PARSING
# ============================================================

def parse_klines(raw):
    candles = []

    for row in raw:
        candles.append({
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": int(row[6]),
        })

    return candles


def get_closed_candles(symbol):
    """
    Binance normally returns the currently forming candle as the last row.
    We explicitly remove it.

    This is the core CLOSED CANDLE LOCK:
    only candles whose close_time <= current UTC time are evaluated.
    """
    raw = get_klines(symbol)
    candles = parse_klines(raw)

    now_ms = int(time.time() * 1000)

    closed = [
        c for c in candles
        if c["close_time"] <= now_ms
    ]

    if len(closed) < 220:
        raise RuntimeError(f"{symbol}: not enough CLOSED candles")

    return closed


# ============================================================
# SIGNAL ENGINE
# ============================================================

def calculate_signal(symbol, candles):
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)

    rsi = pine_rsi(closes, 14)
    vol_ma = sma(volumes, 20)
    atr = pine_atr(highs, lows, closes, 14)

    recent_high = highest_previous(highs, 6)
    recent_low = lowest_previous(lows, 6)

    # The last element is guaranteed to be a CLOSED candle.
    i = len(candles) - 1

    required = [
        ema20[i], ema50[i], ema200[i],
        rsi[i], vol_ma[i], atr[i],
        recent_high[i], recent_low[i]
    ]

    if any(x is None for x in required):
        return None

    close = closes[i]
    open_ = opens[i]
    high = highs[i]
    low = lows[i]
    volume = volumes[i]

    # --------------------------------------------------------
    # Pine:
    # longTrend = close > ema200 and ema20 > ema50
    # shortTrend = close < ema200 and ema20 < ema50
    # --------------------------------------------------------
    long_trend = (
        close > ema200[i]
        and ema20[i] > ema50[i]
    )

    short_trend = (
        close < ema200[i]
        and ema20[i] < ema50[i]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------
    long_rsi = (
        rsi[i] > 50
        and rsi[i] < 72
        and rsi[i] > rsi[i - 1]
    )

    short_rsi = (
        rsi[i] < 50
        and rsi[i] > 28
        and rsi[i] < rsi[i - 1]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------
    volume_ok = volume >= vol_ma[i]

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------
    bull_break = close > recent_high[i]
    bear_break = close < recent_low[i]

    # --------------------------------------------------------
    # EMA PULLBACK
    # --------------------------------------------------------
    long_pullback = (
        low <= ema20[i]
        and close > ema20[i]
    )

    short_pullback = (
        high >= ema20[i]
        and close < ema20[i]
    )

    # --------------------------------------------------------
    # CANDLE CONFIRMATION
    # --------------------------------------------------------
    candle_range = high - low

    if candle_range > 0:
        bull_candle = (
            close > open_
            and ((close - open_) / candle_range) >= 0.40
        )

        bear_candle = (
            close < open_
            and ((open_ - close) / candle_range) >= 0.40
        )
    else:
        bull_candle = False
        bear_candle = False

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------
    volatility_ok = (atr[i] / close) >= 0.002

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

    long_signal = long_score >= REQUIRED_SCORE
    short_signal = short_score >= REQUIRED_SCORE

    close_time = candles[i]["close_time"]

    return {
        "symbol": symbol,
        "candle_open_time": candles[i]["open_time"],
        "candle_close_time": close_time,
        "close": close,
        "ema20": ema20[i],
        "ema50": ema50[i],
        "ema200": ema200[i],
        "rsi": rsi[i],
        "volume": volume,
        "volume_ma": vol_ma[i],
        "atr": atr[i],
        "recent_high": recent_high[i],
        "recent_low": recent_low[i],
        "long_score": long_score,
        "short_score": short_score,
        "long_signal": long_signal,
        "short_signal": short_signal,
        "conditions": {
            "longTrend": long_trend,
            "longRSI": long_rsi,
            "volumeOK": volume_ok,
            "bullBreak": bull_break,
            "longPullback": long_pullback,
            "bullCandle": bull_candle,
            "volatilityOK": volatility_ok,
            "shortTrend": short_trend,
            "shortRSI": short_rsi,
            "bearBreak": bear_break,
            "shortPullback": short_pullback,
            "bearCandle": bear_candle,
        },
    }


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[TELEGRAM NOT CONFIGURED]")
        print(text)
        return False

    url = TELEGRAM_URL.format(TELEGRAM_BOT_TOKEN)

    response = session.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    return True


def format_price(value):
    if value >= 1000:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def build_signal_message(result, direction):
    symbol = result["symbol"]
    entry = result["close"]

    if direction == "LONG":
        tp = entry * 1.01
        sl = entry * 0.995
        score = result["long_score"]
        emoji = "🟢"
    else:
        tp = entry * 0.99
        sl = entry * 1.005
        score = result["short_score"]
        emoji = "🔴"

    close_dt = datetime.fromtimestamp(
        result["candle_close_time"] / 1000,
        tz=timezone.utc,
    )

    return (
        f"{emoji} SCORE HUNTER PRO\n\n"
        f"{symbol} {direction}\n"
        f"⏱ 4H CLOSED CANDLE\n"
        f"📊 Score: {score}/7\n\n"
        f"Entry: {format_price(entry)}\n"
        f"TP: {format_price(tp)} ({'+1%' if direction == 'LONG' else '-1%'})\n"
        f"SL: {format_price(sl)} ({'-0.5%' if direction == 'LONG' else '+0.5%'})\n\n"
        f"RSI: {result['rsi']:.2f}\n"
        f"EMA20: {format_price(result['ema20'])}\n"
        f"EMA50: {format_price(result['ema50'])}\n"
        f"EMA200: {format_price(result['ema200'])}\n\n"
        f"Closed: {close_dt.strftime('%Y-%m-%d %H:%M UTC')}"
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_result(result):
    print(
        f"{result['symbol']} | "
        f"close={format_price(result['close'])} | "
        f"RSI={result['rsi']:.2f} | "
        f"LONG={result['long_score']}/7 | "
        f"SHORT={result['short_score']}/7"
    )


# ============================================================
# MAIN LOOP
# ============================================================

def process_symbol(symbol):
    try:
        candles = get_closed_candles(symbol)

        result = calculate_signal(symbol, candles)

        if result is None:
            print(f"{symbol}: indicator warm-up/incomplete")
            return

        print_result(result)

        close_time = result["candle_close_time"]

        # Process each CLOSED candle exactly once.
        if last_processed_close_time[symbol] == close_time:
            return

        # Mark this closed candle as processed before sending.
        # This prevents duplicate Telegram messages after a restart/poll.
        last_processed_close_time[symbol] = close_time

        if result["long_signal"]:
            message = build_signal_message(result, "LONG")
            print("\n" + message + "\n")
            send_telegram(message)

        if result["short_signal"]:
            message = build_signal_message(result, "SHORT")
            print("\n" + message + "\n")
            send_telegram(message)

    except requests.RequestException as e:
        print(f"{symbol}: Binance/API error: {e}")

    except Exception as e:
        print(f"{symbol}: ERROR: {e}")


def main():
    print("=" * 64)
    print("🟢 SCORE HUNTER PRO - BINANCE FUTURES")
    print("🔒 CLOSED CANDLE MODE")
    print("📊 SOURCE: Binance USDⓈ-M Futures")
    print("⏱ TIMEFRAME: 4H")
    print("🪙 SYMBOLS: ETH / SOL / XRP / APT")
    print("🎯 MINIMUM SCORE: 5/7")
    print("=" * 64)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print("📲 Telegram: CONFIGURED")
    else:
        print("📲 Telegram: NOT CONFIGURED (signals will print to console)")

    print()

    while True:
        cycle_start = time.time()

        for symbol in SYMBOLS:
            process_symbol(symbol)

        elapsed = time.time() - cycle_start
        sleep_for = max(1, POLL_SECONDS - elapsed)

        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
