import os
import json
import requests
from datetime import datetime, timezone

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"
TICKER_URL = "https://api.kraken.com/0/public/Ticker"
STATE_FILE = "state.json"

INTERVAL_4H = 240
INTERVAL_1H = 60

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "XDGUSDT",
}

TP_PERCENT = 1.0
SL_PERCENT = 0.50
TP_BUFFER_PERCENT = 0.10

# 4H is directional context; 1H is the actual trigger.
MIN_4H_QUALITY = 3
MIN_1H_CONFIRMATION = 3

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    r = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    print("Telegram:", r.status_code)
    print(r.text)
    r.raise_for_status()

def get_current_price(pair):
    """Get the latest live price from Kraken Ticker."""
    r = requests.get(
        TICKER_URL,
        params={"pair": pair},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()

    if payload.get("error"):
        raise RuntimeError(payload["error"])

    result = payload.get("result", {})
    if not result:
        raise RuntimeError(f"No ticker data returned for {pair}")

    key = next(iter(result))
    ticker = result[key]

    # Kraken's 'c' field is the last traded price.
    last_price = ticker.get("c", [None])[0]
    if last_price is None:
        raise RuntimeError(f"No last price in ticker for {pair}")

    return float(last_price)


def monitor_open_trades(state):
    """
    Check all previously sent signals against the live Kraken price.
    Each trade gets exactly one final result notification.
    """
    for symbol, coin_state in state.items():
        signal = coin_state.get("last_signal")
        if not signal:
            continue

        # Already finalized.
        if coin_state.get("trade_result"):
            continue

        pair = COINS.get(symbol)
        if not pair:
            continue

        try:
            price = get_current_price(pair)
            direction = signal["direction"]
            entry = float(signal["entry"])

            if direction == "LONG":
                tp = entry * (1 + TP_PERCENT / 100)
                sl = entry * (1 - SL_PERCENT / 100)

                if price >= tp:
                    result = "TP"
                    result_price = price
                elif price <= sl:
                    result = "SL"
                    result_price = price
                else:
                    continue

            else:
                tp = entry * (1 - TP_PERCENT / 100)
                sl = entry * (1 + SL_PERCENT / 100)

                if price <= tp:
                    result = "TP"
                    result_price = price
                elif price >= sl:
                    result = "SL"
                    result_price = price
                else:
                    continue

            if result == "TP":
                emoji = "✅"
                title = "TAKE PROFIT HIT"
                pnl_text = "+1.0%"
            else:
                emoji = "🛑"
                title = "STOP LOSS HIT"
                pnl_text = "-0.5%"

            message = (
                f"{emoji} SCORE HUNTER RESULT\n\n"
                f"💰 {symbol}USDT\n"
                f"📊 {direction}\n"
                f"🏁 {title}\n"
                f"💵 Entry: {entry:.8f}\n"
                f"📍 Price: {result_price:.8f}\n"
                f"📈 Result: {pnl_text}\n\n"
                "⏱ Timeframe: 4H / 1H\n"
                "🕯 Closed candle signal"
            )

            send_telegram(message)

            coin_state["trade_result"] = result
            coin_state["trade_result_price"] = result_price
            coin_state["trade_result_time"] = int(
                datetime.now(timezone.utc).timestamp()
            )
            state[symbol] = coin_state
            save_state(state)

            print(
                f"🏁 {symbol}: {direction} "
                f"{result} HIT at {result_price:.8f}"
            )

        except Exception as e:
            print(
                f"⚠️ {symbol} TP/SL monitor error: "
                f"{type(e).__name__}: {e}"
            )

def get_ohlc(pair, interval):
    r = requests.get(
        KRAKEN_URL,
        params={"pair": pair, "interval": interval},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    result = payload.get("result", {})
    key = next((k for k in result if k != "last"), None)
    if key is None:
        raise RuntimeError("No OHLC data returned")

    candles = []
    for row in result[key]:
        candles.append({
            "time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6]),
        })

    candles.sort(key=lambda x: x["time"])

    # Kraken's final row may still be forming.
    if len(candles) > 1:
        candles = candles[:-1]

    return candles

def ema(values, period):
    if len(values) < period:
        return None
    v = sum(values[:period]) / period
    m = 2.0 / (period + 1)
    for x in values[period:]:
        v = (x - v) * m + v
    return v

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def rsi(values, period=14):
    if len(values) <= period:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))

    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period

    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)

def atr(candles, period=14):
    if len(candles) <= period:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l = candles[i]["high"], candles[i]["low"]
        pc = candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period

def robust_structure(candles, lookback=30):
    """Less brittle 4H structure: EMA alignment + recent HH/LL bias."""
    if len(candles) < 220:
        return "NONE"

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    e200 = ema(closes, 200)
    if None in (e20, e50, e200):
        return "NONE"

    price = closes[-1]
    recent_h = max(highs[-lookback:])
    previous_h = max(highs[-2 * lookback:-lookback])
    recent_l = min(lows[-lookback:])
    previous_l = min(lows[-2 * lookback:-lookback])

    bullish = price > e200 and e20 > e50
    bearish = price < e200 and e20 < e50

    hh = recent_h > previous_h
    hl = recent_l > previous_l
    lh = recent_h < previous_h
    ll = recent_l < previous_l

    long_votes = int(bullish) + int(hh) + int(hl)
    short_votes = int(bearish) + int(lh) + int(ll)

    if long_votes >= 2 and long_votes > short_votes:
        return "LONG"
    if short_votes >= 2 and short_votes > long_votes:
        return "SHORT"
    return "NONE"

def structure_quality_4h(candles, direction):
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    price = closes[-1]

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    e200 = ema(closes, 200)
    rv = rsi(closes, 14)
    va = sma(volumes, 20)
    a = atr(candles, 14)

    if None in (e20, e50, e200, rv, va, a):
        return 0, {}

    trend = (
        price > e200 and e20 > e50
        if direction == "LONG"
        else price < e200 and e20 < e50
    )

    rsi_ok = (
        45 <= rv <= 72
        if direction == "LONG"
        else 28 <= rv <= 55
    )

    volume_ok = volumes[-1] >= 1.05 * va

    volatility_ok = a / price >= 0.0015

    score = (
        int(trend)
        + int(rsi_ok)
        + int(volume_ok)
        + int(volatility_ok)
    )
    return score, {
        "trend": trend,
        "rsi": rv,
        "volume": volume_ok,
        "volatility": volatility_ok,
    }

def one_hour_confirmation(candles, direction):
    if len(candles) < 80:
        return 0, {}

    closes = [c["close"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    price = closes[-1]
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    rv = rsi(closes, 14)
    va = sma(volumes, 20)
    a = atr(candles, 14)

    if None in (e20, e50, rv, va, a):
        return 0, {}

    trend = (
        price > e20 and e20 > e50
        if direction == "LONG"
        else price < e20 and e20 < e50
    )

    # BOS uses the immediately preceding 6 closed candles.
    recent_high = max(highs[-7:-1])
    recent_low = min(lows[-7:-1])
    bos = (
        price > recent_high
        if direction == "LONG"
        else price < recent_low
    )

    # Liquidity sweep: wick through a recent extreme, then close back
    # on the intended side of that level.
    sweep_high = max(highs[-12:-1])
    sweep_low = min(lows[-12:-1])
    sweep = (
        lows[-1] < sweep_low and price > sweep_low
        if direction == "LONG"
        else highs[-1] > sweep_high and price < sweep_high
    )

    # Retest of EMA20 with close back in directional side.
    retest = (
        lows[-1] <= e20 and price > e20
        if direction == "LONG"
        else highs[-1] >= e20 and price < e20
    )

    rng = highs[-1] - lows[-1]
    body_ratio = (
        abs(closes[-1] - opens[-1]) / rng if rng > 0 else 0
    )
    candle = (
        closes[-1] > opens[-1] and body_ratio >= 0.40
        if direction == "LONG"
        else closes[-1] < opens[-1] and body_ratio >= 0.40
    )

    volume = volumes[-1] >= 1.05 * va
    volatility = a / price >= 0.001

    score = sum([
        int(trend),
        int(bos),
        int(sweep),
        int(retest),
        int(candle),
        int(volume or volatility),
    ])

    return score, {
        "trend": trend,
        "bos": bos,
        "sweep": sweep,
        "retest": retest,
        "candle": candle,
        "volume": volume,
        "volatility": volatility,
        "rsi": rv,
    }

def tp_space_ok(candles, direction, entry):
    """Reject only when a nearby opposing level actually blocks the 1% TP."""
    if len(candles) < 35:
        return False, None

    target = (
        entry * (1 + TP_PERCENT / 100)
        if direction == "LONG"
        else entry * (1 - TP_PERCENT / 100)
    )

    buffer = TP_BUFFER_PERCENT / 100
    highs = [c["high"] for c in candles[-30:]]
    lows = [c["low"] for c in candles[-30:]]

    if direction == "LONG":
        blockers = [h for h in highs if h > entry and h < target]
        nearest = min(blockers) if blockers else None
        if nearest is not None and nearest >= target * (1 - buffer):
            return False, nearest
    else:
        blockers = [l for l in lows if l < entry and l > target]
        nearest = max(blockers) if blockers else None
        if nearest is not None and nearest <= target * (1 + buffer):
            return False, nearest

    return True, None

def create_message(symbol, direction, entry, score4, score1):
    if direction == "LONG":
        tp = entry * (1 + TP_PERCENT / 100)
        sl = entry * (1 - SL_PERCENT / 100)
        side = "🟢 LONG"
        tp_text = f"+{TP_PERCENT:.1f}%"
        sl_text = f"-{SL_PERCENT:.1f}%"
    else:
        tp = entry * (1 - TP_PERCENT / 100)
        sl = entry * (1 + SL_PERCENT / 100)
        side = "🔴 SHORT"
        tp_text = f"-{TP_PERCENT:.1f}%"
        sl_text = f"+{SL_PERCENT:.1f}%"

    return (
        "🚨 SCORE HUNTER PRO v3 🚨\n\n"
        f"💰 {symbol}USDT\n"
        f"📊 {side}\n"
        f"⭐ 4H Quality: {score4}/4\n"
        f"⭐ 1H Confirmation: {score1}/6\n"
        f"💵 Entry: {entry:.8f}\n"
        f"🎯 TP: {tp:.8f} ({tp_text})\n"
        f"🛑 SL: {sl:.8f} ({sl_text})\n\n"
        "⏱ Main structure: 4H\n"
        "⏱ Entry confirmation: 1H\n"
        "🕯 Closed candle confirmation\n"
        "⚠️ Manage risk."
    )

def main():
    print("🟢 SCORE HUNTER PRO v3 ROBUST 4H/1H")
    print("🔎 ROBUST 4H STRUCTURE + TRUE 1H TRIGGER")
    print(f"⭐ Minimum 4H quality: {MIN_4H_QUALITY}/4")
    print(f"⭐ Minimum 1H confirmation: {MIN_1H_CONFIRMATION}/6")
    print("🕯 CLOSED CANDLE ONLY - NO MID-CANDLE SIGNAL")
    print("♾️ DAILY SIGNAL LIMIT: DISABLED")
    print("📊 Coins: ETH / SOL / XRP / BTC / ADA / LINK / DOGE")
    print("⏱ Main structure: 4H | Entry confirmation: 1H")
    print("🎯 TP: 1.0% | 🛑 SL: 0.5%")
    print("📐 TP SPACE FILTER: ON")

    state = load_state()

    # TP/SL is monitored independently of new-candle checks.
    print("📡 LIVE TP/SL MONITOR: ON")
    monitor_open_trades(state)

    for symbol, pair in COINS.items():
        print(f"\n========== {symbol} ==========")
        try:
            c4 = get_ohlc(pair, INTERVAL_4H)
            c1 = get_ohlc(pair, INTERVAL_1H)

            if len(c4) < 220 or len(c1) < 80:
                print(f"{symbol}: insufficient data")
                continue

            latest1 = c1[-1]
            candle_id = latest1["time"]
            print(f"{symbol} latest CLOSED 1H: {candle_id}")

            # Only scan once per newly closed 1H candle.
            coin_state = state.get(symbol, {})
            if coin_state.get("last_checked_1h") == candle_id:
                print(f"{symbol}: No new 1H candle.")
                continue

            coin_state["last_checked_1h"] = candle_id

            structure = robust_structure(c4)
            print(f"{symbol} 4H Structure: {structure}")

            directions = [structure] if structure in ("LONG", "SHORT") else []

            signal_sent = False

            for direction in directions:
                q4, details4 = structure_quality_4h(c4, direction)
                print(
                    f"{symbol} {direction} 4H quality: "
                    f"{q4}/4 | trend={int(details4.get('trend', False))} "
                    f"rsi={details4.get('rsi', 0):.2f} "
                    f"volume={int(details4.get('volume', False))} "
                    f"volatility={int(details4.get('volatility', False))}"
                )

                if q4 < MIN_4H_QUALITY:
                    print(f"{symbol}: rejected - 4H quality")
                    continue

                q1, details1 = one_hour_confirmation(c1, direction)
                print(
                    f"{symbol} {direction} 1H confirmation: {q1}/6 "
                    f"| trend={int(details1.get('trend', False))} "
                    f"bos={int(details1.get('bos', False))} "
                    f"sweep={int(details1.get('sweep', False))} "
                    f"retest={int(details1.get('retest', False))} "
                    f"candle={int(details1.get('candle', False))} "
                    f"volume={int(details1.get('volume', False))}"
                )

                if q1 < MIN_1H_CONFIRMATION:
                    print(f"{symbol}: rejected - 1H confirmation")
                    continue

                entry = c1[-1]["close"]
                space_ok, blocker = tp_space_ok(c4, direction, entry)
                if not space_ok:
                    print(
                        f"{symbol}: rejected - TP space blocked "
                        f"by nearby level {blocker}"
                    )
                    continue

                signal_id = f"{candle_id}_{direction}"
                if coin_state.get("last_signal_id") == signal_id:
                    print(f"{symbol}: duplicate signal")
                    continue

                msg = create_message(
                    symbol, direction, entry, q4, q1
                )
                send_telegram(msg)

                coin_state.update({
                    "last_signal_id": signal_id,
                    "last_signal": {
                        "direction": direction,
                        "score_4h": q4,
                        "score_1h": q1,
                        "entry": entry,
                        "candle_1h": candle_id,
                    },
                })
                state[symbol] = coin_state
                save_state(state)

                print(f"🚨 {symbol}: {direction} SIGNAL SENT")
                signal_sent = True
                break

            if not signal_sent:
                print(f"{symbol}: No valid 4H/1H signal.")

            state[symbol] = coin_state
            save_state(state)

        except Exception as e:
            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: {e}"
            )
            continue

    save_state(state)
    print("\n✅ SCORE HUNTER PRO v3 SCAN COMPLETED")

if __name__ == "__main__":
    main()
