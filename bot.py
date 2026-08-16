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

MIN_4H_QUALITY = 3
MIN_1H_CONFIRMATION = 3


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"⚠️ state.json load error: {type(e).__name__}: {e}")
        return {}

    # Keep only valid per-coin dictionaries.
    # Old/malformed top-level metadata entries are ignored.
    state = {}
    if isinstance(raw, dict):
        for symbol in COINS:
            value = raw.get(symbol)
            if isinstance(value, dict):
                state[symbol] = value

    return state


def save_state(state):
    clean = {}
    for symbol in COINS:
        value = state.get(symbol)
        if isinstance(value, dict):
            clean[symbol] = value

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


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

    last_price = ticker.get("c", [None])[0]
    if last_price is None:
        raise RuntimeError(f"No last price in ticker for {pair}")

    return float(last_price)


def monitor_open_trades(state):
    """
    Check currently active signals once per workflow run.

    Important:
    This intentionally does NOT run continuously.
    GitHub Actions starts the script periodically. At the beginning
    of every run, all active trades are checked. If TP or SL is hit,
    exactly one Telegram result is sent and that trade is marked closed.
    """

    print("📡 TP/SL RESULT CHECK: ON")

    changed = False

    for symbol, coin_state in list(state.items()):

        if not isinstance(coin_state, dict):
            continue

        signal = coin_state.get("last_signal")

        if not isinstance(signal, dict):
            continue

        # Already finalized.
        if coin_state.get("trade_result") in ("TP", "SL"):
            continue

        direction = signal.get("direction")
        entry_raw = signal.get("entry")

        if direction not in ("LONG", "SHORT") or entry_raw is None:
            print(f"⚠️ {symbol}: invalid active signal state ignored.")
            continue

        try:
            entry = float(entry_raw)
        except (TypeError, ValueError):
            print(f"⚠️ {symbol}: invalid entry ignored.")
            continue

        pair = COINS.get(symbol)
        if not pair:
            continue

        try:
            price = get_current_price(pair)

            if direction == "LONG":
                tp = entry * (1 + TP_PERCENT / 100)
                sl = entry * (1 - SL_PERCENT / 100)

                if price >= tp:
                    result = "TP"
                    result_emoji = "✅"
                    result_title = "TAKE PROFIT HIT"
                    pnl_text = f"+{TP_PERCENT:.1f}%"

                elif price <= sl:
                    result = "SL"
                    result_emoji = "🛑"
                    result_title = "STOP LOSS HIT"
                    pnl_text = f"-{SL_PERCENT:.1f}%"

                else:
                    print(
                        f"📊 {symbol}: active LONG | "
                        f"price={price:.8f} | "
                        f"TP={tp:.8f} | SL={sl:.8f}"
                    )
                    continue

            else:
                tp = entry * (1 - TP_PERCENT / 100)
                sl = entry * (1 + SL_PERCENT / 100)

                if price <= tp:
                    result = "TP"
                    result_emoji = "✅"
                    result_title = "TAKE PROFIT HIT"
                    pnl_text = f"+{TP_PERCENT:.1f}%"

                elif price >= sl:
                    result = "SL"
                    result_emoji = "🛑"
                    result_title = "STOP LOSS HIT"
                    pnl_text = f"-{SL_PERCENT:.1f}%"

                else:
                    print(
                        f"📊 {symbol}: active SHORT | "
                        f"price={price:.8f} | "
                        f"TP={tp:.8f} | SL={sl:.8f}"
                    )
                    continue

            message = (
                f"{result_emoji} SCORE HUNTER RESULT\n\n"
                f"💰 {symbol}USDT\n"
                f"📊 {direction}\n"
                f"🏁 {result_title}\n"
                f"💵 Entry: {entry:.8f}\n"
                f"📍 Hit Price: {price:.8f}\n"
                f"📈 Result: {pnl_text}\n\n"
                "⏱ Timeframe: 4H / 1H\n"
                "🕯 Signal based on closed candle"
            )

            send_telegram(message)

            coin_state["trade_result"] = result
            coin_state["trade_result_price"] = price
            coin_state["trade_result_time"] = int(
                datetime.now(timezone.utc).timestamp()
            )

            # Keep the signal itself for history, but mark it closed.
            state[symbol] = coin_state
            changed = True

            print(
                f"🏁 {symbol}: {direction} {result} HIT "
                f"at {price:.8f}"
            )

        except Exception as e:
            print(
                f"⚠️ {symbol} TP/SL check error: "
                f"{type(e).__name__}: {e}"
            )

    if changed:
        save_state(state)


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

    # Kraken's final OHLC row may still be forming.
    if len(candles) > 1:
        candles = candles[:-1]

    return candles


def ema(values, period):
    if len(values) < period:
        return None

    value = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)

    for x in values[period:]:
        value = (x - value) * multiplier + value

    return value


def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            avg_gain * (period - 1) + gains[i]
        ) / period

        avg_loss = (
            avg_loss * (period - 1) + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - 100.0 / (1.0 + rs)


def atr(candles, period=14):
    if len(candles) <= period:
        return None

    trs = []

    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        previous_close = candles[i - 1]["close"]

        trs.append(
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        )

    return sum(trs[-period:]) / period


def robust_structure(candles, lookback=30):
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

    recent_high = max(highs[-lookback:])
    previous_high = max(
        highs[-2 * lookback:-lookback]
    )

    recent_low = min(lows[-lookback:])
    previous_low = min(
        lows[-2 * lookback:-lookback]
    )

    bullish = price > e200 and e20 > e50
    bearish = price < e200 and e20 < e50

    hh = recent_high > previous_high
    hl = recent_low > previous_low

    lh = recent_high < previous_high
    ll = recent_low < previous_low

    long_votes = (
        int(bullish)
        + int(hh)
        + int(hl)
    )

    short_votes = (
        int(bearish)
        + int(lh)
        + int(ll)
    )

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
    volume_average = sma(volumes, 20)
    average_true_range = atr(candles, 14)

    if None in (
        e20,
        e50,
        e200,
        rv,
        volume_average,
        average_true_range,
    ):
        return 0, {}

    if direction == "LONG":
        trend = (
            price > e200
            and e20 > e50
        )

        rsi_ok = 45 <= rv <= 72

    else:
        trend = (
            price < e200
            and e20 < e50
        )

        rsi_ok = 28 <= rv <= 55

    volume_ok = (
        volumes[-1] >= 1.05 * volume_average
    )

    volatility_ok = (
        average_true_range / price >= 0.0015
    )

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
    volume_average = sma(volumes, 20)
    average_true_range = atr(candles, 14)

    if None in (
        e20,
        e50,
        rv,
        volume_average,
        average_true_range,
    ):
        return 0, {}

    if direction == "LONG":
        trend = (
            price > e20
            and e20 > e50
        )
    else:
        trend = (
            price < e20
            and e20 < e50
        )

    recent_high = max(highs[-7:-1])
    recent_low = min(lows[-7:-1])

    if direction == "LONG":
        bos = price > recent_high
    else:
        bos = price < recent_low

    sweep_high = max(highs[-12:-1])
    sweep_low = min(lows[-12:-1])

    if direction == "LONG":
        sweep = (
            lows[-1] < sweep_low
            and price > sweep_low
        )
    else:
        sweep = (
            highs[-1] > sweep_high
            and price < sweep_high
        )

    if direction == "LONG":
        retest = (
            lows[-1] <= e20
            and price > e20
        )
    else:
        retest = (
            highs[-1] >= e20
            and price < e20
        )

    candle_range = highs[-1] - lows[-1]

    body_ratio = (
        abs(closes[-1] - opens[-1]) / candle_range
        if candle_range > 0
        else 0
    )

    if direction == "LONG":
        candle = (
            closes[-1] > opens[-1]
            and body_ratio >= 0.40
        )
    else:
        candle = (
            closes[-1] < opens[-1]
            and body_ratio >= 0.40
        )

    volume = (
        volumes[-1] >= 1.05 * volume_average
    )

    volatility = (
        average_true_range / price >= 0.001
    )

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
    if len(candles) < 35:
        return False, None

    target = (
        entry * (1 + TP_PERCENT / 100)
        if direction == "LONG"
        else entry * (1 - TP_PERCENT / 100)
    )

    buffer = TP_BUFFER_PERCENT / 100

    highs = [
        c["high"]
        for c in candles[-30:]
    ]

    lows = [
        c["low"]
        for c in candles[-30:]
    ]

    if direction == "LONG":
        blockers = [
            h for h in highs
            if h > entry and h < target
        ]

        nearest = (
            min(blockers)
            if blockers
            else None
        )

        if (
            nearest is not None
            and nearest >= target * (1 - buffer)
        ):
            return False, nearest

    else:
        blockers = [
            l for l in lows
            if l < entry and l > target
        ]

        nearest = (
            max(blockers)
            if blockers
            else None
        )

        if (
            nearest is not None
            and nearest <= target * (1 + buffer)
        ):
            return False, nearest

    return True, None


def create_message(
    symbol,
    direction,
    entry,
    score4,
    score1,
):
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
    print(
        f"⭐ Minimum 1H confirmation: "
        f"{MIN_1H_CONFIRMATION}/6"
    )
    print("🕯 CLOSED CANDLE ONLY - NO MID-CANDLE SIGNAL")
    print("♾️ DAILY SIGNAL LIMIT: DISABLED")
    print(
        "📊 Coins: "
        "ETH / SOL / XRP / BTC / ADA / LINK / DOGE"
    )
    print(
        "⏱ Main structure: 4H | "
        "Entry confirmation: 1H"
    )
    print(
        f"🎯 TP: {TP_PERCENT:.1f}% | "
        f"🛑 SL: {SL_PERCENT:.1f}%"
    )
    print(
        f"📐 TP SPACE FILTER: ON | "
        f"lookback=30 | buffer={TP_BUFFER_PERCENT:.1f}%"
    )

    state = load_state()

    # IMPORTANT:
    # This checks active trades automatically at the beginning
    # of every GitHub Actions run. It does not require manual
    # checking of every coin.
    monitor_open_trades(state)

    for symbol, pair in COINS.items():
        print(f"\n========== {symbol} ==========")

        try:
            c4 = get_ohlc(pair, INTERVAL_4H)
            c1 = get_ohlc(pair, INTERVAL_1H)

            if len(c4) < 220 or len(c1) < 80:
                print(f"{symbol}: insufficient data")
                continue

            candle_id = c1[-1]["time"]

            print(
                f"{symbol} latest CLOSED 1H: "
                f"{candle_id}"
            )

            coin_state = state.get(symbol, {})

            if not isinstance(coin_state, dict):
                coin_state = {}

            # Only scan a coin once for each newly closed 1H candle.
            if (
                coin_state.get("last_checked_1h")
                == candle_id
            ):
                print(
                    f"{symbol}: No new 1H candle."
                )
                continue

            coin_state["last_checked_1h"] = candle_id

            structure = robust_structure(c4)

            print(
                f"{symbol} 4H Structure: "
                f"{structure}"
            )

            if structure not in ("LONG", "SHORT"):
                print(
                    f"{symbol}: No valid 4H structure."
                )
                state[symbol] = coin_state
                save_state(state)
                continue

            direction = structure

            q4, details4 = (
                structure_quality_4h(
                    c4,
                    direction,
                )
            )

            print(
                f"{symbol} {direction} 4H quality: "
                f"{q4}/4 | "
                f"trend={int(details4.get('trend', False))} "
                f"rsi={details4.get('rsi', 0):.2f} "
                f"volume={int(details4.get('volume', False))} "
                f"volatility={int(details4.get('volatility', False))}"
            )

            if q4 < MIN_4H_QUALITY:
                print(
                    f"{symbol}: rejected - "
                    f"4H quality"
                )
                state[symbol] = coin_state
                save_state(state)
                continue

            q1, details1 = (
                one_hour_confirmation(
                    c1,
                    direction,
                )
            )

            print(
                f"{symbol} {direction} "
                f"1H confirmation: {q1}/6 | "
                f"trend={int(details1.get('trend', False))} "
                f"bos={int(details1.get('bos', False))} "
                f"sweep={int(details1.get('sweep', False))} "
                f"retest={int(details1.get('retest', False))} "
                f"candle={int(details1.get('candle', False))} "
                f"volume={int(details1.get('volume', False))}"
            )

            if q1 < MIN_1H_CONFIRMATION:
                print(
                    f"{symbol}: rejected - "
                    f"1H confirmation"
                )
                state[symbol] = coin_state
                save_state(state)
                continue

            entry = c1[-1]["close"]

            space_ok, blocker = (
                tp_space_ok(
                    c4,
                    direction,
                    entry,
                )
            )

            if not space_ok:
                print(
                    f"{symbol}: rejected - "
                    f"TP space blocked by "
                    f"nearby level {blocker}"
                )
                state[symbol] = coin_state
                save_state(state)
                continue

            signal_id = (
                f"{candle_id}_{direction}"
            )

            if (
                coin_state.get("last_signal_id")
                == signal_id
            ):
                print(
                    f"{symbol}: duplicate signal"
                )
                state[symbol] = coin_state
                save_state(state)
                continue

            # Do not issue another signal while a previous
            # signal for this coin is still active.
            previous_signal = (
                coin_state.get("last_signal")
            )

            previous_result = (
                coin_state.get("trade_result")
            )

            if (
                isinstance(previous_signal, dict)
                and previous_result not in ("TP", "SL")
            ):
                print(
                    f"{symbol}: active previous "
                    f"trade exists - no new signal."
                )
                state[symbol] = coin_state
                save_state(state)
                continue

            message = create_message(
                symbol,
                direction,
                entry,
                q4,
                q1,
            )

            send_telegram(message)

            # New active trade.
            coin_state = {
                "last_checked_1h": candle_id,
                "last_signal_id": signal_id,
                "last_signal": {
                    "direction": direction,
                    "score_4h": q4,
                    "score_1h": q1,
                    "entry": entry,
                    "candle_1h": candle_id,
                },
                "trade_result": None,
                "trade_result_price": None,
                "trade_result_time": None,
            }

            state[symbol] = coin_state
            save_state(state)

            print(
                f"🚨 {symbol}: "
                f"{direction} SIGNAL SENT"
            )

        except Exception as e:
            print(
                f"❌ {symbol} ERROR: "
                f"{type(e).__name__}: {e}"
            )
            continue

    save_state(state)

    print(
        "\n✅ SCORE HUNTER PRO v3 "
        "SCAN COMPLETED"
    )


if __name__ == "__main__":
    main()
