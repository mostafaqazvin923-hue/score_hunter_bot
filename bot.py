import os
import json
import time
import requests

# ============================================================
# SCORE HUNTER 4H - LBANK
# BTC / ETH / SOL / XRP
#
# 7 FACTORS:
# 1 Trend
# 2 RSI
# 3 Volume
# 4 Breakout
# 5 Pullback
# 6 Candle confirmation
# 7 Volatility
#
# NEW:
# - Strong signal filter
# - Entry window
# - Signal expiration
# - ATR + market structure SL
# - Risk/Reward TP
# - Trade result tracking
# - Duplicate protection
# ============================================================

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "2090120004")

LBANK_KLINE_URL = "https://api.lbkex.com/v2/kline.do"
LBANK_TICKER_URL = "https://api.lbkex.com/v2/ticker.do"

COINS = {
    "BTC": "btc_usdt",
    "ETH": "eth_usdt",
    "SOL": "sol_usdt",
    "XRP": "xrp_usdt",
}

TIMEFRAME = "hour4"
CANDLE_LIMIT = 250

# Minimum score
REQUIRED_SCORE = 5

# Entry must be close enough to the signal price
ENTRY_MAX_DISTANCE = 0.003
# 0.003 = 0.30%

# ATR stop-loss
ATR_MULTIPLIER = 1.2

# Minimum distance from entry to structural stop
MIN_SL_PERCENT = 0.004
# 0.4%

# Maximum allowed SL distance
MAX_SL_PERCENT = 0.025
# 2.5%

# Risk / Reward
RISK_REWARD = 1.8

# State file
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

    if response.status_code != 200:
        print(response.text)

    return response.ok


# ============================================================
# STATE
# ============================================================

def load_state():

    if not os.path.exists(STATE_FILE):
        return {
            "last_signals": {},
            "active_trades": {},
            "completed_trades": [],
        }

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            state = json.load(f)

        state.setdefault("last_signals", {})
        state.setdefault("active_trades", {})
        state.setdefault("completed_trades", [])

        return state

    except Exception as e:

        print("State load error:", e)

        return {
            "last_signals": {},
            "active_trades": {},
            "completed_trades": [],
        }


def save_state(state):

    temp_file = STATE_FILE + ".tmp"

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
# LBANK KLINE
# ============================================================

def get_4h_candles(symbol):

    print(
        f"Getting {symbol} 4H candles from LBank..."
    )

    now = int(time.time())

    start_time = (
        now
        - (
            CANDLE_LIMIT
            * 4
            * 60
            * 60
        )
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
            f"Could not find current price: {result}"
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

    # Ignore the currently forming candle.
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
# SIGNAL CALCULATION
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
        20,
    )

    ema50 = ema_series(
        closes,
        50,
    )

    ema200 = ema_series(
        closes,
        200,
    )

    rsi_values = rsi_series(
        closes,
        14,
    )

    atr_values = atr_series(
        candles,
        14,
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
    # 4. BREAKOUT
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
    # 5. PULLBACK
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

    candle_range = high - low

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
        and bull_ratio >= 0.40
    )

    bear_candle = (
        close < open_price
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
    # STRONG SIGNAL FILTER
    #
    # Trend + Pullback + Candle are mandatory.
    # ========================================================

    if (
        long_score >= REQUIRED_SCORE
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
        }

    if (
        short_score >= REQUIRED_SCORE
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
        }

    return None


# ============================================================
# SIGNAL STRENGTH
# ============================================================

def signal_strength(score):

    if score >= 7:
        return "🔥 VERY STRONG"

    if score >= 6:
        return "💪 STRONG"

    return "🟡 NORMAL"


# ============================================================
# CALCULATE SL / TP
# ============================================================

def calculate_risk_levels(signal):

    direction = signal["direction"]
    entry = signal["entry"]
    atr = signal["atr"]

    structure_low = signal["structure_low"]
    structure_high = signal["structure_high"]

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

        # Use the wider protective stop
        sl = min(
            atr_sl,
            structure_sl,
        )

        risk = entry - sl

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

        sl = entry - risk

        tp = (
            entry
            + (
                risk
                * RISK_REWARD
            )
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

        risk = sl - entry

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

        sl = entry + risk

        tp = (
            entry
            - (
                risk
                * RISK_REWARD
            )
        )

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_percent": (
            abs(entry - sl)
            / entry
            * 100
        ),
        "rr": RISK_REWARD,
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

        symbol = trade["symbol"]
        direction = trade["direction"]
        entry = trade["entry"]
        tp = trade["tp"]
        sl = trade["sl"]

        try:

            current_price = get_current_price(
                trade["lbank_symbol"]
            )

        except Exception as e:

            print(
                f"{symbol}: "
                f"price check error: {e}"
            )

            continue

        result = None

        if direction == "LONG":

            if current_price >= tp:
                result = "TP"

            elif current_price <= sl:
                result = "SL"

        else:

            if current_price <= tp:
                result = "TP"

            elif current_price >= sl:
                result = "SL"

        if result is None:
            continue

        if result == "TP":

            pnl_percent = (
                abs(tp - entry)
                / entry
                * 100
            )

        else:

            pnl_percent = -(
                abs(sl - entry)
                / entry
                * 100
            )

        trade["result"] = result
        trade["exit_price"] = current_price
        trade["pnl_percent"] = pnl_percent
        trade["closed_at"] = int(
            time.time()
        )

        state.setdefault(
            "completed_trades",
            [],
        ).append(trade)

        finished.append(
            (
                trade_id,
                trade,
            )
        )

        emoji = (
            "✅"
            if result == "TP"
            else "❌"
        )

        message = (
            f"{emoji} SCORE HUNTER "
            f"TRADE CLOSED\n\n"
            f"💰 {symbol}USDT\n"
            f"📊 {direction}\n"
            f"⭐ Score: "
            f"{trade['score']}/7\n"
            f"📌 Result: {result}\n"
            f"💵 Entry: {entry:.8f}\n"
            f"🏁 Exit: "
            f"{current_price:.8f}\n"
            f"📈 P/L: "
            f"{pnl_percent:+.2f}%\n\n"
            f"⏱ Timeframe: 4H\n"
            f"🏦 Data: LBank"
        )

        send_telegram(message)

    for trade_id, _ in finished:

        del active[trade_id]

    state["active_trades"] = active


# ============================================================
# STATISTICS
# ============================================================

def get_statistics(state):

    trades = state.get(
        "completed_trades",
        [],
    )

    if not trades:
        return None

    total = len(trades)

    wins = sum(
        1
        for t in trades
        if t.get("result") == "TP"
    )

    losses = sum(
        1
        for t in trades
        if t.get("result") == "SL"
    )

    decided = wins + losses

    win_rate = (
        wins / decided * 100
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
        "win_rate": win_rate,
        "pnl": total_pnl,
    }


# ============================================================
# SEND STATISTICS
# ============================================================

def send_statistics_if_needed(
    state,
):

    stats = get_statistics(state)

    if not stats:
        return

    # Only report every 10 completed trades
    if stats["total"] % 10 != 0:
        return

    message = (
        "📊 SCORE HUNTER STATISTICS\n\n"
        f"📌 Closed trades: "
        f"{stats['total']}\n"
        f"✅ TP: {stats['wins']}\n"
        f"❌ SL: {stats['losses']}\n"
        f"🎯 Win rate: "
        f"{stats['win_rate']:.1f}%\n"
        f"📈 Total P/L: "
        f"{stats['pnl']:+.2f}%\n\n"
        "⏱ Timeframe: 4H\n"
        "🏦 Data: LBank"
    )

    send_telegram(message)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🟢 SCORE HUNTER 4H - LBANK"
    )

    print(
        "🛡 Advanced filters enabled"
    )

    print(
        f"Entry window: "
        f"{ENTRY_MAX_DISTANCE * 100:.2f}%"
    )

    print(
        f"ATR SL multiplier: "
        f"{ATR_MULTIPLIER}"
    )

    print(
        f"Risk/Reward: "
        f"1:{RISK_REWARD}"
    )

    state = load_state()

    # --------------------------------------------------------
    # First check existing active trades
    # --------------------------------------------------------

    check_active_trades(
        state
    )

    # --------------------------------------------------------
    # Scan new signals
    # --------------------------------------------------------

    for symbol, lbank_symbol in COINS.items():

        print(
            f"\n========== {symbol} =========="
        )

        try:

            candles = get_4h_candles(
                lbank_symbol
            )

            closed_candles = (
                get_closed_candles(
                    candles
                )
            )

            latest = (
                closed_candles[-1]
            )

            print(
                f"{symbol}: latest closed "
                f"4H candle: "
                f"{latest['time']}"
            )

            signal = calculate_signal(
                closed_candles
            )

            if signal is None:

                print(
                    f"{symbol}: "
                    f"no valid signal"
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

            strength = signal_strength(
                score
            )

            # ------------------------------------------------
            # Current market price
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
                f"signal={direction} "
                f"score={score}/7 "
                f"entry={entry} "
                f"current={current_price} "
                f"distance={distance:.3f}%"
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
                    f"signal expired - "
                    f"price too far from entry"
                )

                continue

            # ------------------------------------------------
            # Duplicate protection
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
                    {}
                )
                .get(symbol)
            )

            if previous_id == signal_id:

                print(
                    f"{symbol}: "
                    f"signal already sent"
                )

                continue

            # ------------------------------------------------
            # Do not open another trade
            # while one is active
            # ------------------------------------------------

            existing_trade = None

            for trade in (
                state
                .get(
                    "active_trades",
                    {}
                )
                .values()
            ):

                if trade["symbol"] == symbol:

                    existing_trade = trade
                    break

            if existing_trade:

                print(
                    f"{symbol}: "
                    f"active trade exists"
                )

                continue

            # ------------------------------------------------
            # Risk levels
            # ------------------------------------------------

            levels = (
                calculate_risk_levels(
                    signal
                )
            )

            sl = levels["sl"]
            tp = levels["tp"]
            risk_percent = (
                levels["risk_percent"]
            )

            # ------------------------------------------------
            # Trade ID
            # ------------------------------------------------

            trade_id = signal_id

            # ------------------------------------------------
            # Save active trade
            # ------------------------------------------------

            trade = {
                "trade_id": trade_id,
                "symbol": symbol,
                "lbank_symbol": lbank_symbol,
                "direction": direction,
                "score": score,
                "strength": strength,
                "entry": entry,
                "current_price_at_signal": current_price,
                "tp": tp,
                "sl": sl,
                "risk_percent": risk_percent,
                "rr": RISK_REWARD,
                "candle_time": candle_time,
                "created_at": int(
                    time.time()
                ),
                "status": "ACTIVE",
            }

            state.setdefault(
                "active_trades",
                {},
            )[trade_id] = trade

            state.setdefault(
                "last_signals",
                {},
            )[symbol] = signal_id

            save_state(state)

            # ------------------------------------------------
            # Telegram
            # ------------------------------------------------

            if direction == "LONG":

                direction_text = (
                    "🟢 LONG"
                )

                tp_percent = (
                    (tp - entry)
                    / entry
                    * 100
                )

                sl_percent = (
                    (sl - entry)
                    / entry
                    * 100
                )

            else:

                direction_text = (
                    "🔴 SHORT"
                )

                tp_percent = (
                    (entry - tp)
                    / entry
                    * 100
                )

                sl_percent = (
                    (entry - sl)
                    / entry
                    * 100
                )

            message = (
                "🚨 SCORE HUNTER 4H 🚨\n\n"
                f"💰 {symbol}USDT\n"
                f"📊 {direction_text}\n"
                f"{strength}\n"
                f"⭐ Score: {score}/7\n\n"
                f"💵 Entry: "
                f"{entry:.8f}\n"
                f"📍 Current: "
                f"{current_price:.8f}\n"
                f"📏 Distance: "
                f"{distance:.2f}%\n\n"
                f"🎯 TP: "
                f"{tp:.8f} "
                f"(+{tp_percent:.2f}% "
                f"target)\n"
                f"🛑 SL: "
                f"{sl:.8f} "
                f"(-{sl_percent:.2f}% "
                f"risk)\n\n"
                f"⚖️ R:R: "
                f"1:{RISK_REWARD}\n"
                f"📐 Risk: "
                f"{risk_percent:.2f}%\n\n"
                "⏱ Timeframe: 4H\n"
                "🏦 Data: LBank\n"
                "🔒 Entry window: ±0.30%\n"
                "⚠️ Manage risk."
            )

            sent = send_telegram(
                message
            )

            if sent:

                print(
                    f"{symbol}: "
                    f"{direction} "
                    f"Score {score}/7 "
                    f"SENT"
                )

            else:

                print(
                    f"{symbol}: "
                    f"Telegram failed"
                )

        except Exception as e:

            print(
                f"{symbol}: ERROR: {e}"
            )

    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    save_state(state)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    send_statistics_if_needed(
        state
    )

    print(
        "\n✅ LBank 4H scan completed."
    )


if __name__ == "__main__":
    main()
