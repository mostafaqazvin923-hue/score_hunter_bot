import requests
from datetime import datetime, timezone

# ============================================================
# SCORE HUNTER PRO v8.1
# LONG ONLY BACKTEST
#
# 4H TREND + 1H ENTRY
# LONG ONLY
# BREAKOUT + REVERSAL ONLY
# NO PULLBACK
# CLOSED CANDLE ONLY
# NO LOOK-AHEAD
# ADX >= 20
# RSI CONFIRMATION
# SL = 1.5 ATR / STRUCTURE
# MAX SL = 3.5 ATR
# TP = 2R
# ENTRY CANDLE EXCLUDED FROM TP/SL
# TP + SL SAME CANDLE = SL
# NO OVERLAPPING POSITIONS
# ============================================================

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"

INTERVAL_4H = 240
INTERVAL_1H = 60

SECONDS_1H = 3600
SECONDS_4H = 14400

COINS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BTC": "XBTUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT",
}

EMA20 = 20
EMA50 = 50
EMA200 = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

ADX_MIN = 20.0

STRUCTURE_LOOKBACK = 5
REVERSAL_LOOKBACK = 6

SL_ATR = 1.50
STRUCTURE_BUFFER_ATR = 0.10
MAX_SL_ATR = 3.50

TP_R_MULTIPLE = 2.0
MIN_RR = 1.50

# Strong breakout requirements
MIN_BODY_RATIO = 0.55
MIN_CLOSE_LOCATION = 0.70

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ScoreHunterPro-v8.1-Backtester/1.0"
})


# ============================================================
# DATA
# ============================================================

def get_ohlc(symbol, interval):

    response = SESSION.get(
        KRAKEN_URL,
        params={
            "pair": COINS[symbol],
            "interval": interval
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"{symbol} Kraken error: {data['error']}"
        )

    result = data.get("result", {})

    keys = [
        k for k in result
        if k != "last"
    ]

    if not keys:
        raise RuntimeError(
            f"{symbol}: no OHLC data."
        )

    pair = keys[0]

    candles = []

    for row in result[pair]:

        candles.append({
            "time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6])
        })

    candles.sort(
        key=lambda x: x["time"]
    )

    unique = {}

    for candle in candles:
        unique[candle["time"]] = candle

    candles = list(unique.values())

    candles.sort(
        key=lambda x: x["time"]
    )

    # Remove currently forming candle.
    if len(candles) > 1:
        candles = candles[:-1]

    return candles


# ============================================================
# EMA
# ============================================================

def ema(values, period):

    if len(values) < period:
        return None

    value = sum(
        values[:period]
    ) / period

    multiplier = 2.0 / (period + 1)

    for price in values[period:]:
        value = (
            (price - value)
            * multiplier
            + value
        )

    return value


# ============================================================
# ATR
# ============================================================

def atr(candles, period=14):

    if len(candles) < period + 1:
        return None

    trs = []

    for i in range(1, len(candles)):

        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current["high"] - current["low"],
            abs(
                current["high"]
                - previous["close"]
            ),
            abs(
                current["low"]
                - previous["close"]
            )
        )

        trs.append(tr)

    if len(trs) < period:
        return None

    value = sum(
        trs[:period]
    ) / period

    for tr in trs[period:]:

        value = (
            value * (period - 1)
            + tr
        ) / period

    return value


# ============================================================
# RSI
# ============================================================

def rsi(candles, period=14):

    if len(candles) < period + 1:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    gains = []
    losses = []

    for i in range(1, len(closes)):

        change = (
            closes[i]
            - closes[i - 1]
        )

        gains.append(
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
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
                avg_gain * (period - 1)
                + gains[i]
            ) / period
        )

        avg_loss = (
            (
                avg_loss * (period - 1)
                + losses[i]
            ) / period
        )

    if avg_loss == 0:
        return 100.0

    if avg_gain == 0:
        return 0.0

    rs_value = (
        avg_gain
        / avg_loss
    )

    return (
        100.0
        - (
            100.0
            / (1.0 + rs_value)
        )
    )


# ============================================================
# ADX
# ============================================================

def adx(candles, period=14):

    if len(candles) < period * 2 + 5:
        return None

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(candles)):

        current = candles[i]
        previous = candles[i - 1]

        high = current["high"]
        low = current["low"]

        prev_high = previous["high"]
        prev_low = previous["low"]
        prev_close = previous["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = (
            up_move
            if (
                up_move > down_move
                and up_move > 0
            )
            else 0.0
        )

        minus_dm = (
            down_move
            if (
                down_move > up_move
                and down_move > 0
            )
            else 0.0
        )

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period * 2:
        return None

    atr_value = (
        sum(tr_list[:period])
        / period
    )

    plus_value = (
        sum(plus_dm_list[:period])
        / period
    )

    minus_value = (
        sum(minus_dm_list[:period])
        / period
    )

    dx_values = []

    for i in range(
        period,
        len(tr_list)
    ):

        atr_value = (
            (
                atr_value * (period - 1)
                + tr_list[i]
            ) / period
        )

        plus_value = (
            (
                plus_value * (period - 1)
                + plus_dm_list[i]
            ) / period
        )

        minus_value = (
            (
                minus_value * (period - 1)
                + minus_dm_list[i]
            ) / period
        )

        if atr_value <= 0:
            continue

        plus_di = (
            100.0
            * plus_value
            / atr_value
        )

        minus_di = (
            100.0
            * minus_value
            / atr_value
        )

        denominator = (
            plus_di
            + minus_di
        )

        if denominator <= 0:
            dx = 0.0
        else:
            dx = (
                100.0
                * abs(
                    plus_di
                    - minus_di
                )
                / denominator
            )

        dx_values.append(dx)

    if len(dx_values) < period:
        return None

    adx_value = (
        sum(dx_values[:period])
        / period
    )

    for value in dx_values[period:]:

        adx_value = (
            (
                adx_value * (period - 1)
                + value
            ) / period
        )

    return adx_value


# ============================================================
# 4H TREND
# ============================================================

def get_4h_direction(candles):

    if len(candles) < EMA200:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    close = closes[-1]

    e20 = ema(closes, EMA20)
    e50 = ema(closes, EMA50)
    e200 = ema(closes, EMA200)

    if (
        e20 is None
        or e50 is None
        or e200 is None
    ):
        return None

    if (
        close > e200
        and e20 > e50
        and e50 > e200
    ):
        return "LONG"

    if (
        close < e200
        and e20 < e50
        and e50 < e200
    ):
        return "SHORT"

    return None


# ============================================================
# CLOSED 4H DATA AVAILABLE AT 1H ENTRY
# ============================================================

def get_closed_4h_for_entry(
    candles_4h,
    entry_candle
):

    entry_close_time = (
        entry_candle["time"]
        + SECONDS_1H
    )

    return [
        candle
        for candle in candles_4h
        if (
            candle["time"]
            + SECONDS_4H
            <= entry_close_time
        )
    ]


# ============================================================
# BREAKOUT
# ============================================================

def detect_breakout_long(candles):

    if len(candles) < STRUCTURE_LOOKBACK + 2:
        return False, None

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    resistance = max(
        c["high"]
        for c in previous
    )

    # Must close above structure.
    if current["close"] <= resistance:
        return False, None

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return False, None

    body = abs(
        current["close"]
        - current["open"]
    )

    body_ratio = (
        body
        / candle_range
    )

    close_location = (
        current["close"]
        - current["low"]
    ) / candle_range

    # Strong bullish candle.
    if current["close"] <= current["open"]:
        return False, None

    if body_ratio < MIN_BODY_RATIO:
        return False, None

    if close_location < MIN_CLOSE_LOCATION:
        return False, None

    # Momentum confirmation.
    if len(candles) >= 2:
        if (
            current["close"]
            <= candles[-2]["close"]
        ):
            return False, None

    return True, resistance


# ============================================================
# LONG REVERSAL
#
# This allows a LONG after a previous 4H SHORT trend,
# but only after a confirmed bullish structure transition.
# ============================================================

def detect_reversal_long(
    candles,
    trend_direction,
    adx_value,
    rsi_value
):

    if trend_direction != "SHORT":
        return None

    required = max(
        60,
        REVERSAL_LOOKBACK + 12
    )

    if len(candles) < required:
        return None

    current = candles[-1]

    previous = candles[
        -REVERSAL_LOOKBACK - 1:-1
    ]

    resistance = max(
        c["high"]
        for c in previous
    )

    if current["close"] <= resistance:
        return None

    # Current candle bullish.
    if current["close"] <= current["open"]:
        return None

    # Body strength.
    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range <= 0:
        return None

    body_ratio = (
        abs(
            current["close"]
            - current["open"]
        )
        / candle_range
    )

    if body_ratio < MIN_BODY_RATIO:
        return None

    close_location = (
        current["close"]
        - current["low"]
    ) / candle_range

    if close_location < MIN_CLOSE_LOCATION:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    e20 = ema(closes, EMA20)
    e50 = ema(closes, EMA50)

    if e20 is None or e50 is None:
        return None

    # EMA transition.
    if e20 <= e50:
        return None

    if current["close"] <= e50:
        return None

    if adx_value < ADX_MIN:
        return None

    if rsi_value < 50:
        return None

    # Previous candle should also support bullish transition.
    if len(candles) >= 2:

        previous_candle = candles[-2]

        if (
            previous_candle["close"]
            <= previous_candle["open"]
        ):
            return None

    return {
        "structure_level": resistance
    }


# ============================================================
# LONG EMA ALIGNMENT
# ============================================================

def ema_alignment_long(candles):

    if len(candles) < EMA200:
        return False

    closes = [
        c["close"]
        for c in candles
    ]

    e20 = ema(closes, EMA20)
    e50 = ema(closes, EMA50)
    e200 = ema(closes, EMA200)

    if (
        e20 is None
        or e50 is None
        or e200 is None
    ):
        return False

    current_close = closes[-1]

    return (
        current_close > e200
        and e20 > e50
        and e50 > e200
    )


# ============================================================
# RSI LONG CONFIRMATION
# ============================================================

def rsi_confirmation_long(rsi_value):

    if rsi_value is None:
        return False

    # RSI > 70 is NOT automatically rejected.
    return (
        50 <= rsi_value <= 85
    )


# ============================================================
# LONG STOP / TARGET
# ============================================================

def calculate_long_levels(
    candles,
    entry,
    atr_value,
    setup
):

    if atr_value is None or atr_value <= 0:
        return None

    recent = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    if not recent:
        return None

    recent_low = min(
        c["low"]
        for c in recent
    )

    atr_stop = (
        entry
        - atr_value * SL_ATR
    )

    structure_stop = (
        recent_low
        - atr_value
        * STRUCTURE_BUFFER_ATR
    )

    # Use the safer / wider stop.
    sl = min(
        atr_stop,
        structure_stop
    )

    risk = (
        entry - sl
    )

    if risk <= 0:
        return None

    risk_atr = (
        risk
        / atr_value
    )

    if risk_atr > MAX_SL_ATR:
        return None

    tp = (
        entry
        + risk * TP_R_MULTIPLE
    )

    reward = (
        tp - entry
    )

    if reward <= 0:
        return None

    rr = (
        reward
        / risk
    )

    if rr < MIN_RR:
        return None

    return {
        "tp": tp,
        "sl": sl,
        "risk": risk,
        "rr": rr,
        "risk_atr": risk_atr
    }


# ============================================================
# SIGNAL
# ============================================================

def analyze_at_index(
    candles_4h,
    candles_1h
):

    if len(candles_1h) < EMA200 + 10:
        return None

    if len(candles_4h) < EMA200:
        return None

    current = candles_1h[-1]

    entry = current["close"]

    trend_direction = get_4h_direction(
        candles_4h
    )

    if trend_direction is None:
        return None

    atr_value = atr(
        candles_1h,
        ATR_PERIOD
    )

    rsi_value = rsi(
        candles_1h,
        RSI_PERIOD
    )

    adx_value = adx(
        candles_1h,
        ADX_PERIOD
    )

    if (
        atr_value is None
        or rsi_value is None
        or adx_value is None
    ):
        return None

    if adx_value < ADX_MIN:
        return None

    if not rsi_confirmation_long(
        rsi_value
    ):
        return None

    # --------------------------------------------------------
    # REVERSAL
    # --------------------------------------------------------

    reversal = detect_reversal_long(
        candles_1h,
        trend_direction,
        adx_value,
        rsi_value
    )

    if reversal is not None:

        setup = "REVERSAL"

    else:

        # ----------------------------------------------------
        # NORMAL LONG
        #
        # Must already have 4H LONG trend.
        # ----------------------------------------------------

        if trend_direction != "LONG":
            return None

        if not ema_alignment_long(
            candles_1h
        ):
            return None

        breakout, level = detect_breakout_long(
            candles_1h
        )

        if not breakout:
            return None

        setup = "BREAKOUT"

    levels = calculate_long_levels(
        candles_1h,
        entry,
        atr_value,
        setup
    )

    if levels is None:
        return None

    return {
        "symbol": None,
        "direction": "LONG",
        "setup": setup,
        "entry_time": current["time"],
        "entry": entry,
        "tp": levels["tp"],
        "sl": levels["sl"],
        "risk": levels["risk"],
        "rr": levels["rr"],
        "risk_atr": levels["risk_atr"],
        "atr": atr_value,
        "adx": adx_value,
        "rsi": rsi_value
    }


# ============================================================
# RESULT CHECK
# ============================================================

def check_trade_result(
    candles,
    entry_index,
    signal
):

    tp = signal["tp"]
    sl = signal["sl"]

    # IMPORTANT:
    # Start from NEXT candle.
    for i in range(
        entry_index + 1,
        len(candles)
    ):

        candle = candles[i]

        hit_tp = (
            candle["high"]
            >= tp
        )

        hit_sl = (
            candle["low"]
            <= sl
        )

        if hit_tp and hit_sl:
            # Conservative assumption.
            return "SL", i

        if hit_sl:
            return "SL", i

        if hit_tp:
            return "TP", i

    return None, None


# ============================================================
# BACKTEST
# ============================================================

def backtest_coin(
    symbol,
    candles_4h,
    candles_1h
):

    trades = []

    i = EMA200 + 10

    while i < len(candles_1h):

        entry_candle = candles_1h[i]

        usable_4h = get_closed_4h_for_entry(
            candles_4h,
            entry_candle
        )

        if len(usable_4h) < EMA200:
            i += 1
            continue

        signal = analyze_at_index(
            usable_4h,
            candles_1h[:i + 1]
        )

        if signal is None:
            i += 1
            continue

        signal["symbol"] = symbol

        result, exit_index = (
            check_trade_result(
                candles_1h,
                i,
                signal
            )
        )

        # Still open at end.
        if result is None:
            break

        exit_candle = candles_1h[
            exit_index
        ]

        if result == "TP":
            r_result = TP_R_MULTIPLE
        else:
            r_result = -1.0

        trades.append({
            "symbol":
                symbol,

            "setup":
                signal["setup"],

            "direction":
                "LONG",

            "entry_time":
                signal["entry_time"],

            "entry":
                signal["entry"],

            "tp":
                signal["tp"],

            "sl":
                signal["sl"],

            "exit_time":
                exit_candle["time"],

            "result":
                result,

            "R":
                r_result,

            "rr":
                signal["rr"],

            "risk_atr":
                signal["risk_atr"],

            "atr":
                signal["atr"],

            "adx":
                signal["adx"],

            "rsi":
                signal["rsi"]
        })

        # No overlapping trades.
        i = exit_index + 1

    return trades


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(trades):

    if not trades:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_r": 0.0,
            "average_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0
        }

    total = len(trades)

    wins = sum(
        t["result"] == "TP"
        for t in trades
    )

    losses = sum(
        t["result"] == "SL"
        for t in trades
    )

    win_rate = (
        wins
        / total
        * 100
    )

    net_r = sum(
        t["R"]
        for t in trades
    )

    gross_profit = sum(
        t["R"]
        for t in trades
        if t["R"] > 0
    )

    gross_loss = abs(
        sum(
            t["R"]
            for t in trades
            if t["R"] < 0
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = float("inf")

    average_r = (
        net_r
        / total
    )

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    current_win = 0
    current_loss = 0

    max_win = 0
    max_loss = 0

    for trade in trades:

        equity += trade["R"]

        peak = max(
            peak,
            equity
        )

        drawdown = (
            peak
            - equity
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

        if trade["result"] == "TP":

            current_win += 1
            current_loss = 0

            max_win = max(
                max_win,
                current_win
            )

        else:

            current_loss += 1
            current_win = 0

            max_loss = max(
                max_loss,
                current_loss
            )

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_r": net_r,
        "average_r": average_r,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_win_streak": max_win,
        "max_loss_streak": max_loss
    }


# ============================================================
# TIME / PRICE
# ============================================================

def utc_time(timestamp):

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M"
    )


def fmt_price(price):

    if price >= 1000:
        return f"{price:.2f}"

    if price >= 1:
        return f"{price:.5f}"

    return f"{price:.8f}"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(" SCORE HUNTER PRO v8.1")
    print(" LONG ONLY - BREAKOUT + REVERSAL")
    print("=" * 110)

    print()
    print("Rules:")
    print("4H Trend + 1H Entry")
    print("LONG ONLY")
    print("BREAKOUT + REVERSAL ONLY")
    print("NO PULLBACK")
    print("Closed 4H only")
    print("Closed 1H only")
    print("NO LOOK-AHEAD")
    print("ADX >= 20")
    print("RSI confirmation")
    print("Strong breakout candle")
    print("SL = 1.5 ATR / Structure")
    print("Maximum SL = 3.5 ATR")
    print("TP = 2R")
    print("Entry candle excluded from TP/SL")
    print("TP + SL same candle = SL")
    print("No overlapping positions")

    all_trades = []

    data_ranges = {}

    for symbol in COINS:

        print()
        print(f"Downloading {symbol}...")

        candles_4h = get_ohlc(
            symbol,
            INTERVAL_4H
        )

        candles_1h = get_ohlc(
            symbol,
            INTERVAL_1H
        )

        print(
            f"{symbol}: "
            f"4H={len(candles_4h)} | "
            f"1H={len(candles_1h)}"
        )

        if candles_4h:
            print(
                f"{symbol}: 4H range = "
                f"{utc_time(candles_4h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_4h[-1]['time'])}"
            )

        if candles_1h:
            print(
                f"{symbol}: 1H range = "
                f"{utc_time(candles_1h[0]['time'])}"
                f" -> "
                f"{utc_time(candles_1h[-1]['time'])}"
            )

        data_ranges[symbol] = (
            candles_4h,
            candles_1h
        )

        trades = backtest_coin(
            symbol,
            candles_4h,
            candles_1h
        )

        print(
            f"{symbol}: "
            f"{len(trades)} completed trades"
        )

        all_trades.extend(
            trades
        )

    # ========================================================
    # OVERALL
    # ========================================================

    stats = calculate_stats(
        all_trades
    )

    print()
    print("=" * 110)
    print("SCORE HUNTER PRO v8.1")
    print("LONG ONLY - BREAKOUT + REVERSAL")
    print("=" * 110)

    print(
        f"Total Trades       : "
        f"{stats['total']}"
    )

    print(
        f"Wins               : "
        f"{stats['wins']}"
    )

    print(
        f"Losses             : "
        f"{stats['losses']}"
    )

    print(
        f"Win Rate           : "
        f"{stats['win_rate']:.2f}%"
    )

    print(
        f"Net Result         : "
        f"{stats['net_r']:+.2f}R"
    )

    print(
        f"Average R          : "
        f"{stats['average_r']:+.3f}R"
    )

    pf = stats["profit_factor"]

    if pf == float("inf"):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.2f}"

    print(
        f"Profit Factor      : "
        f"{pf_text}"
    )

    print(
        f"Max Drawdown       : "
        f"{stats['max_drawdown']:.2f}R"
    )

    print(
        f"Max Win Streak     : "
        f"{stats['max_win_streak']}"
    )

    print(
        f"Max Loss Streak    : "
        f"{stats['max_loss_streak']}"
    )

    # ========================================================
    # RESULT BY COIN
    # ========================================================

    print()
    print("=" * 110)
    print("RESULT BY COIN")
    print("=" * 110)

    for symbol in COINS:

        coin_trades = [
            t for t in all_trades
            if t["symbol"] == symbol
        ]

        if not coin_trades:

            print(
                f"{symbol:5s} | 0 trades"
            )

            continue

        s = calculate_stats(
            coin_trades
        )

        print(
            f"{symbol:5s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f} | "
            f"PF: {s['profit_factor']:.2f}"
        )

    # ========================================================
    # RESULT BY SETUP
    # ========================================================

    print()
    print("=" * 110)
    print("RESULT BY SETUP")
    print("=" * 110)

    for setup in [
        "BREAKOUT",
        "REVERSAL"
    ]:

        setup_trades = [
            t for t in all_trades
            if t["setup"] == setup
        ]

        if not setup_trades:
            continue

        s = calculate_stats(
            setup_trades
        )

        print(
            f"{setup:10s} | "
            f"Trades: {s['total']:3d} | "
            f"W: {s['wins']:3d} | "
            f"L: {s['losses']:3d} | "
            f"WR: {s['win_rate']:6.2f}% | "
            f"R: {s['net_r']:+7.2f} | "
            f"PF: {s['profit_factor']:.2f}"
        )

    # ========================================================
    # FULL TRADE LOG
    # ========================================================

    print()
    print("=" * 110)
    print("FULL TRADE LOG")
    print("=" * 110)

    for number, trade in enumerate(
        all_trades,
        1
    ):

        print(
            f"{number:03d} | "
            f"{trade['symbol']:5s} | "
            f"LONG  | "
            f"{trade['setup']:9s} | "
            f"ENTRY "
            f"{utc_time(trade['entry_time'])} | "
            f"E "
            f"{fmt_price(trade['entry'])} | "
            f"TP "
            f"{fmt_price(trade['tp'])} | "
            f"SL "
            f"{fmt_price(trade['sl'])} | "
            f"EXIT "
            f"{utc_time(trade['exit_time'])} | "
            f"{trade['result']:2s} | "
            f"{trade['R']:+.1f}R | "
            f"ADX "
            f"{trade['adx']:.1f} | "
            f"RSI "
            f"{trade['rsi']:.1f}"
        )

    print()
    print("=" * 110)
    print("BACKTEST FINISHED")
    print("=" * 110)


if __name__ == "__main__":
    main()
