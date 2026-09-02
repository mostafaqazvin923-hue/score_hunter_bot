import json
import urllib.request
from datetime import datetime, timezone
import time
import math


# ============================================================
# CONFIG
# ============================================================

SYMBOLS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
}

RR = 2.0

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

# 1H trend
ADX_MIN = 18.0

# 15M entry
BREAKOUT_LOOKBACK = 8
PULLBACK_LOOKBACK = 6

# ATR stop
SL_ATR_MULT = 1.2

# Maximum bars allowed for trade
MAX_HOLD_BARS = 32


# ============================================================
# YAHOO DATA
# ============================================================

def fetch_yahoo(symbol, interval, range_value):

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?interval={interval}&range={range_value}"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    result = data["chart"]["result"][0]

    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    opens = quote["open"]
    highs = quote["high"]
    lows = quote["low"]
    closes = quote["close"]
    volumes = quote["volume"]

    candles = []

    for i in range(len(timestamps)):

        if (
            opens[i] is None
            or highs[i] is None
            or lows[i] is None
            or closes[i] is None
        ):
            continue

        candles.append({
            "time": timestamps[i],
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "volume": float(volumes[i] or 0)
        })

    return candles


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):

    if len(values) < period:
        return [None] * len(values)

    result = [None] * len(values)

    sma = sum(values[:period]) / period

    result[period - 1] = sma

    multiplier = 2 / (period + 1)

    prev = sma

    for i in range(period, len(values)):

        current = (
            values[i] - prev
        ) * multiplier + prev

        result[i] = current
        prev = current

    return result


def rsi(values, period=14):

    result = [None] * len(values)

    if len(values) <= period:
        return result

    gains = []
    losses = []

    for i in range(1, period + 1):

        change = values[i] - values[i - 1]

        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(values)):

        change = values[i] - values[i - 1]

        gain = max(change, 0)
        loss = max(-change, 0)

        avg_gain = (
            (avg_gain * (period - 1)) + gain
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + loss
        ) / period

        if avg_loss == 0:
            result[i] = 100
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


def atr(candles, period=14):

    result = [None] * len(candles)

    tr = [0]

    for i in range(1, len(candles)):

        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]

        true_range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        tr.append(true_range)

    if len(tr) <= period:
        return result

    first_atr = sum(tr[1:period + 1]) / period

    result[period] = first_atr

    prev = first_atr

    for i in range(period + 1, len(tr)):

        current = (
            (prev * (period - 1)) + tr[i]
        ) / period

        result[i] = current
        prev = current

    return result


def adx(candles, period=14):

    result = [None] * len(candles)

    if len(candles) < period * 2 + 1:
        return result

    tr = [0] * len(candles)
    plus_dm = [0] * len(candles)
    minus_dm = [0] * len(candles)

    for i in range(1, len(candles)):

        high = candles[i]["high"]
        low = candles[i]["low"]

        prev_high = candles[i - 1]["high"]
        prev_low = candles[i - 1]["low"]
        prev_close = candles[i - 1]["close"]

        tr[i] = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr_value = sum(tr[1:period + 1]) / period

    plus_smoothed = sum(plus_dm[1:period + 1])
    minus_smoothed = sum(minus_dm[1:period + 1])

    dx_values = []

    for i in range(period + 1, len(candles)):

        atr_value = (
            (atr_value * (period - 1)) + tr[i]
        ) / period

        plus_smoothed = (
            plus_smoothed
            - (plus_smoothed / period)
            + plus_dm[i]
        )

        minus_smoothed = (
            minus_smoothed
            - (minus_smoothed / period)
            + minus_dm[i]
        )

        if atr_value == 0:
            continue

        plus_di = 100 * plus_smoothed / atr_value
        minus_di = 100 * minus_smoothed / atr_value

        denominator = plus_di + minus_di

        if denominator == 0:
            continue

        dx = (
            100
            * abs(plus_di - minus_di)
            / denominator
        )

        dx_values.append((i, dx))

    if len(dx_values) < period:
        return result

    initial_adx = (
        sum(x[1] for x in dx_values[:period])
        / period
    )

    first_index = dx_values[period - 1][0]

    result[first_index] = initial_adx

    prev_adx = initial_adx

    for j in range(period, len(dx_values)):

        index = dx_values[j][0]
        dx = dx_values[j][1]

        current_adx = (
            (prev_adx * (period - 1)) + dx
        ) / period

        result[index] = current_adx

        prev_adx = current_adx

    return result


# ============================================================
# TIME HELPERS
# ============================================================

def hour_timestamp(ts):
    return ts - (ts % 3600)


def format_time(ts):

    return datetime.fromtimestamp(
        ts,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")


# ============================================================
# BUILD 1H INDICATORS
# ============================================================

def prepare_1h(candles):

    closes = [x["close"] for x in candles]

    ema20 = ema(closes, EMA_FAST)
    ema50 = ema(closes, EMA_MID)
    ema200 = ema(closes, EMA_SLOW)

    rsi14 = rsi(closes, RSI_PERIOD)

    atr14 = atr(candles, ATR_PERIOD)

    adx14 = adx(candles, ADX_PERIOD)

    for i in range(len(candles)):

        candles[i]["ema20"] = ema20[i]
        candles[i]["ema50"] = ema50[i]
        candles[i]["ema200"] = ema200[i]

        candles[i]["rsi"] = rsi14[i]
        candles[i]["atr"] = atr14[i]
        candles[i]["adx"] = adx14[i]

    return candles


# ============================================================
# GET 1H BIAS
# ============================================================

def get_bias(candle):

    if (
        candle["ema20"] is None
        or candle["ema50"] is None
        or candle["ema200"] is None
        or candle["rsi"] is None
        or candle["adx"] is None
    ):
        return None

    price = candle["close"]

    # LONG TREND
    if (
        price > candle["ema20"]
        and candle["ema20"] > candle["ema50"]
        and candle["ema50"] > candle["ema200"]
        and candle["adx"] >= ADX_MIN
        and 50 <= candle["rsi"] <= 70
    ):
        return "LONG"

    # SHORT TREND
    if (
        price < candle["ema20"]
        and candle["ema20"] < candle["ema50"]
        and candle["ema50"] < candle["ema200"]
        and candle["adx"] >= ADX_MIN
        and 30 <= candle["rsi"] <= 50
    ):
        return "SHORT"

    return None


# ============================================================
# MAP 15M -> LAST COMPLETED 1H
# ============================================================

def get_last_completed_1h(candles_1h, ts):

    target = hour_timestamp(ts) - 3600

    selected = None

    for candle in candles_1h:

        if candle["time"] <= target:
            selected = candle
        else:
            break

    return selected


# ============================================================
# BREAKOUT DETECTION
# ============================================================

def get_breakout(candles, index, direction):

    if index < BREAKOUT_LOOKBACK:
        return None

    previous = candles[
        index - BREAKOUT_LOOKBACK:index
    ]

    if direction == "LONG":

        resistance = max(
            x["high"] for x in previous
        )

        if candles[index]["close"] > resistance:

            return resistance

    elif direction == "SHORT":

        support = min(
            x["low"] for x in previous
        )

        if candles[index]["close"] < support:

            return support

    return None


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def backtest_symbol(symbol, yahoo_symbol):

    print("")
    print("=" * 60)
    print(symbol)
    print("=" * 60)

    print("Downloading 1H data...")

    candles_1h = fetch_yahoo(
        yahoo_symbol,
        "1h",
        "1y"
    )

    print(
        "1H candles:",
        len(candles_1h)
    )

    print("Downloading 15M data...")

    candles_15m = fetch_yahoo(
        yahoo_symbol,
        "15m",
        "60d"
    )

    print(
        "15M candles:",
        len(candles_15m)
    )

    if len(candles_1h) < 250:
        print("Not enough 1H data.")
        return []

    if len(candles_15m) < 100:
        print("Not enough 15M data.")
        return []

    candles_1h = prepare_1h(candles_1h)

    trades = []

    position = None
    pending = None

    for i in range(
        BREAKOUT_LOOKBACK + 1,
        len(candles_15m)
    ):

        candle = candles_15m[i]

        # ----------------------------------------------------
        # MANAGE OPEN POSITION
        # ----------------------------------------------------

        if position is not None:

            position["bars"] += 1

            high = candle["high"]
            low = candle["low"]

            result = None

            if position["direction"] == "LONG":

                # Conservative: SL first if both touched
                if low <= position["sl"]:
                    result = -1.0

                elif high >= position["tp"]:
                    result = RR

            else:

                # Conservative: SL first if both touched
                if high >= position["sl"]:
                    result = -1.0

                elif low <= position["tp"]:
                    result = RR

            if result is not None:

                trades.append({
                    "direction": position["direction"],
                    "entry": position["entry"],
                    "sl": position["sl"],
                    "tp": position["tp"],
                    "result_r": result,
                    "entry_time": position["entry_time"],
                    "exit_time": candle["time"]
                })

                position = None

                # Do not open another position
                # on the same candle
                continue

            # Maximum holding period
            if position["bars"] >= MAX_HOLD_BARS:

                trades.append({
                    "direction": position["direction"],
                    "entry": position["entry"],
                    "sl": position["sl"],
                    "tp": position["tp"],
                    "result_r": 0.0,
                    "entry_time": position["entry_time"],
                    "exit_time": candle["time"]
                })

                position = None

                continue

        # ----------------------------------------------------
        # FIND 1H BIAS
        # ----------------------------------------------------

        h1 = get_last_completed_1h(
            candles_1h,
            candle["time"]
        )

        if h1 is None:
            continue

        bias = get_bias(h1)

        if bias is None:
            pending = None
            continue

        # ----------------------------------------------------
        # NEW BREAKOUT
        # ----------------------------------------------------

        breakout = get_breakout(
            candles_15m,
            i,
            bias
        )

        if breakout is not None:

            pending = {
                "direction": bias,
                "level": breakout,
                "bars_left": PULLBACK_LOOKBACK
            }

            continue

        # ----------------------------------------------------
        # PENDING PULLBACK
        # ----------------------------------------------------

        if pending is not None:

            # Cancel if bias changed
            if pending["direction"] != bias:

                pending = None

            else:

                direction = pending["direction"]
                level = pending["level"]

                # LONG PULLBACK
                if direction == "LONG":

                    retest = (
                        candle["low"] <= level
                        and candle["close"] > level
                    )

                    if retest:

                        entry = candle["close"]

                        atr_value = h1["atr"]

                        if atr_value is not None and atr_value > 0:

                            structure_sl = candle["low"]

                            atr_sl = (
                                entry
                                - (SL_ATR_MULT * atr_value)
                            )

                            sl = min(
                                structure_sl,
                                atr_sl
                            )

                            risk = entry - sl

                            if risk > 0:

                                tp = (
                                    entry
                                    + (risk * RR)
                                )

                                position = {
                                    "direction": "LONG",
                                    "entry": entry,
                                    "sl": sl,
                                    "tp": tp,
                                    "entry_time": candle["time"],
                                    "bars": 0
                                }

                                pending = None

                                continue

                # SHORT PULLBACK
                else:

                    retest = (
                        candle["high"] >= level
                        and candle["close"] < level
                    )

                    if retest:

                        entry = candle["close"]

                        atr_value = h1["atr"]

                        if atr_value is not None and atr_value > 0:

                            structure_sl = candle["high"]

                            atr_sl = (
                                entry
                                + (SL_ATR_MULT * atr_value)
                            )

                            sl = max(
                                structure_sl,
                                atr_sl
                            )

                            risk = sl - entry

                            if risk > 0:

                                tp = (
                                    entry
                                    - (risk * RR)
                                )

                                position = {
                                    "direction": "SHORT",
                                    "entry": entry,
                                    "sl": sl,
                                    "tp": tp,
                                    "entry_time": candle["time"],
                                    "bars": 0
                                }

                                pending = None

                                continue

                pending["bars_left"] -= 1

                if pending["bars_left"] <= 0:

                    pending = None

    return trades


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(trades):

    total = len(trades)

    wins = sum(
        1 for x in trades
        if x["result_r"] > 0
    )

    losses = sum(
        1 for x in trades
        if x["result_r"] < 0
    )

    long_trades = sum(
        1 for x in trades
        if x["direction"] == "LONG"
    )

    short_trades = sum(
        1 for x in trades
        if x["direction"] == "SHORT"
    )

    win_rate = (
        (wins / total) * 100
        if total > 0
        else 0
    )

    net_r = sum(
        x["result_r"]
        for x in trades
    )

    gross_profit = sum(
        x["result_r"]
        for x in trades
        if x["result_r"] > 0
    )

    gross_loss = abs(sum(
        x["result_r"]
        for x in trades
        if x["result_r"] < 0
    ))

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = math.inf

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "long": long_trades,
        "short": short_trades,
        "win_rate": win_rate,
        "net_r": net_r,
        "profit_factor": profit_factor
    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def _backtest():

    print("")
    print("=" * 60)
    print("NEW STRATEGY BACKTEST")
    print("1H TREND + 15M BREAKOUT/PULLBACK")
    print("YAHOO FINANCE DATA")
    print("=" * 60)

    all_trades = []

    results = {}

    for symbol, yahoo_symbol in SYMBOLS.items():

        try:

            trades = backtest_symbol(
                symbol,
                yahoo_symbol
            )

            stats = calculate_stats(trades)

            results[symbol] = stats

            all_trades.extend(trades)

            print("")
            print(symbol)
            print("-" * 40)

            print(
                f"Total Trades : {stats['total']}"
            )

            print(
                f"Long Trades  : {stats['long']}"
            )

            print(
                f"Short Trades : {stats['short']}"
            )

            print(
                f"Wins         : {stats['wins']}"
            )

            print(
                f"Losses       : {stats['losses']}"
            )

            print(
                f"Win Rate     : {stats['win_rate']:.2f}%"
            )

            print(
                f"Net R        : {stats['net_r']:.2f}"
            )

            if math.isinf(stats["profit_factor"]):

                print(
                    "Profit Factor: INF"
                )

            else:

                print(
                    f"Profit Factor: "
                    f"{stats['profit_factor']:.2f}"
                )

        except Exception as e:

            print("")
            print(
                f"{symbol} ERROR: {e}"
            )

    # ========================================================
    # TOTAL PORTFOLIO
    # ========================================================

    total_stats = calculate_stats(
        all_trades
    )

    print("")
    print("=" * 40)
    print("TOTAL PORTFOLIO RESULT")
    print("=" * 40)
    print("")

    print(
        f"TOTAL TRADES : "
        f"{total_stats['total']}"
    )

    print(
        f"TOTAL LONG   : "
        f"{total_stats['long']}"
    )

    print(
        f"TOTAL SHORT  : "
        f"{total_stats['short']}"
    )

    print(
        f"TOTAL WINS   : "
        f"{total_stats['wins']}"
    )

    print(
        f"TOTAL LOSSES : "
        f"{total_stats['losses']}"
    )

    print(
        f"TOTAL WINRATE: "
        f"{total_stats['win_rate']:.2f}%"
    )

    print(
        f"TOTAL NET R  : "
        f"{total_stats['net_r']:.2f}"
    )

    if math.isinf(
        total_stats["profit_factor"]
    ):

        print(
            "PROFIT FACTOR: INF"
        )

    else:

        print(
            f"PROFIT FACTOR: "
            f"{total_stats['profit_factor']:.2f}"
        )

    print("")
    print("=" * 40)
    print("BACKTEST FINISHED")
    print("=" * 40)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    _backtest()
