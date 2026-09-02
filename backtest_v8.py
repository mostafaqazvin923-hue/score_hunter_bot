import json
import urllib.request
import urllib.parse
import time
from datetime import datetime, timezone, timedelta
import math

# ============================================================
# WHALE FLOW PRO
# Yahoo Finance Backtest
#
# 1H  = Trend Filter
# 15M = Entry
#
# BTC / ETH / SOL / XRP
# RR = 1:2
# NO PANDAS
# ============================================================

SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD"
}

INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 25.0
TARGET_RR = 2.0

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

MAX_TRADES_PER_DAY = 4
MAX_DAILY_LOSS_R = -2.0

# Yahoo limitations:
# 1H -> up to ~730 days
# 15M -> ~60 days
TREND_RANGE = "1y"
ENTRY_RANGE = "60d"


# ============================================================
# DATA
# ============================================================

def fetch_yahoo(symbol, interval, range_value):

    encoded = urllib.parse.quote(symbol)

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{encoded}?interval={interval}&range={range_value}"
        f"&includePrePost=false"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )

    print(
        f"[*] Downloading {symbol} | "
        f"{interval} | {range_value}"
    )

    try:

        with urllib.request.urlopen(req, timeout=40) as response:

            raw = response.read().decode("utf-8")
            payload = json.loads(raw)

        result = payload.get("chart", {}).get("result")

        if not result:
            print("[!] Yahoo returned no data")
            return []

        data = result[0]

        timestamps = data.get("timestamp", [])

        quote = (
            data.get("indicators", {})
            .get("quote", [{}])[0]
        )

        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        candles = []

        for i in range(len(timestamps)):

            if (
                i >= len(opens)
                or i >= len(highs)
                or i >= len(lows)
                or i >= len(closes)
                or i >= len(volumes)
            ):
                continue

            if (
                opens[i] is None
                or highs[i] is None
                or lows[i] is None
                or closes[i] is None
            ):
                continue

            candles.append({
                "timestamp": int(timestamps[i]) * 1000,
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i] or 0)
            })

        candles.sort(key=lambda x: x["timestamp"])

        print(
            f"    -> {len(candles):,} candles received"
        )

        if candles:

            first = datetime.fromtimestamp(
                candles[0]["timestamp"] / 1000,
                tz=timezone.utc
            )

            last = datetime.fromtimestamp(
                candles[-1]["timestamp"] / 1000,
                tz=timezone.utc
            )

            print(
                f"    -> {first:%Y-%m-%d %H:%M} UTC"
                f" -> {last:%Y-%m-%d %H:%M} UTC"
            )

        return candles

    except Exception as e:

        print(
            f"[!] Yahoo error for {symbol} "
            f"{interval}: {e}"
        )

        return []


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):

    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1)

    result = sum(values[:period]) / period

    for price in values[period:]:
        result = (
            price - result
        ) * multiplier + result

    return result


def ema_series(values, period):

    result = [None] * len(values)

    if len(values) < period:
        return result

    current = sum(values[:period]) / period
    result[period - 1] = current

    multiplier = 2.0 / (period + 1)

    for i in range(period, len(values)):

        current = (
            (values[i] - current)
            * multiplier
            + current
        )

        result[i] = current

    return result


def rsi(values, period=14):

    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):

        change = values[i] - values[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):

        avg_gain = (
            (avg_gain * (period - 1))
            + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1))
            + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles, period=14):

    if len(candles) < period + 1:
        return 0.0

    trs = []

    for i in range(1, len(candles)):

        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]

        tr = max(
            h - l,
            abs(h - pc),
            abs(l - pc)
        )

        trs.append(tr)

    return sum(trs[-period:]) / period


def adx(candles, period=14):

    if len(candles) < period * 2 + 1:
        return 0.0

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):

        high = candles[i]["high"]
        low = candles[i]["low"]

        prev_high = candles[i - 1]["high"]
        prev_low = candles[i - 1]["low"]
        prev_close = candles[i - 1]["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        up_move = high - prev_high
        down_move = prev_low - low

        plus = (
            up_move
            if up_move > down_move and up_move > 0
            else 0
        )

        minus = (
            down_move
            if down_move > up_move and down_move > 0
            else 0
        )

        trs.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(trs) < period:
        return 0.0

    tr_avg = sum(trs[:period]) / period
    plus_avg = sum(plus_dm[:period]) / period
    minus_avg = sum(minus_dm[:period]) / period

    dx_values = []

    for i in range(period, len(trs)):

        tr_avg = (
            (tr_avg * (period - 1))
            + trs[i]
        ) / period

        plus_avg = (
            (plus_avg * (period - 1))
            + plus_dm[i]
        ) / period

        minus_avg = (
            (minus_avg * (period - 1))
            + minus_dm[i]
        ) / period

        if tr_avg == 0:
            continue

        plus_di = 100 * plus_avg / tr_avg
        minus_di = 100 * minus_avg / tr_avg

        denominator = plus_di + minus_di

        if denominator == 0:
            continue

        dx = (
            100
            * abs(plus_di - minus_di)
            / denominator
        )

        dx_values.append(dx)

    if len(dx_values) < period:
        return 0.0

    return sum(dx_values[-period:]) / period


# ============================================================
# 1H TREND
# ============================================================

def get_1h_bias(candles, timestamp):

    usable = [
        c for c in candles
        if c["timestamp"] <= timestamp
    ]

    if len(usable) < 220:
        return None

    closes = [c["close"] for c in usable]

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    e200 = ema(closes, 200)

    current_rsi = rsi(closes, RSI_PERIOD)
    current_adx = adx(usable, ADX_PERIOD)

    price = closes[-1]

    if (
        e20 is None
        or e50 is None
        or e200 is None
        or current_rsi is None
    ):
        return None

    # Strong bullish trend
    long_bias = (
        e20 > e50
        and e50 > e200
        and price > e20
        and current_adx >= 20
        and 50 <= current_rsi <= 70
    )

    # Strong bearish trend
    short_bias = (
        e20 < e50
        and e50 < e200
        and price < e20
        and current_adx >= 20
        and 30 <= current_rsi <= 50
    )

    if long_bias:
        return "LONG"

    if short_bias:
        return "SHORT"

    return None


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def liquidity_sweep(candles, i, direction):

    if i < 5:
        return False

    current = candles[i]
    previous = candles[i - 1]

    recent = candles[i - 5:i]

    recent_high = max(
        x["high"] for x in recent
    )

    recent_low = min(
        x["low"] for x in recent
    )

    if direction == "LONG":

        # Sweep previous low then bullish recovery
        return (
            previous["low"] <= recent_low
            and current["close"] > previous["high"]
            and current["close"] > current["open"]
        )

    if direction == "SHORT":

        # Sweep previous high then bearish recovery
        return (
            previous["high"] >= recent_high
            and current["close"] < previous["low"]
            and current["close"] < current["open"]
        )

    return False


# ============================================================
# BREAKOUT / DISPLACEMENT
# ============================================================

def displacement(candles, i, direction):

    if i < 20:
        return False

    c = candles[i]

    ranges = [
        x["high"] - x["low"]
        for x in candles[i - 20:i]
    ]

    avg_range = (
        sum(ranges) / len(ranges)
        if ranges else 0
    )

    current_range = c["high"] - c["low"]

    if avg_range == 0:
        return False

    strong_move = current_range >= avg_range * 1.2

    if direction == "LONG":

        return (
            strong_move
            and c["close"] > c["open"]
        )

    if direction == "SHORT":

        return (
            strong_move
            and c["close"] < c["open"]
        )

    return False


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

def volume_confirmation(candles, i):

    if i < 20:
        return False

    current_volume = candles[i]["volume"]

    avg_volume = sum(
        x["volume"]
        for x in candles[i - 20:i]
    ) / 20

    if avg_volume == 0:
        return False

    return current_volume >= avg_volume * 1.10


# ============================================================
# ENTRY SIGNAL
# ============================================================

def entry_signal(candles, i, direction):

    if i < 25:
        return None

    sweep = liquidity_sweep(
        candles,
        i - 1,
        direction
    )

    move = displacement(
        candles,
        i,
        direction
    )

    volume_ok = volume_confirmation(
        candles,
        i
    )

    if not sweep:
        return None

    if not move:
        return None

    if not volume_ok:
        return None

    current = candles[i]
    previous = candles[i - 1]

    atr_value = atr(
        candles[:i + 1],
        ATR_PERIOD
    )

    if atr_value <= 0:
        return None

    if direction == "LONG":

        entry = current["close"]

        stop = (
            previous["low"]
            - atr_value * 0.5
        )

        risk = entry - stop

        if risk <= 0:
            return None

        # Prevent abnormal SL
        if risk < entry * 0.002:
            return None

        if risk > entry * 0.04:
            return None

        target = (
            entry
            + risk * TARGET_RR
        )

        return {
            "direction": "LONG",
            "entry": entry,
            "stop": stop,
            "target": target
        }

    if direction == "SHORT":

        entry = current["close"]

        stop = (
            previous["high"]
            + atr_value * 0.5
        )

        risk = stop - entry

        if risk <= 0:
            return None

        if risk < entry * 0.002:
            return None

        if risk > entry * 0.04:
            return None

        target = (
            entry
            - risk * TARGET_RR
        )

        return {
            "direction": "SHORT",
            "entry": entry,
            "stop": stop,
            "target": target
        }

    return None


# ============================================================
# BACKTEST
# ============================================================

def run_symbol(symbol_name, candles_1h, candles_15m):

    if len(candles_1h) < 220:
        print(
            f"[!] {symbol_name}: "
            f"Not enough 1H data"
        )
        return None

    if len(candles_15m) < 100:
        print(
            f"[!] {symbol_name}: "
            f"Not enough 15M data"
        )
        return None

    trades = []

    wins = 0
    losses = 0

    longs = 0
    shorts = 0

    balance_change = 0.0

    daily_trades = {}
    daily_r = {}

    in_position_until = 0

    # Start only where enough 15M history exists
    for i in range(30, len(candles_15m) - 1):

        if i < in_position_until:
            continue

        candle = candles_15m[i]

        ts = candle["timestamp"]

        date_key = datetime.fromtimestamp(
            ts / 1000,
            tz=timezone.utc
        ).strftime("%Y-%m-%d")

        daily_trades.setdefault(
            date_key,
            0
        )

        daily_r.setdefault(
            date_key,
            0.0
        )

        # Daily limits
        if daily_trades[date_key] >= MAX_TRADES_PER_DAY:
            continue

        if daily_r[date_key] <= MAX_DAILY_LOSS_R:
            continue

        bias = get_1h_bias(
            candles_1h,
            ts
        )

        if bias is None:
            continue

        signal = entry_signal(
            candles_15m,
            i,
            bias
        )

        if signal is None:
            continue

        entry = signal["entry"]
        stop = signal["stop"]
        target = signal["target"]

        direction = signal["direction"]

        result = None
        exit_index = None
        exit_price = None

        for j in range(i + 1, len(candles_15m)):

            future = candles_15m[j]

            if direction == "LONG":

                # Conservative:
                # if both TP and SL occur in same candle,
                # SL is assumed first.
                if future["low"] <= stop:

                    result = "LOSS"
                    exit_price = stop
                    exit_index = j
                    break

                if future["high"] >= target:

                    result = "WIN"
                    exit_price = target
                    exit_index = j
                    break

            else:

                if future["high"] >= stop:

                    result = "LOSS"
                    exit_price = stop
                    exit_index = j
                    break

                if future["low"] <= target:

                    result = "WIN"
                    exit_price = target
                    exit_index = j
                    break

        if result is None:
            continue

        daily_trades[date_key] += 1

        if direction == "LONG":
            longs += 1
        else:
            shorts += 1

        if result == "WIN":

            wins += 1
            r_result = TARGET_RR
            balance_change += (
                RISK_PER_TRADE
                * TARGET_RR
            )

        else:

            losses += 1
            r_result = -1.0
            balance_change -= RISK_PER_TRADE

        daily_r[date_key] += r_result

        trades.append({
            "symbol": symbol_name,
            "direction": direction,
            "entry_time": datetime.fromtimestamp(
                entry / 1000000000,
                tz=timezone.utc
            ).isoformat()
            if False else datetime.fromtimestamp(
                candles_15m[i]["timestamp"] / 1000,
                tz=timezone.utc
            ).isoformat(),
            "entry": entry,
            "stop": stop,
            "target": target,
            "result": result,
            "r": r_result,
            "exit_time": datetime.fromtimestamp(
                candles_15m[exit_index]["timestamp"] / 1000,
                tz=timezone.utc
            ).isoformat(),
            "exit": exit_price
        })

        in_position_until = exit_index + 1

    total = wins + losses

    win_rate = (
        wins / total * 100
        if total > 0
        else 0
    )

    return {
        "symbol": symbol_name,
        "trades": total,
        "longs": longs,
        "shorts": shorts,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl": balance_change,
        "trades_data": trades
    }


# ============================================================
# MAIN
# ============================================================

def run_backtest():

    print()
    print("=" * 70)
    print("              WHALE FLOW PRO")
    print("          YAHOO FINANCE BACKTEST")
    print("=" * 70)
    print()
    print("1H  = TREND FILTER")
    print("15M = ENTRY")
    print("RR  = 1:2")
    print("Capital = $1,000")
    print("Risk / Trade = $25")
    print()
    print(
        "IMPORTANT: Yahoo 15M historical data "
        "is limited, so the 15M test period "
        "will NOT be a full year."
    )
    print("=" * 70)
    print()

    all_results = []

    for symbol_name, yahoo_symbol in SYMBOLS.items():

        print()
        print("=" * 70)
        print(f"                 {symbol_name}")
        print("=" * 70)

        candles_1h = fetch_yahoo(
            yahoo_symbol,
            "1h",
            TREND_RANGE
        )

        time.sleep(1)

        candles_15m = fetch_yahoo(
            yahoo_symbol,
            "15m",
            ENTRY_RANGE
        )

        if not candles_1h or not candles_15m:
            print(
                f"[!] Skipping {symbol_name}"
            )
            continue

        result = run_symbol(
            symbol_name,
            candles_1h,
            candles_15m
        )

        if result is None:
            continue

        all_results.append(result)

        print()
        print(
            f"{symbol_name} RESULTS"
        )
        print("-" * 50)
        print(
            f"Total Trades : {result['trades']}"
        )
        print(
            f"Long Trades  : {result['longs']}"
        )
        print(
            f"Short Trades : {result['shorts']}"
        )
        print(
            f"Wins         : {result['wins']}"
        )
        print(
            f"Losses       : {result['losses']}"
        )
        print(
            f"Win Rate     : "
            f"{result['win_rate']:.2f}%"
        )
        print(
            f"P/L          : "
            f"${result['pnl']:.2f}"
        )

    # ========================================================
    # AGGREGATED
    # ========================================================

    total_trades = sum(
        x["trades"]
        for x in all_results
    )

    total_longs = sum(
        x["longs"]
        for x in all_results
    )

    total_shorts = sum(
        x["shorts"]
        for x in all_results
    )

    total_wins = sum(
        x["wins"]
        for x in all_results
    )

    total_losses = sum(
        x["losses"]
        for x in all_results
    )

    total_pnl = sum(
        x["pnl"]
        for x in all_results
    )

    overall_win_rate = (
        total_wins
        / total_trades
        * 100
        if total_trades > 0
        else 0
    )

    final_balance = (
        INITIAL_BALANCE
        + total_pnl
    )

    print()
    print()
    print("=" * 70)
    print("              AGGREGATED RESULTS")
    print("=" * 70)

    print(
        f"Total Trades       : {total_trades}"
    )

    print(
        f"Total Longs        : {total_longs}"
    )

    print(
        f"Total Shorts       : {total_shorts}"
    )

    print(
        f"Winning Trades     : {total_wins}"
    )

    print(
        f"Losing Trades      : {total_losses}"
    )

    print(
        f"Overall Win Rate   : "
        f"{overall_win_rate:.2f}%"
    )

    print(
        f"Total P/L          : "
        f"${total_pnl:.2f}"
    )

    print(
        f"Final Balance      : "
        f"${final_balance:.2f}"
    )

    print("=" * 70)

    print()
    print("PER-SYMBOL SUMMARY")
    print("-" * 70)

    for result in all_results:

        print(
            f"{result['symbol']:5} | "
            f"Trades: {result['trades']:4} | "
            f"Long: {result['longs']:4} | "
            f"Short: {result['shorts']:4} | "
            f"Win: {result['wins']:4} | "
            f"Loss: {result['losses']:4} | "
            f"WR: {result['win_rate']:6.2f}%"
        )

    print()
    print("=" * 70)
    print("BACKTEST FINISHED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    run_backtest()
