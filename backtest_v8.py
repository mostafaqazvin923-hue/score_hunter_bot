import os
import json
import math
import statistics
import requests
from datetime import datetime, timezone, timedelta


# ============================================================
# SCORE HUNTER PRO - BACKTEST
# NO PANDAS VERSION
#
# 4H TREND
# +
# 1H ENTRY
# +
# PULLBACK / BREAKOUT
# +
# REVERSAL
# +
# EMA
# +
# RSI
# +
# ATR
# +
# ADX
# +
# CLOSED CANDLE ONLY
# +
# TP = 2R
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"

INTERVAL_4H = 240
INTERVAL_1H = 60

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

PULLBACK_LOOKBACK = 8
STRUCTURE_LOOKBACK = 5

EMA_PULLBACK_ATR = 1.0

SL_ATR = 1.5
TP_R_MULTIPLE = 2.0

MIN_RR = 1.5

HISTORY_DAYS = 365
OOS_DAYS = 90

STARTING_BALANCE = 1000.0
RISK_PER_TRADE_PERCENT = 1.0

MAX_HOLD_CANDLES = 48

# جلوگیری از ورودهای پشت سر هم
COOLDOWN_CANDLES = 3


# ============================================================
# HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "ScoreHunterBacktest/1.0"
})


# ============================================================
# KRAKEN
# ============================================================

def get_ohlc(symbol, interval, since=None):

    params = {
        "pair": COINS[symbol],
        "interval": interval
    }

    if since is not None:
        params["since"] = since

    response = SESSION.get(
        KRAKEN_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"{symbol} Kraken error: "
            f"{data['error']}"
        )

    result = data.get("result", {})

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
            f"{symbol}: no candle data"
        )

    candles = []

    for row in result[pair_key]:

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

    # Remove duplicate timestamps
    unique = {}

    for candle in candles:
        unique[candle["time"]] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x["time"]
    )

    # Kraken may return current candle.
    # Remove it.
    if len(candles) > 1:
        candles = candles[:-1]

    return candles


# ============================================================
# EMA
# ============================================================

def ema(values, period):

    if len(values) < period:
        return None

    value = (
        sum(values[:period])
        / period
    )

    multiplier = (
        2.0
        / (period + 1)
    )

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
            current["high"]
            - current["low"],

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

    return (
        sum(trs[-period:])
        / period
    )


# ============================================================
# RSI
# ============================================================

def rsi(candles, period=14):

    if len(candles) < period + 1:
        return None

    closes = [
        candle["close"]
        for candle in candles
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

    if len(candles) < (
        period * 3
    ):
        return None

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(
        1,
        len(candles)
    ):

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

        up_move = (
            high
            - prev_high
        )

        down_move = (
            prev_low
            - low
        )

        if (
            up_move > down_move
            and up_move > 0
        ):
            p_dm = up_move
        else:
            p_dm = 0.0

        if (
            down_move > up_move
            and down_move > 0
        ):
            m_dm = down_move
        else:
            m_dm = 0.0

        trs.append(tr)
        plus_dm.append(p_dm)
        minus_dm.append(m_dm)

    if len(trs) < period * 2:
        return None

    atr_value = (
        sum(trs[:period])
        / period
    )

    plus_value = (
        sum(plus_dm[:period])
        / period
    )

    minus_value = (
        sum(minus_dm[:period])
        / period
    )

    dx_values = []

    for i in range(
        period,
        len(trs)
    ):

        atr_value = (
            (
                atr_value
                * (period - 1)
                + trs[i]
            )
            / period
        )

        plus_value = (
            (
                plus_value
                * (period - 1)
                + plus_dm[i]
            )
            / period
        )

        minus_value = (
            (
                minus_value
                * (period - 1)
                + minus_dm[i]
            )
            / period
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
                adx_value
                * (period - 1)
                + value
            )
            / period
        )

    return adx_value


# ============================================================
# 4H TREND
# ============================================================

def get_4h_direction(candles):

    closes = [
        c["close"]
        for c in candles
    ]

    close = closes[-1]

    ema20_value = ema(
        closes,
        EMA20
    )

    ema50_value = ema(
        closes,
        EMA50
    )

    ema200_value = ema(
        closes,
        EMA200
    )

    if (
        ema20_value is None
        or ema50_value is None
        or ema200_value is None
    ):
        return None

    if (
        close > ema200_value
        and ema20_value > ema50_value
        and ema50_value > ema200_value
    ):
        return "LONG"

    if (
        close < ema200_value
        and ema20_value < ema50_value
        and ema50_value < ema200_value
    ):
        return "SHORT"

    return None


# ============================================================
# EMA ALIGNMENT
# ============================================================

def ema_alignment(
    candles,
    direction
):

    closes = [
        c["close"]
        for c in candles
    ]

    e20 = ema(
        closes,
        EMA20
    )

    e50 = ema(
        closes,
        EMA50
    )

    e200 = ema(
        closes,
        EMA200
    )

    if (
        e20 is None
        or e50 is None
        or e200 is None
    ):
        return False

    price = closes[-1]

    if direction == "LONG":

        return (
            price > e20
            and e20 > e50
            and e50 > e200
        )

    return (
        price < e20
        and e20 < e50
        and e50 < e200
    )


# ============================================================
# PULLBACK
# ============================================================

def detect_pullback(
    candles,
    direction,
    atr_value
):

    if atr_value is None:
        return False

    if len(candles) < (
        EMA50
        + PULLBACK_LOOKBACK
    ):
        return False

    start = (
        len(candles)
        - PULLBACK_LOOKBACK
    )

    for i in range(
        start,
        len(candles)
    ):

        partial = candles[:i + 1]

        closes = [
            c["close"]
            for c in partial
        ]

        e20 = ema(
            closes,
            EMA20
        )

        e50 = ema(
            closes,
            EMA50
        )

        if (
            e20 is None
            or e50 is None
        ):
            continue

        candle = candles[i]

        if direction == "LONG":

            if (
                candle["low"]
                <= e20
                + atr_value
                * EMA_PULLBACK_ATR
            ):
                return True

            if (
                candle["low"]
                <= e50
                + atr_value
                * EMA_PULLBACK_ATR
            ):
                return True

        else:

            if (
                candle["high"]
                >= e20
                - atr_value
                * EMA_PULLBACK_ATR
            ):
                return True

            if (
                candle["high"]
                >= e50
                - atr_value
                * EMA_PULLBACK_ATR
            ):
                return True

    return False


# ============================================================
# BREAKOUT
# ============================================================

def detect_breakout(
    candles,
    direction
):

    if len(candles) < (
        STRUCTURE_LOOKBACK + 2
    ):
        return False

    current = candles[-1]

    previous = candles[
        -STRUCTURE_LOOKBACK - 1:-1
    ]

    if direction == "LONG":

        resistance = max(
            c["high"]
            for c in previous
        )

        return (
            current["close"]
            > resistance
        )

    support = min(
        c["low"]
        for c in previous
    )

    return (
        current["close"]
        < support
    )


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def candle_confirmation(
    candle,
    direction
):

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    body = abs(
        candle["close"]
        - candle["open"]
    )

    body_ratio = (
        body
        / candle_range
    )

    if direction == "LONG":

        close_location = (
            candle["close"]
            - candle["low"]
        ) / candle_range

        return (
            candle["close"]
            > candle["open"]
            and body_ratio >= 0.45
            and close_location >= 0.60
        )

    close_location = (
        candle["high"]
        - candle["close"]
    ) / candle_range

    return (
        candle["close"]
        < candle["open"]
        and body_ratio >= 0.45
        and close_location >= 0.60
    )


# ============================================================
# MOMENTUM
# ============================================================

def momentum_confirmation(
    candles,
    direction
):

    if len(candles) < 3:
        return False

    current = candles[-1]
    previous = candles[-2]

    if direction == "LONG":

        return (
            current["close"]
            > previous["close"]
        )

    return (
        current["close"]
        < previous["close"]
    )


# ============================================================
# RSI
# ============================================================

def rsi_confirmation(
    value,
    direction
):

    if value is None:
        return False

    if direction == "LONG":

        return (
            50 <= value <= 75
        )

    return (
        25 <= value <= 50
    )


# ============================================================
# RISK
# ============================================================

def calculate_risk(
    candles,
    direction,
    entry,
    atr_value
):

    recent = candles[-6:]

    recent_low = min(
        c["low"]
        for c in recent
    )

    recent_high = max(
        c["high"]
        for c in recent
    )

    if direction == "LONG":

        atr_sl = (
            entry
            - atr_value * SL_ATR
        )

        structure_sl = (
            recent_low
            - atr_value * 0.10
        )

        sl = max(
            atr_sl,
            structure_sl
        )

        risk = entry - sl

        if risk <= 0:
            return None

        tp = (
            entry
            + risk * TP_R_MULTIPLE
        )

    else:

        atr_sl = (
            entry
            + atr_value * SL_ATR
        )

        structure_sl = (
            recent_high
            + atr_value * 0.10
        )

        sl = min(
            atr_sl,
            structure_sl
        )

        risk = sl - entry

        if risk <= 0:
            return None

        tp = (
            entry
            - risk * TP_R_MULTIPLE
        )

    rr = (
        abs(tp - entry)
        / risk
    )

    if rr < MIN_RR:
        return None

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk": risk,
        "rr": rr
    }


# ============================================================
# SIGNAL
# ============================================================

def analyze(
    candles_4h,
    candles_1h
):

    trend = get_4h_direction(
        candles_4h
    )

    if trend is None:
        return None

    current = candles_1h[-1]

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

    if not ema_alignment(
        candles_1h,
        trend
    ):
        return None

    if not rsi_confirmation(
        rsi_value,
        trend
    ):
        return None

    pullback = detect_pullback(
        candles_1h,
        trend,
        atr_value
    )

    breakout = detect_breakout(
        candles_1h,
        trend
    )

    candle_ok = candle_confirmation(
        current,
        trend
    )

    momentum_ok = momentum_confirmation(
        candles_1h,
        trend
    )

    if not (
        (
            pullback
            and (
                candle_ok
                or momentum_ok
            )
        )
        or
        (
            breakout
            and candle_ok
        )
    ):
        return None

    risk = calculate_risk(
        candles_1h,
        trend,
        current["close"],
        atr_value
    )

    if risk is None:
        return None

    return {
        "direction": trend,
        "entry": risk["entry"],
        "sl": risk["sl"],
        "tp": risk["tp"],
        "risk": risk["risk"],
        "rr": risk["rr"],
        "atr": atr_value,
        "rsi": rsi_value,
        "adx": adx_value,
        "setup":
            "PULLBACK"
            if pullback
            else "BREAKOUT",
        "time": current["time"]
    }


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    candles,
    start_index,
    signal
):

    direction = signal["direction"]

    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]

    end_index = min(
        len(candles),
        start_index
        + MAX_HOLD_CANDLES
    )

    for i in range(
        start_index,
        end_index
    ):

        candle = candles[i]

        high = candle["high"]
        low = candle["low"]

        if direction == "LONG":

            hit_sl = (
                low <= sl
            )

            hit_tp = (
                high >= tp
            )

            # Conservative:
            # if both are hit in same candle,
            # SL wins.
            if hit_sl and hit_tp:
                return {
                    "result": "LOSS",
                    "exit": sl,
                    "bars": i - start_index + 1
                }

            if hit_sl:
                return {
                    "result": "LOSS",
                    "exit": sl,
                    "bars": i - start_index + 1
                }

            if hit_tp:
                return {
                    "result": "WIN",
                    "exit": tp,
                    "bars": i - start_index + 1
                }

        else:

            hit_sl = (
                high >= sl
            )

            hit_tp = (
                low <= tp
            )

            if hit_sl and hit_tp:
                return {
                    "result": "LOSS",
                    "exit": sl,
                    "bars": i - start_index + 1
                }

            if hit_sl:
                return {
                    "result": "LOSS",
                    "exit": sl,
                    "bars": i - start_index + 1
                }

            if hit_tp:
                return {
                    "result": "WIN",
                    "exit": tp,
                    "bars": i - start_index + 1
                }

    # If neither TP nor SL is hit,
    # close at final candle.
    final = candles[end_index - 1]["close"]

    if direction == "LONG":

        pnl_r = (
            final - entry
        ) / (
            entry - sl
        )

    else:

        pnl_r = (
            entry - final
        ) / (
            sl - entry
        )

    if pnl_r > 0:
        result = "WIN"
    else:
        result = "LOSS"

    return {
        "result": result,
        "exit": final,
        "bars": end_index - start_index,
        "pnl_r": pnl_r
    }


# ============================================================
# STATS
# ============================================================

def calculate_stats(
    trades,
    starting_balance
):

    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "net_r": 0,
            "max_drawdown": 0,
            "avg_r": 0,
            "balance": starting_balance
        }

    wins = 0
    losses = 0

    gross_profit = 0.0
    gross_loss = 0.0

    net_r = 0.0

    balance = starting_balance
    peak = balance
    max_drawdown = 0.0

    r_values = []

    for trade in trades:

        if trade["result"] == "WIN":

            wins += 1

            r = trade.get(
                "pnl_r",
                2.0
            )

            if r <= 0:
                r = 2.0

        else:

            losses += 1

            r = trade.get(
                "pnl_r",
                -1.0
            )

            if r >= 0:
                r = -1.0

        r_values.append(r)

        net_r += r

        risk_money = (
            balance
            * RISK_PER_TRADE_PERCENT
            / 100.0
        )

        balance += (
            risk_money * r
        )

        if r > 0:
            gross_profit += r
        else:
            gross_loss += abs(r)

        peak = max(
            peak,
            balance
        )

        drawdown = (
            (peak - balance)
            / peak
            * 100.0
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

    total = len(trades)

    win_rate = (
        wins
        / total
        * 100.0
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = float("inf")

    avg_r = (
        statistics.mean(r_values)
        if r_values
        else 0.0
    )

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_r": net_r,
        "max_drawdown": max_drawdown,
        "avg_r": avg_r,
        "balance": balance
    }


# ============================================================
# BACKTEST ONE COIN
# ============================================================

def backtest_coin(
    symbol,
    candles_4h,
    candles_1h,
    oos_start_time=None
):

    trades = []

    last_trade_time = None

    # 4H candle is used only after it is closed.
    four_h_times = [
        c["time"]
        for c in candles_4h
    ]

    for i in range(
        EMA200,
        len(candles_1h) - 2
    ):

        candle_time = (
            candles_1h[i]["time"]
        )

        if (
            oos_start_time is not None
            and candle_time < oos_start_time
        ):
            continue

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if last_trade_time is not None:

            hours_since = (
                candle_time
                - last_trade_time
            ) / 3600.0

            if hours_since < (
                COOLDOWN_CANDLES
            ):
                continue

        # ----------------------------------------------------
        # Find latest CLOSED 4H candle
        # ----------------------------------------------------

        valid_4h = [
            c
            for c in candles_4h
            if c["time"] <= candle_time
        ]

        if len(valid_4h) < EMA200:
            continue

        candles4 = valid_4h

        # ----------------------------------------------------
        # Closed 1H candles
        # ----------------------------------------------------

        candles1 = candles_1h[:i + 1]

        signal = analyze(
            candles4,
            candles1
        )

        if signal is None:
            continue

        # ----------------------------------------------------
        # Entry on NEXT candle OPEN
        #
        # This is important:
        # signal is generated at closed candle,
        # entry happens after it.
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(candles_1h):
            break

        actual_entry = (
            candles_1h[
                entry_index
            ]["open"]
        )

        # Recalculate SL/TP around actual entry.
        risk = calculate_risk(
            candles1,
            signal["direction"],
            actual_entry,
            signal["atr"]
        )

        if risk is None:
            continue

        live_signal = {
            **signal,
            "entry": actual_entry,
            "sl": risk["sl"],
            "tp": risk["tp"],
            "risk": risk["risk"],
            "rr": risk["rr"]
        }

        result = simulate_trade(
            candles_1h,
            entry_index,
            live_signal
        )

        if "pnl_r" not in result:

            if result["result"] == "WIN":
                pnl_r = 2.0
            else:
                pnl_r = -1.0

            result["pnl_r"] = pnl_r

        trade = {
            "symbol": symbol,
            "time": candle_time,
            "datetime":
                datetime.fromtimestamp(
                    candle_time,
                    tz=timezone.utc
                ).isoformat(),
            "direction":
                live_signal["direction"],
            "setup":
                live_signal["setup"],
            "entry":
                live_signal["entry"],
            "sl":
                live_signal["sl"],
            "tp":
                live_signal["tp"],
            "result":
                result["result"],
            "pnl_r":
                result["pnl_r"],
            "bars":
                result["bars"],
            "rsi":
                live_signal["rsi"],
            "adx":
                live_signal["adx"]
        }

        trades.append(trade)

        # Cooldown begins from entry signal.
        last_trade_time = candle_time

    return trades


# ============================================================
# DATA PERIOD
# ============================================================

def get_period():

    end = datetime.now(
        timezone.utc
    )

    start = (
        end
        - timedelta(
            days=HISTORY_DAYS
        )
    )

    return (
        int(start.timestamp()),
        int(end.timestamp())
    )


# ============================================================
# REPORT
# ============================================================

def print_stats(
    name,
    stats
):

    pf = stats["profit_factor"]

    if math.isinf(pf):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.2f}"

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"{name}"
    )

    print(
        "=" * 60
    )

    print(
        f"Trades       : "
        f"{stats['trades']}"
    )

    print(
        f"Wins         : "
        f"{stats['wins']}"
    )

    print(
        f"Losses       : "
        f"{stats['losses']}"
    )

    print(
        f"Win Rate     : "
        f"{stats['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{pf_text}"
    )

    print(
        f"Net R        : "
        f"{stats['net_r']:.2f}"
    )

    print(
        f"Average R    : "
        f"{stats['avg_r']:.3f}"
    )

    print(
        f"Max Drawdown : "
        f"{stats['max_drawdown']:.2f}%"
    )

    print(
        f"Balance      : "
        f"${stats['balance']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "============================================================"
    )

    print(
        "SCORE HUNTER PRO BACKTEST"
    )

    print(
        "NO PANDAS VERSION"
    )

    print(
        "4H TREND + 1H ENTRY"
    )

    print(
        "CLOSED CANDLE ONLY"
    )

    print(
        "ENTRY = NEXT 1H OPEN"
    )

    print(
        "TP = 2R"
    )

    print(
        "============================================================"
    )

    since, until = get_period()

    all_trades = []

    coin_results = {}

    for symbol in COINS:

        print(
            f"\nDownloading {symbol}..."
        )

        try:

            candles4 = get_ohlc(
                symbol,
                INTERVAL_4H,
                since
            )

            candles1 = get_ohlc(
                symbol,
                INTERVAL_1H,
                since
            )

            print(
                f"{symbol} 4H candles: "
                f"{len(candles4)}"
            )

            print(
                f"{symbol} 1H candles: "
                f"{len(candles1)}"
            )

            if len(candles4) < 210:
                print(
                    f"{symbol}: "
                    f"not enough 4H data"
                )
                continue

            if len(candles1) < 300:
                print(
                    f"{symbol}: "
                    f"not enough 1H data"
                )
                continue

            trades = backtest_coin(
                symbol,
                candles4,
                candles1
            )

            coin_results[symbol] = trades

            all_trades.extend(
                trades
            )

            stats = calculate_stats(
                trades,
                STARTING_BALANCE
            )

            print_stats(
                symbol,
                stats
            )

        except Exception as e:

            print(
                f"{symbol} ERROR: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # ========================================================
    # ALL COINS
    # ========================================================

    overall = calculate_stats(
        all_trades,
        STARTING_BALANCE
    )

    print_stats(
        "ALL COINS",
        overall
    )

    # ========================================================
    # DAILY SIGNAL FREQUENCY
    # ========================================================

    days = set()

    for trade in all_trades:

        day = (
            datetime.fromtimestamp(
                trade["time"],
                tz=timezone.utc
            ).date()
        )

        days.add(day)

    if days:

        signals_per_day = (
            len(all_trades)
            / len(days)
        )

    else:

        signals_per_day = 0.0

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SIGNAL FREQUENCY"
    )

    print(
        "=" * 60
    )

    print(
        f"Trading days : "
        f"{len(days)}"
    )

    print(
        f"Total signals: "
        f"{len(all_trades)}"
    )

    print(
        f"Signals/day  : "
        f"{signals_per_day:.2f}"
    )

    # ========================================================
    # OOS
    # ========================================================

    oos_start = (
        int(
            (
                datetime.now(
                    timezone.utc
                )
                - timedelta(
                    days=OOS_DAYS
                )
            ).timestamp()
        )
    )

    oos_trades = []

    for symbol, trades in coin_results.items():

        for trade in trades:

            if (
                trade["time"]
                >= oos_start
            ):
                oos_trades.append(
                    trade
                )

    oos_stats = calculate_stats(
        oos_trades,
        STARTING_BALANCE
    )

    print_stats(
        "OUT OF SAMPLE - LAST 90 DAYS",
        oos_stats
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    report = {
        "settings": {
            "history_days":
                HISTORY_DAYS,

            "oos_days":
                OOS_DAYS,

            "starting_balance":
                STARTING_BALANCE,

            "risk_per_trade":
                RISK_PER_TRADE_PERCENT,

            "sl_atr":
                SL_ATR,

            "tp_r":
                TP_R_MULTIPLE,

            "adx_min":
                ADX_MIN
        },

        "overall": overall,

        "oos": oos_stats,

        "signals_per_day":
            signals_per_day,

        "trades":
            all_trades
    }

    with open(
        "backtest_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to:"
    )

    print(
        "backtest_results.json"
    )

    print(
        "\nBACKTEST FINISHED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
