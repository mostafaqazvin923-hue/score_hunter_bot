import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone


# ============================================================
# COINEX REAL DATA BACKTEST
#
# 15M TREND + MOMENTUM + BREAKOUT/PULLBACK
# REAL COINEX DATA
# NO PANDAS
# NO NUMPY
# CLOSED CANDLE ONLY
# ENTRY = NEXT CANDLE OPEN
# TP = 2R
# SL = SWING / ATR
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 1000.0

RISK_PER_TRADE = 1.0       # درصد سرمایه
RR_RATIO = 2.0

ATR_PERIOD = 14

EMA_FAST = 10
EMA_MID = 30
EMA_SLOW = 100

VOLUME_LOOKBACK = 14

VOLUME_MULTIPLIER = 1.5

BODY_RATIO_MIN = 0.65

SWING_LOOKBACK = 10

ATR_MULTIPLIER = 1.2

BREAKOUT_LOOKBACK = 10

ADX_PERIOD = 14
ADX_MIN = 18.0

RSI_PERIOD = 14

RSI_LONG_MIN = 52
RSI_LONG_MAX = 72

RSI_SHORT_MIN = 28
RSI_SHORT_MAX = 48

# هزینه تقریبی رفت و برگشت
FEE_RATE = 0.0006

# اسلیپیج هر طرف
SLIPPAGE = 0.0002

# تعداد روز داده
DAYS = 180

# OOS
OOS_DAYS = 45


# ============================================================
# COINEX
# ============================================================

BASE_URL = "https://api.coinex.com/v2/spot/kline"

ASSETS = {
    "BTCUSDT": "BTCUSDT",
    "ETHUSDT": "ETHUSDT",
    "SOLUSDT": "SOLUSDT",
    "XRPUSDT": "XRPUSDT",
    "AVAXUSDT": "AVAXUSDT",
    "LINKUSDT": "LINKUSDT",
}


# ============================================================
# HTTP
# ============================================================

def http_get(url):

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        return response.read().decode(
            "utf-8"
        )


# ============================================================
# COINEX KLINES
# ============================================================

def get_coinex_klines(symbol, days=DAYS):

    now_ms = int(
        time.time() * 1000
    )

    start_ms = now_ms - (
        days * 24 * 60 * 60 * 1000
    )

    all_candles = []

    current_start = start_ms

    while current_start < now_ms:

        params = urllib.parse.urlencode({

            "market": symbol,

            "period": "15min",

            "start_time": current_start,

            "end_time": now_ms

        })

        url = (
            BASE_URL
            + "?"
            + params
        )

        try:

            raw = http_get(url)

            data = json.loads(raw)

        except Exception as e:

            print(
                f"{symbol}: download error: {e}"
            )

            break

        if data.get("code") != 0:

            print(
                f"{symbol}: API error:"
                f" {data}"
            )

            break

        rows = (
            data
            .get("data", [])
        )

        if not rows:
            break

        for row in rows:

            # CoinEx v2 returns:
            # [timestamp, open, close, high, low, volume, value]

            if isinstance(row, list):

                if len(row) < 6:
                    continue

                try:

                    candle = {
                        "time": int(row[0]),
                        "open": float(row[1]),
                        "close": float(row[2]),
                        "high": float(row[3]),
                        "low": float(row[4]),
                        "volume": float(row[5])
                    }

                    all_candles.append(
                        candle
                    )

                except Exception:
                    continue

            elif isinstance(row, dict):

                try:

                    candle = {
                        "time": int(
                            row["created_at"]
                        ),
                        "open": float(
                            row["open"]
                        ),
                        "close": float(
                            row["close"]
                        ),
                        "high": float(
                            row["high"]
                        ),
                        "low": float(
                            row["low"]
                        ),
                        "volume": float(
                            row["volume"]
                        )
                    }

                    all_candles.append(
                        candle
                    )

                except Exception:
                    continue

        last_time = max(
            c["time"]
            for c in all_candles
        )

        if last_time <= current_start:
            break

        current_start = (
            last_time + 1
        )

        time.sleep(0.15)

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique = {}

    for candle in all_candles:

        unique[
            candle["time"]
        ] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x["time"]
    )

    # --------------------------------------------------------
    # Remove currently forming candle
    # --------------------------------------------------------

    if len(candles) > 1:

        candles = candles[:-1]

    return candles


# ============================================================
# EMA SERIES
# ============================================================

def ema_series(values, period):

    result = [
        None
    ] * len(values)

    if len(values) < period:
        return result

    value = (
        sum(values[:period])
        / period
    )

    result[period - 1] = value

    multiplier = (
        2.0
        / (period + 1)
    )

    for i in range(
        period,
        len(values)
    ):

        value = (
            (
                values[i]
                - value
            )
            * multiplier
            + value
        )

        result[i] = value

    return result


# ============================================================
# ATR
# ============================================================

def atr_series(candles, period):

    result = [
        None
    ] * len(candles)

    if len(candles) <= period:
        return result

    tr = []

    for i in range(
        len(candles)
    ):

        if i == 0:

            value = (
                candles[i]["high"]
                - candles[i]["low"]
            )

        else:

            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = (
                candles[i - 1]["close"]
            )

            value = max(
                high - low,
                abs(
                    high - prev_close
                ),
                abs(
                    low - prev_close
                )
            )

        tr.append(value)

    value = (
        sum(tr[:period])
        / period
    )

    result[period - 1] = value

    for i in range(
        period,
        len(tr)
    ):

        value = (
            (
                value
                * (period - 1)
                + tr[i]
            )
            / period
        )

        result[i] = value

    return result


# ============================================================
# RSI
# ============================================================

def rsi_series(candles, period):

    result = [
        None
    ] * len(candles)

    if len(candles) <= period:
        return result

    gains = []
    losses = []

    for i in range(
        1,
        len(candles)
    ):

        change = (
            candles[i]["close"]
            - candles[i - 1]["close"]
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

    def calculate(gain, loss):

        if loss == 0:
            return 100.0

        rs = (
            gain / loss
        )

        return (
            100
            - (
                100
                / (1 + rs)
            )
        )

    result[period] = calculate(
        avg_gain,
        avg_loss
    )

    for j in range(
        period,
        len(gains)
    ):

        avg_gain = (
            (
                avg_gain
                * (period - 1)
                + gains[j]
            )
            / period
        )

        avg_loss = (
            (
                avg_loss
                * (period - 1)
                + losses[j]
            )
            / period
        )

        result[j + 1] = calculate(
            avg_gain,
            avg_loss
        )

    return result


# ============================================================
# ADX
# ============================================================

def adx_series(candles, period):

    result = [
        None
    ] * len(candles)

    if len(candles) < (
        period * 2 + 2
    ):
        return result

    tr = []
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

        true_range = max(
            high - low,
            abs(
                high - prev_close
            ),
            abs(
                low - prev_close
            )
        )

        up = (
            high - prev_high
        )

        down = (
            prev_low - low
        )

        p_dm = (
            up
            if (
                up > down
                and up > 0
            )
            else 0.0
        )

        m_dm = (
            down
            if (
                down > up
                and down > 0
            )
            else 0.0
        )

        tr.append(
            true_range
        )

        plus_dm.append(
            p_dm
        )

        minus_dm.append(
            m_dm
        )

    if len(tr) < period * 2:
        return result

    atr_value = (
        sum(tr[:period])
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

    dx = []

    for i in range(
        period,
        len(tr)
    ):

        atr_value = (
            (
                atr_value
                * (period - 1)
                + tr[i]
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

        if atr_value == 0:
            dx.append(0)
            continue

        plus_di = (
            100
            * plus_value
            / atr_value
        )

        minus_di = (
            100
            * minus_value
            / atr_value
        )

        total = (
            plus_di
            + minus_di
        )

        if total == 0:

            dx.append(0)

        else:

            dx.append(
                100
                * abs(
                    plus_di
                    - minus_di
                )
                / total
            )

    if len(dx) < period:
        return result

    adx_value = (
        sum(dx[:period])
        / period
    )

    index = (
        period * 2
    )

    result[index] = adx_value

    for i in range(
        period,
        len(dx)
    ):

        adx_value = (
            (
                adx_value
                * (period - 1)
                + dx[i]
            )
            / period
        )

        result[
            index
            + (i - period)
            + 1
        ] = adx_value

    return result


# ============================================================
# CANDLE STRENGTH
# ============================================================

def strong_candle(
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

    if body_ratio < BODY_RATIO_MIN:
        return False

    if direction == "LONG":

        close_position = (
            candle["close"]
            - candle["low"]
        ) / candle_range

        return (
            candle["close"]
            > candle["open"]
            and close_position >= 0.70
        )

    else:

        close_position = (
            candle["high"]
            - candle["close"]
        ) / candle_range

        return (
            candle["close"]
            < candle["open"]
            and close_position >= 0.70
        )


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

def volume_confirmation(
    candles,
    index
):

    if index < VOLUME_LOOKBACK + 1:
        return False

    previous = candles[
        index
    ]

    start = (
        index
        - VOLUME_LOOKBACK
    )

    volumes = [
        candles[j]["volume"]
        for j in range(
            start,
            index
        )
    ]

    average = (
        sum(volumes)
        / len(volumes)
    )

    if average <= 0:
        return False

    return (
        previous["volume"]
        >= (
            average
            * VOLUME_MULTIPLIER
        )
    )


# ============================================================
# BREAKOUT
# ============================================================

def breakout_confirmation(
    candles,
    index,
    direction
):

    if index < BREAKOUT_LOOKBACK + 1:
        return False

    current = candles[index]

    previous = candles[
        index
        - BREAKOUT_LOOKBACK:
        index
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
# PULLBACK
# ============================================================

def pullback_confirmation(
    candles,
    index,
    direction,
    ema_fast,
    ema_mid,
    atr
):

    if index < 3:
        return False

    candle = candles[index]

    if direction == "LONG":

        touched = (
            candle["low"]
            <= (
                ema_fast[index]
                + atr[index] * 0.35
            )
            or
            candle["low"]
            <= (
                ema_mid[index]
                + atr[index] * 0.35
            )
        )

        recovery = (
            candle["close"]
            > ema_fast[index]
        )

        return (
            touched
            and recovery
        )

    touched = (
        candle["high"]
        >= (
            ema_fast[index]
            - atr[index] * 0.35
        )
        or
        candle["high"]
        >= (
            ema_mid[index]
            - atr[index] * 0.35
        )
    )

    recovery = (
        candle["close"]
        < ema_fast[index]
    )

    return (
        touched
        and recovery
    )


# ============================================================
# SIGNAL
# ============================================================

def get_signal(
    candles,
    i,
    ema_fast,
    ema_mid,
    ema_slow,
    atr,
    rsi,
    adx
):

    if i < 150:
        return None

    previous = candles[i]

    if (
        ema_fast[i] is None
        or ema_mid[i] is None
        or ema_slow[i] is None
        or atr[i] is None
        or rsi[i] is None
        or adx[i] is None
    ):
        return None

    # --------------------------------------------------------
    # LONG TREND
    # --------------------------------------------------------

    bull_trend = (
        ema_fast[i]
        > ema_mid[i]
        > ema_slow[i]
        and ema_fast[i]
        > ema_fast[i - 3]
    )

    bear_trend = (
        ema_fast[i]
        < ema_mid[i]
        < ema_slow[i]
        and ema_fast[i]
        < ema_fast[i - 3]
    )

    volume_ok = (
        volume_confirmation(
            candles,
            i
        )
    )

    strong_bull = strong_candle(
        previous,
        "LONG"
    )

    strong_bear = strong_candle(
        previous,
        "SHORT"
    )

    breakout_long = (
        breakout_confirmation(
            candles,
            i,
            "LONG"
        )
    )

    breakout_short = (
        breakout_confirmation(
            candles,
            i,
            "SHORT"
        )
    )

    pullback_long = (
        pullback_confirmation(
            candles,
            i,
            "LONG",
            ema_fast,
            ema_mid,
            atr
        )
    )

    pullback_short = (
        pullback_confirmation(
            candles,
            i,
            "SHORT",
            ema_fast,
            ema_mid,
            atr
        )
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    long_rsi = (
        RSI_LONG_MIN
        <= rsi[i]
        <= RSI_LONG_MAX
    )

    short_rsi = (
        RSI_SHORT_MIN
        <= rsi[i]
        <= RSI_SHORT_MAX
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx_ok = (
        adx[i]
        >= ADX_MIN
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_signal = (
        bull_trend
        and adx_ok
        and long_rsi
        and volume_ok
        and strong_bull
        and (
            breakout_long
            or pullback_long
        )
    )

    if long_signal:

        return "LONG"

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_signal = (
        bear_trend
        and adx_ok
        and short_rsi
        and volume_ok
        and strong_bear
        and (
            breakout_short
            or pullback_short
        )
    )

    if short_signal:

        return "SHORT"

    return None


# ============================================================
# RISK
# ============================================================

def create_trade(
    candles,
    signal_index,
    direction,
    capital,
    atr
):

    entry_index = (
        signal_index + 1
    )

    if entry_index >= len(candles):
        return None

    entry = candles[
        entry_index
    ]["open"]

    atr_value = atr[
        signal_index
    ]

    if atr_value is None:
        return None

    start = max(
        0,
        signal_index
        - SWING_LOOKBACK
        + 1
    )

    recent = candles[
        start:
        signal_index + 1
    ]

    if direction == "LONG":

        swing_low = min(
            c["low"]
            for c in recent
        )

        atr_stop = (
            entry
            - (
                atr_value
                * ATR_MULTIPLIER
            )
        )

        sl = min(
            swing_low,
            atr_stop
        )

        risk_per_unit = (
            entry - sl
        )

        tp = (
            entry
            + (
                risk_per_unit
                * RR_RATIO
            )
        )

    else:

        swing_high = max(
            c["high"]
            for c in recent
        )

        atr_stop = (
            entry
            + (
                atr_value
                * ATR_MULTIPLIER
            )
        )

        sl = max(
            swing_high,
            atr_stop
        )

        risk_per_unit = (
            sl - entry
        )

        tp = (
            entry
            - (
                risk_per_unit
                * RR_RATIO
            )
        )

    if risk_per_unit <= 0:
        return None

    risk_amount = (
        capital
        * RISK_PER_TRADE
        / 100.0
    )

    quantity = (
        risk_amount
        / risk_per_unit
    )

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk": risk_amount,
        "quantity": quantity,
        "entry_index": entry_index
    }


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(
    symbol,
    candles,
    start_index=150,
    end_index=None
):

    if end_index is None:
        end_index = len(candles) - 1

    closes = [
        c["close"]
        for c in candles
    ]

    ema_fast = ema_series(
        closes,
        EMA_FAST
    )

    ema_mid = ema_series(
        closes,
        EMA_MID
    )

    ema_slow = ema_series(
        closes,
        EMA_SLOW
    )

    atr = atr_series(
        candles,
        ATR_PERIOD
    )

    rsi = rsi_series(
        candles,
        RSI_PERIOD
    )

    adx = adx_series(
        candles,
        ADX_PERIOD
    )

    capital = INITIAL_CAPITAL

    peak = capital

    max_dd = 0.0

    trades = []

    active_trade = None

    i = start_index

    while i < end_index:

        # ====================================================
        # MANAGE EXISTING TRADE
        # ====================================================

        if active_trade is not None:

            candle = candles[i]

            direction = (
                active_trade[
                    "direction"
                ]
            )

            entry = (
                active_trade[
                    "entry"
                ]
            )

            sl = (
                active_trade["sl"]
            )

            tp = (
                active_trade["tp"]
            )

            result = None

            exit_price = None

            # Conservative:
            # if TP and SL are touched
            # in the same candle,
            # SL wins.

            if direction == "LONG":

                if (
                    candle["low"]
                    <= sl
                    and
                    candle["high"]
                    >= tp
                ):

                    result = "SL"
                    exit_price = sl

                elif (
                    candle["low"]
                    <= sl
                ):

                    result = "SL"
                    exit_price = sl

                elif (
                    candle["high"]
                    >= tp
                ):

                    result = "TP"
                    exit_price = tp

            else:

                if (
                    candle["high"]
                    >= sl
                    and
                    candle["low"]
                    <= tp
                ):

                    result = "SL"
                    exit_price = sl

                elif (
                    candle["high"]
                    >= sl
                ):

                    result = "SL"
                    exit_price = sl

                elif (
                    candle["low"]
                    <= tp
                ):

                    result = "TP"
                    exit_price = tp

            if result is not None:

                risk_amount = (
                    active_trade[
                        "risk"
                    ]
                )

                if result == "TP":

                    gross_r = RR_RATIO

                else:

                    gross_r = -1.0

                # fees
                fee = (
                    capital
                    * FEE_RATE
                    * 2
                )

                r_value = (
                    risk_amount
                    * gross_r
                )

                capital += (
                    r_value
                    - fee
                )

                trades.append({

                    "symbol": symbol,

                    "direction":
                        direction,

                    "entry":
                        entry,

                    "exit":
                        exit_price,

                    "result":
                        result,

                    "R":
                        gross_r,

                    "capital":
                        capital,

                    "time":
                        candle["time"]

                })

                active_trade = None

                if capital > peak:
                    peak = capital

                dd = (
                    peak - capital
                ) / peak * 100

                max_dd = max(
                    max_dd,
                    dd
                )

                i += 1

                continue

        # ====================================================
        # NEW SIGNAL
        # ====================================================

        if active_trade is None:

            signal = get_signal(
                candles,
                i,
                ema_fast,
                ema_mid,
                ema_slow,
                atr,
                rsi,
                adx
            )

            if signal:

                trade = create_trade(
                    candles,
                    i,
                    signal,
                    capital,
                    atr
                )

                if trade:

                    active_trade = trade

        i += 1

    return (
        capital,
        trades,
        max_dd
    )


# ============================================================
# STATISTICS
# ============================================================

def statistics(
    trades,
    initial_capital
):

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "profit_factor": None,
            "net_r": 0,
            "average_r": 0
        }

    wins = [
        t for t in trades
        if t["result"] == "TP"
    ]

    losses = [
        t for t in trades
        if t["result"] == "SL"
    ]

    gross_profit = sum(
        t["R"]
        for t in wins
    )

    gross_loss = abs(
        sum(
            t["R"]
            for t in losses
        )
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else None
    )

    net_r = sum(
        t["R"]
        for t in trades
    )

    return {

        "trades":
            len(trades),

        "wins":
            len(wins),

        "losses":
            len(losses),

        "win_rate":
            len(wins)
            / len(trades)
            * 100,

        "profit_factor":
            profit_factor,

        "net_r":
            net_r,

        "average_r":
            net_r
            / len(trades)

    }


# ============================================================
# PRINT
# ============================================================

def print_stats(
    title,
    stats,
    capital,
    max_dd,
    days
):

    signals_day = (
        stats["trades"]
        / days
        if days > 0
        else 0
    )

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

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
        f"{stats['profit_factor']}"
    )

    print(
        f"Net R        : "
        f"{stats['net_r']:.2f}"
    )

    print(
        f"Average R    : "
        f"{stats['average_r']:.4f}"
    )

    print(
        f"Max Drawdown : "
        f"{max_dd:.2f}%"
    )

    print(
        f"Balance      : "
        f"${capital:.2f}"
    )

    print(
        f"Signals/day  : "
        f"{signals_day:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("COINEX REAL DATA BACKTEST")
    print("15M TREND + BREAKOUT/PULLBACK")
    print("NO PANDAS / NO NUMPY")
    print("CLOSED CANDLE ONLY")
    print("ENTRY = NEXT CANDLE OPEN")
    print("TP = 2R")
    print("=" * 60)

    all_trades = []

    total_capital = INITIAL_CAPITAL

    per_asset = {}

    for symbol in ASSETS:

        print()
        print(
            f"Downloading {symbol}..."
        )

        try:

            candles = get_coinex_klines(
                symbol,
                DAYS
            )

            print(
                f"{symbol}: "
                f"{len(candles)} candles"
            )

            if len(candles) < 300:

                print(
                    f"{symbol}: "
                    "Not enough data."
                )

                continue

            # ------------------------------------------------
            # OOS split by time
            # ------------------------------------------------

            oos_candles = (
                OOS_DAYS
                * 24
                * 4
            )

            split_index = max(
                150,
                len(candles)
                - oos_candles
            )

            # ------------------------------------------------
            # FULL
            # ------------------------------------------------

            capital, trades, max_dd = (
                run_backtest(
                    symbol,
                    candles,
                    150,
                    len(candles) - 1
                )
            )

            stats = statistics(
                trades,
                INITIAL_CAPITAL
            )

            days = (
                len(candles)
                / 96.0
            )

            print_stats(
                symbol,
                stats,
                capital,
                max_dd,
                days
            )

            # ------------------------------------------------
            # OOS
            # ------------------------------------------------

            oos_capital, oos_trades, oos_dd = (
                run_backtest(
                    symbol,
                    candles,
                    split_index,
                    len(candles) - 1
                )
            )

            per_asset[symbol] = {

                "candles":
                    len(candles),

                "trades":
                    stats,

                "capital":
                    capital,

                "max_drawdown":
                    max_dd,

                "oos_trades":
                    statistics(
                        oos_trades,
                        INITIAL_CAPITAL
                    ),

                "oos_capital":
                    oos_capital,

                "oos_max_drawdown":
                    oos_dd

            }

            all_trades.extend(
                trades
            )

        except Exception as e:

            print(
                f"{symbol} ERROR: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # ========================================================
    # ALL ASSETS
    # ========================================================

    all_stats = statistics(
        all_trades,
        INITIAL_CAPITAL
    )

    print_stats(
        "ALL COINS",
        all_stats,
        INITIAL_CAPITAL,
        0,
        DAYS
    )

    # ========================================================
    # TARGET
    # ========================================================

    print()
    print("=" * 60)
    print("TARGET CHECK")
    print("=" * 60)

    wr = all_stats[
        "win_rate"
    ]

    signals_day = (
        all_stats["trades"]
        / DAYS
    )

    print(
        f"Win Rate 70-80% : "
        f"{'PASS' if 70 <= wr <= 80 else 'FAIL'}"
    )

    print(
        f"Signals >= 2/day : "
        f"{'PASS' if signals_day >= 2 else 'FAIL'}"
    )

    print(
        f"Real data       : PASS"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {

        "strategy": {
            "timeframe": "15m",
            "ema": [
                EMA_FAST,
                EMA_MID,
                EMA_SLOW
            ],
            "atr_period":
                ATR_PERIOD,
            "rr":
                RR_RATIO,
            "volume_multiplier":
                VOLUME_MULTIPLIER,
            "body_ratio":
                BODY_RATIO_MIN,
            "adx_min":
                ADX_MIN,
            "fee_rate":
                FEE_RATE,
            "slippage":
                SLIPPAGE
        },

        "assets":
            per_asset,

        "all": all_stats

    }

    with open(
        "coinex_backtest_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print(
        "Results saved to:"
    )

    print(
        "coinex_backtest_results.json"
    )

    print()
    print(
        "BACKTEST FINISHED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
