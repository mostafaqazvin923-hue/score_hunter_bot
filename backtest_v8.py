import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone


# ============================================================
# COINEX FUTURES REAL DATA BACKTEST
#
# 15M TREND + BREAKOUT / PULLBACK
# REAL COINEX DATA
# PAGINATED HISTORICAL DOWNLOAD
# NO PANDAS
# NO NUMPY
# CLOSED CANDLE ONLY
# ENTRY = NEXT CANDLE OPEN
#
# Indicators:
# EMA 20 / 50 / 200
# RSI 14
# ATR 14
# ADX 14
# Volume confirmation
#
# Risk:
# SL = max(structure, ATR)
# TP = 2R
#
# ============================================================


API_URL = "https://api.coinex.com/v2/futures/kline"


# ============================================================
# SETTINGS
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "AVAXUSDT",
    "LINKUSDT",
]

PERIOD = "15min"

# Six months approximately
DAYS = 180

INITIAL_CAPITAL = 1000.0

RISK_PER_TRADE = 0.01

RR = 2.0

ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

BREAKOUT_LOOKBACK = 20
PULLBACK_LOOKBACK = 8

VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.20

ADX_MIN = 20.0

ATR_SL_MULTIPLIER = 1.20

MAX_BARS = 18000

FEE_RATE = 0.0005

SLIPPAGE = 0.0002

REQUEST_LIMIT = 1000

REQUEST_SLEEP = 0.15


# ============================================================
# HTTP
# ============================================================

def http_get(url, params):

    query = urllib.parse.urlencode(params)

    full_url = url + "?" + query

    req = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(
        req,
        timeout=30
    ) as response:

        raw = response.read().decode(
            "utf-8"
        )

    return json.loads(raw)


# ============================================================
# COINEX DATA
# ============================================================

def download_coinex_klines(
    symbol,
    days=DAYS
):

    now_ms = int(
        time.time() * 1000
    )

    start_ms = (
        now_ms
        - days
        * 24
        * 60
        * 60
        * 1000
    )

    end_ms = now_ms

    all_candles = []

    print(
        f"Downloading {symbol}..."
    )

    while start_ms < end_ms:

        params = {
            "market": symbol,
            "period": PERIOD,
            "limit": REQUEST_LIMIT,
            "start_time": start_ms,
            "end_time": end_ms
        }

        try:

            data = http_get(
                API_URL,
                params
            )

        except Exception as e:

            print(
                f"{symbol} request error: "
                f"{type(e).__name__}: {e}"
            )

            time.sleep(2)

            continue

        if data.get("code") != 0:

            raise RuntimeError(
                f"{symbol}: "
                f"{data.get('message')}"
            )

        rows = data.get(
            "data",
            []
        )

        if not rows:

            break

        batch = []

        for row in rows:

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

            batch.append(candle)

        all_candles.extend(batch)

        oldest = min(
            c["time"]
            for c in batch
        )

        newest = max(
            c["time"]
            for c in batch
        )

        # Move backward in time.
        new_end = oldest - 1

        if new_end >= end_ms:

            break

        end_ms = new_end

        print(
            f"{symbol}: "
            f"{len(all_candles)} candles"
        )

        if len(all_candles) >= MAX_BARS:

            break

        time.sleep(
            REQUEST_SLEEP
        )

    # --------------------------------------------------------
    # Remove duplicates
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
    # Remove current candle
    # --------------------------------------------------------

    if len(candles) > 1:

        candles = candles[:-1]

    print(
        f"{symbol}: "
        f"{len(candles)} closed candles"
    )

    return candles


# ============================================================
# EMA
# ============================================================

def calculate_ema(
    values,
    period
):

    if len(values) < period:

        return [None] * len(values)

    result = [None] * len(values)

    seed = sum(
        values[:period]
    ) / period

    result[
        period - 1
    ] = seed

    multiplier = (
        2.0
        / (period + 1)
    )

    previous = seed

    for i in range(
        period,
        len(values)
    ):

        current = (
            (
                values[i]
                - previous
            )
            * multiplier
            + previous
        )

        result[i] = current

        previous = current

    return result


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    candles,
    period
):

    result = [
        None
    ] * len(candles)

    if len(candles) <= period:

        return result

    trs = []

    for i in range(
        1,
        len(candles)
    ):

        current = candles[i]

        previous = candles[
            i - 1
        ]

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

        return result

    value = (
        sum(
            trs[:period]
        )
        / period
    )

    result[period] = value

    for j in range(
        period,
        len(trs)
    ):

        value = (
            (
                value
                * (period - 1)
                + trs[j]
            )
            / period
        )

        result[j + 1] = value

    return result


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    candles,
    period
):

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
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
        )

    avg_gain = (
        sum(
            gains[:period]
        )
        / period
    )

    avg_loss = (
        sum(
            losses[:period]
        )
        / period
    )

    def rsi_value():

        if avg_loss == 0:

            return 100.0

        rs = (
            avg_gain
            / avg_loss
        )

        return (
            100
            - 100
            / (1 + rs)
        )

    result[period] = rsi_value()

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

            result[
                i + 1
            ] = 100.0

        else:

            rs = (
                avg_gain
                / avg_loss
            )

            result[
                i + 1
            ] = (
                100
                - 100
                / (1 + rs)
            )

    return result


# ============================================================
# ADX
# ============================================================

def calculate_adx(
    candles,
    period
):

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

        c = candles[i]
        p = candles[i - 1]

        tr_value = max(
            c["high"] - c["low"],

            abs(
                c["high"]
                - p["close"]
            ),

            abs(
                c["low"]
                - p["close"]
            )
        )

        up = (
            c["high"]
            - p["high"]
        )

        down = (
            p["low"]
            - c["low"]
        )

        plus = (
            up
            if (
                up > down
                and up > 0
            )
            else 0.0
        )

        minus = (
            down
            if (
                down > up
                and down > 0
            )
            else 0.0
        )

        tr.append(
            tr_value
        )

        plus_dm.append(
            plus
        )

        minus_dm.append(
            minus
        )

    atr_value = (
        sum(
            tr[:period]
        )
        / period
    )

    plus_value = (
        sum(
            plus_dm[:period]
        )
        / period
    )

    minus_value = (
        sum(
            minus_dm[:period]
        )
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

            dx.append(0.0)

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

            dx.append(0.0)

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
        sum(
            dx[:period]
        )
        / period
    )

    index = (
        period
        + period
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

        index = (
            i
            + period
        )

        if index < len(result):

            result[index] = (
                adx_value
            )

    return result


# ============================================================
# UTILITY
# ============================================================

def average(
    values
):

    if not values:

        return 0.0

    return (
        sum(values)
        / len(values)
    )


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(
    candles,
    i,
    ema20,
    ema50,
    ema200,
    rsi,
    atr,
    adx
):

    if i < 250:

        return None

    if (
        ema20[i] is None
        or ema50[i] is None
        or ema200[i] is None
        or rsi[i] is None
        or atr[i] is None
        or adx[i] is None
    ):

        return None

    candle = candles[i]

    previous = candles[
        i - 1
    ]

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    bullish_trend = (
        ema20[i]
        > ema50[i]
        > ema200[i]
    )

    bearish_trend = (
        ema20[i]
        < ema50[i]
        < ema200[i]
    )

    # --------------------------------------------------------
    # EMA slope
    # --------------------------------------------------------

    if i < 5:

        return None

    bull_slope = (
        ema20[i]
        > ema20[i - 5]
        and ema50[i]
        > ema50[i - 5]
    )

    bear_slope = (
        ema20[i]
        < ema20[i - 5]
        and ema50[i]
        < ema50[i - 5]
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    if adx[i] < ADX_MIN:

        return None

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if i < (
        VOLUME_LOOKBACK
        + 1
    ):

        return None

    volumes = [
        candles[j]["volume"]
        for j in range(
            i - VOLUME_LOOKBACK,
            i
        )
    ]

    avg_volume = average(
        volumes
    )

    if avg_volume <= 0:

        return None

    volume_ok = (
        candle["volume"]
        >=
        avg_volume
        * VOLUME_MULTIPLIER
    )

    # --------------------------------------------------------
    # Previous structure
    # --------------------------------------------------------

    start = max(
        0,
        i - BREAKOUT_LOOKBACK
    )

    previous_high = max(
        candles[j]["high"]
        for j in range(
            start,
            i
        )
    )

    previous_low = min(
        candles[j]["low"]
        for j in range(
            start,
            i
        )
    )

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    bullish_breakout = (
        candle["close"]
        > previous_high
    )

    bearish_breakout = (
        candle["close"]
        < previous_low
    )

    # --------------------------------------------------------
    # Candle strength
    # --------------------------------------------------------

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:

        return None

    body = abs(
        candle["close"]
        - candle["open"]
    )

    body_ratio = (
        body
        / candle_range
    )

    bullish_candle = (
        candle["close"]
        > candle["open"]
        and body_ratio >= 0.55
    )

    bearish_candle = (
        candle["close"]
        < candle["open"]
        and body_ratio >= 0.55
    )

    # --------------------------------------------------------
    # Pullback
    # --------------------------------------------------------

    pullback_long = False
    pullback_short = False

    for j in range(
        max(
            0,
            i - PULLBACK_LOOKBACK
        ),
        i
    ):

        c = candles[j]

        if (
            c["low"]
            <= ema20[j]
            and c["close"]
            > ema20[j]
        ):

            pullback_long = True

        if (
            c["high"]
            >= ema20[j]
            and c["close"]
            < ema20[j]
        ):

            pullback_short = True

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_breakout = (
        bullish_trend
        and bull_slope
        and bullish_breakout
        and bullish_candle
        and volume_ok
        and 52 <= rsi[i] <= 72
    )

    long_pullback = (
        bullish_trend
        and bull_slope
        and pullback_long
        and bullish_candle
        and volume_ok
        and 50 <= rsi[i] <= 68
    )

    if (
        long_breakout
        or long_pullback
    ):

        setup = (
            "BREAKOUT"
            if long_breakout
            else "PULLBACK"
        )

        return {
            "direction": "LONG",
            "setup": setup
        }

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_breakout = (
        bearish_trend
        and bear_slope
        and bearish_breakout
        and bearish_candle
        and volume_ok
        and 28 <= rsi[i] <= 48
    )

    short_pullback = (
        bearish_trend
        and bear_slope
        and pullback_short
        and bearish_candle
        and volume_ok
        and 32 <= rsi[i] <= 50
    )

    if (
        short_breakout
        or short_pullback
    ):

        setup = (
            "BREAKOUT"
            if short_breakout
            else "PULLBACK"
        )

        return {
            "direction": "SHORT",
            "setup": setup
        }

    return None


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    candles
):

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = calculate_ema(
        closes,
        EMA_FAST
    )

    ema50 = calculate_ema(
        closes,
        EMA_MID
    )

    ema200 = calculate_ema(
        closes,
        EMA_SLOW
    )

    atr = calculate_atr(
        candles,
        ATR_PERIOD
    )

    rsi = calculate_rsi(
        candles,
        RSI_PERIOD
    )

    adx = calculate_adx(
        candles,
        ADX_PERIOD
    )

    capital = INITIAL_CAPITAL

    peak = capital

    max_drawdown = 0.0

    trades = []

    active = None

    for i in range(
        250,
        len(candles) - 1
    ):

        candle = candles[i]

        # ----------------------------------------------------
        # Manage existing trade
        # ----------------------------------------------------

        if active is not None:

            result = None
            exit_price = None

            if active[
                "direction"
            ] == "LONG":

                if (
                    candle["low"]
                    <= active["sl"]
                ):

                    result = "SL"

                    exit_price = (
                        active["sl"]
                    )

                elif (
                    candle["high"]
                    >= active["tp"]
                ):

                    result = "TP"

                    exit_price = (
                        active["tp"]
                    )

            else:

                if (
                    candle["high"]
                    >= active["sl"]
                ):

                    result = "SL"

                    exit_price = (
                        active["sl"]
                    )

                elif (
                    candle["low"]
                    <= active["tp"]
                ):

                    result = "TP"

                    exit_price = (
                        active["tp"]
                    )

            if result:

                if active[
                    "direction"
                ] == "LONG":

                    gross_r = (
                        exit_price
                        - active["entry"]
                    ) / active["risk_distance"]

                else:

                    gross_r = (
                        active["entry"]
                        - exit_price
                    ) / active["risk_distance"]

                # ------------------------------------------------
                # Fees
                # ------------------------------------------------

                fee_cost = (
                    FEE_RATE
                    * 2.0
                )

                net_r = (
                    gross_r
                    - (
                        fee_cost
                        / active["risk_percent"]
                    )
                )

                pnl = (
                    active[
                        "risk_amount"
                    ]
                    * net_r
                )

                capital += pnl

                trades.append({
                    "entry_time":
                        active["entry_time"],

                    "exit_time":
                        candle["time"],

                    "direction":
                        active["direction"],

                    "setup":
                        active["setup"],

                    "entry":
                        active["entry"],

                    "sl":
                        active["sl"],

                    "tp":
                        active["tp"],

                    "result":
                        result,

                    "gross_r":
                        gross_r,

                    "net_r":
                        net_r,

                    "pnl":
                        pnl
                })

                active = None

                if capital > peak:

                    peak = capital

                dd = (
                    (
                        peak
                        - capital
                    )
                    / peak
                    * 100
                )

                max_drawdown = max(
                    max_drawdown,
                    dd
                )

                continue

        # ----------------------------------------------------
        # Do not open another position
        # ----------------------------------------------------

        if active is not None:

            continue

        # ----------------------------------------------------
        # Signal on CLOSED candle
        # ----------------------------------------------------

        signal = generate_signal(
            candles,
            i,
            ema20,
            ema50,
            ema200,
            rsi,
            atr,
            adx
        )

        if signal is None:

            continue

        # ----------------------------------------------------
        # NEXT CANDLE OPEN
        # ----------------------------------------------------

        next_candle = candles[
            i + 1
        ]

        entry = (
            next_candle["open"]
        )

        # Slippage
        if signal[
            "direction"
        ] == "LONG":

            entry *= (
                1.0
                + SLIPPAGE
            )

        else:

            entry *= (
                1.0
                - SLIPPAGE
            )

        atr_value = atr[i]

        if atr_value is None:

            continue

        # ----------------------------------------------------
        # Structure
        # ----------------------------------------------------

        structure_start = max(
            0,
            i - 10
        )

        if signal[
            "direction"
        ] == "LONG":

            swing_low = min(
                candles[j]["low"]
                for j in range(
                    structure_start,
                    i + 1
                )
            )

            atr_sl = (
                entry
                - atr_value
                * ATR_SL_MULTIPLIER
            )

            sl = min(
                swing_low,
                atr_sl
            )

            risk_distance = (
                entry
                - sl
            )

            if risk_distance <= 0:

                continue

            tp = (
                entry
                + risk_distance
                * RR
            )

        else:

            swing_high = max(
                candles[j]["high"]
                for j in range(
                    structure_start,
                    i + 1
                )
            )

            atr_sl = (
                entry
                + atr_value
                * ATR_SL_MULTIPLIER
            )

            sl = max(
                swing_high,
                atr_sl
            )

            risk_distance = (
                sl
                - entry
            )

            if risk_distance <= 0:

                continue

            tp = (
                entry
                - risk_distance
                * RR
            )

        risk_amount = (
            capital
            * RISK_PER_TRADE
        )

        active = {

            "direction":
                signal[
                    "direction"
                ],

            "setup":
                signal[
                    "setup"
                ],

            "entry":
                entry,

            "sl":
                sl,

            "tp":
                tp,

            "risk_distance":
                risk_distance,

            "risk_amount":
                risk_amount,

            "risk_percent":
                RISK_PER_TRADE,

            "entry_time":
                next_candle[
                    "time"
                ]
        }

    return {
        "capital":
            capital,

        "trades":
            trades,

        "max_drawdown":
            max_drawdown
    }


# ============================================================
# STATISTICS
# ============================================================

def statistics(
    trades,
    initial_capital,
    final_capital,
    max_drawdown,
    days
):

    wins = [
        t
        for t in trades
        if t["result"] == "TP"
    ]

    losses = [
        t
        for t in trades
        if t["result"] == "SL"
    ]

    win_rate = (
        len(wins)
        / len(trades)
        * 100
        if trades
        else 0
    )

    gross_profit = sum(
        max(
            t["pnl"],
            0
        )
        for t in trades
    )

    gross_loss = abs(
        sum(
            min(
                t["pnl"],
                0
            )
            for t in trades
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = None

    net_r = sum(
        t["net_r"]
        for t in trades
    )

    avg_r = (
        net_r
        / len(trades)
        if trades
        else 0
    )

    signals_day = (
        len(trades)
        / days
    )

    return {
        "trades":
            len(trades),

        "wins":
            len(wins),

        "losses":
            len(losses),

        "win_rate":
            win_rate,

        "profit_factor":
            profit_factor,

        "net_r":
            net_r,

        "average_r":
            avg_r,

        "max_drawdown":
            max_drawdown,

        "initial":
            initial_capital,

        "final":
            final_capital,

        "signals_day":
            signals_day
    }


# ============================================================
# PRINT
# ============================================================

def print_stats(
    title,
    stats
):

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

    pf = stats[
        "profit_factor"
    ]

    if pf is None:

        print(
            "Profit Factor: None"
        )

    else:

        print(
            f"Profit Factor: "
            f"{pf:.2f}"
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
        f"{stats['max_drawdown']:.2f}%"
    )

    print(
        f"Balance      : "
        f"${stats['final']:.2f}"
    )

    print(
        f"Signals/day  : "
        f"{stats['signals_day']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "COINEX REAL DATA BACKTEST v10"
    )
    print(
        "15M TREND + BREAKOUT / PULLBACK"
    )
    print(
        "PAGINATED HISTORICAL DATA"
    )
    print(
        "NO PANDAS / NO NUMPY"
    )
    print(
        "CLOSED CANDLE ONLY"
    )
    print(
        "ENTRY = NEXT CANDLE OPEN"
    )
    print(
        "TP = 2R"
    )
    print("=" * 60)

    all_trades = []

    per_coin = {}

    for symbol in SYMBOLS:

        try:

            candles = (
                download_coinex_klines(
                    symbol
                )
            )

            if len(candles) < 1000:

                print(
                    f"{symbol}: "
                    f"Not enough data: "
                    f"{len(candles)}"
                )

                continue

            result = run_backtest(
                candles
            )

            trades = result[
                "trades"
            ]

            stats = statistics(
                trades,
                INITIAL_CAPITAL,
                result["capital"],
                result[
                    "max_drawdown"
                ],
                DAYS
            )

            per_coin[
                symbol
            ] = stats

            all_trades.extend(
                trades
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

    all_stats = statistics(
        all_trades,
        INITIAL_CAPITAL,
        INITIAL_CAPITAL,
        0.0,
        DAYS
    )

    # Reconstruct combined equity approximately
    # from trade PNL sequence.

    balance = INITIAL_CAPITAL
    peak = balance
    combined_dd = 0.0

    ordered_trades = sorted(
        all_trades,
        key=lambda x: x[
            "exit_time"
        ]
    )

    for trade in ordered_trades:

        balance += trade[
            "pnl"
        ]

        if balance > peak:

            peak = balance

        dd = (
            (
                peak
                - balance
            )
            / peak
            * 100
        )

        combined_dd = max(
            combined_dd,
            dd
        )

    all_stats = statistics(
        all_trades,
        INITIAL_CAPITAL,
        balance,
        combined_dd,
        DAYS
    )

    print_stats(
        "ALL COINS",
        all_stats
    )

    # ========================================================
    # TARGET
    # ========================================================

    print()
    print("=" * 60)
    print("TARGET CHECK")
    print("=" * 60)

    wr_pass = (
        70
        <= all_stats[
            "win_rate"
        ]
        <= 80
    )

    freq_pass = (
        all_stats[
            "signals_day"
        ]
        >= 2
    )

    print(
        "Win Rate 70-80% : "
        + (
            "PASS"
            if wr_pass
            else "FAIL"
        )
    )

    print(
        "Signals >= 2/day : "
        + (
            "PASS"
            if freq_pass
            else "FAIL"
        )
    )

    print(
        "Both targets     : "
        + (
            "PASS"
            if (
                wr_pass
                and freq_pass
            )
            else "FAIL"
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "settings": {
            "period":
                PERIOD,

            "days":
                DAYS,

            "rr":
                RR,

            "risk_per_trade":
                RISK_PER_TRADE,

            "fee_rate":
                FEE_RATE,

            "slippage":
                SLIPPAGE,

            "adx_min":
                ADX_MIN,

            "volume_multiplier":
                VOLUME_MULTIPLIER
        },

        "all_coins":
            all_stats,

        "per_coin":
            per_coin,

        "trades":
            all_trades
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


if __name__ == "__main__":

    main()
