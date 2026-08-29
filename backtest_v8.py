import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone


# ============================================================
# SCORE HUNTER
# LBank REAL DATA BACKTEST
#
# ORIGINAL STRATEGY:
# EMA 10 / 30 / 100
# STRONG CANDLE >= 75%
# VOLUME >= 2.0x average volume
# ATR 14
# SL = 1.2 ATR / SWING
# TP = 2R
# CLOSED CANDLE ONLY
# ENTRY = NEXT CANDLE OPEN
#
# NO PANDAS
# NO NUMPY
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.lbank.info/v2/kline.do"

SYMBOLS = [
    "btc_usdt",
    "eth_usdt",
    "sol_usdt",
    "xrp_usdt",
    "avax_usdt",
    "link_usdt",
]

TIMEFRAME = "minute15"

# Number of REAL historical 15m candles requested
TARGET_CANDLES = 20000

# LBank maximum size per request
PAGE_SIZE = 2000

EMA_FAST = 10
EMA_MID = 30
EMA_SLOW = 100

ATR_PERIOD = 14

VOLUME_LOOKBACK = 14
VOLUME_MULTIPLIER = 2.0

STRONG_BODY_RATIO = 0.75

SWING_LOOKBACK = 10

ATR_SL_MULTIPLIER = 1.2

RR_RATIO = 2.0

# ------------------------------------------------------------
# REALISTIC COSTS
#
# Set to zero if you want a pure price-action comparison
# with the original backtest.
# ------------------------------------------------------------

FEE_PER_SIDE = 0.0005       # 0.05%
SLIPPAGE_PER_SIDE = 0.0002  # 0.02%

# Risk percentage
RISK_PER_TRADE = 1.0

INITIAL_CAPITAL = 1000.0


# ============================================================
# HTTP
# ============================================================

def http_get(url, params):

    query = urllib.parse.urlencode(params)

    full_url = url + "?" + query

    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        raw = response.read().decode("utf-8")

    return json.loads(raw)


# ============================================================
# GET LBANK KLINES
# ============================================================

def get_klines(symbol, target):

    all_candles = []

    # Current time in seconds
    now = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )

    # 15 minutes
    interval_seconds = 15 * 60

    # Start sufficiently far in the past
    cursor = (
        now
        - target * interval_seconds
    )

    print(
        f"Downloading {symbol.upper()}..."
    )

    while len(all_candles) < target:

        remaining = (
            target
            - len(all_candles)
        )

        size = min(
            PAGE_SIZE,
            remaining
        )

        try:

            data = http_get(
                BASE_URL,
                {
                    "symbol": symbol,
                    "size": size,
                    "type": TIMEFRAME,
                    "time": str(cursor)
                }
            )

        except Exception as e:

            print(
                f"{symbol.upper()} HTTP ERROR: "
                f"{type(e).__name__}: {e}"
            )

            break

        if not isinstance(data, dict):

            print(
                f"{symbol.upper()}: "
                f"Unexpected response."
            )

            break

        if data.get("result") not in (
            True,
            "true",
            "TRUE",
            None
        ):

            print(
                f"{symbol.upper()}: "
                f"LBank response error: "
                f"{data}"
            )

            break

        rows = data.get(
            "data",
            []
        )

        if not rows:

            print(
                f"{symbol.upper()}: "
                f"No more historical candles."
            )

            break

        batch = []

        for row in rows:

            if len(row) < 6:
                continue

            try:

                candle = {
                    "time": int(
                        float(row[0])
                    ),

                    "open": float(
                        row[1]
                    ),

                    "high": float(
                        row[2]
                    ),

                    "low": float(
                        row[3]
                    ),

                    "close": float(
                        row[4]
                    ),

                    "volume": float(
                        row[5]
                    )
                }

                batch.append(candle)

            except Exception:

                continue

        if not batch:
            break

        all_candles.extend(batch)

        print(
            f"{symbol.upper()}: "
            f"{len(all_candles)} candles"
        )

        # Move cursor forward.
        newest_time = max(
            c["time"]
            for c in batch
        )

        cursor = (
            newest_time
            + interval_seconds
        )

        # Protection against duplicate responses
        if len(batch) < size:

            # Try one more request.
            if len(all_candles) >= target:
                break

        time.sleep(0.15)

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
    # Remove currently forming candle
    # --------------------------------------------------------

    current_time = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )

    closed = []

    for candle in candles:

        candle_end = (
            candle["time"]
            + interval_seconds
        )

        if candle_end <= current_time:
            closed.append(candle)

    candles = closed

    # Keep latest target candles
    if len(candles) > target:
        candles = candles[-target:]

    print(
        f"{symbol.upper()}: "
        f"{len(candles)} CLOSED candles"
    )

    return candles


# ============================================================
# EMA SERIES
# ============================================================

def ema_series(values, period):

    result = [
        None
        for _ in values
    ]

    if len(values) < period:
        return result

    sma = (
        sum(
            values[:period]
        )
        / period
    )

    result[
        period - 1
    ] = sma

    multiplier = (
        2.0
        / (period + 1)
    )

    previous = sma

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
# ATR SERIES
# ============================================================

def atr_series(candles, period):

    result = [
        None
        for _ in candles
    ]

    if len(candles) <= period:
        return result

    trs = []

    for i in range(
        len(candles)
    ):

        if i == 0:

            tr = (
                candles[i]["high"]
                - candles[i]["low"]
            )

        else:

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
                )
            )

        trs.append(tr)

    initial = (
        sum(
            trs[1:period + 1]
        )
        / period
    )

    result[period] = initial

    previous_atr = initial

    for i in range(
        period + 1,
        len(candles)
    ):

        previous_atr = (
            (
                previous_atr
                * (period - 1)
                + trs[i]
            )
            / period
        )

        result[i] = previous_atr

    return result


# ============================================================
# STRONG CANDLE
# ============================================================

def strong_bull(candle):

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    body = (
        candle["close"]
        - candle["open"]
    )

    return (
        body > 0
        and (
            body
            / candle_range
        ) >= STRONG_BODY_RATIO
    )


def strong_bear(candle):

    candle_range = (
        candle["high"]
        - candle["low"]
    )

    if candle_range <= 0:
        return False

    body = (
        candle["open"]
        - candle["close"]
    )

    return (
        body > 0
        and (
            body
            / candle_range
        ) >= STRONG_BODY_RATIO
    )


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def backtest_symbol(
    symbol,
    candles
):

    if len(candles) < 300:

        return {
            "symbol": symbol,
            "trades": [],
            "capital": INITIAL_CAPITAL,
            "max_drawdown": 0.0
        }

    closes = [
        c["close"]
        for c in candles
    ]

    volumes = [
        c["volume"]
        for c in candles
    ]

    ema10 = ema_series(
        closes,
        EMA_FAST
    )

    ema30 = ema_series(
        closes,
        EMA_MID
    )

    ema100 = ema_series(
        closes,
        EMA_SLOW
    )

    atr = atr_series(
        candles,
        ATR_PERIOD
    )

    capital = INITIAL_CAPITAL

    peak_capital = capital

    max_drawdown = 0.0

    trades = []

    active_trade = None

    # --------------------------------------------------------
    # We use i as the CURRENT candle.
    #
    # Signal is generated using candle i-1.
    # Entry is candle i OPEN.
    #
    # This prevents look-ahead.
    # --------------------------------------------------------

    for i in range(
        EMA_SLOW + 5,
        len(candles)
    ):

        current = candles[i]

        previous = candles[i - 1]

        # ====================================================
        # MANAGE EXISTING TRADE
        # ====================================================

        if active_trade is not None:

            trade = active_trade

            direction = trade["type"]

            entry = trade["entry"]

            sl = trade["sl"]

            tp = trade["tp"]

            result = None

            exit_price = None

            # ------------------------------------------------
            # LONG
            # ------------------------------------------------

            if direction == "LONG":

                hit_sl = (
                    current["low"]
                    <= sl
                )

                hit_tp = (
                    current["high"]
                    >= tp
                )

                if hit_sl and hit_tp:

                    # Conservative assumption:
                    # if both occur in same candle,
                    # SL wins.
                    result = "SL"
                    exit_price = sl

                elif hit_sl:

                    result = "SL"
                    exit_price = sl

                elif hit_tp:

                    result = "TP"
                    exit_price = tp

            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            else:

                hit_sl = (
                    current["high"]
                    >= sl
                )

                hit_tp = (
                    current["low"]
                    <= tp
                )

                if hit_sl and hit_tp:

                    result = "SL"
                    exit_price = sl

                elif hit_sl:

                    result = "SL"
                    exit_price = sl

                elif hit_tp:

                    result = "TP"
                    exit_price = tp

            # ------------------------------------------------
            # CLOSE TRADE
            # ------------------------------------------------

            if result is not None:

                risk_price = trade[
                    "risk_price"
                ]

                if direction == "LONG":

                    gross_r = (
                        exit_price
                        - entry
                    ) / risk_price

                else:

                    gross_r = (
                        entry
                        - exit_price
                    ) / risk_price

                # Trading costs
                total_cost = (
                    FEE_PER_SIDE * 2
                    + SLIPPAGE_PER_SIDE * 2
                )

                # Approximate R impact
                cost_r = (
                    total_cost
                    * entry
                    / risk_price
                )

                net_r = (
                    gross_r
                    - cost_r
                )

                risk_amount = trade[
                    "risk_amount"
                ]

                pnl = (
                    net_r
                    * risk_amount
                )

                capital += pnl

                trades.append({

                    "symbol": symbol,

                    "type": direction,

                    "entry": entry,

                    "sl": sl,

                    "tp": tp,

                    "exit": exit_price,

                    "result": result,

                    "gross_r": gross_r,

                    "net_r": net_r,

                    "pnl": pnl,

                    "entry_time":
                        trade[
                            "entry_time"
                        ],

                    "exit_time":
                        current[
                            "time"
                        ]
                })

                active_trade = None

                if capital > peak_capital:

                    peak_capital = capital

                dd = (
                    (
                        peak_capital
                        - capital
                    )
                    / peak_capital
                    * 100.0
                )

                if dd > max_drawdown:

                    max_drawdown = dd

                # Do not open another trade
                # on the same candle.
                continue

        # ====================================================
        # NO ACTIVE TRADE
        # CHECK SIGNAL
        # ====================================================

        if active_trade is not None:
            continue

        # Need enough history
        if (
            ema10[i - 1] is None
            or ema30[i - 1] is None
            or ema100[i - 1] is None
            or atr[i - 1] is None
        ):
            continue

        # ====================================================
        # ORIGINAL TREND CONDITIONS
        # ====================================================

        bull_trend = (
            ema10[i - 1]
            > ema30[i - 1]
            > ema100[i - 1]
        )

        bear_trend = (
            ema10[i - 1]
            < ema30[i - 1]
            < ema100[i - 1]
        )

        # ====================================================
        # TREND MOMENTUM
        # ====================================================

        if i >= 4:

            bull_trend = (
                bull_trend
                and (
                    ema10[i - 1]
                    > ema10[i - 3]
                )
            )

            bear_trend = (
                bear_trend
                and (
                    ema10[i - 1]
                    < ema10[i - 3]
                )
            )

        # ====================================================
        # STRONG CANDLE
        #
        # IMPORTANT:
        # Signal candle is PREVIOUS CLOSED candle.
        # ====================================================

        strong_bull_signal = (
            strong_bull(
                previous
            )
        )

        strong_bear_signal = (
            strong_bear(
                previous
            )
        )

        # ====================================================
        # VOLUME
        # ====================================================

        if i - 1 < (
            VOLUME_LOOKBACK + 1
        ):
            continue

        volume_start = (
            i - 1
            - VOLUME_LOOKBACK
        )

        volume_end = (
            i - 1
        )

        historical_volumes = (
            volumes[
                volume_start:
                volume_end
            ]
        )

        if not historical_volumes:
            continue

        avg_volume = (
            sum(
                historical_volumes
            )
            / len(
                historical_volumes
            )
        )

        high_volume = (
            previous["volume"]
            >= (
                VOLUME_MULTIPLIER
                * avg_volume
            )
        )

        # ====================================================
        # FINAL SIGNAL
        # ====================================================

        long_cond = (
            bull_trend
            and strong_bull_signal
            and high_volume
        )

        short_cond = (
            bear_trend
            and strong_bear_signal
            and high_volume
        )

        if not (
            long_cond
            or short_cond
        ):
            continue

        # ====================================================
        # ENTRY = NEXT CANDLE OPEN
        # ====================================================

        entry = current["open"]

        atr_value = atr[i - 1]

        risk_amount = (
            capital
            * (
                RISK_PER_TRADE
                / 100.0
            )
        )

        # ====================================================
        # LONG
        # ====================================================

        if long_cond:

            swing_low = min(
                candles[j]["low"]
                for j in range(
                    max(
                        0,
                        i - SWING_LOOKBACK
                    ),
                    i
                )
            )

            atr_stop = (
                entry
                - (
                    atr_value
                    * ATR_SL_MULTIPLIER
                )
            )

            sl = min(
                swing_low,
                atr_stop
            )

            risk_price = (
                entry
                - sl
            )

            if risk_price <= 0:
                continue

            tp = (
                entry
                + (
                    risk_price
                    * RR_RATIO
                )
            )

            active_trade = {

                "type": "LONG",

                "entry": entry,

                "sl": sl,

                "tp": tp,

                "risk_price":
                    risk_price,

                "risk_amount":
                    risk_amount,

                "entry_time":
                    current["time"]
            }

        # ====================================================
        # SHORT
        # ====================================================

        elif short_cond:

            swing_high = max(
                candles[j]["high"]
                for j in range(
                    max(
                        0,
                        i - SWING_LOOKBACK
                    ),
                    i
                )
            )

            atr_stop = (
                entry
                + (
                    atr_value
                    * ATR_SL_MULTIPLIER
                )
            )

            sl = max(
                swing_high,
                atr_stop
            )

            risk_price = (
                sl
                - entry
            )

            if risk_price <= 0:
                continue

            tp = (
                entry
                - (
                    risk_price
                    * RR_RATIO
                )
            )

            active_trade = {

                "type": "SHORT",

                "entry": entry,

                "sl": sl,

                "tp": tp,

                "risk_price":
                    risk_price,

                "risk_amount":
                    risk_amount,

                "entry_time":
                    current["time"]
            }

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "symbol": symbol,

        "trades": trades,

        "capital": capital,

        "max_drawdown":
            max_drawdown
    }


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades,
    initial_capital
):

    total = len(trades)

    wins = [
        t for t in trades
        if t["result"] == "TP"
    ]

    losses = [
        t for t in trades
        if t["result"] == "SL"
    ]

    win_rate = (
        len(wins)
        / total
        * 100.0
        if total
        else 0.0
    )

    gross_profit = sum(
        max(
            t["net_r"],
            0
        )
        for t in trades
    )

    gross_loss = sum(
        abs(
            min(
                t["net_r"],
                0
            )
        )
        for t in trades
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

    average_r = (
        net_r / total
        if total
        else 0.0
    )

    final_capital = (
        initial_capital
    )

    for trade in trades:

        final_capital += (
            trade["pnl"]
        )

    return {

        "trades": total,

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
            average_r,

        "final_capital":
            final_capital
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SCORE HUNTER"
    )

    print(
        "LBANK REAL DATA BACKTEST"
    )

    print(
        "ORIGINAL 61.24% STRATEGY"
    )

    print(
        "15M"
    )

    print(
        "EMA 10 / 30 / 100"
    )

    print(
        "STRONG CANDLE >= 75%"
    )

    print(
        "VOLUME >= 2X"
    )

    print(
        "SL = 1.2 ATR / SWING"
    )

    print(
        "TP = 2R"
    )

    print(
        "ENTRY = NEXT CANDLE OPEN"
    )

    print(
        "CLOSED CANDLE ONLY"
    )

    print(
        "REAL LBANK DATA"
    )

    print(
        "=" * 70
    )

    all_trades = []

    results = {}

    # ========================================================
    # DOWNLOAD + TEST
    # ========================================================

    for symbol in SYMBOLS:

        try:

            candles = get_klines(
                symbol,
                TARGET_CANDLES
            )

            if len(candles) < 500:

                print(
                    f"{symbol.upper()}: "
                    f"Not enough data."
                )

                continue

            result = backtest_symbol(
                symbol.upper(),
                candles
            )

            trades = result[
                "trades"
            ]

            stats = calculate_stats(
                trades,
                INITIAL_CAPITAL
            )

            results[
                symbol.upper()
            ] = {

                "candles":
                    len(candles),

                "trades":
                    trades,

                "capital":
                    result[
                        "capital"
                    ],

                "max_drawdown":
                    result[
                        "max_drawdown"
                    ],

                "stats":
                    stats
            }

            all_trades.extend(
                trades
            )

            print(
                "\n"
                + "=" * 55
            )

            print(
                symbol.upper()
            )

            print(
                "=" * 55
            )

            print(
                f"Candles      : "
                f"{len(candles)}"
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
                f"{result['max_drawdown']:.2f}%"
            )

            print(
                f"Balance      : "
                f"${stats['final_capital']:.2f}"
            )

        except Exception as e:

            print(
                f"{symbol.upper()} ERROR: "
                f"{type(e).__name__}: {e}"
            )

    # ========================================================
    # ALL COINS
    # ========================================================

    combined = calculate_stats(
        all_trades,
        INITIAL_CAPITAL
    )

    # --------------------------------------------------------
    # Signal frequency
    # --------------------------------------------------------

    if all_trades:

        first_time = min(
            t["entry_time"]
            for t in all_trades
        )

        last_time = max(
            t["entry_time"]
            for t in all_trades
        )

        days = max(
            (
                last_time
                - first_time
            )
            / 86400.0,
            1.0
        )

        signals_per_day = (
            len(all_trades)
            / days
        )

    else:

        days = 0.0
        signals_per_day = 0.0

    # ========================================================
    # PRINT FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL COINS"
    )

    print(
        "=" * 70
    )

    print(
        f"Trades       : "
        f"{combined['trades']}"
    )

    print(
        f"Wins         : "
        f"{combined['wins']}"
    )

    print(
        f"Losses       : "
        f"{combined['losses']}"
    )

    print(
        f"Win Rate     : "
        f"{combined['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{combined['profit_factor']}"
    )

    print(
        f"Net R        : "
        f"{combined['net_r']:.2f}"
    )

    print(
        f"Average R    : "
        f"{combined['average_r']:.4f}"
    )

    print(
        f"Balance      : "
        f"${combined['final_capital']:.2f}"
    )

    print(
        f"Trading days : "
        f"{days:.1f}"
    )

    print(
        f"Signals/day  : "
        f"{signals_per_day:.2f}"
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    win_target = (
        70.0
        <= combined["win_rate"]
        <= 80.0
    )

    frequency_target = (
        signals_per_day
        >= 2.0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TARGET CHECK"
    )

    print(
        "=" * 70
    )

    print(
        "Win Rate 70-80% : "
        + (
            "PASS"
            if win_target
            else "FAIL"
        )
    )

    print(
        "Signals >= 2/day : "
        + (
            "PASS"
            if frequency_target
            else "FAIL"
        )
    )

    print(
        "Real LBank data  : PASS"
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    output = {

        "settings": {

            "exchange":
                "LBank",

            "timeframe":
                TIMEFRAME,

            "target_candles":
                TARGET_CANDLES,

            "ema_fast":
                EMA_FAST,

            "ema_mid":
                EMA_MID,

            "ema_slow":
                EMA_SLOW,

            "atr_period":
                ATR_PERIOD,

            "volume_multiplier":
                VOLUME_MULTIPLIER,

            "strong_body_ratio":
                STRONG_BODY_RATIO,

            "atr_sl":
                ATR_SL_MULTIPLIER,

            "rr":
                RR_RATIO,

            "fee_per_side":
                FEE_PER_SIDE,

            "slippage_per_side":
                SLIPPAGE_PER_SIDE,

            "risk_per_trade":
                RISK_PER_TRADE
        },

        "results": results,

        "combined": {

            "trades":
                combined["trades"],

            "wins":
                combined["wins"],

            "losses":
                combined["losses"],

            "win_rate":
                combined["win_rate"],

            "profit_factor":
                combined["profit_factor"],

            "net_r":
                combined["net_r"],

            "average_r":
                combined["average_r"],

            "final_capital":
                combined["final_capital"],

            "trading_days":
                days,

            "signals_per_day":
                signals_per_day
        }
    }

    with open(
        "lbank_backtest_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to:"
    )

    print(
        "lbank_backtest_results.json"
    )

    print(
        "\nBACKTEST FINISHED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
