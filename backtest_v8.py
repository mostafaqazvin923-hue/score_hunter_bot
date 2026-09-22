import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt


# ============================================================
# HUNTER-V144 — INSTITUTIONAL PIVOT PULLBACK
# LIVE-SAFE / NO LOOKAHEAD / RR 1:2
# ============================================================


exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {
        "defaultType": "swap"
    }
})


SYMBOLS = {
    "CRV": "CRV/USDT",
    "DOGE": "DOGE/USDT",
    "ICP": "ICP/USDT",
    "APT": "APT/USDT",
    "PENDLE": "PENDLE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "NEAR": "NEAR/USDT",
    "SEI": "SEI/USDT",
    "XLM": "XLM/USDT",
    "ADA": "ADA/USDT",
    "BNB": "BNB/USDT",
    "SOL": "SOL/USDT",
    "ETH": "ETH/USDT",
}


SLIPPAGE = 0.0003
FEE_RATE = 0.0007

TRADE_MARGIN = 100.0
LEVERAGE = 50.0

DAYS = 365

MAX_OPEN_POSITIONS = 3

PIVOT_LEFT = 2
PIVOT_RIGHT = 2



# ============================================================
# DATA FETCH
# ============================================================


def fetch_chunk_data(symbol, start_dt, end_dt):

    since_ts = int((start_dt - timedelta(days=15)).timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    candles = []
    current_since = since_ts

    try:

        while current_since < end_ts:

            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe="15m",
                since=current_since,
                limit=1000
            )

            if not batch:
                break

            candles.extend(batch)

            last_ts = batch[-1][0]

            if last_ts <= current_since:
                break

            current_since = last_ts + 1

            if len(batch) < 1000:
                break

            if last_ts >= end_ts:
                break

            time.sleep(0.2)


    except Exception as e:

        print(f"ERROR {symbol}: {e}")
        return None


    if not candles:
        return None


    df = pd.DataFrame(
        candles,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )


    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms"
    )


    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]


    df.dropna(inplace=True)

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )


    df.set_index(
        "Date",
        inplace=True
    )


    df = df[
        (df.index >= start_dt) &
        (df.index <= end_dt)
    ]


    return df




# ============================================================
# ADX
# ============================================================


def calculate_adx(df, period=14):

    df = df.copy()


    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        )
    )


    up_move = df["High"] - df["High"].shift(1)

    down_move = df["Low"].shift(1) - df["Low"]


    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0
    )


    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0
    )


    tr_smooth = (
        pd.Series(df["TR"])
        .ewm(alpha=1/period, adjust=False)
        .mean()
    )


    plus_smooth = (
        pd.Series(plus_dm)
        .ewm(alpha=1/period, adjust=False)
        .mean()
    )


    minus_smooth = (
        pd.Series(minus_dm)
        .ewm(alpha=1/period, adjust=False)
        .mean()
    )


    plus_di = 100 * plus_smooth / tr_smooth

    minus_di = 100 * minus_smooth / tr_smooth


    dx = (
        abs(plus_di - minus_di)
        /
        (plus_di + minus_di)
    ) * 100


    adx = (
        dx
        .ewm(alpha=1/period, adjust=False)
        .mean()
    )


    return adx




# ============================================================
# REAL CONFIRMED PIVOTS
# NO REPAINT
# ============================================================


def calculate_confirmed_pivots(df):

    df = df.copy()


    df["Pivot_High"] = False
    df["Pivot_Low"] = False


    highs = df["High"].values
    lows = df["Low"].values


    for i in range(
        PIVOT_LEFT,
        len(df) - PIVOT_RIGHT
    ):

        left_high = highs[
            i-PIVOT_LEFT:i
        ]

        right_high = highs[
            i+1:i+PIVOT_RIGHT+1
        ]


        left_low = lows[
            i-PIVOT_LEFT:i
        ]

        right_low = lows[
            i+1:i+PIVOT_RIGHT+1
        ]


        if (
            highs[i] > max(left_high)
            and
            highs[i] > max(right_high)
        ):
            df.iloc[i,
                    df.columns.get_loc("Pivot_High")] = True



        if (
            lows[i] < min(left_low)
            and
            lows[i] < min(right_low)
        ):
            df.iloc[i,
                    df.columns.get_loc("Pivot_Low")] = True



    # فقط بعد از تایید سمت راست قابل استفاده است

    df["Confirmed_High"] = (
        df["High"]
        .where(df["Pivot_High"])
        .shift(PIVOT_RIGHT)
    )


    df["Confirmed_Low"] = (
        df["Low"]
        .where(df["Pivot_Low"])
        .shift(PIVOT_RIGHT)
    )


    df["Last_Swing_High"] = df["Confirmed_High"].ffill()
    df["Last_Swing_Low"] = df["Confirmed_Low"].ffill()

    return df




# ============================================================
# PREPARE MULTI TIMEFRAME DATA
# ============================================================


def prepare_data(df_15m):

    df_15m = df_15m.copy()


    # -------------------------------
    # 4H MARKET REGIME
    # -------------------------------

    df_4h = (
        df_15m
        .resample("4h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
        .dropna()
    )


    df_4h["EMA50"] = (
        df_4h["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )


    df_4h["EMA200"] = (
        df_4h["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )


    df_4h["ADX"] = calculate_adx(df_4h)


    df_4h["Bull_Regime"] = (
        (df_4h["Close"] > df_4h["EMA200"]) &
        (df_4h["EMA50"] > df_4h["EMA200"]) &
        (df_4h["ADX"] > 18)
    )


    df_4h["Bear_Regime"] = (
        (df_4h["Close"] < df_4h["EMA200"]) &
        (df_4h["EMA50"] < df_4h["EMA200"]) &
        (df_4h["ADX"] > 18)
    )



    # -------------------------------
    # 1H STRUCTURE
    # -------------------------------

    df_1h = (
        df_15m
        .resample("1h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
        .dropna()
    )


    # Pivot تایید شده بدون ریپینت

    df_1h = calculate_confirmed_pivots(df_1h)



    # BOS واقعی

    df_1h["BOS_UP"] = (
        df_1h["Close"] >
        df_1h["Last_Swing_High"]
    )


    df_1h["BOS_DOWN"] = (
        df_1h["Close"] <
        df_1h["Last_Swing_Low"]
    )



    # -------------------------------
    # 15M ENTRY LAYER
    # -------------------------------


    df_15m["ATR"] = (
        (df_15m["High"] - df_15m["Low"])
        .rolling(14)
        .mean()
    )


    df_15m["Body"] = (
        abs(
            df_15m["Close"]
            -
            df_15m["Open"]
        )
    )


    df_15m["Avg_Body"] = (
        df_15m["Body"]
        .rolling(20)
        .mean()
    )


    df_15m["Avg_Volume"] = (
        df_15m["Volume"]
        .rolling(20)
        .mean()
    )



    return (
        df_15m,
        df_1h,
        df_4h
    )





# ============================================================
# SIGNAL GENERATION
# ============================================================


def generate_signal(
        symbol,
        idx,
        df_15,
        df_1h,
        df_4h
):


    t = df_15.index[idx]


    candle = df_15.iloc[idx]

    previous = df_15.iloc[idx-1]



    # فقط اطلاعات بسته شده قبلی

    h1 = df_1h[
        df_1h.index < t
    ]

    h4 = df_4h[
        df_4h.index < t
    ]


    if len(h1) < 20 or len(h4) < 5:
        return None



    regime_long = bool(
        h4.iloc[-1]["Bull_Regime"]
    )


    regime_short = bool(
        h4.iloc[-1]["Bear_Regime"]
    )


    bos_long = bool(
        h1.iloc[-1]["BOS_UP"]
    )


    bos_short = bool(
        h1.iloc[-1]["BOS_DOWN"]
    )



    support = h1.iloc[-1]["Last_Swing_Low"]

    resistance = h1.iloc[-1]["Last_Swing_High"]



    if pd.isna(support) or pd.isna(resistance):
        return None



    # -------------------------------
    # Pullback Logic
    # -------------------------------


    pullback_long = (
        previous["Low"]
        <=
        support * 1.003
    )


    pullback_short = (
        previous["High"]
        >=
        resistance * 0.997
    )



    # -------------------------------
    # Confirmation Candle
    # -------------------------------


    volume_confirm_long = (
        previous["Volume"]
        >
        previous["Avg_Volume"] * 1.1
    )


    volume_confirm_short = (
        previous["Volume"]
        >
        previous["Avg_Volume"] * 1.1
    )


    bullish_candle = (
        previous["Close"]
        >
        previous["Open"]
    )


    bearish_candle = (
        previous["Close"]
        <
        previous["Open"]
    )



    long_signal = (
        regime_long
        and
        bos_long
        and
        pullback_long
        and
        volume_confirm_long
        and
        bullish_candle
    )


    short_signal = (
        regime_short
        and
        bos_short
        and
        pullback_short
        and
        volume_confirm_short
        and
        bearish_candle
    )



    if long_signal:

        return {
            "Symbol": symbol,
            "Side": "LONG",
            "Time": t,
            "Entry": candle["Open"] * (1 + SLIPPAGE),
            "ATR": previous["ATR"]
        }



    if short_signal:

        return {
            "Symbol": symbol,
            "Side": "SHORT",
            "Time": t,
            "Entry": candle["Open"] * (1 - SLIPPAGE),
            "ATR": previous["ATR"]
        }



    return None





# ============================================================
# POSITION ENGINE
# ============================================================


def simulate_trade(signal, df_15, start_idx):


    side = signal["Side"]

    entry = signal["Entry"]

    atr = signal["ATR"]


    if not np.isfinite(atr):
        return None



    # -------------------------------
    # Fixed Risk Reward 1:2
    # -------------------------------


    if side == "LONG":

        sl = entry - (1.5 * atr)

        tp = entry + (3.0 * atr)


    else:

        sl = entry + (1.5 * atr)

        tp = entry - (3.0 * atr)



    outcome = None

    exit_price = None

    exit_time = None



    # بررسی فقط کندل‌های بعد از ورود

    for i in range(
        start_idx + 1,
        len(df_15)
    ):

        candle = df_15.iloc[i]


        high = candle["High"]

        low = candle["Low"]



        if side == "LONG":


            # اولویت SL برای حالت برخورد همزمان

            if low <= sl:

                outcome = "LOSS"

                exit_price = sl

                exit_time = df_15.index[i]

                break



            if high >= tp:

                outcome = "WIN"

                exit_price = tp

                exit_time = df_15.index[i]

                break



        else:



            if high >= sl:

                outcome = "LOSS"

                exit_price = sl

                exit_time = df_15.index[i]

                break



            if low <= tp:

                outcome = "WIN"

                exit_price = tp

                exit_time = df_15.index[i]

                break



    # اگر هنوز باز باشد

    if outcome is None:

        outcome = "OPEN"



        exit_price = df_15.iloc[-1]["Close"]

        exit_time = df_15.index[-1]



    return {


        "Symbol": signal["Symbol"],

        "Side": side,

        "Entry_Time": signal["Time"],

        "Exit_Time": exit_time,

        "Entry": entry,

        "Exit": exit_price,

        "SL": sl,

        "TP": tp,

        "Outcome": outcome

    }





# ============================================================
# PORTFOLIO SIMULATION
# ============================================================


def run_portfolio(symbol_dfs):


    events = []



    # تولید تمام سیگنال‌ها

    for symbol, data in symbol_dfs.items():


        df_15, df_1h, df_4h = data


        for i in range(
            50,
            len(df_15)
        ):


            signal = generate_signal(
                symbol,
                i,
                df_15,
                df_1h,
                df_4h
            )


            if signal:

                signal["Index"] = i

                signal["df_15"] = df_15

                events.append(signal)



    # مرتب‌سازی زمانی

    events.sort(
        key=lambda x:x["Time"]
    )



    trades = []


    active_positions = []


    consecutive_losses = 0



    for event in events:



        current_time = event["Time"]



        # حذف معاملات بسته شده

        active_positions = [

            p for p in active_positions

            if p["Exit_Time"] > current_time

        ]



        # محدودیت یک معامله روی هر ارز

        if any(

            p["Symbol"] == event["Symbol"]

            for p in active_positions

        ):

            continue



        # حداکثر ۳ پوزیشن باز

        if len(active_positions) >= MAX_OPEN_POSITIONS:

            continue



        # توقف بعد از ۴ ضرر

        if consecutive_losses >= 4:

            continue




        result = simulate_trade(

            event,

            event["df_15"],

            event["Index"]

        )



        if result is None:

            continue



        # محاسبه سود زیان


        if result["Outcome"] != "OPEN":


            notional = (
                TRADE_MARGIN *
                LEVERAGE
            )


            if result["Side"] == "LONG":

                price_change = (

                    result["Exit"]
                    -
                    result["Entry"]

                ) / result["Entry"]


            else:


                price_change = (

                    result["Entry"]
                    -
                    result["Exit"]

                ) / result["Entry"]



            pnl = (

                notional *
                price_change

            ) - (

                notional *
                FEE_RATE *
                2

            )


        else:

            pnl = 0




        result["PnL"] = pnl



        if result["Outcome"] == "LOSS":

            consecutive_losses += 1


        elif result["Outcome"] == "WIN":

            consecutive_losses = 0




        active_positions.append(result)


        trades.append(result)



    return trades





# ============================================================
# REPORTING & MAIN
# ============================================================


def print_report(trades):


    if not trades:

        print("No trades generated")

        return



    df = pd.DataFrame(trades)



    closed = df[
        df["Outcome"] != "OPEN"
    ]



    if len(closed) == 0:

        print("No closed trades")

        return



    total = len(closed)



    wins = closed[
        closed["Outcome"] == "WIN"
    ]



    losses = closed[
        closed["Outcome"] == "LOSS"
    ]



    win_rate = (

        len(wins)
        /
        total
        *
        100

    )



    pnl = closed["PnL"].sum()



    gross_profit = wins["PnL"].sum()

    gross_loss = abs(
        losses["PnL"].sum()
    )


    profit_factor = (

        gross_profit /
        gross_loss

        if gross_loss > 0

        else 0

    )



    # محاسبه استریک ضرر

    max_loss_streak = 0

    current = 0


    for x in closed["Outcome"]:


        if x == "LOSS":

            current += 1

            max_loss_streak = max(
                max_loss_streak,
                current
            )


        else:

            current = 0




    print("="*80)

    print("HUNTER-V144 PERFORMANCE REPORT")

    print("="*80)

    print(
        f"Total Trades : {total}"
    )

    print(
        f"Win Rate : {win_rate:.2f}%"
    )

    print(
        f"Profit Factor : {profit_factor:.2f}"
    )

    print(
        f"Net PnL : ${pnl:.2f}"
    )

    print(
        f"Max Losing Streak : {max_loss_streak}"
    )


    print("="*80)



    print("\nSYMBOL PERFORMANCE")


    stats = closed.groupby(
        "Symbol"
    ).agg(

        Trades=("Outcome","count"),

        Wins=(
            "Outcome",
            lambda x:
            (x=="WIN").sum()
        ),

        PnL=(
            "PnL",
            "sum"
        )

    )



    stats["WinRate"] = (

        stats["Wins"]
        /
        stats["Trades"]
        *
        100

    ).round(2)



    print(stats)



# ============================================================
# MAIN
# ============================================================


def main():


    print("="*80)

    print(
        "HUNTER-V144 BACKTEST START"
    )

    print("="*80)



    end = datetime.utcnow()


    start = end - timedelta(
        days=DAYS
    )



    symbol_dfs = {}



    for symbol, market in SYMBOLS.items():


        print(
            f"Loading {symbol}..."
        )


        df = fetch_chunk_data(

            market,

            start,

            end

        )



        if df is None:

            continue



        if len(df) < 500:

            continue




        df_15, df_1h, df_4h = prepare_data(
            df
        )



        symbol_dfs[symbol] = (

            df_15,

            df_1h,

            df_4h

        )




    if not symbol_dfs:


        print(
            "No market data"
        )

        return




    print(
        "Running engine..."
    )



    trades = run_portfolio(
        symbol_dfs
    )



    print_report(
        trades
    )




if __name__ == "__main__":

    main()
