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
# HUNTER-V145 — ADAPTIVE MARKET STRUCTURE PULLBACK
# LIVE SAFE / NO LOOKAHEAD / RR 1:2
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


# ===============================
# SETTINGS
# ===============================

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

TRADE_MARGIN = 100
LEVERAGE = 50

DAYS = 365

MAX_OPEN_POSITIONS = 3


# ===============================
# DATA FETCH
# ===============================

def fetch_chunk_data(symbol, start_dt, end_dt):

    since = int((start_dt - timedelta(days=15)).timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    candles = []

    while since < end_ts:

        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe="15m",
                since=since,
                limit=1000
            )

        except Exception as e:
            print("FETCH ERROR:", symbol, e)
            return None


        if not batch:
            break


        candles.extend(batch)

        last = batch[-1][0]

        if last <= since:
            break


        since = last + 1


        if len(batch) < 1000:
            break


        time.sleep(0.2)


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



# ===============================
# ATR
# ===============================

def calculate_atr(df, period=14):

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] -
        df["Close"].shift()
    ).abs()


    low_close = (
        df["Low"] -
        df["Close"].shift()
    ).abs()


    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)


    return tr.rolling(period).mean()



# ===============================
# ADX
# ===============================

def calculate_adx(df, period=14):

    up = df["High"].diff()

    down = -df["Low"].diff()


    plus_dm = np.where(
        (up > down) &
        (up > 0),
        up,
        0
    )


    minus_dm = np.where(
        (down > up) &
        (down > 0),
        down,
        0
    )


    tr = calculate_atr(df, period)


    plus_di = (
        pd.Series(plus_dm, index=df.index)
        .rolling(period)
        .mean()
        /
        tr
        * 100
    )


    minus_di = (
        pd.Series(minus_dm, index=df.index)
        .rolling(period)
        .mean()
        /
        tr
        * 100
    )


    dx = (
        abs(plus_di - minus_di)
        /
        (plus_di + minus_di)
    ) * 100


    return dx.rolling(period).mean()



# ============================================================
# MARKET STRUCTURE PREPARATION
# ============================================================


def prepare_data(df_15m):

    df_15m = df_15m.copy()



    # ==============================
    # 4H TREND REGIME
    # ==============================


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



    df_4h["BULL"] = (
        (df_4h["Close"] > df_4h["EMA200"])
        &
        (df_4h["EMA50"] > df_4h["EMA200"])
        &
        (df_4h["ADX"] > 15)
    )


    df_4h["BEAR"] = (
        (df_4h["Close"] < df_4h["EMA200"])
        &
        (df_4h["EMA50"] < df_4h["EMA200"])
        &
        (df_4h["ADX"] > 15)
    )





    # ==============================
    # 1H STRUCTURE
    # ==============================


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



    # Swing های تایید شده فقط با گذشته
    # بدون center=True


    df_1h["Swing_High"] = (
        df_1h["High"]
        .rolling(6)
        .max()
        .shift(1)
    )


    df_1h["Swing_Low"] = (
        df_1h["Low"]
        .rolling(6)
        .min()
        .shift(1)
    )


    df_1h["Resistance"] = (
        df_1h["Swing_High"]
        .ffill()
    )


    df_1h["Support"] = (
        df_1h["Swing_Low"]
        .ffill()
    )



    # BOS


    df_1h["BOS_UP"] = (
        df_1h["Close"]
        >
        df_1h["Resistance"]
    )


    df_1h["BOS_DOWN"] = (
        df_1h["Close"]
        <
        df_1h["Support"]
    )



    # Liquidity Sweep ساده و بدون آینده


    df_1h["Sweep_Low"] = (
        (df_1h["Low"] < df_1h["Support"])
        &
        (df_1h["Close"] > df_1h["Support"])
    )


    df_1h["Sweep_High"] = (
        (df_1h["High"] > df_1h["Resistance"])
        &
        (df_1h["Close"] < df_1h["Resistance"])
    )





    # ==============================
    # 15M EXECUTION DATA
    # ==============================


    df_15m["ATR"] = calculate_atr(
        df_15m
    )


    df_15m["AVG_VOLUME"] = (
        df_15m["Volume"]
        .rolling(20)
        .mean()
    )


    df_15m["BODY"] = (
        abs(
            df_15m["Close"]
            -
            df_15m["Open"]
        )
    )



    return df_15m, df_1h, df_4h






# ============================================================
# SIGNAL GENERATOR
# ============================================================


def generate_signal(symbol, idx, df_15, df_1h, df_4h):


    current_time = df_15.index[idx]



    # فقط اطلاعات قبل از ورود

    h1 = df_1h[
        df_1h.index < current_time
    ]


    h4 = df_4h[
        df_4h.index < current_time
    ]



    if len(h1) < 20 or len(h4) < 20:

        return None



    last1 = h1.iloc[-1]

    last4 = h4.iloc[-1]


    candle = df_15.iloc[idx-1]



    atr = candle["ATR"]



    if not np.isfinite(atr):

        return None




    score_long = 0

    score_short = 0




    # ------------------------------
    # LONG SCORE
    # ------------------------------


    if last4["BULL"]:
        score_long += 2


    if last1["BOS_UP"]:
        score_long += 2


    if last1["Sweep_Low"]:
        score_long += 1



    if candle["Close"] > candle["Open"]:
        score_long += 1



    if candle["Volume"] > candle["AVG_VOLUME"]:
        score_long += 1




    # ------------------------------
    # SHORT SCORE
    # ------------------------------


    if last4["BEAR"]:
        score_short += 2


    if last1["BOS_DOWN"]:
        score_short += 2


    if last1["Sweep_High"]:
        score_short += 1



    if candle["Close"] < candle["Open"]:
        score_short += 1



    if candle["Volume"] > candle["AVG_VOLUME"]:
        score_short += 1



    # ==============================
    # FINAL ENTRY FILTER
    # ==============================


    if score_long >= 5 and score_short < score_long:


        return {

            "Symbol": symbol,

            "Side": "LONG",

            "Time": current_time,

            "Index": idx,

            "Entry": df_15.iloc[idx]["Open"],

            "ATR": atr,

            "DF": df_15

        }



    if score_short >= 5 and score_short > score_long:


        return {

            "Symbol": symbol,

            "Side": "SHORT",

            "Time": current_time,

            "Index": idx,

            "Entry": df_15.iloc[idx]["Open"],

            "ATR": atr,

            "DF": df_15

        }



    return None





# ============================================================
# TRADE EXECUTION
# ============================================================


def execute_trade(signal):


    entry = signal["Entry"]

    atr = signal["ATR"]

    side = signal["Side"]



    if side == "LONG":


        entry = entry * (1 + SLIPPAGE)

        sl = entry - (atr * 1.5)

        tp = entry + (atr * 3.0)



    else:


        entry = entry * (1 - SLIPPAGE)

        sl = entry + (atr * 1.5)

        tp = entry - (atr * 3.0)



    df = signal["DF"]

    start = signal["Index"]



    outcome = "OPEN_END"

    exit_price = entry

    exit_time = df.index[-1]



    for i in range(start + 1, len(df)):


        candle = df.iloc[i]


        high = candle["High"]

        low = candle["Low"]



        if side == "LONG":



            if low <= sl:

                outcome = "LOSS"

                exit_price = sl

                exit_time = df.index[i]

                break



            if high >= tp:

                outcome = "WIN"

                exit_price = tp

                exit_time = df.index[i]

                break



        else:



            if high >= sl:

                outcome = "LOSS"

                exit_price = sl

                exit_time = df.index[i]

                break



            if low <= tp:

                outcome = "WIN"

                exit_price = tp

                exit_time = df.index[i]

                break




    if outcome == "OPEN_END":

        pnl = 0


    else:


        notional = TRADE_MARGIN * LEVERAGE


        if side == "LONG":

            move = (
                exit_price - entry
            ) / entry


        else:

            move = (
                entry - exit_price
            ) / entry



        pnl = (
            notional * move
        ) - (
            notional * FEE_RATE * 2
        )



    return {


        "Symbol": signal["Symbol"],

        "Side": side,

        "Entry_Time": signal["Time"],

        "Exit_Time": exit_time,

        "Entry": entry,

        "Exit": exit_price,

        "SL": sl,

        "TP": tp,

        "Outcome": outcome,

        "PnL": pnl

    }







# ============================================================
# PORTFOLIO ENGINE
# ============================================================


def run_engine(symbol_dfs):


    signals = []



    # تولید همه سیگنال ها

    for symbol, data in symbol_dfs.items():


        df15, df1h, df4h = data



        for i in range(60, len(df15)):


            sig = generate_signal(

                symbol,

                i,

                df15,

                df1h,

                df4h

            )



            if sig:

                signals.append(sig)




    # مرتب سازی زمانی

    signals.sort(

        key=lambda x: x["Time"]

    )



    trades = []

    active = []

    loss_streak = 0




    for sig in signals:



        current = sig["Time"]



        # حذف پوزیشن های بسته

        active = [

            x for x in active

            if x["Exit_Time"] > current

        ]




        # یک پوزیشن برای هر نماد

        if any(

            x["Symbol"] == sig["Symbol"]

            for x in active

        ):

            continue




        # حداکثر پوزیشن باز

        if len(active) >= MAX_OPEN_POSITIONS:

            continue




        # توقف بعد از ۴ ضرر

        if loss_streak >= 4:

            continue



        trade = execute_trade(sig)



        if trade["Outcome"] == "LOSS":

            loss_streak += 1


        elif trade["Outcome"] == "WIN":

            loss_streak = 0




        active.append(trade)


        trades.append(trade)



    return trades





# ============================================================
# MAIN BACKTEST
# ============================================================


def main():

    print("=" * 80)
    print("HUNTER-V145 ADAPTIVE MARKET STRUCTURE BACKTEST")
    print("=" * 80)



    end_dt = datetime.utcnow()

    start_dt = (
        end_dt
        -
        timedelta(days=DAYS)
    )



    symbol_dfs = {}



    for name, symbol in SYMBOLS.items():


        print(f"Loading {name}...")


        df = fetch_chunk_data(

            symbol,

            start_dt,

            end_dt

        )


        if df is None or len(df) < 500:

            print(
                f"Skipping {name}"
            )

            continue



        df15, df1h, df4h = prepare_data(df)



        symbol_dfs[name] = (

            df15,

            df1h,

            df4h

        )



    if not symbol_dfs:


        print(
            "No data loaded"
        )

        return




    print("=" * 80)

    print(
        "Running engine..."
    )

    print("=" * 80)



    trades = run_engine(

        symbol_dfs

    )



    if not trades:


        print(
            "No trades generated"
        )

        return



    df = pd.DataFrame(trades)



    closed = df[

        df["Outcome"] != "OPEN_END"

    ]



    if len(closed) == 0:


        print(
            "No closed trades"
        )

        return



    total = len(closed)



    wins = len(

        closed[
            closed["Outcome"] == "WIN"
        ]

    )



    losses = len(

        closed[
            closed["Outcome"] == "LOSS"
        ]

    )



    win_rate = (

        wins / total

    ) * 100



    pnl = closed["PnL"].sum()



    avg_win = (

        closed[

            closed["Outcome"] == "WIN"

        ]["PnL"]

        .mean()

    )



    avg_loss = abs(

        closed[

            closed["Outcome"] == "LOSS"

        ]["PnL"]

        .mean()

    )



    profit_factor = (

        closed[

            closed["PnL"] > 0

        ]["PnL"].sum()

        /

        abs(

            closed[

                closed["PnL"] < 0

            ]["PnL"].sum()

        )

    )




    # محاسبه بیشترین ضرر متوالی

    max_loss_streak = 0

    current_loss = 0



    for x in closed["Outcome"]:


        if x == "LOSS":

            current_loss += 1

            max_loss_streak = max(

                max_loss_streak,

                current_loss

            )


        else:

            current_loss = 0




    print("\n")

    print("=" * 80)

    print("HUNTER-V145 PERFORMANCE REPORT")

    print("=" * 80)



    print(
        f"Total Trades: {total}"
    )


    print(
        f"Win Rate: {win_rate:.2f}%"
    )


    print(
        f"Wins: {wins}"
    )


    print(
        f"Losses: {losses}"
    )


    print(
        f"Net PnL: ${pnl:.2f}"
    )


    print(
        f"Profit Factor: {profit_factor:.2f}"
    )


    print(
        f"Average Win: ${avg_win:.2f}"
    )


    print(
        f"Average Loss: ${avg_loss:.2f}"
    )


    print(
        f"Max Losing Streak: {max_loss_streak}"
    )


    print(
        f"Signals Per Day: {total / DAYS:.2f}"
    )



    print("=" * 80)




    print("\n")

    print(
        "SYMBOL PERFORMANCE"
    )



    symbol_report = closed.groupby(

        "Symbol"

    ).agg(


        Trades=("Outcome", "count"),


        Wins=(

            "Outcome",

            lambda x:

            (x == "WIN").sum()

        ),


        PnL=(

            "PnL",

            "sum"

        )


    )



    symbol_report["WinRate"] = (

        symbol_report["Wins"]

        /

        symbol_report["Trades"]

        *

        100

    ).round(2)



    print(symbol_report)



    print("=" * 80)



    print(
        "BACKTEST COMPLETE"
    )




if __name__ == "__main__":

    main()
