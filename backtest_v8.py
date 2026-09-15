import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ccxt"]
    )
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V64
# Smart Long/Short Regime + Market Breadth + Dynamic Risk
# ============================================================


exchange = ccxt.lbank({
    "enableRateLimit": True
})


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}


REMOVED_COINS = {
    "NEAR",
    "OP",
    "HYPE",
    "HBAR",
    "AVAX",
    "SUI",
    "PENDLE",
    "TIA",
    "FET",
    "SEI",
    "ARB",
    "DOT",
    "ETC",
    "SHIB",
    "STX",
    "RUNE",
    "MKR",
    "APT",
    "LTC",
    "AR",
    "IMX",
    "PEPE",
    "BONK",
}


SYMBOLS = {
    k: v
    for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}


# ============================================================
# SETTINGS
# ============================================================

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14

INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0

TIMEOUT_CANDLES = 45

EMA_WARMUP = 200


# مالی

INITIAL_CAPITAL = 1000
TRADE_MARGIN = 100
LEVERAGE = 80


# فیلترهای جدید

MIN_BREADTH_BULL = 0.45
MIN_BREADTH_BEAR = 0.35

SHORT_MOMENTUM_LIMIT = -0.035
LONG_MOMENTUM_LIMIT = 0.035


start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)

since_timestamp = int(
    start_date.timestamp() * 1000
)



# ============================================================
# DATA FETCH
# ============================================================


def fetch_symbol_data(symbol):

    all_ohlcv = []

    current_since = since_timestamp

    last_seen = None


    while current_since < exchange.milliseconds():

        batch = None


        for attempt in range(3):

            try:

                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000
                )

                break


            except Exception:

                if attempt == 2:
                    return None


        if not batch:
            break


        last_ts = batch[-1][0]


        if (
            last_seen is not None
            and last_ts <= last_seen
        ):
            return None


        all_ohlcv.extend(batch)

        last_seen = last_ts

        current_since = last_ts + 1


        if len(batch) < 1000:
            break



    if not all_ohlcv:
        return None



    df = pd.DataFrame(
        all_ohlcv,
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


    df.reset_index(
        drop=True,
        inplace=True
    )


    # حذف کندل ناقص

    if len(df) >= 2:

        now = exchange.milliseconds()

        last = int(
            df.iloc[-1]["Date"].timestamp()
            * 1000
        )

        if last + 4*60*60*1000 > now:

            df = df.iloc[:-1].copy()



    if len(df) < EMA_WARMUP + 50:

        return None



    # ATR

    tr1 = (
        df["High"]
        -
        df["Low"]
    )


    tr2 = np.abs(
        df["High"]
        -
        df["Close"].shift(1)
    )


    tr3 = np.abs(
        df["Low"]
        -
        df["Close"].shift(1)
    )


    df["ATR"] = (
        pd.concat(
            [
                tr1,
                tr2,
                tr3
            ],
            axis=1
        )
        .max(axis=1)
        .rolling(
            ATR_PERIOD
        )
        .mean()
    )


    # EMA

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )


    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )


    df["EMA200"] = (
        df["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )


    # Momentum

    df["Mom_Short"] = (
        df["Close"]
        -
        df["Close"].shift(10)
    ) / df["Close"].shift(10)


    df["Mom_Long"] = (
        df["Close"]
        -
        df["Close"].shift(30)
    ) / df["Close"].shift(30)



    df.set_index(
        "Date",
        inplace=True
    )


    return df
    # ============================================================
# LOAD DATA
# ============================================================

processed_data = {}


print("=" * 70)
print("📥 دریافت داده‌ها - HUNTER-V64")
print("=" * 70)


for name, symbol in SYMBOLS.items():

    print(f"📡 {name} ...")

    data = fetch_symbol_data(symbol)

    if data is not None:

        processed_data[name] = data

        print(
            f"✅ {name}: {len(data)} candles"
        )

    else:

        print(
            f"❌ {name}: skipped"
        )


print(
    f"\n✅ Valid symbols: {len(processed_data)}"
)



# ============================================================
# MARKET REGIME
# ============================================================


def get_market_state(ts):

    btc_state = 0

    breadth_bull = 0

    breadth_total = 0



    # BTC leader

    if (
        "BTC" in processed_data
        and
        ts in processed_data["BTC"].index
    ):

        btc = processed_data["BTC"].loc[ts]


        if (
            btc["Close"] > btc["EMA200"]
            and
            btc["EMA20"] > btc["EMA50"]
        ):

            btc_state = 1


        elif (
            btc["Close"] < btc["EMA200"]
            and
            btc["EMA20"] < btc["EMA50"]
        ):

            btc_state = -1



    # Market breadth

    for symbol, df in processed_data.items():

        if ts not in df.index:
            continue


        c = df.loc[ts]


        breadth_total += 1


        if (
            c["Close"] > c["EMA20"]
            and
            c["EMA20"] > c["EMA50"]
            and
            c["Close"] > c["EMA200"]
        ):

            breadth_bull += 1



    breadth = 0


    if breadth_total:

        breadth = (
            breadth_bull
            /
            breadth_total
        )



    # Final regime


    if (
        btc_state == 1
        and
        breadth >= MIN_BREADTH_BULL
    ):

        return "BULL"



    if (
        btc_state == -1
        and
        breadth <= MIN_BREADTH_BEAR
    ):

        return "BEAR"



    return "NEUTRAL"





# ============================================================
# BACKTEST ENGINE
# ============================================================


def run_backtest(data):


    timestamps = sorted(
        {
            t
            for df in data.values()
            for t in df.index
        }
    )


    active_positions = {}

    trades = []



    for ts in timestamps:



        # ================================================
        # EXIT MANAGEMENT
        # ================================================


        closed = []


        for symbol, pos in list(
            active_positions.items()
        ):


            df = data[symbol]


            if ts not in df.index:

                continue



            candle = df.loc[ts]



            # LONG TRAILING

            if pos["side"] == "LONG":


                if (
                    candle["High"]
                    >
                    pos["extreme"]
                ):

                    pos["extreme"] = candle["High"]


                    trail = (
                        pos["extreme"]
                        -
                        TRAILING_ATR_MULTIPLIER
                        *
                        candle["ATR"]
                    )


                    if trail > pos["stop"]:

                        pos["stop"] = trail



                hit_stop = (
                    candle["Low"]
                    <=
                    pos["stop"]
                )



            # SHORT TRAILING

            else:


                if (
                    candle["Low"]
                    <
                    pos["extreme"]
                ):

                    pos["extreme"] = candle["Low"]


                    trail = (
                        pos["extreme"]
                        +
                        TRAILING_ATR_MULTIPLIER
                        *
                        candle["ATR"]
                    )


                    if trail < pos["stop"]:

                        pos["stop"] = trail



                hit_stop = (
                    candle["High"]
                    >=
                    pos["stop"]
                )




            idx = df.index.get_loc(ts)


            timeout = (
                idx
                -
                pos["entry_index"]
            ) >= TIMEOUT_CANDLES



            if hit_stop or timeout:



                if pos["side"] == "LONG":


                    if hit_stop:

                        exit_price = min(
                            pos["stop"],
                            candle["Open"]
                        )

                    else:

                        exit_price = candle["Close"]



                    price_return = (
                        exit_price
                        -
                        pos["entry"]
                    ) / pos["entry"]



                    r_value = (
                        (
                            exit_price
                            -
                            pos["entry"]
                        )
                        /
                        pos["risk"]
                    )



                else:



                    if hit_stop:

                        exit_price = max(
                            pos["stop"],
                            candle["Open"]
                        )

                    else:

                        exit_price = candle["Close"]



                    price_return = (
                        pos["entry"]
                        -
                        exit_price
                    ) / pos["entry"]



                    r_value = (
                        (
                            pos["entry"]
                            -
                            exit_price
                        )
                        /
                        pos["risk"]
                    )



                r_value -= (
                    FEE_RATE
                    *
                    2
                )


                pnl = (
                    TRADE_MARGIN
                    *
                    LEVERAGE
                    *
                    price_return
                )

                pnl -= (
                    TRADE_MARGIN
                    *
                    LEVERAGE
                    *
                    FEE_RATE
                    *
                    2
                )



                trades.append(
                    {
                        "Timestamp": ts,
                        "Symbol": symbol,
                        "Side": pos["side"],
                        "R": r_value,
                        "PnL": pnl,
                        "Outcome":
                            "WIN"
                            if r_value > 0
                            else
                            "LOSS",
                        "ExitOrder":
                            len(trades)
                    }
                )


                closed.append(symbol)



        for s in closed:

            del active_positions[s]



        # ادامه بخش ۳/۴...
                # ================================================
        # ENTRY MANAGEMENT
        # ================================================


        if len(active_positions) >= MAX_POSITIONS:
            continue



        regime = get_market_state(ts)



        if regime == "NEUTRAL":
            continue



        scores = {}



        for symbol, df in data.items():


            if ts not in df.index:
                continue


            mom = df.loc[
                ts,
                "Mom_Long"
            ]


            if not np.isnan(mom):

                scores[symbol] = mom



        if not scores:
            continue



        # مرتب سازی هوشمند

        if regime == "BULL":

            ranked = sorted(
                scores.keys(),
                key=lambda x: scores[x],
                reverse=True
            )

        else:

            ranked = sorted(
                scores.keys(),
                key=lambda x: scores[x]
            )




        for symbol in ranked:



            if len(active_positions) >= MAX_POSITIONS:
                break



            if symbol in active_positions:
                continue



            df = data[symbol]



            i = df.index.get_loc(ts)



            if i < EMA_WARMUP:
                continue



            c = df.iloc[i]



            if (
                pd.isna(c["ATR"])
                or
                c["ATR"] <= 0
            ):
                continue




            # ============================================
            # LONG SIGNAL
            # ============================================


            if regime == "BULL":



                signal = (

                    c["Close"]
                    >
                    c["EMA20"]

                    and

                    c["EMA20"]
                    >
                    c["EMA50"]

                    and

                    c["Close"]
                    >
                    c["EMA200"]

                    and

                    c["Mom_Short"]
                    >
                    0.012

                    and

                    c["Mom_Long"]
                    >
                    LONG_MOMENTUM_LIMIT

                )



                side = "LONG"




            # ============================================
            # SHORT SIGNAL
            # ============================================


            else:



                signal = (


                    c["Close"]
                    <
                    c["EMA20"]


                    and


                    c["EMA20"]
                    <
                    c["EMA50"]


                    and


                    c["Close"]
                    <
                    c["EMA200"]


                    and


                    c["Mom_Short"]
                    <
                    -0.012


                    and


                    c["Mom_Long"]
                    <
                    SHORT_MOMENTUM_LIMIT

                )



                side = "SHORT"




            if not signal:

                continue





            # ============================================
            # ENTRY
            # ============================================


            if side == "LONG":


                entry = (
                    c["Open"]
                    *
                    (
                        1
                        +
                        SLIPPAGE
                    )
                )


                stop = (
                    entry
                    -
                    INITIAL_ATR_MULTIPLIER
                    *
                    c["ATR"]
                )



            else:



                entry = (
                    c["Open"]
                    *
                    (
                        1
                        -
                        SLIPPAGE
                    )
                )


                stop = (
                    entry
                    +
                    INITIAL_ATR_MULTIPLIER
                    *
                    c["ATR"]
                )





            risk = abs(
                entry
                -
                stop
            )



            if risk <= 0:

                continue



            risk_pct = (
                risk
                /
                entry
            )



            # فیلتر ریسک

            if not (
                0.01
                <=
                risk_pct
                <=
                0.04
            ):

                continue




            active_positions[symbol] = {


                "side": side,


                "entry": entry,


                "stop": stop,


                "risk": risk,


                "entry_index": i,


                "extreme": entry


            }



    return pd.DataFrame(trades)





# ================================================
# METRICS
# ================================================


def calculate_metrics(df):


    if df.empty:

        return None



    total = len(df)


    wins = (
        df["Outcome"]
        ==
        "WIN"
    ).sum()


    losses = (
        df["Outcome"]
        ==
        "LOSS"
    ).sum()



    winrate = (
        wins
        /
        total
        *
        100
    )



    net_r = (
        df["R"]
        .sum()
    )



    pnl = (
        df["PnL"]
        .sum()
    )



    profit = (
        df.loc[
            df["R"] > 0,
            "R"
        ]
        .sum()
    )



    loss = abs(
        df.loc[
            df["R"] < 0,
            "R"
        ]
        .sum()
    )



    profit_factor = (

        profit
        /
        loss

        if loss > 0
        else 0

    )



    avg_r = (
        net_r
        /
        total
    )



    longs = df[
        df["Side"]
        ==
        "LONG"
    ]


    shorts = df[
        df["Side"]
        ==
        "SHORT"
    ]



    return {

        "Trades": total,

        "Wins": int(wins),

        "Losses": int(losses),

        "WR": winrate,

        "NetR": net_r,

        "PnL": pnl,

        "PF": profit_factor,

        "AVG_R": avg_r,

        "Longs": len(longs),

        "Shorts": len(shorts)

    }
    # ============================================================
# FINAL REPORT
# ============================================================


def summarize_result(trades):

    print("\n")
    print("=" * 70)
    print("📊 HUNTER-V64 FINAL REPORT")
    print("=" * 70)


    if trades.empty:

        print("⚠️ No trades")

        return



    trades = trades.sort_values(
        [
            "Timestamp",
            "ExitOrder"
        ],
        kind="stable"
    ).reset_index(drop=True)



    metrics = calculate_metrics(
        trades
    )



    print(
        f"🔹 Total Trades : {metrics['Trades']}"
    )

    print(
        f"🟢 Wins         : {metrics['Wins']}"
    )

    print(
        f"🔴 Losses       : {metrics['Losses']}"
    )

    print(
        f"🎯 Win Rate     : {metrics['WR']:.2f}%"
    )

    print(
        f"💰 Net R        : {metrics['NetR']:.2f}R"
    )

    print(
        f"💵 Net PnL      : ${metrics['PnL']:.2f}"
    )

    print(
        f"📈 Profit Factor: {metrics['PF']:.2f}"
    )

    print(
        f"📊 Average R    : {metrics['AVG_R']:.3f}"
    )



    print("\n------------------------------")

    print(
        f"🟢 Long Trades  : {metrics['Longs']}"
    )

    print(
        f"🔴 Short Trades : {metrics['Shorts']}"
    )



    # Long / Short breakdown


    for side in [
        "LONG",
        "SHORT"
    ]:


        temp = trades[
            trades["Side"]
            ==
            side
        ]


        if len(temp) == 0:

            continue



        w = (
            temp["Outcome"]
            ==
            "WIN"
        ).sum()



        wr = (
            w
            /
            len(temp)
            *
            100
        )



        print(
            f"\n{side}:"
        )

        print(
            f"Trades={len(temp)} | "
            f"WR={wr:.2f}% | "
            f"R={temp['R'].sum():.2f}"
        )




    # Loss streak


    max_loss = 0

    current = 0


    for x in trades["Outcome"]:


        if x == "LOSS":

            current += 1

            max_loss = max(
                max_loss,
                current
            )


        else:

            current = 0



    print(
        f"\n❄️ Max Loss Streak: {max_loss}"
    )



    print("=" * 70)







# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":



    print("\n")
    print("=" * 70)

    print(
        "🚀 RUNNING HUNTER-V64"
    )

    print(
        "Smart Long/Short Regime System"
    )

    print("=" * 70)



    print(
        f"Symbols: {len(processed_data)}"
    )

    print(
        f"Timeframe: {TIMEFRAME}"
    )

    print(
        f"Lookback: {LOOKBACK_DAYS} days"
    )



    trades = run_backtest(
        processed_data
    )



    summarize_result(
        trades
    )



    print("\n✨ HUNTER-V64 FINISHED")
