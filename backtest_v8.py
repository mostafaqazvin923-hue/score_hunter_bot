import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V75
# Adaptive Risk + Anti Loss Cluster Engine
# ============================================================


exchange = ccxt.lbank({
    "enableRateLimit": True
})


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
    "NEAR","OP","HYPE","HBAR","AVAX","SUI",
    "PENDLE","TIA","FET","SEI","ARB",
    "DOT","ETC","SHIB","STX","RUNE",
    "MKR","APT","LTC","AR","IMX",
    "PEPE","BONK"
}


SYMBOLS = {
    k:v for k,v in SYMBOLS.items()
    if k not in REMOVED_COINS
}


LOOKBACK_DAYS = 365
TIMEFRAME = "4h"


MAX_POSITIONS = 5


# هزینه‌ها
SLIPPAGE = 0.0003
FEE_RATE = 0.0007


# ATR
ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.2


# مدیریت ضرر
TIMEOUT_CANDLES = 35

# بعد از ضرر روی یک کوین
COOLDOWN_CANDLES = 12


EMA_WARMUP = 200


INITIAL_CAPITAL = 1000
TRADE_MARGIN = 100
LEVERAGE = 80



start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp()*1000)



print("="*70)
print("🚀 HUNTER-V75 Adaptive Risk Engine")
print("="*70)



processed_data = {}



def fetch_symbol_data(symbol):

    candles = []

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

            except:

                if attempt == 2:
                    return None


        if not batch:
            break


        last_ts = batch[-1][0]


        if last_seen and last_ts <= last_seen:
            return None


        candles.extend(batch)

        last_seen = last_ts

        current_since = last_ts + 1


        if len(batch)<1000:
            break



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
        "Date",
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


    if len(df)<EMA_WARMUP+50:
        return None



    # ATR

    tr1 = df["High"]-df["Low"]

    tr2 = abs(
        df["High"]-
        df["Close"].shift(1)
    )

    tr3 = abs(
        df["Low"]-
        df["Close"].shift(1)
    )


    df["ATR"] = pd.concat(
        [tr1,tr2,tr3],
        axis=1
    ).max(axis=1).rolling(
        ATR_PERIOD
    ).mean()



    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()


    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()


    df["EMA200"] = df["Close"].ewm(
        span=200,
        adjust=False
    ).mean()



    df["Mom_Short"] = (
        df["Close"]-
        df["Close"].shift(10)
    ) / df["Close"].shift(10)



    df["Mom_Long"] = (
        df["Close"]-
        df["Close"].shift(30)
    ) / df["Close"].shift(30)



    df.set_index(
        "Date",
        inplace=True
    )


    return df



for name,symbol in SYMBOLS.items():

    data = fetch_symbol_data(symbol)

    if data is not None:
        processed_data[name]=data



print(
    f"✅ Valid symbols: {len(processed_data)}"
)
def run_backtest(processed_data):

    all_timestamps = sorted(
        {
            ts
            for df in processed_data.values()
            for ts in df.index
        }
    )


    active_positions = {}

    all_trades = []


    # ثبت آخرین ضرر هر نماد
    symbol_cooldown = {}


    # شمارنده ضرر کلی
    consecutive_losses = 0



    for ts in all_timestamps:


        symbols_to_close = []



        # ====================================================
        # مدیریت معاملات باز
        # ====================================================

        for symbol,pos in list(active_positions.items()):


            df = processed_data[symbol]


            if ts not in df.index:
                continue



            c4h = df.loc[ts]



            # -----------------------------
            # LONG TRAILING
            # -----------------------------

            if pos["side"]=="LONG":


                if c4h["High"] > pos["highest_price"]:


                    pos["highest_price"] = c4h["High"]


                    new_sl = (
                        pos["highest_price"]
                        -
                        TRAILING_ATR_MULTIPLIER*c4h["ATR"]
                    )


                    if new_sl > pos["stop_loss"]:
                        pos["stop_loss"]=new_sl



                hit_sl = (
                    c4h["Low"]
                    <=
                    pos["stop_loss"]
                )



            # -----------------------------
            # SHORT TRAILING
            # -----------------------------

            else:


                if c4h["Low"] < pos["lowest_price"]:


                    pos["lowest_price"]=c4h["Low"]


                    new_sl = (
                        pos["lowest_price"]
                        +
                        TRAILING_ATR_MULTIPLIER*c4h["ATR"]
                    )


                    if new_sl < pos["stop_loss"]:
                        pos["stop_loss"]=new_sl



                hit_sl = (
                    c4h["High"]
                    >=
                    pos["stop_loss"]
                )




            current_index = (
                df.index.get_loc(ts)
            )


            candles_held = (
                current_index -
                pos["entry_index"]
            )



            # خروج زمانی
            timeout = (
                candles_held
                >=
                TIMEOUT_CANDLES
            )



            # خروج زود هنگام معامله بی‌جان
            dead_trade = False

            if candles_held >= 12:


                if pos["side"]=="LONG":

                    if (
                        c4h["Close"]
                        <
                        pos["entry_price"]
                    ):
                        dead_trade=True


                else:

                    if (
                        c4h["Close"]
                        >
                        pos["entry_price"]
                    ):
                        dead_trade=True





            if hit_sl or timeout or dead_trade:



                if pos["side"]=="LONG":


                    exit_price = (
                        min(
                            pos["stop_loss"],
                            c4h["Open"]
                        )
                        if hit_sl
                        else
                        c4h["Close"]
                    )


                    pnl_pct = (
                        exit_price -
                        pos["entry_price"]
                    ) / pos["entry_price"]



                    r_real = (
                        (
                        exit_price -
                        pos["entry_price"]
                        )
                        /
                        pos["initial_risk"]
                    )



                else:


                    exit_price = (
                        max(
                            pos["stop_loss"],
                            c4h["Open"]
                        )
                        if hit_sl
                        else
                        c4h["Close"]
                    )


                    pnl_pct = (
                        pos["entry_price"]
                        -
                        exit_price
                    ) / pos["entry_price"]



                    r_real = (
                        (
                        pos["entry_price"]
                        -
                        exit_price
                        )
                        /
                        pos["initial_risk"]
                    )




                r_real -= (
                    FEE_RATE*2
                )



                outcome = (
                    "WIN"
                    if r_real>0
                    else
                    "LOSS"
                )



                if outcome=="LOSS":

                    symbol_cooldown[symbol]=(
                        current_index
                        +
                        COOLDOWN_CANDLES
                    )


                    consecutive_losses += 1


                else:

                    consecutive_losses=0





                notional = (
                    TRADE_MARGIN*
                    LEVERAGE
                )


                dollar_pnl = (
                    notional*pnl_pct
                    -
                    notional*FEE_RATE*2
                )




                all_trades.append(
                    {

                    "Timestamp":ts,

                    "Symbol":symbol,

                    "Side":pos["side"],

                    "Outcome":outcome,

                    "Return":r_real,

                    "Dollar_PnL":dollar_pnl,

                    "ExitOrder":len(all_trades)

                    }
                )


                symbols_to_close.append(symbol)





        for s in symbols_to_close:

            del active_positions[s]





        # ====================================================
        # وضعیت کلی بازار
        # ====================================================


        btc_bull=True


        if (
            "BTC" in processed_data
            and
            ts in processed_data["BTC"].index
        ):


            btc=processed_data["BTC"].loc[ts]


            btc_bull = (
                btc["Close"]
                >
                btc["EMA200"]
            )




        # ====================================================
        # Market Breadth Dynamic
        # ====================================================


        bullish=0
        total=0



        for symbol,df in processed_data.items():

            if ts in df.index:

                total+=1


                if (
                    df.loc[ts,"Close"]
                    >
                    df.loc[ts,"EMA200"]
                ):
                    bullish+=1




        breadth = (
            bullish/total
            if total>0
            else .5
        )



        # بازار ضعیف = سختگیری بیشتر

        if breadth < .35:

            max_allowed_positions=2


        elif breadth > .70:

            max_allowed_positions=MAX_POSITIONS


        else:

            max_allowed_positions=3




        # جلوگیری از ورود هنگام زنجیره ضرر شدید

        if consecutive_losses >= 5:

            max_allowed_positions=2



        current_scores={}



        for symbol,df in processed_data.items():

            if ts in df.index:


                score=df.loc[
                    ts,
                    "Mom_Long"
                ]


                if not np.isnan(score):

                    current_scores[symbol]=score




        if not current_scores:
            continue




        ranked_symbols=sorted(
            current_scores.keys(),
            key=lambda x:current_scores[x],
            reverse=btc_bull
        )
                # ====================================================
        # ساخت پوزیشن‌های جدید
        # ====================================================


        for symbol in ranked_symbols:


            if len(active_positions) >= max_allowed_positions:
                break



            if symbol in active_positions:
                continue



            df = processed_data[symbol]



            if ts not in df.index:
                continue



            i = df.index.get_loc(ts)



            if i < EMA_WARMUP + 5:
                continue




            # بررسی Cooldown

            if symbol in symbol_cooldown:


                if i < symbol_cooldown[symbol]:
                    continue




            c4h = df.iloc[i]

            prev = df.iloc[i-1]




            # ====================================================
            # تعیین سمت بازار
            # ====================================================


            if btc_bull:


                # فقط لانگ‌های با کیفیت

                regime_ok = (

                    c4h["Close"]
                    >
                    c4h["EMA20"]

                    and

                    c4h["EMA20"]
                    >
                    c4h["EMA50"]

                    and

                    c4h["Close"]
                    >
                    c4h["EMA200"]

                )



                pullback_ok = (

                    prev["Low"]
                    <=
                    prev["EMA20"]*1.02

                )



                momentum_ok = (

                    c4h["Mom_Short"]
                    >
                    0.012

                    and

                    c4h["Mom_Long"]
                    >
                    0.035

                )



                # در بازار خیلی ضعیف سختگیری بیشتر

                if breadth < 0.45:


                    momentum_ok = (

                        c4h["Mom_Short"]
                        >
                        0.018

                        and

                        c4h["Mom_Long"]
                        >
                        0.045

                    )



                valid_signal = (

                    regime_ok

                    and

                    pullback_ok

                    and

                    momentum_ok

                )



                side="LONG"




            else:


                # ====================================================
                # SHORT FILTER
                # ====================================================


                regime_ok = (

                    c4h["Close"]
                    <
                    c4h["EMA20"]

                    and

                    c4h["EMA20"]
                    <
                    c4h["EMA50"]

                    and

                    c4h["Close"]
                    <
                    c4h["EMA200"]

                )



                pullback_ok = (

                    prev["High"]
                    >=
                    prev["EMA20"]*0.98

                )



                momentum_ok = (

                    c4h["Mom_Short"]
                    <
                    -0.012

                    and

                    c4h["Mom_Long"]
                    <
                    -0.035

                )




                # شورت‌ها کمی سخت‌تر

                if breadth > 0.55:


                    momentum_ok = (

                        c4h["Mom_Short"]
                        <
                        -0.018

                        and

                        c4h["Mom_Long"]
                        <
                        -0.045

                    )




                valid_signal = (

                    regime_ok

                    and

                    pullback_ok

                    and

                    momentum_ok

                )



                side="SHORT"







            if not valid_signal:
                continue




            # ====================================================
            # جلوگیری از ورودهای همبسته
            # ====================================================


            same_side_count=0


            for p in active_positions.values():

                if p["side"]==side:

                    same_side_count += 1



            if same_side_count >= 3:
                continue






            # ====================================================
            # ساخت قیمت ورود و استاپ
            # ====================================================


            if side=="LONG":


                entry_price = (
                    c4h["Open"]
                    *
                    (1+SLIPPAGE)
                )


                stop_loss = (

                    entry_price

                    -
                    INITIAL_ATR_MULTIPLIER
                    *
                    c4h["ATR"]

                )



            else:


                entry_price = (

                    c4h["Open"]
                    *
                    (1-SLIPPAGE)

                )


                stop_loss = (

                    entry_price

                    +
                    INITIAL_ATR_MULTIPLIER
                    *
                    c4h["ATR"]

                )





            initial_risk = abs(
                entry_price-stop_loss
            )



            sl_percent = (
                initial_risk /
                entry_price
            )



            # حذف استاپ‌های خیلی کوچک یا خیلی بزرگ

            if not (
                0.012
                <=
                sl_percent
                <=
                0.045
            ):

                continue





            active_positions[symbol]={


                "side":side,


                "entry_price":entry_price,


                "stop_loss":stop_loss,


                "highest_price":entry_price,


                "lowest_price":entry_price,


                "initial_risk":initial_risk,


                "entry_index":i

            }
            def summarize_result(trades_df):

    print("\n" + "="*70)
    print("📊 HUNTER-V75 FINAL REPORT")
    print("="*70)


    if trades_df.empty:

        print("⚠️ هیچ معامله‌ای ثبت نشد")
        return



    trades_df = trades_df.sort_values(
        ["Timestamp","ExitOrder"],
        kind="stable"
    ).reset_index(drop=True)



    total = len(trades_df)


    wins = (
        trades_df["Outcome"]
        ==
        "WIN"
    ).sum()


    losses = (
        trades_df["Outcome"]
        ==
        "LOSS"
    ).sum()



    winrate = (
        wins/total*100
    )



    net_r = (
        trades_df["Return"]
        .sum()
    )



    pnl = (
        trades_df["Dollar_PnL"]
        .sum()
    )



    final_capital = (
        INITIAL_CAPITAL+pnl
    )




    # Profit Factor

    gross_profit = (
        trades_df.loc[
            trades_df["Dollar_PnL"]>0,
            "Dollar_PnL"
        ].sum()
    )


    gross_loss = abs(
        trades_df.loc[
            trades_df["Dollar_PnL"]<0,
            "Dollar_PnL"
        ].sum()
    )


    profit_factor = (

        gross_profit/gross_loss

        if gross_loss>0

        else 0

    )




    # زنجیره ضرر

    max_loss_streak=0

    current=0


    loss_list=[]

    temp=0



    for x in trades_df["Outcome"]:


        if x=="LOSS":

            current+=1

            temp+=1


            max_loss_streak=max(
                max_loss_streak,
                current
            )


        else:


            current=0


            if temp>0:

                loss_list.append(temp)

                temp=0




    if temp>0:

        loss_list.append(temp)






    # تفکیک معاملات


    long_df = trades_df[
        trades_df["Side"]=="LONG"
    ]


    short_df = trades_df[
        trades_df["Side"]=="SHORT"
    ]



    long_wr = (

        (
            long_df["Outcome"]
            =="WIN"
        ).sum()
        /
        len(long_df)
        *
        100

        if len(long_df)>0

        else 0

    )



    short_wr = (

        (
            short_df["Outcome"]
            =="WIN"
        ).sum()
        /
        len(short_df)
        *
        100

        if len(short_df)>0

        else 0

    )





    print(
        f"🔹 Total Trades : {total}"
    )


    print(
        f"🟢 Wins         : {wins}"
    )


    print(
        f"🔴 Losses       : {losses}"
    )


    print(
        f"🎯 Win Rate     : {winrate:.2f}%"
    )


    print(
        f"💰 Net R        : {net_r:.2f}R"
    )


    print(
        f"💵 Net PnL      : ${pnl:,.2f}"
    )


    print(
        f"📈 Profit Factor : {profit_factor:.2f}"
    )


    print(
        f"🏦 Final Capital : ${final_capital:,.2f}"
    )


    print("\n------------------------------")


    print(
        f"LONG  : {len(long_df)} trades | WR={long_wr:.2f}%"
    )


    print(
        f"SHORT : {len(short_df)} trades | WR={short_wr:.2f}%"
    )


    print(
        f"\n❄️ Max Loss Streak : {max_loss_streak}"
    )


    print(
        "\nLoss sequences:"
    )


    print(
        ", ".join(
            map(str,loss_list)
        )
    )



    print("="*70)






if __name__=="__main__":


    trades = run_backtest(
        processed_data
    )


    summarize_result(
        trades
    )


    print(
        "\n✨ HUNTER-V75 FINISHED"
    )
