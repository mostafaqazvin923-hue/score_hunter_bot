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
# Golden Core + Market Breadth + Loss Shield
# ============================================================


exchange = ccxt.lbank({
    "enableRateLimit": True
})


# ================= SYMBOLS =================

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
    "NEAR", "OP", "HYPE", "HBAR",
    "AVAX", "SUI", "PENDLE",
    "TIA", "FET", "SEI",
    "ARB", "DOT", "ETC",
    "SHIB", "STX", "RUNE",
    "MKR", "APT", "LTC",
    "AR", "IMX", "PEPE",
    "BONK"
}


SYMBOLS = {
    k: v for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}


# ================= BACKTEST SETTINGS =================

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



# ================= LOSS SHIELD =================

# بعد از ضررهای متوالی ورودها محدود می‌شوند

MAX_LOSS_CHAIN_SOFT = 3
MAX_LOSS_CHAIN_HARD = 5

COOLDOWN_SOFT = 3
COOLDOWN_HARD = 8



# ================= FINANCE =================

INITIAL_CAPITAL = 1000.0

TRADE_MARGIN = 100.0

LEVERAGE = 80.0



# ================= DATA RANGE =================

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)

since_timestamp = int(
    start_date.timestamp() * 1000
)



print("=" * 70)
print(
    "📥 HUNTER-V75 "
    "Golden Core + Loss Shield"
)
print("=" * 70)



processed_data = {}
# ============================================================
# DATA FETCH
# ============================================================


def fetch_symbol_data(lbank_symbol):

    all_ohlcv = []

    current_since = since_timestamp

    last_seen = None


    while current_since < exchange.milliseconds():

        batch = None


        for attempt in range(3):

            try:

                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
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


        if last_seen is not None and last_ts <= last_seen:
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



    # حذف کندل در حال تشکیل

    if len(df) >= 2:

        now_ms = exchange.milliseconds()

        last_ms = int(
            df.iloc[-1]["Date"].timestamp()
            * 1000
        )


        if last_ms + (
            4 * 60 * 60 * 1000
        ) > now_ms:

            df = df.iloc[:-1].copy()



    if len(df) < EMA_WARMUP + 50:

        return None



    # چک گپ دیتا

    deltas = df["Date"].diff().dropna()


    if (
        not deltas.empty
        and deltas.max()
        > pd.Timedelta(
            hours=4,
            minutes=10
        )
    ):

        return None



    # ================= ATR =================


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


    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)



    df["ATR"] = (
        true_range
        .rolling(ATR_PERIOD)
        .mean()
    )



    # ================= EMA =================


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



    # ================= MOMENTUM =================


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



    # فاصله قیمت از EMA200 برای امتیازدهی

    df["EMA200_Distance"] = (
        df["Close"]
        -
        df["EMA200"]
    ) / df["EMA200"]



    df.set_index(
        "Date",
        inplace=True
    )


    return df



# ============================================================
# LOAD ALL SYMBOLS
# ============================================================


for symbol, lbank_symbol in SYMBOLS.items():

    df4h = fetch_symbol_data(
        lbank_symbol
    )


    if df4h is not None:

        processed_data[symbol] = df4h



print(
    f"✅ تعداد نمادهای معتبر: "
    f"{len(processed_data)} "
    f"از {len(SYMBOLS)}"
)


print(
    "⚙️ شروع اجرای بک‌تست HUNTER-V75..."
    
)# ============================================================
# BACKTEST ENGINE
# ============================================================


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


    # کنترل ضررهای متوالی

    loss_chain = 0

    cooldown_counter = 0



    for ts in all_timestamps:


        symbols_to_close = []



        # ====================================================
        # مدیریت پوزیشن های باز
        # ====================================================


        for symbol, pos in list(active_positions.items()):


            df = processed_data[symbol]


            if ts not in df.index:

                continue



            c4h = df.loc[ts]



            # ---------------- LONG ----------------


            if pos["side"] == "LONG":


                if c4h["High"] > pos["highest_price"]:


                    pos["highest_price"] = c4h["High"]


                    new_sl = (
                        pos["highest_price"]
                        -
                        TRAILING_ATR_MULTIPLIER
                        *
                        c4h["ATR"]
                    )


                    if new_sl > pos["stop_loss"]:

                        pos["stop_loss"] = new_sl



                hit_sl = (
                    c4h["Low"]
                    <=
                    pos["stop_loss"]
                )



            # ---------------- SHORT ----------------


            else:


                if c4h["Low"] < pos["lowest_price"]:


                    pos["lowest_price"] = c4h["Low"]


                    new_sl = (
                        pos["lowest_price"]
                        +
                        TRAILING_ATR_MULTIPLIER
                        *
                        c4h["ATR"]
                    )


                    if new_sl < pos["stop_loss"]:

                        pos["stop_loss"] = new_sl



                hit_sl = (
                    c4h["High"]
                    >=
                    pos["stop_loss"]
                )




            curr_i = (
                df.index
                .get_loc(ts)
            )


            candles_held = (
                curr_i
                -
                pos["entry_index"]
            )


            timeout = (
                candles_held
                >=
                TIMEOUT_CANDLES
            )



            if hit_sl or timeout:


                initial_risk = pos["initial_risk"]



                # ================= EXIT PRICE =================


                if pos["side"] == "LONG":


                    exit_p = (
                        min(
                            pos["stop_loss"],
                            c4h["Open"]
                        )
                        if hit_sl
                        else c4h["Close"]
                    )



                    r_real = (
                        (
                            exit_p
                            -
                            pos["entry_price"]
                        )
                        /
                        initial_risk
                    ) - (
                        FEE_RATE * 2
                    )



                    price_return_pct = (
                        exit_p
                        -
                        pos["entry_price"]
                    ) / pos["entry_price"]



                else:


                    exit_p = (
                        max(
                            pos["stop_loss"],
                            c4h["Open"]
                        )
                        if hit_sl
                        else c4h["Close"]
                    )



                    r_real = (
                        (
                            pos["entry_price"]
                            -
                            exit_p
                        )
                        /
                        initial_risk
                    ) - (
                        FEE_RATE * 2
                    )



                    price_return_pct = (
                        pos["entry_price"]
                        -
                        exit_p
                    ) / pos["entry_price"]





                outcome = (
                    "WIN"
                    if r_real > 0
                    else
                    "LOSS"
                )



                # ================= LOSS SHIELD UPDATE =================


                if outcome == "LOSS":

                    loss_chain += 1


                else:

                    loss_chain = 0



                position_notional = (
                    TRADE_MARGIN
                    *
                    LEVERAGE
                )


                dollar_pnl = (
                    position_notional
                    *
                    price_return_pct
                ) - (
                    position_notional
                    *
                    FEE_RATE
                    *
                    2
                )



                all_trades.append(
                    {
                        "Timestamp": ts,
                        "Symbol": symbol,
                        "Side": pos["side"],
                        "Outcome": outcome,
                        "Return": r_real,
                        "Dollar_PnL": dollar_pnl,
                        "ExitOrder": len(all_trades),
                        "LossChain": loss_chain,
                    }
                )



                symbols_to_close.append(symbol)



        # حذف پوزیشن های بسته شده


        for sym in symbols_to_close:

            del active_positions[sym]



        # ====================================================
        # COOLDOWN AFTER LOSING STREAK
        # ====================================================


        if cooldown_counter > 0:

            cooldown_counter -= 1



        if loss_chain >= MAX_LOSS_CHAIN_HARD:

            cooldown_counter = max(
                cooldown_counter,
                COOLDOWN_HARD
            )


        elif loss_chain >= MAX_LOSS_CHAIN_SOFT:

            cooldown_counter = max(
                cooldown_counter,
                COOLDOWN_SOFT
            )



        # اگر cooldown فعال است،
        # فقط مدیریت معاملات باز انجام شود


        if cooldown_counter > 0:

            continue
                    # ====================================================
        # MARKET REGIME + BREADTH FILTER
        # ====================================================


        market_bull = True


        if (
            "BTC" in processed_data
            and ts in processed_data["BTC"].index
        ):

            btc_c = processed_data["BTC"].loc[ts]

            market_bull = (
                btc_c["Close"]
                >
                btc_c["EMA200"]
            )



        # ================= MARKET BREADTH =================


        bullish_count = 0

        total_active = 0



        for symbol, df in processed_data.items():


            if ts in df.index:


                total_active += 1


                if (
                    df.loc[ts, "Close"]
                    >
                    df.loc[ts, "EMA200"]
                ):

                    bullish_count += 1




        breadth_ratio = (
            bullish_count
            /
            total_active
            if total_active > 0
            else 0.5
        )



        # فیلتر جهت بازار

        allow_longs = (
            breadth_ratio >= 0.35
        )


        allow_shorts = (
            breadth_ratio <= 0.65
        )





        # ====================================================
        # BUILD SMART SCORE
        # ====================================================


        current_scores = {}



        for symbol, df in processed_data.items():


            if ts not in df.index:

                continue



            row = df.loc[ts]



            if np.isnan(row["Mom_Long"]):

                continue



            score = (

                row["Mom_Long"] * 0.5

                +

                row["Mom_Short"] * 0.2

                +

                row["EMA200_Distance"] * 0.3

            )



            current_scores[symbol] = score




        if not current_scores:

            continue





        ranked_symbols = sorted(

            current_scores.keys(),

            key=lambda x:
                current_scores[x],

            reverse=True

        )



        # در بازار نزولی برعکس مرتب می‌کنیم

        if not market_bull:


            ranked_symbols = sorted(

                current_scores.keys(),

                key=lambda x:
                    current_scores[x],

                reverse=False

            )





        # ====================================================
        # ENTRY LOGIC
        # ====================================================


        for symbol in ranked_symbols:


            if len(active_positions) >= MAX_POSITIONS:

                break



            if symbol in active_positions:

                continue



            df = processed_data[symbol]



            if ts not in df.index:

                continue



            i = df.index.get_loc(ts)



            if i < EMA_WARMUP + 1:

                continue



            c4h = df.iloc[i]

            prev_c = df.iloc[i-1]





            # ================= LONG =================


            if market_bull:



                if not allow_longs:

                    continue



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

                    prev_c["Low"]
                    <=

                    prev_c["EMA20"]
                    *
                    1.015

                )



                valid_signal = (

                    regime_ok

                    and

                    pullback_ok

                    and

                    c4h["Mom_Short"]
                    >
                    0.012

                    and

                    c4h["Mom_Long"]
                    >
                    0.035

                )



                side = "LONG"





            # ================= SHORT =================


            else:



                if not allow_shorts:

                    continue



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

                    prev_c["High"]
                    >=

                    prev_c["EMA20"]
                    *
                    0.985

                )



                valid_signal = (

                    regime_ok

                    and

                    pullback_ok

                    and

                    c4h["Mom_Short"]
                    <
                    -0.012

                    and

                    c4h["Mom_Long"]
                    <
                    -0.035

                )



                side = "SHORT"
                            # ====================================================
            # OPEN POSITION
            # ====================================================


            if valid_signal:


                if side == "LONG":


                    entry_price = (
                        c4h["Open"]
                        *
                        (1 + SLIPPAGE)
                    )


                    initial_sl = (
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
                        (1 - SLIPPAGE)
                    )


                    initial_sl = (
                        entry_price
                        +
                        INITIAL_ATR_MULTIPLIER
                        *
                        c4h["ATR"]
                    )




                initial_risk = abs(
                    entry_price
                    -
                    initial_sl
                )



                sl_dist_pct = (
                    initial_risk
                    /
                    entry_price
                )



                # جلوگیری از استاپ خیلی نزدیک یا خیلی دور

                if (
                    0.01
                    <=
                    sl_dist_pct
                    <=
                    0.04
                ):



                    active_positions[symbol] = {


                        "side": side,


                        "entry_price":
                            entry_price,


                        "stop_loss":
                            initial_sl,


                        "highest_price":
                            entry_price,


                        "lowest_price":
                            entry_price,


                        "initial_risk":
                            initial_risk,


                        "entry_index":
                            i,

                    }




    return pd.DataFrame(
        all_trades
    )



# ============================================================
# RESULT SUMMARY
# ============================================================


def summarize_result(trades_df):


    print("\n" + "=" * 70)

    print(
        "📊 HUNTER-V75 FINAL REPORT"
    )

    print("=" * 70)



    if trades_df.empty:


        print(
            "⚠️ هیچ معامله‌ای ثبت نشد."
        )

        return




    trades_df = trades_df.sort_values(
        [
            "Timestamp",
            "ExitOrder"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )




    total_trades = len(
        trades_df
    )


    wins = int(
        (
            trades_df["Outcome"]
            ==
            "WIN"
        ).sum()
    )


    losses = int(
        (
            trades_df["Outcome"]
            ==
            "LOSS"
        ).sum()
    )



    win_rate = (
        wins
        /
        total_trades
        *
        100
    )



    net_r = float(
        trades_df["Return"]
        .sum()
    )



    total_pnl = float(
        trades_df["Dollar_PnL"]
        .sum()
    )



    final_capital = (
        INITIAL_CAPITAL
        +
        total_pnl
    )




    # ================= LONG / SHORT =================


    longs = trades_df[
        trades_df["Side"]
        ==
        "LONG"
    ]


    shorts = trades_df[
        trades_df["Side"]
        ==
        "SHORT"
    ]



    def calc_wr(df):

        if len(df) == 0:

            return 0

        return (
            (
                df["Outcome"]
                ==
                "WIN"
            ).sum()
            /
            len(df)
            *
            100
        )



    long_wr = calc_wr(
        longs
    )


    short_wr = calc_wr(
        shorts
    )




    # ================= LOSS STREAK =================


    max_losses = 0

    current_losses = 0


    streaks = []

    temp = 0



    for outcome in trades_df["Outcome"]:


        if outcome == "WIN":


            current_losses = 0


            if temp > 0:

                streaks.append(temp)

                temp = 0



        else:


            current_losses += 1

            temp += 1


            max_losses = max(
                max_losses,
                current_losses
            )



    if temp > 0:

        streaks.append(temp)




    print(
        f"🔹 Total Trades : {total_trades}"
    )


    print(
        f"🟢 Wins         : {wins}"
    )


    print(
        f"🔴 Losses       : {losses}"
    )


    print(
        f"🎯 Win Rate     : {win_rate:.2f}%"
    )


    print(
        f"💰 Net R        : {net_r:.2f}R"
    )


    print(
        f"💵 Net PnL      : ${total_pnl:,.2f}"
    )


    print(
        f"🏦 Capital      : ${final_capital:,.2f}"
    )


    print(
        f"❄️ Max Loss Streak : {max_losses}"
    )



    print("\n------------------------------")


    print(
        f"🟢 Long Trades  : {len(longs)}"
    )


    print(
        f"LONG WR        : {long_wr:.2f}%"
    )


    print(
        f"🔴 Short Trades : {len(shorts)}"
    )


    print(
        f"SHORT WR       : {short_wr:.2f}%"
    )



    print("\nLoss sequences:")

    print(
        streaks
        # ============================================================
# MAIN EXECUTION
# ============================================================


if __name__ == "__main__":


    print("\n")
    print("=" * 70)
    print(
        "🚀 RUNNING HUNTER-V75"
    )
    print(
        "Golden Core + Market Breadth + Loss Shield"
    )
    print("=" * 70)



    df_trades = run_backtest(
        processed_data
    )



    summarize_result(
        df_trades
    )



    print("\n")
    print(
        "✨ HUNTER-V75 FINISHED"
    )
    )
