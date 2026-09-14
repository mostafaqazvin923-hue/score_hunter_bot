import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V47
# ============================================================
# پایه: HUNTER-V44 / V46
#
# تغییر جدید V47:
# اگر حداقل 2 معامله در یک timestamp با LOSS بسته شوند،
# ورودهای جدید برای تعداد مشخصی کندل متوقف می‌شود.
#
# سه تست:
#   BASE   = بدون Cluster Control
#   V47-A  = Cluster Cooldown = 2 candles
#   V47-B  = Cluster Cooldown = 4 candles
#
# نکته:
# پوزیشن‌های باز هرگز به خاطر Cluster Control بسته نمی‌شوند.
# SL / Trailing / Exit / Fee / Entry منطق اصلی تغییر نکرده است.
# ============================================================


# ============================================================
# اتصال به LBank
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True
})


# ============================================================
# Universe
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


# ============================================================
# حذف نمادهای ضعیف قبلی
# ============================================================

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
# تنظیمات اصلی
# ============================================================

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14

TRAILING_ATR_MULTIPLIER = 2.0
INITIAL_ATR_MULTIPLIER = 1.8

TIMEOUT_CANDLES = 45

EMA_WARMUP = 200


# ============================================================
# تاریخ شروع
# ============================================================

start_date = datetime.now() - timedelta(
    days=LOOKBACK_DAYS
)

since_timestamp = int(
    start_date.timestamp() * 1000
)


# ============================================================
# دریافت داده از LBank
# ============================================================

print("=" * 68)
print("📥 دریافت داده‌ها - HUNTER-V47")
print("=" * 68)


def fetch_symbol_data(lbank_symbol):

    all_ohlcv = []

    current_since = since_timestamp

    last_seen = None

    while current_since < exchange.milliseconds():

        batch = None

        # ----------------------------------------------------
        # Retry API
        # ----------------------------------------------------

        for attempt in range(3):

            try:

                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000,
                )

                break

            except Exception as e:

                if attempt == 2:

                    print(
                        f"⚠️ دریافت ناقص {lbank_symbol}: {e}"
                    )

                    return None

        if not batch:
            break

        first_ts = batch[0][0]
        last_ts = batch[-1][0]

        # ----------------------------------------------------
        # جلوگیری از loop pagination
        # ----------------------------------------------------

        if (
            last_seen is not None
            and last_ts <= last_seen
        ):

            print(
                f"⚠️ pagination متوقف شد: "
                f"{lbank_symbol}"
            )

            return None

        all_ohlcv.extend(batch)

        last_seen = last_ts

        current_since = last_ts + 1

        if len(batch) < 1000:
            break

    if not all_ohlcv:
        return None

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
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
            "Volume",
        ]
    ]

    # --------------------------------------------------------
    # پاکسازی
    # --------------------------------------------------------

    df.dropna(
        inplace=True
    )

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True,
    )

    df.sort_values(
        "Date",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    # ========================================================
    # حذف آخرین کندل ناقص
    # ========================================================

    if len(df) >= 2:

        now_ms = exchange.milliseconds()

        last_ms = int(
            df.iloc[-1]["Date"].timestamp() * 1000
        )

        candle_ms = (
            4
            * 60
            * 60
            * 1000
        )

        if last_ms + candle_ms > now_ms:

            df = df.iloc[:-1].copy()

    # ========================================================
    # حداقل داده
    # ========================================================

    if len(df) < EMA_WARMUP + 50:

        return None

    # ========================================================
    # کنترل Gap
    # ========================================================

    deltas = (
        df["Date"]
        .diff()
        .dropna()
    )

    if (
        not deltas.empty
        and deltas.max()
        > pd.Timedelta(
            hours=4,
            minutes=10
        )
    ):

        print(
            f"⚠️ gap بزرگ در "
            f"{lbank_symbol}؛ "
            f"نماد حذف شد."
        )

        return None

    # ========================================================
    # ATR
    # ========================================================

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = np.abs(
        df["High"]
        - df["Close"].shift(1)
    )

    tr3 = np.abs(
        df["Low"]
        - df["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .rolling(
            ATR_PERIOD
        )
        .mean()
    )

    # ========================================================
    # EMA
    # ========================================================

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

    # ========================================================
    # Momentum
    # ========================================================

    df["Mom_Short"] = (
        (
            df["Close"]
            - df["Close"].shift(10)
        )
        /
        df["Close"].shift(10)
    )

    df["Mom_Long"] = (
        (
            df["Close"]
            - df["Close"].shift(30)
        )
        /
        df["Close"].shift(30)
    )

    # ========================================================
    # Index
    # ========================================================

    df.set_index(
        "Date",
        inplace=True
    )

    return df


# ============================================================
# دریافت تمام نمادها
# ============================================================

processed_data = {}


for symbol, lbank_symbol in SYMBOLS.items():

    df4h = fetch_symbol_data(
        lbank_symbol
    )

    if df4h is not None:

        processed_data[symbol] = df4h

        print(
            f"✅ {symbol}: "
            f"{len(df4h)} کندل"
        )

    else:

        print(
            f"❌ {symbol}: حذف شد"
        )


print(
    f"\n✅ تعداد نمادهای معتبر: "
    f"{len(processed_data)} "
    f"از {len(SYMBOLS)}"
)


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    processed_data,
    cluster_cooldown_candles=0
):

    # ========================================================
    # تمام timestampها
    # ========================================================

    all_timestamps = sorted(
        {
            ts
            for df in processed_data.values()
            for ts in df.index
        }
    )

    # ========================================================
    # وضعیت پوزیشن‌ها
    # ========================================================

    active_positions = {}

    all_trades = []

    # ========================================================
    # V47 Cluster Cooldown
    # ========================================================

    cluster_cooldown_remaining = 0

    # ========================================================
    # Loop بازار
    # ========================================================

    for ts in all_timestamps:

        # ====================================================
        # مدیریت معاملات باز
        # ====================================================

        symbols_to_close = []

        # تعداد LOSSهایی که در همین timestamp بسته می‌شوند
        losses_this_timestamp = 0

        # ----------------------------------------------------
        # بررسی پوزیشن‌های باز
        # ----------------------------------------------------

        for symbol, pos in list(
            active_positions.items()
        ):

            df = processed_data[symbol]

            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            # =================================================
            # Trailing Stop
            # =================================================

            if (
                c4h["High"]
                > pos["highest_price"]
            ):

                pos["highest_price"] = (
                    c4h["High"]
                )

                new_trailing_sl = (
                    pos["highest_price"]
                    -
                    TRAILING_ATR_MULTIPLIER
                    *
                    c4h["ATR"]
                )

                if (
                    new_trailing_sl
                    > pos["stop_loss"]
                ):

                    pos["stop_loss"] = (
                        new_trailing_sl
                    )

            # =================================================
            # Stop Loss
            # =================================================

            hit_sl = (
                c4h["Low"]
                <= pos["stop_loss"]
            )

            # =================================================
            # Timeout
            # =================================================

            curr_i = df.index.get_loc(
                ts
            )

            candles_held = (
                curr_i
                -
                pos["entry_index"]
            )

            is_timeout = (
                candles_held
                >= TIMEOUT_CANDLES
            )

            # =================================================
            # خروج
            # =================================================

            if hit_sl or is_timeout:

                initial_risk = (
                    pos["initial_risk"]
                )

                # ------------------------------------------------
                # منطق خروج اصلی V33
                # ------------------------------------------------

                exit_p = (

                    min(
                        pos["stop_loss"],
                        c4h["Open"]
                    )

                    if hit_sl

                    else c4h["Close"]
                )

                # =================================================
                # Return بر اساس R
                # =================================================

                r_real = (

                    (
                        exit_p
                        -
                        pos["entry_price"]
                    )
                    /
                    initial_risk

                    -
                    (FEE_RATE * 2)
                )

                # =================================================
                # Outcome
                # =================================================

                outcome = (

                    "WIN"

                    if r_real > 0

                    else "LOSS"
                )

                # =================================================
                # V47 Loss Cluster Counter
                # =================================================

                if outcome == "LOSS":

                    losses_this_timestamp += 1

                # =================================================
                # Diagnostic data
                # =================================================

                btc_regime = "UNKNOWN"

                market_breadth = np.nan

                # ------------------------------------------------
                # BTC Regime
                # ------------------------------------------------

                if (
                    "BTC" in processed_data
                    and
                    ts in processed_data[
                        "BTC"
                    ].index
                ):

                    btc = (
                        processed_data[
                            "BTC"
                        ].loc[ts]
                    )

                    btc_regime = (

                        "STRONG"

                        if (
                            btc["Close"]
                            > btc["EMA20"]

                            and

                            btc["EMA20"]
                            > btc["EMA50"]

                            and

                            btc["Close"]
                            > btc["EMA200"]
                        )

                        else

                        "WEAK"
                    )

                # ------------------------------------------------
                # Market Breadth
                # ------------------------------------------------

                breadth_total = 0

                breadth_bull = 0

                for (
                    _sym,
                    _df
                ) in processed_data.items():

                    if ts not in _df.index:
                        continue

                    breadth_total += 1

                    _c = _df.loc[ts]

                    if (
                        _c["Close"]
                        > _c["EMA20"]

                        and

                        _c["EMA20"]
                        > _c["EMA50"]

                        and

                        _c["Close"]
                        > _c["EMA200"]
                    ):

                        breadth_bull += 1

                if breadth_total:

                    market_breadth = (
                        breadth_bull
                        /
                        breadth_total
                    )

                # =================================================
                # ثبت معامله
                # =================================================

                all_trades.append(

                    {
                        "Timestamp": ts,

                        "Symbol": symbol,

                        "Side": "LONG",

                        "Outcome": outcome,

                        "Return": r_real,

                        "ExitOrder": len(
                            all_trades
                        ),

                        "BTC_Regime":
                            btc_regime,

                        "Market_Breadth":
                            market_breadth,
                    }
                )

                symbols_to_close.append(
                    symbol
                )

        # ========================================================
        # حذف پوزیشن‌های بسته‌شده
        # ========================================================

        for sym in symbols_to_close:

            del active_positions[
                sym
            ]

        # ========================================================
        # V47 — Loss Cluster Detection
        # ========================================================
        #
        # اگر حداقل 2 LOSS در همین timestamp
        # بسته شده باشد، cooldown فعال می‌شود.
        #
        # نکته:
        # cooldown فقط ورود جدید را متوقف می‌کند.
        # پوزیشن باز وجود داشته باشد، همچنان مدیریت می‌شود.
        # ========================================================

        if (
            cluster_cooldown_candles > 0
            and
            losses_this_timestamp >= 2
        ):

            cluster_cooldown_remaining = max(

                cluster_cooldown_remaining,

                cluster_cooldown_candles
            )

        # ========================================================
        # V47 — Cooldown
        # ========================================================

        if (
            cluster_cooldown_remaining
            > 0
        ):

            cluster_cooldown_remaining -= 1

            continue

        # ========================================================
        # امتیازدهی
        # ========================================================

        current_scores = {}

        for (
            symbol,
            df
        ) in processed_data.items():

            if ts not in df.index:
                continue

            val = df.loc[
                ts,
                "Mom_Long"
            ]

            if not np.isnan(val):

                current_scores[
                    symbol
                ] = val

        if not current_scores:
            continue

        # ========================================================
        # Ranking
        # ========================================================

        ranked_symbols = sorted(

            current_scores.keys(),

            key=lambda x:
                current_scores[x],

            reverse=True
        )

        # ========================================================
        # ورود
        # ========================================================

        for symbol in ranked_symbols:

            # ----------------------------------------------------
            # MAX POSITIONS
            # ----------------------------------------------------

            if (
                len(active_positions)
                >= MAX_POSITIONS
            ):

                break

            # ----------------------------------------------------
            # اگر از قبل پوزیشن داریم
            # ----------------------------------------------------

            if symbol in active_positions:

                continue

            df = processed_data[
                symbol
            ]

            if ts not in df.index:

                continue

            # ====================================================
            # Index
            # ====================================================

            i = df.index.get_loc(
                ts
            )

            if i < EMA_WARMUP:

                continue

            c4h = df.iloc[i]

            # ====================================================
            # Bull Regime
            # ====================================================

            regime_bull = (

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

            # ====================================================
            # V33 Signal
            # ====================================================

            valid_trend = (

                regime_bull

                and

                c4h["Mom_Short"]
                > 0.012

                and

                c4h["Mom_Long"]
                > 0.035
            )

            if not valid_trend:

                continue

            # ====================================================
            # Entry
            # ====================================================
            #
            # عمداً همان منطق قبلی حفظ شده:
            # ورود روی Open همان کندل.
            # ====================================================

            entry_price = (

                c4h["Open"]
                *
                (1 + SLIPPAGE)
            )

            # ====================================================
            # Initial Stop
            # ====================================================

            initial_sl = (

                entry_price

                -
                INITIAL_ATR_MULTIPLIER
                *
                c4h["ATR"]
            )

            # ====================================================
            # Initial Risk
            # ====================================================

            initial_risk = (

                entry_price
                -
                initial_sl
            )

            if (
                initial_risk <= 0
                or
                np.isnan(initial_risk)
            ):

                continue

            # ====================================================
            # SL Distance
            # ====================================================

            sl_dist_pct = (

                initial_risk
                /
                entry_price
            )

            # ====================================================
            # Risk Filter
            # ====================================================

            if not (
                0.01
                <= sl_dist_pct
                <= 0.04
            ):

                continue

            # ====================================================
            # ثبت پوزیشن
            # ====================================================

            active_positions[
                symbol
            ] = {

                "side":
                    "LONG",

                "entry_price":
                    entry_price,

                "stop_loss":
                    initial_sl,

                "highest_price":
                    entry_price,

                "initial_risk":
                    initial_risk,

                "entry_index":
                    i,
            }

    # ========================================================
    # خروجی
    # ========================================================

    return pd.DataFrame(
        all_trades
    )


# ============================================================
# Summary
# ============================================================

def summarize_result(
    trades_df,
    label
):

    print(
        "\n"
        +
        "=" * 68
    )

    print(
        f"📊 {label}"
    )

    print(
        "=" * 68
    )

    if trades_df.empty:

        print(
            "⚠️ هیچ معامله‌ای ثبت نشد."
        )

        return {

            "Trades": 0,

            "WR": 0.0,

            "NetR": 0.0,

            "MaxDD": 0.0,

            "MaxLossStreak": 0,
        }

    # ========================================================
    # ترتیب واقعی خروج
    # ========================================================

    trades_df = (

        trades_df

        .sort_values(
            [
                "Timestamp",
                "ExitOrder"
            ],
            kind="stable"
        )

        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Stats
    # ========================================================

    wins = int(
        (
            trades_df["Outcome"]
            == "WIN"
        ).sum()
    )

    losses = int(
        (
            trades_df["Outcome"]
            == "LOSS"
        ).sum()
    )

    trades = len(
        trades_df
    )

    wr = (

        wins
        /
        trades
        *
        100
    )

    net_r = float(
        trades_df[
            "Return"
        ].sum()
    )

    # ========================================================
    # Max Loss Streak
    # ========================================================

    cur = 0

    max_streak = 0

    # ========================================================
    # Max Drawdown
    # ========================================================

    eq = 0.0

    peak = 0.0

    max_dd = 0.0

    for (
        outcome,
        r
    ) in zip(
        trades_df[
            "Outcome"
        ],
        trades_df[
            "Return"
        ]
    ):

        if outcome == "LOSS":

            cur += 1

            max_streak = max(
                max_streak,
                cur
            )

        else:

            cur = 0

        eq += float(r)

        peak = max(
            peak,
            eq
        )

        max_dd = min(
            max_dd,
            eq - peak
        )

    # ========================================================
    # Print
    # ========================================================

    print(
        f"🔸 Trades: {trades}"
    )

    print(
        f"🔸 Wins: {wins}"
    )

    print(
        f"🔸 Losses: {losses}"
    )

    print(
        f"🎯 Win Rate: {wr:.2f}%"
    )

    print(
        f"💰 Net R: {net_r:.2f}R"
    )

    print(
        f"❄️ Max Loss Streak: "
        f"{max_streak}"
    )

    print(
        f"📉 Max DD: "
        f"{max_dd:.2f}R"
    )

    return {

        "Trades":
            trades,

        "WR":
            wr,

        "NetR":
            net_r,

        "MaxDD":
            max_dd,

        "MaxLossStreak":
            max_streak,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        +
        "=" * 68
    )

   
