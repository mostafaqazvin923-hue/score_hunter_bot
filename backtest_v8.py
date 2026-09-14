import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...", flush=True)
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "ccxt"
    ])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# HUNTER-V47
# ============================================================
#
# پایه:
# HUNTER-V46
#
# تغییر V47:
#
# اگر حداقل 2 معامله در یک timestamp با LOSS بسته شوند،
# ورودهای جدید برای تعداد مشخصی کندل متوقف می‌شوند.
#
# تست‌ها:
#
# BASE  = بدون Cluster Control
# V47-A = Cooldown = 2 candles
# V47-B = Cooldown = 4 candles
#
# نکته مهم:
#
# Cluster Control فقط NEW ENTRY را متوقف می‌کند.
# پوزیشن‌های باز همچنان با همان:
#   SL
#   Trailing
#   Timeout
#   Exit
# مدیریت می‌شوند.
#
# هیچ Position Sizing جدیدی اضافه نشده.
# هیچ Short اضافه نشده.
# هیچ فیلتر جدیدی اضافه نشده.
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

start_date = (
    datetime.now()
    - timedelta(days=LOOKBACK_DAYS)
)

since_timestamp = int(
    start_date.timestamp() * 1000
)


# ============================================================
# دریافت داده از LBank
# ============================================================

print("=" * 68, flush=True)

print(
    "📥 دریافت داده‌ها - HUNTER-V47",
    flush=True
)

print("=" * 68, flush=True)


def fetch_symbol_data(lbank_symbol):

    all_ohlcv = []

    current_since = since_timestamp

    last_seen = None

    while current_since < exchange.milliseconds():

        batch = None

        # ====================================================
        # Retry
        # ====================================================

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
                        f"⚠️ دریافت ناقص "
                        f"{lbank_symbol}: {e}",
                        flush=True
                    )

                    return None

        if not batch:

            break

        first_ts = batch[0][0]

        last_ts = batch[-1][0]

        # ====================================================
        # Pagination protection
        # ====================================================

        if (
            last_seen is not None
            and last_ts <= last_seen
        ):

            print(
                f"⚠️ pagination متوقف شد: "
                f"{lbank_symbol}",
                flush=True
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

    # ========================================================
    # Date
    # ========================================================

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

    # ========================================================
    # Cleaning
    # ========================================================

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
            df.iloc[-1]["Date"].timestamp()
            * 1000
        )

        candle_ms = (
            4
            * 60
            * 60
            * 1000
        )

        if (
            last_ms + candle_ms
            > now_ms
        ):

            df = df.iloc[:-1].copy()

    # ========================================================
    # Minimum data
    # ========================================================

    if len(df) < EMA_WARMUP + 50:

        return None

    # ========================================================
    # Gap check
    # ========================================================

    deltas = (
        df["Date"]
        .diff()
        .dropna()
    )

    if (
        not deltas.empty
        and
        deltas.max()
        >
        pd.Timedelta(
            hours=4,
            minutes=10
        )
    ):

        print(
            f"⚠️ gap بزرگ در "
            f"{lbank_symbol}؛ "
            f"نماد حذف شد.",
            flush=True
        )

        return None

    # ========================================================
    # ATR
    # ========================================================

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
            -
            df["Close"].shift(10)
        )
        /
        df["Close"].shift(10)
    )

    df["Mom_Long"] = (
        (
            df["Close"]
            -
            df["Close"].shift(30)
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

    print(
        f"📡 دریافت {symbol} ...",
        flush=True
    )

    df4h = fetch_symbol_data(
        lbank_symbol
    )

    if df4h is not None:

        processed_data[symbol] = df4h

        print(
            f"✅ {symbol}: "
            f"{len(df4h)} کندل",
            flush=True
        )

    else:

        print(
            f"❌ {symbol}: حذف شد",
            flush=True
        )


print(
    "",
    flush=True
)

print(
    f"✅ تعداد نمادهای معتبر: "
    f"{len(processed_data)} "
    f"از {len(SYMBOLS)}",
    flush=True
)

print(
    "=" * 68,
    flush=True
)


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    processed_data,
    cluster_cooldown_candles=0
):

    # ========================================================
    # Timestampهای مشترک/موجود
    # ========================================================

    all_timestamps = sorted(
        {
            ts
            for df in processed_data.values()
            for ts in df.index
        }
    )

    # ========================================================
    # Active Positions
    # ========================================================

    active_positions = {}

    # ========================================================
    # Trades
    # ========================================================

    all_trades = []

    # ========================================================
    # Cluster Cooldown State
    # ========================================================
    #
    # تعداد کندل‌هایی که ورود جدید ممنوع است.
    #
    # مقدار 0:
    # هیچ محدودیتی وجود ندارد.
    #
    # ========================================================

    cluster_cooldown_remaining = 0

    # ========================================================
    # Market Loop
    # ========================================================

    for ts in all_timestamps:

        # ====================================================
        # این متغیر فقط برای همین timestamp است.
        # ====================================================

        losses_this_timestamp = 0

        symbols_to_close = []

        # ====================================================
        # 1) مدیریت پوزیشن‌های باز
        # ====================================================

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
                >
                pos["highest_price"]
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
                    >
                    pos["stop_loss"]
                ):

                    pos["stop_loss"] = (
                        new_trailing_sl
                    )

            # =================================================
            # Stop Loss
            # =================================================

            hit_sl = (
                c4h["Low"]
                <=
                pos["stop_loss"]
            )

            # =================================================
            # Timeout
            # =================================================

            curr_i = (
                df.index.get_loc(ts)
            )

            candles_held = (
                curr_i
                -
                pos["entry_index"]
            )

            is_timeout = (
                candles_held
                >=
                TIMEOUT_CANDLES
            )

            # =================================================
            # Exit
            # =================================================

            if (
                hit_sl
                or
                is_timeout
            ):

                initial_risk = (
                    pos["initial_risk"]
                )

                # =================================================
                # Exit price
                # =================================================

                if hit_sl:

                    exit_p = min(
                        pos["stop_loss"],
                        c4h["Open"]
                    )

                else:

                    exit_p = c4h["Close"]

                # =================================================
                # R result
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
                    (
                        FEE_RATE
                        *
                        2
                    )
                )

                # =================================================
                # Outcome
                # =================================================

                if r_real > 0:

                    outcome = "WIN"

                else:

                    outcome = "LOSS"

                    # فقط LOSSهای همین timestamp
                    losses_this_timestamp += 1

                # =================================================
                # Diagnostic
                # =================================================

                btc_regime = "UNKNOWN"

                market_breadth = np.nan

                # =================================================
                # BTC Regime
                # =================================================

                if (
                    "BTC"
                    in
                    processed_data
                ):

                    btc_df = (
                        processed_data["BTC"]
                    )

                    if ts in btc_df.index:

                        btc = (
                            btc_df.loc[ts]
                        )

                        if (
                            btc["Close"]
                            >
                            btc["EMA20"]

                            and

                            btc["EMA20"]
                            >
                            btc["EMA50"]

                            and

                            btc["Close"]
                            >
                            btc["EMA200"]
                        ):

                            btc_regime = (
                                "STRONG"
                            )

                        else:

                            btc_regime = (
                                "WEAK"
                            )

                # =================================================
                # Market Breadth
                # =================================================

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
                        >
                        _c["EMA20"]

                        and

                        _c["EMA20"]
                        >
                        _c["EMA50"]

                        and

                        _c["Close"]
                        >
                        _c["EMA200"]
                    ):

                        breadth_bull += 1

                if breadth_total > 0:

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

                        "ExitOrder":
                            len(all_trades),

                        "BTC_Regime":
                            btc_regime,

                        "Market_Breadth":
                            market_breadth,
                    }
                )

                symbols_to_close.append(
                    symbol
                )

        # ====================================================
        # 2) حذف پوزیشن‌های بسته‌شده
        # ====================================================

        for sym in symbols_to_close:

            if sym in active_positions:

                del active_positions[sym]

        # ====================================================
        # 3) Cluster Detection
        # ====================================================
        #
        # بسیار مهم:
        #
        # این قسمت فقط زمانی فعال است که حداقل 2 LOSS
        # در همین timestamp بسته شده باشد.
        #
        # برای BASE چون مقدار cooldown = 0 است،
        # هیچ تغییری در رفتار ایجاد نمی‌کند.
        #
        # ====================================================

        if (
            cluster_cooldown_candles > 0
            and
            losses_this_timestamp >= 2
        ):

            cluster_cooldown_remaining = max(
                cluster_cooldown_remaining,
                cluster_cooldown_candles
            )

        # ====================================================
        # 4) Cluster Cooldown
        # ====================================================
        #
        # اگر cooldown فعال باشد:
        #
        # - پوزیشن‌های باز قبلاً مدیریت شده‌اند.
        # - هیچ پوزیشن جدیدی باز نمی‌شود.
        # - سپس یک کندل از cooldown مصرف می‌شود.
        #
        # ====================================================

        if (
            cluster_cooldown_remaining
            > 0
        ):

            cluster_cooldown_remaining -= 1

            continue

        # ====================================================
        # 5) Ranking
        # ====================================================

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

        # ====================================================
        # 6) Sort by Mom_Long
        # ====================================================

        ranked_symbols = sorted(
            current_scores.keys(),
            key=lambda x:
                current_scores[x],
            reverse=True
        )

        # ====================================================
        # 7) Entry
        # ====================================================

        for symbol in ranked_symbols:

            # =================================================
            # Max Positions
            # =================================================

            if (
                len(active_positions)
                >= MAX_POSITIONS
            ):

                break

            # =================================================
            # Already open
            # =================================================

            if symbol in active_positions:

                continue

            df = processed_data[
                symbol
            ]

            if ts not in df.index:

                continue

            # =================================================
            # Index
            # =================================================

            i = df.index.get_loc(ts)

            if i < EMA_WARMUP:

                continue

            c4h = df.iloc[i]

            # =================================================
            # ATR validation
            # =================================================

            if (
                pd.isna(c4h["ATR"])
                or
                c4h["ATR"] <= 0
            ):

                continue

            # =================================================
            # Bull Regime
            # =================================================

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

            # =================================================
            # Original V33 Signal
            # =================================================

            valid_trend = (

                regime_bull

                and

                c4h["Mom_Short"]
                >
                0.012

                and

                c4h["Mom_Long"]
                >
                0.035
            )

            if not valid_trend:

                continue

            # =================================================
            # Entry Price
            # =================================================
            #
            # عمداً همان منطق قبلی:
            # Open همان کندل.
            #
            # =================================================

            entry_price = (

                c4h["Open"]
                *
                (
                    1
                    +
                    SLIPPAGE
                )
            )

            # =================================================
            # Initial Stop
            # =================================================

            initial_sl = (

                entry_price

                -
                INITIAL_ATR_MULTIPLIER
                *
                c4h["ATR"]
            )

            # =================================================
            # Initial Risk
            # =================================================

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

            # =================================================
            # SL Distance
            # =================================================

            sl_dist_pct = (

                initial_risk
                /
                entry_price
            )

            # =================================================
            # Original Risk Filter
            # =================================================

            if not (
                0.01
                <=
                sl_dist_pct
                <=
                0.04
            ):

                continue

            # =================================================
            # Open Position
            # =================================================

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
    # Return Trades
    # ========================================================

    return pd.DataFrame(
        all_trades
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_result(
    trades_df,
    label
):

    print(
        "\n"
        +
        "=" * 68,
        flush=True
    )

    print(
        f"📊 {label}",
        flush=True
    )

    print(
        "=" * 68,
        flush=True
    )

    if trades_df.empty:

        print(
            "⚠️ هیچ معامله‌ای ثبت نشد.",
            flush=True
        )

        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "WR": 0.0,
            "NetR": 0.0,
            "MaxDD": 0.0,
            "MaxLossStreak": 0,
        }

    # ========================================================
    # Actual exit order
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
    # Wins / Losses
    # ========================================================

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

    trades = len(
        trades_df
    )

    # ========================================================
    # Win Rate
    # ========================================================

    wr = (
        wins
        /
        trades
        *
        100
    )

    # ========================================================
    # Net R
    # ========================================================

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
        trades_df["Outcome"],
        trades_df["Return"]
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
        f"🔸 Trades: {trades}",
        flush=True
    )

    print(
        f"🔸 Wins: {wins}",
        flush=True
    )

    print(
        f"🔸 Losses: {losses}",
        flush=True
    )

    print(
        f"🎯 Win Rate: {wr:.2f}%",
        flush=True
    )

    print(
        f"💰 Net R: {net_r:.2f}R",
        flush=True
    )

    print(
        f"❄️ Max Loss Streak: "
        f"{max_streak}",
        flush=True
    )

    print(
        f"📉 Max DD: "
        f"{max_dd:.2f}R",
        flush=True
    )

    return {
        "Trades": trades,
        "Wins": wins,
        "Losses": losses,
        "WR": wr,
        "NetR": net_r,
        "MaxDD": max_dd,
        "MaxLossStreak": max_streak,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        +
        "=" * 68,
        flush=True
    )

    print(
        "🚀 HUNTER-V47",
        flush=True
    )

    print(
        "Confirmed Loss-Cluster Control Test",
        flush=True
    )

    print(
        "=" * 68,
        flush=True
    )

    print(
        "\n⚙️ Universe: "
        f"{len(processed_data)} symbols",
        flush=True
    )

    print(
        "⚙️ Timeframe: 4h",
        flush=True
    )

    print(
        "⚙️ Lookback: 365 days",
        flush=True
    )

    print(
        "⚙️ Max Positions: 5",
        flush=True
    )

    print(
        "\n"
        "⚠️ منطق Entry / Exit / SL / "
        "Trailing / Timeout / Fee ثابت است.",
        flush=True
    )

    # ========================================================
    # Results container
    # ========================================================

    results = {}

    # ========================================================
    # BASE
    # ========================================================

    print(
        "\n"
        +
        "-" * 68,
        flush=True
    )

    print(
        "🧪 اجرای BASE",
        flush=True
    )

    print(
        "Cluster Control = OFF",
        flush=True
    )

    print(
        "-" * 68,
        flush=True
    )

    df_base = run_backtest(
        processed_data,
        cluster_cooldown_candles=0
    )

    results["BASE"] = summarize_result(
        df_base,
        "BASE — No Cluster Control"
    )

    # ========================================================
    # V47-A
    # ========================================================

    print(
        "\n"
        +
        "-" * 68,
        flush=True
    )

    print(
        "🧪 اجرای V47-A",
        flush=True
    )

    print(
        "Cluster Cooldown = 2 candles",
        flush=True
    )

    print(
        "-" * 68,
        flush=True
    )

    df_a = run_backtest(
        processed_data,
        cluster_cooldown_candles=2
    )

    results["V47-A"] = summarize_result(
        df_a,
        "V47-A — Cluster Cooldown 2"
    )

    # ========================================================
    # V47-B
    # ========================================================

    print(
        "\n"
        +
        "-" * 68,
        flush=True
    )

    print(
        "🧪 اجرای V47-B",
        flush=True
    )

    print(
        "Cluster Cooldown = 4 candles",
        flush=True
    )

    print(
        "-" * 68,
        flush=True
    )

    df_b = run_backtest(
        processed_data,
        cluster_cooldown_candles=4
    )

    results["V47-B"] = summarize_result(
        df_b,
        "V47-B — Cluster Cooldown 4"
    )

    # ========================================================
    # Comparison
    # ========================================================

    print(
        "\n"
        +
        "=" * 82,
        flush=True
    )

    print(
        "🏁 HUNTER-V47 COMPARISON",
        flush=True
    )

    print(
        "=" * 82,
        flush=True
    )

    print(
        f"{'Variant':<15}"
        f"{'Trades':>10}"
        f"{'Wins':>8}"
        f"{'Losses':>9}"
        f"{'WR%':>10}"
        f"{'NetR':>12}"
        f"{'MaxDD':>12}"
        f"{'MaxLS':>10}",
        flush=True
    )

    print(
        "-" * 82,
        flush=True
    )

    for (
        label,
        result
    ) in results.items():

        print(
            f"{label:<15}"
            f"{result['Trades']:>10}"
            f"{result['Wins']:>8}"
            f"{result['Losses']:>9}"
            f"{result['WR']:>9.2f}%"
            f"{result['NetR']:>11.2f}R"
            f"{result['MaxDD']:>11.2f}R"
            f"{result['MaxLossStreak']:>10}",
            flush=True
        )

    # ========================================================
    # Comparison versus BASE
    # ========================================================

    base = results["BASE"]

    print(
        "\n"
        +
        "=" * 82,
        flush=True
    )

    print(
        "📈 تغییر نسبت به BASE",
        flush=True
    )

    print(
        "=" * 82,
        flush=True
    )

    for label in [
        "V47-A",
        "V47-B"
    ]:

        r = results[label]

        print(
            f"\n🔹 {label}",
            flush=True
        )

        print(
            f"Trades Δ: "
            f"{r['Trades'] - base['Trades']:+d}",
            flush=True
        )

        print(
            f"WR Δ: "
            f"{r['WR'] - base['WR']:+.2f}%",
            flush=True
        )

        print(
            f"Net R Δ: "
            f"{r['NetR'] - base['NetR']:+.2f}R",
            flush=True
        )

        print(
            f"Max DD Δ: "
            f"{r['MaxDD'] - base['MaxDD']:+.2f}R",
            flush=True
        )

        print(
            f"Max Loss Streak Δ: "
            f"{r['MaxLossStreak'] - base['MaxLossStreak']:+d}",
            flush=True
        )

    # ========================================================
    # Validation
    # ========================================================

    print(
        "\n"
        +
        "=" * 82,
        flush=True
    )

    print(
        "🔍 BASE VALIDATION",
        flush=True
    )

    print(
        "=" * 82,
        flush=True
    )

    expected = {
        "Trades": 267,
        "Wins": 162,
        "Losses": 105,
        "WR": 60.67,
        "NetR": 97.45,
        "MaxDD": -5.73,
        "MaxLossStreak": 9,
    }

    print(
        "نتیجه مرجع V46:",
        flush=True
    )

    print(
        "267 Trades | "
        "162 Wins | "
        "105 Losses | "
        "60.67% WR | "
        "+97.45R | "
        "-5.73R DD | "
        "9 MaxLS",
        flush=True
    )

    print(
        "\nنتیجه BASE فعلی:",
        flush=True
    )

    print(
        f"{base['Trades']} Trades | "
        f"{base['Wins']} Wins | "
        f"{base['Losses']} Losses | "
        f"{base['WR']:.2f}% WR | "
        f"{base['NetR']:.2f}R | "
        f"{base['MaxDD']:.2f}R DD | "
        f"{base['MaxLossStreak']} MaxLS",
        flush=True
    )

    # ========================================================
    # Check
    # ========================================================

    base_matches = (

        base["Trades"]
        ==
        expected["Trades"]

        and

        base["Wins"]
        ==
        expected["Wins"]

        and

        base["Losses"]
        ==
        expected["Losses"]

        and

        abs(
            base["WR"]
            -
            expected["WR"]
        )
        < 0.01

        and

        abs(
            base["NetR"]
            -
            expected["NetR"]
        )
        < 0.01

        and

        abs(
            base["MaxDD"]
            -
            expected["MaxDD"]
        )
        < 0.01

        and

        base["MaxLossStreak"]
        ==
        expected["MaxLossStreak"]
    )

    if base_matches:

        print(
            "\n✅ BASE با نتیجه مرجع V46 "
            "مطابقت دارد.",
            flush=True
        )

        print(
            "✅ حالا V47-A و V47-B قابل مقایسه هستند.",
            flush=True
        )

    else:

        print(
            "\n⚠️ هشدار:",
            flush=True
        )

        print(
            "BASE با نتیجه مرجع V46 "
            "مطابقت ندارد.",
            flush=True
        )

        print(
            "قبل از قضاوت درباره Cluster Control، "
            "داده یا منطق نسخه پایه باید بررسی شود.",
            flush=True
        )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        +
        "=" * 82,
        flush=True
    )

    print(
        "✨ HUNTER-V47 به پایان رسید.",
        flush=True
    )

    print(
        "=" * 82,
        flush=True
    )
