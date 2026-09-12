import os
import subprocess
import sys
from datetime import datetime, timedelta

# ============================================================
# نصب خودکار کتابخانه‌ها
# ============================================================

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب ccxt...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "ccxt"
    ])
    import ccxt

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "pandas"
    ])
    import pandas as pd

try:
    import numpy as np
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "numpy"
    ])
    import numpy as np


# ============================================================
# تنظیمات
# ============================================================

exchange = ccxt.lbank({
    'enableRateLimit': True,
    'timeout': 30000
})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT",
    "DOT": "DOT/USDT"
}

BACKTEST_DAYS = int(
    os.getenv("BACKTEST_DAYS", "365")
)

FETCH_LIMIT = 1000

# ============================================================
# استراتژی
# ============================================================

TP_PCT = 0.0200
SL_PCT = 0.0100

# حداکثر زمان نگهداری معامله
MAX_HOLD_BARS = 24

# کارمزد هر طرف
FEE_RATE = 0.0006

# اسلیپیج هر طرف
SLIPPAGE_RATE = 0.0002

OUTPUT_CSV = "hunter_x_v9_lbank_trades.csv"


# ============================================================
# اندیکاتورها
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]

    tr2 = (
        df["High"] - prev_close
    ).abs()

    tr3 = (
        df["Low"] - prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # --------------------------------------------------------
    # Volume MA
    # --------------------------------------------------------

    df["VOL_MA20"] = (
        df["Volume"].rolling(20).mean()
    )

    # --------------------------------------------------------
    # Candle Body
    # --------------------------------------------------------

    df["BODY"] = (
        df["Close"] - df["Open"]
    ).abs()

    df["RANGE"] = (
        df["High"] - df["Low"]
    )

    df["BODY_RATIO"] = np.where(
        df["RANGE"] > 0,
        df["BODY"] / df["RANGE"],
        0
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()

    plus_dm = np.where(
        (plus_dm > minus_dm) & (plus_dm > 0),
        plus_dm,
        0
    )

    minus_dm = np.where(
        (minus_dm > plus_dm) & (minus_dm > 0),
        minus_dm,
        0
    )

    atr14 = df["ATR"]

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index
        ).rolling(14).mean()
        / atr14
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index
        ).rolling(14).mean()
        / atr14
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    df["ADX"] = dx.rolling(14).mean()

    return df


# ============================================================
# دریافت دیتای LBank
# ============================================================

def fetch_lbank_1h(symbol, days=365):

    print("")
    print("=" * 60)
    print(
        f"📥 شروع دریافت دیتای {symbol} از LBank"
    )
    print("=" * 60)

    start_date = (
        datetime.utcnow()
        - timedelta(days=days + 10)
    )

    since_timestamp = int(
        start_date.timestamp() * 1000
    )

    now_timestamp = exchange.milliseconds()

    all_ohlcv = []

    current_since = since_timestamp

    request_number = 0

    while current_since < now_timestamp:

        request_number += 1

        try:

            print(
                f"  🔄 درخواست #{request_number} | "
                f"کندل دریافت‌شده: "
                f"{len(all_ohlcv)}"
            )

            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe="1h",
                since=current_since,
                limit=FETCH_LIMIT
            )

            if not ohlcv:

                print(
                    f"  ⚠️ LBank برای {symbol} "
                    f"دیتای بیشتری نداد."
                )

                break

            all_ohlcv.extend(
                ohlcv
            )

            last_timestamp = ohlcv[-1][0]

            current_since = (
                last_timestamp + 1
            )

            print(
                f"  ✔️ +{len(ohlcv)} کندل | "
                f"مجموع: {len(all_ohlcv)}"
            )

            if len(ohlcv) < FETCH_LIMIT:
                break

        except Exception as e:

            print(
                f"  ❌ خطا در دریافت "
                f"{symbol}: {e}"
            )

            break

    if not all_ohlcv:

        print(
            f"❌ هیچ دیتایی برای "
            f"{symbol} دریافت نشد."
        )

        return None

    # --------------------------------------------------------
    # حذف duplicate
    # --------------------------------------------------------

    unique = {}

    for row in all_ohlcv:
        unique[row[0]] = row

    all_ohlcv = list(
        unique.values()
    )

    all_ohlcv.sort(
        key=lambda x: x[0]
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

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

    # مهم:
    # index را عمداً timezone-naive می‌سازیم
    # تا تمام مقایسه‌های زمانی یکسان باشند.
    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
    ).dt.tz_localize(None)

    df.set_index(
        "Date",
        inplace=True
    )

    # --------------------------------------------------------
    # فقط داده موردنیاز
    # --------------------------------------------------------

    # FIX:
    # هر دو طرف مقایسه timezone-naive هستند.
    now_naive = (
        pd.Timestamp.now(tz="UTC")
        .tz_localize(None)
    )

    cutoff = (
        now_naive
        - pd.Timedelta(days=days)
    )

    df = df[
        df.index >= cutoff
    ].copy()

    # --------------------------------------------------------
    # حذف کندل جاری
    # --------------------------------------------------------

    if len(df) > 1:

        current_hour = (
            pd.Timestamp.now(tz="UTC")
            .tz_localize(None)
            .replace(
                minute=0,
                second=0,
                microsecond=0
            )
        )

        df = df[
            df.index < current_hour
        ].copy()

    print("")

    print(
        f"  ✅ {symbol} آماده شد | "
        f"{len(df)} کندل 1H"
    )

    if len(df) > 0:

        print(
            f"  📅 از {df.index[0]} "
            f"تا {df.index[-1]}"
        )

    return df


# ============================================================
# ساخت 4H از دیتای 1H
# ============================================================

def make_4h(df1h):

    df4h = df1h.resample(
        "4h"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return df4h


# ============================================================
# اضافه کردن ویژگی 4H قبلی به 1H
# ============================================================

def attach_previous_4h(
    df1h,
    df4h
):

    h4 = df4h.copy()

    h4 = calculate_indicators(
        h4
    )

    # فقط کندل 4H کاملاً بسته‌شده قبلی
    h4_features = h4[
        [
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI",
            "ADX",
            "ATR"
        ]
    ].shift(1)

    h4_features = h4_features.rename(
        columns={
            "EMA20": "H4_EMA20",
            "EMA50": "H4_EMA50",
            "EMA200": "H4_EMA200",
            "RSI": "H4_RSI",
            "ADX": "H4_ADX",
            "ATR": "H4_ATR"
        }
    )

    # --------------------------------------------------------
    # زمان کندل 4H
    # --------------------------------------------------------

    h4_features["H4_BUCKET"] = (
        h4_features.index
        .floor("4h")
    )

    result = df1h.copy()

    result["H4_BUCKET"] = (
        result.index
        .floor("4h")
    )

    result = result.join(
        h4_features.set_index(
            "H4_BUCKET"
        ),
        on="H4_BUCKET",
        rsuffix="_H4"
    )

    return result


# ============================================================
# شبیه‌سازی معامله
# ============================================================

def simulate_trade(
    df,
    entry_idx,
    side,
    entry_price
):

    if entry_idx >= len(df):
        return None

    entry_row = df.iloc[
        entry_idx
    ]

    atr = entry_row["ATR"]

    if pd.isna(atr) or atr <= 0:
        return None

    # --------------------------------------------------------
    # قیمت‌ها
    # --------------------------------------------------------

    if side == "LONG":

        raw_sl = entry_price * (
            1 - SL_PCT
        )

        raw_tp = entry_price * (
            1 + TP_PCT
        )

        sl = raw_sl * (
            1 - SLIPPAGE_RATE
        )

        tp = raw_tp * (
            1 - SLIPPAGE_RATE
        )

    else:

        raw_sl = entry_price * (
            1 + SL_PCT
        )

        raw_tp = entry_price * (
            1 - TP_PCT
        )

        sl = raw_sl * (
            1 + SLIPPAGE_RATE
        )

        tp = raw_tp * (
            1 + SLIPPAGE_RATE
        )

    max_exit = min(
        entry_idx + MAX_HOLD_BARS,
        len(df) - 1
    )

    exit_idx = None
    outcome = None
    exit_price = None

    # --------------------------------------------------------
    # بررسی کندل‌ها
    # --------------------------------------------------------

    for j in range(
        entry_idx,
        max_exit + 1
    ):

        candle = df.iloc[j]

        high = candle["High"]
        low = candle["Low"]

        if side == "LONG":

            hit_sl = low <= sl
            hit_tp = high >= tp

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

        # برخورد همزمان:
        # SL اول
        if hit_sl and hit_tp:

            outcome = "LOSS"
            exit_price = sl
            exit_idx = j

            break

        if hit_sl:

            outcome = "LOSS"
            exit_price = sl
            exit_idx = j

            break

        if hit_tp:

            outcome = "WIN"
            exit_price = tp
            exit_idx = j

            break

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    if outcome is None:

        exit_idx = max_exit

        exit_price = df.iloc[
            exit_idx
        ]["Close"]

        outcome = "TIMEOUT"

    # --------------------------------------------------------
    # محاسبه بازده
    # --------------------------------------------------------

    if side == "LONG":

        gross_return = (
            exit_price / entry_price
        ) - 1

    else:

        gross_return = (
            entry_price / exit_price
        ) - 1

    # کارمزد ورود + خروج
    total_cost = (
        FEE_RATE * 2
        + SLIPPAGE_RATE * 2
    )

    net_return = (
        gross_return
        - total_cost
    )

    # تبدیل به R
    risk = SL_PCT

    net_r = (
        net_return / risk
    )

    return {
        "EntryIndex": entry_idx,
        "ExitIndex": exit_idx,
        "Side": side,
        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "Outcome": outcome,
        "GrossReturn": gross_return * 100,
        "NetReturn": net_return * 100,
        "R": net_r
    }


# ============================================================
# بک‌تست یک نماد
# ============================================================

def backtest_symbol(
    symbol,
    df
):

    print("")

    print(
        f"🚀 بک‌تست {symbol}..."
    )

    trades = []

    # حداقل warmup
    start_idx = 250

    locked_until = -1

    for i in range(
        start_idx,
        len(df) - MAX_HOLD_BARS - 2
    ):

        if i < locked_until:
            continue

        row = df.iloc[i]

        # ----------------------------------------------------
        # داده‌های 4H
        # ----------------------------------------------------

        h4_ema20 = row[
            "H4_EMA20"
        ]

        h4_ema50 = row[
            "H4_EMA50"
        ]

        h4_ema200 = row[
            "H4_EMA200"
        ]

        h4_rsi = row[
            "H4_RSI"
        ]

        h4_adx = row[
            "H4_ADX"
        ]

        if any(
            pd.isna(x)
            for x in [
                h4_ema20,
                h4_ema50,
                h4_ema200,
                h4_rsi,
                h4_adx
            ]
        ):
            continue

        # ----------------------------------------------------
        # فیلتر روند 4H
        # ----------------------------------------------------

        long_regime = (
            row["Close"] > h4_ema20
            and
            h4_ema20 > h4_ema50
            and
            h4_ema50 > h4_ema200
            and
            h4_rsi >= 50
            and
            h4_adx >= 15
        )

        short_regime = (
            row["Close"] < h4_ema20
            and
            h4_ema20 < h4_ema50
            and
            h4_ema50 < h4_ema200
            and
            h4_rsi <= 50
            and
            h4_adx >= 15
        )

        # ----------------------------------------------------
        # فیلترهای 1H
        # ----------------------------------------------------

        atr = row["ATR"]

        vol_ma = row[
            "VOL_MA20"
        ]

        if (
            pd.isna(atr)
            or atr <= 0
            or pd.isna(vol_ma)
            or vol_ma <= 0
        ):
            continue

        volume_ok = (
            row["Volume"]
            >= vol_ma * 1.20
        )

        body_ok = (
            row["BODY_RATIO"] >= 0.50
        )

        if (
            not volume_ok
            or not body_ok
        ):
            continue

        # ----------------------------------------------------
        # Setup 1: Pullback
        # ----------------------------------------------------

        previous = df.iloc[
            i - 1
        ]

        pullback_long = (
            row["Low"] <= row["EMA20"]
            and
            row["Close"] > row["EMA20"]
            and
            row["Close"] > row["Open"]
            and
            row["Close"] > previous["High"]
        )

        pullback_short = (
            row["High"] >= row["EMA20"]
            and
            row["Close"] < row["EMA20"]
            and
            row["Close"] < row["Open"]
            and
            row["Close"] < previous["Low"]
        )

        # ----------------------------------------------------
        # Setup 2: Momentum breakout
        # ----------------------------------------------------

        momentum_long = (
            row["Close"] > previous["High"]
            and
            row["Close"] > row["EMA20"]
            and
            row["Close"] > row["Open"]
        )

        momentum_short = (
            row["Close"] < previous["Low"]
            and
            row["Close"] < row["EMA20"]
            and
            row["Close"] < row["Open"]
        )

        # ----------------------------------------------------
        # ورود در OPEN کندل بعدی
        # ----------------------------------------------------

        entry_idx = i + 1

        if entry_idx >= len(df):
            break

        entry_price = df.iloc[
            entry_idx
        ]["Open"]

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if long_regime:

            setup = None

            if pullback_long:

                setup = "PULLBACK"

            elif momentum_long:

                setup = "MOMENTUM"

            if setup:

                trade = simulate_trade(
                    df,
                    entry_idx,
                    "LONG",
                    entry_price
                )

                if trade:

                    trade["Symbol"] = symbol

                    trade["Setup"] = setup

                    trade["EntryTime"] = (
                        df.index[entry_idx]
                    )

                    trade["ExitTime"] = (
                        df.index[
                            trade["ExitIndex"]
                        ]
                    )

                    trades.append(
                        trade
                    )

                    locked_until = (
                        trade["ExitIndex"]
                        + 1
                    )

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        elif short_regime:

            setup = None

            if pullback_short:

                setup = "PULLBACK"

            elif momentum_short:

                setup = "MOMENTUM"

            if setup:

                trade = simulate_trade(
                    df,
                    entry_idx,
                    "SHORT",
                    entry_price
                )

                if trade:

                    trade["Symbol"] = symbol

                    trade["Setup"] = setup

                    trade["EntryTime"] = (
                        df.index[entry_idx]
                    )

                    trade["ExitTime"] = (
                        df.index[
                            trade["ExitIndex"]
                        ]
                    )

                    trades.append(
                        trade
                    )

                    locked_until = (
                        trade["ExitIndex"]
                        + 1
                    )

    print(
        f"  ✔️ {symbol}: "
        f"{len(trades)} معامله"
    )

    return trades


# ============================================================
# MAIN
# ============================================================

def main():

    print("")

    print("=" * 70)

    print(
        "HUNTER-X V9 — "
        "LBank 1H → 4H CLEAN BACKTEST"
    )

    print("=" * 70)

    print(
        f"📅 دوره بک‌تست: "
        f"{BACKTEST_DAYS} روز"
    )

    print(
        "📡 منبع داده: LBank / CCXT"
    )

    print(
        "⏱️ دیتای اصلی: 1H"
    )

    print(
        "⏱️ تایم‌فریم بالاتر: "
        "4H ساخته‌شده از 1H"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # دریافت داده
    # --------------------------------------------------------

    data_1h = {}
    data_4h = {}

    for symbol, lbank_symbol in SYMBOLS.items():

        df1h = fetch_lbank_1h(
            lbank_symbol,
            BACKTEST_DAYS
        )

        if df1h is None:
            continue

        if len(df1h) < 400:

            print(
                f"⚠️ {symbol}: "
                f"داده کافی نیست "
                f"({len(df1h)})"
            )

            continue

        df4h = make_4h(
            df1h
        )

        data_1h[symbol] = df1h

        data_4h[symbol] = df4h

        print(
            f"  📊 {symbol}: "
            f"1H={len(df1h)} | "
            f"4H={len(df4h)}"
        )

    # --------------------------------------------------------
    # آماده‌سازی
    # --------------------------------------------------------

    all_trades = []

    for symbol in SYMBOLS.keys():

        if symbol not in data_1h:
            continue

        print("")

        print("-" * 70)

        df1h = calculate_indicators(
            data_1h[symbol]
        )

        df4h = data_4h[
            symbol
        ]

        df = attach_previous_4h(
            df1h,
            df4h
        )

        symbol_trades = (
            backtest_symbol(
                symbol,
                df
            )
        )

        all_trades.extend(
            symbol_trades
        )

    # --------------------------------------------------------
    # گزارش
    # --------------------------------------------------------

    print("")

    print("=" * 70)

    print(
        "📊 گزارش نهایی HUNTER-X V9"
    )

    print("=" * 70)

    if not all_trades:

        print(
            "⚠️ هیچ معامله‌ای پیدا نشد."
        )

        return

    trades_df = pd.DataFrame(
        all_trades
    )

    # --------------------------------------------------------
    # مرتب‌سازی
    # --------------------------------------------------------

    trades_df.sort_values(
        "EntryTime",
        inplace=True
    )

    # --------------------------------------------------------
    # Portfolio Lock
    # --------------------------------------------------------

    accepted = []

    portfolio_free_time = None

    for _, trade in trades_df.iterrows():

        if (
            portfolio_free_time is None
            or
            trade["EntryTime"]
            >= portfolio_free_time
        ):

            accepted.append(
                trade
            )

            portfolio_free_time = (
                trade["ExitTime"]
            )

    if not accepted:

        print(
            "⚠️ بعد از Portfolio Lock "
            "هیچ معامله‌ای باقی نماند."
        )

        return

    portfolio_df = pd.DataFrame(
        accepted
    )

    # --------------------------------------------------------
    # آمار
    # --------------------------------------------------------

    total = len(
        portfolio_df
    )

    wins = len(
        portfolio_df[
            portfolio_df["Outcome"]
            == "WIN"
        ]
    )

    losses = len(
        portfolio_df[
            portfolio_df["Outcome"]
            == "LOSS"
        ]
    )

    timeouts = len(
        portfolio_df[
            portfolio_df["Outcome"]
            == "TIMEOUT"
        ]
    )

    decisive = wins + losses

    win_rate = (
        wins / decisive * 100
        if decisive > 0
        else 0
    )

    total_r = portfolio_df[
        "R"
    ].sum()

    avg_r = portfolio_df[
        "R"
    ].mean()

    gross_profit = portfolio_df.loc[
        portfolio_df["R"] > 0,
        "R"
    ].sum()

    gross_loss = abs(
        portfolio_df.loc[
            portfolio_df["R"] < 0,
            "R"
        ].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    equity = portfolio_df[
        "R"
    ].cumsum()

    peak = equity.cummax()

    drawdown = (
        equity - peak
    )

    max_dd = drawdown.min()

    # --------------------------------------------------------
    # گزارش اصلی
    # --------------------------------------------------------

    print("")

    print(
        f"🔸 Candidate Trades: "
        f"{len(trades_df)}"
    )

    print(
        f"🔸 Accepted Trades: "
        f"{total}"
    )

    print(
        f"🔸 WIN: {wins}"
    )

    print(
        f"🔸 LOSS: {losses}"
    )

    print(
        f"🔸 TIMEOUT: {timeouts}"
    )

    print(
        f"🎯 Win Rate: "
        f"{win_rate:.2f}%"
    )

    print(
        f"📈 Profit Factor: "
        f"{profit_factor:.3f}"
    )

    print(
        f"💰 Net R: "
        f"{total_r:.2f}R"
    )

    print(
        f"📊 Average Trade: "
        f"{avg_r:.3f}R"
    )

    print(
        f"📉 Max Drawdown: "
        f"{max_dd:.2f}R"
    )

    # --------------------------------------------------------
    # تعداد معاملات سالانه
    # --------------------------------------------------------

    if len(portfolio_df) > 1:

        first_date = (
            portfolio_df[
                "EntryTime"
            ].min()
        )

        last_date = (
            portfolio_df[
                "EntryTime"
            ].max()
        )

        days = max(
            1,
            (
                last_date - first_date
            ).total_seconds()
            / 86400
        )

        trades_per_year = (
            total / days * 365
        )

    else:

        trades_per_year = total

    print(
        f"📅 Trades/Year: "
        f"{trades_per_year:.1f}"
    )

    # --------------------------------------------------------
    # Symbol
    # --------------------------------------------------------

    print("")

    print("=" * 70)

    print(
        "📌 عملکرد به تفکیک نماد"
    )

    print("=" * 70)

    symbol_stats = []

    for symbol, group in (
        portfolio_df.groupby(
            "Symbol"
        )
    ):

        sw = len(
            group[
                group["Outcome"]
                == "WIN"
            ]
        )

        sl = len(
            group[
                group["Outcome"]
                == "LOSS"
            ]
        )

        sto = len(
            group[
                group["Outcome"]
                == "TIMEOUT"
            ]
        )

        dec = sw + sl

        wr = (
            sw / dec * 100
            if dec > 0
            else 0
        )

        symbol_stats.append({
            "Symbol": symbol,
            "Trades": len(group),
            "Wins": sw,
            "Losses": sl,
            "Timeouts": sto,
            "WinRate": wr,
            "NetR": group["R"].sum()
        })

    symbol_df = pd.DataFrame(
        symbol_stats
    )

    print(
        symbol_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Long / Short
    # --------------------------------------------------------

    print("")

    print("=" * 70)

    print(
        "📌 LONG / SHORT"
    )

    print("=" * 70)

    side_stats = (
        portfolio_df
        .groupby("Side")
        .agg(
            Trades=("R", "count"),
            NetR=("R", "sum"),
            AvgR=("R", "mean")
        )
    )

    print(
        side_stats.to_string()
    )

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    print("")

    print("=" * 70)

    print(
        "📌 SETUP"
    )

    print("=" * 70)

    setup_stats = (
        portfolio_df
        .groupby("Setup")
        .agg(
            Trades=("R", "count"),
            NetR=("R", "sum"),
            AvgR=("R", "mean")
        )
    )

    print(
        setup_stats.to_string()
    )

    # --------------------------------------------------------
    # ماهانه
    # --------------------------------------------------------

    portfolio_df[
        "Month"
    ] = (
        portfolio_df[
            "EntryTime"
        ].dt.to_period("M")
    )

    monthly = (
        portfolio_df
        .groupby("Month")
        .agg(
            Trades=("R", "count"),
            NetR=("R", "sum"),
            AvgR=("R", "mean")
        )
    )

    print("")

    print("=" * 70)

    print(
        "📅 عملکرد ماهانه"
    )

    print("=" * 70)

    print(
        monthly.to_string()
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    portfolio_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("")

    print(
        f"💾 فایل معاملات ذخیره شد: "
        f"{OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    print("")

    print("=" * 70)

    print(
        "🔍 ANTI-LOOKAHEAD AUDIT"
    )

    print("=" * 70)

    print(
        "✔️ ورود در OPEN کندل بعد از سیگنال"
    )

    print(
        "✔️ دیتای 1H از LBank"
    )

    print(
        "✔️ 4H از دیتای 1H ساخته شده"
    )

    print(
        "✔️ کندل جاری حذف شده"
    )

    print(
        "✔️ فقط 4H قبلی برای فیلتر استفاده می‌شود"
    )

    print(
        "✔️ معاملات هم‌زمان روی هر نماد ممنوع"
    )

    print(
        "✔️ Portfolio Lock فعال است"
    )

    print(
        "✔️ برخورد همزمان TP/SL → LOSS"
    )

    print(
        "✔️ Fee + Slippage لحاظ شده"
    )

    print("")

    print("=" * 70)

    print(
        "✅ پایان بک‌تست"
    )

    print("=" * 70)


# ============================================================
# اجرا
# ============================================================

if __name__ == "__main__":
    main()
