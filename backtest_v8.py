import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

# ============================================================
# نصب خودکار کتابخانه‌ها
# ============================================================

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np


# ============================================================
# تنظیمات اصلی
# ============================================================

DAYS = 365

TIMEFRAME_1H = "1h"

TP_PCT = 0.02          # +2%
SL_PCT = 0.01          # -1%

RR = 2.0

VOLUME_MULTIPLIER = 1.30

BREAKOUT_LOOKBACK = 20

RSI_PERIOD = 14

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

MAX_HOLD_BARS = 24     # حداکثر 24 ساعت

LOCK_BARS = 2

# کارمزد تقریبی رفت + برگشت
FEE_RATE = 0.0006

# اسلیپیج هر طرف
SLIPPAGE_RATE = 0.0003

MAX_RETRIES = 3

FETCH_LIMIT = 1000


# ============================================================
# اتصال به LBank
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 30000,
})


# ============================================================
# سبد 10 ارز
# ============================================================

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "SUI": "SUI/USDT",
    "LINK": "LINK/USDT",
    "AVAX": "AVAX/USDT",
    "NEAR": "NEAR/USDT",
    "ADA": "ADA/USDT",
    "DOT": "DOT/USDT",
}


# ============================================================
# بازه بک‌تست
# ============================================================

end_date = datetime.utcnow()
start_date = end_date - timedelta(days=DAYS)

since_timestamp = int(start_date.timestamp() * 1000)
now_timestamp = int(end_date.timestamp() * 1000)


print("=" * 70)
print("🚀 HUNTER-X 2% / 1%")
print("📊 بک‌تست یک‌ساله LBank")
print("=" * 70)

print(f"📅 شروع: {start_date}")
print(f"📅 پایان: {end_date}")
print(f"🎯 TP: {TP_PCT * 100:.2f}%")
print(f"🛑 SL: {SL_PCT * 100:.2f}%")
print(f"⚖️ RR: 1:{RR}")
print(f"📈 تایم‌فریم ورود: 1H")
print(f"📊 تایم‌فریم روند: 4H")
print("=" * 70)


# ============================================================
# دریافت داده
# ============================================================

def download_symbol(symbol, lbank_symbol):

    print(f"\n🔹 {symbol} → شروع دانلود...")

    all_ohlcv = []

    current_since = since_timestamp

    request_count = 0

    while current_since < now_timestamp:

        success = False

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                ohlcv = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=TIMEFRAME_1H,
                    since=current_since,
                    limit=FETCH_LIMIT
                )

                if not ohlcv:
                    success = True
                    break

                # جلوگیری از داده تکراری
                if all_ohlcv:
                    last_timestamp = all_ohlcv[-1][0]
                    ohlcv = [
                        candle for candle in ohlcv
                        if candle[0] > last_timestamp
                    ]

                if not ohlcv:
                    success = True
                    break

                all_ohlcv.extend(ohlcv)

                current_since = ohlcv[-1][0] + 1

                request_count += 1

                first_dt = datetime.utcfromtimestamp(
                    ohlcv[0][0] / 1000
                )

                last_dt = datetime.utcfromtimestamp(
                    ohlcv[-1][0] / 1000
                )

                print(
                    f"   📥 Request {request_count} | "
                    f"{len(all_ohlcv)} candles | "
                    f"{first_dt} → {last_dt}"
                )

                success = True

                # اگر کمتر از limit برگشت یعنی به انتهای داده رسیده‌ایم
                if len(ohlcv) < FETCH_LIMIT:
                    current_since = now_timestamp

                break

            except Exception as e:

                print(
                    f"   ⚠️ خطا {symbol} "
                    f"(تلاش {attempt}/{MAX_RETRIES}): {e}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(3)

        if not success:

            print(
                f"   ❌ دریافت داده {symbol} "
                f"بعد از {MAX_RETRIES} تلاش شکست خورد."
            )

            return None

    if not all_ohlcv:

        print(f"   ❌ هیچ داده‌ای برای {symbol} دریافت نشد.")
        return None

    # ========================================================
    # ساخت DataFrame
    # ========================================================

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

    # حذف duplicate
    df = df.drop_duplicates(
        subset=["Timestamp"]
    )

    df = df.sort_values("Timestamp")

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
    )

    df.set_index("Date", inplace=True)

    # فقط محدوده موردنظر
    df = df[
        (df.index >= pd.Timestamp(start_date, tz="UTC")) &
        (df.index <= pd.Timestamp(end_date, tz="UTC"))
    ]

    df = df.reset_index()

    print(
        f"   ✅ {symbol} آماده شد | "
        f"{len(df)} کندل 1H"
    )

    if len(df) > 0:

        print(
            f"   🕐 {df['Date'].iloc[0]} → "
            f"{df['Date'].iloc[-1]}"
        )

    return df


# ============================================================
# محاسبه اندیکاتورها
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA20"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # Volume MA
    # --------------------------------------------------------

    df["Volume_MA20"] = df["Volume"].rolling(
        20
    ).mean()

    df["RVOL"] = (
        df["Volume"] /
        df["Volume_MA20"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        RSI_PERIOD
    ).mean()

    avg_loss = loss.rolling(
        RSI_PERIOD
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]

    tr2 = (
        df["High"] -
        previous_close
    ).abs()

    tr3 = (
        df["Low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(
        14
    ).mean()

    # --------------------------------------------------------
    # Candle strength
    # --------------------------------------------------------

    df["Body"] = (
        df["Close"] -
        df["Open"]
    ).abs()

    df["Range"] = (
        df["High"] -
        df["Low"]
    )

    df["Body_ATR"] = (
        df["Body"] /
        df["ATR"]
    )

    # --------------------------------------------------------
    # Close Location
    # --------------------------------------------------------

    df["Close_Location"] = np.where(
        df["Range"] > 0,
        (
            df["Close"] -
            df["Low"]
        ) / df["Range"],
        0.5
    )

    # --------------------------------------------------------
    # Breakout levels
    #
    # shift(1) خیلی مهم است:
    # سقف/کف فقط از کندل‌های قبلی ساخته می‌شود.
    # --------------------------------------------------------

    df["Prior20High"] = (
        df["High"]
        .rolling(BREAKOUT_LOOKBACK)
        .max()
        .shift(1)
    )

    df["Prior20Low"] = (
        df["Low"]
        .rolling(BREAKOUT_LOOKBACK)
        .min()
        .shift(1)
    )

    # --------------------------------------------------------
    # EMA slope
    # --------------------------------------------------------

    df["EMA20_Slope"] = (
        df["EMA20"] -
        df["EMA20"].shift(3)
    )

    df["EMA50_Slope"] = (
        df["EMA50"] -
        df["EMA50"].shift(3)
    )

    return df


# ============================================================
# ساخت 4H از 1H
# ============================================================

def build_4h(df1h):

    temp = df1h.copy()

    temp = temp.set_index("Date")

    df4h = temp.resample("4h").agg({

        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",

    }).dropna().reset_index()

    df4h = calculate_indicators(
        df4h
    )

    return df4h


# ============================================================
# قفل کردن Context چهار ساعته
#
# shift(1) باعث می‌شود در کندل 1H،
# از 4H در حال تشکیل استفاده نکنیم.
# ============================================================

def merge_4h_context(df1h, df4h):

    df1 = df1h.copy()

    df4 = df4h.copy()

    df4["Date"] = pd.to_datetime(
        df4["Date"],
        utc=True
    )

    # زمان کندل 4H
    df1["4H_Time"] = (
        df1["Date"]
        .dt.floor("4h")
    )

    # --------------------------------------------------------
    # بسیار مهم:
    # Context مربوط به 4H قبلی را استفاده می‌کنیم.
    # --------------------------------------------------------

    context_columns = [
        "Date",
        "Close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "ATR",
        "EMA50_Slope"
    ]

    df4_context = df4[
        context_columns
    ].copy()

    df4_context = df4_context.rename(
        columns={
            "Date": "4H_Context_Time",
            "Close": "HTF_Close",
            "EMA20": "HTF_EMA20",
            "EMA50": "HTF_EMA50",
            "EMA200": "HTF_EMA200",
            "RSI": "HTF_RSI",
            "ATR": "HTF_ATR",
            "EMA50_Slope": "HTF_EMA50_Slope",
        }
    )

    # فقط کندل 4H کاملاً بسته‌شده
    df4_context["4H_Time"] = (
        df4_context["4H_Context_Time"] +
        pd.Timedelta(hours=4)
    )

    df1 = df1.merge(
        df4_context[
            [
                "4H_Time",
                "HTF_Close",
                "HTF_EMA20",
                "HTF_EMA50",
                "HTF_EMA200",
                "HTF_RSI",
                "HTF_ATR",
                "HTF_EMA50_Slope"
            ]
        ],
        on="4H_Time",
        how="left"
    )

    return df1


# ============================================================
# بک‌تست یک نماد
# ============================================================

def backtest_symbol(symbol, df):

    trades = []

    if len(df) < 350:

        print(
            f"⚠️ {symbol}: "
            f"داده کافی نیست ({len(df)})"
        )

        return trades

    locked_until = -1

    # از 250 شروع می‌کنیم تا EMA200 و سایر اندیکاتورها آماده باشند
    for i in range(250, len(df) - 2):

        if i <= locked_until:
            continue

        row = df.iloc[i]

        # ----------------------------------------------------
        # جلوگیری از NaN
        # ----------------------------------------------------

        required = [
            "HTF_Close",
            "HTF_EMA20",
            "HTF_EMA50",
            "HTF_EMA200",
            "HTF_RSI",
            "HTF_EMA50_Slope",
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI",
            "RVOL",
            "Body_ATR",
            "Close_Location",
            "Prior20High",
            "Prior20Low",
            "EMA20_Slope",
        ]

        if any(
            pd.isna(row[x])
            for x in required
        ):
            continue

        # ====================================================
        # 4H LONG REGIME
        # ====================================================

        long_4h = (

            row["HTF_Close"] >
            row["HTF_EMA200"]

            and

            row["HTF_EMA20"] >
            row["HTF_EMA50"]

            and

            row["HTF_EMA50"] >
            row["HTF_EMA200"]

            and

            row["HTF_EMA50_Slope"] > 0

            and

            row["HTF_RSI"] >= 52
        )

        # ====================================================
        # 4H SHORT REGIME
        # ====================================================

        short_4h = (

            row["HTF_Close"] <
            row["HTF_EMA200"]

            and

            row["HTF_EMA20"] <
            row["HTF_EMA50"]

            and

            row["HTF_EMA50"] <
            row["HTF_EMA200"]

            and

            row["HTF_EMA50_Slope"] < 0

            and

            row["HTF_RSI"] <= 48
        )

        if not long_4h and not short_4h:
            continue

        # ====================================================
        # 1H TREND
        # ====================================================

        long_1h = (

            row["Close"] >
            row["EMA50"]

            and

            row["EMA20"] >
            row["EMA50"]

            and

            row["EMA50"] >
            row["EMA200"]

            and

            row["EMA20_Slope"] > 0
        )

        short_1h = (

            row["Close"] <
            row["EMA50"]

            and

            row["EMA20"] <
            row["EMA50"]

            and

            row["EMA50"] <
            row["EMA200"]

            and

            row["EMA20_Slope"] < 0
        )

        # ====================================================
        # حجم
        # ====================================================

        volume_ok = (
            row["RVOL"] >=
            VOLUME_MULTIPLIER
        )

        if not volume_ok:
            continue

        # ====================================================
        # LONG BREAKOUT
        # ====================================================

        long_breakout = (

            long_4h

            and

            long_1h

            and

            row["Close"] >
            row["Prior20High"]

            and

            row["Close"] >
            row["Open"]

            and

            row["Body_ATR"] >= 0.40

            and

            row["Close_Location"] >= 0.70

            and

            row["RSI"] >= 52
        )

        # ====================================================
        # SHORT BREAKOUT
        # ====================================================

        short_breakout = (

            short_4h

            and

            short_1h

            and

            row["Close"] <
            row["Prior20Low"]

            and

            row["Close"] <
            row["Open"]

            and

            row["Body_ATR"] >= 0.40

            and

            row["Close_Location"] <= 0.30

            and

            row["RSI"] <= 48
        )

        if not long_breakout and not short_breakout:
            continue

        # ====================================================
        # ورود در Open کندل بعدی
        # ====================================================

        entry_idx = i + 1

        if entry_idx >= len(df):
            break

        entry_candle = df.iloc[
            entry_idx
        ]

        raw_entry = float(
            entry_candle["Open"]
        )

        # Slippage هنگام ورود
        if long_breakout:
            entry_price = (
                raw_entry *
                (1 + SLIPPAGE_RATE)
            )
        else:
            entry_price = (
                raw_entry *
                (1 - SLIPPAGE_RATE)
            )

        # ====================================================
        # LONG
        # ====================================================

        if long_breakout:

            tp_price = (
                entry_price *
                (1 + TP_PCT)
            )

            sl_price = (
                entry_price *
                (1 - SL_PCT)
            )

            outcome = None
            exit_idx = None
            exit_price = None

            last_idx = min(
                entry_idx +
                MAX_HOLD_BARS,
                len(df) - 1
            )

            for j in range(
                entry_idx,
                last_idx + 1
            ):

                candle = df.iloc[j]

                # ------------------------------------------------
                # اگر هر دو در یک کندل لمس شوند:
                # SL اول -> محافظه‌کارانه
                # ------------------------------------------------

                if candle["Low"] <= sl_price:

                    outcome = "LOSS"

                    exit_idx = j

                    exit_price = (
                        sl_price *
                        (1 - SLIPPAGE_RATE)
                    )

                    break

                if candle["High"] >= tp_price:

                    outcome = "WIN"

                    exit_idx = j

                    exit_price = (
                        tp_price *
                        (1 - SLIPPAGE_RATE)
                    )

                    break

            if outcome is None:

                exit_idx = last_idx

                exit_price = float(
                    df.iloc[last_idx]["Close"]
                )

                outcome = "TIMEOUT"

            # ----------------------------------------------------
            # P/L خام
            # ----------------------------------------------------

            gross_return = (
                exit_price /
                entry_price
            ) - 1

            # ----------------------------------------------------
            # هزینه رفت و برگشت
            # ----------------------------------------------------

            net_return = (
                gross_return -
                (2 * FEE_RATE)
            )

            # R بر اساس ریسک 1%
            r_multiple = (
                net_return /
                SL_PCT
            )

            trades.append({

                "Symbol": symbol,
                "Side": "LONG",

                "Signal_Time":
                    row["Date"],

                "Entry_Time":
                    entry_candle["Date"],

                "Exit_Time":
                    df.iloc[
                        exit_idx
                    ]["Date"],

                "Entry": entry_price,
                "TP": tp_price,
                "SL": sl_price,

                "Exit": exit_price,

                "Outcome": outcome,

                "Gross_Return_%":
                    gross_return * 100,

                "Net_Return_%":
                    net_return * 100,

                "R": r_multiple,

                "Bars_Held":
                    exit_idx - entry_idx,

                "RSI_1H":
                    row["RSI"],

                "RVOL":
                    row["RVOL"],

            })

            locked_until = exit_idx + LOCK_BARS

        # ====================================================
        # SHORT
        # ====================================================

        elif short_breakout:

            tp_price = (
                entry_price *
                (1 - TP_PCT)
            )

            sl_price = (
                entry_price *
                (1 + SL_PCT)
            )

            outcome = None
            exit_idx = None
            exit_price = None

            last_idx = min(
                entry_idx +
                MAX_HOLD_BARS,
                len(df) - 1
            )

            for j in range(
                entry_idx,
                last_idx + 1
            ):

                candle = df.iloc[j]

                # SL اول
                if candle["High"] >= sl_price:

                    outcome = "LOSS"

                    exit_idx = j

                    exit_price = (
                        sl_price *
                        (1 + SLIPPAGE_RATE)
                    )

                    break

                if candle["Low"] <= tp_price:

                    outcome = "WIN"

                    exit_idx = j

                    exit_price = (
                        tp_price *
                        (1 + SLIPPAGE_RATE)
                    )

                    break

            if outcome is None:

                exit_idx = last_idx

                exit_price = float(
                    df.iloc[last_idx]["Close"]
                )

                outcome = "TIMEOUT"

            # ----------------------------------------------------
            # P/L SHORT
            # ----------------------------------------------------

            gross_return = (
                entry_price /
                exit_price
            ) - 1

            net_return = (
                gross_return -
                (2 * FEE_RATE)
            )

            r_multiple = (
                net_return /
                SL_PCT
            )

            trades.append({

                "Symbol": symbol,
                "Side": "SHORT",

                "Signal_Time":
                    row["Date"],

                "Entry_Time":
                    entry_candle["Date"],

                "Exit_Time":
                    df.iloc[
                        exit_idx
                    ]["Date"],

                "Entry": entry_price,
                "TP": tp_price,
                "SL": sl_price,

                "Exit": exit_price,

                "Outcome": outcome,

                "Gross_Return_%":
                    gross_return * 100,

                "Net_Return_%":
                    net_return * 100,

                "R": r_multiple,

                "Bars_Held":
                    exit_idx - entry_idx,

                "RSI_1H":
                    row["RSI"],

                "RVOL":
                    row["RVOL"],

            })

            locked_until = exit_idx + LOCK_BARS

    return trades


# ============================================================
# دانلود تمام داده‌ها
# ============================================================

data_1h = {}
data_4h = {}

print("\n")
print("=" * 70)
print("📥 مرحله 1: دریافت داده‌های واقعی LBank")
print("=" * 70)

for symbol, lbank_symbol in SYMBOLS.items():

    df1h = download_symbol(
        symbol,
        lbank_symbol
    )

    if df1h is None:
        continue

    if len(df1h) < 350:

        print(
            f"⚠️ {symbol}: "
            f"داده کافی نیست."
        )

        continue

    data_1h[symbol] = df1h

    print(
        f"   🔄 ساخت داده 4H برای {symbol}..."
    )

    df4h = build_4h(
        df1h
    )

    data_4h[symbol] = df4h

    print(
        f"   ✔️ 1H = {len(df1h)} | "
        f"4H = {len(df4h)}"
    )


# ============================================================
# اجرای بک‌تست
# ============================================================

print("\n")
print("=" * 70)
print("🚀 مرحله 2: اجرای بک‌تست HUNTER-X 2% / 1%")
print("=" * 70)

all_trades = []


for symbol in SYMBOLS.keys():

    if symbol not in data_1h:
        print(
            f"⚠️ {symbol}: "
            f"داده موجود نیست."
        )
        continue

    print(
        f"\n🔬 بک‌تست {symbol}..."
    )

    df1h = calculate_indicators(
        data_1h[symbol]
    )

    df4h = data_4h[symbol]

    # اتصال Context چهار ساعته
    df = merge_4h_context(
        df1h,
        df4h
    )

    trades = backtest_symbol(
        symbol,
        df
    )

    all_trades.extend(
        trades
    )

    print(
        f"   ✅ {symbol}: "
        f"{len(trades)} معامله"
    )


# ============================================================
# گزارش نهایی
# ============================================================

print("\n")
print("=" * 70)
print("📊 گزارش نهایی HUNTER-X 2% / 1%")
print("=" * 70)


if not all_trades:

    print(
        "❌ هیچ معامله‌ای پیدا نشد."
    )

    sys.exit(0)


pf = pd.DataFrame(
    all_trades
)


# ============================================================
# آمار کلی
# ============================================================

total = len(pf)

wins = len(
    pf[
        pf["Outcome"] == "WIN"
    ]
)

losses = len(
    pf[
        pf["Outcome"] == "LOSS"
    ]
)

timeouts = len(
    pf[
        pf["Outcome"] == "TIMEOUT"
    ]
)

resolved = wins + losses

win_rate_all = (
    wins / total * 100
    if total > 0
    else 0
)

win_rate_resolved = (
    wins / resolved * 100
    if resolved > 0
    else 0
)

total_R = pf["R"].sum()

avg_R = pf["R"].mean()

gross_profit_R = pf.loc[
    pf["R"] > 0,
    "R"
].sum()

gross_loss_R = abs(
    pf.loc[
        pf["R"] < 0,
        "R"
    ].sum()
)

profit_factor = (
    gross_profit_R /
    gross_loss_R
    if gross_loss_R > 0
    else np.inf
)

avg_win_R = (
    pf.loc[
        pf["Outcome"] == "WIN",
        "R"
    ].mean()
    if wins > 0
    else 0
)

avg_loss_R = (
    pf.loc[
        pf["Outcome"] == "LOSS",
        "R"
    ].mean()
    if losses > 0
    else 0
)


# ============================================================
# Max Drawdown
# ============================================================

equity = pf["R"].cumsum()

peak = equity.cummax()

drawdown = equity - peak

max_drawdown_R = drawdown.min()


# ============================================================
# تعداد معاملات روزانه
# ============================================================

pf["DateOnly"] = (
    pd.to_datetime(
        pf["Entry_Time"]
    ).dt.date
)

daily_trades = (
    pf.groupby("DateOnly")
    .size()
)

avg_daily_trades = (
    daily_trades.mean()
    if len(daily_trades) > 0
    else 0
)

max_daily_trades = (
    daily_trades.max()
    if len(daily_trades) > 0
    else 0
)


# ============================================================
# چاپ گزارش
# ============================================================

print(
    f"🔸 تعداد کل معاملات      : {total}"
)

print(
    f"🟢 بردها (WIN)           : {wins}"
)

print(
    f"🔴 باخت‌ها (LOSS)        : {losses}"
)

print(
    f"🟡 Timeout               : {timeouts}"
)

print(
    f"🎯 Win Rate کل           : {win_rate_all:.2f}%"
)

print(
    f"🎯 Win Rate معاملات بسته : {win_rate_resolved:.2f}%"
)

print(
    f"💰 Net Profit            : {total_R:.2f}R"
)

print(
    f"📈 Profit Factor         : {profit_factor:.2f}"
)

print(
    f"📊 Expectancy            : {avg_R:.4f}R"
)

print(
    f"🟢 Average Win           : {avg_win_R:.2f}R"
)

print(
    f"🔴 Average Loss          : {avg_loss_R:.2f}R"
)

print(
    f"📉 Max Drawdown          : {max_drawdown_R:.2f}R"
)

print(
    f"📅 میانگین معامله/روز    : {avg_daily_trades:.2f}"
)

print(
    f"🔥 بیشترین معامله/روز    : {max_daily_trades}"
)


# ============================================================
# عملکرد هر ارز
# ============================================================

print("\n")
print("=" * 70)
print("📊 عملکرد تک‌تک ارزها")
print("=" * 70)


summary = []


for symbol in SYMBOLS.keys():

    sdf = pf[
        pf["Symbol"] == symbol
    ]

    if sdf.empty:
        continue

    s_total = len(sdf)

    s_wins = len(
        sdf[
            sdf["Outcome"] == "WIN"
        ]
    )

    s_losses = len(
        sdf[
            sdf["Outcome"] == "LOSS"
        ]
    )

    s_timeout = len(
        sdf[
            sdf["Outcome"] == "TIMEOUT"
        ]
    )

    s_resolved = (
        s_wins +
        s_losses
    )

    s_wr = (
        s_wins /
        s_resolved *
        100
        if s_resolved > 0
        else 0
    )

    s_R = sdf["R"].sum()

    s_pf_profit = sdf.loc[
        sdf["R"] > 0,
        "R"
    ].sum()

    s_pf_loss = abs(
        sdf.loc[
            sdf["R"] < 0,
            "R"
        ].sum()
    )

    s_pf = (
        s_pf_profit /
        s_pf_loss
        if s_pf_loss > 0
        else np.inf
    )

    summary.append({

        "Symbol": symbol,

        "Trades": s_total,

        "Wins": s_wins,

        "Losses": s_losses,

        "Timeout": s_timeout,

        "WinRate_%": s_wr,

        "Net_R": s_R,

        "ProfitFactor": s_pf,

    })


summary_df = pd.DataFrame(
    summary
)

summary_df = summary_df.sort_values(
    "Net_R",
    ascending=False
)


print(
    summary_df.to_string(
        index=False,
        formatters={
            "WinRate_%":
                "{:.2f}".format,
            "Net_R":
                "{:.2f}".format,
            "ProfitFactor":
                "{:.2f}".format,
        }
    )
)


# ============================================================
# Long / Short
# ============================================================

print("\n")
print("=" * 70)
print("📈 عملکرد LONG / SHORT")
print("=" * 70)


for side in ["LONG", "SHORT"]:

    sdf = pf[
        pf["Side"] == side
    ]

    if sdf.empty:
        continue

    sw = len(
        sdf[
            sdf["Outcome"] == "WIN"
        ]
    )

    sl = len(
        sdf[
            sdf["Outcome"] == "LOSS"
        ]
    )

    sr = sw + sl

    swr = (
        sw /
        sr *
        100
        if sr > 0
        else 0
    )

    print(
        f"{side:5s} | "
        f"Trades: {len(sdf):4d} | "
        f"W: {sw:4d} | "
        f"L: {sl:4d} | "
        f"WR: {swr:6.2f}% | "
        f"R: {sdf['R'].sum():8.2f}"
    )


# ============================================================
# ذخیره معاملات
# ============================================================

output_file = (
    "hunter_x_2pct_1pct_trades.csv"
)

pf.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 70)
print(
    f"💾 فایل معاملات ذخیره شد: "
    f"{output_file}"
)
print("=" * 70)

print("\n✨ بک‌تست با موفقیت تمام شد.")
