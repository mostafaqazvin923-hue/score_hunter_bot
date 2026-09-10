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

import pandas as pd
import numpy as np


# ============================================================
# صرافی LBank با سبد 10 ارز
# ============================================================

exchange = ccxt.lbank({'enableRateLimit': True})

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


# ============================================================
# دریافت داده یک سال گذشته
# ============================================================

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌های 1 ساعته و ساخت کندل‌های 4 ساعته")
print("   نسخه اصلاح‌شده ضد Look-ahead")
print("============================================================")

data_1h = {}


for symbol, lbank_symbol in SYMBOLS.items():

    filename_1h = f"{symbol}_1h_expanded_data.csv"

    print(f"🔹 در حال دریافت دیتای 1 ساعته {symbol}...")

    all_ohlcv = []
    current_since = since_timestamp
    now_timestamp = exchange.milliseconds()

    while current_since < now_timestamp:

        try:

            ohlcv = exchange.fetch_ohlcv(
                lbank_symbol,
                timeframe='1h',
                since=current_since,
                limit=1000
            )

            if not ohlcv:
                break

            current_since = ohlcv[-1][0] + 1

            all_ohlcv.extend(ohlcv)

            if len(ohlcv) < 1000:
                break

        except Exception as e:

            print(f"  ❌ خطا در دریافت داده {symbol}: {e}")
            break


    if all_ohlcv:

        df1h = pd.DataFrame(
            all_ohlcv,
            columns=[
                'Timestamp',
                'Open',
                'High',
                'Low',
                'Close',
                'Volume'
            ]
        )

        df1h['Date'] = pd.to_datetime(
            df1h['Timestamp'],
            unit='ms'
        )

        df1h = df1h[
            [
                'Date',
                'Open',
                'High',
                'Low',
                'Close',
                'Volume'
            ]
        ]

        df1h.dropna(inplace=True)

        df1h.drop_duplicates(
            subset=['Date'],
            inplace=True
        )

        df1h.sort_values(
            'Date',
            inplace=True
        )

        df1h.reset_index(
            drop=True,
            inplace=True
        )

        df1h.to_csv(
            filename_1h,
            index=False
        )

        data_1h[symbol] = df1h

        print(
            f"  ✔️ دیتای 1 ساعته {symbol} آماده شد "
            f"(تعداد کندل: {len(df1h)})"
        )

    else:

        print(
            f"  ❌ دیتایی برای {symbol} دریافت نشد."
        )


# ============================================================
# محاسبه اندیکاتورها
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df['EMA_20'] = (
        df['Close']
        .ewm(span=20, adjust=False)
        .mean()
    )

    df['EMA_50'] = (
        df['Close']
        .ewm(span=50, adjust=False)
        .mean()
    )

    df['EMA_200'] = (
        df['Close']
        .ewm(span=200, adjust=False)
        .mean()
    )

    # RSI
    delta = df['Close'].diff()

    gain = (
        delta
        .where(delta > 0, 0)
        .rolling(window=14)
        .mean()
    )

    loss = (
        -delta
        .where(delta < 0, 0)
        .rolling(window=14)
        .mean()
    )

    rs = gain / loss

    df['RSI'] = (
        100 - (100 / (1 + rs))
    )

    # ATR
    high_low = (
        df['High'] - df['Low']
    )

    high_close = np.abs(
        df['High'] - df['Close'].shift()
    )

    low_close = np.abs(
        df['Low'] - df['Close'].shift()
    )

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df['ATR'] = (
        tr
        .rolling(window=14)
        .mean()
    )

    # ADX
    plus_dm = (
        df['High']
        .diff()
        .clip(lower=0)
    )

    minus_dm = (
        -df['Low']
        .diff()
        .clip(lower=0)
    )

    tr14 = (
        tr
        .rolling(window=14)
        .mean()
    )

    plus_di = (
        100 *
        (
            plus_dm
            .rolling(window=14)
            .mean()
            / tr14
        )
    )

    minus_di = (
        100 *
        (
            minus_dm
            .rolling(window=14)
            .mean()
            / tr14
        )
    )

    dx = (
        100 *
        np.abs(plus_di - minus_di)
        /
        (
            plus_di +
            minus_di +
            1e-9
        )
    )

    df['ADX'] = (
        dx
        .rolling(window=14)
        .mean()
        .fillna(20)
    )

    return df


# ============================================================
# اجرای موتور بک‌تست
# ============================================================

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست با فیلترهای متعادل‌تر")
print("   ضد Look-ahead / بدون استفاده از اطلاعات آینده")
print("============================================================")

all_portfolio_trades = []


for symbol, df1h_original in data_1h.items():

    if len(df1h_original) < 300:
        continue


    # --------------------------------------------------------
    # کپی داده 1H
    # --------------------------------------------------------

    df1h = df1h_original.copy()

    df1h = calculate_indicators(df1h)


    # --------------------------------------------------------
    # ساخت کندل‌های 4H
    # --------------------------------------------------------

    df4h = (
        df1h
        .set_index('Date')
        .resample('4h')
        .agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        .dropna()
        .reset_index()
    )

    df4h = calculate_indicators(df4h)


    # --------------------------------------------------------
    # ایجاد زمان کندل 4H برای هر کندل 1H
    # --------------------------------------------------------

    df1h['Date_4H'] = (
        df1h['Date']
        .dt.floor('4h')
    )


    # --------------------------------------------------------
    # ایندکس 4H
    # --------------------------------------------------------

    df4h_indexed = (
        df4h
        .set_index('Date')
    )


    # --------------------------------------------------------
    # قفل معاملات هر نماد
    # --------------------------------------------------------

    locked_until_index = 0


    # ========================================================
    # موتور اصلی
    # ========================================================

    for i in range(
        200,
        len(df1h) - 40
    ):

        if i < locked_until_index:
            continue


        c1h = df1h.iloc[i]

        t4h_time = c1h['Date_4H']


        # ----------------------------------------------------
        # مهم:
        # استفاده فقط از آخرین کندل 4H کاملاً بسته‌شده
        # ----------------------------------------------------

        current_4h_position = (
            df4h.index[
                df4h['Date'] == t4h_time
            ]
        )


        if len(current_4h_position) == 0:
            continue


        current_4h_idx = (
            current_4h_position[0]
        )


        # اگر کندل 4H فعلی در حال تشکیل است،
        # از کندل 4H قبلی استفاده می‌کنیم.
        #
        # مثال:
        #
        # 08:00 تا 12:00 = کندل فعلی
        # ساعت 10:00 هستیم
        #
        # اطلاعات 08 تا 12 هنوز کامل نشده.
        #
        # بنابراین فقط 04 تا 08 مجاز است.

        if current_4h_idx <= 0:
            continue


        closed_4h_idx = (
            current_4h_idx - 1
        )


        r4h = (
            df4h.iloc[
                closed_4h_idx
            ]
        )


        # ====================================================
        # اندیکاتورهای 4H
        # ====================================================

        ema20_4h = r4h['EMA_20']

        ema50_4h = r4h['EMA_50']

        ema200_4h = r4h['EMA_200']

        rsi_4h = r4h['RSI']

        adx_4h = r4h['ADX']


        # ----------------------------------------------------
        # اصلاح واقعی EMA200 Slope
        # ----------------------------------------------------

        if closed_4h_idx > 0:

            prev_ema200_4h = (
                df4h.iloc[
                    closed_4h_idx - 1
                ]['EMA_200']
            )

            slope_positive = (
                ema200_4h >=
                prev_ema200_4h
            )

        else:

            slope_positive = True


        # ====================================================
        # رژیم LONG
        # ====================================================

        is_long_regime = (

            (r4h['Close'] > ema200_4h)

            and

            (ema20_4h > ema50_4h)

            and

            slope_positive

            and

            (adx_4h >= 17)

            and

            (rsi_4h > 50)
        )


        # ====================================================
        # رژیم SHORT
        # ====================================================

        is_short_regime = (

            (r4h['Close'] < ema200_4h)

            and

            (ema20_4h < ema50_4h)

            and

            (adx_4h >= 17)

            and

            (rsi_4h < 50)
        )


        if (
            not is_long_regime
            and
            not is_short_regime
        ):
            continue


        # ====================================================
        # ساختار 1H
        # ====================================================

        lookback_slice = (
            df1h.iloc[
                i - 15:i
            ]
        )


        struct_high = (
            lookback_slice['High']
            .max()
        )

        struct_low = (
            lookback_slice['Low']
            .min()
        )


        avg_vol = (
            lookback_slice['Volume']
            .mean()
        )


        # ====================================================
        # Breakout LONG
        # ====================================================

        is_breakout_long = (

            (c1h['Close'] > struct_high)

            and

            (
                c1h['Volume']
                >=
                avg_vol * 0.95
            )
        )


        # ====================================================
        # Breakout SHORT
        # ====================================================

        is_breakout_short = (

            (c1h['Close'] < struct_low)

            and

            (
                c1h['Volume']
                >=
                avg_vol * 0.95
            )
        )


        # ====================================================
        # LONG
        # ====================================================

        if (
            is_long_regime
            and
            is_breakout_long
        ):

            entered = False


            for p in range(1, 14):

                if (
                    i + p
                    >=
                    len(df1h) - 10
                ):
                    break


                p_candle = (
                    df1h.iloc[
                        i + p
                    ]
                )


                # ------------------------------------------------
                # Pullback
                # ------------------------------------------------

                if (
                    p_candle['Low']
                    <=
                    struct_high * 1.004
                ):

                    # --------------------------------------------
                    # Confirmation
                    # --------------------------------------------

                    if (
                        p_candle['Close']
                        >
                        p_candle['Open']

                        and

                        p_candle['RSI']
                        >
                        48
                    ):

                        # ----------------------------------------
                        # Entry
                        # ----------------------------------------

                        entry_price = (
                            p_candle['Close']
                        )


                        # ----------------------------------------
                        # Swing Low
                        # ----------------------------------------

                        swing_low_pullback = (
                            df1h.iloc[
                                i:i + p + 1
                            ]['Low'].min()
                        )


                        # ----------------------------------------
                        # Stop Loss
                        # ----------------------------------------

                        sl = (
                            swing_low_pullback
                            -
                            (
                                0.25 *
                                p_candle['ATR']
                            )
                        )


                        risk = (
                            entry_price -
                            sl
                        )


                        if (
                            risk <= 0
                            or
                            (
                                risk /
                                entry_price
                            ) > 0.05
                        ):
                            break


                        # ----------------------------------------
                        # TP = 2R
                        # ----------------------------------------

                        tp = (
                            entry_price
                            +
                            (
                                2.0 *
                                risk
                            )
                        )


                        # ====================================================
                        # FUTURE_WINDOW حذف شد
                        #
                        # دیگر قبل از ثبت معامله بررسی نمی‌کنیم
                        # که آیا TP در آینده لمس خواهد شد یا خیر.
                        #
                        # این مهم‌ترین اصلاح ضد Look-ahead است.
                        # ====================================================


                        outcome = 'OPEN'

                        exit_idx = (
                            i + p + 1
                        )


                        # ----------------------------------------
                        # مدیریت معامله
                        # ----------------------------------------

                        for j in range(
                            i + p + 1,
                            min(
                                i + p + 40,
                                len(df1h)
                            )
                        ):

                            f_c = (
                                df1h.iloc[j]
                            )

                            exit_idx = j


                            # ------------------------------------
                            # SL اول
                            # ------------------------------------

                            if (
                                f_c['Low']
                                <=
                                sl
                            ):

                                outcome = 'LOSS'
                                break


                            # ------------------------------------
                            # TP
                            # ------------------------------------

                            elif (
                                f_c['High']
                                >=
                                tp
                            ):

                                outcome = 'WIN'
                                break


                        # ----------------------------------------
                        # ثبت معامله
                        # ----------------------------------------

                        if outcome in [
                            'WIN',
                            'LOSS'
                        ]:

                            all_portfolio_trades.append({

                                'Symbol':
                                    symbol,

                                'Side':
                                    'LONG',

                                'Outcome':
                                    outcome

                            })


                            locked_until_index = (
                                exit_idx
                            )


                            entered = True

                            break


            if entered:
                continue


        # ====================================================
        # SHORT
        # ====================================================

        elif (
            is_short_regime
            and
            is_breakout_short
        ):

            entered = False


            for p in range(1, 14):

                if (
                    i + p
                    >=
                    len(df1h) - 10
                ):
                    break


                p_candle = (
                    df1h.iloc[
                        i + p
                    ]
                )


                # ------------------------------------------------
                # Pullback
                # ------------------------------------------------

                if (
                    p_candle['High']
                    >=
                    struct_low * 0.996
                ):

                    # --------------------------------------------
                    # Confirmation
                    # --------------------------------------------

                    if (
                        p_candle['Close']
                        <
                        p_candle['Open']

                        and

                        p_candle['RSI']
                        <
                        52
                    ):

                        # ----------------------------------------
                        # Entry
                        # ----------------------------------------

                        entry_price = (
                            p_candle['Close']
                        )


                        # ----------------------------------------
                        # Swing High
                        # ----------------------------------------

                        swing_high_pullback = (
                            df1h.iloc[
                                i:i + p + 1
                            ]['High'].max()
                        )


                        # ----------------------------------------
                        # Stop Loss
                        # ----------------------------------------

                        sl = (
                            swing_high_pullback
                            +
                            (
                                0.25 *
                                p_candle['ATR']
                            )
                        )


                        risk = (
                            sl -
                            entry_price
                        )


                        if (
                            risk <= 0
                            or
                            (
                                risk /
                                entry_price
                            ) > 0.05
                        ):
                            break


                        # ----------------------------------------
                        # TP = 2R
                        # ----------------------------------------

                        tp = (
                            entry_price
                            -
                            (
                                2.0 *
                                risk
                            )
                        )


                        # ====================================================
                        # FUTURE_WINDOW حذف شد
                        # ====================================================


                        outcome = 'OPEN'

                        exit_idx = (
                            i + p + 1
                        )


                        # ----------------------------------------
                        # مدیریت معامله
                        # ----------------------------------------

                        for j in range(
                            i + p + 1,
                            min(
                                i + p + 40,
                                len(df1h)
                            )
                        ):

                            f_c = (
                                df1h.iloc[j]
                            )

                            exit_idx = j


                            # ------------------------------------
                            # SL اول
                            # ------------------------------------

                            if (
                                f_c['High']
                                >=
                                sl
                            ):

                                outcome = 'LOSS'
                                break


                            # ------------------------------------
                            # TP
                            # ------------------------------------

                            elif (
                                f_c['Low']
                                <=
                                tp
                            ):

                                outcome = 'WIN'
                                break


                        # ----------------------------------------
                        # ثبت معامله
                        # ----------------------------------------

                        if outcome in [
                            'WIN',
                            'LOSS'
                        ]:

                            all_portfolio_trades.append({

                                'Symbol':
                                    symbol,

                                'Side':
                                    'SHORT',

                                'Outcome':
                                    outcome

                            })


                            locked_until_index = (
                                exit_idx
                            )


                            entered = True

                            break


# ============================================================
# گزارش نهایی
# ============================================================

print("\n============================================================")
print("📊 گزارش تجمیعی نهایی پورتفوی")
print("   نسخه اصلاح‌شده ضد Look-ahead")
print("============================================================")


if all_portfolio_trades:

    pf_df = pd.DataFrame(
        all_portfolio_trades
    )


    total_trades = len(
        pf_df
    )


    total_wins = len(
        pf_df[
            pf_df['Outcome'] == 'WIN'
        ]
    )


    total_losses = len(
        pf_df[
            pf_df['Outcome'] == 'LOSS'
        ]
    )


    portfolio_win_rate = (
        total_wins /
        total_trades
    ) * 100 if total_trades > 0 else 0


    # RR = 1:2
    net_profit_score = (
        total_wins * 2.0
    ) - total_losses


    print(
        f"🔸 تعداد کل معاملات کل سبد (پورتفوی): "
        f"{total_trades}"
    )

    print(
        f"🔸 کل معاملات برنده (WIN): "
        f"{total_wins}"
    )

    print(
        f"🔸 کل معاملات بازنده (LOSS): "
        f"{total_losses}"
    )

    print(
        f"🎯 وین‌ریت تجمیعی کل پورتفوی: "
        f"{portfolio_win_rate:.2f}%"
    )

    print(
        f"💰 امتیاز سودآوری خالص: "
        f"{net_profit_score:.2f}R"
    )


    print(
        "\nتفکیک عملکرد به تفکیک هر نماد:"
    )

    print(
        pf_df
        .groupby('Symbol')['Outcome']
        .value_counts()
        .unstack(fill_value=0)
    )


else:

    print(
        "⚠️ هیچ معامله‌ای با شرایط ثبت نشد."
    )


print(
    "\n✨ بک‌تست اصلاح‌شده به اتمام رسید."
)

print(
    "🔒 اطلاعات آینده برای انتخاب معامله استفاده نشده است."
)
