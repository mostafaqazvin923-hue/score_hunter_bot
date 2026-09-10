import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

# ============================================================
# نصب ccxt در صورت نیاز
# ============================================================

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np


# ============================================================
# تنظیمات اصلی
# ============================================================

exchange = ccxt.lbank({
    'enableRateLimit': True
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

LOOKBACK_DAYS = 365

# Strategy parameters — V10 baseline
DONCHIAN_LENGTH = 24
ATR_LENGTH = 14
ATR_MA_LENGTH = 50
ADX_LENGTH = 14
VOLUME_MA_LENGTH = 20

ADX_THRESHOLD = 18

ATR_MIN_RATIO = 0.90
ATR_MAX_RATIO = 2.50

BREAKOUT_BODY_ATR = 0.35
LONG_CLOSE_LOCATION = 0.65
SHORT_CLOSE_LOCATION = 0.35
VOLUME_MULTIPLIER = 1.10

STOP_ATR_MULTIPLIER = 1.50
TARGET_RR = 2.00

COOLDOWN_BARS = 6
MAX_HOLD_BARS = 80

# 0.08% total round-trip = 0.04% per side
COMMISSION_RATE = 0.0008
COMMISSION_PER_SIDE = COMMISSION_RATE / 2

# 0.04% slippage per side
SLIPPAGE_RATE = 0.04 / 100


# ============================================================
# دریافت داده
# ============================================================

start_date = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌ها برای HUNTER-X V10 — Pristine Final")
print("============================================================")


data_1h = {}


for symbol, lbank_symbol in SYMBOLS.items():

    filename_1h = f"{symbol}_1h_v10_pristine_final.csv"

    print(f"\n🔹 در حال دریافت دیتای 1 ساعته {symbol}...")

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

            all_ohlcv.extend(ohlcv)

            last_timestamp = ohlcv[-1][0]

            # جلوگیری از گیر کردن pagination
            next_since = last_timestamp + 1

            if next_since <= current_since:
                break

            current_since = next_since

            if len(ohlcv) < 1000:
                break

        except Exception as e:

            print(f"  ❌ خطا در دریافت داده {symbol}: {e}")
            break


    if not all_ohlcv:

        print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")
        continue


    # ========================================================
    # ساخت DataFrame
    # ========================================================

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

    # UTC
    df1h['Date'] = pd.to_datetime(
        df1h['Timestamp'],
        unit='ms',
        utc=True
    ).dt.tz_localize(None)

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


    # ========================================================
    # حذف آخرین کندل ناقص 1H
    # ========================================================

    current_hour_utc = (
        pd.Timestamp.now(tz='UTC')
        .floor('h')
        .tz_localize(None)
    )

    df1h = df1h[
        df1h['Date'] < current_hour_utc
    ].copy()

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
        f"  ✔️ {symbol}: "
        f"{len(df1h)} کندل 1H آماده شد."
    )


# ============================================================
# محاسبه اندیکاتورها
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df['EMA_20'] = (
        df['Close']
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df['EMA_50'] = (
        df['Close']
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df['EMA_200'] = (
        df['Close']
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )


    # --------------------------------------------------------
    # True Range
    # --------------------------------------------------------

    high_low = (
        df['High'] - df['Low']
    )

    high_close = np.abs(
        df['High'] - df['Close'].shift(1)
    )

    low_close = np.abs(
        df['Low'] - df['Close'].shift(1)
    )

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)


    # --------------------------------------------------------
    # ATR — Wilder style
    # --------------------------------------------------------

    df['ATR'] = (
        tr
        .ewm(
            alpha=1 / ATR_LENGTH,
            adjust=False
        )
        .mean()
    )

    df['ATR_MA'] = (
        df['ATR']
        .rolling(
            ATR_MA_LENGTH
        )
        .mean()
    )


    # --------------------------------------------------------
    # ADX — Wilder style
    # --------------------------------------------------------

    up_move = df['High'].diff()

    down_move = -df['Low'].diff()

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0.0
    )


    plus_dm_smooth = (
        pd.Series(
            plus_dm,
            index=df.index
        )
        .ewm(
            alpha=1 / ADX_LENGTH,
            adjust=False
        )
        .mean()
    )

    minus_dm_smooth = (
        pd.Series(
            minus_dm,
            index=df.index
        )
        .ewm(
            alpha=1 / ADX_LENGTH,
            adjust=False
        )
        .mean()
    )

    tr_smooth = (
        tr
        .ewm(
            alpha=1 / ADX_LENGTH,
            adjust=False
        )
        .mean()
    )


    plus_di = (
        100 *
        plus_dm_smooth /
        tr_smooth.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm_smooth /
        tr_smooth.replace(0, np.nan)
    )


    dx = (
        100 *
        np.abs(
            plus_di - minus_di
        ) /
        (
            plus_di +
            minus_di +
            1e-9
        )
    )


    df['ADX'] = (
        dx
        .ewm(
            alpha=1 / ADX_LENGTH,
            adjust=False
        )
        .mean()
    )


    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df['Vol_MA'] = (
        df['Volume']
        .rolling(
            VOLUME_MA_LENGTH
        )
        .mean()
    )


    # --------------------------------------------------------
    # Donchian
    # IMPORTANT:
    # current candle is excluded
    # --------------------------------------------------------

    df['Donchian_High_24'] = (
        df['High']
        .shift(1)
        .rolling(DONCHIAN_LENGTH)
        .max()
    )

    df['Donchian_Low_24'] = (
        df['Low']
        .shift(1)
        .rolling(DONCHIAN_LENGTH)
        .min()
    )


    return df


# ============================================================
# موتور بک‌تست
# ============================================================

print("\n============================================================")
print("🚀 اجرای HUNTER-X V10 — Pristine Final")
print("============================================================")


all_trades = []


for symbol, original_df1h in data_1h.items():

    if len(original_df1h) < 500:

        print(
            f"⚠️ {symbol}: "
            f"داده کافی نیست."
        )

        continue


    # --------------------------------------------------------
    # 1H indicators
    # --------------------------------------------------------

    df1h = calculate_indicators(
        original_df1h
    )


    # --------------------------------------------------------
    # 4H resampling
    # --------------------------------------------------------

    df4h = (
        df1h
        .set_index('Date')
        .resample(
            '4h',
            label='left',
            closed='left'
        )
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


    df4h = calculate_indicators(
        df4h
    )


    # --------------------------------------------------------
    # IS / OOS split
    # --------------------------------------------------------

    split_idx = int(
        len(df1h) * 0.80
    )


    locked_until_index = 0

    i = 200


    # آخرین سیگنالی که می‌تواند
    # حداقل MAX_HOLD_BARS کندل آینده داشته باشد
    last_valid_signal_index = (
        len(df1h)
        - MAX_HOLD_BARS
        - 2
    )


    while i <= last_valid_signal_index:


        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if i < locked_until_index:

            i += 1
            continue


        # ----------------------------------------------------
        # Signal candle
        # ----------------------------------------------------

        c1h = df1h.iloc[i]

        current_time = c1h['Date']


        # ----------------------------------------------------
        # آخرین 4H کاملاً بسته‌شده
        # ----------------------------------------------------

        closed_4h_time = (
            current_time -
            timedelta(hours=4)
        )

        available_4h = df4h[
            df4h['Date'] <= closed_4h_time
        ]


        if len(available_4h) < 2:

            i += 1
            continue


        r4h = available_4h.iloc[-1]

        prev_r4h = available_4h.iloc[-2]


        # ----------------------------------------------------
        # جلوگیری از استفاده از ADX نامعتبر
        # ----------------------------------------------------

        if (
            pd.isna(r4h['ADX']) or
            pd.isna(r4h['EMA_200']) or
            pd.isna(r4h['EMA_50'])
        ):

            i += 1
            continue


        # ----------------------------------------------------
        # EMA200 slope
        # ----------------------------------------------------

        ema200_slope_up = (
            r4h['EMA_200'] >=
            prev_r4h['EMA_200']
        )

        ema200_slope_down = (
            r4h['EMA_200'] <=
            prev_r4h['EMA_200']
        )


        # ----------------------------------------------------
        # 4H regime
        # ----------------------------------------------------

        is_long_regime = (
            (r4h['Close'] > r4h['EMA_200']) and
            (r4h['EMA_50'] > r4h['EMA_200']) and
            (r4h['ADX'] >= ADX_THRESHOLD) and
            ema200_slope_up
        )


        is_short_regime = (
            (r4h['Close'] < r4h['EMA_200']) and
            (r4h['EMA_50'] < r4h['EMA_200']) and
            (r4h['ADX'] >= ADX_THRESHOLD) and
            ema200_slope_down
        )


        if (
            not is_long_regime and
            not is_short_regime
        ):

            i += 1
            continue


        # ----------------------------------------------------
        # 1H volatility
        # ----------------------------------------------------

        if (
            pd.isna(c1h['ATR']) or
            pd.isna(c1h['ATR_MA']) or
            c1h['ATR_MA'] <= 0
        ):

            i += 1
            continue


        volatility_ratio = (
            c1h['ATR'] /
            c1h['ATR_MA']
        )


        pass_volatility = (
            volatility_ratio >= ATR_MIN_RATIO and
            volatility_ratio <= ATR_MAX_RATIO
        )


        if not pass_volatility:

            i += 1
            continue


        # ----------------------------------------------------
        # Breakout candle
        # ----------------------------------------------------

        body_size = abs(
            c1h['Close'] -
            c1h['Open']
        )

        total_range = (
            c1h['High'] -
            c1h['Low']
        )


        if total_range <= 0:

            i += 1
            continue


        close_location = (
            c1h['Close'] -
            c1h['Low']
        ) / total_range


        # ----------------------------------------------------
        # Volume / Donchian validity
        # ----------------------------------------------------

        if (
            pd.isna(c1h['Donchian_High_24']) or
            pd.isna(c1h['Donchian_Low_24']) or
            pd.isna(c1h['Vol_MA'])
        ):

            i += 1
            continue


        # ----------------------------------------------------
        # Long breakout
        # ----------------------------------------------------

        is_long_breakout = (
            is_long_regime and
            (
                c1h['Close'] >
                c1h['Donchian_High_24']
            ) and
            (
                body_size >=
                BREAKOUT_BODY_ATR *
                c1h['ATR']
            ) and
            (
                close_location >=
                LONG_CLOSE_LOCATION
            ) and
            (
                c1h['Volume'] >=
                VOLUME_MULTIPLIER *
                c1h['Vol_MA']
            )
        )


        # ----------------------------------------------------
        # Short breakout
        # ----------------------------------------------------

        is_short_breakout = (
            is_short_regime and
            (
                c1h['Close'] <
                c1h['Donchian_Low_24']
            ) and
            (
                body_size >=
                BREAKOUT_BODY_ATR *
                c1h['ATR']
            ) and
            (
                close_location <=
                SHORT_CLOSE_LOCATION
            ) and
            (
                c1h['Volume'] >=
                VOLUME_MULTIPLIER *
                c1h['Vol_MA']
            )
        )


        if (
            not is_long_breakout and
            not is_short_breakout
        ):

            i += 1
            continue


        # ====================================================
        # ENTRY
        # ====================================================

        entry_candle_idx = i + 1


        if (
            entry_candle_idx >=
            len(df1h)
        ):

            break


        entry_candle = (
            df1h.iloc[entry_candle_idx]
        )


        # ----------------------------------------------------
        # IS / OOS بر اساس کندل ورود
        # ----------------------------------------------------

        if entry_candle_idx < split_idx:

            data_sample = "IS"

            sample_end_index = split_idx

        else:

            data_sample = "OOS"

            sample_end_index = len(df1h)


        # ----------------------------------------------------
        # ATR فقط از SIGNAL CANDLE
        # ----------------------------------------------------

        signal_atr = c1h['ATR']


        if (
            pd.isna(signal_atr) or
            signal_atr <= 0
        ):

            i += 1
            continue


        risk = (
            STOP_ATR_MULTIPLIER *
            signal_atr
        )


        # ====================================================
        # LONG TRADE
        # ====================================================

        if is_long_breakout:

            entry_price = (
                entry_candle['Open'] *
                (1 + SLIPPAGE_RATE)
            )


            sl = (
                entry_price -
                risk
            )


            tp = (
                entry_price +
                TARGET_RR * risk
            )


            outcome = 'OPEN'

            exit_idx = entry_candle_idx

            max_favorable = 0.0
            max_adverse = 0.0

            executed_exit_price = 0.0


            max_exit_index = min(
                entry_candle_idx +
                MAX_HOLD_BARS,
                len(df1h)
            )


            # ----------------------------------------------
            # مهم:
            # اجازه عبور از مرز IS/OOS داده نمی‌شود
            # ----------------------------------------------

            max_allowed_index = min(
                max_exit_index,
                sample_end_index
            )


            for j in range(
                entry_candle_idx,
                max_allowed_index
            ):

                f_c = df1h.iloc[j]

                exit_idx = j


                # MFE / MAE
                cur_mfe = (
                    f_c['High'] -
                    entry_price
                )

                cur_mae = (
                    entry_price -
                    f_c['Low']
                )


                if cur_mfe > max_favorable:

                    max_favorable = cur_mfe


                if cur_mae > max_adverse:

                    max_adverse = cur_mae


                hit_tp = (
                    f_c['High'] >= tp
                )

                hit_sl = (
                    f_c['Low'] <= sl
                )


                # محافظه‌کارانه:
                # اگر هر دو در یک کندل لمس شوند،
                # SL اول فرض می‌شود.

                if hit_sl:

                    outcome = 'LOSS'

                    executed_exit_price = (
                        sl *
                        (1 - SLIPPAGE_RATE)
                    )

                    break


                elif hit_tp:

                    outcome = 'WIN'

                    executed_exit_price = (
                        tp *
                        (1 - SLIPPAGE_RATE)
                    )

                    break


            # ----------------------------------------------
            # ثبت معامله فقط اگر واقعاً بسته شده باشد
            # ----------------------------------------------

            if outcome in ['WIN', 'LOSS']:

                gross_pnl = (
                    executed_exit_price -
                    entry_price
                )


                entry_fee = (
                    entry_price *
                    COMMISSION_PER_SIDE
                )

                exit_fee = (
                    executed_exit_price *
                    COMMISSION_PER_SIDE
                )


                net_pnl = (
                    gross_pnl -
                    entry_fee -
                    exit_fee
                )


                net_R = (
                    net_pnl /
                    risk
                )


                all_trades.append({

                    'Symbol': symbol,

                    'Side': 'LONG',

                    'Sample': data_sample,

                    'Outcome': outcome,

                    'NetR': net_R,

                    'Date': entry_candle['Date'],

                    'MFE': (
                        max_favorable /
                        risk
                    ),

                    'MAE': (
                        max_adverse /
                        risk
                    )
                })


                # cooldown بعد از خروج
                locked_until_index = (
                    exit_idx +
                    COOLDOWN_BARS +
                    1
                )


                i = locked_until_index

                continue


            # اگر معامله تا مرز نمونه یا پایان داده
            # بسته نشد، آن را ثبت نمی‌کنیم.
            i = max(
                i + 1,
                exit_idx + 1
            )

            continue


        # ====================================================
        # SHORT TRADE
        # ====================================================

        elif is_short_breakout:

            entry_price = (
                entry_candle['Open'] *
                (1 - SLIPPAGE_RATE)
            )


            sl = (
                entry_price +
                risk
            )


            tp = (
                entry_price -
                TARGET_RR * risk
            )


            outcome = 'OPEN'

            exit_idx = entry_candle_idx

            max_favorable = 0.0
            max_adverse = 0.0

            executed_exit_price = 0.0


            max_exit_index = min(
                entry_candle_idx +
                MAX_HOLD_BARS,
                len(df1h)
            )


            max_allowed_index = min(
                max_exit_index,
                sample_end_index
            )


            for j in range(
                entry_candle_idx,
                max_allowed_index
            ):

                f_c = df1h.iloc[j]

                exit_idx = j


                # MFE / MAE
                cur_mfe = (
                    entry_price -
                    f_c['Low']
                )

                cur_mae = (
                    f_c['High'] -
                    entry_price
                )


                if cur_mfe > max_favorable:

                    max_favorable = cur_mfe


                if cur_mae > max_adverse:

                    max_adverse = cur_mae


                hit_tp = (
                    f_c['Low'] <= tp
                )

                hit_sl = (
                    f_c['High'] >= sl
                )


                # محافظه‌کارانه:
                # اگر هر دو در یک کندل لمس شوند،
                # SL اول فرض می‌شود.

                if hit_sl:

                    outcome = 'LOSS'

                    executed_exit_price = (
                        sl *
                        (1 + SLIPPAGE_RATE)
                    )

                    break


                elif hit_tp:

                    outcome = 'WIN'

                    executed_exit_price = (
                        tp *
                        (1 + SLIPPAGE_RATE)
                    )

                    break


            # ----------------------------------------------
            # ثبت معامله
            # ----------------------------------------------

            if outcome in ['WIN', 'LOSS']:

                gross_pnl = (
                    entry_price -
                    executed_exit_price
                )


                entry_fee = (
                    entry_price *
                    COMMISSION_PER_SIDE
                )

                exit_fee = (
                    executed_exit_price *
                    COMMISSION_PER_SIDE
                )


                net_pnl = (
                    gross_pnl -
                    entry_fee -
                    exit_fee
                )


                net_R = (
                    net_pnl /
                    risk
                )


                all_trades.append({

                    'Symbol': symbol,

                    'Side': 'SHORT',

                    'Sample': data_sample,

                    'Outcome': outcome,

                    'NetR': net_R,

                    'Date': entry_candle['Date'],

                    'MFE': (
                        max_favorable /
                        risk
                    ),

                    'MAE': (
                        max_adverse /
                        risk
                    )
                })


                locked_until_index = (
                    exit_idx +
                    COOLDOWN_BARS +
                    1
                )


                i = locked_until_index

                continue


            i = max(
                i + 1,
                exit_idx + 1
            )

            continue


        i += 1


# ============================================================
# گزارش نهایی
# ============================================================

print("\n============================================================")
print("📊 گزارش نهایی HUNTER-X V10 — Pristine Final")
print("============================================================")


if not all_trades:

    print("⚠️ هیچ معامله‌ای ثبت نشد.")

else:

    tdf = pd.DataFrame(
        all_trades
    )


    tdf.sort_values(
        'Date',
        inplace=True
    )


    tdf.to_csv(
        "detailed_trades_v10_pristine_final.csv",
        index=False
    )


    # ========================================================
    # IS / OOS REPORT
    # ========================================================

    for sample_type in ['IS', 'OOS']:

        sample_df = tdf[
            tdf['Sample'] ==
            sample_type
        ].copy()


        print(
            f"\n--- نتایج {sample_type} "
            f"(تعداد معاملات: "
            f"{len(sample_df)}) ---"
        )


        if sample_df.empty:

            print(
                "معامله‌ای ثبت نشد."
            )

            continue


        t_count = len(
            sample_df
        )


        w_count = len(
            sample_df[
                sample_df['Outcome'] ==
                'WIN'
            ]
        )


        l_count = len(
            sample_df[
                sample_df['Outcome'] ==
                'LOSS'
            ]
        )


        win_rate = (
            w_count /
            t_count
        ) * 100


        gross_profit = (
            sample_df[
                sample_df['NetR'] > 0
            ]['NetR']
            .sum()
        )


        gross_loss = abs(
            sample_df[
                sample_df['NetR'] < 0
            ]['NetR']
            .sum()
        )


        if gross_loss > 0:

            profit_factor = (
                gross_profit /
                gross_loss
            )

        else:

            profit_factor = np.nan


        expectancy = (
            sample_df['NetR']
            .mean()
        )


        total_net_r = (
            sample_df['NetR']
            .sum()
        )


        # ----------------------------------------------------
        # Drawdown
        # ----------------------------------------------------

        sample_df.sort_values(
            'Date',
            inplace=True
        )


        sample_df['CumulativeR'] = (
            sample_df['NetR']
            .cumsum()
        )


        sample_df['Peak'] = (
            sample_df['CumulativeR']
            .cummax()
        )


        sample_df['Drawdown'] = (
            sample_df['Peak'] -
            sample_df['CumulativeR']
        )


        max_dd = (
            sample_df['Drawdown']
            .max()
        )


        # ----------------------------------------------------
        # Monthly frequency
        # ----------------------------------------------------

        s_min_date = (
            sample_df['Date']
            .min()
        )

        s_max_date = (
            sample_df['Date']
            .max()
        )


        sample_months = max(
            1,
            (
                s_max_date -
                s_min_date
            ).days / 30.44
        )


        monthly_trades = (
            t_count /
            sample_months
        )


        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        print(
            f"🎯 معاملات: {t_count} | "
            f"WIN: {w_count} | "
            f"LOSS: {l_count}"
        )

        print(
            f"📅 معاملات ماهانه: "
            f"{monthly_trades:.1f}"
        )

        print(
            f"🎯 Win Rate: "
            f"{win_rate:.2f}%"
        )

        print(
            f"💰 Net R: "
            f"{total_net_r:.2f}R"
        )

        print(
            f"⚖️ Profit Factor: "
            f"{profit_factor:.2f}"
        )

        print(
            f"📐 Expectancy: "
            f"{expectancy:.3f}R"
        )

        print(
            f"📉 Max Drawdown: "
            f"{max_dd:.2f}R"
        )

        print(
            f"📈 Average MFE: "
            f"{sample_df['MFE'].mean():.2f}R"
        )

        print(
            f"📉 Average MAE: "
            f"{sample_df['MAE'].mean():.2f}R"
        )


    # ========================================================
    # LONG / SHORT
    # ========================================================

    print(
        "\n--- عملکرد Long / Short ---"
    )


    side_table = (
        tdf
        .groupby('Side')['Outcome']
        .value_counts()
        .unstack(
            fill_value=0
        )
    )


    print(
        side_table
    )


    # ========================================================
    # SYMBOL REPORT
    # ========================================================

    print(
        "\n--- عملکرد تفکیکی نمادها ---"
    )


    symbol_grouped = (
        tdf
        .groupby('Symbol')
        .agg(
            Trades=(
                'Outcome',
                'count'
            ),

            Wins=(
                'Outcome',
                lambda x:
                (x == 'WIN').sum()
            ),

            Losses=(
                'Outcome',
                lambda x:
                (x == 'LOSS').sum()
            ),

            NetR=(
                'NetR',
                'sum'
            )
        )
    )


    symbol_grouped['WinRate'] = (
        symbol_grouped['Wins'] /
        symbol_grouped['Trades'] *
        100
    ).round(2)


    symbol_grouped = (
        symbol_grouped
        .sort_values(
            'NetR',
            ascending=False
        )
    )


    print(
        symbol_grouped
    )


# ============================================================
# پایان
# ============================================================

print(
    "\n✨ بک‌تست HUNTER-X V10 "
    "Pristine Final به پایان رسید."
)
