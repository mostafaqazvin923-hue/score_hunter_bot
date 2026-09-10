import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

# ============================================================
# نصب ccxt
# ============================================================

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "ccxt"
    ])
    import ccxt

import pandas as pd
import numpy as np


# ============================================================
# تنظیمات
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


# ============================================================
# V10 Strategy — ثابت
# ============================================================

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

COOLDOWN_BARS = 6

MAX_HOLD_BARS = 80


# ============================================================
# مهم:
# فقط TP را آزمایش می‌کنیم
# ============================================================

TP_TESTS = [
    1.00,
    1.25,
    1.50,
    1.75,
    2.00
]


# ============================================================
# هزینه‌ها
# ============================================================

# 0.08% round trip
COMMISSION_RATE = 0.0008

COMMISSION_PER_SIDE = (
    COMMISSION_RATE / 2
)


# 0.04% per side
SLIPPAGE_RATE = 0.04 / 100


# ============================================================
# دریافت داده
# ============================================================

start_date = (
    datetime.now(timezone.utc)
    - timedelta(days=LOOKBACK_DAYS)
)

since_timestamp = int(
    start_date.timestamp() * 1000
)


print("============================================================")
print("📥 دانلود داده‌ها — HUNTER-X V10 Diagnostic")
print("============================================================")


data_1h = {}


for symbol, lbank_symbol in SYMBOLS.items():

    print(
        f"\n🔹 در حال دریافت دیتای 1H برای {symbol}..."
    )

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


            all_ohlcv.extend(
                ohlcv
            )


            last_timestamp = ohlcv[-1][0]

            next_since = (
                last_timestamp + 1
            )


            if next_since <= current_since:
                break


            current_since = next_since


            if len(ohlcv) < 1000:
                break


        except Exception as e:

            print(
                f"  ❌ خطا در {symbol}: {e}"
            )

            break


    if not all_ohlcv:

        print(
            f"  ❌ دیتایی برای {symbol} دریافت نشد."
        )

        continue


    # ========================================================
    # DataFrame
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


    df1h.dropna(
        inplace=True
    )


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
    # حذف کندل ناقص فعلی
    # ========================================================

    current_hour_utc = (
        pd.Timestamp.now(tz='UTC')
        .floor('h')
        .tz_localize(None)
    )


    df1h = df1h[
        df1h['Date'] <
        current_hour_utc
    ].copy()


    df1h.reset_index(
        drop=True,
        inplace=True
    )


    data_1h[symbol] = df1h


    print(
        f"  ✔️ {symbol}: "
        f"{len(df1h)} کندل"
    )


# ============================================================
# اندیکاتورها
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
        df['High'] -
        df['Low']
    )


    high_close = np.abs(
        df['High'] -
        df['Close'].shift(1)
    )


    low_close = np.abs(
        df['Low'] -
        df['Close'].shift(1)
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
    # ATR
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
    # ADX
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
        tr_smooth.replace(
            0,
            np.nan
        )
    )


    minus_di = (
        100 *
        minus_dm_smooth /
        tr_smooth.replace(
            0,
            np.nan
        )
    )


    dx = (
        100 *
        np.abs(
            plus_di -
            minus_di
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
    # current candle excluded
    # --------------------------------------------------------

    df['Donchian_High_24'] = (
        df['High']
        .shift(1)
        .rolling(
            DONCHIAN_LENGTH
        )
        .max()
    )


    df['Donchian_Low_24'] = (
        df['Low']
        .shift(1)
        .rolling(
            DONCHIAN_LENGTH
        )
        .min()
    )


    return df


# ============================================================
# آماده‌سازی داده‌ها
# ============================================================

prepared_data = {}


for symbol, original_df1h in data_1h.items():

    if len(original_df1h) < 500:

        print(
            f"⚠️ {symbol}: داده کافی نیست."
        )

        continue


    df1h = calculate_indicators(
        original_df1h
    )


    # ========================================================
    # 4H
    # ========================================================

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


    prepared_data[symbol] = (
        df1h,
        df4h
    )


# ============================================================
# موتور بک‌تست
# ============================================================

def run_backtest(target_rr):

    all_trades = []


    for symbol, (
        df1h,
        df4h
    ) in prepared_data.items():


        # ----------------------------------------------------
        # 80% IS / 20% OOS
        # ----------------------------------------------------

        split_idx = int(
            len(df1h) * 0.80
        )


        i = 200


        last_valid_signal_index = (
            len(df1h)
            - MAX_HOLD_BARS
            - 2
        )


        # ----------------------------------------------------
        # نکته مهم:
        # cooldown به صورت sample-specific است.
        #
        # یعنی cooldown از IS وارد OOS نمی‌شود.
        # ----------------------------------------------------

        locked_until_index = 0


        current_sample = "IS"


        while i <= last_valid_signal_index:


            # =================================================
            # تشخیص مرز IS/OOS
            # =================================================

            if i < split_idx:

                sample_type = "IS"

            else:

                sample_type = "OOS"


            # اگر وارد OOS شدیم،
            # cooldown قبلی را پاک می‌کنیم.

            if (
                sample_type == "OOS" and
                current_sample == "IS"
            ):

                locked_until_index = 0


            current_sample = sample_type


            # =================================================
            # Cooldown
            # =================================================

            if i < locked_until_index:

                i += 1
                continue


            # =================================================
            # Signal candle
            # =================================================

            c1h = df1h.iloc[i]

            current_time = c1h['Date']


            # =================================================
            # آخرین 4H کاملاً بسته‌شده
            # =================================================

            closed_4h_time = (
                current_time -
                timedelta(hours=4)
            )


            available_4h = df4h[
                df4h['Date'] <=
                closed_4h_time
            ]


            if len(available_4h) < 2:

                i += 1
                continue


            r4h = available_4h.iloc[-1]

            prev_r4h = available_4h.iloc[-2]


            # =================================================
            # Validity
            # =================================================

            if (
                pd.isna(r4h['ADX']) or
                pd.isna(r4h['EMA_200']) or
                pd.isna(r4h['EMA_50'])
            ):

                i += 1
                continue


            # =================================================
            # EMA200 slope
            # =================================================

            ema200_slope_up = (
                r4h['EMA_200'] >=
                prev_r4h['EMA_200']
            )


            ema200_slope_down = (
                r4h['EMA_200'] <=
                prev_r4h['EMA_200']
            )


            # =================================================
            # 4H regime
            # =================================================

            is_long_regime = (
                (r4h['Close'] >
                 r4h['EMA_200']) and

                (r4h['EMA_50'] >
                 r4h['EMA_200']) and

                (r4h['ADX'] >=
                 ADX_THRESHOLD) and

                ema200_slope_up
            )


            is_short_regime = (
                (r4h['Close'] <
                 r4h['EMA_200']) and

                (r4h['EMA_50'] <
                 r4h['EMA_200']) and

                (r4h['ADX'] >=
                 ADX_THRESHOLD) and

                ema200_slope_down
            )


            if (
                not is_long_regime and
                not is_short_regime
            ):

                i += 1
                continue


            # =================================================
            # 1H volatility
            # =================================================

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
                volatility_ratio >=
                ATR_MIN_RATIO and

                volatility_ratio <=
                ATR_MAX_RATIO
            )


            if not pass_volatility:

                i += 1
                continue


            # =================================================
            # Breakout candle
            # =================================================

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


            # =================================================
            # Validity
            # =================================================

            if (
                pd.isna(
                    c1h[
                        'Donchian_High_24'
                    ]
                ) or

                pd.isna(
                    c1h[
                        'Donchian_Low_24'
                    ]
                ) or

                pd.isna(
                    c1h['Vol_MA']
                )
            ):

                i += 1
                continue


            # =================================================
            # LONG
            # =================================================

            is_long_breakout = (
                is_long_regime and

                (
                    c1h['Close'] >
                    c1h[
                        'Donchian_High_24'
                    ]
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


            # =================================================
            # SHORT
            # =================================================

            is_short_breakout = (
                is_short_regime and

                (
                    c1h['Close'] <
                    c1h[
                        'Donchian_Low_24'
                    ]
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


            # =================================================
            # Entry
            # =================================================

            entry_candle_idx = i + 1


            if (
                entry_candle_idx >=
                len(df1h)
            ):

                break


            entry_candle = (
                df1h.iloc[
                    entry_candle_idx
                ]
            )


            # =================================================
            # Sample based on ENTRY
            # =================================================

            if (
                entry_candle_idx <
                split_idx
            ):

                data_sample = "IS"

                sample_end_index = (
                    split_idx
                )

            else:

                data_sample = "OOS"

                sample_end_index = (
                    len(df1h)
                )


            # =================================================
            # Signal ATR
            # =================================================

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


            # =================================================
            # LONG TRADE
            # =================================================

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
                    target_rr * risk
                )


                outcome = "OPEN"

                exit_idx = (
                    entry_candle_idx
                )

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


                    # MFE
                    cur_mfe = (
                        f_c['High'] -
                        entry_price
                    )


                    # MAE
                    cur_mae = (
                        entry_price -
                        f_c['Low']
                    )


                    max_favorable = max(
                        max_favorable,
                        cur_mfe
                    )


                    max_adverse = max(
                        max_adverse,
                        cur_mae
                    )


                    hit_tp = (
                        f_c['High'] >= tp
                    )


                    hit_sl = (
                        f_c['Low'] <= sl
                    )


                    # Conservative:
                    # SL first if both occur
                    # in the same candle.

                    if hit_sl:

                        outcome = "LOSS"


                        executed_exit_price = (
                            sl *
                            (1 - SLIPPAGE_RATE)
                        )


                        break


                    elif hit_tp:

                        outcome = "WIN"


                        executed_exit_price = (
                            tp *
                            (1 - SLIPPAGE_RATE)
                        )


                        break


                # =================================================
                # ثبت معامله
                # =================================================

                if outcome in [
                    "WIN",
                    "LOSS"
                ]:


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

                        'Date':
                            entry_candle['Date'],

                        'MFE':
                            max_favorable / risk,

                        'MAE':
                            max_adverse / risk,

                        'TP_RR':
                            target_rr

                    })


                    locked_until_index = (
                        exit_idx +
                        COOLDOWN_BARS +
                        1
                    )


                    i = locked_until_index

                    continue


                # معامله باز تا مرز نمونه
                i = max(
                    i + 1,
                    exit_idx + 1
                )

                continue


            # =================================================
            # SHORT TRADE
            # =================================================

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
                    target_rr * risk
                )


                outcome = "OPEN"

                exit_idx = (
                    entry_candle_idx
                )

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


                    # MFE
                    cur_mfe = (
                        entry_price -
                        f_c['Low']
                    )


                    # MAE
                    cur_mae = (
                        f_c['High'] -
                        entry_price
                    )


                    max_favorable = max(
                        max_favorable,
                        cur_mfe
                    )


                    max_adverse = max(
                        max_adverse,
                        cur_mae
                    )


                    hit_tp = (
                        f_c['Low'] <= tp
                    )


                    hit_sl = (
                        f_c['High'] >= sl
                    )


                    # Conservative:
                    # SL first

                    if hit_sl:

                        outcome = "LOSS"


                        executed_exit_price = (
                            sl *
                            (1 + SLIPPAGE_RATE)
                        )


                        break


                    elif hit_tp:

                        outcome = "WIN"


                        executed_exit_price = (
                            tp *
                            (1 + SLIPPAGE_RATE)
                        )


                        break


                # =================================================
                # ثبت معامله
                # =================================================

                if outcome in [
                    "WIN",
                    "LOSS"
                ]:


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

                        'Date':
                            entry_candle['Date'],

                        'MFE':
                            max_favorable / risk,

                        'MAE':
                            max_adverse / risk,

                        'TP_RR':
                            target_rr

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


    return pd.DataFrame(
        all_trades
    )


# ============================================================
# گزارش یک تست
# ============================================================

def calculate_report(
    trades_df,
    sample_type
):

    sample_df = trades_df[
        trades_df['Sample'] ==
        sample_type
    ].copy()


    if sample_df.empty:

        return {
            'Trades': 0,
            'Wins': 0,
            'Losses': 0,
            'WR': np.nan,
            'NetR': 0.0,
            'PF': np.nan,
            'Expectancy': np.nan,
            'MaxDD': np.nan,
            'MFE': np.nan,
            'MAE': np.nan
        }


    trades = len(
        sample_df
    )


    wins = (
        sample_df['Outcome']
        .eq('WIN')
        .sum()
    )


    losses = (
        sample_df['Outcome']
        .eq('LOSS')
        .sum()
    )


    wr = (
        wins /
        trades *
        100
    )


    gross_profit = (
        sample_df.loc[
            sample_df['NetR'] > 0,
            'NetR'
        ].sum()
    )


    gross_loss = abs(
        sample_df.loc[
            sample_df['NetR'] < 0,
            'NetR'
        ].sum()
    )


    pf = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else np.nan
    )


    expectancy = (
        sample_df['NetR']
        .mean()
    )


    net_r = (
        sample_df['NetR']
        .sum()
    )


    sample_df.sort_values(
        'Date',
        inplace=True
    )


    cumulative = (
        sample_df['NetR']
        .cumsum()
    )


    peak = (
        cumulative
        .cummax()
    )


    drawdown = (
        peak -
        cumulative
    )


    max_dd = (
        drawdown.max()
    )


    return {

        'Trades':
            trades,

        'Wins':
            wins,

        'Losses':
            losses,

        'WR':
            wr,

        'NetR':
            net_r,

        'PF':
            pf,

        'Expectancy':
            expectancy,

        'MaxDD':
            max_dd,

        'MFE':
            sample_df['MFE'].mean(),

        'MAE':
            sample_df['MAE'].mean()
    }


# ============================================================
# اجرای تمام TPها
# ============================================================

print("\n============================================================")
print("🧪 اجرای تست تشخیصی TP")
print("============================================================")

results = []

all_detailed = []


for target_rr in TP_TESTS:

    print(
        f"\n▶️ در حال تست TP = {target_rr:.2f}R ..."
    )


    trades_df = run_backtest(
        target_rr
    )


    if trades_df.empty:

        print(
            "  ⚠️ هیچ معامله‌ای ثبت نشد."
        )

        continue


    trades_df.to_csv(
        f"detailed_v10_tp_{target_rr:.2f}R.csv",
        index=False
    )


    all_detailed.append(
        trades_df
    )


    is_report = calculate_report(
        trades_df,
        "IS"
    )


    oos_report = calculate_report(
        trades_df,
        "OOS"
    )


    results.append({

        'TP': target_rr,

        'IS_Trades':
            is_report['Trades'],

        'IS_WR':
            is_report['WR'],

        'IS_NetR':
            is_report['NetR'],

        'IS_PF':
            is_report['PF'],

        'IS_Expectancy':
            is_report['Expectancy'],

        'IS_MaxDD':
            is_report['MaxDD'],

        'IS_MFE':
            is_report['MFE'],

        'IS_MAE':
            is_report['MAE'],


        'OOS_Trades':
            oos_report['Trades'],

        'OOS_WR':
            oos_report['WR'],

        'OOS_NetR':
            oos_report['NetR'],

        'OOS_PF':
            oos_report['PF'],

        'OOS_Expectancy':
            oos_report['Expectancy'],

        'OOS_MaxDD':
            oos_report['MaxDD'],

        'OOS_MFE':
            oos_report['MFE'],

        'OOS_MAE':
            oos_report['MAE']
    })


    # ========================================================
    # نمایش
    # ========================================================

    print(
        "\n  ───────────── IS ─────────────"
    )


    print(
        f"  Trades:      {is_report['Trades']}"
    )

    print(
        f"  WR:          {is_report['WR']:.2f}%"
    )

    print(
        f"  Net R:       {is_report['NetR']:.2f}R"
    )

    print(
        f"  PF:          {is_report['PF']:.2f}"
    )

    print(
        f"  Expectancy:  "
        f"{is_report['Expectancy']:.3f}R"
    )

    print(
        f"  Max DD:      "
        f"{is_report['MaxDD']:.2f}R"
    )

    print(
        f"  MFE:         "
        f"{is_report['MFE']:.2f}R"
    )

    print(
        f"  MAE:         "
        f"{is_report['MAE']:.2f}R"
    )


    print(
        "\n  ───────────── OOS ────────────"
    )


    print(
        f"  Trades:      {oos_report['Trades']}"
    )

    print(
        f"  WR:          {oos_report['WR']:.2f}%"
    )

    print(
        f"  Net R:       {oos_report['NetR']:.2f}R"
    )

    print(
        f"  PF:          {oos_report['PF']:.2f}"
    )

    print(
        f"  Expectancy:  "
        f"{oos_report['Expectancy']:.3f}R"
    )

    print(
        f"  Max DD:      "
        f"{oos_report['MaxDD']:.2f}R"
    )

    print(
        f"  MFE:         "
        f"{oos_report['MFE']:.2f}R"
    )

    print(
        f"  MAE:         "
        f"{oos_report['MAE']:.2f}R"
    )


# ============================================================
# جدول نهایی
# ============================================================

print("\n============================================================")
print("📊 مقایسه نهایی TPها")
print("============================================================")


if results:

    results_df = pd.DataFrame(
        results
    )


    pd.set_option(
        'display.max_columns',
        None
    )

    pd.set_option(
        'display.width',
        250
    )

    pd.set_option(
        'display.float_format',
        lambda x: f"{x:.3f}"
    )


    print(
        results_df.to_string(
            index=False
        )
    )


    results_df.to_csv(
        "V10_TP_DIAGNOSTIC_RESULTS.csv",
        index=False
    )


    # ========================================================
    # بهترین OOS بر اساس PF
    # ========================================================

    valid_pf = results_df[
        results_df['OOS_PF'].notna()
    ].copy()


    if not valid_pf.empty:

        best_pf = (
            valid_pf
            .sort_values(
                'OOS_PF',
                ascending=False
            )
            .iloc[0]
        )


        print(
            "\n🏆 بهترین TP از نظر OOS Profit Factor:"
        )


        print(
            f"TP = "
            f"{best_pf['TP']:.2f}R | "
            f"OOS PF = "
            f"{best_pf['OOS_PF']:.2f} | "
            f"OOS WR = "
            f"{best_pf['OOS_WR']:.2f}% | "
            f"OOS NetR = "
            f"{best_pf['OOS_NetR']:.2f}R | "
            f"OOS Exp = "
            f"{best_pf['OOS_Expectancy']:.3f}R"
        )


    # ========================================================
    # بهترین OOS بر اساس Expectancy
    # ========================================================

    valid_exp = results_df[
        results_df['OOS_Expectancy'].notna()
    ].copy()


    if not valid_exp.empty:

        best_exp = (
            valid_exp
            .sort_values(
                'OOS_Expectancy',
                ascending=False
            )
            .iloc[0]
        )


        print(
            "\n🏆 بهترین TP از نظر OOS Expectancy:"
        )


        print(
            f"TP = "
            f"{best_exp['TP']:.2f}R | "
            f"OOS PF = "
            f"{best_exp['OOS_PF']:.2f} | "
            f"OOS WR = "
            f"{best_exp['OOS_WR']:.2f}% | "
            f"OOS NetR = "
            f"{best_exp['OOS_NetR']:.2f}R | "
            f"OOS Exp = "
            f"{best_exp['OOS_Expectancy']:.3f}R"
        )


    # ========================================================
    # TPهای مثبت OOS
    # ========================================================

    positive_oos = results_df[
        (results_df['OOS_PF'] > 1.0) &
        (results_df['OOS_Expectancy'] > 0)
    ]


    print(
        "\n🔎 TPهایی که OOS واقعاً مثبت هستند:"
    )


    if positive_oos.empty:

        print(
            "❌ هیچ TPای همزمان "
            "PF > 1 و Expectancy > 0 ندارد."
        )

    else:

        print(
            positive_oos[
                [
                    'TP',
                    'OOS_Trades',
                    'OOS_WR',
                    'OOS_NetR',
                    'OOS_PF',
                    'OOS_Expectancy',
                    'OOS_MaxDD'
                ]
            ].to_string(
                index=False
            )
        )


    # ========================================================
    # تمام معاملات
    # ========================================================

    if all_detailed:

        combined = pd.concat(
            all_detailed,
            ignore_index=True
        )


        combined.to_csv(
            "V10_TP_DIAGNOSTIC_ALL_TRADES.csv",
            index=False
        )


else:

    print(
        "⚠️ هیچ نتیجه‌ای تولید نشد."
    )


# ============================================================
# پایان
# ============================================================

print("\n============================================================")
print("✅ V10 TP Diagnostic به پایان رسید.")
print("============================================================")
