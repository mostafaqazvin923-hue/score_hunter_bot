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

exchange = ccxt.lbank({'enableRateLimit': True})

# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT',
    'SOL': 'SOL/USDT',
    'XRP': 'XRP/USDT',
    'ADA': 'ADA/USDT',
    'AVAX': 'AVAX/USDT',
    'DOGE': 'DOGE/USDT',
    'DOT': 'DOT/USDT',
    'LTC': 'LTC/USDT',
    'UNI': 'UNI/USDT',
    'RENDER': 'RENDER/USDT',
    'LINK': 'LINK/USDT',
    'ATOM': 'ATOM/USDT',
}

# ============================================================
# SETTINGS
# ============================================================

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

MAX_CONCURRENT_POSITIONS = 3
MAX_HOLD_CANDLES = 24

RISK_PER_TRADE_R = 1.0
TARGET_R = 2.0

# ============================================================
# DOWNLOAD DATA
# ============================================================

print('============================================================')
print('📥 دریافت داده‌ها از LBank')
print('============================================================')

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():

    print(f'📥 {symbol} ...')

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
            print(f'⚠️ خطا در دریافت {symbol}: {e}')
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

        data_1h[symbol] = df1h

        print(
            f'   ✅ {symbol}: {len(df1h)} candles'
        )


# ============================================================
# ICHIMOKU
# ============================================================

def calculate_ichimoku(df):

    df = df.copy()

    period9_high = (
        df['High']
        .rolling(window=9)
        .max()
    )

    period9_low = (
        df['Low']
        .rolling(window=9)
        .min()
    )

    df['Tenkan'] = (
        period9_high + period9_low
    ) / 2

    period26_high = (
        df['High']
        .rolling(window=26)
        .max()
    )

    period26_low = (
        df['Low']
        .rolling(window=26)
        .min()
    )

    df['Kijun'] = (
        period26_high + period26_low
    ) / 2

    df['Senkou_A'] = (
        (df['Tenkan'] + df['Kijun']) / 2
    ).shift(26)

    period52_high = (
        df['High']
        .rolling(window=52)
        .max()
    )

    period52_low = (
        df['Low']
        .rolling(window=52)
        .min()
    )

    df['Senkou_B'] = (
        (period52_high + period52_low) / 2
    ).shift(26)

    tr1 = df['High'] - df['Low']

    tr2 = np.abs(
        df['High'] - df['Close'].shift(1)
    )

    tr3 = np.abs(
        df['Low'] - df['Close'].shift(1)
    )

    df['ATR'] = (
        pd.concat(
            [tr1, tr2, tr3],
            axis=1
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )

    return df


# ============================================================
# PROCESS DATA
# ============================================================

processed_data = {}

for symbol, df1h in data_1h.items():

    if len(df1h) < 100:
        continue

    df1h = calculate_ichimoku(df1h)

    df4h = (
        df1h
        .set_index('Date')
        .resample('4h')
        .agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        })
        .dropna()
        .reset_index()
    )

    df4h = calculate_ichimoku(df4h)

    df4h['Cloud_Top'] = df4h[
        ['Senkou_A', 'Senkou_B']
    ].max(axis=1)

    df4h['Cloud_Bottom'] = df4h[
        ['Senkou_A', 'Senkou_B']
    ].min(axis=1)

    df4h['Trend_Long'] = (
        df4h['Close'] > df4h['Cloud_Top']
    )

    df4h['Trend_Short'] = (
        df4h['Close'] < df4h['Cloud_Bottom']
    )

    # استفاده از کندل 4H بسته‌شده قبلی
    for col in [
        'Trend_Long',
        'Trend_Short',
        'Cloud_Top',
        'Cloud_Bottom'
    ]:
        df4h[col] = df4h[col].shift(1)

    df1h['Date_4H'] = (
        df1h['Date']
        .dt.floor('4h')
    )

    processed_data[symbol] = {
        '1h': df1h,
        '4h': df4h.set_index('Date')
    }


# ============================================================
# BACKTEST
# ============================================================

print()
print('============================================================')
print('⚙️ شروع بک‌تست')
print('============================================================')

all_timestamps = set()

for dat in processed_data.values():

    all_timestamps.update(
        dat['1h']['Date'].tolist()
    )

sorted_timestamps = sorted(
    list(all_timestamps)
)

active_positions = {}
all_trades = []

dfs_1h = {
    sym: dat['1h'].set_index('Date')
    for sym, dat in processed_data.items()
}

dfs_4h = {
    sym: dat['4h']
    for sym, dat in processed_data.items()
}


# ============================================================
# MAIN LOOP
# ============================================================

for ts in sorted_timestamps:

    # --------------------------------------------------------
    # BTC MARKET FILTER
    # --------------------------------------------------------

    btc_allows_long = True
    btc_allows_short = True

    btc_trend = 'UNKNOWN'

    if 'BTC' in processed_data:

        t4h_btc = (
            ts
            - timedelta(
                hours=ts.hour % 4,
                minutes=ts.minute,
                seconds=ts.second
            )
        )

        df4h_btc = processed_data['BTC']['4h']

        if t4h_btc in df4h_btc.index:

            btc_row = df4h_btc.loc[t4h_btc]

            btc_allows_long = bool(
                btc_row.get(
                    'Trend_Long',
                    True
                )
            )

            btc_allows_short = bool(
                btc_row.get(
                    'Trend_Short',
                    True
                )
            )

            if btc_allows_long:
                btc_trend = 'LONG'
            elif btc_allows_short:
                btc_trend = 'SHORT'
            else:
                btc_trend = 'NEUTRAL'


    # ========================================================
    # CLOSE EXISTING POSITIONS
    # ========================================================

    symbols_to_close = []

    for symbol, pos in list(
        active_positions.items()
    ):

        if ts not in dfs_1h[symbol].index:
            continue

        c1h = dfs_1h[symbol].loc[ts]

        entry_index = pos['entry_index']

        df1h_local = processed_data[
            symbol
        ]['1h']

        match_rows = df1h_local[
            df1h_local['Date'] == ts
        ]

        if match_rows.empty:
            continue

        curr_i = match_rows.index[0]

        candles_held = (
            curr_i - entry_index
        )

        pos['candles_held'] = candles_held

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if pos['side'] == 'LONG':

            hit_sl = (
                c1h['Low']
                <= pos['stop_loss']
            )

            hit_tp = (
                c1h['High']
                >= pos['take_profit']
            )

            timeout = (
                candles_held
                >= MAX_HOLD_CANDLES
            )

            reason = None
            outcome = None

            # هر دو در یک کندل
            if hit_sl and hit_tp:

                outcome = 'LOSS'
                reason = 'SL+TP_SAME_CANDLE'

                r_real = (
                    -1.0
                    - (FEE_RATE * 2)
                )

            elif hit_sl:

                outcome = 'LOSS'
                reason = 'SL'

                r_real = (
                    -1.0
                    - (FEE_RATE * 2)
                )

            elif hit_tp:

                outcome = 'WIN'
                reason = 'TP'

                r_real = (
                    2.0
                    - (FEE_RATE * 2)
                )

            elif timeout:

                outcome = 'TIMEOUT'
                reason = 'TIMEOUT'

                # نتیجه واقعی خروج در Close
                exit_price = c1h['Close']

                gross_r = (
                    exit_price
                    - pos['entry_price']
                ) / (
                    pos['entry_price']
                    - pos['stop_loss']
                )

                r_real = (
                    gross_r
                    - (FEE_RATE * 2)
                )

            else:

                continue

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            hit_sl = (
                c1h['High']
                >= pos['stop_loss']
            )

            hit_tp = (
                c1h['Low']
                <= pos['take_profit']
            )

            timeout = (
                candles_held
                >= MAX_HOLD_CANDLES
            )

            reason = None
            outcome = None

            if hit_sl and hit_tp:

                outcome = 'LOSS'
                reason = 'SL+TP_SAME_CANDLE'

                r_real = (
                    -1.0
                    - (FEE_RATE * 2)
                )

            elif hit_sl:

                outcome = 'LOSS'
                reason = 'SL'

                r_real = (
                    -1.0
                    - (FEE_RATE * 2)
                )

            elif hit_tp:

                outcome = 'WIN'
                reason = 'TP'

                r_real = (
                    2.0
                    - (FEE_RATE * 2)
                )

            elif timeout:

                outcome = 'TIMEOUT'
                reason = 'TIMEOUT'

                exit_price = c1h['Close']

                gross_r = (
                    pos['entry_price']
                    - exit_price
                ) / (
                    pos['stop_loss']
                    - pos['entry_price']
                )

                r_real = (
                    gross_r
                    - (FEE_RATE * 2)
                )

            else:

                continue


        # ----------------------------------------------------
        # SAVE TRADE
        # ----------------------------------------------------

        all_trades.append({

            'Timestamp': ts,

            'Symbol': symbol,

            'Side': pos['side'],

            'Entry_Time': pos['entry_time'],

            'Entry_Price': pos['entry_price'],

            'Stop_Loss': pos['stop_loss'],

            'Take_Profit': pos['take_profit'],

            'Exit_Price':
                c1h['Close']
                if reason == 'TIMEOUT'
                else (
                    pos['stop_loss']
                    if reason in [
                        'SL',
                        'SL+TP_SAME_CANDLE'
                    ]
                    else pos['take_profit']
                ),

            'Candles_Held':
                candles_held,

            'Hours_Held':
                candles_held,

            'Outcome':
                outcome,

            'Reason':
                reason,

            'BTC_Trend':
                btc_trend,

            'Return':
                r_real,
        })

        symbols_to_close.append(symbol)


    # --------------------------------------------------------
    # DELETE CLOSED POSITIONS
    # --------------------------------------------------------

    for sym in symbols_to_close:

        if sym in active_positions:

            del active_positions[sym]


    # ========================================================
    # OPEN NEW POSITIONS
    # ========================================================

    for symbol, dat in processed_data.items():

        if symbol in active_positions:
            continue

        if (
            len(active_positions)
            >= MAX_CONCURRENT_POSITIONS
        ):
            break

        df1h = dat['1h']

        if ts not in df1h['Date'].values:
            continue

        match_rows = df1h[
            df1h['Date'] == ts
        ]

        if match_rows.empty:
            continue

        i = match_rows.index[0]

        if i < 2:
            continue

        c1h = df1h.iloc[i]

        prev = df1h.iloc[i - 1]

        t4h_time = c1h['Date_4H']

        df4h_idx = dat['4h']

        if t4h_time not in df4h_idx.index:
            continue

        r4h = df4h_idx.loc[t4h_time]

        cloud_top_1h = max(
            prev['Senkou_A'],
            prev['Senkou_B']
        )

        cloud_bot_1h = min(
            prev['Senkou_A'],
            prev['Senkou_B']
        )

        # ====================================================
        # LONG
        # ====================================================

        if (
            btc_allows_long
            and r4h.get(
                'Trend_Long',
                False
            )
            and prev['Close']
            > cloud_top_1h
        ):

            tk_cross_long = (

                prev['Tenkan']
                > prev['Kijun']

                and

                df1h.iloc[i - 2]['Tenkan']
                <=
                df1h.iloc[i - 2]['Kijun']
            )

            pullback_long = (

                prev['Low']
                <= prev['Kijun']

                and

                prev['Close']
                > prev['Kijun']
            )

            if (
                tk_cross_long
                or pullback_long
            ):

                entry_price = (
                    c1h['Open']
                    * (1 + SLIPPAGE)
                )

                stop_loss = (
                    min(
                        prev['Low'],
                        cloud_bot_1h
                    )
                    - 0.2 * prev['ATR']
                )

                sl_dist_pct = (
                    entry_price
                    - stop_loss
                ) / entry_price

                if (
                    0.003
                    <= sl_dist_pct
                    <= 0.04
                ):

                    risk = (
                        entry_price
                        - stop_loss
                    )

                    take_profit = (
                        entry_price
                        + 2.0 * risk
                    )

                    active_positions[
                        symbol
                    ] = {

                        'side':
                            'LONG',

                        'entry_price':
                            entry_price,

                        'stop_loss':
                            stop_loss,

                        'take_profit':
                            take_profit,

                        'entry_index':
                            i,

                        'entry_time':
                            ts,
                    }

                    continue


        # ====================================================
        # SHORT
        # ====================================================

        elif (
            btc_allows_short
            and r4h.get(
                'Trend_Short',
                False
            )
            and prev['Close']
            < cloud_bot_1h
        ):

            tk_cross_short = (

                prev['Tenkan']
                < prev['Kijun']

                and

                df1h.iloc[i - 2]['Tenkan']
                >=
                df1h.iloc[i - 2]['Kijun']
            )

            pullback_short = (

                prev['High']
                >= prev['Kijun']

                and

                prev['Close']
                < prev['Kijun']
            )

            if (
                tk_cross_short
                or pullback_short
            ):

                entry_price = (
                    c1h['Open']
                    * (1 - SLIPPAGE)
                )

                stop_loss = (
                    max(
                        prev['High'],
                        cloud_top_1h
                    )
                    + 0.2 * prev['ATR']
                )

                sl_dist_pct = (
                    stop_loss
                    - entry_price
                ) / entry_price

                if (
                    0.003
                    <= sl_dist_pct
                    <= 0.04
                ):

                    risk = (
                        stop_loss
                        - entry_price
                    )

                    take_profit = (
                        entry_price
                        - 2.0 * risk
                    )

                    active_positions[
                        symbol
                    ] = {

                        'side':
                            'SHORT',

                        'entry_price':
                            entry_price,

                        'stop_loss':
                            stop_loss,

                        'take_profit':
                            take_profit,

                        'entry_index':
                            i,

                        'entry_time':
                            ts,
                    }

                    continue


# ============================================================
# FINAL ANALYSIS
# ============================================================

print()
print('============================================================')
print('📊 گزارش کامل HUNTER-X V8.1')
print('============================================================')


if not all_trades:

    print('⚠️ معامله‌ای ثبت نشد.')

else:

    trades_df = pd.DataFrame(
        all_trades
    )

    trades_df.sort_values(
        'Timestamp',
        inplace=True
    )

    trades_df.reset_index(
        drop=True,
        inplace=True
    )

    # ========================================================
    # BASIC STATS
    # ========================================================

    total = len(trades_df)

    wins = len(
        trades_df[
            trades_df['Outcome'] == 'WIN'
        ]
    )

    losses = len(
        trades_df[
            trades_df['Outcome'] == 'LOSS'
        ]
    )

    timeouts = len(
        trades_df[
            trades_df['Outcome'] == 'TIMEOUT'
        ]
    )

    closed_trades = wins + losses

    win_rate = (
        wins / closed_trades * 100
        if closed_trades > 0
        else 0
    )

    net_r = (
        trades_df['Return'].sum()
    )

    avg_r = (
        trades_df['Return'].mean()
    )

    print()
    print('📌 آمار اصلی')
    print('------------------------------------------------------------')

    print(
        f'🔸 کل معاملات: {total}'
    )

    print(
        f'🟢 WIN: {wins}'
    )

    print(
        f'🔴 LOSS: {losses}'
    )

    print(
        f'⏱️ TIMEOUT: {timeouts}'
    )

    print(
        f'🎯 Win Rate بدون Timeout: {win_rate:.2f}%'
    )

    print(
        f'💰 Net R: {net_r:.2f}R'
    )

    print(
        f'📊 Average R / Trade: {avg_r:.4f}R'
    )


    # ========================================================
    # REASON ANALYSIS
    # ========================================================

    print()
    print('📌 علت خروج معاملات')
    print('------------------------------------------------------------')

    reason_counts = (
        trades_df['Reason']
        .value_counts()
    )

    for reason, count in reason_counts.items():

        pct = (
            count / total * 100
        )

        print(
            f'   {reason:<25} '
            f'{count:>4} '
            f'({pct:.2f}%)'
        )


    # ========================================================
    # EQUITY CURVE + MAX DRAWDOWN
    # ========================================================

    trades_df['Equity_R'] = (
        trades_df['Return']
        .cumsum()
    )

    trades_df['Peak_R'] = (
        trades_df['Equity_R']
        .cummax()
    )

    trades_df['Drawdown_R'] = (
        trades_df['Equity_R']
        - trades_df['Peak_R']
    )

    max_drawdown = (
        trades_df['Drawdown_R'].min()
    )

    max_drawdown_idx = (
        trades_df['Drawdown_R'].idxmin()
    )

    peak_before_dd = (
        trades_df.loc[
            :max_drawdown_idx,
            'Peak_R'
        ].max()
    )

    dd_end_time = (
        trades_df.loc[
            max_drawdown_idx,
            'Timestamp'
        ]
    )

    print()
    print('📉 MAX DRAWDOWN')
    print('------------------------------------------------------------')

    print(
        f'🔻 Maximum Drawdown: '
        f'{max_drawdown:.2f}R'
    )

    print(
        f'🏔️ Peak Equity Before DD: '
        f'{peak_before_dd:.2f}R'
    )

    print(
        f'📅 DD Low Date: '
        f'{dd_end_time}'
    )


    # ========================================================
    # MAX CONSECUTIVE WINS / LOSSES
    # ========================================================

    max_wins = 0
    max_losses = 0

    current_wins = 0
    current_losses = 0

    for outcome in trades_df['Outcome']:

        if outcome == 'WIN':

            current_wins += 1
            current_losses = 0

            max_wins = max(
                max_wins,
                current_wins
            )

        elif outcome == 'LOSS':

            current_losses += 1
            current_wins = 0

            max_losses = max(
                max_losses,
                current_losses
            )

        else:

            current_wins = 0
            current_losses = 0


    print()
    print('🔥 STREAK ANALYSIS')
    print('------------------------------------------------------------')

    print(
        f'🔥 بیشترین WIN متوالی: '
        f'{max_wins}'
    )

    print(
        f'❄️ بیشترین LOSS متوالی: '
        f'{max_losses}'
    )


    # ========================================================
    # FIND ALL LOSS STREAKS
    # ========================================================

    print()
    print('🔴 تمام زنجیره‌های 5+ باخت متوالی')
    print('------------------------------------------------------------')

    loss_streaks = []

    start_idx = None

    for idx, outcome in enumerate(
        trades_df['Outcome']
    ):

        if outcome == 'LOSS':

            if start_idx is None:
                start_idx = idx

        else:

            if start_idx is not None:

                end_idx = idx - 1

                length = (
                    end_idx
                    - start_idx
                    + 1
                )

                if length >= 5:

                    streak = (
                        trades_df
                        .iloc[
                            start_idx:
                            end_idx + 1
                        ]
                    )

                    loss_streaks.append(
                        streak
                    )

                start_idx = None


    # اگر فایل با LOSS تمام شده باشد
    if start_idx is not None:

        end_idx = (
            len(trades_df) - 1
        )

        length = (
            end_idx
            - start_idx
            + 1
        )

        if length >= 5:

            streak = (
                trades_df
                .iloc[
                    start_idx:
                    end_idx + 1
                ]
            )

            loss_streaks.append(
                streak
            )


    if not loss_streaks:

        print(
            '✅ هیچ زنجیره 5+ باختی وجود ندارد.'
        )

    else:

        for n, streak in enumerate(
            loss_streaks,
            1
        ):

            streak_r = (
                streak['Return'].sum()
            )

            print()
            print(
                f'🔴 Streak #{n}: '
                f'{len(streak)} LOSS'
            )

            print(
                f'   شروع: '
                f'{streak.iloc[0]["Timestamp"]}'
            )

            print(
                f'   پایان: '
                f'{streak.iloc[-1]["Timestamp"]}'
            )

            print(
                f'   مجموع R: '
                f'{streak_r:.2f}R'
            )

            print(
                f'   ارزها: '
                f'{", ".join(streak["Symbol"].tolist())}'
            )


            if len(streak) >= 10:

                print()
                print(
                    '   🚨 جزئیات زنجیره 10+ باخت:'
                )

                for j, (_, row) in enumerate(
                    streak.iterrows(),
                    1
                ):

                    print(
                        f'      {j:02d}. '
                        f'{row["Timestamp"]} | '
                        f'{row["Symbol"]} | '
                        f'{row["Side"]} | '
                        f'{row["Reason"]} | '
                        f'BTC={row["BTC_Trend"]} | '
                        f'R={row["Return"]:.2f}'
                    )


    # ========================================================
    # WORST LOSS STREAK BY R
    # ========================================================

    worst_streak_r = None

    for streak in loss_streaks:

        r = streak['Return'].sum()

        if (
            worst_streak_r is None
            or r < worst_streak_r['R']
        ):

            worst_streak_r = {
                'R': r,
                'length': len(streak),
                'start':
                    streak.iloc[0]['Timestamp'],
                'end':
                    streak.iloc[-1]['Timestamp'],
            }


    if worst_streak_r:

        print()
        print(
            '💥 بدترین زنجیره بر اساس R'
        )

        print('------------------------------------------------------------')

        print(
            f'تعداد معاملات: '
            f'{worst_streak_r["length"]}'
        )

        print(
            f'ضرر: '
            f'{worst_streak_r["R"]:.2f}R'
        )

        print(
            f'شروع: '
            f'{worst_streak_r["start"]}'
        )

        print(
            f'پایان: '
            f'{worst_streak_r["end"]}'
        )


    # ========================================================
    # TIMEOUT ANALYSIS
    # ========================================================

    timeout_df = trades_df[
        trades_df['Outcome']
        == 'TIMEOUT'
    ]

    print()
    print('⏱️ TIMEOUT ANALYSIS')
    print('------------------------------------------------------------')

    if timeout_df.empty:

        print(
            '✅ هیچ Timeout وجود ندارد.'
        )

    else:

        timeout_wins = len(
            timeout_df[
                timeout_df['Return'] > 0
            ]
        )

        timeout_losses = len(
            timeout_df[
                timeout_df['Return'] < 0
            ]
        )

        timeout_avg = (
            timeout_df['Return'].mean()
        )

        timeout_net = (
            timeout_df['Return'].sum()
        )

        print(
            f'⏱️ تعداد Timeout: '
            f'{len(timeout_df)}'
        )

        print(
            f'🟢 Timeout مثبت: '
            f'{timeout_wins}'
        )

        print(
            f'🔴 Timeout منفی: '
            f'{timeout_losses}'
        )

        print(
            f'📊 میانگین Timeout: '
            f'{timeout_avg:.4f}R'
        )

        print(
            f'💰 مجموع Timeout: '
            f'{timeout_net:.2f}R'
        )


        print()
        print(
            '📋 جزئیات Timeout ها:'
        )

        for _, row in timeout_df.iterrows():

            print(
                f'   {row["Timestamp"]} | '
                f'{row["Symbol"]} | '
                f'{row["Side"]} | '
                f'{row["Candles_Held"]}h | '
                f'R={row["Return"]:.3f}'
            )


    # ========================================================
    # LOSS REASON ANALYSIS
    # ========================================================

    loss_df = trades_df[
        trades_df['Outcome']
        == 'LOSS'
    ]

    print()
    print('❌ تحلیل علت باخت‌ها')
    print('------------------------------------------------------------')

    if loss_df.empty:

        print(
            '🎉 هیچ باختی ثبت نشده.'
        )

    else:

        loss_reasons = (
            loss_df['Reason']
            .value_counts()
        )

        for reason, count in (
            loss_reasons.items()
        ):

            pct = (
                count
                / len(loss_df)
                * 100
            )

            print(
                f'🔴 {reason}: '
                f'{count} '
                f'({pct:.2f}% از کل باخت‌ها)'
            )


    # ========================================================
    # LONG / SHORT ANALYSIS
    # ========================================================

    print()
    print('📈 تحلیل LONG / SHORT')
    print('------------------------------------------------------------')

    for side in ['LONG', 'SHORT']:

        side_df = trades_df[
            trades_df['Side'] == side
        ]

        if side_df.empty:
            continue

        side_wins = len(
            side_df[
                side_df['Outcome'] == 'WIN'
            ]
        )

        side_losses = len(
            side_df[
                side_df['Outcome'] == 'LOSS'
            ]
        )

        side_timeout = len(
            side_df[
                side_df['Outcome'] == 'TIMEOUT'
            ]
        )

        side_closed = (
            side_wins
            + side_losses
        )

        side_wr = (
            side_wins
            / side_closed
            * 100
            if side_closed > 0
            else 0
        )

        side_r = (
            side_df['Return'].sum()
        )

        print(
            f'{side}: '
            f'Trades={len(side_df)} | '
            f'WIN={side_wins} | '
            f'LOSS={side_losses} | '
            f'TIMEOUT={side_timeout} | '
            f'WR={side_wr:.2f}% | '
            f'Net={side_r:.2f}R'
        )


    # ========================================================
    # BTC TREND ANALYSIS
    # ========================================================

    print()
    print('₿ تحلیل بر اساس روند BTC')
    print('------------------------------------------------------------')

    for trend in [
        'LONG',
        'SHORT',
        'NEUTRAL',
        'UNKNOWN'
    ]:

        btc_df = trades_df[
            trades_df['BTC_Trend']
            == trend
        ]

        if btc_df.empty:
            continue

        b_wins = len(
            btc_df[
                btc_df['Outcome'] == 'WIN'
            ]
        )

        b_losses = len(
            btc_df[
                btc_df['Outcome'] == 'LOSS'
            ]
        )

        b_closed = (
            b_wins + b_losses
        )

        b_wr = (
            b_wins
            / b_closed
            * 100
            if b_closed > 0
            else 0
        )

        b_r = (
            btc_df['Return'].sum()
        )

        print(
            f'BTC={trend}: '
            f'Trades={len(btc_df)} | '
            f'WR={b_wr:.2f}% | '
            f'Net={b_r:.2f}R'
        )


    # ========================================================
    # SYMBOL ANALYSIS
    # ========================================================

    print()
    print('🪙 گزارش تفکیک‌شده ارزها')
    print('------------------------------------------------------------')

    symbol_summary = []

    for sym in SYMBOLS.keys():

        sym_df = trades_df[
            trades_df['Symbol']
            == sym
        ]

        if sym_df.empty:
            continue

        s_wins = len(
            sym_df[
                sym_df['Outcome'] == 'WIN'
            ]
        )

        s_loss = len(
            sym_df[
                sym_df['Outcome'] == 'LOSS'
            ]
        )

        s_timeout = len(
            sym_df[
                sym_df['Outcome'] == 'TIMEOUT'
            ]
        )

        s_closed = (
            s_wins + s_loss
        )

        s_wr = (
            s_wins
            / s_closed
            * 100
            if s_closed > 0
            else 0
        )

        s_net_r = (
            sym_df['Return'].sum()
        )

        symbol_summary.append({

            'Symbol':
                sym,

            'Trades':
                len(sym_df),

            'Wins':
                s_wins,

            'Losses':
                s_loss,

            'Timeout':
                s_timeout,

            'WinRate':
                round(
                    s_wr,
                    2
                ),

            'Net_R':
                round(
                    s_net_r,
                    2
                ),
        })


    summary_df = pd.DataFrame(
        symbol_summary
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


    # ========================================================
    # WORST SYMBOLS
    # ========================================================

    print()
    print('⚠️ ارزهای ضعیف‌تر')
    print('------------------------------------------------------------')

    weak_symbols = (
        summary_df
        .sort_values(
            'WinRate'
        )
        .head(5)
    )

    print(
        weak_symbols.to_string(
            index=False
        )
    )


    # ========================================================
    # WORST LOSS EVENTS
    # ========================================================

    print()
    print('🚨 آخرین/بدترین باخت‌ها')
    print('------------------------------------------------------------')

    worst_losses = (
        loss_df
        .sort_values(
            'Timestamp'
        )
        .tail(20)
    )

    for _, row in worst_losses.iterrows():

        print(
            f'{row["Timestamp"]} | '
            f'{row["Symbol"]} | '
            f'{row["Side"]} | '
            f'{row["Reason"]} | '
            f'BTC={row["BTC_Trend"]} | '
            f'Hold={row["Candles_Held"]}h'
        )


    # ========================================================
    # FINAL RISK SUMMARY
    # ========================================================

    print()
    print('============================================================')
    print('🏁 خلاصه نهایی ریسک')
    print('============================================================')

    print(
        f'📊 Trades: {total}'
    )

    print(
        f'🎯 Win Rate: {win_rate:.2f}%'
    )

    print(
        f'💰 Net: {net_r:.2f}R'
    )

    print(
        f'📉 Max Drawdown: '
        f'{max_drawdown:.2f}R'
    )

    print(
        f'🔥 Max Win Streak: '
        f'{max_wins}'
    )

    print(
        f'❄️ Max Loss Streak: '
        f'{max_losses}'
    )

    print(
        f'⏱️ Timeouts: '
        f'{timeouts}'
    )

    print(
        f'📊 Avg R/Trade: '
        f'{avg_r:.4f}R'
    )

    print()
    print('✨ بک‌تست V8.1 به پایان رسید.')
