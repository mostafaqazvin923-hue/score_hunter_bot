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
# HUNTER-X V8.2
# TIMEOUT + DRAWDOWN + STREAK + REGIME DIAGNOSTIC
# ============================================================

exchange = ccxt.lbank({'enableRateLimit': True})

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

LOOKBACK_DAYS = 365

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

MAX_CONCURRENT_POSITIONS = 3

MAX_HOLD_CANDLES = 24

TARGET_R = 2.0
RISK_R = 1.0

# Diagnostic timeout alternatives
TIMEOUT_TESTS = [12, 24, 36]

# ============================================================
# DOWNLOAD DATA
# ============================================================

print("=" * 60)
print("📥 دریافت داده‌ها از LBank")
print("=" * 60)

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():

    print(f"📥 {symbol} ...")

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

            print(f"   ⚠️ Error {symbol}: {e}")
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
            f"   ✅ {symbol}: "
            f"{len(df1h)} candles"
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
        df4h['Close'] >
        df4h['Cloud_Top']
    )

    df4h['Trend_Short'] = (
        df4h['Close'] <
        df4h['Cloud_Bottom']
    )

    # Preserve V8.1 logic
    for col in [
        'Trend_Long',
        'Trend_Short',
        'Cloud_Top',
        'Cloud_Bottom'
    ]:
        df4h[col] = df4h[col].shift(1)

    df1h['Date_4H'] = (
        df1h['Date'].dt.floor('4h')
    )

    processed_data[symbol] = {
        '1h': df1h,
        '4h': df4h.set_index('Date')
    }


# ============================================================
# TIMESTAMPS
# ============================================================

all_timestamps = set()

for dat in processed_data.values():

    all_timestamps.update(
        dat['1h']['Date'].tolist()
    )

sorted_timestamps = sorted(
    list(all_timestamps)
)

# ============================================================
# DATAFRAMES
# ============================================================

dfs_1h = {
    sym: dat['1h'].set_index('Date')
    for sym, dat in processed_data.items()
}

dfs_4h = {
    sym: dat['4h']
    for sym, dat in processed_data.items()
}

# ============================================================
# BACKTEST
# ============================================================

active_positions = {}

all_trades = []

print()
print("=" * 60)
print("⚙️ شروع بک‌تست V8.2")
print("=" * 60)

for ts in sorted_timestamps:

    # --------------------------------------------------------
    # BTC REGIME
    # --------------------------------------------------------

    btc_allows_long = True
    btc_allows_short = True

    btc_trend_label = "NEUTRAL"

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

            btc_allows_long = btc_row.get(
                'Trend_Long',
                True
            )

            btc_allows_short = btc_row.get(
                'Trend_Short',
                True
            )

            if btc_allows_long:
                btc_trend_label = "LONG"

            elif btc_allows_short:
                btc_trend_label = "SHORT"

            else:
                btc_trend_label = "NEUTRAL"

    # --------------------------------------------------------
    # CLOSE EXISTING POSITIONS
    # --------------------------------------------------------

    symbols_to_close = []

    for symbol, pos in list(
        active_positions.items()
    ):

        if ts not in dfs_1h[symbol].index:
            continue

        c1h = dfs_1h[symbol].loc[ts]

        entry_time = pos['entry_time']
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

        hours_held = (
            ts - entry_time
        ).total_seconds() / 3600.0

        exit_price = None
        outcome = None
        reason = None

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

            if hit_sl and hit_tp:

                outcome = 'LOSS'
                reason = 'SL+TP_SAME_CANDLE'
                exit_price = pos['stop_loss']

            elif hit_sl:

                outcome = 'LOSS'
                reason = 'SL'
                exit_price = pos['stop_loss']

            elif hit_tp:

                outcome = 'WIN'
                reason = 'TP'
                exit_price = pos['take_profit']

            elif timeout:

                outcome = 'TIMEOUT'
                reason = 'TIMEOUT'
                exit_price = c1h['Close']

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        elif pos['side'] == 'SHORT':

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

            if hit_sl and hit_tp:

                outcome = 'LOSS'
                reason = 'SL+TP_SAME_CANDLE'
                exit_price = pos['stop_loss']

            elif hit_sl:

                outcome = 'LOSS'
                reason = 'SL'
                exit_price = pos['stop_loss']

            elif hit_tp:

                outcome = 'WIN'
                reason = 'TP'
                exit_price = pos['take_profit']

            elif timeout:

                outcome = 'TIMEOUT'
                reason = 'TIMEOUT'
                exit_price = c1h['Close']

        # ----------------------------------------------------
        # RECORD TRADE
        # ----------------------------------------------------

        if outcome is not None:

            entry_price = pos['entry_price']
            stop_loss = pos['stop_loss']
            take_profit = pos['take_profit']

            risk_distance = abs(
                entry_price - stop_loss
            )

            if risk_distance <= 0:
                r_raw = 0.0

            else:

                if pos['side'] == 'LONG':

                    price_pnl = (
                        exit_price -
                        entry_price
                    )

                else:

                    price_pnl = (
                        entry_price -
                        exit_price
                    )

                r_raw = (
                    price_pnl /
                    risk_distance
                )

            # Fees expressed in R
            fee_r = FEE_RATE * 2

            if outcome == 'WIN':

                r_real = (
                    TARGET_R -
                    fee_r
                )

            elif outcome == 'LOSS':

                r_real = (
                    -RISK_R -
                    fee_r
                )

            else:

                r_real = (
                    r_raw -
                    fee_r
                )

            # Maximum favorable excursion
            if pos['side'] == 'LONG':

                mfe = (
                    c1h['High'] -
                    entry_price
                ) / risk_distance

                mae = (
                    c1h['Low'] -
                    entry_price
                ) / risk_distance

            else:

                mfe = (
                    entry_price -
                    c1h['Low']
                ) / risk_distance

                mae = (
                    entry_price -
                    c1h['High']
                ) / risk_distance

            all_trades.append({

                'Timestamp': ts,
                'Symbol': symbol,
                'Side': pos['side'],

                'Entry_Time':
                    entry_time,

                'Entry_Price':
                    entry_price,

                'Stop_Loss':
                    stop_loss,

                'Take_Profit':
                    take_profit,

                'Exit_Price':
                    exit_price,

                'Candles_Held':
                    candles_held,

                'Hours_Held':
                    hours_held,

                'Outcome':
                    outcome,

                'Reason':
                    reason,

                'BTC_Trend':
                    btc_trend_label,

                'R_Raw':
                    r_raw,

                'Return':
                    r_real,

                'MFE_R':
                    mfe,

                'MAE_R':
                    mae,

                'SL_Distance_Pct':
                    (
                        risk_distance /
                        entry_price
                    ) * 100,

            })

            symbols_to_close.append(
                symbol
            )

    # --------------------------------------------------------
    # DELETE CLOSED POSITIONS
    # --------------------------------------------------------

    for sym in symbols_to_close:

        if sym in active_positions:
            del active_positions[sym]

    # --------------------------------------------------------
    # NEW ENTRIES
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

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
            ) and (
                df1h.iloc[i - 2]['Tenkan']
                <=
                df1h.iloc[i - 2]['Kijun']
            )

            kijun_pullback_long = (
                prev['Low']
                <= prev['Kijun']
                and
                prev['Close']
                > prev['Kijun']
            )

            if (
                tk_cross_long
                or kijun_pullback_long
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
                    entry_price -
                    stop_loss
                ) / entry_price

                if (
                    0.003
                    <= sl_dist_pct
                    <= 0.04
                ):

                    risk = (
                        entry_price -
                        stop_loss
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

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

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
            ) and (
                df1h.iloc[i - 2]['Tenkan']
                >=
                df1h.iloc[i - 2]['Kijun']
            )

            kijun_pullback_short = (
                prev['High']
                >= prev['Kijun']
                and
                prev['Close']
                < prev['Kijun']
            )

            if (
                tk_cross_short
                or kijun_pullback_short
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
                    stop_loss -
                    entry_price
                ) / entry_price

                if (
                    0.003
                    <= sl_dist_pct
                    <= 0.04
                ):

                    risk = (
                        stop_loss -
                        entry_price
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
# DATAFRAME
# ============================================================

trades = pd.DataFrame(
    all_trades
)

if trades.empty:

    print("❌ هیچ معامله‌ای ثبت نشد.")
    sys.exit(0)

trades.sort_values(
    'Timestamp',
    inplace=True
)

trades.reset_index(
    drop=True,
    inplace=True
)

# ============================================================
# BASIC STATS
# ============================================================

total_trades = len(trades)

wins = trades[
    trades['Outcome'] == 'WIN'
]

losses = trades[
    trades['Outcome'] == 'LOSS'
]

timeouts = trades[
    trades['Outcome'] == 'TIMEOUT'
]

win_count = len(wins)
loss_count = len(losses)
timeout_count = len(timeouts)

closed_count = (
    win_count +
    loss_count
)

win_rate = (
    win_count /
    closed_count * 100
    if closed_count
    else 0
)

net_r = trades['Return'].sum()

avg_r = trades['Return'].mean()

# ============================================================
# PROFIT FACTOR
# ============================================================

gross_profit = (
    trades.loc[
        trades['Return'] > 0,
        'Return'
    ].sum()
)

gross_loss = abs(
    trades.loc[
        trades['Return'] < 0,
        'Return'
    ].sum()
)

profit_factor = (
    gross_profit /
    gross_loss
    if gross_loss > 0
    else np.inf
)

# ============================================================
# EXPECTANCY
# ============================================================

expectancy = avg_r

# ============================================================
# EQUITY / DRAWDOWN
# ============================================================

trades['Equity_R'] = (
    trades['Return'].cumsum()
)

trades['Peak_R'] = (
    trades['Equity_R']
    .cummax()
)

trades['Drawdown_R'] = (
    trades['Equity_R']
    - trades['Peak_R']
)

max_drawdown = (
    trades['Drawdown_R'].min()
)

max_dd_index = (
    trades['Drawdown_R'].idxmin()
)

dd_low_date = (
    trades.loc[
        max_dd_index,
        'Timestamp'
    ]
)

peak_before_dd = (
    trades.loc[
        :max_dd_index,
        'Equity_R'
    ].max()
)

# ============================================================
# STREAK ANALYSIS
# ============================================================

max_win_streak = 0
max_loss_streak = 0

current_win = 0
current_loss = 0

loss_streaks = []
current_loss_rows = []

for idx, row in trades.iterrows():

    if row['Outcome'] == 'WIN':

        current_win += 1
        current_loss = 0

        if current_loss_rows:

            if len(current_loss_rows) >= 5:

                loss_streaks.append(
                    current_loss_rows
                )

            current_loss_rows = []

    elif row['Outcome'] == 'LOSS':

        current_loss += 1
        current_win = 0

        current_loss_rows.append(
            row
        )

    else:

        current_win = 0
        current_loss = 0

        if current_loss_rows:

            if len(current_loss_rows) >= 5:

                loss_streaks.append(
                    current_loss_rows
                )

            current_loss_rows = []

    max_win_streak = max(
        max_win_streak,
        current_win
    )

    max_loss_streak = max(
        max_loss_streak,
        current_loss
    )

if current_loss_rows:

    if len(current_loss_rows) >= 5:
        loss_streaks.append(
            current_loss_rows
        )

# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("📊 گزارش کامل HUNTER-X V8.2")
print("=" * 60)

print()
print("📌 آمار اصلی")
print("-" * 60)

print(
    f"🔸 کل معاملات: {total_trades}"
)

print(
    f"🟢 WIN: {win_count}"
)

print(
    f"🔴 LOSS: {loss_count}"
)

print(
    f"⏱️ TIMEOUT: {timeout_count}"
)

print(
    f"🎯 Win Rate بدون Timeout: "
    f"{win_rate:.2f}%"
)

print(
    f"💰 Net R: {net_r:.2f}R"
)

print(
    f"📊 Average R / Trade: "
    f"{avg_r:.4f}R"
)

print(
    f"📈 Profit Factor: "
    f"{profit_factor:.3f}"
)

print(
    f"📐 Expectancy: "
    f"{expectancy:.4f}R"
)

# ============================================================
# EXIT REASONS
# ============================================================

print()
print("📌 علت خروج")
print("-" * 60)

reason_counts = (
    trades['Reason']
    .value_counts()
)

for reason, count in reason_counts.items():

    pct = (
        count /
        total_trades *
        100
    )

    print(
        f"{reason:<28} "
        f"{count:>5} "
        f"({pct:.2f}%)"
    )

# ============================================================
# TIMEOUT ANALYSIS
# ============================================================

print()
print("=" * 60)
print("⏱️ TIMEOUT DEEP ANALYSIS")
print("=" * 60)

print(
    f"Total Timeout: {timeout_count}"
)

if timeout_count > 0:

    timeout_net = (
        timeouts['Return'].sum()
    )

    timeout_avg = (
        timeouts['Return'].mean()
    )

    timeout_median = (
        timeouts['Return'].median()
    )

    timeout_positive = (
        timeouts[
            timeouts['Return'] > 0
        ]
    )

    timeout_negative = (
        timeouts[
            timeouts['Return'] < 0
        ]
    )

    timeout_zero = (
        timeouts[
            timeouts['Return'] == 0
        ]
    )

    print(
        f"🟢 Timeout سودده: "
        f"{len(timeout_positive)}"
    )

    print(
        f"🔴 Timeout ضررده: "
        f"{len(timeout_negative)}"
    )

    print(
        f"⚪ Timeout خنثی: "
        f"{len(timeout_zero)}"
    )

    print(
        f"💰 Timeout Net R: "
        f"{timeout_net:.2f}R"
    )

    print(
        f"📊 Timeout Avg R: "
        f"{timeout_avg:.4f}R"
    )

    print(
        f"📊 Timeout Median R: "
        f"{timeout_median:.4f}R"
    )

    print()
    print("⏱️ مدت نگهداری Timeout")

    print(
        timeouts[
            'Hours_Held'
        ].describe().to_string()
    )

# ============================================================
# TIMEOUT BY SYMBOL
# ============================================================

print()
print("⏱️ TIMEOUT بر اساس ارز")
print("-" * 60)

if timeout_count > 0:

    timeout_symbol = (
        timeouts
        .groupby('Symbol')
        .agg(
            Trades=('Symbol', 'size'),
            Avg_R=('Return', 'mean'),
            Net_R=('Return', 'sum'),
            Avg_Hours=('Hours_Held', 'mean')
        )
        .sort_values(
            'Net_R'
        )
    )

    print(
        timeout_symbol.to_string(
            float_format=lambda x:
            f"{x:.3f}"
        )
    )

# ============================================================
# TIMEOUT LONG / SHORT
# ============================================================

print()
print("⏱️ TIMEOUT بر اساس جهت")
print("-" * 60)

if timeout_count > 0:

    timeout_side = (
        timeouts
        .groupby('Side')
        .agg(
            Trades=('Side', 'size'),
            Avg_R=('Return', 'mean'),
            Net_R=('Return', 'sum'),
            Avg_Hours=('Hours_Held', 'mean')
        )
    )

    print(
        timeout_side.to_string(
            float_format=lambda x:
            f"{x:.3f}"
        )
    )

# ============================================================
# TIMEOUT BTC REGIME
# ============================================================

print()
print("⏱️ TIMEOUT بر اساس وضعیت BTC")
print("-" * 60)

if timeout_count > 0:

    timeout_btc = (
        timeouts
        .groupby('BTC_Trend')
        .agg(
            Trades=('BTC_Trend', 'size'),
            Avg_R=('Return', 'mean'),
            Net_R=('Return', 'sum')
        )
        .sort_values(
            'Net_R'
        )
    )

    print(
        timeout_btc.to_string(
            float_format=lambda x:
            f"{x:.3f}"
        )
    )

# ============================================================
# TIMEOUT BY HOLDING TIME
# ============================================================

print()
print("⏱️ TIMEOUT بر اساس مدت نگهداری")
print("-" * 60)

if timeout_count > 0:

    bins = [
        -1,
        12,
        18,
        24,
        30,
        36,
        1000
    ]

    labels = [
        "<=12h",
        "13-18h",
        "19-24h",
        "25-30h",
        "31-36h",
        ">36h"
    ]

    timeout_copy = (
        timeouts.copy()
    )

    timeout_copy[
        'Hold_Bucket'
    ] = pd.cut(
        timeout_copy[
            'Hours_Held'
        ],
        bins=bins,
        labels=labels
    )

    hold_analysis = (
        timeout_copy
        .groupby(
            'Hold_Bucket',
            observed=True
        )
        .agg(
            Trades=('Return', 'size'),
            Avg_R=('Return', 'mean'),
            Net_R=('Return', 'sum')
        )
    )

    print(
        hold_analysis.to_string(
            float_format=lambda x:
            f"{x:.3f}"
        )
    )

# ============================================================
# MFE / MAE TIMEOUT
# ============================================================

print()
print("🎯 TIMEOUT MFE / MAE")
print("-" * 60)

if timeout_count > 0:

    print(
        f"Average MFE: "
        f"{timeouts['MFE_R'].mean():.3f}R"
    )

    print(
        f"Median MFE: "
        f"{timeouts['MFE_R'].median():.3f}R"
    )

    print(
        f"Average MAE: "
        f"{timeouts['MAE_R'].mean():.3f}R"
    )

    print(
        f"Median MAE: "
        f"{timeouts['MAE_R'].median():.3f}R"
    )

    print()

    reached_1R = (
        timeouts[
            timeouts['MFE_R'] >= 1.0
        ]
    )

    reached_1_5R = (
        timeouts[
            timeouts['MFE_R'] >= 1.5
        ]
    )

    reached_2R = (
        timeouts[
            timeouts['MFE_R'] >= 2.0
        ]
    )

    print(
        f"Timeoutهایی که حداقل +1R "
        f"رفته‌اند: "
        f"{len(reached_1R)} "
        f"({len(reached_1R)/timeout_count*100:.2f}%)"
    )

    print(
        f"Timeoutهایی که حداقل +1.5R "
        f"رفته‌اند: "
        f"{len(reached_1_5R)} "
        f"({len(reached_1_5R)/timeout_count*100:.2f}%)"
    )

    print(
        f"Timeoutهایی که حداقل +2R "
        f"رفته‌اند: "
        f"{len(reached_2R)} "
        f"({len(reached_2R)/timeout_count*100:.2f}%)"
    )

# ============================================================
# LOSS ANALYSIS
# ============================================================

print()
print("=" * 60)
print("🔴 LOSS ANALYSIS")
print("=" * 60)

loss_symbol = (
    losses
    .groupby('Symbol')
    .agg(
        Losses=('Symbol', 'size'),
        Avg_R=('Return', 'mean'),
        Net_R=('Return', 'sum')
    )
    .sort_values(
        'Losses',
        ascending=False
    )
)

print(
    loss_symbol.to_string(
        float_format=lambda x:
        f"{x:.3f}"
    )
)

print()
print("🔴 LOSS بر اساس BTC Trend")

loss_btc = (
    losses
    .groupby('BTC_Trend')
    .agg(
        Trades=('BTC_Trend', 'size'),
        Avg_R=('Return', 'mean'),
        Net_R=('Return', 'sum')
    )
)

print(
    loss_btc.to_string(
        float_format=lambda x:
        f"{x:.3f}"
    )
)

# ============================================================
# LONG / SHORT PERFORMANCE
# ============================================================

print()
print("=" * 60)
print("📊 LONG / SHORT ANALYSIS")
print("=" * 60)

side_stats = (
    trades
    .groupby('Side')
    .agg(
        Trades=('Side', 'size'),
        Avg_R=('Return', 'mean'),
        Net_R=('Return', 'sum')
    )
)

for side in ['LONG', 'SHORT']:

    if side in side_stats.index:

        side_data = trades[
            trades['Side'] == side
        ]

        side_wins = len(
            side_data[
                side_data['Outcome']
                == 'WIN'
            ]
        )

        side_losses = len(
            side_data[
                side_data['Outcome']
                == 'LOSS'
            ]
        )

        side_closed = (
            side_wins +
            side_losses
        )

        side_wr = (
            side_wins /
            side_closed * 100
            if side_closed
            else 0
        )

        print(
            f"{side}: "
            f"Trades={len(side_data)} | "
            f"WR={side_wr:.2f}% | "
            f"Net={side_data['Return'].sum():.2f}R"
        )

# ============================================================
# STREAK REPORT
# ============================================================

print()
print("=" * 60)
print("🔥 LOSS STREAK ANALYSIS")
print("=" * 60)

print(
    f"🔥 Max WIN streak: "
    f"{max_win_streak}"
)

print(
    f"❄️ Max LOSS streak: "
    f"{max_loss_streak}"
)

for n, streak in enumerate(
    loss_streaks,
    start=1
):

    if len(streak) < 5:
        continue

    streak_df = pd.DataFrame(
        streak
    )

    print()
    print(
        f"🔴 Streak #{n}: "
        f"{len(streak_df)} LOSS"
    )

    print(
        f"   شروع: "
        f"{streak_df['Timestamp'].iloc[0]}"
    )

    print(
        f"   پایان: "
        f"{streak_df['Timestamp'].iloc[-1]}"
    )

    print(
        f"   مجموع R: "
        f"{streak_df['Return'].sum():.2f}R"
    )

    print(
        "   ارزها: "
        +
        ", ".join(
            streak_df['Symbol']
            .tolist()
        )
    )

# ============================================================
# WORST TIMEOUTS
# ============================================================

print()
print("=" * 60)
print("🔻 بدترین TIMEOUT ها")
print("=" * 60)

if timeout_count > 0:

    worst_timeouts = (
        timeouts
        .sort_values(
            'Return'
        )
        .head(20)
    )

    for _, row in (
        worst_timeouts.iterrows()
    ):

        print(
            f"{row['Timestamp']} | "
            f"{row['Symbol']} | "
            f"{row['Side']} | "
            f"R={row['Return']:.3f} | "
            f"Hold={row['Hours_Held']:.1f}h | "
            f"BTC={row['BTC_Trend']} | "
            f"MFE={row['MFE_R']:.2f}R"
        )

# ============================================================
# BEST TIMEOUTS
# ============================================================

print()
print("=" * 60)
print("🟢 بهترین TIMEOUT ها")
print("=" * 60)

if timeout_count > 0:

    best_timeouts = (
        timeouts
        .sort_values(
            'Return',
            ascending=False
        )
        .head(20)
    )

    for _, row in (
        best_timeouts.iterrows()
    ):

        print(
            f"{row['Timestamp']} | "
            f"{row['Symbol']} | "
            f"{row['Side']} | "
            f"R={row['Return']:.3f} | "
            f"Hold={row['Hours_Held']:.1f}h | "
            f"BTC={row['BTC_Trend']} | "
            f"MFE={row['MFE_R']:.2f}R"
        )

# ============================================================
# TIMEOUT SCENARIO SIMULATION
#
# IMPORTANT:
# This is a diagnostic approximation.
# It does NOT rerun entries for different timeout values.
# It only evaluates existing trades at their observed path.
# ============================================================

print()
print("=" * 60)
print("🧪 TIMEOUT SCENARIO DIAGNOSTIC")
print("=" * 60)

print(
    "⚠️ این بخش فقط برای Diagnostic است "
    "و Entryها را دوباره تولید نمی‌کند."
)

for timeout_hours in TIMEOUT_TESTS:

    scenario_returns = []

    for _, trade in trades.iterrows():

        if trade['Outcome'] != 'TIMEOUT':

            scenario_returns.append(
                trade['Return']
            )

            continue

        # We do not have every intermediate
        # candle stored in the trade table.
        # Therefore we cannot honestly
        # reconstruct 12h/36h exits here.

        scenario_returns.append(
            trade['Return']
        )

    scenario_net = sum(
        scenario_returns
    )

    print(
        f"Timeout={timeout_hours}h | "
        f"Net observed={scenario_net:.2f}R"
    )

print()
print(
    "⚠️ برای مقایسه واقعی 12h/24h/36h "
    "باید کندل‌های داخل هر معامله ذخیره شوند."
)

# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

print()
print("=" * 60)
print("🏁 خلاصه نهایی V8.2")
print("=" * 60)

print(
    f"📊 Trades: "
    f"{total_trades}"
)

print(
    f"🟢 Wins: "
    f"{win_count}"
)

print(
    f"🔴 Losses: "
    f"{loss_count}"
)

print(
    f"⏱️ Timeouts: "
    f"{timeout_count}"
)

print(
    f"🎯 WR without Timeout: "
    f"{win_rate:.2f}%"
)

print(
    f"💰 Net R: "
    f"{net_r:.2f}R"
)

print(
    f"📈 Profit Factor: "
    f"{profit_factor:.3f}"
)

print(
    f"📐 Expectancy: "
    f"{expectancy:.4f}R"
)

print(
    f"📉 Max Drawdown: "
    f"{max_drawdown:.2f}R"
)

print(
    f"🔥 Max Win Streak: "
    f"{max_win_streak}"
)

print(
    f"❄️ Max Loss Streak: "
    f"{max_loss_streak}"
)

# ============================================================
# SAVE CSV
# ============================================================

csv_file = "hunter_x_v8_2_trades.csv"

trades.to_csv(
    csv_file,
    index=False
)

print()
print(
    f"💾 فایل معاملات ذخیره شد: "
    f"{csv_file}"
)

print()
print("=" * 60)
print("✨ HUNTER-X V8.2 به پایان رسید")
print("=" * 60)
