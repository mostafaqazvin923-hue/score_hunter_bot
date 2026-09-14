import subprocess
import sys
import time
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# V41 - 4H LBank Backtest
# اصلاحات:
# 1) Entry fee همان لحظه از cash کسر می‌شود.
# 2) Exposure شامل Entry Fee است.
# 3) Cash accounting خروج دقیق است.
# 4) Pending signal فقط برای کندل بلافاصله بعد معتبر است.
# 5) Trailing از ATR کندل قبل استفاده می‌کند.
# 6) Mark-to-Market Equity در هر کندل محاسبه می‌شود.
# 7) Gap detection سخت‌گیرانه.
# 8) Timestamp/index lookup بهینه شده.
# 9) در صورت شکست دریافت کامل دیتا، نماد ناقص وارد بک‌تست نمی‌شود.
# ============================================================

exchange = ccxt.lbank({'enableRateLimit': True})

SYMBOLS = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT',
    'SOL': 'SOL/USDT',
    'XRP': 'XRP/USDT',
    'LINK': 'LINK/USDT',
    'UNI': 'UNI/USDT',
    'ICP': 'ICP/USDT',
    'INJ': 'INJ/USDT',
    'ATOM': 'ATOM/USDT',
    'RENDER': 'RENDER/USDT',
    'XLM': 'XLM/USDT',
    'AAVE': 'AAVE/USDT',
    'WIF': 'WIF/USDT',
    'ONDO': 'ONDO/USDT',
    'SUI': 'SUI/USDT',
    'TIA': 'TIA/USDT',
    'FET': 'FET/USDT',
}

REMOVED_COINS = {
    'NEAR', 'OP', 'SEI', 'ARB', 'AVAX', 'DOT', 'ETC', 'SHIB',
    'STX', 'RUNE', 'MKR', 'APT', 'LTC', 'PENDLE', 'AR', 'IMX',
    'PEPE', 'BONK',
}

SYMBOLS = {k: v for k, v in SYMBOLS.items() if k not in REMOVED_COINS}

# -----------------------------
# Parameters
# -----------------------------
LOOKBACK_DAYS = 365
TIMEFRAME_HOURS = 4
EMA_WARMUP = 200

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

MAX_POSITIONS = 5
RISK_PER_TRADE = 0.01
MAX_CASH_ALLOCATION = 0.95

INITIAL_CAPITAL = 10000.0
ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0

TIMEOUT_CANDLES = 45
COOLDOWN_CANDLES = 10

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print('============================================================')
print('📥 دریافت و پاکسازی داده‌ها - V41')
print('============================================================')

processed_data = {}


def fetch_symbol_data(symbol, lbank_symbol):
    """دریافت کامل داده 4H با retry و جلوگیری از ورود دیتای ناقص."""
    all_ohlcv = []
    current_since = since_timestamp
    now_timestamp = exchange.milliseconds()

    while current_since < now_timestamp:
        success = False
        ohlcv = None

        for attempt in range(3):
            try:
                ohlcv = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe='4h',
                    since=current_since,
                    limit=1000,
                )
                if ohlcv:
                    success = True
                    break
            except Exception as e:
                print(
                    f'⚠️ خطای اتصال {symbol} '
                    f'(تلاش {attempt + 1}/3): {e}'
                )
                if attempt < 2:
                    time.sleep(2)

        if not success or not ohlcv:
            print(f'❌ دریافت کامل دیتای {symbol} شکست خورد؛ حذف نماد.')
            return None

        # جلوگیری از pagination بی‌نهایت
        if ohlcv[-1][0] < current_since:
            print(f'❌ Timestamp نامعتبر برای {symbol}; حذف نماد.')
            return None

        all_ohlcv.extend(ohlcv)
        next_since = ohlcv[-1][0] + 1

        if next_since <= current_since:
            print(f'❌ Pagination نامعتبر برای {symbol}; حذف نماد.')
            return None

        current_since = next_since

        if len(ohlcv) < 1000:
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'],
    )

    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # پاکسازی
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # حذف آخرین کندل، چون ممکن است هنوز کامل نشده باشد.
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        print(f'⚠️ دیتای کافی برای {symbol} وجود ندارد؛ حذف نماد.')
        return None

    # -----------------------------
    # Gap detection
    # -----------------------------
    time_diffs = df['Date'].diff().dropna()
    expected_delta = timedelta(hours=TIMEFRAME_HOURS)
    max_gap = time_diffs.max()

    if max_gap > timedelta(hours=4, minutes=10):
        print(
            f'⚠️ گپ دیتای {symbol}: {max_gap}؛ حذف نماد.'
        )
        return None

    # timestamp alignment
    invalid_alignment = (
        (df['Date'].dt.minute != 0)
        | (df['Date'].dt.second != 0)
        | (df['Date'].dt.microsecond != 0)
    )

    if invalid_alignment.any():
        print(f'⚠️ Timestamp غیر استاندارد در {symbol}; حذف نماد.')
        return None

    # -----------------------------
    # Indicators
    # -----------------------------
    prev_close = df['Close'].shift(1)

    tr1 = df['High'] - df['Low']
    tr2 = np.abs(df['High'] - prev_close)
    tr3 = np.abs(df['Low'] - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder ATR
    df['ATR'] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False
    ).mean()

    df['EMA20'] = df['Close'].ewm(
        span=20,
        adjust=False
    ).mean()

    df['EMA50'] = df['Close'].ewm(
        span=50,
        adjust=False
    ).mean()

    df['EMA200'] = df['Close'].ewm(
        span=200,
        adjust=False
    ).mean()

    df['Mom_Short'] = (
        df['Close'] - df['Close'].shift(10)
    ) / df['Close'].shift(10)

    df['Mom_Long'] = (
        df['Close'] - df['Close'].shift(30)
    ) / df['Close'].shift(30)

    df.set_index('Date', inplace=True)

    # mapping برای lookup سریع
    index_to_i = {ts: i for i, ts in enumerate(df.index)}

    return {
        'df': df,
        'index_to_i': index_to_i,
    }


for symbol, lbank_symbol in SYMBOLS.items():
    result = fetch_symbol_data(symbol, lbank_symbol)
    if result is not None:
        processed_data[symbol] = result

print(
    f'✅ تعداد نمادهای معتبر: {len(processed_data)} از {len(SYMBOLS)}'
)

if not processed_data:
    raise RuntimeError('هیچ دیتای معتبری دریافت نشد.')

print('⚙️ شروع بک‌تست V41...')

# ============================================================
# Unified timestamps
# ============================================================
all_timestamps = set()

for item in processed_data.values():
    all_timestamps.update(item['df'].index.tolist())

sorted_timestamps = sorted(all_timestamps)

# ============================================================
# Portfolio state
# ============================================================
active_positions = {}
all_trades = []
cooldown_timers = {}
pending_signals = {}

cash = INITIAL_CAPITAL
equity_curve = []


def mark_to_market_equity(ts):
    floating_pnl = 0.0

    for symbol, pos in active_positions.items():
        data = processed_data[symbol]
        df = data['df']

        if ts in df.index:
            close_price = float(df.loc[ts, 'Close'])
            floating_pnl += (
                close_price - pos['entry_price']
            ) * pos['position_size']

    return cash + floating_pnl


for ts in sorted_timestamps:

    # ========================================================
    # 0. Cooldown
    # ========================================================
    for sym in list(cooldown_timers.keys()):
        cooldown_timers[sym] -= 1

        if cooldown_timers[sym] <= 0:
            del cooldown_timers[sym]

    # ========================================================
    # 1. Execute pending orders
    # ========================================================
    for symbol in list(pending_signals.keys()):

        sig_info = pending_signals[symbol]

        # Signal فقط روی کندل بلافاصله بعد معتبر است.
        if sig_info['valid_timestamp'] != ts:
            del pending_signals[symbol]
            continue

        if len(active_positions) >= MAX_POSITIONS:
            del pending_signals[symbol]
            continue

        if symbol in active_positions:
            del pending_signals[symbol]
            continue

        if symbol not in processed_data:
            del pending_signals[symbol]
            continue

        data = processed_data[symbol]
        df = data['df']

        if ts not in df.index:
            del pending_signals[symbol]
            continue

        del pending_signals[symbol]

        c4h = df.loc[ts]

        entry_price = float(c4h['Open']) * (1 + SLIPPAGE)

        initial_sl = (
            entry_price
            - INITIAL_ATR_MULTIPLIER * sig_info['atr']
        )

        initial_risk_per_unit = entry_price - initial_sl

        if initial_risk_per_unit <= 0:
            continue

        sl_dist_pct = (
            initial_risk_per_unit / entry_price
        )

        if not (0.01 <= sl_dist_pct <= 0.04):
            continue

        # ----------------------------------------------------
        # Equity فعلی برای position sizing
        # ----------------------------------------------------
        current_equity = mark_to_market_equity(ts)

        risk_amount_usd = current_equity * RISK_PER_TRADE

        position_size_units = (
            risk_amount_usd / initial_risk_per_unit
        )

        # ----------------------------------------------------
        # Exposure:
        # حداکثر 95% از cash و شامل fee ورود
        # ----------------------------------------------------
        max_allowed_cash = cash * MAX_CASH_ALLOCATION

        if max_allowed_cash <= 0:
            continue

        max_units_by_cash = (
            max_allowed_cash
            / (entry_price * (1 + FEE_RATE))
        )

        position_size_units = min(
            position_size_units,
            max_units_by_cash
        )

        if position_size_units <= 0:
            continue

        entry_value = (
            position_size_units * entry_price
        )

        entry_fee = entry_value * FEE_RATE

        total_entry_cost = entry_value + entry_fee

        if total_entry_cost > max_allowed_cash:
            continue

        # ----------------------------------------------------
        # ثبت پوزیشن
        # ----------------------------------------------------
        active_positions[symbol] = {
            'side': 'LONG',
            'entry_price': entry_price,
            'stop_loss': initial_sl,
            'highest_price': entry_price,

            # High همین کندل فقط برای کندل بعد استفاده می‌شود.
            'pending_highest_price': float(c4h['High']),

            'initial_risk_unit': initial_risk_per_unit,
            'position_size': position_size_units,
            'entry_timestamp': ts,

            'entry_value': entry_value,
            'entry_fee': entry_fee,
        }

        # پرداخت واقعی هزینه خرید + fee
        cash -= total_entry_cost

    # ========================================================
    # 2. Manage open positions
    # ========================================================
    symbols_to_close = []

    for symbol, pos in list(active_positions.items()):

        if symbol not in processed_data:
            continue

        data = processed_data[symbol]
        df = data['df']
        index_to_i = data['index_to_i']

        if ts not in df.index:
            continue

        c4h = df.loc[ts]
        curr_i = index_to_i[ts]

        # ----------------------------------------------------
        # ATR کندل قبل
        # ----------------------------------------------------
        if curr_i > 0:
            prev_atr = float(
                df.iloc[curr_i - 1]['ATR']
            )
        else:
            prev_atr = float(c4h['ATR'])

        # ----------------------------------------------------
        # Trailing stop
        # High کندل قبلی + ATR کندل قبلی
        # ----------------------------------------------------
        if pos['pending_highest_price'] > pos['highest_price']:

            pos['highest_price'] = (
                pos['pending_highest_price']
            )

            new_trailing_sl = (
                pos['highest_price']
                - TRAILING_ATR_MULTIPLIER * prev_atr
            )

            if new_trailing_sl > pos['stop_loss']:
                pos['stop_loss'] = new_trailing_sl

        # ----------------------------------------------------
        # Stop check
        # ----------------------------------------------------
        hit_sl = (
            float(c4h['Low']) <= pos['stop_loss']
        )

        # High کندل فعلی فقط برای کندل بعد ذخیره می‌شود.
        current_high = float(c4h['High'])

        if current_high > pos['pending_highest_price']:
            pos['pending_highest_price'] = current_high

        # ----------------------------------------------------
        # Holding period
        # ----------------------------------------------------
        entry_i = index_to_i.get(
            pos['entry_timestamp'],
            curr_i
        )

        candles_held = curr_i - entry_i

        is_timeout = (
            candles_held >= TIMEOUT_CANDLES
        )

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------
        if hit_sl or is_timeout:

            if hit_sl:
                # اگر قیمت افتتاحیه زیر stop باشد، gap-down
                # با Open اجرا می‌شود؛ در غیر این صورت stop.
                stop_execution_price = min(
                    pos['stop_loss'],
                    float(c4h['Open'])
                )

                exit_p = (
                    stop_execution_price
                    * (1 - SLIPPAGE)
                )

            else:
                exit_p = (
                    float(c4h['Close'])
                    * (1 - SLIPPAGE)
                )

            exit_value = (
                exit_p * pos['position_size']
            )

            exit_fee = exit_value * FEE_RATE

            # PnL خالص واقعی
            dollar_pnl = (
                exit_value
                - pos['entry_value']
                - pos['entry_fee']
                - exit_fee
            )

            # دریافت مبلغ فروش پس از exit fee
            cash += exit_value - exit_fee

            # ------------------------------------------------
            # R calculation
            # ------------------------------------------------
            price_diff = (
                exit_p - pos['entry_price']
            )

            raw_r = (
                price_diff
                / pos['initial_risk_unit']
            )

            fee_impact_r = (
                (
                    pos['entry_price'] * FEE_RATE
                    + exit_p * FEE_RATE
                )
                / pos['initial_risk_unit']
            )

            r_real = raw_r - fee_impact_r

            outcome = (
                'WIN'
                if r_real > 0
                else 'LOSS'
            )

            all_trades.append({
                'Timestamp': ts,
                'Symbol': symbol,
                'Outcome': outcome,
                'Return': r_real,
                'PnL': dollar_pnl,
                'EntryPrice': pos['entry_price'],
                'ExitPrice': exit_p,
            })

            symbols_to_close.append(symbol)

            if hit_sl:
                cooldown_timers[symbol] = (
                    COOLDOWN_CANDLES
                )

    # حذف پوزیشن‌های بسته‌شده
    for symbol in symbols_to_close:
        del active_positions[symbol]

    # ========================================================
    # 3. Mark-to-Market Equity
    # ========================================================
    total_equity = mark_to_market_equity(ts)

    equity_curve.append({
        'Timestamp': ts,
        'Equity': total_equity,
        'Cash': cash,
        'OpenPositions': len(active_positions),
    })

    # ========================================================
    # 4. Generate signals
    # ========================================================
    current_scores = {}

    for symbol, data in processed_data.items():

        df = data['df']

        if ts not in df.index:
            continue

        val = df.loc[ts, 'Mom_Long']

        if pd.notna(val):
            current_scores[symbol] = float(val)

    ranked_symbols = sorted(
        current_scores.keys(),
        key=lambda x: current_scores[x],
        reverse=True
    )

    for symbol in ranked_symbols:

        if symbol in active_positions:
            continue

        if symbol in cooldown_timers:
            continue

        df = processed_data[symbol]['df']
        index_to_i = processed_data[symbol]['index_to_i']

        if ts not in df.index:
            continue

        i = index_to_i[ts]

        if i < EMA_WARMUP:
            continue

        # برای اجرای سفارش در کندل بعد باید واقعاً کندل بعد وجود داشته باشد.
        if i + 1 >= len(df):
            continue

        c4h = df.iloc[i]

        regime_bull = (
            c4h['Close'] > c4h['EMA20']
            and c4h['EMA20'] > c4h['EMA50']
            and c4h['EMA50'] > c4h['EMA200']
        )

        valid_trend = (
            regime_bull
            and c4h['Mom_Short'] > 0.015
            and c4h['Mom_Long'] > 0.04
            and pd.notna(c4h['ATR'])
        )

        if not valid_trend:
            continue

        next_ts = df.index[i + 1]

        pending_signals[symbol] = {
            'atr': float(c4h['ATR']),
            'valid_timestamp': next_ts,
        }


# ============================================================
# Final report
# ============================================================
print('\n============================================================')
print('📊 گزارش نهایی استراتژی V41')
print('============================================================')

if all_trades:

    trades_df = pd.DataFrame(all_trades)

    # ترتیب واقعی زمانی خروج
    trades_df.sort_values(
        ['Timestamp', 'Symbol'],
        inplace=True
    )

    tot_trades = len(trades_df)

    tot_wins = int(
        (trades_df['Outcome'] == 'WIN').sum()
    )

    tot_losses = int(
        (trades_df['Outcome'] == 'LOSS').sum()
    )

    win_rate = (
        tot_wins / tot_trades * 100
        if tot_trades > 0
        else 0
    )

    net_r = trades_df['Return'].sum()
    net_pnl = trades_df['PnL'].sum()

    # --------------------------------------------------------
    # Equity / Drawdown
    # --------------------------------------------------------
    eq_df = pd.DataFrame(equity_curve)

    eq_df['Peak'] = (
        eq_df['Equity'].cummax()
    )

    eq_df['Drawdown'] = (
        (eq_df['Equity'] - eq_df['Peak'])
        / eq_df['Peak']
        * 100
    )

    max_dd = eq_df['Drawdown'].min()

    # --------------------------------------------------------
    # Losing streak
    # --------------------------------------------------------
    max_losses = 0
    current_losses = 0
    loss_sequences = []
    temp_loss_seq = 0

    for outcome in trades_df['Outcome']:

        if outcome == 'WIN':

            current_losses = 0

            if temp_loss_seq > 0:
                loss_sequences.append(
                    temp_loss_seq
                )
                temp_loss_seq = 0

        else:

            current_losses += 1
            temp_loss_seq += 1

            max_losses = max(
                max_losses,
                current_losses
            )

    if temp_loss_seq > 0:
        loss_sequences.append(
            temp_loss_seq
        )

    final_equity = (
        eq_df.iloc[-1]['Equity']
        if not eq_df.empty
        else INITIAL_CAPITAL
    )

    print(f'🔸 تعداد کل معاملات: {tot_trades}')
    print(f'🔸 WIN: {tot_wins}')
    print(f'🔸 LOSS: {tot_losses}')
    print(f'❄️ حداکثر ضررهای متوالی: {max_losses}')
    print(f'📉 Max Drawdown: {max_dd:.2f}%')
    print(f'🎯 Win Rate: {win_rate:.2f}%')
    print(f'💰 مجموع بازدهی خالص: {net_r:.2f}R')
    print(f'💵 مجموع PnL خالص: ${net_pnl:,.2f}')
    print(
        f'🏦 سرمایه نهایی Mark-to-Market: '
        f'${final_equity:,.2f}'
    )

    print('\n------------------------------------------------------------')
    print('📉 زنجیره‌های ضرر')
    print('------------------------------------------------------------')

    if loss_sequences:
        print(
            ', '.join(
                map(str, loss_sequences)
            )
        )
    else:
        print('هیچ زنجیره ضرری ثبت نشد.')

else:
    print('⚠️ معامله‌ای ثبت نشد.')

print('\n✨ بک‌تست V41 به پایان رسید.')
