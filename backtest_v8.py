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
# منبع دیتای تاریخی برای بک‌تست
# ============================================================
CANDIDATE_DATA_EXCHANGES = ['lbank', 'kucoin', 'okx', 'bybit', 'gateio', 'mexc', 'bitget']
_exchange_instances = {}


def get_data_exchange(exchange_id):
    if exchange_id not in _exchange_instances:
        exchange_class = getattr(ccxt, exchange_id)
        _exchange_instances[exchange_id] = exchange_class({'enableRateLimit': True})
    return _exchange_instances[exchange_id]


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

TIMEFRAME = '1h'
DAYS_BACK = 365
COMMISSION_RATE = 0.0008
COMMISSION_PER_SIDE = COMMISSION_RATE / 2
SLIPPAGE_RATE = 0.04 / 100

# ---------------- پارامترهای اندیکاتور ----------------
BB_PERIOD = 20
BB_STD_MULT = 2.0
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MA_PERIOD = 50
VOL_MA_PERIOD = 20
VP_WINDOW = 100
VP_BINS = 24
VALUE_AREA_PCT = 0.70
SWING_LOOKBACK = 10
ADX_PERIOD = 14
MAX_HOLD_CANDLES = 60
COOLDOWN_CANDLES = 3
MIN_WARMUP = max(250, VP_WINDOW + ATR_MA_PERIOD + 5)

# ---------------- آستانه‌ی z-score برای ورود (مطابق تحقیق: نسخه‌ی سخت‌گیرانه‌تر) ----------------
Z_ENTRY_THRESHOLD = 2.2   # کلاسیک بولینگر 2.0 است؛ کمی سخت‌گیرانه‌تر برای سیگنال‌های باکیفیت‌تر

# ---------------- فیلتر روند بالادستی (4 ساعته) ----------------
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 200

# ---------------- فیلتر فاندینگ ریت ----------------
CANDIDATE_FUNDING_EXCHANGES = ['lbank', 'bybit', 'okx', 'gateio', 'mexc', 'bitget', 'kucoin']
FUNDING_Z_WINDOW = 30
FUNDING_Z_LONG_MAX = 0.5
FUNDING_Z_SHORT_MIN = -0.5
_funding_instances = {}


def get_funding_exchange(exchange_id):
    if exchange_id not in _funding_instances:
        exchange_class = getattr(ccxt, exchange_id)
        _funding_instances[exchange_id] = exchange_class({'enableRateLimit': True})
    return _funding_instances[exchange_id]


# ---------------- پارامترهای مدیریت ریسک در سطح پرتفوی (جدید) ----------------
STARTING_EQUITY = 100.0            # واحد درصدی سرمایه (نه دلار مشخص)
RISK_PER_TRADE_PCT = 0.01          # ۱٪ از سرمایه‌ی فعلی (نه ثابت) در هر معامله — امکان کامپوند شدن
MAX_CONCURRENT_POSITIONS = 4       # حداکثر تعداد پوزیشن باز هم‌زمان در کل پرتفوی
MAX_SAME_DIRECTION_POSITIONS = 3   # حداکثر پوزیشن هم‌جهت هم‌زمان (کنترل هم‌بستگی بین آلت‌کوین‌ها)
DAILY_LOSS_LIMIT_PCT = 0.03        # اگر ضرر یک روز به ۳٪ سرمایه‌ی ابتدای همان روز رسید، ورود جدید متوقف می‌شود


# ============================================================
# دریافت داده تاریخی — به ترتیب چند صرافی را امتحان می‌کند
# ============================================================
def fetch_ohlcv_full(symbol_ccxt, timeframe, since_ts):
    for exchange_id in CANDIDATE_DATA_EXCHANGES:
        try:
            ex = get_data_exchange(exchange_id)
            all_ohlcv = []
            current_since = since_ts
            now_ts = ex.milliseconds()

            while current_since < now_ts:
                ohlcv = ex.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, since=current_since, limit=1000)
                if not ohlcv:
                    break
                current_since = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)
                if len(ohlcv) < 1000:
                    break

            if len(all_ohlcv) > 200:
                print(f"    ✔️ دیتا از {exchange_id} دریافت شد ({len(all_ohlcv)} کندل).")
                return all_ohlcv
            else:
                print(f"    ⚠️ {exchange_id}: دیتای کافی برنگشت، رفتن به صرافی بعدی...")
        except Exception as e:
            print(f"    ⚠️ {exchange_id} ناموفق ({type(e).__name__}): {e}")
            continue
    return []


def fetch_funding_rate_full(base_symbol, since_ts):
    symbol_variants = [base_symbol, f"{base_symbol}:USDT"]
    for exchange_id in CANDIDATE_FUNDING_EXCHANGES:
        try:
            ex = get_funding_exchange(exchange_id)
            if not ex.has.get('fetchFundingRateHistory'):
                continue
            for sym_variant in symbol_variants:
                try:
                    all_funding = []
                    current_since = since_ts
                    now_ts = ex.milliseconds()
                    safety_counter = 0
                    while current_since < now_ts and safety_counter < 200:
                        batch = ex.fetch_funding_rate_history(sym_variant, since=current_since, limit=200)
                        if not batch:
                            break
                        current_since = batch[-1]['timestamp'] + 1
                        all_funding.extend(batch)
                        safety_counter += 1
                        if len(batch) < 200:
                            break
                    if len(all_funding) > 20:
                        print(f"    ✔️ فاندینگ ریت از {exchange_id} ({sym_variant}) دریافت شد ({len(all_funding)} رکورد).")
                        return all_funding
                except Exception:
                    continue
        except Exception:
            continue
    return []


# ============================================================
# اندیکاتورها — همه بر اساس کندل‌های کاملاً بسته‌شده، بدون لوک‌آهد
# ============================================================
def calculate_indicators(df):
    df = df.copy()

    df['BB_MID'] = df['Close'].rolling(BB_PERIOD).mean()
    df['BB_STD'] = df['Close'].rolling(BB_PERIOD).std()
    df['BB_UPPER'] = df['BB_MID'] + BB_STD_MULT * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - BB_STD_MULT * df['BB_STD']
    # z-score واقعی انحراف قیمت از میانگین — معیار اصلی ورود (به‌جای فقط لمس باند)
    df['Z_SCORE'] = (df['Close'] - df['BB_MID']) / df['BB_STD'].replace(0, np.nan)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(ATR_PERIOD).mean()
    df['ATR_MA'] = df['ATR'].rolling(ATR_MA_PERIOD).mean()
    df['Vol_MA'] = df['Volume'].rolling(VOL_MA_PERIOD).mean()

    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    atr_for_adx = tr.rolling(ADX_PERIOD).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(ADX_PERIOD).mean() / atr_for_adx)
    minus_di = 100 * (minus_dm.rolling(ADX_PERIOD).mean() / atr_for_adx)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX'] = dx.rolling(ADX_PERIOD).mean()

    df['SWING_LOW'] = df['Low'].shift(1).rolling(SWING_LOOKBACK).min()
    df['SWING_HIGH'] = df['High'].shift(1).rolling(SWING_LOOKBACK).max()

    return df


# ============================================================
# Fixed Range Volume Profile — رولینگ، بدون لوک‌آهد
# ============================================================
def calculate_volume_profile(df, window=VP_WINDOW, bins=VP_BINS, va_pct=VALUE_AREA_PCT):
    n = len(df)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    vols = df['Volume'].values

    for i in range(window, n):
        w_high = highs[i - window:i]
        w_low = lows[i - window:i]
        w_close = closes[i - window:i]
        w_vol = vols[i - window:i]

        price_max = w_high.max()
        price_min = w_low.min()
        if price_max <= price_min:
            continue

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_vol = np.zeros(bins)
        bin_idx = np.clip(np.digitize(w_close, bin_edges) - 1, 0, bins - 1)
        np.add.at(bin_vol, bin_idx, w_vol)

        total_vol = bin_vol.sum()
        if total_vol <= 0:
            continue

        poc_bin = int(np.argmax(bin_vol))
        poc[i] = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2

        target_vol = total_vol * va_pct
        acc_vol = bin_vol[poc_bin]
        lo_bin = hi_bin = poc_bin
        while acc_vol < target_vol and (lo_bin > 0 or hi_bin < bins - 1):
            expand_lo = bin_vol[lo_bin - 1] if lo_bin > 0 else -1
            expand_hi = bin_vol[hi_bin + 1] if hi_bin < bins - 1 else -1
            if expand_hi >= expand_lo:
                hi_bin += 1
                acc_vol += bin_vol[hi_bin]
            else:
                lo_bin -= 1
                acc_vol += bin_vol[lo_bin]

        val[i] = bin_edges[lo_bin]
        vah[i] = bin_edges[hi_bin + 1]

    df = df.copy()
    df['POC'] = poc
    df['VAH'] = vah
    df['VAL'] = val
    return df


def calculate_trend_filter(df1h):
    df1h = df1h.copy().sort_values('Date').reset_index(drop=True)

    df4h = (
        df1h.set_index('Date')
        .resample('4h', label='left', closed='left')
        .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        .dropna()
        .reset_index()
    )
    df4h['EMA_FAST'] = df4h['Close'].ewm(span=TREND_EMA_FAST, adjust=False).mean()
    df4h['EMA_SLOW'] = df4h['Close'].ewm(span=TREND_EMA_SLOW, adjust=False).mean()
    df4h['EMA_SLOW_PREV'] = df4h['EMA_SLOW'].shift(1)
    df4h['CLOSE_4H'] = df4h['Close']
    df4h['AVAILABLE_AT'] = df4h['Date'] + pd.Timedelta(hours=4)

    merge_cols = df4h[['AVAILABLE_AT', 'CLOSE_4H', 'EMA_FAST', 'EMA_SLOW', 'EMA_SLOW_PREV']].sort_values('AVAILABLE_AT')

    merged = pd.merge_asof(
        df1h, merge_cols,
        left_on='Date', right_on='AVAILABLE_AT',
        direction='backward'
    )

    is_long_regime = (
        (merged['CLOSE_4H'] > merged['EMA_SLOW'])
        & (merged['EMA_FAST'] > merged['EMA_SLOW'])
        & (merged['EMA_SLOW'] >= merged['EMA_SLOW_PREV'] * 0.9995)
    )
    is_short_regime = (
        (merged['CLOSE_4H'] < merged['EMA_SLOW'])
        & (merged['EMA_FAST'] < merged['EMA_SLOW'])
        & (merged['EMA_SLOW'] <= merged['EMA_SLOW_PREV'] * 1.0005)
    )

    merged['TREND_LONG_OK'] = is_long_regime.fillna(False)
    merged['TREND_SHORT_OK'] = is_short_regime.fillna(False)
    return merged


def calculate_funding_filter(df1h, funding_records):
    df = df1h.copy()

    if not funding_records:
        df['FUNDING_RATE'] = np.nan
        df['FUNDING_Z'] = np.nan
        return df

    fdf = pd.DataFrame(funding_records)
    if 'timestamp' not in fdf.columns or 'fundingRate' not in fdf.columns:
        df['FUNDING_RATE'] = np.nan
        df['FUNDING_Z'] = np.nan
        return df

    fdf['Date'] = pd.to_datetime(fdf['timestamp'], unit='ms').astype('datetime64[ns]')
    fdf = fdf[['Date', 'fundingRate']].dropna().sort_values('Date').reset_index(drop=True)
    fdf.rename(columns={'fundingRate': 'FUNDING_RATE'}, inplace=True)

    fdf['FUNDING_MEAN'] = fdf['FUNDING_RATE'].rolling(FUNDING_Z_WINDOW, min_periods=5).mean()
    fdf['FUNDING_STD'] = fdf['FUNDING_RATE'].rolling(FUNDING_Z_WINDOW, min_periods=5).std()
    fdf['FUNDING_Z'] = (fdf['FUNDING_RATE'] - fdf['FUNDING_MEAN']) / fdf['FUNDING_STD'].replace(0, np.nan)

    df = df.sort_values('Date')
    df['Date'] = df['Date'].astype('datetime64[ns]')
    merged = pd.merge_asof(
        df, fdf[['Date', 'FUNDING_RATE', 'FUNDING_Z']],
        on='Date', direction='backward'
    )
    return merged


def compute_r(entry_price, exit_price, sl_dist, is_long):
    entry_fee = entry_price * COMMISSION_PER_SIDE
    exit_fee = exit_price * COMMISSION_PER_SIDE
    gross_pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    net_pnl = gross_pnl - entry_fee - exit_fee
    return net_pnl / sl_dist


# ============================================================
# فاز ۱: پیش‌محاسبه‌ی سیگنال‌ها برای هر نماد (بدون شبیه‌سازی معامله)
# خروجی این تابع فقط "کجا سیگنال هست" را مشخص می‌کند؛ تصمیم واقعی برای باز
# کردن پوزیشن (با توجه به محدودیت‌های کل پرتفوی) در فاز ۲ گرفته می‌شود.
# ============================================================
def precompute_signals_for_symbol(symbol, df1h, funding_records=None):
    if len(df1h) < MIN_WARMUP + MAX_HOLD_CANDLES + 5:
        return None

    df = calculate_indicators(df1h)
    df = calculate_volume_profile(df)
    df = calculate_trend_filter(df)
    df = calculate_funding_filter(df, funding_records or [])
    funding_available = df['FUNDING_Z'].notna().any()

    n = len(df)
    long_signal = np.zeros(n, dtype=bool)
    short_signal = np.zeros(n, dtype=bool)
    sl_dist_arr = np.full(n, np.nan)

    needed_cols = ['BB_MID', 'BB_STD', 'Z_SCORE', 'RSI', 'ATR', 'ATR_MA',
                   'ADX', 'POC', 'VAH', 'VAL', 'SWING_LOW', 'SWING_HIGH', 'Vol_MA']

    for i in range(MIN_WARMUP, n - 2):
        c = df.iloc[i]
        prev = df.iloc[i - 1]

        if c[needed_cols].isna().any() or pd.isna(prev['RSI']):
            continue
        if c['ATR_MA'] == 0 or c['ATR'] <= 0:
            continue

        atr_ratio = c['ATR'] / c['ATR_MA']
        vol_ok = c['Volume'] >= 0.75 * c['Vol_MA']
        volatility_ok = (0.65 <= atr_ratio <= 2.7) and (c['ADX'] >= 12)
        adx_not_extreme = c['ADX'] <= 45

        if funding_available and not pd.isna(c['FUNDING_Z']):
            funding_ok_long = c['FUNDING_Z'] <= FUNDING_Z_LONG_MAX
            funding_ok_short = c['FUNDING_Z'] >= FUNDING_Z_SHORT_MIN
        else:
            funding_ok_long = True
            funding_ok_short = True

        # ---- ورود لانگ: z-score واقعی به‌جای فقط لمس باند ----
        is_long = (
            volatility_ok and vol_ok and adx_not_extreme and funding_ok_long
            and bool(c['TREND_LONG_OK'])
            and c['Z_SCORE'] <= -Z_ENTRY_THRESHOLD
            and c['Close'] <= max(c['POC'], c['VAL'] * 1.02)
            and 28 <= c['RSI'] <= 45
            and c['RSI'] > prev['RSI']
            and c['Close'] > c['Open']
        )

        is_short = (
            volatility_ok and vol_ok and adx_not_extreme and funding_ok_short
            and bool(c['TREND_SHORT_OK'])
            and c['Z_SCORE'] >= Z_ENTRY_THRESHOLD
            and c['Close'] >= min(c['POC'], c['VAH'] * 0.98)
            and 55 <= c['RSI'] <= 72
            and c['RSI'] < prev['RSI']
            and c['Close'] < c['Open']
        )

        if is_long:
            atr_sl_dist = 1.5 * c['ATR']
            struct_sl_dist = c['Close'] - (c['SWING_LOW'] - 0.2 * c['ATR'])
            sl_dist = struct_sl_dist if 0 < struct_sl_dist < atr_sl_dist else atr_sl_dist
            if sl_dist > 0:
                long_signal[i] = True
                sl_dist_arr[i] = sl_dist
        elif is_short:
            atr_sl_dist = 1.5 * c['ATR']
            struct_sl_dist = (c['SWING_HIGH'] + 0.2 * c['ATR']) - c['Close']
            sl_dist = struct_sl_dist if 0 < struct_sl_dist < atr_sl_dist else atr_sl_dist
            if sl_dist > 0:
                short_signal[i] = True
                sl_dist_arr[i] = sl_dist

    out = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
    out['LONG_SIGNAL'] = long_signal
    out['SHORT_SIGNAL'] = short_signal
    out['SL_DIST'] = sl_dist_arr
    return out


# ============================================================
# فاز ۲: موتور شبیه‌سازی پرتفوی — یک حساب مشترک برای همه‌ی نمادها
# با مدیریت ریسک واقعی: سایزینگ پویا بر مبنای سرمایه، محدودیت هم‌بستگی
# (حداکثر پوزیشن هم‌جهت هم‌زمان)، محدودیت کل پوزیشن باز، و سقف ضرر روزانه.
# پردازش زمانی-ترتیبی روی خط‌زمان مشترک همه‌ی نمادها — بدون لوک‌آهد.
# ============================================================
def run_portfolio_backtest(precomputed):
    symbols = list(precomputed.keys())
    if not symbols:
        return [], []

    all_dates = sorted(set().union(*[set(df['Date']) for df in precomputed.values()]))
    n_master = len(all_dates)

    arrays = {}
    for symbol, df in precomputed.items():
        aligned = df.set_index('Date').reindex(all_dates)
        arrays[symbol] = {
            'Open': aligned['Open'].values,
            'High': aligned['High'].values,
            'Low': aligned['Low'].values,
            'Close': aligned['Close'].values,
            'LONG_SIGNAL': aligned['LONG_SIGNAL'].fillna(False).values,
            'SHORT_SIGNAL': aligned['SHORT_SIGNAL'].fillna(False).values,
            'SL_DIST': aligned['SL_DIST'].values,
        }

    equity = STARTING_EQUITY
    equity_curve = []
    open_positions = {}     # symbol -> position dict
    recently_closed = {}    # symbol -> idx بسته‌شدن (برای cooldown)
    trades = []

    current_day = None
    day_start_equity = equity
    daily_loss = 0.0

    for idx in range(n_master):
        date_t = all_dates[idx]
        day = date_t.date() if hasattr(date_t, 'date') else pd.Timestamp(date_t).date()
        if day != current_day:
            current_day = day
            day_start_equity = equity
            daily_loss = 0.0

        # ---- ۱) بررسی خروج پوزیشن‌های باز ----
        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            arr = arrays[symbol]
            if idx >= len(arr['High']) or np.isnan(arr['High'][idx]):
                continue  # کندل این نماد در این لحظه موجود نیست؛ پوزیشن باز می‌ماند

            high = arr['High'][idx]
            low = arr['Low'][idx]
            close_p = arr['Close'][idx]
            exit_price = None

            if pos['side'] == 'LONG':
                if low <= pos['sl_price']:
                    exit_price = pos['sl_price'] * (1 - SLIPPAGE_RATE)
                elif high >= pos['tp_price']:
                    exit_price = pos['tp_price'] * (1 - SLIPPAGE_RATE)
            else:
                if high >= pos['sl_price']:
                    exit_price = pos['sl_price'] * (1 + SLIPPAGE_RATE)
                elif low <= pos['tp_price']:
                    exit_price = pos['tp_price'] * (1 + SLIPPAGE_RATE)

            if exit_price is None and idx >= pos['max_hold_until_idx']:
                exit_price = close_p  # خروج زمانی

            if exit_price is not None:
                net_r = compute_r(pos['entry_price'], exit_price, pos['sl_dist'], pos['side'] == 'LONG')
                pnl_equity = pos['risk_amount'] * net_r
                equity += pnl_equity
                if pnl_equity < 0:
                    daily_loss += -pnl_equity
                trades.append({
                    'Symbol': symbol, 'Side': pos['side'],
                    'Outcome': 'WIN' if net_r > 0 else 'LOSS',
                    'NetR': net_r, 'PnL_Equity': pnl_equity,
                    'Date': pos['entry_date'], 'ExitDate': date_t,
                    'EquityAfter': equity
                })
                del open_positions[symbol]
                recently_closed[symbol] = idx

        equity_curve.append((date_t, equity))

        # ---- ۲) سقف ضرر روزانه ----
        if daily_loss >= DAILY_LOSS_LIMIT_PCT * day_start_equity:
            continue  # امروز دیگر پوزیشن جدید باز نمی‌شود

        # ---- ۳) بررسی سیگنال‌های ورود جدید (با رعایت محدودیت‌های پرتفوی) ----
        current_long_count = sum(1 for p in open_positions.values() if p['side'] == 'LONG')
        current_short_count = sum(1 for p in open_positions.values() if p['side'] == 'SHORT')

        for symbol in symbols:
            if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
                break

            if symbol in open_positions:
                continue
            if symbol in recently_closed and (idx - recently_closed[symbol]) < COOLDOWN_CANDLES:
                continue

            arr = arrays[symbol]
            if idx >= len(arr['Close']) or np.isnan(arr['Close'][idx]):
                continue

            entry_idx = idx + 1
            if entry_idx >= len(arr['Open']) or np.isnan(arr['Open'][entry_idx]):
                continue

            sl_dist_raw = arr['SL_DIST'][idx]
            if np.isnan(sl_dist_raw) or sl_dist_raw <= 0:
                continue

            if arr['LONG_SIGNAL'][idx] and current_long_count < MAX_SAME_DIRECTION_POSITIONS:
                entry_price = arr['Open'][entry_idx] * (1 + SLIPPAGE_RATE)
                sl_price = entry_price - sl_dist_raw
                tp_price = entry_price + 2 * sl_dist_raw
                risk_amount = equity * RISK_PER_TRADE_PCT
                open_positions[symbol] = {
                    'side': 'LONG', 'entry_price': entry_price, 'sl_price': sl_price,
                    'tp_price': tp_price, 'sl_dist': sl_dist_raw, 'risk_amount': risk_amount,
                    'entry_date': all_dates[entry_idx] if entry_idx < n_master else date_t,
                    'max_hold_until_idx': idx + 1 + MAX_HOLD_CANDLES
                }
                current_long_count += 1

            elif arr['SHORT_SIGNAL'][idx] and current_short_count < MAX_SAME_DIRECTION_POSITIONS:
                entry_price = arr['Open'][entry_idx] * (1 - SLIPPAGE_RATE)
                sl_price = entry_price + sl_dist_raw
                tp_price = entry_price - 2 * sl_dist_raw
                risk_amount = equity * RISK_PER_TRADE_PCT
                open_positions[symbol] = {
                    'side': 'SHORT', 'entry_price': entry_price, 'sl_price': sl_price,
                    'tp_price': tp_price, 'sl_dist': sl_dist_raw, 'risk_amount': risk_amount,
                    'entry_date': all_dates[entry_idx] if entry_idx < n_master else date_t,
                    'max_hold_until_idx': idx + 1 + MAX_HOLD_CANDLES
                }
                current_short_count += 1

    return trades, equity_curve


# ============================================================
# اجرای کامل: دانلود داده + پیش‌محاسبه‌ی سیگنال + شبیه‌سازی پرتفوی + گزارش
# ============================================================
def main():
    start_date = datetime.now() - timedelta(days=DAYS_BACK)
    since_timestamp = int(start_date.timestamp() * 1000)

    print("============================================================")
    print("📥 دانلود داده‌های یک‌ساله (تایم‌فریم 1 ساعته)")
    print(f"   ترتیب امتحان صرافی‌ها: {', '.join(CANDIDATE_DATA_EXCHANGES)}")
    print("============================================================")

    data_1h = {}
    for symbol, lbank_symbol in SYMBOLS.items():
        print(f"🔹 در حال دریافت دیتای 1 ساعته {symbol}...")
        raw = fetch_ohlcv_full(lbank_symbol, TIMEFRAME, since_timestamp)
        if not raw:
            print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")
            continue
        df = pd.DataFrame(raw, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        df.drop_duplicates(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(f"{symbol}_1h_portfolio_v2.csv", index=False)
        data_1h[symbol] = df
        print(f"  ✔️ {symbol}: {len(df)} کندل دریافت شد.")

    print("\n============================================================")
    print("📥 دریافت تاریخچه فاندینگ ریت")
    print(f"   ترتیب امتحان صرافی‌ها: {', '.join(CANDIDATE_FUNDING_EXCHANGES)}")
    print("============================================================")

    funding_data = {}
    for symbol, ccxt_symbol in SYMBOLS.items():
        if symbol not in data_1h:
            continue
        print(f"🔹 در حال دریافت فاندینگ ریت {symbol}...")
        records = fetch_funding_rate_full(ccxt_symbol, since_timestamp)
        if not records:
            print(f"  ⚠️ فاندینگ ریتی برای {symbol} پیدا نشد — فیلتر فاندینگ برای این نماد نادیده گرفته می‌شود.")
        funding_data[symbol] = records

    print("\n============================================================")
    print("🧮 پیش‌محاسبه‌ی سیگنال‌ها (z-score + BB + RSI + VP + روند + فاندینگ)")
    print("============================================================")

    precomputed = {}
    for symbol, df1h in data_1h.items():
        sig_df = precompute_signals_for_symbol(symbol, df1h, funding_data.get(symbol))
        if sig_df is not None:
            precomputed[symbol] = sig_df
            n_long = int(sig_df['LONG_SIGNAL'].sum())
            n_short = int(sig_df['SHORT_SIGNAL'].sum())
            print(f"  ✔️ {symbol}: {n_long} سیگنال لانگ, {n_short} سیگنال شورت (خام، قبل از اعمال محدودیت پرتفوی)")

    print("\n============================================================")
    print("🚀 اجرای شبیه‌سازی پرتفوی مشترک (با مدیریت ریسک و هم‌بستگی)")
    print(f"   ریسک هر معامله: {RISK_PER_TRADE_PCT*100:.1f}% از سرمایه | "
          f"حداکثر پوزیشن هم‌زمان: {MAX_CONCURRENT_POSITIONS} | "
          f"حداکثر هم‌جهت: {MAX_SAME_DIRECTION_POSITIONS} | "
          f"سقف ضرر روزانه: {DAILY_LOSS_LIMIT_PCT*100:.1f}%")
    print("============================================================")

    trades, equity_curve = run_portfolio_backtest(precomputed)
    print_report(trades, equity_curve)


def print_report(all_trades, equity_curve):
    print("\n============================================================")
    print("📊 گزارش نهایی عملکرد یک‌ساله (سطح پرتفوی)")
    print("============================================================")

    if not all_trades:
        print("⚠️ هیچ معامله‌ای در این دوره ثبت نشد. پارامترهای فیلتر را بازبینی کنید.")
        return

    tdf = pd.DataFrame(all_trades)
    total_trades = len(tdf)
    longs = int((tdf['Side'] == 'LONG').sum())
    shorts = int((tdf['Side'] == 'SHORT').sum())
    wins = int((tdf['Outcome'] == 'WIN').sum())
    losses = int((tdf['Outcome'] == 'LOSS').sum())
    win_rate = wins / total_trades * 100
    total_net_r = tdf['NetR'].sum()
    avg_r = tdf['NetR'].mean()

    final_equity = equity_curve[-1][1] if equity_curve else STARTING_EQUITY
    total_return_pct = (final_equity / STARTING_EQUITY - 1) * 100

    equity_series = pd.Series([e for _, e in equity_curve])
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    print(f"🎯 تعداد کل معاملات (یک سال / ۱۰ ارز، سطح پرتفوی): {total_trades}")
    print(f"📈 لانگ: {longs}  |  📉 شورت: {shorts}")
    print(f"🏆 برد: {wins}  |  ❌ باخت: {losses}")
    print(f"✅ وین‌ریت کل: {win_rate:.2f}%")
    print(f"💰 مجموع Net R: {total_net_r:.2f}R  |  میانگین هر معامله: {avg_r:.3f}R")
    print(f"📆 میانگین تعداد معامله در ماه: {total_trades / 12:.1f}")
    print(f"💵 رشد سرمایه (کامپوند، با ریسک {RISK_PER_TRADE_PCT*100:.0f}٪ هر معامله): {total_return_pct:+.2f}%")
    print(f"📉 بیشینه افت سرمایه (Max Drawdown): {max_drawdown:.2f}%")

    print("\n--- تفکیک بر اساس نماد ---")
    g = tdf.groupby('Symbol').agg(
        Trades=('Outcome', 'count'),
        Long=('Side', lambda x: (x == 'LONG').sum()),
        Short=('Side', lambda x: (x == 'SHORT').sum()),
        Wins=('Outcome', lambda x: (x == 'WIN').sum()),
        Losses=('Outcome', lambda x: (x == 'LOSS').sum()),
        NetR=('NetR', 'sum')
    )
    g['WinRate%'] = (g['Wins'] / g['Trades'] * 100).round(2)
    print(g)

    tdf.to_csv("all_trades_portfolio_v2.csv", index=False)
    print("\n📁 جزئیات تمام معاملات در فایل all_trades_portfolio_v2.csv ذخیره شد.")


if __name__ == "__main__":
    main()
