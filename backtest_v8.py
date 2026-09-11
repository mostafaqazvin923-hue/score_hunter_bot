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
# چون بایننس محدوده IP سرورهای GitHub Actions را هم مسدود می‌کند (خطای 451)،
# به‌جای وابستگی به یک صرافی، لیستی از صرافی‌ها را به ترتیب امتحان می‌کنیم و
# اولین موردی که واقعاً دیتا برگرداند مبنای بک‌تست قرار می‌گیرد. اجرای زنده
# همچنان می‌تواند روی LBank باشد؛ این‌ها فقط برای گرفتن تاریخچه قیمت هستند.
CANDIDATE_DATA_EXCHANGES = ['kucoin', 'okx', 'bybit', 'gateio', 'mexc', 'bitget']
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

TIMEFRAME = '15m'
DAYS_BACK = 365
COMMISSION_RATE = 0.0008          # کارمزد کل رفت‌وبرگشت (تقریبی)
COMMISSION_PER_SIDE = COMMISSION_RATE / 2
SLIPPAGE_RATE = 0.04 / 100        # اسلیپیج تخمینی هر ضلع

# ---------------- مقیاس‌دهی پارامترهای مبتنی بر «بازه زمانی واقعی» ----------------
# چون از 1h به 15m رفتیم (۴ برابر کندل بیشتر در واحد زمان)، پارامترهایی که معنای
# "طول یک بازه تقویمی" دارند (نه اندیکاتورهای استاندارد مثل RSI/BB) را ۴ برابر
# می‌کنیم تا همان "پنجره زمانی" قبلی حفظ شود؛ در غیر این صورت فیلترها عملاً شل‌تر
# می‌شدند بدون اینکه واقعاً تصمیم گرفته باشیم شل‌ترشان کنیم.
TF_SCALE = 4

# ---------------- پارامترهای استراتژی ----------------
BB_PERIOD = 20             # استاندارد اندیکاتور، مستقل از تایم‌فریم
BB_STD = 2
RSI_PERIOD = 14            # استاندارد اندیکاتور، مستقل از تایم‌فریم
ATR_PERIOD = 14            # استاندارد اندیکاتور، مستقل از تایم‌فریم
ATR_MA_PERIOD = 50 * TF_SCALE
VOL_MA_PERIOD = 20 * TF_SCALE
VP_WINDOW = 100 * TF_SCALE        # طول پنجره Fixed Range Volume Profile (بازه تقویمی ثابت)
VP_BINS = 24                      # تعداد باکت‌های قیمتی پروفایل حجم
VALUE_AREA_PCT = 0.70             # درصد حجم برای تعیین Value Area (VAH/VAL)
SWING_LOOKBACK = 10 * TF_SCALE    # تعداد کندل برای سقف/کف ساختاری (بازه تقویمی ثابت)
ADX_PERIOD = 14            # استاندارد اندیکاتور، مستقل از تایم‌فریم
MAX_HOLD_CANDLES = 60 * TF_SCALE  # حداکثر مدت نگه‌داشتن معامله باز (بازه تقویمی ثابت)
COOLDOWN_CANDLES = 3 * TF_SCALE   # فاصله امنیتی بعد از بسته‌شدن معامله (جلوگیری از هم‌پوشانی)
MIN_WARMUP = max(250 * TF_SCALE, VP_WINDOW + ATR_MA_PERIOD + 5)


# ============================================================
# دریافت داده تاریخی — به ترتیب چند صرافی را امتحان می‌کند تا یکی جواب بدهد
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


# ============================================================
# اندیکاتورها: Bollinger Bands, RSI, ATR, ADX, Swing High/Low
# همه بر اساس کندل‌های کاملاً بسته‌شده — بدون لوک‌آهد
# ============================================================
def calculate_indicators(df):
    df = df.copy()

    # Bollinger Bands
    df['BB_MID'] = df['Close'].rolling(BB_PERIOD).mean()
    bb_std = df['Close'].rolling(BB_PERIOD).std()
    df['BB_UPPER'] = df['BB_MID'] + BB_STD * bb_std
    df['BB_LOWER'] = df['BB_MID'] - BB_STD * bb_std

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # True Range / ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['TR'] = tr
    df['ATR'] = tr.rolling(ATR_PERIOD).mean()
    df['ATR_MA'] = df['ATR'].rolling(ATR_MA_PERIOD).mean()

    # Volume MA
    df['Vol_MA'] = df['Volume'].rolling(VOL_MA_PERIOD).mean()

    # ADX (نسخه ساده‌شده، هم‌راستا با روش محاسبه ATR بالا)
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

    # سقف/کف ساختاری — بر اساس کندل‌های *قبل* از کندل جاری (shift(1)) تا هیچ نگاهی به آینده نباشد
    df['SWING_LOW'] = df['Low'].shift(1).rolling(SWING_LOOKBACK).min()
    df['SWING_HIGH'] = df['High'].shift(1).rolling(SWING_LOOKBACK).max()

    return df


# ============================================================
# Fixed Range Volume Profile — محاسبه غلتان (rolling)
# برای کندل i فقط از کندل‌های i-window ... i-1 استفاده می‌شود (کاملاً بسته‌شده)
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


def compute_r(entry_price, exit_price, sl_dist, is_long):
    entry_fee = entry_price * COMMISSION_PER_SIDE
    exit_fee = exit_price * COMMISSION_PER_SIDE
    gross_pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    net_pnl = gross_pnl - entry_fee - exit_fee
    return net_pnl / sl_dist


# ============================================================
# شبیه‌سازی معامله دو مرحله‌ای (TP1 پارشیال ۵۰٪ + بریک‌ایون + TP2)
# فقط کندل‌های بعد از ورود بررسی می‌شوند — بدون لوک‌آهد
# ============================================================
def simulate_long_trade(df, entry_idx, entry_price, sl_price, tp1_price, tp2_price, sl_dist):
    leg1_done = False
    r1 = 0.0
    current_sl = sl_price
    last_idx = min(entry_idx + MAX_HOLD_CANDLES, len(df)) - 1

    for j in range(entry_idx, last_idx + 1):
        f = df.iloc[j]
        if not leg1_done:
            if f['Low'] <= current_sl:
                exit_price = current_sl * (1 - SLIPPAGE_RATE)
                return compute_r(entry_price, exit_price, sl_dist, True), j
            if f['High'] >= tp1_price:
                exit1_price = tp1_price * (1 - SLIPPAGE_RATE)
                r1 = compute_r(entry_price, exit1_price, sl_dist, True)
                leg1_done = True
                current_sl = entry_price
                if f['High'] >= tp2_price:
                    exit2_price = tp2_price * (1 - SLIPPAGE_RATE)
                    r2 = compute_r(entry_price, exit2_price, sl_dist, True)
                    return 0.5 * r1 + 0.5 * r2, j
        else:
            if f['Low'] <= current_sl:
                exit2_price = current_sl * (1 - SLIPPAGE_RATE)
                r2 = compute_r(entry_price, exit2_price, sl_dist, True)
                return 0.5 * r1 + 0.5 * r2, j
            if f['High'] >= tp2_price:
                exit2_price = tp2_price * (1 - SLIPPAGE_RATE)
                r2 = compute_r(entry_price, exit2_price, sl_dist, True)
                return 0.5 * r1 + 0.5 * r2, j

    # پایان بازه بدون رسیدن به SL/TP2 → خروج زمانی با قیمت بسته‌شدن آخرین کندل
    exit_price = df.iloc[last_idx]['Close']
    if not leg1_done:
        return compute_r(entry_price, exit_price, sl_dist, True), last_idx
    r2 = compute_r(entry_price, exit_price, sl_dist, True)
    return 0.5 * r1 + 0.5 * r2, last_idx


def simulate_short_trade(df, entry_idx, entry_price, sl_price, tp1_price, tp2_price, sl_dist):
    leg1_done = False
    r1 = 0.0
    current_sl = sl_price
    last_idx = min(entry_idx + MAX_HOLD_CANDLES, len(df)) - 1

    for j in range(entry_idx, last_idx + 1):
        f = df.iloc[j]
        if not leg1_done:
            if f['High'] >= current_sl:
                exit_price = current_sl * (1 + SLIPPAGE_RATE)
                return compute_r(entry_price, exit_price, sl_dist, False), j
            if f['Low'] <= tp1_price:
                exit1_price = tp1_price * (1 + SLIPPAGE_RATE)
                r1 = compute_r(entry_price, exit1_price, sl_dist, False)
                leg1_done = True
                current_sl = entry_price
                if f['Low'] <= tp2_price:
                    exit2_price = tp2_price * (1 + SLIPPAGE_RATE)
                    r2 = compute_r(entry_price, exit2_price, sl_dist, False)
                    return 0.5 * r1 + 0.5 * r2, j
        else:
            if f['High'] >= current_sl:
                exit2_price = current_sl * (1 + SLIPPAGE_RATE)
                r2 = compute_r(entry_price, exit2_price, sl_dist, False)
                return 0.5 * r1 + 0.5 * r2, j
            if f['Low'] <= tp2_price:
                exit2_price = tp2_price * (1 + SLIPPAGE_RATE)
                r2 = compute_r(entry_price, exit2_price, sl_dist, False)
                return 0.5 * r1 + 0.5 * r2, j

    exit_price = df.iloc[last_idx]['Close']
    if not leg1_done:
        return compute_r(entry_price, exit_price, sl_dist, False), last_idx
    r2 = compute_r(entry_price, exit_price, sl_dist, False)
    return 0.5 * r1 + 0.5 * r2, last_idx


# ============================================================
# موتور اصلی سیگنال‌یابی و بک‌تست برای هر نماد
# ============================================================
def run_backtest_for_symbol(symbol, df1h):
    if len(df1h) < MIN_WARMUP + MAX_HOLD_CANDLES + 5:
        return []

    df = calculate_indicators(df1h)
    df = calculate_volume_profile(df)

    trades = []
    locked_until_index = 0
    n = len(df)
    needed_cols = ['BB_UPPER', 'BB_LOWER', 'BB_MID', 'RSI', 'ATR', 'ATR_MA',
                   'ADX', 'POC', 'VAH', 'VAL', 'SWING_LOW', 'SWING_HIGH', 'Vol_MA']

    for i in range(MIN_WARMUP, n - MAX_HOLD_CANDLES - 2):
        if i < locked_until_index:
            continue

        c = df.iloc[i]
        prev = df.iloc[i - 1]

        if c[needed_cols].isna().any() or pd.isna(prev['RSI']):
            continue
        if c['ATR_MA'] == 0 or c['ATR'] <= 0:
            continue

        # ---- فیلتر رژیم بازار (جلوگیری از ترید در بازار رنج/پرخطر) ----
        atr_ratio = c['ATR'] / c['ATR_MA']
        vol_ok = c['Volume'] >= 0.8 * c['Vol_MA']
        regime_ok = (0.7 <= atr_ratio <= 2.5) and (c['ADX'] >= 15) and vol_ok
        if not regime_ok:
            continue

        adx_not_extreme = c['ADX'] <= 45

        # ---- شرایط ورود لانگ ----
        long_signal = (
            c['Low'] <= c['BB_LOWER']
            and c['Close'] > c['BB_LOWER']
            and c['Close'] <= max(c['POC'], c['VAL'] * 1.01)
            and 25 <= c['RSI'] <= 40
            and c['RSI'] > prev['RSI']
            and c['Close'] > c['Open']
            and adx_not_extreme
        )

        # ---- شرایط ورود شورت ----
        short_signal = (
            c['High'] >= c['BB_UPPER']
            and c['Close'] < c['BB_UPPER']
            and c['Close'] >= min(c['POC'], c['VAH'] * 0.99)
            and 60 <= c['RSI'] <= 75
            and c['RSI'] < prev['RSI']
            and c['Close'] < c['Open']
            and adx_not_extreme
        )

        if not long_signal and not short_signal:
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_candle = df.iloc[entry_idx]

        if long_signal:
            entry_price = entry_candle['Open'] * (1 + SLIPPAGE_RATE)

            atr_sl_dist = 1.5 * c['ATR']
            struct_sl_price = c['SWING_LOW'] - 0.2 * c['ATR']
            struct_sl_dist = entry_price - struct_sl_price
            sl_dist = struct_sl_dist if 0 < struct_sl_dist < atr_sl_dist else atr_sl_dist
            if sl_dist <= 0:
                continue
            sl_price = entry_price - sl_dist

            tp1_price = c['POC'] if c['POC'] > entry_price else c['VAH']
            tp2_r_price = entry_price + 2 * sl_dist
            tp2_price = min(tp2_r_price, c['SWING_HIGH']) if c['SWING_HIGH'] > entry_price else tp2_r_price

            if not (sl_price < entry_price < tp1_price < tp2_price):
                continue

            net_r, exit_idx = simulate_long_trade(df, entry_idx, entry_price, sl_price, tp1_price, tp2_price, sl_dist)
            trades.append({
                'Symbol': symbol, 'Side': 'LONG',
                'Outcome': 'WIN' if net_r > 0 else 'LOSS',
                'NetR': net_r, 'Date': entry_candle['Date']
            })
            locked_until_index = exit_idx + COOLDOWN_CANDLES

        elif short_signal:
            entry_price = entry_candle['Open'] * (1 - SLIPPAGE_RATE)

            atr_sl_dist = 1.5 * c['ATR']
            struct_sl_price = c['SWING_HIGH'] + 0.2 * c['ATR']
            struct_sl_dist = struct_sl_price - entry_price
            sl_dist = struct_sl_dist if 0 < struct_sl_dist < atr_sl_dist else atr_sl_dist
            if sl_dist <= 0:
                continue
            sl_price = entry_price + sl_dist

            tp1_price = c['POC'] if c['POC'] < entry_price else c['VAL']
            tp2_r_price = entry_price - 2 * sl_dist
            tp2_price = max(tp2_r_price, c['SWING_LOW']) if c['SWING_LOW'] < entry_price else tp2_r_price

            if not (tp2_price < tp1_price < entry_price < sl_price):
                continue

            net_r, exit_idx = simulate_short_trade(df, entry_idx, entry_price, sl_price, tp1_price, tp2_price, sl_dist)
            trades.append({
                'Symbol': symbol, 'Side': 'SHORT',
                'Outcome': 'WIN' if net_r > 0 else 'LOSS',
                'NetR': net_r, 'Date': entry_candle['Date']
            })
            locked_until_index = exit_idx + COOLDOWN_CANDLES

    return trades


# ============================================================
# اجرای کامل: دانلود داده + بک‌تست + گزارش
# ============================================================
def main():
    start_date = datetime.now() - timedelta(days=DAYS_BACK)
    since_timestamp = int(start_date.timestamp() * 1000)

    print("============================================================")
    print("📥 دانلود داده‌های یک‌ساله (تایم‌فریم 15 دقیقه)")
    print(f"   ترتیب امتحان صرافی‌ها: {', '.join(CANDIDATE_DATA_EXCHANGES)}")
    print("   (فقط برای منبع دیتا؛ اجرای زنده می‌تواند روی LBank باشد)")
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
        df.to_csv(f"{symbol}_1h_frvp_bb_rsi_atr.csv", index=False)
        data_1h[symbol] = df
        print(f"  ✔️ {symbol}: {len(df)} کندل دریافت شد.")

    print("\n============================================================")
    print("🚀 اجرای بک‌تست ستاپ FRVP + Bollinger + RSI + ATR")
    print("============================================================")

    all_trades = []
    for symbol, df1h in data_1h.items():
        print(f"⏳ در حال پردازش {symbol} ...")
        trades = run_backtest_for_symbol(symbol, df1h)
        all_trades.extend(trades)
        print(f"  ✔️ {symbol}: {len(trades)} معامله ثبت شد.")

    print_report(all_trades)


def print_report(all_trades):
    print("\n============================================================")
    print("📊 گزارش نهایی عملکرد یک‌ساله")
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

    print(f"🎯 تعداد کل معاملات (یک سال / ۱۰ ارز): {total_trades}")
    print(f"📈 لانگ: {longs}  |  📉 شورت: {shorts}")
    print(f"🏆 برد: {wins}  |  ❌ باخت: {losses}")
    print(f"✅ وین‌ریت کل: {win_rate:.2f}%")
    print(f"💰 مجموع Net R: {total_net_r:.2f}R  |  میانگین هر معامله: {avg_r:.3f}R")
    print(f"📆 میانگین تعداد معامله در ماه: {total_trades / 12:.1f}")

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

    tdf.to_csv("all_trades_frvp_bb_rsi_atr.csv", index=False)
    print("\n📁 جزئیات تمام معاملات در فایل all_trades_frvp_bb_rsi_atr.csv ذخیره شد.")


if __name__ == "__main__":
    main()
