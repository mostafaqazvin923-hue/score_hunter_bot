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

# صرافی LBank با سبد 10 ارز
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

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌های 1 ساعته برای استراتژی شکست محدوده (Breakout)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    print(f"🔹 در حال دریافت دیتای {symbol}...")
    all_ohlcv = []
    current_since = since_timestamp
    now_timestamp = exchange.milliseconds()
    
    while current_since < now_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(lbank_symbol, timeframe='1h', since=current_since, limit=1000)
            if not ohlcv:
                break
            current_since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            break
            
    if all_ohlcv:
        df1h = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df1h['Date'] = pd.to_datetime(df1h['Timestamp'], unit='ms')
        df1h.set_index('Date', inplace=True)
        data_1h[symbol] = df1h
        print(f"  ✔️ دیتای {symbol} آماده شد (تعداد کندل: {len(df1h)})")

def calculate_breakout_indicators(df):
    # کانال Donchian برای تشخیص سقف و کف 24 کندل گذشته (24 ساعت اخیر)
    df['Highest_24'] = df['High'].shift(1).rolling(window=24).max()
    df['Lowest_24'] = df['Low'].shift(1).rolling(window=24).min()
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست شکست محدوده باکیفیت بالا (TP 2% / SL 1%)")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 100:
        continue
        
    df1h = calculate_breakout_indicators(df1h)
    locked_until_index = 0
    
    for i in range(50, len(df1h) - 20):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        
        # فیلتر حجم سنگین (حداقل 1.5 برابر میانگین برای تایید شکست واقعی)
        if c1h['Volume'] < c1h['Vol_MA'] * 1.5:
            continue
            
        highest_24 = c1h['Highest_24']
        lowest_24 = c1h['Lowest_24']
        
        if pd.isna(highest_24) or pd.isna(lowest_24):
            continue
            
        # شرایط شکست سقف (لانگ قدرتمند)
        is_breakout_long = c1h['Close'] > highest_24
        # شرایط شکست کف (شورت قدرتمند)
        is_breakout_short = c1h['Close'] < lowest_24
        
        if not is_breakout_long and not is_breakout_short:
            continue
            
        if is_breakout_long:
            # ورود در بازگشایی کندل بعدی
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            tp = entry_price * 1.02  # حد سود 2 درصد
            sl = entry_price * 0.99  # حد ضرر 1 درصد
            
            outcome = None
            exit_idx = entry_idx
            
            for j in range(entry_idx, len(df1h)):
                f_c = df1h.iloc[j]
                exit_idx = j
                
                if f_c['Low'] <= sl:
                    outcome = 'LOSS'
                    break
                elif f_c['High'] >= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'LONG',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx + 3  # قفل طولانی‌تر برای جلوگیری از ترید رگباری
                
        elif is_breakout_short:
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            tp = entry_price * 0.98  # حد سود 2 درصد
            sl = entry_price * 1.01  # حد ضرر 1 درصد
            
            outcome = None
            exit_idx = entry_idx
            
            for j in range(entry_idx, len(df1h)):
                f_c = df1h.iloc[j]
                exit_idx = j
                
                if f_c['High'] >= sl:
                    outcome = 'LOSS'
                    break
                elif f_c['Low'] <= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'SHORT',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx + 3

print("\n============================================================")
print("📊 گزارش نهایی استراتژی شکست محدوده (Breakout Strategy)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات پورتفوی: {total_trades}")
    print(f"🔸 معاملات برنده (WIN): {total_wins}")
    print(f"🔸 معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی پورتفوی:** {portfolio_win_rate:.2f}%")
    print(f"💰 **امتیاز سودآوری خالص (Net Profit Score):** {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با این شرایط ثبت نشد.")

print("\n✨ پایان بک‌تست.")
