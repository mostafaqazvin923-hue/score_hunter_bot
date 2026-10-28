import os
import pandas as pd
import numpy as np

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'AVAXUSDT', 
    'NEARUSDT', 'ADAUSDT', 'BNBUSDT', 'APTUSDT', 'CRVUSDT', 
    'ONDOUSDT', 'PENDLEUSDT', 'ICPUSDT', 'WIFUSDT'
]
DATA_DIR = 'data_cache'

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
MAX_LEVERAGE = 50.0
MIN_LEVERAGE = 2.0
TAKER_FEE_RATE = 0.05 / 100
SLIPPAGE_RATE = 0.03 / 100

def load_symbol_data(symbol):
    ohlcv_path = os.path.join(DATA_DIR, f"{symbol}_OHLCV.csv")
    fr_path = os.path.join(DATA_DIR, f"{symbol}_FundingRate.csv")
    
    if not os.path.exists(ohlcv_path) or not os.path.exists(fr_path):
        return None
        
    df_c = pd.read_csv(ohlcv_path)
    df_c['timestamp'] = pd.to_datetime(df_c['timestamp'], format='mixed', utc=True)
    
    df_f = pd.read_csv(fr_path)
    df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], format='mixed', utc=True)
    
    df = pd.merge_asof(df_c.sort_values('timestamp'), 
                       df_f.sort_values('timestamp'), 
                       on='timestamp', 
                       direction='backward')
    df['fundingRate'] = df['fundingRate'].fillna(0.0)
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['tr'] = np.max(ranges, axis=1)
    
    df['close_prev'] = df['close'].shift(1)
    df['high_prev'] = df['high'].shift(1)
    df['low_prev'] = df['low'].shift(1)
    df['volume_prev'] = df['volume'].shift(1)
    
    df['atr'] = df['tr'].rolling(14).mean().shift(1)
    df['structure_high'] = df['high'].rolling(24).max().shift(1)
    df['structure_low'] = df['low'].rolling(24).min().shift(1)
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean().shift(1)
    df['vol_ma'] = df['volume'].rolling(20).mean().shift(1)
    
    df['date'] = df['timestamp'].dt.date
    df['symbol'] = symbol
    
    return df

def main():
    all_oos_dfs = []
    for symbol in SYMBOLS:
        df = load_symbol_data(symbol)
        if df is None:
            continue
            
        total_len = len(df)
        val_idx = int(total_len * 0.80)
        df_oos = df.iloc[val_idx:].reset_index(drop=True)
        all_oos_dfs.append(df_oos)
        
    if not all_oos_dfs:
        print("[خطا] هیچ داده‌ای بارگذاری نشد.")
        return
        
    master_df = pd.concat(all_oos_dfs, ignore_index=True)
    master_df = master_df.sort_values('timestamp').reset_index(drop=True)
    
    active_trades = {}      
    last_exit_time = {}     
    traded_dates = set()    
    trades_list = []
    
    min_utc_timestamp = pd.Timestamp("1970-01-01 00:00:00", tz="UTC")
    
    n = len(master_df)
    for idx in range(50, n):
        row = master_df.loc[idx]
        sym = row['symbol']
        ts = row['timestamp']
        current_date = row['date']
        
        if sym in active_trades:
            trade = active_trades[sym]
            high_val = row['high']
            low_val = row['low']
            leverage = trade['leverage']
            notional = MARGIN_PER_TRADE * leverage
            
            hit_tp = False
            hit_sl = False
            exit_price = 0.0
            
            if trade['side'] == 'LONG':
                liq_price = trade['entry_price'] * (1.0 - (0.90 / leverage))
                hit_liq = low_val <= liq_price
                hit_tp = high_val >= trade['tp']
                hit_sl = low_val <= trade['sl']
                
                if hit_liq:
                    exit_price = liq_price
                    hit_sl = True
                elif hit_sl and hit_tp:
                    exit_price = trade['sl']
                    hit_sl = True
                elif hit_sl:
                    exit_price = trade['sl']
                elif hit_tp:
                    exit_price = trade['tp']
                    hit_tp = True
            else:
                liq_price = trade['entry_price'] * (1.0 + (0.90 / leverage))
                hit_liq = high_val >= liq_price
                hit_tp = low_val <= trade['tp']
                hit_sl = high_val >= trade['sl']
                
                if hit_liq:
                    exit_price = liq_price
                    hit_sl = True
                elif hit_sl and hit_tp:
                    exit_price = trade['sl']
                    hit_sl = True
                elif hit_sl:
                    exit_price = trade['sl']
                elif hit_tp:
                    exit_price = trade['tp']
                    hit_tp = True
                    
            if hit_tp or hit_sl:
                if trade['side'] == 'LONG':
                    gross_pnl = notional * (exit_price - trade['entry_price']) / trade['entry_price']
                else:
                    gross_pnl = notional * (trade['entry_price'] - exit_price) / trade['entry_price']
                    
                fees = notional * TAKER_FEE_RATE * 2
                net_pnl = gross_pnl - fees
                
                trades_list.append({
                    'symbol': sym,
                    'side': trade['side'],
                    'entry_time': trade['entry_time'],
                    'exit_time': ts,
                    'entry_date': trade['entry_date'],
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'leverage': leverage,
                    'net_pnl': net_pnl,
                    'win': net_pnl > 0
                })
                last_exit_time[sym] = ts
                del active_trades[sym]
            continue
            
        if ts <= last_exit_time.get(sym, min_utc_timestamp):
            continue
            
        if current_date in traded_dates:
            continue
            
        if pd.isna(row['atr']) or row['atr'] <= 0 or pd.isna(row['vol_ma']) or pd.isna(row['ema50']) or pd.isna(row['structure_high']):
            continue
            
        c1 = row['close_prev']
        v1 = row['volume_prev']
        vol_ma1 = row['vol_ma']
        sh1 = row['structure_high']
        sl1 = row['structure_low']
        ema50 = row['ema50']
        current_atr = row['atr']
        
        # تعدیل ضریب حجم به 1.2 برای اطمینان از ثبت تعداد معاملات کافی در سبد
        buy_signal = (c1 > sh1) and (v1 > vol_ma1 * 1.2) and (c1 > ema50)
        sell_signal = (c1 < sl1) and (v1 > vol_ma1 * 1.2) and (c1 < ema50)
        
        if buy_signal:
            if current_date in traded_dates:
                continue
                
            entry_price = row['open'] * (1.0 + SLIPPAGE_RATE)
            sl_dist = max(current_atr * 1.0, (entry_price - row['low_prev']))
            sl = entry_price - sl_dist
            tp = entry_price + (2.0 * sl_dist)  # ریسک به ریوارد ۱ به ۲
            
            risk_pct = sl_dist / entry_price
            dynamic_lev = 0.50 / risk_pct
            dynamic_lev = max(MIN_LEVERAGE, min(MAX_LEVERAGE, dynamic_lev))
            
            traded_dates.add(current_date)
            active_trades[sym] = {
                'side': 'LONG',
                'entry_time': ts,
                'entry_date': current_date,
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'leverage': dynamic_lev
            }
        elif sell_signal:
            if current_date in traded_dates:
                continue
                
            entry_price = row['open'] * (1.0 - SLIPPAGE_RATE)
            sl_dist = max(current_atr * 1.0, (row['high_prev'] - entry_price))
            sl = entry_price + sl_dist
            tp = entry_price - (2.0 * sl_dist)  # ریسک به ریوارد ۱ به ۲
            
            risk_pct = sl_dist / entry_price
            dynamic_lev = 0.50 / risk_pct
            dynamic_lev = max(MIN_LEVERAGE, min(MAX_LEVERAGE, dynamic_lev))
            
            traded_dates.add(current_date)
            active_trades[sym] = {
                'side': 'SHORT',
                'entry_time': ts,
                'entry_date': current_date,
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'leverage': dynamic_lev
            }
            
    if not trades_list:
        print("[نتیجه] هیچ معامله‌ای ثبت نشد.")
        return
        
    df_trades = pd.DataFrame(trades_list)
    total_trades = len(df_trades)
    wins = df_trades['win'].sum()
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit = df_trades['net_pnl'].sum()
    avg_lev = df_trades['leverage'].mean()
    
    max_losing_streak = 0
    current_streak = 0
    for win_status in df_trades['win']:
        if not win_status:
            current_streak += 1
            if current_streak > max_losing_streak:
                max_losing_streak = current_streak
        else:
            current_streak = 0
            
    print("\n" + "="*50)
    print("گزارش نتایج نهایی (با حجم بهینه‌شده)")
    print("="*50)
    print(f"تعداد کل معاملات OOS: {total_trades}")
    print(f"میانگین اهرم‌های پویا: {avg_lev:.1f}x")
    print(f"نرخ برد (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص کل (Net PnL): ${net_profit:.2f}")
    print(f"حداکثر ضرر متوالی (Max Losing Streak): {max_losing_streak}")
    print("="*50)

if __name__ == '__main__':
    main()
