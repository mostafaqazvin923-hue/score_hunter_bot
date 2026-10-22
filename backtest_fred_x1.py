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
LEVERAGE = 50.0
NOTIONAL_SIZE = MARGIN_PER_TRADE * LEVERAGE
TAKER_FEE_RATE = 0.05 / 100
SLIPPAGE_RATE = 0.03 / 100

def load_symbol_data(symbol):
    ohlcv_path = os.path.join(DATA_DIR, f"{symbol}_OHLCV.csv")
    fr_path = os.path.join(DATA_DIR, f"{symbol}_FundingRate.csv")
    
    if not os.path.exists(ohlcv_path) or not os.path.exists(fr_path):
        return None
        
    df_c = pd.read_csv(ohlcv_path)
    # استفاده از format='mixed' برای مدیریت امن انواع فرمت‌های تاریخ و زمان
    df_c['timestamp'] = pd.to_datetime(df_c['timestamp'], format='mixed', utc=True)
    
    df_f = pd.read_csv(fr_path)
    df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], format='mixed', utc=True)
    
    df = pd.merge_asof(df_c.sort_values('timestamp'), 
                       df_f.sort_values('timestamp'), 
                       on='timestamp', 
                       direction='backward')
    df['fundingRate'] = df['fundingRate'].fillna(0.0)
    return df

def run_backtest_for_symbol(df, symbol):
    trades = []
    active_trade = None
    last_exit_idx = -999
    
    window = 168
    df['fr_mean'] = df['fundingRate'].rolling(window=window, min_periods=24).mean()
    df['fr_std'] = df['fundingRate'].rolling(window=window, min_periods=24).std()
    df['fr_zscore'] = (df['fundingRate'] - df['fr_mean']) / (df['fr_std'] + 1e-8)
    
    for i in range(window + 10, len(df) - 1):
        current_time = df.loc[i, 'timestamp']
        
        if active_trade is not None:
            high_val = df.loc[i, 'high']
            low_val = df.loc[i, 'low']
            
            hit_tp = False
            hit_sl = False
            exit_price = 0.0
            
            if active_trade['side'] == 'LONG':
                liq_price = active_trade['entry_price'] * (1.0 - (1.0 / LEVERAGE) * 0.95)
                if low_val <= liq_price:
                    exit_price = liq_price
                    hit_sl = True
                elif high_val >= active_trade['tp']:
                    exit_price = active_trade['tp']
                    hit_tp = True
                elif low_val <= active_trade['sl']:
                    exit_price = active_trade['sl']
                    hit_sl = True
            else:
                liq_price = active_trade['entry_price'] * (1.0 + (1.0 / LEVERAGE) * 0.95)
                if high_val >= liq_price:
                    exit_price = liq_price
                    hit_sl = True
                elif low_val <= active_trade['tp']:
                    exit_price = active_trade['tp']
                    hit_tp = True
                elif high_val >= active_trade['sl']:
                    exit_price = active_trade['sl']
                    hit_sl = True
                    
            if hit_tp or hit_sl:
                notional = NOTIONAL_SIZE
                if active_trade['side'] == 'LONG':
                    gross_pnl = notional * (exit_price - active_trade['entry_price']) / active_trade['entry_price']
                else:
                    gross_pnl = notional * (active_trade['entry_price'] - exit_price) / active_trade['entry_price']
                    
                fees = notional * TAKER_FEE_RATE * 2
                net_pnl = gross_pnl - fees
                
                trades.append({
                    'symbol': symbol,
                    'side': active_trade['side'],
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'entry_price': active_trade['entry_price'],
                    'exit_price': exit_price,
                    'net_pnl': net_pnl,
                    'win': net_pnl > 0
                })
                last_exit_idx = i
                active_trade = None
            continue
            
        if i <= last_exit_idx + 1:
            continue
            
        z_prev = df.loc[i-1, 'fr_zscore']
        fr_curr = df.loc[i, 'fundingRate']
        fr_prev = df.loc[i-1, 'fundingRate']
        
        if z_prev >= 2.5 and 0 < fr_curr < fr_prev:
            entry_price = df.loc[i+1, 'open'] * (1.0 + SLIPPAGE_RATE)
            sl = df.loc[i-9:i, 'high'].max() * 1.002
            risk = sl - entry_price
            if risk <= 0:
                continue
            tp = entry_price - (2.0 * risk)
            
            liq_price = entry_price * (1.0 + (1.0 / LEVERAGE) * 0.95)
            if sl >= liq_price:
                continue
                
            active_trade = {
                'side': 'SHORT',
                'entry_time': df.loc[i+1, 'timestamp'],
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp
            }
            
        elif z_prev <= -2.5 and fr_prev < fr_curr < 0:
            entry_price = df.loc[i+1, 'open'] * (1.0 - SLIPPAGE_RATE)
            sl = df.loc[i-9:i, 'low'].min() * 0.998
            risk = entry_price - sl
            if risk <= 0:
                continue
            tp = entry_price + (2.0 * risk)
            
            liq_price = entry_price * (1.0 - (1.0 / LEVERAGE) * 0.95)
            if sl <= liq_price:
                continue
                
            active_trade = {
                'side': 'LONG',
                'entry_time': df.loc[i+1, 'timestamp'],
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp
            }
            
    return trades

def main():
    all_trades = []
    for symbol in SYMBOLS:
        df = load_symbol_data(symbol)
        if df is None:
            continue
            
        total_len = len(df)
        val_idx = int(total_len * 0.80)
        
        df_oos = df.iloc[val_idx:].reset_index(drop=True)
        trades = run_backtest_for_symbol(df_oos, symbol)
        all_trades.extend(trades)
        
    if not all_trades:
        print("[نتیجه] هیچ معامله‌ای در بخش Out-of-Sample ثبت نشد.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    total_trades = len(df_trades)
    wins = df_trades['win'].sum()
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit = df_trades['net_pnl'].sum()
    
    df_trades['loss'] = ~df_trades['win']
    df_trades['streak'] = df_trades['loss'].groupby((~df_trades['loss']).cumsum()).cumsum()
    max_losing_streak = df_trades['streak'].max()
    
    print("\n" + "="*50)
    print("گزارش نتایج نهایی بک‌تست Out-of-Sample (OOS)")
    print("="*50)
    print(f"تعداد کل معاملات OOS: {total_trades}")
    print(f"نرخ برد (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص کل (Net PnL): ${net_profit:.2f}")
    print(f"حداکثر ضرر متوالی (Max Losing Streak): {max_losing_streak}")
    print("="*50)

if __name__ == '__main__':
    main()
