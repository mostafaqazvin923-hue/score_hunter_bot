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
MARGIN_PER_TRADE = 100.0  # مارجین ثابت ۱۰۰ دلار برای هر معامله
MAX_LEVERAGE = 75.0       # سقف مجاز اهرم صرافی
MIN_LEVERAGE = 2.0        # کف مجاز اهرم
TAKER_FEE_RATE = 0.05 / 100
SLIPPAGE_RATE = 0.03 / 100

BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
ATR_PERIOD = 14

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
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    
    df['bb_middle'] = df['close'].rolling(BB_PERIOD).mean()
    rolling_std = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * BB_STD)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * BB_STD)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['bb_upper'] = df['bb_upper'].shift(1)
    df['bb_lower'] = df['bb_lower'].shift(1)
    df['rsi'] = df['rsi'].shift(1)
    df['atr'] = df['atr'].shift(1)
    
    return df

def run_backtest_for_symbol(df, symbol):
    trades = []
    active_trade = None
    last_exit_idx = -999
    
    n = len(df)
    for i in range(50, n):
        if active_trade is not None:
            high_val = df.loc[i, 'high']
            low_val = df.loc[i, 'low']
            
            hit_tp = False
            hit_sl = False
            exit_price = 0.0
            
            leverage = active_trade['leverage']
            notional = MARGIN_PER_TRADE * leverage
            
            if active_trade['side'] == 'LONG':
                liq_price = active_trade['entry_price'] * (1.0 - (1.0 / leverage) * 0.95)
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
                liq_price = active_trade['entry_price'] * (1.0 + (1.0 / leverage) * 0.95)
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
                    'exit_time': df.loc[i, 'timestamp'],
                    'entry_price': active_trade['entry_price'],
                    'exit_price': exit_price,
                    'leverage': leverage,
                    'net_pnl': net_pnl,
                    'win': net_pnl > 0
                })
                last_exit_idx = i
                active_trade = None
            continue
            
        if i <= last_exit_idx + 1:
            continue
            
        close_prev = df.loc[i-1, 'close']
        bb_lower = df.loc[i-1, 'bb_lower']
        bb_upper = df.loc[i-1, 'bb_upper']
        rsi_val = df.loc[i-1, 'rsi']
        current_atr = df.loc[i, 'atr']
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
            
        buy_signal = (close_prev <= bb_lower) and (rsi_val < 30)
        sell_signal = (close_prev >= bb_upper) and (rsi_val > 70)
        
        if buy_signal:
            entry_price = df.loc[i, 'open'] * (1.0 - SLIPPAGE_RATE)
            sl_dist = current_atr * 1.0
            sl = entry_price - sl_dist
            tp = entry_price + (2.0 * sl_dist)  # ریوارد ۱ به ۲
            
            # محاسبه اهرم پویا بر اساس حد ضرر ۵۰ درصدی مارجین
            risk_pct = sl_dist / entry_price
            dynamic_lev = 0.50 / risk_pct
            dynamic_lev = max(MIN_LEVERAGE, min(MAX_LEVERAGE, dynamic_lev))
            
            active_trade = {
                'side': 'LONG',
                'entry_time': df.loc[i, 'timestamp'],
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'leverage': dynamic_lev
            }
        elif sell_signal:
            entry_price = df.loc[i, 'open'] * (1.0 + SLIPPAGE_RATE)
            sl_dist = current_atr * 1.0
            sl = entry_price + sl_dist
            tp = entry_price - (2.0 * sl_dist)  # ریوارد ۱ به ۲
            
            # محاسبه اهرم پویا بر اساس حد ضرر ۵۰ درصدی مارجین
            risk_pct = sl_dist / entry_price
            dynamic_lev = 0.50 / risk_pct
            dynamic_lev = max(MIN_LEVERAGE, min(MAX_LEVERAGE, dynamic_lev))
            
            active_trade = {
                'side': 'SHORT',
                'entry_time': df.loc[i, 'timestamp'],
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'leverage': dynamic_lev
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
        print("[نتیجه] هیچ معامله‌ای ثبت نشد.")
        return
        
    df_trades = pd.DataFrame(all_trades)
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
    print("گزارش نتایج استراتژی با «اهرم پویا» (TP=100% | SL=50%)")
    print("="*50)
    print(f"تعداد کل معاملات OOS: {total_trades}")
    print(f"میانگین اهرم‌های محاسبه‌شده: {avg_lev:.1f}x")
    print(f"نرخ برد (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص کل (Net PnL): ${net_profit:.2f}")
    print(f"حداکثر ضرر متوالی (Max Losing Streak): {max_losing_streak}")
    print("="*50)

if __name__ == '__main__':
    main()
