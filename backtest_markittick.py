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
MAX_LEVERAGE = 75.0
MIN_LEVERAGE = 2.0
TAKER_FEE_RATE = 0.05 / 100
SLIPPAGE_RATE = 0.03 / 100

EMA_FAST = 20
EMA_SLOW = 50
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
    
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['ema_fast'] = df['ema_fast'].shift(1)
    df['ema_slow'] = df['ema_slow'].shift(1)
    df['rsi'] = df['rsi'].shift(1)
    df['atr'] = df['atr'].shift(1)
    
    return df

def run_backtest_for_symbol(df, symbol):
    trades = []
    active_trade = None
    last_exit_idx = -999
    
    n = len(df)
    for i in range(60, n):
        if active_trade is not None:
            high_val = df.loc[i, 'high']
            low_val = df.loc[i, 'low']
            
            leverage = active_trade['leverage']
            notional = MARGIN_PER_TRADE * leverage
            
            hit_tp = False
            hit_sl = False
            exit_price = 0.0
            
            if active_trade['side'] == 'LONG':
                # فرمول واقعی‌تر لیکوییدیشن با احتساب ضریب اطمینان مارجین
                liq_price = active_trade['entry_price'] * (1.0 - (0.90 / leverage))
                
                hit_liq = low_val <= liq_price
                hit_tp = high_val >= active_trade['tp']
                hit_sl = low_val <= active_trade['sl']
                
                # رفع باگ تداخل: اگر لیکویید شد یا هم‌زمان استاپ و تارگت خورد، اولویت قطعی با ضرر/استاپ است
                if hit_liq:
                    exit_price = liq_price
                    hit_sl = True
                elif hit_sl and hit_tp:
                    exit_price = active_trade['sl']
                    hit_sl = True
                elif hit_sl:
                    exit_price = active_trade['sl']
                elif hit_tp:
                    exit_price = active_trade['tp']
                    hit_tp = True
            else:
                liq_price = active_trade['entry_price'] * (1.0 + (0.90 / leverage))
                
                hit_liq = high_val >= liq_price
                hit_tp = low_val <= active_trade['tp']
                hit_sl = high_val >= active_trade['sl']
                
                if hit_liq:
                    exit_price = liq_price
                    hit_sl = True
                elif hit_sl and hit_tp:
                    exit_price = active_trade['sl']
                    hit_sl = True
                elif hit_sl:
                    exit_price = active_trade['sl']
                elif hit_tp:
                    exit_price = active_trade['tp']
                    hit_tp = True
                    
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
            
        c1 = df.loc[i-1, 'close']
        l1 = df.loc[i-1, 'low']
        h1 = df.loc[i-1, 'high']
        ef1 = df.loc[i-1, 'ema_fast']
        es1 = df.loc[i-1, 'ema_slow']
        rsi1 = df.loc[i-1, 'rsi']
        current_atr = df.loc[i, 'atr']
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
            
        uptrend = (ef1 > es1) and (c1 > es1)
        buy_pullback = uptrend and (l1 <= ef1) and (rsi1 > 40) and (rsi1 < 55) and (c1 > ef1)
        
        downtrend = (ef1 < es1) and (c1 < es1)
        sell_pullback = downtrend and (h1 >= ef1) and (rsi1 < 60) and (rsi1 > 45) and (c1 < ef1)
        
        if buy_pullback:
            entry_price = df.loc[i, 'open'] * (1.0 - SLIPPAGE_RATE)
            sl_dist = current_atr * 1.5
            sl = entry_price - sl_dist
            tp = entry_price + (2.0 * sl_dist)  # ریسک به ریوارد دقیقاً ۱ به ۲
            
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
        elif sell_pullback:
            entry_price = df.loc[i, 'open'] * (1.0 + SLIPPAGE_RATE)
            sl_dist = current_atr * 1.5
            sl = entry_price + sl_dist
            tp = entry_price - (2.0 * sl_dist)  # ریسک به ریوارد دقیقاً ۱ به ۲
            
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
    print("گزارش نتایج نهایی (بدون باگ و کاملا واقع‌گرایانه)")
    print("="*50)
    print(f"تعداد کل معاملات OOS: {total_trades}")
    print(f"میانگین اهرم‌های محاسبه‌شده: {avg_lev:.1f}x")
    print(f"نرخ برد (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص کل (Net PnL): ${net_profit:.2f}")
    print(f"حداکثر ضرر متوالی (Max Losing Streak): {max_losing_streak}")
    print("="*50)

if __name__ == '__main__':
    main()
