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

# پارامترهای استخراج شده از تصویر تنظیمات شما
ATR_MULT = 1.5
AP = 14
BO_SENS = 2.5

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
    
    # محاسبه True Range و ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['tr'] = np.max(ranges, axis=1)
    df['atr'] = df['tr'].rolling(AP).mean()
    
    # محاسبه ساده Money Flow Index (MFI) معادل تریدینگ‌ویو
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    money_flow = typical_price * df['volume']
    pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(AP).sum()
    neg_mf = pd.Series(neg_flow).rolling(AP).sum()
    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-8)))
    df['mfi'] = mfi.shift(1) # شیفت برای جلوگیری از تقلب (مشابه [1] در پاین‌اسکریپت)
    
    return df

def run_backtest_for_symbol(df, symbol):
    trades = []
    active_trade = None
    last_exit_idx = -999
    
    n = len(df)
    alpha_trend = np.zeros(n)
    direction = np.ones(n, dtype=int)
    
    up_t = df['low'].shift(1) - df['atr'] * ATR_MULT
    down_t = df['high'].shift(1) + df['atr'] * ATR_MULT
    
    for i in range(AP + 2, n):
        mfi_val = df.loc[i, 'mfi']
        prev_at = alpha_trend[i-1] if i > 0 else 0.0
        
        if mfi_val >= 50:
            alpha_trend[i] = max(up_t.iloc[i], prev_at) if not pd.isna(up_t.iloc[i]) else prev_at
        else:
            alpha_trend[i] = min(down_t.iloc[i], prev_at) if not pd.isna(down_t.iloc[i]) else prev_at
            
        # تعیین جهت روند
        if i >= 3:
            if alpha_trend[i-1] > alpha_trend[i-3]:
                direction[i] = 1
            elif alpha_trend[i-1] < alpha_trend[i-3]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
        # بررسی سیگنال‌ها بدون تقلب
        if i < window_limit := 15:
            continue
            
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
                    'exit_time': df.loc[i, 'timestamp'],
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
            
        # منطق ورود بر اساس کراس و Breakout Momentum
        c1 = df.loc[i-1, 'close']
        c2 = df.loc[i-2, 'close']
        at1 = alpha_trend[i-1]
        at2 = alpha_trend[i-2]
        
        is_cross_up = (c1 > at1) and (c2 <= at2)
        is_cross_dn = (c1 < at1) and (c2 >= at2)
        bo_momentum = abs(c1 - c2) > (df.loc[i-1, 'atr'] * BO_SENS)
        
        is_trend_cross_up = (alpha_trend[i-1] > alpha_trend[i-3]) and (alpha_trend[i-2] <= alpha_trend[i-4] if i >= 4 else True)
        is_trend_cross_dn = (alpha_trend[i-1] < alpha_trend[i-3]) and (alpha_trend[i-2] >= alpha_trend[i-4] if i >= 4 else True)
        
        buy_signal = is_trend_cross_up or (bo_momentum and is_cross_up)
        sell_signal = is_trend_cross_dn or (bo_momentum and is_cross_dn)
        
        current_atr = df.loc[i, 'atr']
        if pd.isna(current_atr) or current_atr <= 0:
            continue
            
        if buy_signal:
            entry_price = df.loc[i+1, 'open'] * (1.0 - SLIPPAGE_RATE)
            sl = alpha_trend[i] if not pd.isna(alpha_trend[i]) else entry_price - (2.0 * current_atr)
            risk = entry_price - sl
            if risk <= 0:
                continue
            tp = entry_price + (2.0 * risk) # ریوارد 1 به 2
            
            active_trade = {
                'side': 'LONG',
                'entry_time': df.loc[i+1, 'timestamp'],
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp
            }
        elif sell_signal:
            entry_price = df.loc[i+1, 'open'] * (1.0 + SLIPPAGE_RATE)
            sl = alpha_trend[i] if not pd.isna(alpha_trend[i]) else entry_price + (2.0 * current_atr)
            risk = sl - entry_price
            if risk <= 0:
                continue
            tp = entry_price - (2.0 * risk) # ریوارد 1 به 2
            
            active_trade = {
                'side': 'SHORT',
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
    print("گزارش نتایج نهایی بک‌تست Out-of-Sample (OOS)")
    print("="*50)
    print(f"تعداد کل معاملات OOS: {total_trades}")
    print(f"نرخ برد (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص کل (Net PnL): ${net_profit:.2f}")
    print(f"حداکثر ضرر متوالی (Max Losing Streak): {max_losing_streak}")
    print("="*50)

if __name__ == '__main__':
    main()
            
