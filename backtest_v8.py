import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

def fetch_lbank_data(symbol, timeframe='1h', limit=1000):
    exchange = ccxt.lbank({'enableRateLimit': True})
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    # EMA 50 for trend
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # ATR 14 for volatility & SL buffer
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['shift'] if 'shift' in df else df['close'].shift())
    low_close = np.abs(df['low'] - df['shift'] if 'shift' in df else df['close'].shift())
    df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Swing Highs and Lows (20 periods)
    df['swing_high'] = df['high'].rolling(window=20).max()
    df['swing_low'] = df['low'].rolling(window=20).min()
    return df

def run_backtest():
    # Top 10 liquid crypto futures on LBank (using unified/ccxt symbols)
    symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
        'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'ADA/USDT:USDT', 
        'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT', 'NEAR/USDT:USDT'
    ]
    
    initial_capital = 1000.0
    capital = initial_capital
    fixed_margin = 100.0
    leverage = 5 # Standard futures leverage
    commission_rate = 0.0005 # 0.05%
    slippage = 0.0002 # 0.02%
    
    all_trades = []
    asset_performance = {}
    
    print("Fetching data and running backtest across top assets...")
    
    for symbol in symbols:
        df_1h = fetch_lbank_data(symbol, timeframe='1h', limit=3000)
        if df_1h.empty or len(df_1h) < 200:
            continue
            
        # Resample to 4H for trend filter
        df_4h = df_1h.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
        
        # Merge 4H trend into 1H dataframe (forward fill to avoid look-ahead bias)
        df_1h['trend_4h'] = df_4h['ema50'].reindex(df_1h.index, method='ffill')
        df_1h = calculate_indicators(df_1h)
        df_1h.dropna(inplace=True)
        
        position = None # 'LONG' or 'SHORT'
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        trade_results = []
        
        # Simulation Loop (No look-ahead bias: iterate strictly chronologically)
        for i in range(21, len(df_1h) - 1):
            current_time = df_1h.index[i]
            row = df_1h.iloc[i]
            prev_row = df_1h.iloc[i-1]
            next_open = df_1h.iloc[i+1]['open']
            
            # If in position, check exit on current candle high/low
            if position is not None:
                hit_tp = False
                hit_sl = False
                
                if position == 'LONG':
                    if row['high'] >= take_profit:
                        hit_tp = True
                    if row['low'] <= stop_loss:
                        hit_sl = True
                        
                    # Conservative rule: if both hit, SL first
                    if hit_sl and hit_tp:
                        exit_price = stop_loss * (1 - slippage)
                        pnl = (exit_price - entry_price) * (fixed_margin * leverage / entry_price) - (fixed_margin * leverage * commission_rate * 2)
                        trade_results.append({'type': 'LOSS', 'pnl': pnl, 'symbol': symbol})
                        capital += pnl
                        position = None
                    elif hit_sl:
                        exit_price = stop_loss * (1 - slippage)
                        pnl = (exit_price - entry_price) * (fixed_margin * leverage / entry_price) - (fixed_margin * leverage * commission_rate * 2)
                        trade_results.append({'type': 'LOSS', 'pnl': pnl, 'symbol': symbol})
                        capital += pnl
                        position = None
                    elif hit_tp:
                        exit_price = take_profit * (1 - slippage)
                        pnl = (exit_price - entry_price) * (fixed_margin * leverage / entry_price) - (fixed_margin * leverage * commission_rate * 2)
                        trade_results.append({'type': 'WIN', 'pnl': pnl, 'symbol': symbol})
                        capital += pnl
                        position = None
                        
                elif position == 'SHORT':
                    if row['low'] <= take_profit:
                        hit_tp = True
                    if row['high'] >= stop_loss:
                        hit_sl = True
                        
                    if hit_sl and hit_tp:
                        exit_price = stop_loss * (1 + slippage)
                        pnl = (entry_price - exit_price) * (fixed_margin * leverage / entry_price) - (fixed_margin * leverage * commission_rate * 2)
                        trade_results.append({'type': 'LOSS', 'pnl': pnl, 'symbol': symbol})
                        capital += pnl
                        position = None
                    elif hit_sl:
                        exit_price = stop_loss * (1 + slippage)
                        pnl = (entry_price - exit_price) * (fixed_margin * leverage / entry_price) - (fixed_margin * leverage * commission_rate * 2)
                        trade_results.append({'type': 'LOSS', 'pnl': pnl, 'symbol': symbol})
                        capital += pnl
                        position = None
                    elif hit_tp:
                        exit_price = take_profit * (1 + slippage)
                        pnl = (entry_price - exit_price) * (fixed_margin * leverage / entry_price) - (fixed_margin * leverage * commission_rate * 2)
                        trade_results.append({'type': 'WIN', 'pnl': pnl, 'symbol': symbol})
                        capital += pnl
                        position = None
            
            # If no position, look for setup based on closed candle [i]
            if position is None:
                # Trend condition from 4H
                is_bullish_trend = row['close'] > row['trend_4h']
                is_bearish_trend = row['close'] < row['trend_4h']
                
                # Liquidity Sweep & CHOCH Logic
                # Bullish setup: Sweep recent swing low then close higher
                swept_low = row['low'] < prev_row['swing_low']
                bullish_choch = row['close'] > prev_row['high']
                
                # Bearish setup: Sweep recent swing high then close lower
                swept_high = row['high'] > prev_row['swing_high']
                bearish_choch = row['close'] < prev_row['low']
                
                if is_bullish_trend and swept_low and bullish_choch:
                    position = 'LONG'
                    entry_price = next_open * (1 + slippage)
                    stop_loss = row['low'] - (0.2 * row['atr'])
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (2.0 * risk)
                    
                elif is_bearish_trend and swept_high and bearish_choch:
                    position = 'SHORT'
                    entry_price = next_open * (1 - slippage)
                    stop_loss = row['high'] + (0.2 * row['atr'])
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (2.0 * risk)
                    
        asset_performance[symbol] = trade_results
        all_trades.extend(trade_results)

    # Calculate metrics
    total_trades = len(all_trades)
    winning_trades = len([t for t in all_trades if t['type'] == 'WIN'])
    losing_trades = len([t for t in all_trades if t['type'] == 'LOSS'])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in all_trades])
    final_balance = initial_capital + total_pnl
    
    # Loss streaks calculation
    streaks = []
    current_streak = 0
    for t in all_trades:
        if t['type'] == 'LOSS':
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
        
    # Profit Factor & Avg Win/Loss
    total_wins_pnl = sum([t['pnl'] for t in all_trades if t['type'] == 'WIN'])
    total_losses_pnl = abs(sum([t['pnl'] for t in all_trades if t['type'] == 'LOSS']))
    profit_factor = (total_wins_pnl / total_losses_pnl) if total_losses_pnl > 0 else float('inf')
    avg_win = (total_wins_pnl / winning_trades) if winning_trades > 0 else 0
    avg_loss = (total_losses_pnl / losing_trades) if losing_trades > 0 else 0
    
    # Print Performance Report
    print("\n" + "="*40)
    print("BACKTEST PERFORMANCE REPORT (LBank Futures)")
    print("="*40)
    print(f"1- Total Trades: {total_trades}")
    print(f"2- Winning Trades: {winning_trades}")
    print(f"3- Losing Trades: {losing_trades}")
    print(f"4- Win Rate: {win_rate:.2f}%")
    print(f"5- Total PnL ($): ${total_pnl:.2f}")
    print(f"6- Final Balance ($): ${final_balance:.2f} (from $1000)")
    
    print("\n7 & 8 - PnL and Trade Count per Asset:")
    for sym, trades in asset_performance.items():
        sym_pnl = sum([t['pnl'] for t in trades])
        print(f"   - {sym}: {len(trades)} trades | PnL: ${sym_pnl:.2f}")
        
    print("\n9- Loss Streaks:")
    if streaks:
        for s in set(streaks):
            count = streaks.count(s)
            print(f"   {s} losses: {count} time(s)")
    else:
        print("   No losing streaks recorded.")
        
    print(f"\n11- Profit Factor: {profit_factor:.2f}")
    print(f"12- Average Win: ${avg_win:.2f} | Average Loss: ${avg_loss:.2f}")

if __name__ == '__main__':
    run_backtest()
