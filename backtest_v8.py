import pandas as pd
import numpy as np
import datetime

# ==============================================================================
# اسکریپت کامل بک‌تست استراتژی High Win-Rate با نسبت ریسک به ریوارد ۱ به ۲ (R:R = 1:2)
# ==============================================================================

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rma(series, length):
    return series.ewm(alpha=1.0/length, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_cp = (df['high'] - df['close'].shift(1)).abs()
    low_cp = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return calculate_rma(tr, period)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = calculate_rma(gain, period)
    avg_loss = calculate_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_adx(df, period=14):
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    high_low = df['high'] - df['low']
    high_cp = (df['high'] - df['close'].shift(1)).abs()
    low_cp = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    
    smooth_tr = calculate_rma(pd.Series(tr, index=df.index), period)
    smooth_plus_dm = calculate_rma(pd.Series(plus_dm, index=df.index), period)
    smooth_minus_dm = calculate_rma(pd.Series(minus_dm, index=df.index), period)
    
    plus_di = 100 * (smooth_plus_dm / smooth_tr.replace(0, np.nan))
    minus_di = 100 * (smooth_minus_dm / smooth_tr.replace(0, np.nan))
    
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = calculate_rma(dx.fillna(0), period)
    return adx.fillna(0)

def run_backtest(df, symbol="BTC-USDT", rr_ratio=2.0, atr_sl_mult=1.2, initial_capital=1000.0, risk_per_trade_pct=2.0):
    df = df.copy()
    df['ema100'] = calculate_ema(df['close'], 100)
    df['atr'] = calculate_atr(df, 14)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    
    df['avg_vol'] = df['volume'].rolling(20).mean()
    df['vol_break'] = df['volume'] > (1.2 * df['avg_vol'])
    
    candle_range = df['high'] - df['low']
    body_size = (df['close'] - df['open']).abs()
    df['strong_candle'] = np.where(candle_range > 0, (body_size / candle_range) > 0.55, False)
    df['strong_trend'] = df['adx'] > 22
    
    df['upper_break'] = df['high'].shift(1).rolling(10).max()
    df['lower_break'] = df['low'].shift(1).rolling(10).min()

    trades = []
    active_trade = None
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0

    for i in range(101, len(df)):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        
        # ۱. مدیریت پوزیشن فعال
        if active_trade is not None:
            t = active_trade
            if t['type'] == 'LONG':
                if current_candle['low'] <= t['sl']:
                    pnl = -t['risk_amount']
                    trades.append({'symbol': symbol, 'type': 'LONG', 'entry_time': t['entry_time'], 'exit_time': current_candle['timestamp'], 'entry': t['entry'], 'exit': t['sl'], 'pnl': pnl, 'result': 'SL', 'r_multiple': -1.0})
                    capital += pnl
                    active_trade = None
                elif current_candle['high'] >= t['tp']:
                    pnl = t['risk_amount'] * rr_ratio
                    trades.append({'symbol': symbol, 'type': 'LONG', 'entry_time': t['entry_time'], 'exit_time': current_candle['timestamp'], 'entry': t['entry'], 'exit': t['tp'], 'pnl': pnl, 'result': 'TP', 'r_multiple': rr_ratio})
                    capital += pnl
                    active_trade = None
                    
            elif t['type'] == 'SHORT':
                if current_candle['high'] >= t['sl']:
                    pnl = -t['risk_amount']
                    trades.append({'symbol': symbol, 'type': 'SHORT', 'entry_time': t['entry_time'], 'exit_time': current_candle['timestamp'], 'entry': t['entry'], 'exit': t['sl'], 'pnl': pnl, 'result': 'SL', 'r_multiple': -1.0})
                    capital += pnl
                    active_trade = None
                elif current_candle['low'] <= t['tp']:
                    pnl = t['risk_amount'] * rr_ratio
                    trades.append({'symbol': symbol, 'type': 'SHORT', 'entry_time': t['entry_time'], 'exit_time': current_candle['timestamp'], 'entry': t['entry'], 'exit': t['tp'], 'pnl': pnl, 'result': 'TP', 'r_multiple': rr_ratio})
                    capital += pnl
                    active_trade = None

        # ۲. ورود به پوزیشن جدید
        if active_trade is None:
            c_close = prev_candle['close']
            c_atr = prev_candle['atr']
            
            long_cond = (
                (c_close > prev_candle['ema100']) and
                (c_close > prev_candle['upper_break']) and
                (prev_candle['rsi'] > 52) and
                prev_candle['vol_break'] and
                prev_candle['strong_candle'] and
                prev_candle['strong_trend']
            )
            
            short_cond = (
                (c_close < prev_candle['ema100']) and
                (c_close < prev_candle['lower_break']) and
                (prev_candle['rsi'] < 48) and
                prev_candle['vol_break'] and
                prev_candle['strong_candle'] and
                prev_candle['strong_trend']
            )
            
            if long_cond:
                entry = current_candle['open']
                sl = entry - (c_atr * atr_sl_mult)
                tp = entry + ((entry - sl) * rr_ratio)
                risk_amt = capital * (risk_per_trade_pct / 100.0)
                active_trade = {'type': 'LONG', 'entry_time': current_candle['timestamp'], 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
            elif short_cond:
                entry = current_candle['open']
                sl = entry + (c_atr * atr_sl_mult)
                tp = entry - ((sl - entry) * rr_ratio)
                risk_amt = capital * (risk_per_trade_pct / 100.0)
                active_trade = {'type': 'SHORT', 'entry_time': current_candle['timestamp'], 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}

        if capital > peak_capital:
            peak_capital = capital
        dd = (peak_capital - capital) / peak_capital * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    trades_df = pd.DataFrame(trades)
    return capital, trades_df, max_drawdown
