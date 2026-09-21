import os
import subprocess
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

# ============================================================
# HUNTER-V116 — 180-DAY INSTITUTIONAL MULTI-TIMEFRAME ENGINE
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True, "timeout": 20000})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}

LOOKBACK_DAYS = 180  # تنظیم روی ۱۸۰ روز برای دریافت کامل و بدون نقص دیتای ۱۵ دقیقه
TIMEFRAME_BASE = "15m"

SLIPPAGE = 0.0003
FEE_RATE = 0.0007
INITIAL_CAPITAL = 2000.0
TRADE_MARGIN = 100.0
LEVERAGE = 5.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS + 10)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V116 — 180-DAY INSTITUTIONAL CONFLUENCE ENGINE INITIALIZED")
print("=" * 68)

processed_data = {}

def fetch_ohlcv_data(lbank_symbol, timeframe):
    all_ohlcv = []
    current_since = since_timestamp
    try:
        while current_since < exchange.milliseconds():
            batch = exchange.fetch_ohlcv(lbank_symbol, timeframe=timeframe, since=current_since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= current_since:
                break
            current_since = last_ts + 1
            if len(batch) < 1000:
                break
    except Exception as e:
        print(f"Error fetching {lbank_symbol}: {e}")
        return None

    if not all_ohlcv:
        return None

    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index("Date", inplace=True)
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    print(f"Downloading 15m data for {symbol} (180 Days)...")
    df_15m = fetch_ohlcv_data(lbank_symbol, TIMEFRAME_BASE)

    if df_15m is None or len(df_15m) < 500:
        continue

    # ساخت تایم فریم‌های بالاتر بدون نگاه به آینده (Resample دقیق)
    df_1h = df_15m.resample('1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    df_4h = df_15m.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # 1. رژیم ۴ ساعته
    df_4h["EMA_50"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
    df_4h["EMA_200"] = df_4h["Close"].ewm(span=200, adjust=False).mean()
    df_4h["Regime_Bullish"] = (df_4h["Close"] > df_4h["EMA_200"]) & (df_4h["EMA_50"] > df_4h["EMA_200"])
    df_4h["Regime_Bearish"] = (df_4h["Close"] < df_4h["EMA_200"]) & (df_4h["EMA_50"] < df_4h["EMA_200"])

    # 2. ساختار و نقدینگی ۱ ساعته
    df_1h["Swing_High"] = df_1h["High"].rolling(5, center=True).max()
    df_1h["Swing_Low"] = df_1h["Low"].rolling(5, center=True).min()
    df_1h["ATR"] = (df_1h["High"] - df_1h["Low"]).rolling(14).mean()

    # 3. اندیکاتورهای ۱۵ دقیقه برای ورود
    df_15m["ATR"] = (df_15m["High"] - df_15m["Low"]).rolling(14).mean()
    df_15m["Body"] = (df_15m["Close"] - df_15m["Open"]).abs()
    df_15m["Avg_Body"] = df_15m["Body"].rolling(20).mean()

    # فیلتر بازه دقیق ۱۸۰ روزه
    cutoff_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    df_15m = df_15m[df_15m.index >= cutoff_date]

    if len(df_15m) > 100:
        processed_data[symbol] = {
            "15m": df_15m,
            "1h": df_1h,
            "4h": df_4h
        }

print(f"Successfully processed symbols: {len(processed_data)}")

def run_backtest(data_dict):
    all_trades = []

    for symbol, dsets in data_dict.items():
        df_15 = dsets["15m"]
        df_1h = dsets["1h"]
        df_4h = dsets["4h"]

        for i in range(50, len(df_15)):
            t_curr = df_15.index[i]
            c_row = df_15.iloc[i]
            p_row = df_15.iloc[i-1]

            h_sub = df_1h[df_1h.index <= t_curr]
            h_4sub = df_4h[df_4h.index <= t_curr]

            if len(h_sub) < 10 or len(h_4sub) < 10:
                continue

            regime_bull = h_4sub.iloc[-1]["Regime_Bullish"]
            regime_bear = h_4sub.iloc[-1]["Regime_Bearish"]

            recent_lows = h_sub["Low"].iloc[-10:-2]
            recent_highs = h_sub["High"].iloc[-10:-2]
            if len(recent_lows) == 0 or len(recent_highs) == 0:
                continue

            min_support = recent_lows.min()
            max_resistance = recent_highs.max()

            sweep_low = (p_row["Low"] < min_support) and (p_row["Close"] > min_support)
            sweep_high = (p_row["High"] > max_resistance) and (p_row["Close"] < max_resistance)

            displacement_up = (c_row["Close"] > c_row["Open"]) and (c_row["Body"] > 1.5 * c_row["Avg_Body"])
            displacement_down = (c_row["Close"] < c_row["Open"]) and (c_row["Body"] > 1.5 * c_row["Avg_Body"])

            valid_long = regime_bull and sweep_low and displacement_up
            valid_short = regime_bear and sweep_high and displacement_down

            if not (valid_long or valid_short):
                continue

            side = "LONG" if valid_long else "SHORT"
            entry_price = c_row["Open"] * (1 + SLIPPAGE) if side == "LONG" else c_row["Open"] * (1 - SLIPPAGE)
            atr = c_row["ATR"]

            if np.isnan(atr) or atr <= 0:
                continue

            if side == "LONG":
                sl = entry_price - (1.5 * atr)
                tp = entry_price + (3.0 * atr)
            else:
                sl = entry_price + (1.5 * atr)
                tp = entry_price - (3.0 * atr)

            outcome = "LOSS"
            exit_price = sl

            for j in range(i + 1, min(i + 35, len(df_15))):
                fut = df_15.iloc[j]
                if side == "LONG":
                    hit_sl = fut["Low"] <= sl
                    hit_tp = fut["High"] >= tp
                    if hit_sl and hit_tp:
                        outcome = "LOSS"
                        exit_price = sl
                        break
                    elif hit_sl:
                        outcome = "LOSS"
                        exit_price = sl
                        break
                    elif hit_tp:
                        outcome = "WIN"
                        exit_price = tp
                        break
                else:
                    hit_sl = fut["High"] >= sl
                    hit_tp = fut["Low"] <= tp
                    if hit_sl and hit_tp:
                        outcome = "LOSS"
                        exit_price = sl
                        break
                    elif hit_sl:
                        outcome = "LOSS"
                        exit_price = sl
                        break
                    elif hit_tp:
                        outcome = "WIN"
                        exit_price = tp
                        break

            price_ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
            dollar_pnl = (TRADE_MARGIN * LEVERAGE * price_ret) - (TRADE_MARGIN * LEVERAGE * FEE_RATE * 2)

            all_trades.append({
                "Timestamp": t_curr,
                "Symbol": symbol,
                "Side": side,
                "Outcome": outcome,
                "Dollar_PnL": dollar_pnl,
                "Month": t_curr.strftime("%Y-%m")
            })

    return pd.DataFrame(all_trades)

if __name__ == "__main__":
    trades_df = run_backtest(processed_data)
    
    n = len(trades_df)
    win_rate = (trades_df["Outcome"].eq("WIN").mean() * 100) if n > 0 else 0.0
    loss_rate = 100.0 - win_rate
    net_pnl = float(trades_df["Dollar_PnL"].sum()) if n > 0 else 0.0
    
    wins = trades_df[trades_df["Outcome"] == "WIN"]["Dollar_PnL"]
    losses = trades_df[trades_df["Outcome"] == "LOSS"]["Dollar_PnL"]
    
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    trades_df["Cumulative_PnL"] = trades_df["Dollar_PnL"].cumsum()
    trades_df["Peak"] = trades_df["Cumulative_PnL"].cummax()
    trades_df["Drawdown"] = trades_df["Cumulative_PnL"] - trades_df["Peak"]
    max_dd = trades_df["Drawdown"].min() if n > 0 else 0.0

    streaks = []
    current_streak = 0
    for outcome in trades_df["Outcome"]:
        if outcome == "LOSS":
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
    
    max_consecutive_losses = max(streaks) if streaks else 0

    print("=" * 72)
    print("===== BACKTEST RESULT =====")
    print(f"Period: 180 Days (LBank)")
    print(f"Total Trades: {n}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Loss Rate: {loss_rate:.2f}%")
    print(f"Net PnL: ${net_pnl:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    print(f"Maximum Consecutive Losses: {max_consecutive_losses}")
    print("-" * 72)
    
    if n > 0:
        print("LONG / SHORT BREAKDOWN:")
        for side in ["LONG", "SHORT"]:
            sub = trades_df[trades_df["Side"] == side]
            s_n = len(sub)
            s_wr = (sub["Outcome"].eq("WIN").mean() * 100) if s_n > 0 else 0.0
            s_pnl = sub["Dollar_PnL"].sum() if s_n > 0 else 0.0
            print(f"  {side} -> Trades: {s_n}, Win Rate: {s_wr:.2f}%, PnL: ${s_pnl:,.2f}")
        
        print("-" * 72)
        print("BY SYMBOL:")
        for sym in SYMBOLS.keys():
            sub = trades_df[trades_df["Symbol"] == sym]
            if len(sub) > 0:
                s_wr = sub["Outcome"].eq("WIN").mean() * 100
                s_pnl = sub["Dollar_PnL"].sum()
                print(f"  {sym:6} -> Trades: {len(sub):3}, Win Rate: {s_wr:5.2f}%, PnL: ${s_pnl:10,.2f}")

        print("-" * 72)
        print("BY MONTH:")
        for m, sub in trades_df.groupby("Month"):
            m_wr = sub["Outcome"].eq("WIN").mean() * 100
            m_pnl = sub["Dollar_PnL"].sum()
            print(f"  {m} -> Trades: {len(sub):3}, Win Rate: {m_wr:5.2f}%, PnL: ${m_pnl:10,.2f}")
            
        print("-" * 72)
        print(f"LOSS STREAKS DISTRIBUTION: {streaks}")
    print("=" * 72)
