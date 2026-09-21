import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

# ============================================================
# HUNTER-V123 — 15 SYMBOLS (INCLUDING SOL, ETH, XRP) 1-YEAR BACKTEST
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True, "timeout": 20000})

# لیست ۱۵ تایی (۱۲ نماد قبل + سولانا، اتریوم و ریپل)
SYMBOLS = {
    "CRV": "CRV/USDT",
    "DOGE": "DOGE/USDT",
    "ICP": "ICP/USDT",
    "APT": "APT/USDT",
    "PENDLE": "PENDLE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "NEAR": "NEAR/USDT",
    "SEI": "SEI/USDT",
    "XLM": "XLM/USDT",
    "ADA": "ADA/USDT",
    "BNB": "BNB/USDT",
    "SOL": "SOL/USDT",
    "ETH": "ETH/USDT",
    "XRP": "XRP/USDT"
}

TIMEFRAME_BASE = "15m"
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

# تعریف ۴ پارت زمانی (از امروز تا ۳۶۵ روز گذشته)
QUARTERS = [
    {"name": "Q1 (Recent 90 Days)", "start_days_ago": 90, "end_days_ago": 0},
    {"name": "Q2 (90 to 180 Days)", "start_days_ago": 180, "end_days_ago": 90},
    {"name": "Q3 (180 to 270 Days)", "start_days_ago": 270, "end_days_ago": 180},
    {"name": "Q4 (270 to 365 Days)", "start_days_ago": 365, "end_days_ago": 270},
]

def fetch_chunk_data(lbank_symbol, start_dt, end_dt):
    since_ts = int((start_dt - timedelta(days=15)).timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    all_ohlcv = []
    current_since = since_ts
    try:
        while current_since < end_ts:
            batch = exchange.fetch_ohlcv(lbank_symbol, timeframe=TIMEFRAME_BASE, since=current_since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= current_since:
                break
            current_since = last_ts + 1
            if len(batch) < 1000 or last_ts >= end_ts:
                break
            time.sleep(0.2)
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
    
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df

def run_backtest_on_data(processed_data):
    all_trades = []

    for symbol, dsets in processed_data.items():
        df_15 = dsets["15m"]
        df_1h = dsets["1h"]
        df_4h = dsets["4h"]

        cooldown_bars = 0
        consecutive_losses = 0

        for i in range(50, len(df_15)):
            if cooldown_bars > 0:
                cooldown_bars -= 1
                continue

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

            displacement_up = (c_row["Close"] > c_row["Open"]) and (c_row["Body"] > 2.0 * c_row["Avg_Body"])
            displacement_down = (c_row["Close"] < c_row["Open"]) and (c_row["Body"] > 2.0 * c_row["Avg_Body"])

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
                be_trigger = entry_price + (1.5 * atr)
            else:
                sl = entry_price + (1.5 * atr)
                tp = entry_price - (3.0 * atr)
                be_trigger = entry_price - (1.5 * atr)

            outcome = "LOSS"
            exit_price = sl
            current_sl = sl
            breakeven_activated = False

            for j in range(i + 1, min(i + 35, len(df_15))):
                fut = df_15.iloc[j]
                if side == "LONG":
                    if not breakeven_activated and fut["High"] >= be_trigger:
                        current_sl = entry_price
                        breakeven_activated = True

                    hit_sl = fut["Low"] <= current_sl
                    hit_tp = fut["High"] >= tp

                    if hit_sl and hit_tp:
                        outcome = "LOSS" if current_sl != entry_price else "BE"
                        exit_price = current_sl if current_sl != entry_price else entry_price
                        break
                    elif hit_sl:
                        outcome = "LOSS" if current_sl != entry_price else "BE"
                        exit_price = current_sl
                        break
                    elif hit_tp:
                        outcome = "WIN"
                        exit_price = tp
                        break
                else:
                    if not breakeven_activated and fut["Low"] <= be_trigger:
                        current_sl = entry_price
                        breakeven_activated = True

                    hit_sl = fut["High"] >= current_sl
                    hit_tp = fut["Low"] <= tp

                    if hit_sl and hit_tp:
                        outcome = "LOSS" if current_sl != entry_price else "BE"
                        exit_price = current_sl if current_sl != entry_price else entry_price
                        break
                    elif hit_sl:
                        outcome = "LOSS" if current_sl != entry_price else "BE"
                        exit_price = current_sl
                        break
                    elif hit_tp:
                        outcome = "WIN"
                        exit_price = tp
                        break

            if outcome == "LOSS":
                consecutive_losses += 1
                if consecutive_losses >= 2:
                    cooldown_bars = 16
                    consecutive_losses = 0
            elif outcome == "WIN":
                consecutive_losses = 0

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
    print("=" * 72)
    print("HUNTER-V123 — 15 SYMBOLS (SOL, ETH, XRP INCLUDED) BACKTEST INITIALIZED")
    print("=" * 72)

    all_quarter_trades = []
    now = datetime.now()

    for q in QUARTERS:
        start_dt = now - timedelta(days=q["start_days_ago"])
        end_dt = now - timedelta(days=q["end_days_ago"])
        print(f"\nProcessing {q['name']} ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})...")

        processed_data = {}
        for symbol, lbank_symbol in SYMBOLS.items():
            fetch_start = start_dt - timedelta(days=35)
            df_15m = fetch_chunk_data(lbank_symbol, fetch_start, end_dt)

            if df_15m is None or len(df_15m) < 200:
                continue

            df_1h = df_15m.resample('1h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

            df_4h = df_15m.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

            df_4h["EMA_50"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
            df_4h["EMA_200"] = df_4h["Close"].ewm(span=200, adjust=False).mean()
            df_4h["Regime_Bullish"] = (df_4h["Close"] > df_4h["EMA_200"]) & (df_4h["EMA_50"] > df_4h["EMA_200"])
            df_4h["Regime_Bearish"] = (df_4h["Close"] < df_4h["EMA_200"]) & (df_4h["EMA_50"] < df_4h["EMA_200"])

            df_1h["Swing_High"] = df_1h["High"].rolling(5, center=True).max()
            df_1h["Swing_Low"] = df_1h["Low"].rolling(5, center=True).min()
            df_1h["ATR"] = (df_1h["High"] - df_1h["Low"]).rolling(14).mean()

            df_15m["ATR"] = (df_15m["High"] - df_15m["Low"]).rolling(14).mean()
            df_15m["Body"] = (df_15m["Close"] - df_15m["Open"]).abs()
            df_15m["Avg_Body"] = df_15m["Body"].rolling(20).mean()

            df_15m = df_15m[(df_15m.index >= start_dt) & (df_15m.index <= end_dt)]

            if len(df_15m) > 50:
                processed_data[symbol] = {
                    "15m": df_15m,
                    "1h": df_1h,
                    "4h": df_4h
                }

        q_trades_df = run_backtest_on_data(processed_data)
        if not q_trades_df.empty:
            all_quarter_trades.append(q_trades_df)
            print(f" -> {q['name']} Done. Trades found: {len(q_trades_df)}")
        else:
            print(f" -> {q['name']} Produced 0 trades.")

    if all_quarter_trades:
        trades_df = pd.concat(all_quarter_trades, ignore_index=True)
        trades_df.sort_values("Timestamp", inplace=True)
        trades_df.reset_index(drop=True, inplace=True)

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

        print("\n" + "=" * 72)
        print("===== 15 SYMBOLS — AGGREGATED 1-YEAR (365 DAYS) BACKTEST RESULT =====")
        print(f"Total Trades (Full Year): {n}")
        print(f"Trades Per Month (Avg): {n / 12.0:.1f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Loss Rate: {loss_rate:.2f}%")
        print(f"Net PnL: ${net_pnl:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Max Drawdown: ${max_dd:,.2f}")
        print(f"Maximum Consecutive Losses: {max_consecutive_losses}")
        print("-" * 72)
        
        print("BY SYMBOL (15 GIANTS):")
        for sym in list(SYMBOLS.keys()):
            sub = trades_df[trades_df["Symbol"] == sym]
            if len(sub) > 0:
                s_wr = sub["Outcome"].eq("WIN").mean() * 100
                s_pnl = sub["Dollar_PnL"].sum()
                print(f"  {sym:6} -> Trades: {len(sub):3}, Win Rate: {s_wr:5.2f}%, PnL: ${s_pnl:10,.2f}")

        print("-" * 72)
        print("BY MONTH (15 GIANTS):")
        for m, sub in trades_df.groupby("Month"):
            m_wr = sub["Outcome"].eq("WIN").mean() * 100
            m_pnl = sub["Dollar_PnL"].sum()
            print(f"  {m} -> Trades: {len(sub):3}, Win Rate: {m_wr:5.2f}%, PnL: ${m_pnl:10,.2f}")
        print("=" * 72)
    else:
        print("No trades generated across quarters.")
