import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

# ============================================================
# HUNTER-V69 (Golden Core + ADX Trend Strength Filter)
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True})

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

REMOVED_COINS = {
    "NEAR", "OP", "HYPE", "HBAR", "AVAX", "SUI", "PENDLE", "TIA",
    "FET", "SEI", "ARB", "DOT", "ETC", "SHIB", "STX", "RUNE",
    "MKR", "APT", "LTC", "AR", "IMX", "PEPE", "BONK",
}

SYMBOLS = {k: v for k, v in SYMBOLS.items() if k not in REMOVED_COINS}

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"
MAX_POSITIONS = 5
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
ATR_PERIOD = 14
TRAILING_ATR_MULTIPLIER = 2.0
INITIAL_ATR_MULTIPLIER = 1.8
TIMEOUT_CANDLES = 45
EMA_WARMUP = 200

# تنظیمات مالی
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 80.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 60)
print("📥 دریافت داده‌ها - HUNTER-V69 (هسته طلایی + فیلتر ADX)")
print("=" * 60)

processed_data = {}

def fetch_symbol_data(lbank_symbol):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000,
                )
                break
            except Exception as e:
                if attempt == 2:
                    return None

        if not batch:
            break

        first_ts = batch[0][0]
        last_ts = batch[-1][0]

        if last_seen is not None and last_ts <= last_seen:
            return None

        all_ohlcv.extend(batch)
        last_seen = last_ts
        current_since = last_ts + 1

        if len(batch) < 1000:
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(df.iloc[-1]["Date"].timestamp() * 1000)
        if last_ms + 4 * 60 * 60 * 1000 > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    deltas = df["Date"].diff().dropna()
    if not deltas.empty and deltas.max() > pd.Timedelta(hours=4, minutes=10):
        return None

    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))
    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()

    # محاسبه ADX (شاخص میانگین جهت‌دار) برای سنجش قدرت روند
    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    df["Plus_DM"] = pd.Series(plus_dm, index=df.index)
    df["Minus_DM"] = pd.Series(minus_dm, index=df.index)
    
    tr_smoothed = df["ATR"].rolling(14).mean()
    plus_di = 100 * (df["Plus_DM"].rolling(14).mean() / (tr_smoothed + 1e-9))
    minus_di = 100 * (df["Minus_DM"].rolling(14).mean() / (tr_smoothed + 1e-9))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df["ADX"] = dx.rolling(14).mean()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["Mom_Short"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
    df["Mom_Long"] = (df["Close"] - df["Close"].shift(30)) / df["Close"].shift(30)

    df.set_index("Date", inplace=True)
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h

print(f"✅ تعداد نمادهای معتبر: {len(processed_data)} از {len(SYMBOLS)}")
print("⚙️ شروع اجرای بک‌تست HUNTER-V69...")

def run_backtest(processed_data):
    all_timestamps = sorted({
        ts for df in processed_data.values() for ts in df.index
    })
    
    active_positions = {}
    all_trades = []
    
    for ts in all_timestamps:
        symbols_to_close = []
        
        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue
            
            c4h = df.loc[ts]
            
            if pos["side"] == "LONG":
                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_trailing_sl = pos["highest_price"] - TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_trailing_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
                hit_sl = c4h["Low"] <= pos["stop_loss"]
            else: # SHORT
                if c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]
                    new_trailing_sl = pos["lowest_price"] + TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_trailing_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
                hit_sl = c4h["High"] >= pos["stop_loss"]
            
            curr_i = df.index.get_loc(ts)
            candles_held = curr_i - pos["entry_index"]
            is_timeout = candles_held >= TIMEOUT_CANDLES
            
            if hit_sl or is_timeout:
                initial_risk = pos["initial_risk"]
                if pos["side"] == "LONG":
                    exit_p = min(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                    r_real = ((exit_p - pos["entry_price"]) / initial_risk) - (FEE_RATE * 2)
                    price_return_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
                else: # SHORT
                    exit_p = max(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                    r_real = ((pos["entry_price"] - exit_p) / initial_risk) - (FEE_RATE * 2)
                    price_return_pct = (pos["entry_price"] - exit_p) / pos["entry_price"]
                
                outcome = "WIN" if r_real > 0 else "LOSS"
                position_notional = TRADE_MARGIN * LEVERAGE
                dollar_pnl = (position_notional * price_return_pct) - (position_notional * FEE_RATE * 2)
                
                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Return": r_real,
                    "Dollar_PnL": dollar_pnl,
                    "ExitOrder": len(all_trades),
                })
                symbols_to_close.append(symbol)
        
        for sym in symbols_to_close:
            del active_positions[sym]
        
        market_bull = True
        if "BTC" in processed_data and ts in processed_data["BTC"].index:
            btc_c = processed_data["BTC"].loc[ts]
            market_bull = btc_c["Close"] > btc_c["EMA200"]
        
        current_scores = {}
        for symbol, df in processed_data.items():
            if ts in df.index:
                val = df.loc[ts, "Mom_Long"]
                if not np.isnan(val):
                    current_scores[symbol] = val
        
        if not current_scores:
            continue
        
        ranked_symbols = sorted(
            current_scores.keys(), key=lambda x: current_scores[x], reverse=market_bull
        )
        
        for symbol in ranked_symbols:
            if len(active_positions) >= MAX_POSITIONS:
                break
            if symbol in active_positions:
                continue
            
            df = processed_data[symbol]
            if ts not in df.index:
                continue
            
            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1:
                continue
            
            c4h = df.iloc[i]
            prev_c = df.iloc[i - 1]
            
            # فیلتر قدرت روند ADX: اگر روند ضعیف باشد (ADX < 20)، به هیچ وجه وارد معامله نمی‌شویم
            adx_val = c4h["ADX"]
            if np.isnan(adx_val) or adx_val < 20.0:
                continue

            if market_bull:
                regime_ok = (c4h["Close"] > c4h["EMA20"]) and (c4h["EMA20"] > c4h["EMA50"]) and (c4h["Close"] > c4h["EMA200"])
                pullback_ok = prev_c["Low"] <= prev_c["EMA20"] * 1.015
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] > 0.012) and (c4h["Mom_Long"] > 0.035)
                side = "LONG"
            else:
                regime_ok = (c4h["Close"] < c4h["EMA20"]) and (c4h["EMA20"] < c4h["EMA50"]) and (c4h["Close"] < c4h["EMA200"])
                pullback_ok = prev_c["High"] >= prev_c["EMA20"] * 0.985
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] < -0.012) and (c4h["Mom_Long"] < -0.035)
                side = "SHORT"
            
            if valid_signal:
                entry_price = c4h["Open"] * (1 + SLIPPAGE) if side == "LONG" else c4h["Open"] * (1 - SLIPPAGE)
                initial_sl = (entry_price - INITIAL_ATR_MULTIPLIER * c4h["ATR"]) if side == "LONG" else (entry_price + INITIAL_ATR_MULTIPLIER * c4h["ATR"])
                initial_risk = abs(entry_price - initial_sl)
                sl_dist_pct = initial_risk / entry_price
                
                if 0.01 <= sl_dist_pct <= 0.04:
                    active_positions[symbol] = {
                        "side": side,
                        "entry_price": entry_price,
                        "stop_loss": initial_sl,
                        "highest_price": entry_price,
                        "lowest_price": entry_price,
                        "initial_risk": initial_risk,
                        "entry_index": i,
                    }
                    
    return pd.DataFrame(all_trades)

def summarize_result(trades_df):
    print("\n" + "=" * 68)
    print("📊 گزارش نهایی استراتژی با فیلتر قدرت روند ADX - HUNTER-V69")
    print("=" * 68)

    if trades_df.empty:
        print("⚠️ هیچ معامله‌ای ثبت نشد.")
        return

    trades_df = trades_df.sort_values(["Timestamp", "ExitOrder"], kind="stable").reset_index(drop=True)

    total_trades = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())
    wr = (wins / total_trades * 100) if total_trades > 0 else 0
    net_r = float(trades_df["Return"].sum())
    
    total_dollar_pnl = float(trades_df["Dollar_PnL"].sum())
    final_capital = INITIAL_CAPITAL + total_dollar_pnl

    longs_df = trades_df[trades_df["Side"] == "LONG"]
    shorts_df = trades_df[trades_df["Side"] == "SHORT"]

    long_total = len(longs_df)
    long_wins = int((longs_df["Outcome"] == "WIN").sum()) if long_total > 0 else 0
    long_losses = int((longs_df["Outcome"] == "LOSS").sum()) if long_total > 0 else 0
    long_wr = (long_wins / long_total * 100) if long_total > 0 else 0

    short_total = len(shorts_df)
    short_wins = int((shorts_df["Outcome"] == "WIN").sum()) if short_total > 0 else 0
    short_losses = int((shorts_df["Outcome"] == "LOSS").sum()) if short_total > 0 else 0
    short_wr = (short_wins / short_total * 100) if short_total > 0 else 0

    max_losses = 0
    current_losses = 0
    loss_sequences = []
    temp_loss_seq = 0

    for outcome in trades_df["Outcome"]:
        if outcome == "WIN":
            current_losses = 0
            if temp_loss_seq > 0:
                loss_sequences.append(temp_loss_seq)
                temp_loss_seq = 0
        else:
            current_losses += 1
            temp_loss_seq += 1
            max_losses = max(max_losses, current_losses)

    if temp_loss_seq > 0:
        loss_sequences.append(temp_loss_seq)

    print(f"🔸 سرمایه اولیه: ${INITIAL_CAPITAL:,.2f}")
    print(f"🔸 مارجین: ${TRADE_MARGIN:,.2f} | لورج: {LEVERAGE}x")
    print(f"🔸 تعداد کل معاملات: {total_trades}")
    print(f"   🔹 معاملات لانگ: کل = {long_total} | برنده = {long_wins} | بازنده = {long_losses} | وین‌ریت = {long_wr:.2f}%")
    print(f"   🔸 معاملات شورت: کل = {short_total} | برنده = {short_wins} | بازنده = {short_losses} | وین‌ریت = {short_wr:.2f}%")
    print(f"🎯 وین‌ریت کلی کل سبد: {wr:.2f}%")
    print(f"💰 مجموع بازدهی خالص: {net_r:.2f}R")
    print(f"💵 مجموع سود/زیان دلاری خالص: ${total_dollar_pnl:,.2f}")
    print(f"🏦 سرمایه نهایی: ${final_capital:,.2f}")
    print(f"❄️ حداکثر ضررهای متوالی کل سبد: {max_losses}")

    print("\n------------------------------------------------------------")
    print("📉 لیست کامل زنجیره‌های ضرر متوالی:")
    print("------------------------------------------------------------")
    if loss_sequences:
        print(", ".join(map(str, loss_sequences)))
    else:
        print("هیچ زنجیره ضرری ثبت نشد.")

if __name__ == "__main__":
    df_trades = run_backtest(processed_data)
    summarize_result(df_trades)
    print("\n✨ بک‌تست HUNTER-V69 به پایان رسید.")
