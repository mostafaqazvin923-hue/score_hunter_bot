from dataclasses import dataclass
from typing import Optional, Dict, Any
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
# تنظیمات اصلی LBank - سیستم چندتایم‌فریمی (4h Trend + 1h Execution)
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
    "NEAR": "NEAR/USDT",
    "OP": "OP/USDT",
    "HBAR": "HBAR/USDT",
    "AVAX": "AVAX/USDT",
    "SUI": "SUI/USDT",
    "TIA": "TIA/USDT",
    "FET": "FET/USDT",
    "SEI": "SEI/USDT",
    "ARB": "ARB/USDT",
    "DOT": "DOT/USDT",
    "ETC": "ETC/USDT",
    "SHIB": "SHIB/USDT",
    "STX": "STX/USDT",
    "APT": "APT/USDT",
    "LTC": "LTC/USDT",
    "AR": "AR/USDT",
    "IMX": "IMX/USDT",
    "PEPE": "PEPE/USDT",
    "BONK": "BONK/USDT",
}

LOOKBACK_DAYS = 365
MAX_POSITIONS = 8
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
ATR_PERIOD = 14
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 60)
print("📥 دریافت داده‌های چندتایم‌فریمی (1h و 4h) از صرافی LBank")
print("=" * 60)

processed_data_1h = {}
processed_data_4h = {}

def fetch_data(lbank_symbol, timeframe):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000,
                )
                break
            except Exception:
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

    if len(df) < 50:
        return None

    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))
    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df.set_index("Date", inplace=True)
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    df1h = fetch_data(lbank_symbol, "1h")
    df4h = fetch_data(lbank_symbol, "4h")
    
    if df1h is not None and df4h is not None:
        processed_data_1h[symbol] = df1h
        processed_data_4h[symbol] = df4h

print(f"✅ تعداد نمادهای بارگذاری شده: {len(processed_data_1h)} نماد")
print("⚙️ شروع اجرای بک‌تست Multi-Timeframe (روند 4h + ورود 1h)...")


# ============================================================
# موتور معاملاتی چندتایم‌فریمی
# ============================================================

def run_multi_timeframe_backtest(data_1h, data_4h):
    all_timestamps = sorted({
        ts for df in data_1h.values() for ts in df.index
    })
    
    active_positions = {}
    all_trades = []
    equity = INITIAL_CAPITAL
    consecutive_losses = 0
    trading_paused = False
    pause_timer = 0

    for ts in all_timestamps:
        if trading_paused:
            pause_timer -= 1
            if pause_timer <= 0:
                trading_paused = False
                consecutive_losses = 0
            continue

        symbols_to_close = []
        
        # 1. مدیریت پوزیشن‌های باز روی تایم‌فریم 1 ساعته
        for symbol, pos in list(active_positions.items()):
            df1 = data_1h[symbol]
            if ts not in df1.index:
                continue
            
            c1h = df1.loc[ts]
            hit_stop = False
            hit_target = False

            if pos["side"] == "LONG":
                hit_stop = c1h["Low"] <= pos["stop_price"]
                hit_target = c1h["High"] >= pos["target_price"]
            else:
                hit_stop = c1h["High"] >= pos["stop_price"]
                hit_target = c1h["Low"] <= pos["target_price"]

            if hit_stop or hit_target:
                is_win = hit_target
                notional = TRADE_MARGIN * LEVERAGE
                fee_cost = notional * FEE_RATE * 2.0
                
                if is_win:
                    pnl_amount = abs(pos["target_price"] - pos["entry_price"]) * pos["size"]
                else:
                    pnl_amount = -abs(pos["entry_price"] - pos["stop_price"]) * pos["size"]
                
                net_pnl = pnl_amount - fee_cost
                equity += net_pnl
                
                outcome = "WIN" if is_win else "LOSS"
                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Net_PnL": net_pnl,
                })
                
                if not is_win:
                    consecutive_losses += 1
                    if consecutive_losses >= 4:
                        trading_paused = True
                        pause_timer = 12 # مکث طولانی‌تر در تایم 1h
                else:
                    consecutive_losses = 0

                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        if len(active_positions) >= MAX_POSITIONS or equity < TRADE_MARGIN:
            continue

        # 2. بررسی سیگنال ورود روی هر نماد
        for symbol, df1 in data_1h.items():
            if symbol in active_positions or ts not in df1.index:
                continue

            i1 = df1.index.get_loc(ts)
            if i1 < 30:
                continue

            subset_1h = df1.iloc[:i1 + 1]
            c1h = subset_1h.iloc[-1]
            
            # --- دریافت رژیم روند از تایم‌فریم 4 ساعته بدون نگاه به آینده ---
            df4 = data_4h[symbol]
            # پیدا کردن آخرین کندل 4 ساعته که قبل از زمان ts بسته شده است
            past_4h = df4[df4.index <= ts]
            if len(past_4h) < 30:
                continue
            
            c4h = past_4h.iloc[-1]
            recent_highs_4h = past_4h["High"].rolling(10).max().iloc[-1]
            recent_lows_4h = past_4h["Low"].rolling(10).min().iloc[-1]

            regime = "Neutral"
            if c4h["Close"] > c4h["EMA50"] and c4h["Close"] >= recent_highs_4h * 0.99:
                regime = "Bull"
            elif c4h["Close"] < c4h["EMA50"] and c4h["Close"] <= recent_lows_4h * 1.01:
                regime = "Bear"

            if regime == "Neutral":
                continue

            score = 0
            direction = "LONG" if regime == "Bull" else "SHORT"

            # 1. رژیم روند کلان 4h -> 3 امتیاز
            score += 3

            # 2. سویپ نقدینگی در تایم‌فریم 1h -> 3 امتیاز
            lookback_liq = 20
            if len(subset_1h) > lookback_liq:
                recent_high_1h = subset_1h["High"].iloc[-lookback_liq:-1].max()
                recent_low_1h = subset_1h["Low"].iloc[-lookback_liq:-1].min()

                if direction == "LONG" and c1h["Low"] < recent_low_1h and c1h["Close"] > recent_low_1h:
                    score += 3
                elif direction == "SHORT" and c1h["High"] > recent_high_1h and c1h["Close"] < recent_high_1h:
                    score += 3

            # 3. شتاب در 1h -> 2 امتیاز
            atr_1h = c1h["ATR"]
            if not np.isfinite(atr_1h) or atr_1h <= 0:
                continue
                
            candle_range = c1h["High"] - c1h["Low"]
            if candle_range > (1.05 * atr_1h):
                score += 2

            # 4. حجم در 1h -> 2 امتیاز
            vol_mean_1h = subset_1h["Volume"].rolling(14).mean().iloc[-1]
            if np.isfinite(vol_mean_1h) and c1h["Volume"] > (1.05 * vol_mean_1h):
                score += 2

            if score < 4:
                continue

            entry_price = c1h["Open"] * (1 + SLIPPAGE) if direction == "LONG" else c1h["Open"] * (1 - SLIPPAGE)
            
            if direction == "LONG":
                stop_price = recent_low_1h - (atr_1h * 0.5)
                if entry_price - stop_price <= 0:
                    stop_price = entry_price - atr_1h
                risk_dist = entry_price - stop_price
                target_price = entry_price + (risk_dist * 2.0)
            else:
                stop_price = recent_high_1h + (atr_1h * 0.5)
                if stop_price - entry_price <= 0:
                    stop_price = entry_price + atr_1h
                risk_dist = stop_price - entry_price
                target_price = entry_price - (risk_dist * 2.0)

            notional_value = TRADE_MARGIN * LEVERAGE
            size = notional_value / entry_price

            active_positions[symbol] = {
                "side": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "size": size,
            }

            if len(active_positions) >= MAX_POSITIONS:
                break

    return pd.DataFrame(all_trades), equity


def summarize_result(trades_df, final_equity):
    print("\n" + "=" * 68)
    print("📊 گزارش نهایی استراتژی چندتایم‌فریمی (Trend 4h + Entry 1h)")
    print("=" * 68)

    if trades_df.empty:
        print("⚠️ هیچ معامله‌ای ثبت نشد.")
        return

    trades_df = trades_df.sort_values(["Timestamp"], kind="stable").reset_index(drop=True)

    total_trades = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())
    wr = (wins / total_trades * 100) if total_trades > 0 else 0
    total_dollar_pnl = float(trades_df["Net_PnL"].sum())

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

    longs_df = trades_df[trades_df["Side"] == "LONG"]
    shorts_df = trades_df[trades_df["Side"] == "SHORT"]

    print(f"🔸 سرمایه اولیه: ${INITIAL_CAPITAL:,.2f}")
    print(f"🔸 مارجین: ${TRADE_MARGIN:,.2f} | لورج: {LEVERAGE}x")
    print(f"🔸 تعداد کل معاملات: {total_trades}")
    print(f"   🔹 معاملات لانگ: کل = {len(longs_df)} | برنده = {int((longs_df['Outcome'] == 'WIN').sum())} | بازنده = {int((longs_df['Outcome'] == 'LOSS').sum())}")
    print(f"   🔸 معاملات شورت: کل = {len(shorts_df)} | برنده = {int((shorts_df['Outcome'] == 'WIN').sum())} | بازنده = {int((shorts_df['Outcome'] == 'LOSS').sum())}")
    print(f"🎯 وین‌ریت کلی: {wr:.2f}%")
    print(f"💵 مجموع سود/زیان دلاری خالص: ${total_dollar_pnl:,.2f}")
    print(f"🏦 سرمایه نهایی: ${final_equity:,.2f}")
    print(f"❄️ حداکثر ضررهای متوالی کل سبد: {max_losses}")

    print("\n------------------------------------------------------------")
    print("📉 لیست کامل زنجیره‌های ضرر متوالی:")
    print("------------------------------------------------------------")
    if loss_sequences:
        print(", ".join(map(str, loss_sequences)))
    else:
        print("هیچ زنجیره ضرری ثبت نشد.")
    print("=" * 68)


if __name__ == "__main__":
    df_trades, final_equity = run_multi_timeframe_backtest(processed_data_1h, processed_data_4h)
    summarize_result(df_trades, final_equity)
    print("\n✨ بک‌تست سیستم چندتایم‌فریمی به پایان رسید.")
