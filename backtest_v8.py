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
# تنظیمات حرفه‌ای و پارامتریک سیستم (Professional Config)
# ============================================================

EXCHANGE_ID = "lbank"
TIMEFRAME = "4h"
LOOKBACK_DAYS = 365
MAX_POSITIONS = 8
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
ATR_PERIOD = 14
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0  # ثابت و بهینه
LEVERAGE = 50.0       # ثابت و بهینه

exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})

SYMBOLS = {
    "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT",
    "XRP": "XRP/USDT", "LINK": "LINK/USDT", "UNI": "UNI/USDT",
    "ICP": "ICP/USDT", "INJ": "INJ/USDT", "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT", "XLM": "XLM/USDT", "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT", "ONDO": "ONDO/USDT", "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT", "ADA": "ADA/USDT", "NEAR": "NEAR/USDT",
    "OP": "OP/USDT", "HBAR": "HBAR/USDT", "AVAX": "AVAX/USDT",
    "SUI": "SUI/USDT", "TIA": "TIA/USDT", "FET": "FET/USDT",
    "SEI": "SEI/USDT", "ARB": "ARB/USDT", "DOT": "DOT/USDT",
    "ETC": "ETC/USDT", "SHIB": "SHIB/USDT", "STX": "STX/USDT",
    "APT": "APT/USDT", "LTC": "LTC/USDT", "AR": "AR/USDT",
    "IMX": "IMX/USDT", "PEPE": "PEPE/USDT", "BONK": "BONK/USDT",
}

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 60)
print(f"📥 دریافت داده‌های ساختاریافته {TIMEFRAME} از صرافی {EXCHANGE_ID.upper()}")
print("=" * 60)

processed_data = {}

def fetch_symbol_data(lbank_symbol: str) -> Optional[pd.DataFrame]:
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
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h

print(f"✅ تعداد نمادهای بارگذاری شده معتبر: {len(processed_data)} نماد")
print("⚙️ شروع اجرای موتور بک‌تست حرفه‌ای (Break-Even + کنترل استریک سخت‌گیرانه)...")


# ============================================================
# موتور اجرایی حرفه‌ای
# ============================================================

def run_professional_backtest(data_dict: Dict[str, pd.DataFrame]):
    all_timestamps = sorted({
        ts for df in data_dict.values() for ts in df.index
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
        
        # مدیریت پوزیشن‌های باز و Break-Even
        for symbol, pos in list(active_positions.items()):
            df = data_dict[symbol]
            if ts not in df.index:
                continue
            
            c4h = df.loc[ts]
            
            # منطق Break-Even در ۵۰ درصد مسیر
            if not pos["is_breakeven"]:
                if pos["side"] == "LONG":
                    halfway = pos["entry_price"] + (pos["target_price"] - pos["entry_price"]) * 0.5
                    if c4h["High"] >= halfway:
                        pos["stop_price"] = pos["entry_price"]
                        pos["is_breakeven"] = True
                else:
                    halfway = pos["entry_price"] - (pos["entry_price"] - pos["target_price"]) * 0.5
                    if c4h["Low"] <= halfway:
                        pos["stop_price"] = pos["entry_price"]
                        pos["is_breakeven"] = True

            hit_stop = False
            hit_target = False

            if pos["side"] == "LONG":
                hit_stop = c4h["Low"] <= pos["stop_price"]
                hit_target = c4h["High"] >= pos["target_price"]
            else:
                hit_stop = c4h["High"] >= pos["stop_price"]
                hit_target = c4h["Low"] <= pos["target_price"]

            if hit_stop or hit_target:
                is_win = hit_target
                is_be = (not is_win) and pos["is_breakeven"] and (pos["stop_price"] == pos["entry_price"])
                
                notional = TRADE_MARGIN * LEVERAGE
                fee_cost = notional * FEE_RATE * 2.0
                
                if is_win:
                    pnl_amount = abs(pos["target_price"] - pos["entry_price"]) * pos["size"]
                    outcome = "WIN"
                elif is_be:
                    pnl_amount = 0.0
                    outcome = "BREAK_EVEN"
                else:
                    pnl_amount = -abs(pos["entry_price"] - pos["stop_price"]) * pos["size"]
                    outcome = "LOSS"
                
                net_pnl = pnl_amount - fee_cost
                equity += net_pnl
                
                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Net_PnL": net_pnl,
                })
                
                # کنترل استریک روی ۳ باخت متوالی برای حفظ سقف ۳ یا ۴
                if outcome == "LOSS":
                    consecutive_losses += 1
                    if consecutive_losses >= 3:
                        trading_paused = True
                        pause_timer = 8  # استراحت بهینه برای عبور از بازار رنک و پرنویز
                elif outcome == "WIN":
                    consecutive_losses = 0

                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        if len(active_positions) >= MAX_POSITIONS or equity < TRADE_MARGIN:
            continue

        # بررسی سیگنال ورود با بهینه‌سازی فرکانس
        for symbol, df in data_dict.items():
            if symbol in active_positions or ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < 30:
                continue

            subset = df.iloc[:i + 1]
            current_candle = subset.iloc[-1]
            close = current_candle["Close"]
            ema_50 = current_candle["EMA50"]
            atr = current_candle["ATR"]

            if not np.isfinite(atr) or atr <= 0:
                continue

            recent_highs = subset["High"].rolling(10).max().iloc[-1]
            recent_lows = subset["Low"].rolling(10).min().iloc[-1]

            regime = "Neutral"
            if close > ema_50 and close >= recent_highs * 0.99:
                regime = "Bull"
            elif close < ema_50 and close <= recent_lows * 1.01:
                regime = "Bear"

            if regime == "Neutral":
                continue

            score = 0
            direction = "LONG" if regime == "Bull" else "SHORT"

            # 1. رژیم بازار -> 3 امتیاز
            score += 3

            # 2. نقدینگی و Sweep -> 3 امتیاز
            lookback_liq = 15
            if len(subset) > lookback_liq:
                recent_high = subset["High"].iloc[-lookback_liq:-1].max()
                recent_low = subset["Low"].iloc[-lookback_liq:-1].min()

                if direction == "LONG" and current_candle["Low"] < recent_low and current_candle["Close"] > recent_low:
                    score += 3
                elif direction == "SHORT" and current_candle["High"] > recent_high and current_candle["Close"] < recent_high:
                    score += 3

            # 3. شتاب -> 2 امتیاز (بهینه‌شده برای افزایش جزئی معاملات)
            candle_range = current_candle["High"] - current_candle["Low"]
            if candle_range > (0.95 * atr):
                score += 2

            # 4. حجم -> 2 امتیاز
            vol_mean = subset["Volume"].rolling(14).mean().iloc[-1]
            if np.isfinite(vol_mean) and current_candle["Volume"] > (0.95 * vol_mean):
                score += 2

            if score < 4:
                continue

            entry_price = current_candle["Open"] * (1 + SLIPPAGE) if direction == "LONG" else current_candle["Open"] * (1 - SLIPPAGE)
            
            if direction == "LONG":
                stop_price = recent_low - (atr * 0.5)
                if entry_price - stop_price <= 0:
                    stop_price = entry_price - atr
                risk_dist = entry_price - stop_price
                target_price = entry_price + (risk_dist * 2.0)
            else:
                stop_price = recent_high + (atr * 0.5)
                if stop_price - entry_price <= 0:
                    stop_price = entry_price + atr
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
                "is_breakeven": False,
            }

            if len(active_positions) >= MAX_POSITIONS:
                break

    return pd.DataFrame(all_trades), equity


def summarize_results(trades_df: pd.DataFrame, final_equity: float):
    print("\n" + "=" * 68)
    print("📊 گزارش حرفه‌ای بک‌تست 4h (کنترل استریک روی ۳ + Break-Even)")
    print("=" * 68)

    if trades_df.empty:
        print("⚠️ هیچ معامله‌ای ثبت نشد.")
        return

    trades_df = trades_df.sort_values(["Timestamp"], kind="stable").reset_index(drop=True)

    total_trades = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())
    breakevens = int((trades_df["Outcome"] == "BREAK_EVEN").sum())
    
    decisive_trades = wins + losses
    wr = (wins / decisive_trades * 100) if decisive_trades > 0 else 0
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
        elif outcome == "LOSS":
            current_losses += 1
            temp_loss_seq += 1
            max_losses = max(max_losses, current_losses)

    if temp_loss_seq > 0:
        loss_sequences.append(temp_loss_seq)

    longs_df = trades_df[trades_df["Side"] == "LONG"]
    shorts_df = trades_df[trades_df["Side"] == "SHORT"]

    print(f"🔸 سرمایه اولیه: ${INITIAL_CAPITAL:,.2f}")
    print(f"🔸 مارجین: ${TRADE_MARGIN:,.2f} | لورج: {LEVERAGE}x")
    print(f"🔸 تعداد کل معاملات: {total_trades} (برنده: {wins} | بازنده: {losses} | سر‌به‌سر: {breakevens})")
    print(f"   🔹 معاملات لانگ: کل = {len(longs_df)}")
    print(f"   🔸 معاملات شورت: کل = {len(shorts_df)}")
    print(f"🎯 وین‌ریت واقعی: {wr:.2f}%")
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
    df_trades, final_equity = run_professional_backtest(processed_data)
    summarize_results(df_trades, final_equity)
    print("\n✨ بک‌تست حرفه‌ای به پایان رسید.")
