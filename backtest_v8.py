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
# تنظیمات سیستماتیک و پیشرفته (Institutional Setup Config)
# ============================================================

EXCHANGE_ID = "lbank"
TIMEFRAME = "4h"
LOOKBACK_DAYS = 365
MAX_POSITIONS = 6
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

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
print(f"📥 دریافت و آماده‌سازی داده‌های ساختاریافته {TIMEFRAME} از صرافی {EXCHANGE_ID.upper()}")
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

    # اندیکاتورهای پیشرفته کمی (Statistical & Quantitative Indicators)
    period = 20
    df["EMA_20"] = df["Close"].ewm(span=period, adjust=False).mean()
    rolling_std = df["Close"].rolling(window=period).std()
    
    # محاسبه Z-Score قیمت
    df["Z_Score"] = (df["Close"] - df["EMA_20"]) / (rolling_std + 1e-9)
    
    # محاسبه ATR و رژیم نوسانی
    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))
    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df["ATR_MA"] = df["ATR"].rolling(20).mean()

    df.set_index("Date", inplace=True)
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h

print(f"✅ تعداد نمادهای معتبر تاییدشده: {len(processed_data)}")
print("⚙️ اجرای موتور کوانت پیشرفته (بدون Lookahead و با مدیریت کامل پوزیشن‌ها)...")


# ============================================================
# موتور اجرای استراتژی کوانت (Institutional Backtest Engine)
# ============================================================

def run_institutional_backtest(data_dict: Dict[str, pd.DataFrame]):
    all_timestamps = sorted({
        ts for df in data_dict.values() for ts in df.index
    })
    
    active_positions = {}
    pending_signals = []  # سیگنال‌های کندل N برای اجرا در Open کندل N+1
    all_trades = []
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    
    consecutive_losses = 0
    trading_paused = False
    pause_timer = 0

    for ts in all_timestamps:
        # ۱. مدیریت تایمر مکث (فقط مانع ورودهای جدید می‌شود و کاری به مدیریت پوزیشن‌های باز ندارد)
        if trading_paused:
            pause_timer -= 1
            if pause_timer <= 0:
                trading_paused = False
                consecutive_losses = 0

        # ۲. اجرای سیگنال‌های کندل قبل در Open کندل جاری (بدون Lookahead)
        signals_to_execute = pending_signals
        pending_signals = []

        for sig in signals_to_execute:
            symbol = sig["symbol"]
            if symbol in active_positions or len(active_positions) >= MAX_POSITIONS or equity < TRADE_MARGIN:
                continue
            
            df = data_dict[symbol]
            if ts not in df.index:
                continue
            
            c4h = df.loc[ts]
            open_price = c4h["Open"]
            entry_price = open_price * (1 + SLIPPAGE) if sig["side"] == "LONG" else open_price * (1 - SLIPPAGE)
            
            notional_value = TRADE_MARGIN * LEVERAGE
            size = notional_value / entry_price

            active_positions[symbol] = {
                "side": sig["side"],
                "entry_price": entry_price,
                "stop_price": sig["stop_price"],
                "target_price": sig["target_price"],
                "size": size,
                "is_breakeven": False,
            }

        # ۳. مدیریت کامل پوزیشن‌های باز (همیشه و حتی زمان Pause اجرا می‌شود)
        symbols_to_close = []
        for symbol, pos in list(active_positions.items()):
            df = data_dict[symbol]
            if ts not in df.index:
                continue
            
            c4h = df.loc[ts]
            
            # مدیریت Break-Even محافظه‌کارانه در ۵۰ درصد مسیر
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
                # اگر همزمان استاپ و تارجت زده شد، حالت محافظه‌کارانه (استاپ) اعمال می‌شود
                is_win = hit_target and not hit_stop
                is_be = (not is_win) and pos["is_breakeven"] and (pos["stop_price"] == pos["entry_price"])
                
                notional = TRADE_MARGIN * LEVERAGE
                fee_cost = notional * FEE_RATE * 2.0
                exit_slippage_cost = notional * SLIPPAGE
                
                if is_win:
                    pnl_amount = abs(pos["target_price"] - pos["entry_price"]) * pos["size"]
                    outcome = "WIN"
                elif is_be:
                    pnl_amount = 0.0
                    outcome = "BREAK_EVEN"
                else:
                    pnl_amount = -abs(pos["entry_price"] - pos["stop_price"]) * pos["size"]
                    outcome = "LOSS"
                
                net_pnl = pnl_amount - fee_cost - exit_slippage_cost
                equity += net_pnl
                
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                max_drawdown = max(max_drawdown, dd)
                
                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Net_PnL": net_pnl,
                })
                
                # کنترل استریک ضد ضرر متوالی
                if outcome == "LOSS":
                    consecutive_losses += 1
                    if consecutive_losses >= 3 and not trading_paused:
                        trading_paused = True
                        pause_timer = 6  # استراحت بهینه
                elif outcome == "WIN":
                    consecutive_losses = 0

                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        # ۴. اسکن سیگنال‌های جدید بر اساس Z-Score و رژیم نوسانی (برای ثبت در صف کندل بعدی)
        if not trading_paused:
            for symbol, df in data_dict.items():
                if symbol in active_positions or ts not in df.index:
                    continue

                i = df.index.get_loc(ts)
                if i < 30:
                    continue

                subset = df.iloc[:i + 1]
                current = subset.iloc[-1]
                
                z_score = current["Z_Score"]
                atr = current["ATR"]
                atr_ma = current["ATR_MA"]

                if not np.isfinite(z_score) or not np.isfinite(atr) or atr <= 0:
                    continue

                # فیلتر رژیم بازار: نوسان باید بالاتر از میانگین باشد (بازار پویا، نه رِنج مرده)
                if atr < atr_ma * 0.9:
                    continue

                direction = None
                # استراتژی بازگشت آماری به میانگین (Mean Reversion) با انحراف شدید Z-Score
                if z_score <= -2.0:  # اشباع فروش شدید -> پتانسیل لانگ
                    direction = "LONG"
                elif z_score >= 2.0:  # اشباع خرید شدید -> پتانسیل شورت
                    direction = "SHORT"

                if direction is None:
                    continue

                # تاییدیه حجم
                vol_mean = subset["Volume"].rolling(14).mean().iloc[-1]
                if np.isfinite(vol_mean) and current["Volume"] < vol_mean * 0.9:
                    continue

                # محاسبات ریسک پویا با ATR
                if direction == "LONG":
                    stop_price = current["Close"] - (atr * 1.5)
                    risk_dist = current["Close"] - stop_price
                    target_price = current["Close"] + (risk_dist * 2.0)
                else:
                    stop_price = current["Close"] + (atr * 1.5)
                    risk_dist = stop_price - current["Close"]
                    target_price = current["Close"] - (risk_dist * 2.0)

                pending_signals.append({
                    "symbol": symbol,
                    "side": direction,
                    "stop_price": stop_price,
                    "target_price": target_price,
                })

    return pd.DataFrame(all_trades), equity, max_drawdown


def summarize_institutional_results(trades_df: pd.DataFrame, final_equity: float, max_dd: float):
    print("\n" + "=" * 68)
    print("📊 گزارش نهایی استراتژی کوانت جدید (بدون باگ و کاملاً ایزوله)")
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
    decisive_wr = (wins / decisive_trades * 100) if decisive_trades > 0 else 0
    total_wr = (wins / total_trades * 100) if total_trades > 0 else 0
    be_rate = (breakevens / total_trades * 100) if total_trades > 0 else 0
    
    total_dollar_pnl = float(trades_df["Net_PnL"].sum())
    
    gross_profits = trades_df[trades_df["Net_PnL"] > 0]["Net_PnL"].sum()
    gross_losses = abs(trades_df[trades_df["Net_PnL"] < 0]["Net_PnL"].sum())
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else 0.0

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
    print(f"🔸 تعداد کل معاملات: {total_trades}")
    print(f"   🔹 معاملات برنده (WIN): {wins}")
    print(f"   🔸 معاملات بازنده (LOSS): {losses}")
    print(f"   🔹 معاملات سر‌به‌سر (BREAK-EVEN): {breakevens}")
    print(f"   🔹 معاملات لانگ: {len(longs_df)} | شورت: {len(shorts_df)}")
    print("-" * 68)
    print(f"🎯 وین‌ریت قطعی (Decisive Win Rate): {decisive_wr:.2f}%")
    print(f"🎯 وین‌ریت کل (Total Win Rate): {total_wr:.2f}%")
    print(f"⚖️ نرخ معاملات سر‌به‌سر (BE Rate): {be_rate:.2f}%")
    print(f"📈 فاکتور سود (Profit Factor): {profit_factor:.2f}")
    print(f"📉 حداکثر افت سرمایه (Max Drawdown): {max_dd * 100:.2f}%")
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
    df_trades, final_equity, max_dd = run_institutional_backtest(processed_data)
    summarize_institutional_results(df_trades, final_equity, max_dd)
    print("\n✨ اجرای تست استراتژی کوانت جدید به پایان رسید.")
