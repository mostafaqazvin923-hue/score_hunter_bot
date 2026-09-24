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
# تنظیمات سیستماتیک چندتایم‌فریمه (Multi-Timeframe Setup)
# ============================================================

EXCHANGE_ID = "lbank"
BASE_TIMEFRAME = "1h"
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
print(f"📥 دریافت داده‌های ۱ ساعته و ساختاربندی سه‌تایم‌فریمه از صرافی {EXCHANGE_ID.upper()}")
print("=" * 60)

multi_tf_data = {}

def fetch_and_prepare_data(lbank_symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=BASE_TIMEFRAME,
                    since=current_since,
                    limit=1000,
                )
                break
            except Exception:
                if attempt == 2:
                    return None

        if not batch:
            break

        last_ts = batch[-1][0]
        if last_seen is not None and last_ts <= last_seen:
            return None

        all_ohlcv.extend(batch)
        last_seen = last_ts
        current_since = last_ts + 1

        if len(batch) < 1000:
            break

    if not all_ohlcv or len(all_ohlcv) < 200:
        return None

    df_1h = pd.DataFrame(
        all_ohlcv,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    df_1h["Date"] = pd.to_datetime(df_1h["Timestamp"], unit="ms")
    df_1h = df_1h[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df_1h.dropna(inplace=True)
    df_1h.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df_1h.set_index("Date", inplace=True)

    df_4h = df_1h.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

    df_1d = df_1h.resample("1d").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

    if len(df_4h) < 50 or len(df_1d) < 50:
        return None

    df_1d["EMA_200"] = df_1d["Close"].ewm(span=200, adjust=False).mean()
    
    tr1 = df_1h["High"] - df_1h["Low"]
    tr2 = np.abs(df_1h["High"] - df_1h["Close"].shift(1))
    tr3 = np.abs(df_1h["Low"] - df_1h["Close"].shift(1))
    df_1h["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df_1h["Vol_MA"] = df_1h["Volume"].rolling(20).mean()

    return {"1h": df_1h, "4h": df_4h, "1d": df_1d}

for symbol, lbank_symbol in SYMBOLS.items():
    data_pack = fetch_and_prepare_data(lbank_symbol)
    if data_pack is not None:
        multi_tf_data[symbol] = data_pack

print(f"✅ تعداد نمادهای معتبر سه‌تایم‌فریمه تاییدشده: {len(multi_tf_data)}")
print("⚙️ اجرای مجدد موتور بک‌تست با اصلاح منطق اسکن ساختار ۴ ساعته...")


# ============================================================
# موتور اجرای بک‌تست اصلاح‌شده
# ============================================================

def run_multi_tf_backtest(data_dict: Dict[str, Dict[str, pd.DataFrame]]):
    all_timestamps = sorted({
        ts for d in data_dict.values() for ts in d["1h"].index
    })
    
    active_positions = {}
    pending_signals = []
    all_trades = []
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0.0
    
    consecutive_losses = 0
    trading_paused = False
    pause_timer = 0

    for ts in all_timestamps:
        if trading_paused:
            pause_timer -= 1
            if pause_timer <= 0:
                trading_paused = False
                consecutive_losses = 0

        signals_to_execute = pending_signals
        pending_signals = []

        for sig in signals_to_execute:
            symbol = sig["symbol"]
            if symbol in active_positions or len(active_positions) >= MAX_POSITIONS or equity < TRADE_MARGIN:
                continue
            
            df_1h = data_dict[symbol]["1h"]
            if ts not in df_1h.index:
                continue
            
            c1h = df_1h.loc[ts]
            open_price = c1h["Open"]
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

        symbols_to_close = []
        for symbol, pos in list(active_positions.items()):
            df_1h = data_dict[symbol]["1h"]
            if ts not in df_1h.index:
                continue
            
            c1h = df_1h.loc[ts]
            
            if not pos["is_breakeven"]:
                if pos["side"] == "LONG":
                    halfway = pos["entry_price"] + (pos["target_price"] - pos["entry_price"]) * 0.5
                    if c1h["High"] >= halfway:
                        pos["stop_price"] = pos["entry_price"]
                        pos["is_breakeven"] = True
                else:
                    halfway = pos["entry_price"] - (pos["entry_price"] - pos["target_price"]) * 0.5
                    if c1h["Low"] <= halfway:
                        pos["stop_price"] = pos["entry_price"]
                        pos["is_breakeven"] = True

            hit_stop = False
            hit_target = False

            if pos["side"] == "LONG":
                hit_stop = c1h["Low"] <= pos["stop_price"]
                hit_target = c1h["High"] >= pos["target_price"]
            else:
                hit_stop = c1h["High"] >= pos["stop_price"]
                hit_target = c1h["Low"] <= pos["target_price"]

            if hit_stop or hit_target:
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
                
                if outcome == "LOSS":
                    consecutive_losses += 1
                    if consecutive_losses >= 3 and not trading_paused:
                        trading_paused = True
                        pause_timer = 12
                elif outcome == "WIN":
                    consecutive_losses = 0

                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        if not trading_paused:
            for symbol, d_pack in data_dict.items():
                if symbol in active_positions:
                    continue

                df_1h = d_pack["1h"]
                df_4h = d_pack["4h"]
                df_1d = d_pack["1d"]

                if ts not in df_1h.index:
                    continue

                d_ts = pd.Timestamp(ts).normalize()
                if d_ts not in df_1d.index:
                    continue
                daily_row = df_1d.loc[d_ts]

                macro_close = daily_row["Close"]
                macro_ema = daily_row["EMA_200"]
                if not np.isfinite(macro_ema):
                    continue

                allowed_side = "LONG" if macro_close > macro_ema else "SHORT"

                # اصلاح کلیدی: فقط کندل‌های ۴ ساعته‌ی کاملاً بسته‌شده‌ی قبل از زمان فعلی (برای جلوگیری از خطای خودمقایسه‌ای)
                current_4h_floor = pd.Timestamp(ts).floor("4h")
                valid_4h = df_4h[df_4h.index < current_4h_floor]
                if len(valid_4h) < 20:
                    continue
                
                recent_4h = valid_4h.iloc[-10:]
                struct_high = recent_4h["High"].max()
                struct_low = recent_4h["Low"].min()

                i_1h = df_1h.index.get_loc(ts)
                if i_1h < 20:
                    continue

                current_1h = df_1h.iloc[i_1h]
                atr = current_1h["ATR"]
                vol_ma = current_1h["Vol_MA"]

                if not np.isfinite(atr) or atr <= 0 or not np.isfinite(vol_ma):
                    continue

                direction = None
                stop_price = 0.0
                target_price = 0.0

                if allowed_side == "LONG":
                    is_sweep = current_1h["Low"] < struct_low and current_1h["Close"] > struct_low
                    is_vol = current_1h["Volume"] > (vol_ma * 1.1)
                    if is_sweep and is_vol:
                        direction = "LONG"
                        stop_price = current_1h["Low"] - (atr * 0.5)
                        risk_dist = current_1h["Close"] - stop_price
                        if risk_dist <= 0:
                            risk_dist = atr
                        target_price = current_1h["Close"] + (risk_dist * 2.0)

                elif allowed_side == "SHORT":
                    is_sweep = current_1h["High"] > struct_high and current_1h["Close"] < struct_high
                    is_vol = current_1h["Volume"] > (vol_ma * 1.1)
                    if is_sweep and is_vol:
                        direction = "SHORT"
                        stop_price = current_1h["High"] + (atr * 0.5)
                        risk_dist = stop_price - current_1h["Close"]
                        if risk_dist <= 0:
                            risk_dist = atr
                        target_price = current_1h["Close"] - (risk_dist * 2.0)

                if direction is not None:
                    pending_signals.append({
                        "symbol": symbol,
                        "side": direction,
                        "stop_price": stop_price,
                        "target_price": target_price,
                    })

    return pd.DataFrame(all_trades), equity, max_drawdown


def summarize_multi_tf_results(trades_df: pd.DataFrame, final_equity: float, max_dd: float):
    print("\n" + "=" * 68)
    print("📊 گزارش نهایی استراتژی چندتایم‌فریمه (اصلاح‌شده)")
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
    print("=" * 68)


if __name__ == "__main__":
    df_trades, final_equity, max_dd = run_multi_tf_backtest(multi_tf_data)
    summarize_multi_tf_results(df_trades, final_equity, max_dd)
    print("\n✨ تست مجدد به پایان رسید.")
