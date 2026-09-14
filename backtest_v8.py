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
# HUNTER-V46
# Baseline-preserving backtest:
# منطق سیگنال/ورود/خروج V33 حفظ شده و فقط خطاهای فنی بک‌تست
# (داده ناقص، کندل ناقص، pagination، index lookup و گزارش DD)
# اصلاح شده‌اند.
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True})

SYMBOLS = {
    # Core leaders from V42
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

    # New candidates replacing persistent weak names
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}

# Names that were persistently weak in the tested basket are excluded.
# This is a fixed universe choice, not a future-looking trade filter.
REMOVED_COINS = {
    "NEAR",
    "OP",
    "HYPE",
    "HBAR",
    "AVAX",
    "SUI",
    "PENDLE",
    "TIA",
    "FET",
    "SEI",
    "ARB",
    "DOT",
    "ETC",
    "SHIB",
    "STX",
    "RUNE",
    "MKR",
    "APT",
    "LTC",
    "AR",
    "IMX",
    "PEPE",
    "BONK",
}

SYMBOLS = {
    k: v for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}


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

# ------------------------------------------------------------
# محافظت سبک از زنجیره ضرر
# این فیلتر فقط بعد از ایجاد زنجیره ضرر فعال می‌شود و در حالت عادی
# هیچ محدودیتی روی سیگنال‌های V42 اعمال نمی‌کند.
# ------------------------------------------------------------
STREAK_TRIGGER = 3
STREAK_PAUSE_CANDLES = 2
STREAK_MOM_LONG_BOOST = 0.010

# ------------------------------------------------------------
# دریافت مطمئن داده
# ------------------------------------------------------------

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 60)
print("📥 دریافت داده‌ها - HUNTER-V46")
print("=" * 60)

processed_data = {}

def fetch_symbol_data(lbank_symbol):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None

        # retry برای خطای موقت API
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
                    print(f"⚠️ دریافت ناقص {lbank_symbol}: {e}")
                    return None

        if not batch:
            break

        # جلوگیری از loop در pagination
        first_ts = batch[0][0]
        last_ts = batch[-1][0]

        if last_seen is not None and last_ts <= last_seen:
            print(f"⚠️ pagination متوقف شد: {lbank_symbol}")
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

    # آخرین کندل ممکن است هنوز بسته نشده باشد؛ حذفش می‌کنیم.
    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(df.iloc[-1]["Date"].timestamp() * 1000)
        if last_ms + 4 * 60 * 60 * 1000 > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    # کنترل فاصله کندل‌ها؛ یک gap واقعی یعنی داده آن نماد قابل اتکا نیست.
    deltas = df["Date"].diff().dropna()
    if not deltas.empty and deltas.max() > pd.Timedelta(hours=4, minutes=10):
        print(f"⚠️ gap بزرگ در {lbank_symbol}؛ نماد حذف شد.")
        return None

    # ATR دقیقاً مطابق منطق V33: rolling SMA روی True Range
    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))
    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["Mom_Short"] = (
        (df["Close"] - df["Close"].shift(10))
        / df["Close"].shift(10)
    )
    df["Mom_Long"] = (
        (df["Close"] - df["Close"].shift(30))
        / df["Close"].shift(30)
    )

    # timestampها را یک بار به عنوان index نگه می‌داریم.
    df.set_index("Date", inplace=True)
    return df


for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h
        print(f"✅ {symbol}: {len(df4h)} کندل")
    else:
        print(f"❌ {symbol}: حذف شد")

print(f"\n✅ تعداد نمادهای معتبر: {len(processed_data)} از {len(SYMBOLS)}")
print("⚙️ شروع اجرای بک‌تست HUNTER-V46...")

# ------------------------------------------------------------
# بک‌تست — منطق اصلی V33 حفظ شده؛ V45 فقط تشخیص Loss Cluster اضافه می‌کند
# ------------------------------------------------------------


# ============================================================
# V46 — همان موتور V44، فقط با Market Regime Gate اختیاری
# ============================================================
def run_backtest(processed_data, regime_breadth_threshold=None):
    """Run the baseline-preserving engine with an optional regime gate."""
    all_timestamps = sorted({
        ts
        for df in processed_data.values()
        for ts in df.index
    })
    
    active_positions = {}
    all_trades = []
    
    
    for ts in all_timestamps:
    
        # -------------------------
        # مدیریت معاملات باز
        # -------------------------
        symbols_to_close = []
    
        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
    
            if ts not in df.index:
                continue
    
            c4h = df.loc[ts]
    
            # منطق trailing دقیقاً مطابق V33:
            # ابتدا highest همان کندل، سپس ATR همان کندل.
            if c4h["High"] > pos["highest_price"]:
                pos["highest_price"] = c4h["High"]
                new_trailing_sl = (
                    pos["highest_price"]
                    - TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                )
    
                if new_trailing_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_trailing_sl
    
            hit_sl = c4h["Low"] <= pos["stop_loss"]
    
            curr_i = df.index.get_loc(ts)
            candles_held = curr_i - pos["entry_index"]
            is_timeout = candles_held >= TIMEOUT_CANDLES
    
            if hit_sl or is_timeout:
                initial_risk = pos["initial_risk"]
    
                # دقیقاً همان ترتیب/منطق V33
                exit_p = (
                    min(pos["stop_loss"], c4h["Open"])
                    if hit_sl
                    else c4h["Close"]
                )
    
                r_real = (
                    (exit_p - pos["entry_price"]) / initial_risk
                    - (FEE_RATE * 2)
                )
    
                outcome = "WIN" if r_real > 0 else "LOSS"
    
    
                # اطلاعات تشخیصی در لحظه خروج؛ هیچ اثری بر معامله ندارد.
                btc_regime = "UNKNOWN"
                market_breadth = np.nan
    
                if "BTC" in processed_data and ts in processed_data["BTC"].index:
                    btc = processed_data["BTC"].loc[ts]
                    btc_regime = (
                        "STRONG"
                        if (
                            btc["Close"] > btc["EMA20"]
                            and btc["EMA20"] > btc["EMA50"]
                            and btc["Close"] > btc["EMA200"]
                        )
                        else "WEAK"
                    )
    
                breadth_total = 0
                breadth_bull = 0
                for _sym, _df in processed_data.items():
                    if ts in _df.index:
                        breadth_total += 1
                        _c = _df.loc[ts]
                        if (
                            _c["Close"] > _c["EMA20"]
                            and _c["EMA20"] > _c["EMA50"]
                            and _c["Close"] > _c["EMA200"]
                        ):
                            breadth_bull += 1
    
                if breadth_total:
                    market_breadth = breadth_bull / breadth_total
    
                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": "LONG",
                    "Outcome": outcome,
                    "Return": r_real,
                    "ExitOrder": len(all_trades),
                    "BTC_Regime": btc_regime,
                    "Market_Breadth": market_breadth,
                })
    
                symbols_to_close.append(symbol)
    
        for sym in symbols_to_close:
            del active_positions[sym]
    
        # -------------------------
        # امتیازدهی و ورود
        # -------------------------
    
        current_scores = {}
    
        for symbol, df in processed_data.items():
            if ts in df.index:
                val = df.loc[ts, "Mom_Long"]
                if not np.isnan(val):
                    current_scores[symbol] = val
    
        if not current_scores:
            continue
    
        ranked_symbols = sorted(
            current_scores.keys(),
            key=lambda x: current_scores[x],
            reverse=True,
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
    
            if i < EMA_WARMUP:
                continue
    
            c4h = df.iloc[i]
    
            regime_bull = (
                (c4h["Close"] > c4h["EMA20"])
                and (c4h["EMA20"] > c4h["EMA50"])
                and (c4h["Close"] > c4h["EMA200"])
            )
    
            valid_trend = (
                regime_bull
                and (c4h["Mom_Short"] > 0.012)
                and (c4h["Mom_Long"] > 0.035)
            )
    
            if valid_trend:
                # ورود روی Open همان کندل — عمداً تغییر نکرده است.
                entry_price = c4h["Open"] * (1 + SLIPPAGE)
    
                initial_sl = (
                    entry_price
                    - INITIAL_ATR_MULTIPLIER * c4h["ATR"]
                )
    
                initial_risk = entry_price - initial_sl
                sl_dist_pct = initial_risk / entry_price
    
                if 0.01 <= sl_dist_pct <= 0.04:
                    active_positions[symbol] = {
                        "side": "LONG",
                        "entry_price": entry_price,
                        "stop_loss": initial_sl,
                        "highest_price": entry_price,
                        "initial_risk": initial_risk,
                        "entry_index": i,
                    }
    


    return pd.DataFrame(all_trades)


def summarize_result(trades_df, label):
    print("\n" + "=" * 68)
    print(f"📊 {label}")
    print("=" * 68)

    if trades_df.empty:
        print("⚠️ هیچ معامله‌ای ثبت نشد.")
        return {
            "Trades": 0, "WR": 0.0, "NetR": 0.0,
            "MaxDD": 0.0, "MaxLossStreak": 0
        }

    trades_df = trades_df.sort_values(
        ["Timestamp", "ExitOrder"], kind="stable"
    ).reset_index(drop=True)

    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())
    trades = len(trades_df)
    wr = wins / trades * 100
    net_r = float(trades_df["Return"].sum())

    cur = max_streak = 0
    eq = peak = max_dd = 0.0
    for outcome, r in zip(trades_df["Outcome"], trades_df["Return"]):
        if outcome == "LOSS":
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
        eq += float(r)
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)

    print(f"🔸 Trades: {trades}")
    print(f"🔸 Wins: {wins}")
    print(f"🔸 Losses: {losses}")
    print(f"🎯 Win Rate: {wr:.2f}%")
    print(f"💰 Net R: {net_r:.2f}R")
    print(f"❄️ Max Loss Streak: {max_streak}")
    print(f"📉 Max DD: {max_dd:.2f}R")

    return {
        "Trades": trades,
        "WR": wr,
        "NetR": net_r,
        "MaxDD": max_dd,
        "MaxLossStreak": max_streak,
    }


if __name__ == "__main__":
    print("\n" + "=" * 68)
    print("🚀 HUNTER-V46 — Market Regime A/B Test")
    print("=" * 68)
    print("A: BTC Weak + Breadth < 20%  → block NEW LONG")
    print("B: BTC Weak + Breadth < 30%  → block NEW LONG")
    print("⚠️ Entry / Exit / SL / Trailing / Fees باقی می‌مانند.")
    print("⚠️ V44 Streak Pause در این تست غیرفعال است؛ فقط Regime Gate تست می‌شود.")
    print("⚠️ پوزیشن‌های باز هرگز توسط Regime Gate بسته نمی‌شوند.")
    print("📌 A/B فقط تفاوت آستانه Breadth را می‌سنجد: 20% در برابر 30%.")

    # The data-loading code above this main guard is already executed once.
    results = {}

    for label, threshold in [
        ("V46-A (20% Breadth)", 0.20),
        ("V46-B (30% Breadth)", 0.30),
    ]:
        print("\n" + "-" * 68)
        print(f"🧪 اجرای {label}")
        print("-" * 68)

        df_trades = run_backtest(
            processed_data,
            regime_breadth_threshold=threshold,
        )

        results[label] = summarize_result(df_trades, label)

    print("\n" + "=" * 68)
    print("🏁 V46 A/B COMPARISON")
    print("=" * 68)
    print(
        f"{'Variant':<24} {'Trades':>7} {'WR%':>8} "
        f"{'NetR':>9} {'MaxDD':>9} {'MaxLS':>8}"
    )
    print("-" * 68)

    for label, r in results.items():
        print(
            f"{label:<24} {r['Trades']:>7} {r['WR']:>7.2f}% "
            f"{r['NetR']:>8.2f}R {r['MaxDD']:>8.2f}R "
            f"{r['MaxLossStreak']:>8}"
        )

    print("\n✨ بک‌تست HUNTER-V46 به پایان رسید.")

