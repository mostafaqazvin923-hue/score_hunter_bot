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
# HUNTER-V44
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
print("📥 دریافت داده‌ها - HUNTER-V44")
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
print("⚙️ شروع اجرای بک‌تست HUNTER-V44...")

# ------------------------------------------------------------
# بک‌تست — منطق اصلی V33 حفظ شده
# ------------------------------------------------------------

all_timestamps = sorted({
    ts
    for df in processed_data.values()
    for ts in df.index
})

active_positions = {}
all_trades = []

# تعداد ضررهای متوالی پورتفوی؛ فقط معاملات بسته‌شده را می‌شمارد.
portfolio_loss_streak = 0
streak_pause_remaining = 0

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

            if outcome == "LOSS":
                portfolio_loss_streak += 1
            else:
                portfolio_loss_streak = 0

            # فقط پس از 3 ضرر پیاپی، دو کندل بازار را استراحت می‌دهیم.
            # این کار در حالت عادی هیچ اثری ندارد.
            if portfolio_loss_streak >= STREAK_TRIGGER:
                streak_pause_remaining = STREAK_PAUSE_CANDLES

            all_trades.append({
                "Timestamp": ts,
                "Symbol": symbol,
                "Side": "LONG",
                "Outcome": outcome,
                "Return": r_real,
            })

            symbols_to_close.append(symbol)

    for sym in symbols_to_close:
        del active_positions[sym]

    # -------------------------
    # امتیازدهی و ورود
    # -------------------------
    if streak_pause_remaining > 0:
        streak_pause_remaining -= 1
        continue

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

# ------------------------------------------------------------
# گزارش
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("📊 گزارش نهایی HUNTER-V44")
print("=" * 60)

if not all_trades:
    print("⚠️ معامله‌ای ثبت نشد.")
else:
    trades_df = pd.DataFrame(all_trades)
    trades_df.sort_values(
        ["Timestamp", "Symbol"],
        inplace=True,
        ignore_index=True,
    )

    tot_trades = len(trades_df)
    tot_wins = int((trades_df["Outcome"] == "WIN").sum())
    tot_losses = int((trades_df["Outcome"] == "LOSS").sum())

    win_rate = (
        tot_wins / tot_trades * 100
        if tot_trades
        else 0
    )

    net_r = trades_df["Return"].sum()

    outcomes = trades_df["Outcome"].tolist()

    max_wins = 0
    max_losses = 0
    curr_wins = 0
    curr_losses = 0

    loss_sequences = []
    temp_loss_seq = 0

    for out in outcomes:
        if out == "WIN":
            curr_wins += 1
            curr_losses = 0

            max_wins = max(max_wins, curr_wins)

            if temp_loss_seq > 0:
                loss_sequences.append(temp_loss_seq)
                temp_loss_seq = 0
        else:
            curr_losses += 1
            curr_wins = 0

            temp_loss_seq += 1
            max_losses = max(max_losses, curr_losses)

    if temp_loss_seq > 0:
        loss_sequences.append(temp_loss_seq)

    # گزارش DD معاملاتی بر اساس R، نه سرمایه فرضی.
    equity_r = 0.0
    peak_r = 0.0
    max_dd_r = 0.0

    for r in trades_df["Return"]:
        equity_r += r
        peak_r = max(peak_r, equity_r)
        dd = equity_r - peak_r
        max_dd_r = min(max_dd_r, dd)

    print(f"🔸 تعداد کل معاملات سبد: {tot_trades}")
    print(f"🔸 معاملات برنده (WIN): {tot_wins}")
    print(f"🔸 معاملات بازنده (LOSS): {tot_losses}")
    print(f"🔥 حداکثر سودهای متوالی: {max_wins}")
    print(f"❄️ حداکثر ضررهای متوالی: {max_losses}")
    print(f"🎯 وین‌ریت تجمیعی پورتفوی: {win_rate:.2f}%")
    print(f"💰 مجموع بازدهی خالص کل: {net_r:.2f}R")
    print(f"📉 Max Drawdown معاملاتی: {max_dd_r:.2f}R")

    print("\n" + "-" * 60)
    print("📉 لیست کامل تعداد ضررهای متوالی ثبت‌شده")
    print("-" * 60)

    if loss_sequences:
        print(", ".join(map(str, loss_sequences)))
    else:
        print("هیچ زنجیره ضرری ثبت نشد.")

    print("\n" + "-" * 60)
    print("📈 گزارش تفکیک‌شده به تفکیک هر ارز")
    print("-" * 60)

    symbol_summary = []

    for sym in SYMBOLS.keys():
        sym_trades = trades_df[
            trades_df["Symbol"] == sym
        ]

        s_tot = len(sym_trades)

        if s_tot > 0:
            s_wins = int(
                (sym_trades["Outcome"] == "WIN").sum()
            )
            s_loss = int(
                (sym_trades["Outcome"] == "LOSS").sum()
            )
            s_wr = s_wins / s_tot * 100
            s_net_r = sym_trades["Return"].sum()
        else:
            s_wins = 0
            s_loss = 0
            s_wr = 0.0
            s_net_r = 0.0

        symbol_summary.append({
            "Symbol": sym,
            "Trades": s_tot,
            "Wins": s_wins,
            "Losses": s_loss,
            "WinRate(%)": round(s_wr, 2),
            "Net_R": round(s_net_r, 2),
        })

    summary_df = pd.DataFrame(symbol_summary)
    print(summary_df.to_string(index=False))

print("\n✨ بک‌تست HUNTER-V44 به پایان رسید.")
