"""
TTR (Turtle-style Trend Rider) — Portfolio Backtest on LBank
============================================================================
طبق درخواست صریح: به‌جای هدف بی‌واقعیت "WR بالا با R:R ثابت"، این نسخه دقیقاً
همان چیزی را پیاده می‌کند که تریدرهای Trend-Following واقعی (مشهورترین‌شان:
Turtle Traders) انجام می‌دهند — استاپ اولیه کوچک و ثابت، و خروج **صبورانه**
که اجازه می‌دهد بردها هرقدر بازار اجازه بدهد رشد کنند (۲R، ۵R، حتی ۱۰R+)،
نه بستن زودهنگام با Take-Profit ثابت یا Trailing تنگ.

چرا نسخه قبلی (HUNTER-V74) این کار را نمی‌کرد، با اینکه اسمش Trailing بود:
  - Trailing Stop فقط ۲ برابر ATR بود (خیلی تنگ برای نوسان عادی کریپتو)
  - سقف نگهداری فقط ۴۵ کندل ۴ساعته (~۷.۵ روز) بود — قبل از اینکه یک روند
    واقعی حتی فرصت شکل‌گیری پیدا کند، معامله به‌زور بسته می‌شد
  - نتیجه: میانگین برد واقعی فقط ۰.۷۴R شد، نه چیزی نزدیک ۱۰R

راه‌حل این نسخه (سیستم کلاسیک Donchian Channel Breakout، همان منطق Turtle):
  - ورود: شکست کانال N-کندلی (مثلاً بالاترین سقف ۴۰ کندل اخیر)
  - استاپ اولیه: ثابت و کوچک، ۲ برابر ATR (ریسک مشخص و محدود از همان اول)
  - خروج: نه Take-Profit ثابت، نه Timeout کوتاه — فقط وقتی قیمت کانال
    مخالفِ کوتاه‌تری (مثلاً پایین‌ترین کف ۱۵ کندل اخیر) را بشکند. این یعنی
    تا وقتی روند واقعی ادامه دارد، معامله باز می‌ماند؛ فقط با یک بازگشت
    واقعی و معنادار بسته می‌شود — دقیقاً منطق «اجازه به بردها برای رشد».

نکات ضد-تقلب (طبق همان اصولی که در جلسه قبل یاد گرفتیم و اصلاح کردیم):
  - کانال ورود/خروج در هر لحظه فقط از کندل‌های *قبل* از کندل جاری ساخته
    می‌شود (کندل جاری هرگز در محاسبه کانال خودش شرکت نمی‌کند).
  - سیگنال از آخرین کندل کاملاً بسته گرفته می‌شود؛ ورود در بازِ کندل بعدی.
  - رژیم 1D با merge_asof و تأخیر بسته‌شدن متصل می‌شود.
  - هیچ فایلی ذخیره نمی‌شود؛ همه‌چیز مستقیم در کنسول چاپ می‌شود.
"""

import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

# ============================================================================
# پیکربندی — طبق منطق کلاسیک Turtle (Donchian Breakout)
# ============================================================================
CONFIG = {
    "days_back": 365,
    "min_1h_bars": 1400,

    "entry_channel": 40,     # کانال ورود: شکست بالاترین/پایین‌ترین N کندل 4H اخیر
    "exit_channel": 15,      # کانال خروج: کوتاه‌تر و سریع‌تر — برگشت واقعی را زودتر تشخیص می‌دهد
    "initial_atr_mult": 2.0, # استاپ اولیه: فاصله ثابت و کوچک (واحد ریسک)
    "min_atr_pct": 0.003,
    "vol_mult": 1.1,         # تأیید حجم روی کندل شکست
    "adx_min": 15,           # حداقل قدرت روند (1D) برای فیلترکردن بازار کاملاً بی‌روند
    "max_hold_bars": 540,    # فقط یک محافظ ایمنی خیلی سخاوتمندانه (~۹۰ روز)، نه Take-Profit
    "max_open_positions": 5,
    "taker_fee": 0.0005,
    "slippage_pct": 0.0005,
}

SYMBOLS = {
    "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "XRP": "XRP/USDT",
    "ADA": "ADA/USDT", "AVAX": "AVAX/USDT", "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT", "SUI": "SUI/USDT", "DOT": "DOT/USDT",
}

exchange = ccxt.lbank({"enableRateLimit": True})


# ============================================================================
# ۱. دانلود داده 1H
# ============================================================================
def fetch_1h_data(symbol_key, lbank_symbol, days_back):
    start_date = datetime.now() - timedelta(days=days_back)
    since_ts = int(start_date.timestamp() * 1000)
    now_ts = exchange.milliseconds()

    all_ohlcv = []
    current_since = since_ts
    print(f"🔹 در حال دریافت دیتای 1 ساعته {symbol_key}...")
    while current_since < now_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(lbank_symbol, timeframe="1h",
                                          since=current_since, limit=1000)
            if not ohlcv:
                break
            current_since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"  ❌ خطا در دریافت داده {symbol_key}: {e}")
            break

    if not all_ohlcv:
        print(f"  ❌ دیتایی برای {symbol_key} دریافت نشد.")
        return None

    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  ✔️ دیتای 1 ساعته {symbol_key} آماده شد (تعداد کندل: {len(df)})")
    return df


# ============================================================================
# ۲. اندیکاتورها — فقط رو به گذشته
# ============================================================================
def calculate_indicators(df):
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    plus_dm = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    tr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean().fillna(20)

    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df


# ============================================================================
# ۳. ساخت 4H (ورود) و 1D (رژیم) — بدون Lookahead
# ============================================================================
def resample_htf(df1h, rule, close_delta):
    htf = (
        df1h.set_index("Date")
        .resample(rule, label="left", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
        .reset_index()
    )
    htf = calculate_indicators(htf)
    htf["available_at"] = htf["Date"] + close_delta
    return htf


def attach_regime(df4h, df1d):
    df4h = df4h.rename(columns={"available_at": "available_at_4h"})
    ctx = df1d[["available_at", "EMA_50", "EMA_200", "ADX"]].rename(
        columns={"EMA_50": "EMA50_1D", "EMA_200": "EMA200_1D", "ADX": "ADX_1D"}
    )
    merged = pd.merge_asof(
        df4h.sort_values("Date"), ctx.sort_values("available_at"),
        left_on="Date", right_on="available_at", direction="backward",
    )
    return merged.drop(columns=["available_at", "available_at_4h"])


def add_regime_flags(df):
    df = df.copy()
    df["regime_long"] = ((df["EMA50_1D"] > df["EMA200_1D"]) & (df["ADX_1D"] >= CONFIG["adx_min"])).fillna(False)
    df["regime_short"] = ((df["EMA50_1D"] < df["EMA200_1D"]) & (df["ADX_1D"] >= CONFIG["adx_min"])).fillna(False)
    return df


# ============================================================================
# ۴. کانال‌های Donchian — همیشه از کندل‌های *قبل* از کندل جاری (بدون Lookahead)
# ============================================================================
def add_donchian_channels(df):
    df = df.copy()
    n_in, n_out = CONFIG["entry_channel"], CONFIG["exit_channel"]
    # shift(1): کانال در ایندکس i فقط از کندل‌های 0..i-1 ساخته می‌شود، نه خودِ i
    df["entry_high"] = df["High"].rolling(n_in).max().shift(1)
    df["entry_low"] = df["Low"].rolling(n_in).min().shift(1)
    df["exit_high"] = df["High"].rolling(n_out).max().shift(1)
    df["exit_low"] = df["Low"].rolling(n_out).min().shift(1)
    return df


# ============================================================================
# ۵. موتور اصلی
# ============================================================================
def run_symbol_backtest(symbol, df, diag=None):
    if diag is None:
        diag = {}
    for key in ["bars_scanned", "breakout_events", "trades_taken"]:
        diag.setdefault(key, 0)

    n = len(df)
    trades = []
    position = None  # فقط یک پوزیشن هم‌زمان روی هر نماد

    start_idx = max(CONFIG["entry_channel"] + 5, 210)
    for i in range(start_idx, n - 2):
        row = df.iloc[i]

        # ---------------- مدیریت پوزیشن باز (اگر هست) ----------------
        if position is not None:
            side = position["side"]
            hit_hard_sl = (row["Low"] <= position["hard_sl"]) if side == "LONG" else (row["High"] >= position["hard_sl"])
            # خروج صبورانه: شکست کانال مخالفِ کوتاه‌تر (نه Take-Profit ثابت)
            hit_channel_exit = (row["Low"] <= row["exit_low"]) if side == "LONG" else (row["High"] >= row["exit_high"])
            held = i - position["entry_idx"]
            is_timeout = held >= CONFIG["max_hold_bars"]

            if hit_hard_sl or hit_channel_exit or is_timeout:
                if side == "LONG":
                    exit_price = min(position["hard_sl"], row["Open"]) if hit_hard_sl else row["Open"]
                    r_real = (exit_price - position["entry_price"]) / position["initial_risk"]
                else:
                    exit_price = max(position["hard_sl"], row["Open"]) if hit_hard_sl else row["Open"]
                    r_real = (position["entry_price"] - exit_price) / position["initial_risk"]

                cost_R = (2 * CONFIG["taker_fee"] + 2 * CONFIG["slippage_pct"]) / (position["initial_risk"] / position["entry_price"])
                net_R = r_real - cost_R
                outcome = "WIN" if net_R > 0 else "LOSS"

                trades.append({
                    "Symbol": symbol, "Side": side, "Outcome": outcome,
                    "Entry": round(position["entry_price"], 5), "Exit": round(exit_price, 5),
                    "EntryTime": position["entry_time"], "ExitTime": row["Date"],
                    "HoursHeld": held * 4, "Gross_R": round(r_real, 3), "Net_R": round(net_R, 3),
                    "ExitReason": "HardSL" if hit_hard_sl else ("ChannelExit" if hit_channel_exit else "Timeout"),
                })
                diag["trades_taken"] += 1
                position = None
            else:
                continue  # پوزیشن باز است، دنبال سیگنال جدید روی این نماد نباش

        # ---------------- جست‌وجوی سیگنال جدید ----------------
        if pd.isna(row["ATR"]) or row["ATR"] / row["Close"] < CONFIG["min_atr_pct"]:
            continue
        if pd.isna(row["entry_high"]) or pd.isna(row["entry_low"]):
            continue
        diag["bars_scanned"] += 1

        avg_vol = df.iloc[i - 20:i]["Volume"].mean()
        vol_ok = row["Volume"] >= CONFIG["vol_mult"] * avg_vol

        breakout_long = row["regime_long"] and row["Close"] > row["entry_high"] and vol_ok
        breakout_short = row["regime_short"] and row["Close"] < row["entry_low"] and vol_ok

        if not (breakout_long or breakout_short):
            continue
        diag["breakout_events"] += 1

        # ورود در بازِ کندل بعدی (نه همین کندل که سیگنال از آن گرفته شد)
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        entry_row = df.iloc[entry_idx]
        side = "LONG" if breakout_long else "SHORT"
        entry_price = entry_row["Open"] * (1 + CONFIG["slippage_pct"]) if side == "LONG" else entry_row["Open"] * (1 - CONFIG["slippage_pct"])
        atr_now = row["ATR"]  # ATR همان کندلِ سیگنال (کاملاً بسته)، نه کندل ورود
        hard_sl = entry_price - CONFIG["initial_atr_mult"] * atr_now if side == "LONG" else entry_price + CONFIG["initial_atr_mult"] * atr_now
        initial_risk = abs(entry_price - hard_sl)
        if initial_risk <= 0:
            continue

        position = {
            "side": side, "entry_price": entry_price, "hard_sl": hard_sl,
            "initial_risk": initial_risk, "entry_idx": entry_idx, "entry_time": entry_row["Date"],
        }

    return trades, diag


# ============================================================================
# ۶. متریک‌ها
# ============================================================================
def compute_metrics(trades_df, r_col, label):
    if trades_df.empty:
        print(f"⚠️ هیچ معامله‌ای ({label}) ثبت نشد.")
        return
    total = len(trades_df)
    wins = trades_df[trades_df["Outcome"] == "WIN"]
    losses = trades_df[trades_df["Outcome"] == "LOSS"]
    win_rate = len(wins) / total * 100
    gross_profit, gross_loss = wins[r_col].sum(), losses[r_col].sum()
    net = trades_df[r_col].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.nan
    expectancy = trades_df[r_col].mean()
    avg_win = wins[r_col].mean() if len(wins) else float("nan")
    avg_loss = losses[r_col].mean() if len(losses) else float("nan")
    biggest_win = trades_df[r_col].max()

    equity = trades_df.sort_values("EntryTime")[r_col].cumsum()
    max_dd = (equity - equity.cummax()).min()

    outcomes = trades_df.sort_values("EntryTime")["Outcome"].tolist()
    max_w = max_l = cur_w = cur_l = 0
    for o in outcomes:
        if o == "WIN":
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w, max_l = max(max_w, cur_w), max(max_l, cur_l)

    span_days = max((trades_df["EntryTime"].max() - trades_df["EntryTime"].min()).days, 1)
    avg_hold_days = (trades_df["HoursHeld"] / 24).mean()

    print(f"\n--- نتایج ({label}) ---")
    print(f"🔸 تعداد کل معاملات: {total}")
    print(f"🔸 برنده: {len(wins)}   بازنده: {len(losses)}")
    print(f"🎯 Win Rate: {win_rate:.2f}%")
    print(f"💰 Net: {net:.2f}R   Profit Factor: {profit_factor:.2f}   Expectancy: {expectancy:.3f}R")
    print(f"📊 میانگین برد: {avg_win:+.2f}R   میانگین باخت: {avg_loss:+.2f}R   بزرگ‌ترین برد: {biggest_win:+.2f}R")
    print(f"📉 Max Drawdown: {max_dd:.2f}R")
    print(f"🔁 حداکثر برد متوالی: {max_w}   حداکثر باخت متوالی: {max_l}")
    print(f"⏱️ میانگین مدت نگهداری: {avg_hold_days:.1f} روز")
    print("\nتفکیک به هر نماد:")
    print(trades_df.groupby("Symbol")["Outcome"].value_counts().unstack(fill_value=0).to_string())
    print("\nتفکیک به دلیل خروج:")
    print(trades_df["ExitReason"].value_counts().to_string())


# ============================================================================
# اجرای کامل
# ============================================================================
def main():
    print("=" * 70)
    print("📥 دانلود داده 1H از LBank و ساخت 4H (ورود) / 1D (رژیم) بدون Lookahead")
    print("=" * 70)

    all_trades, all_diag = [], {}
    for key, sym in SYMBOLS.items():
        df1h_raw = fetch_1h_data(key, sym, CONFIG["days_back"])
        if df1h_raw is None or len(df1h_raw) < CONFIG["min_1h_bars"]:
            print(f"  ⏭️ داده کافی برای {key} نیست، رد شد.")
            continue

        df1h = calculate_indicators(df1h_raw)
        df4h = resample_htf(df1h, "4h", pd.Timedelta(hours=4))
        df1d = resample_htf(df1h, "1D", pd.Timedelta(days=1))
        df = attach_regime(df4h, df1d)
        df = add_regime_flags(df)
        df = add_donchian_channels(df)
        df = df.dropna(subset=["ATR", "ADX", "EMA200_1D"]).reset_index(drop=True)

        trades, diag = run_symbol_backtest(key, df)
        print(f"  ✅ {key}: {len(trades)} معامله یافت شد.  "
              f"[Bars={diag['bars_scanned']} Breakout={diag['breakout_events']} Trades={diag['trades_taken']}]")
        all_trades.extend(trades)
        all_diag[key] = diag

    print("\n" + "=" * 70)
    print("📊 گزارش تجمیعی نهایی پورتفوی — TTR (Turtle-style Trend Rider)")
    print("=" * 70)

    if not all_trades:
        print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")
        return

    df_trades = pd.DataFrame(all_trades)
    compute_metrics(df_trades, "Gross_R", "قبل از هزینه‌ها")
    compute_metrics(df_trades, "Net_R", "بعد از هزینه‌ها (کارمزد + اسلیپیج)")

    print("\n" + "=" * 70)
    print("📋 لیست کامل معاملات")
    print("=" * 70)
    pd.set_option("display.max_rows", None)
    print(df_trades.sort_values("EntryTime").to_string(index=False))

    print("\n✨ بک‌تست TTR به اتمام رسید.")


if __name__ == "__main__":
    main()
