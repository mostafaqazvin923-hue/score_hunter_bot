"""
LSCS (Liquidity-Structure Confluence System) — Portfolio Backtest on LBank
============================================================================
این اسکریپت دقیقاً طبق سند طراحی (Sweep -> BOS -> Retest -> Score -> Entry)
پیاده‌سازی شده و روی داده واقعی 1H صرافی LBank (از طریق ccxt) اجرا می‌شود.

نکات ضد-تقلب (طبق درخواست صریح کاربر) که نسبت به کد نمونه اصلاح شده‌اند:
  1) داده 4H با merge_asof و "available_at = close_time + 4h" ساخته می‌شود،
     یعنی هر کندل 1H فقط به آخرین کندل 4H که واقعاً بسته شده دسترسی دارد
     (باگ کد نمونه: resample هم‌زمان از کندل‌های آینده همان سبد 4H استفاده می‌کرد).
  2) هیچ "نگاه به آینده برای فیلتر کردن معامله" وجود ندارد — در کد نمونه خط
     `if future_window < tp: break` معاملاتی را که از قبل می‌دانست بازنده‌اند حذف
     می‌کرد. این خط کاملاً حذف شده. نتیجه معامله فقط با اسکن گام‌به‌گام روبه‌جلو
     (دقیقاً مثل واقعیت لایو) تعیین می‌شود.
  3) Overlap Lock: تا زمانی که تکلیف یک معامله باز روشن نشده، هیچ سیگنال جدیدی
     (حتی روی همان کندلی که SL/TP آن معامله فعال می‌شود) صادر نمی‌شود.
     scan از exit_idx + 1 از سر گرفته می‌شود، نه exit_idx.
  4) Swing High/Low فقط از کندل‌های *قبل* از کندل جاری محاسبه می‌شود (lookback
     window که کندل جاری را شامل نمی‌شود).
  5) اندیکاتورها (EMA/RSI/ATR/ADX) فقط rolling/ewm رو به گذشته هستند، بدون
     center=True و بدون هیچ تابع Repaint.
"""

import os
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
# پیکربندی
# ============================================================================
CONFIG = {
    "days_back": 365,
    "min_1h_bars": 1400,          # حداقل کندل لازم برای EMA200(4H) با حاشیه امن
    "structure_lookback": 20,     # کندل قبل از سیگنال برای تشخیص Swing H/L
    "sweep_confirm_window": 16,   # حداکثر کندل بعد از Sweep برای وقوع BOS (قبلاً 10 — طبق قیف، بزرگ‌ترین گلوگاه بود)
    "retest_window": 10,          # حداکثر کندل بعد از BOS برای Retest (قبلاً 6)
    "retest_atr_mult": 0.25,      # عرض ناحیه Retest بر حسب ATR
    "sl_atr_buffer": 0.30,        # بافر ATR اضافه به Structure برای SL
    "min_sl_atr": 0.7,            # حداقل فاصله SL بر حسب ATR (قبلاً 0.8)
    "max_sl_atr": 2.8,            # حداکثر فاصله SL بر حسب ATR (قبلاً 2.5)
    "rr": 2.0,                    # Risk:Reward ثابت — هدف تغییر نکرده: همچنان 1:2
    "vol_mult": 1.15,             # حداقل نسبت حجم کندل BOS به میانگین (قبلاً 1.3)
    "min_atr_pct": 0.005,         # حداقل ATR/Price برای اجازه معامله (0.5%)
    "adx_min": 18,                # حداقل ADX(4H) برای رژیم روند‌دار (قبلاً 20)
    "min_score": 55,              # حداقل امتیاز برای پذیرش سیگنال (قبلاً 65 — هدف جدید WR≈50% نه 70%)
    "max_forward_bars": 200,      # سقف اسکن روبه‌جلو برای رسیدن SL/TP
    "taker_fee": 0.0005,          # کارمزد هر طرف (فرض ~0.05%)
    "slippage_pct": 0.0005,       # اسلیپیج تخمینی هر طرف
}

SYMBOLS = {
    "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "XRP": "XRP/USDT",
    "ADA": "ADA/USDT", "AVAX": "AVAX/USDT", "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT", "SUI": "SUI/USDT", "DOT": "DOT/USDT",
}

exchange = ccxt.lbank({"enableRateLimit": True})


# ============================================================================
# ۱. دانلود داده 1H واقعی از LBank
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
    df.to_csv(f"{symbol_key}_1h_lscs_data.csv", index=False)
    print(f"  ✔️ دیتای 1 ساعته {symbol_key} آماده شد (تعداد کندل: {len(df)})")
    return df


# ============================================================================
# ۲. اندیکاتورها — همه فقط رو به گذشته (بدون center=True، بدون Repaint)
# ============================================================================
def calculate_indicators(df):
    df = df.copy()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

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
    return df


# ============================================================================
# ۳. ساخت داده 4H بدون Lookahead
#    هر کندل 4H فقط از لحظه "بسته‌شدن" آن (Date + 4h) در دسترس کندل‌های 1H قرار
#    می‌گیرد؛ merge_asof(direction="backward") تضمین می‌کند کندل 1H هرگز به
#    کندل 4H در حالِ تشکیل دسترسی نداشته باشد.
# ============================================================================
def build_4h_context(df1h):
    df4h = (
        df1h.set_index("Date")
        .resample("4h", label="left", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
        .reset_index()
    )
    df4h = calculate_indicators(df4h)
    df4h["EMA200_prev"] = df4h["EMA_200"].shift(1)
    df4h["available_at"] = df4h["Date"] + pd.Timedelta(hours=4)  # فقط بعد از بسته‌شدن کندل

    ctx_cols = ["available_at", "EMA_20", "EMA_50", "EMA_200", "EMA200_prev", "RSI", "ADX", "ATR"]
    df4h_ctx = df4h[ctx_cols].rename(columns={
        "EMA_20": "EMA20_4H", "EMA_50": "EMA50_4H", "EMA_200": "EMA200_4H",
        "RSI": "RSI_4H", "ADX": "ADX_4H", "ATR": "ATR_4H",
    })

    merged = pd.merge_asof(
        df1h.sort_values("Date"), df4h_ctx.sort_values("available_at"),
        left_on="Date", right_on="available_at", direction="backward",
    )
    return merged


# ============================================================================
# ۴. رژیم بازار (4H) — بدون تغییر بعد از محاسبه (Non-repainting)
# ============================================================================
def add_regime_flags(df):
    df = df.copy()
    long_ok = (
        (df["EMA20_4H"] > df["EMA50_4H"]) & (df["EMA50_4H"] > df["EMA200_4H"]) &
        (df["EMA200_4H"] >= df["EMA200_prev"]) &  # شیب EMA200(4H) واقعاً صعودی نسبت به کندلِ 4H قبلی
        (df["ADX_4H"] >= CONFIG["adx_min"]) & (df["RSI_4H"] > 55)
    )
    short_ok = (
        (df["EMA20_4H"] < df["EMA50_4H"]) & (df["EMA50_4H"] < df["EMA200_4H"]) &
        (df["ADX_4H"] >= CONFIG["adx_min"]) & (df["RSI_4H"] < 45)
    )
    df["regime_long"] = long_ok.fillna(False)
    df["regime_short"] = short_ok.fillna(False)
    return df


# ============================================================================
# ۵. امتیازدهی سیگنال (0-100)
# ============================================================================
def score_signal(adx, sweep_depth_atr, bos_size_atr, retest_dist_atr, vol_ratio, rsi_slope_ok):
    s = 0.0
    s += min(20, (adx / 40) * 20)
    s += min(20, (sweep_depth_atr / 0.5) * 20)
    s += min(15, (bos_size_atr / 1.0) * 15)
    s += max(0, 15 - (retest_dist_atr / CONFIG["retest_atr_mult"]) * 15)
    s += min(15, ((vol_ratio - 1.0) / 0.5) * 15) if vol_ratio > 1 else 0
    s += 15 if rsi_slope_ok else 0
    return max(0, min(100, s))


# ============================================================================
# ۶. موتور اصلی: Sweep -> BOS -> Retest -> Score -> Entry -> Forward Resolution
#    (بدون هیچ نگاهی به آینده برای فیلتر کردن نتیجه معامله)
# ============================================================================
def run_symbol_backtest(symbol, df1h, diag=None):
    """
    diag: دیکشنری شمارنده اختیاری برای تشخیص اینکه سیگنال‌ها در کدام مرحله
    از قیف (Regime -> Sweep -> BOS -> Retest -> Score -> Risk-Bounds -> Trade)
    رد می‌شوند. اگر None باشد، شمارش انجام نمی‌شود.
    """
    if diag is None:
        diag = {}
    for key in ["regime_bars", "atr_ok_bars", "sweep_events", "bos_confirmed",
                "retest_zone_hit", "score_passed", "risk_bounds_passed", "trades_taken"]:
        diag.setdefault(key, 0)

    n = len(df1h)
    lb = CONFIG["structure_lookback"]
    trades = []
    locked_until = -1  # آخرین ایندکسی که هنوز قفلِ معامله باز روی آن است

    i = lb
    while i < n - CONFIG["sweep_confirm_window"] - CONFIG["retest_window"] - 5:
        if i <= locked_until:
            i += 1
            continue  # هیچ سیگنال جدیدی تا رفع‌قفل کامل معامله قبلی صادر نمی‌شود

        row = df1h.iloc[i]
        if not (row["regime_long"] or row["regime_short"]):
            i += 1
            continue
        diag["regime_bars"] += 1
        if pd.isna(row["ATR"]) or row["ATR"] / row["Close"] < CONFIG["min_atr_pct"]:
            i += 1
            continue
        diag["atr_ok_bars"] += 1

        # ساختار مرجع فقط از کندل‌های *قبل* از کندل جاری (i را شامل نمی‌شود)
        lookback = df1h.iloc[i - lb:i]
        struct_high = lookback["High"].max()
        struct_low = lookback["Low"].min()
        atr_now = row["ATR"]

        traded = False

        # ---------------- سناریوی LONG ----------------
        if row["regime_long"] and row["Low"] < struct_low and row["Close"] > struct_low:
            diag["sweep_events"] += 1
            sweep_depth = (struct_low - row["Low"]) / atr_now
            # جست‌وجوی BOS در کندل‌های بعدی (فقط رو به جلو، مثل واقعیت زنده)
            for k in range(i + 1, min(i + 1 + CONFIG["sweep_confirm_window"], n)):
                bos_c = df1h.iloc[k]
                if bos_c["Close"] > struct_high:
                    avg_vol = df1h.iloc[k - lb:k]["Volume"].mean()
                    if bos_c["Volume"] < avg_vol * CONFIG["vol_mult"]:
                        break
                    diag["bos_confirmed"] += 1
                    bos_size_atr = (bos_c["Close"] - struct_high) / atr_now
                    # جست‌وجوی Retest
                    for p in range(k + 1, min(k + 1 + CONFIG["retest_window"], n)):
                        rt_c = df1h.iloc[p]
                        dist_atr = abs(rt_c["Low"] - struct_high) / atr_now
                        in_zone = rt_c["Low"] <= struct_high + CONFIG["retest_atr_mult"] * atr_now
                        if in_zone and rt_c["Close"] > rt_c["Open"] and 45 <= rt_c["RSI"] <= 65:
                            diag["retest_zone_hit"] += 1
                            prev_rsi = df1h.iloc[p - 1]["RSI"]
                            rsi_slope_ok = rt_c["RSI"] > prev_rsi
                            sc = score_signal(row["ADX_4H"], sweep_depth, bos_size_atr,
                                               dist_atr, bos_c["Volume"] / avg_vol, rsi_slope_ok)
                            if sc < CONFIG["min_score"]:
                                break
                            diag["score_passed"] += 1
                            entry_price = rt_c["Close"]
                            swing_low_pullback = df1h.iloc[k:p + 1]["Low"].min()
                            sl = swing_low_pullback - CONFIG["sl_atr_buffer"] * atr_now
                            risk = entry_price - sl
                            if risk <= 0:
                                break
                            risk_atr = risk / atr_now
                            if not (CONFIG["min_sl_atr"] <= risk_atr <= CONFIG["max_sl_atr"]):
                                break
                            diag["risk_bounds_passed"] += 1
                            tp = entry_price + CONFIG["rr"] * risk

                            outcome, exit_idx, bars_held = resolve_forward(
                                df1h, p, entry_price, sl, tp, "LONG"
                            )
                            if outcome in ("WIN", "LOSS"):
                                trades.append(build_trade_record(
                                    symbol, "LONG", entry_price, sl, tp, outcome,
                                    df1h.iloc[p]["Date"], df1h.iloc[exit_idx]["Date"],
                                    bars_held, sc
                                ))
                                locked_until = exit_idx  # قفل تا انتهای همان کندلِ رفع‌تکلیف
                                traded = True
                                diag["trades_taken"] += 1
                            break
                    break
                # اگر ساختار به‌کل نقض شد (سقوط بیشتر) از انتظار BOS خارج شو
                if bos_c["Close"] < struct_low - atr_now:
                    break

        # ---------------- سناریوی SHORT ----------------
        if not traded and row["regime_short"] and row["High"] > struct_high and row["Close"] < struct_high:
            diag["sweep_events"] += 1
            sweep_depth = (row["High"] - struct_high) / atr_now
            for k in range(i + 1, min(i + 1 + CONFIG["sweep_confirm_window"], n)):
                bos_c = df1h.iloc[k]
                if bos_c["Close"] < struct_low:
                    avg_vol = df1h.iloc[k - lb:k]["Volume"].mean()
                    if bos_c["Volume"] < avg_vol * CONFIG["vol_mult"]:
                        break
                    diag["bos_confirmed"] += 1
                    bos_size_atr = (struct_low - bos_c["Close"]) / atr_now
                    for p in range(k + 1, min(k + 1 + CONFIG["retest_window"], n)):
                        rt_c = df1h.iloc[p]
                        dist_atr = abs(rt_c["High"] - struct_low) / atr_now
                        in_zone = rt_c["High"] >= struct_low - CONFIG["retest_atr_mult"] * atr_now
                        if in_zone and rt_c["Close"] < rt_c["Open"] and 35 <= rt_c["RSI"] <= 55:
                            diag["retest_zone_hit"] += 1
                            prev_rsi = df1h.iloc[p - 1]["RSI"]
                            rsi_slope_ok = rt_c["RSI"] < prev_rsi
                            sc = score_signal(row["ADX_4H"], sweep_depth, bos_size_atr,
                                               dist_atr, bos_c["Volume"] / avg_vol, rsi_slope_ok)
                            if sc < CONFIG["min_score"]:
                                break
                            diag["score_passed"] += 1
                            entry_price = rt_c["Close"]
                            swing_high_pullback = df1h.iloc[k:p + 1]["High"].max()
                            sl = swing_high_pullback + CONFIG["sl_atr_buffer"] * atr_now
                            risk = sl - entry_price
                            if risk <= 0:
                                break
                            risk_atr = risk / atr_now
                            if not (CONFIG["min_sl_atr"] <= risk_atr <= CONFIG["max_sl_atr"]):
                                break
                            diag["risk_bounds_passed"] += 1
                            tp = entry_price - CONFIG["rr"] * risk

                            outcome, exit_idx, bars_held = resolve_forward(
                                df1h, p, entry_price, sl, tp, "SHORT"
                            )
                            if outcome in ("WIN", "LOSS"):
                                trades.append(build_trade_record(
                                    symbol, "SHORT", entry_price, sl, tp, outcome,
                                    df1h.iloc[p]["Date"], df1h.iloc[exit_idx]["Date"],
                                    bars_held, sc
                                ))
                                locked_until = exit_idx
                                traded = True
                                diag["trades_taken"] += 1
                            break
                    break
                if bos_c["Close"] > struct_high + atr_now:
                    break

        i = (locked_until + 1) if traded else (i + 1)

    return trades, diag


def resolve_forward(df1h, entry_idx, entry_price, sl, tp, side):
    """
    اسکن گام‌به‌گام روبه‌جلو — تنها راه تعیین نتیجه معامله.
    هیچ اطلاعاتی از فراتر از کندلی که SL/TP در آن لمس می‌شود استفاده نمی‌شود؛
    اگر در همان کندل هر دو سطح لمس شوند (کندل پرنوسان)، به‌صورت محافظه‌کارانه
    LOSS در نظر گرفته می‌شود (فرض بدترین حالت، نه خوش‌بینانه).
    """
    n = len(df1h)
    last = min(entry_idx + 1 + CONFIG["max_forward_bars"], n)
    for j in range(entry_idx + 1, last):
        c = df1h.iloc[j]
        if side == "LONG":
            hit_sl = c["Low"] <= sl
            hit_tp = c["High"] >= tp
        else:
            hit_sl = c["High"] >= sl
            hit_tp = c["Low"] <= tp
        if hit_sl and hit_tp:
            return "LOSS", j, j - entry_idx  # بدترین حالت محافظه‌کارانه
        if hit_sl:
            return "LOSS", j, j - entry_idx
        if hit_tp:
            return "WIN", j, j - entry_idx
    return "OPEN", last - 1, last - 1 - entry_idx


def build_trade_record(symbol, side, entry, sl, tp, outcome, entry_time, exit_time, bars_held, score):
    risk_pct = abs(entry - sl) / entry
    # هزینه تقریبی معامله بر حسب واحد R (کارمزد دو طرف + اسلیپیج دو طرف)
    cost_R = (2 * CONFIG["taker_fee"] + 2 * CONFIG["slippage_pct"]) / risk_pct
    gross_R = CONFIG["rr"] if outcome == "WIN" else -1.0
    net_R = gross_R - cost_R
    return {
        "Symbol": symbol, "Side": side, "Outcome": outcome,
        "Entry": entry, "SL": sl, "TP": tp,
        "EntryTime": entry_time, "ExitTime": exit_time, "BarsHeld": bars_held,
        "Score": round(score, 1), "Gross_R": gross_R, "Net_R": net_R,
    }


# ============================================================================
# ۷. متریک‌های ارزیابی
# ============================================================================
def compute_metrics(trades_df, r_col, label):
    if trades_df.empty:
        print(f"⚠️ هیچ معامله‌ای ({label}) ثبت نشد.")
        return
    total = len(trades_df)
    wins = trades_df[trades_df["Outcome"] == "WIN"]
    losses = trades_df[trades_df["Outcome"] == "LOSS"]
    win_rate = len(wins) / total * 100
    gross_profit = wins[r_col].sum()
    gross_loss = losses[r_col].sum()  # منفی
    net = trades_df[r_col].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.nan
    expectancy = trades_df[r_col].mean()

    equity = trades_df.sort_values("EntryTime")[r_col].cumsum()
    running_max = equity.cummax()
    drawdown = (equity - running_max)
    max_dd = drawdown.min()

    # حداکثر بردهای/باخت‌های متوالی
    outcomes = trades_df.sort_values("EntryTime")["Outcome"].tolist()
    max_w = max_l = cur_w = cur_l = 0
    for o in outcomes:
        if o == "WIN":
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w, max_l = max(max_w, cur_w), max(max_l, cur_l)

    span_days = max((trades_df["EntryTime"].max() - trades_df["EntryTime"].min()).days, 1)
    avg_trades_per_day = total / span_days
    avg_hold_hours = trades_df["BarsHeld"].mean()  # هر بار = 1 ساعت
    sharpe = (trades_df[r_col].mean() / trades_df[r_col].std() * np.sqrt(252)) if trades_df[r_col].std() > 0 else np.nan

    print(f"\n--- نتایج ({label}) ---")
    print(f"🔸 تعداد کل معاملات: {total}")
    print(f"🔸 برنده: {len(wins)}   بازنده: {len(losses)}")
    print(f"🎯 Win Rate: {win_rate:.2f}%")
    print(f"💰 Gross Profit: {gross_profit:.2f}R   Gross Loss: {gross_loss:.2f}R   Net: {net:.2f}R")
    print(f"📈 Profit Factor: {profit_factor:.2f}   Expectancy: {expectancy:.3f}R")
    print(f"📉 Max Drawdown: {max_dd:.2f}R")
    print(f"🔁 حداکثر برد متوالی: {max_w}   حداکثر باخت متوالی: {max_l}")
    print(f"📅 میانگین معامله در روز (کل پورتفوی): {avg_trades_per_day:.2f}")
    print(f"⏱️ میانگین مدت نگهداری: {avg_hold_hours:.1f} ساعت")
    print(f"📐 Sharpe (تقریبی، بر پایه R هر معامله): {sharpe:.2f}")
    print("\nتفکیک به هر نماد:")
    print(trades_df.groupby("Symbol")["Outcome"].value_counts().unstack(fill_value=0))


# ============================================================================
# اجرای پورتفوی روی مجموعه‌ای از دیتافریم‌های آماده (بدون دانلود مجدد)
# ============================================================================
def run_portfolio(symbol_frames):
    """
    symbol_frames: dict[symbol] -> df1h (خروجی build_4h_context، هنوز بدون
    regime_long/short). رژیم با CONFIG فعلی محاسبه می‌شود، پس تغییر پارامترهای
    رژیم (مثل adx_min) هم بدون دانلود مجدد اثر می‌کند.
    """
    all_trades, all_diag = [], {}
    for key, df1h_ctx in symbol_frames.items():
        df1h = add_regime_flags(df1h_ctx)
        df1h = df1h.dropna(subset=["ATR", "ADX", "EMA200_4H"]).reset_index(drop=True)
        trades, diag = run_symbol_backtest(key, df1h)
        all_trades.extend(trades)
        all_diag[key] = diag
    return all_trades, all_diag


def print_diag_table(all_diag):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    diag_df = pd.DataFrame(all_diag).T
    diag_df.loc["TOTAL"] = diag_df.sum(numeric_only=True)
    cols = ["regime_bars", "atr_ok_bars", "sweep_events", "bos_confirmed",
            "retest_zone_hit", "score_passed", "risk_bounds_passed", "trades_taken"]
    print(diag_df[cols].to_string())


# ============================================================================
# ابزار Sensitivity Sweep — تک‌تک پارامترها را حول مقدار پایه تغییر می‌دهد و اثر
# را روی کل پورتفوی گزارش می‌کند (طبق اصل «جلوگیری از Curve Fitting»، نه یک
# حدس دستیِ بی‌پشتوانه).
# ============================================================================
SENSITIVITY_GRID = {
    "vol_mult": [1.0, 1.1, 1.3, 1.5],
    "sweep_confirm_window": [8, 10, 15, 20],
    "retest_window": [4, 6, 10, 14],
    "min_score": [45, 55, 65, 75],
    "adx_min": [15, 18, 20, 25],
}


def run_sensitivity(symbol_frames):
    baseline = {k: CONFIG[k] for k in SENSITIVITY_GRID}
    print("\n" + "=" * 70)
    print("🧪 Sensitivity Sweep — اثر هر پارامتر روی کل پورتفوی (بقیه پارامترها ثابت)")
    print("=" * 70)
    for param, values in SENSITIVITY_GRID.items():
        print(f"\n--- پارامتر: {param}  (مقدار پایه فعلی: {baseline[param]}) ---")
        for v in values:
            CONFIG[param] = v
            trades, _ = run_portfolio(symbol_frames)
            if trades:
                df_t = pd.DataFrame(trades)
                total = len(df_t)
                wr = (df_t["Outcome"] == "WIN").mean() * 100
                net = df_t["Net_R"].sum()
            else:
                total, wr, net = 0, 0.0, 0.0
            marker = " <-- baseline" if v == baseline[param] else ""
            print(f"   {param}={v:<6}  Trades={total:<4}  WinRate={wr:5.1f}%  Net_R={net:6.2f}{marker}")
        CONFIG[param] = baseline[param]  # بازگرداندن به مقدار پایه قبل از پارامتر بعدی


# ============================================================================
# اجرای کامل
# ============================================================================
def main():
    print("=" * 70)
    print("📥 دانلود داده 1H از LBank و ساخت زمینه 4H بدون Lookahead")
    print("=" * 70)

    symbol_frames = {}
    for key, sym in SYMBOLS.items():
        df1h_raw = fetch_1h_data(key, sym, CONFIG["days_back"])
        if df1h_raw is None or len(df1h_raw) < CONFIG["min_1h_bars"]:
            print(f"  ⏭️ داده کافی برای {key} نیست، رد شد.")
            continue
        df1h = calculate_indicators(df1h_raw)
        df1h = build_4h_context(df1h)
        symbol_frames[key] = df1h

    print("\n" + "=" * 70)
    print("🚀 اجرای بک‌تست پایه (پارامترهای فعلی CONFIG)")
    print("=" * 70)
    all_trades, all_diag = run_portfolio(symbol_frames)
    for key, diag in all_diag.items():
        trades_n = diag["trades_taken"]
        print(f"  ✅ {key}: {trades_n} معامله یافت شد.  "
              f"[Regime={diag['regime_bars']} ATR-ok={diag['atr_ok_bars']} "
              f"Sweep={diag['sweep_events']} BOS={diag['bos_confirmed']} "
              f"Retest={diag['retest_zone_hit']} Score-ok={diag['score_passed']} "
              f"Risk-ok={diag['risk_bounds_passed']}]")

    print("\n" + "=" * 70)
    print("🔬 قیف تشخیصی (کجای زنجیره فیلترها سیگنال‌ها رد می‌شوند)")
    print("=" * 70)
    print_diag_table(all_diag)

    print("\n" + "=" * 70)
    print("📊 گزارش تجمیعی نهایی پورتفوی — LSCS (پایه)")
    print("=" * 70)

    if not all_trades:
        print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")
    else:
        df_trades = pd.DataFrame(all_trades)
        df_trades.to_csv("lscs_all_trades.csv", index=False)
        compute_metrics(df_trades, "Gross_R", "قبل از هزینه‌ها")
        compute_metrics(df_trades, "Net_R", "بعد از هزینه‌ها (کارمزد + اسلیپیج)")
        print("\n✨ فایل کامل معاملات: lscs_all_trades.csv")

    # اجرای Sensitivity Sweep روی داده‌های همین اجرا (بدون دانلود مجدد)
    if "--sensitivity" in sys.argv or "-s" in sys.argv:
        run_sensitivity(symbol_frames)


if __name__ == "__main__":
    main()
