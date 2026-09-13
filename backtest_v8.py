"""
TPCS (Trend-Pullback Continuation System) — Portfolio Backtest on LBank
============================================================================
این نسخه از پایه بازطراحی شده است. طراحی قبلی (Sweep -> BOS -> Retest به‌صورت
زنجیره‌ای) از نظر ساختاری ایراد داشت: هر مرحله یک رویداد نادر بود و چون باید
پشت‌سرِ هم رخ می‌دادند، احتمالات در هم ضرب می‌شدند و تعداد معامله به‌شدت پایین
می‌آمد (~۲٪ از موقعیت‌های واجد رژیم). نسخه جدید یک الگوی تک‌مرحله‌ای و
پرتکرارتر (بازگشت به EMA20 در جهت روند + تأیید مومنتوم) را بررسی می‌کند —
همان اصول حرفه‌ای (Regime + ATR-Stop + R:R=1:2 ثابت + Score + بدون Lookahead
+ قفل هم‌پوشانی) حفظ شده، فقط زنجیره رویدادهای نادر حذف شده است.

نکات ضد-تقلب (بدون تغییر نسبت به نسخه‌های قبل):
  1) داده 4H با merge_asof و "available_at = close_time + 4h" ساخته می‌شود —
     هر کندل 1H فقط به آخرین کندل 4H که واقعاً بسته شده دسترسی دارد.
  2) هیچ نگاه به آینده برای فیلتر کردن نتیجه معامله وجود ندارد؛ نتیجه فقط با
     اسکن گام‌به‌گام روبه‌جلو (resolve_forward) تعیین می‌شود.
  3) Overlap Lock: تا وقتی تکلیف معامله باز روشن نشود (حتی روی همان کندلِ
     رفع‌تکلیف)، سیگنال جدید صادر نمی‌شود. اسکن از exit_idx+1 از سر گرفته می‌شود.
  4) همه اندیکاتورها فقط rolling/ewm رو به گذشته هستند، بدون center=True.
  5) هیچ فایلی (CSV/خروجی) ذخیره نمی‌شود؛ همه‌چیز مستقیم در کنسول چاپ می‌شود،
     چون اجرای CI/GitHub Actions به پوشه خروجی دائمی دسترسی ندارد.
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
# پیکربندی
# ============================================================================
CONFIG = {
    "days_back": 365,
    "min_1h_bars": 1400,        # حداقل کندل لازم برای EMA200(4H) با حاشیه امن
    "adx_min": 15,              # حداقل ADX(4H) برای رژیم روند‌دار (سخت‌گیری متوسط)
    "pullback_band_atr": 0.40,  # عرض ناحیه Pullback حول EMA20 (بر حسب ATR)
    "min_atr_pct": 0.003,       # حداقل ATR/Price برای اجازه معامله (0.3%)
    "sl_lookback": 5,           # کندل‌های قبل برای یافتن Swing کوچک جهت SL
    "sl_atr_buffer": 0.30,      # بافر ATR اضافه به Swing برای SL
    "min_sl_atr": 1.0,          # حداقل فاصله SL بر حسب ATR (بازگردانده‌شده از 0.5 — استاپ تنگ باعث افت شدید بعد از هزینه می‌شد)
    "max_sl_atr": 3.0,          # حداکثر فاصله SL بر حسب ATR
    "rr": 2.0,                  # Risk:Reward ثابت — 1:2
    "vol_mult": 0.9,            # حداقل نسبت حجم کندل Trigger به میانگین (ملایم)
    "min_score": 50,            # حداقل امتیاز از 100 برای پذیرش سیگنال
    "max_forward_bars": 200,    # سقف اسکن روبه‌جلو برای رسیدن SL/TP
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
# ۳. ساخت داده 4H بدون Lookahead (merge_asof با تأخیر بسته‌شدن کندل)
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
    # پایداری روند: ADX باید حداقل 3 کندل 4H متوالی (قبل از کندل جاری هم) بالای
    # آستانه بماند، نه فقط یک لحظه — جلوگیری از Whipsaw در بازارهای رنج
    df4h["ADX_min3"] = df4h["ADX"].rolling(3).min()
    df4h["available_at"] = df4h["Date"] + pd.Timedelta(hours=4)

    ctx_cols = ["available_at", "EMA_20", "EMA_50", "EMA_200", "EMA200_prev", "RSI", "ADX", "ADX_min3"]
    df4h_ctx = df4h[ctx_cols].rename(columns={
        "EMA_20": "EMA20_4H", "EMA_50": "EMA50_4H", "EMA_200": "EMA200_4H",
        "RSI": "RSI_4H", "ADX": "ADX_4H", "ADX_min3": "ADX_4H_min3",
    })

    merged = pd.merge_asof(
        df1h.sort_values("Date"), df4h_ctx.sort_values("available_at"),
        left_on="Date", right_on="available_at", direction="backward",
    )
    return merged


def add_regime_flags(df):
    df = df.copy()
    long_ok = (
        (df["EMA20_4H"] > df["EMA50_4H"]) & (df["EMA50_4H"] > df["EMA200_4H"]) &
        (df["EMA200_4H"] >= df["EMA200_prev"]) &
        (df["ADX_4H_min3"] >= CONFIG["adx_min"]) & (df["RSI_4H"] > 50)
    )
    short_ok = (
        (df["EMA20_4H"] < df["EMA50_4H"]) & (df["EMA50_4H"] < df["EMA200_4H"]) &
        (df["ADX_4H_min3"] >= CONFIG["adx_min"]) & (df["RSI_4H"] < 50)
    )
    df["regime_long"] = long_ok.fillna(False)
    df["regime_short"] = short_ok.fillna(False)
    return df


# ============================================================================
# ۴. امتیازدهی سیگنال (0-100)
# ============================================================================
def score_signal(adx, pullback_tightness, rsi_slope_mag, vol_ratio, ema_stack_strength):
    s = 0.0
    s += min(25, (adx / 35) * 25)
    s += min(20, (1 - pullback_tightness) * 20)          # هرچه نزدیک‌تر به EMA20، امتیاز بیشتر
    s += min(20, (rsi_slope_mag / 5) * 20)
    s += min(20, (vol_ratio / 1.3) * 20)
    s += min(15, ema_stack_strength * 15)
    return max(0, min(100, s))


# ============================================================================
# ۵. موتور اصلی — تک‌مرحله‌ای: در هر کندلِ واجد رژیم، شرط Pullback+Trigger را
#    مستقیم بررسی می‌کند (بدون جست‌وجوی زنجیره‌ای چندمرحله‌ای در آینده).
# ============================================================================
def run_symbol_backtest(symbol, df1h, diag=None):
    if diag is None:
        diag = {}
    for key in ["regime_bars", "atr_ok_bars", "pullback_zone_hit",
                "trigger_confirmed", "score_passed", "risk_bounds_passed", "trades_taken"]:
        diag.setdefault(key, 0)

    n = len(df1h)
    lb = CONFIG["sl_lookback"]
    trades = []
    locked_until = -1

    for i in range(20, n - 5):
        if i <= locked_until:
            continue

        row = df1h.iloc[i]
        if not (row["regime_long"] or row["regime_short"]):
            continue
        diag["regime_bars"] += 1

        atr_now = row["ATR"]
        if pd.isna(atr_now) or atr_now / row["Close"] < CONFIG["min_atr_pct"]:
            continue
        diag["atr_ok_bars"] += 1

        ema20 = row["EMA_20"]
        band = CONFIG["pullback_band_atr"] * atr_now
        avg_vol = df1h.iloc[i - 20:i]["Volume"].mean()
        prev_rsi = df1h.iloc[i - 1]["RSI"]

        traded = False

        # ---------------- LONG: بازگشت به EMA20 در روند صعودی ----------------
        if row["regime_long"] and row["Close"] > row["EMA_50"]:
            touched_zone = row["Low"] <= ema20 + band and row["High"] >= ema20 - band
            if touched_zone:
                diag["pullback_zone_hit"] += 1
                bullish_trigger = row["Close"] > row["Open"] and 40 <= row["RSI"] <= 62
                rsi_rising = row["RSI"] > prev_rsi
                vol_ok = row["Volume"] >= CONFIG["vol_mult"] * avg_vol
                if bullish_trigger and rsi_rising and vol_ok:
                    diag["trigger_confirmed"] += 1
                    pullback_tightness = min(1.0, abs(row["Close"] - ema20) / (band + 1e-9))
                    rsi_slope_mag = row["RSI"] - prev_rsi
                    vol_ratio = row["Volume"] / (avg_vol + 1e-9)
                    ema_stack = 1.0 if (row["EMA_20"] > row["EMA_50"] > row["EMA_200"]) else 0.5
                    sc = score_signal(row["ADX_4H"], pullback_tightness, rsi_slope_mag, vol_ratio, ema_stack)

                    if sc >= CONFIG["min_score"]:
                        diag["score_passed"] += 1
                        entry_price = row["Close"]
                        swing_low = df1h.iloc[i - lb:i + 1]["Low"].min()
                        sl = swing_low - CONFIG["sl_atr_buffer"] * atr_now
                        risk = entry_price - sl
                        risk_atr = risk / atr_now if risk > 0 else -1
                        if risk > 0 and CONFIG["min_sl_atr"] <= risk_atr <= CONFIG["max_sl_atr"]:
                            diag["risk_bounds_passed"] += 1
                            tp = entry_price + CONFIG["rr"] * risk
                            outcome, exit_idx, bars_held = resolve_forward(df1h, i, entry_price, sl, tp, "LONG")
                            if outcome in ("WIN", "LOSS"):
                                trades.append(build_trade_record(
                                    symbol, "LONG", entry_price, sl, tp, outcome,
                                    row["Date"], df1h.iloc[exit_idx]["Date"], bars_held, sc
                                ))
                                locked_until = exit_idx
                                traded = True
                                diag["trades_taken"] += 1

        # ---------------- SHORT: بازگشت به EMA20 در روند نزولی ----------------
        if not traded and row["regime_short"] and row["Close"] < row["EMA_50"]:
            touched_zone = row["High"] >= ema20 - band and row["Low"] <= ema20 + band
            if touched_zone:
                diag["pullback_zone_hit"] += 1
                bearish_trigger = row["Close"] < row["Open"] and 38 <= row["RSI"] <= 60
                rsi_falling = row["RSI"] < prev_rsi
                vol_ok = row["Volume"] >= CONFIG["vol_mult"] * avg_vol
                if bearish_trigger and rsi_falling and vol_ok:
                    diag["trigger_confirmed"] += 1
                    pullback_tightness = min(1.0, abs(row["Close"] - ema20) / (band + 1e-9))
                    rsi_slope_mag = prev_rsi - row["RSI"]
                    vol_ratio = row["Volume"] / (avg_vol + 1e-9)
                    ema_stack = 1.0 if (row["EMA_20"] < row["EMA_50"] < row["EMA_200"]) else 0.5
                    sc = score_signal(row["ADX_4H"], pullback_tightness, rsi_slope_mag, vol_ratio, ema_stack)

                    if sc >= CONFIG["min_score"]:
                        diag["score_passed"] += 1
                        entry_price = row["Close"]
                        swing_high = df1h.iloc[i - lb:i + 1]["High"].max()
                        sl = swing_high + CONFIG["sl_atr_buffer"] * atr_now
                        risk = sl - entry_price
                        risk_atr = risk / atr_now if risk > 0 else -1
                        if risk > 0 and CONFIG["min_sl_atr"] <= risk_atr <= CONFIG["max_sl_atr"]:
                            diag["risk_bounds_passed"] += 1
                            tp = entry_price - CONFIG["rr"] * risk
                            outcome, exit_idx, bars_held = resolve_forward(df1h, i, entry_price, sl, tp, "SHORT")
                            if outcome in ("WIN", "LOSS"):
                                trades.append(build_trade_record(
                                    symbol, "SHORT", entry_price, sl, tp, outcome,
                                    row["Date"], df1h.iloc[exit_idx]["Date"], bars_held, sc
                                ))
                                locked_until = exit_idx
                                traded = True
                                diag["trades_taken"] += 1

    return trades, diag


def resolve_forward(df1h, entry_idx, entry_price, sl, tp, side):
    """اسکن گام‌به‌گام روبه‌جلو — تنها راه تعیین نتیجه معامله (بدون نگاه به آینده)."""
    n = len(df1h)
    last = min(entry_idx + 1 + CONFIG["max_forward_bars"], n)
    for j in range(entry_idx + 1, last):
        c = df1h.iloc[j]
        if side == "LONG":
            hit_sl, hit_tp = c["Low"] <= sl, c["High"] >= tp
        else:
            hit_sl, hit_tp = c["High"] >= sl, c["Low"] <= tp
        if hit_sl:  # اگر هردو در یک کندل لمس شوند، محافظه‌کارانه LOSS در نظر گرفته می‌شود
            return "LOSS", j, j - entry_idx
        if hit_tp:
            return "WIN", j, j - entry_idx
    return "OPEN", last - 1, last - 1 - entry_idx


def build_trade_record(symbol, side, entry, sl, tp, outcome, entry_time, exit_time, bars_held, score):
    risk_pct = abs(entry - sl) / entry
    cost_R = (2 * CONFIG["taker_fee"] + 2 * CONFIG["slippage_pct"]) / risk_pct
    gross_R = CONFIG["rr"] if outcome == "WIN" else -1.0
    net_R = gross_R - cost_R
    return {
        "Symbol": symbol, "Side": side, "Outcome": outcome,
        "Entry": round(entry, 4), "SL": round(sl, 4), "TP": round(tp, 4),
        "EntryTime": entry_time, "ExitTime": exit_time, "BarsHeld": bars_held,
        "Score": round(score, 1), "Gross_R": gross_R, "Net_R": round(net_R, 4),
    }


# ============================================================================
# ۶. متریک‌های ارزیابی
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
    gross_loss = losses[r_col].sum()
    net = trades_df[r_col].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.nan
    expectancy = trades_df[r_col].mean()

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
    avg_trades_per_day = total / span_days
    avg_hold_hours = trades_df["BarsHeld"].mean()
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
    print(f"📐 Sharpe (تقریبی): {sharpe:.2f}")
    print("\nتفکیک به هر نماد:")
    print(trades_df.groupby("Symbol")["Outcome"].value_counts().unstack(fill_value=0).to_string())


def print_diag_table(all_diag):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    diag_df = pd.DataFrame(all_diag).T
    diag_df.loc["TOTAL"] = diag_df.sum(numeric_only=True)
    cols = ["regime_bars", "atr_ok_bars", "pullback_zone_hit", "trigger_confirmed",
            "score_passed", "risk_bounds_passed", "trades_taken"]
    print(diag_df[cols].to_string())


# ============================================================================
# Score Sweep — بدون دانلود مجدد، فقط min_score را عبور می‌دهد تا نقطه واقعی
# تعادل بین تعداد معامله و Win Rate/Expectancy بعد از هزینه پیدا شود.
# ============================================================================
def run_score_sweep(regime_frames, score_values):
    baseline_score = CONFIG["min_score"]
    print("\n" + "=" * 70)
    print("🧪 Score Sweep — همان داده‌ها، فقط آستانه min_score عوض می‌شود")
    print("=" * 70)
    for sc_val in score_values:
        CONFIG["min_score"] = sc_val
        all_trades = []
        for key, df1h in regime_frames.items():
            trades, _ = run_symbol_backtest(key, df1h)
            all_trades.extend(trades)
        if all_trades:
            df_t = pd.DataFrame(all_trades)
            total = len(df_t)
            wr = (df_t["Outcome"] == "WIN").mean() * 100
            net_before = df_t["Gross_R"].sum()
            net_after = df_t["Net_R"].sum()
            span_days = max((df_t["EntryTime"].max() - df_t["EntryTime"].min()).days, 1)
            per_day = total / span_days
            outcomes = df_t.sort_values("EntryTime")["Outcome"].tolist()
            max_l = cur_l = 0
            for o in outcomes:
                cur_l = cur_l + 1 if o == "LOSS" else 0
                max_l = max(max_l, cur_l)
        else:
            total, wr, net_before, net_after, per_day, max_l = 0, 0.0, 0.0, 0.0, 0.0, 0
        marker = " <-- تنظیم فعلی" if sc_val == baseline_score else ""
        print(f"  min_score={sc_val:<4} Trades={total:<4} ({per_day:4.2f}/day)  "
              f"WinRate={wr:5.1f}%  Net(before)={net_before:7.2f}R  Net(after)={net_after:7.2f}R  "
              f"MaxLossStreak={max_l}{marker}")
    CONFIG["min_score"] = baseline_score


# ============================================================================
# اجرای کامل — همه‌چیز مستقیم در کنسول چاپ می‌شود، هیچ فایلی ذخیره نمی‌شود
# ============================================================================
def main():
    print("=" * 70)
    print("📥 دانلود داده 1H از LBank و ساخت زمینه 4H بدون Lookahead")
    print("=" * 70)

    all_trades, all_diag, regime_frames = [], {}, {}
    for key, sym in SYMBOLS.items():
        df1h_raw = fetch_1h_data(key, sym, CONFIG["days_back"])
        if df1h_raw is None or len(df1h_raw) < CONFIG["min_1h_bars"]:
            print(f"  ⏭️ داده کافی برای {key} نیست، رد شد.")
            continue

        df1h = calculate_indicators(df1h_raw)
        df1h = build_4h_context(df1h)
        df1h = add_regime_flags(df1h)
        df1h = df1h.dropna(subset=["ATR", "ADX", "EMA200_4H", "ADX_4H_min3"]).reset_index(drop=True)
        regime_frames[key] = df1h  # کش می‌شود تا Score Sweep نیازی به دانلود مجدد نداشته باشد

        trades, diag = run_symbol_backtest(key, df1h)
        print(f"  ✅ {key}: {len(trades)} معامله یافت شد.  "
              f"[Regime={diag['regime_bars']} ATR-ok={diag['atr_ok_bars']} "
              f"Pullback={diag['pullback_zone_hit']} Trigger={diag['trigger_confirmed']} "
              f"Score-ok={diag['score_passed']} Risk-ok={diag['risk_bounds_passed']}]")
        all_trades.extend(trades)
        all_diag[key] = diag

    print("\n" + "=" * 70)
    print("🔬 قیف تشخیصی")
    print("=" * 70)
    print_diag_table(all_diag)

    print("\n" + "=" * 70)
    print("📊 گزارش تجمیعی نهایی پورتفوی — TPCS")
    print("=" * 70)

    if not all_trades:
        print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")
        return

    df_trades = pd.DataFrame(all_trades)

    compute_metrics(df_trades, "Gross_R", "قبل از هزینه‌ها")
    compute_metrics(df_trades, "Net_R", "بعد از هزینه‌ها (کارمزد + اسلیپیج)")

    print("\n" + "=" * 70)
    print("📋 لیست کامل معاملات (مستقیم در همین خروجی، بدون فایل جداگانه)")
    print("=" * 70)
    pd.set_option("display.max_rows", None)
    print(df_trades.sort_values("EntryTime").to_string(index=False))

    # Score Sweep همیشه اجرا می‌شود (هزینه محاسباتی‌اش ناچیز است، چون داده تکرار دانلود نمی‌شود)
    run_score_sweep(regime_frames, [45, 50, 55, 60, 65, 70, 75, 80, 85])

    print("\n✨ بک‌تست TPCS به اتمام رسید.")


if __name__ == "__main__":
    main()
