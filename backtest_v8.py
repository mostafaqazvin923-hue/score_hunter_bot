"""
MTBR (Multi-Touch Break-and-Retest) — Portfolio Backtest on LBank
============================================================================
بر پایه تحقیق این جلسه: پرتکرارترین الگوی مستند برای Win Rate بالاتر همراه با
R:R ثابت، «Break-and-Retest» روی سطوح حمایت/مقاومتی است که **چندبار قبلاً لمس
و محترم شمرده شده‌اند» — نه یک Swing تصادفی اخیر. این نسخه دقیقاً همین اصل را
پیاده می‌کند، به‌علاوه فیلتر روند بالادست (1D) طبق توصیه تحقیق («معامله در
جهت روند اصلی»).

هدف اعلام‌شده: Win Rate ≥ ۶۰٪، R:R اکنون ۱:۱.۵ ثابت (تغییر از ۱:۲)، حداکثر ۴-۵ ضرر متوالی.
صادقانه: این هدف بسیار بالاتر از نقطه سربه‌سر (۳۳٪) است و در این جلسه، هر
سیستمی که با نمونه بزرگ (n>300) تست شده، به ۳۵-۴۵٪ رسیده، نه ۶۰٪+. این
نسخه محتمل‌ترین و مستندترین مسیر برای رسیدن به WR بالاتر است، اما نتیجه
واقعی را فقط اجرا روی داده واقعی نشان می‌دهد — نه وعده از پیش.

نکات ضد-تقلب (رعایت‌شده از ابتدا، از جمله رفع باگی که در فایل قبلی پیدا شد):
  1) سطوح حمایت/مقاومت (Pivot) فقط بعد از تأیید N کندل بعدی «شناخته‌شده»
     تلقی می‌شوند (shift(N) + ffill) — نه در لحظه وقوع.
  2) سیگنال (Breakout/Retest/Score) هرگز از کندلی که هنوز به‌طور کامل بسته
     نشده گرفته نمی‌شود.
  3) ورود همیشه در باز شدنِ کندل بعد از کندلِ تأییدِ سیگنال انجام می‌شود —
     هرگز از Close و Open یک کندلِ واحد با هم برای سیگنال+ورود استفاده
     نمی‌شود (این دقیقاً همان باگی بود که در فایل قبلی پیدا و اصلاح شد).
  4) داده 1D (رژیم) با merge_asof و تأخیر بسته‌شدن به کندل‌های 4H متصل
     می‌شود — کندل 4H هرگز به کندل روزانه‌ای که هنوز نبسته دسترسی ندارد.
  5) نتیجه هر معامله فقط با اسکن گام‌به‌گام روبه‌جلو تعیین می‌شود، بدون
     هیچ نگاهی به آینده برای فیلترکردن نتیجه.
  6) هیچ فایلی ذخیره نمی‌شود؛ همه‌چیز مستقیم در کنسول چاپ می‌شود.
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
    "min_1h_bars": 1400,

    # --- تشخیص سطح (Multi-Touch Level) ---
    "pivot_lag": 3,            # کندل قبل/بعد لازم برای تأیید یک Pivot
    "level_lookback": 150,     # چند کندل 4H اخیر برای جست‌وجوی سطوح بررسی شود
    "level_cluster_atr": 0.5,  # دو Pivot در این فاصله (بر حسب ATR) یک سطح حساب می‌شوند
    "min_touches": 2,          # حداقل تعداد لمس برای «سطح معتبر»

    # --- Breakout / Retest ---
    "breakout_vol_mult": 1.2,
    "breakout_window": 12,     # حداکثر کندل بعد از عبور از سطح برای شمردنش Breakout
    "retest_window": 8,
    "retest_atr_mult": 0.35,

    # --- رژیم (1D) ---
    "adx_min": 15,

    # --- مدیریت ریسک (R:R ثابت ۱:۱.۵) ---
    "min_atr_pct": 0.003,
    "sl_atr_buffer": 0.25,
    "min_sl_atr": 0.8,
    "max_sl_atr": 3.0,
    "rr": 1.5,   # تغییر از 2.0 به 1.5 طبق درخواست — نقطه سربه‌سر از 33.3% به 40% بالا می‌رود؛
                 # هدف این است که هدف کوچک‌تر زودتر لمس شود و WR واقعی بیشتر از این افزایش بالا برود.
    "min_score": 60,
    "max_forward_bars": 150,
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
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

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
# ۴. تشخیص سطوح چندبارلمس‌شده (Multi-Touch Levels) — بدون Lookahead
#    Pivot در ایندکس k فقط در ایندکس k+lag «شناخته‌شده» می‌شود.
# ============================================================================
def compute_confirmed_pivots(df, lag):
    n = len(df)
    win = 2 * lag + 1
    is_piv_high = df["High"] == df["High"].rolling(win, center=True).max()
    is_piv_low = df["Low"] == df["Low"].rolling(win, center=True).min()

    piv_high_val = df["High"].where(is_piv_high)
    piv_low_val = df["Low"].where(is_piv_low)

    # shift(lag): مقدار Pivot که در ایندکس k رخ داده، در ایندکس k+lag ظاهر می‌شود
    # (یعنی همان لحظه‌ای که واقعاً «تأیید» می‌شود).
    confirmed_high = piv_high_val.shift(lag)
    confirmed_low = piv_low_val.shift(lag)
    return confirmed_high, confirmed_low


def build_level_clusters(confirmed_vals, atr_series, cluster_atr_mult, up_to_idx, lookback):
    """
    از میان Pivotهای تأییدشده تا ایندکس `up_to_idx` (شامل)، در بازه
    `lookback` کندل اخیر، سطوحی که در فاصله cluster_atr_mult*ATR از هم
    هستند را یک سطح حساب می‌کند و برای هرکدام تعداد لمس را می‌شمارد.
    فقط از داده‌ی تا همین لحظه استفاده می‌شود (بدون آینده).
    """
    start = max(0, up_to_idx - lookback)
    window_vals = confirmed_vals.iloc[start:up_to_idx + 1].dropna()
    if window_vals.empty:
        return []
    atr_now = atr_series.iloc[up_to_idx]
    if pd.isna(atr_now) or atr_now <= 0:
        return []
    tol = cluster_atr_mult * atr_now

    points = sorted(window_vals.values)
    clusters = []
    current = [points[0]]
    for p in points[1:]:
        if abs(p - current[-1]) <= tol:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)
    return [{"level": float(np.mean(c)), "touches": len(c)} for c in clusters]


# ============================================================================
# ۵. امتیازدهی
# ============================================================================
def score_signal(touches, adx, vol_ratio, retest_dist_atr, rsi_ok):
    s = 0.0
    s += min(30, (touches / 4) * 30)
    s += min(20, (adx / 35) * 20)
    s += min(25, (vol_ratio / 1.5) * 25)
    s += max(0, 15 - (retest_dist_atr / CONFIG["retest_atr_mult"]) * 15)
    s += 10 if rsi_ok else 0
    return max(0, min(100, s))


# ============================================================================
# ۶. موتور اصلی
# ============================================================================
def run_symbol_backtest(symbol, df, diag=None):
    if diag is None:
        diag = {}
    for key in ["bars_scanned", "levels_found", "breakout_events",
                "retest_events", "score_passed", "risk_bounds_passed", "trades_taken"]:
        diag.setdefault(key, 0)

    n = len(df)
    lag = CONFIG["pivot_lag"]
    conf_high, conf_low = compute_confirmed_pivots(df, lag)

    trades = []
    locked_until = -1

    start_idx = max(CONFIG["level_lookback"], 210)  # حاشیه امن برای EMA200/ATR
    for i in range(start_idx, n - CONFIG["breakout_window"] - CONFIG["retest_window"] - 3):
        if i <= locked_until:
            continue

        row = df.iloc[i]
        if not (row["regime_long"] or row["regime_short"]):
            continue
        atr_now = row["ATR"]
        if pd.isna(atr_now) or atr_now / row["Close"] < CONFIG["min_atr_pct"]:
            continue
        diag["bars_scanned"] += 1

        # سطوح معتبر تا همین لحظه (i)، فقط از Pivotهای تأییدشده
        resistances = build_level_clusters(conf_high, df["ATR"], CONFIG["level_cluster_atr"], i, CONFIG["level_lookback"])
        supports = build_level_clusters(conf_low, df["ATR"], CONFIG["level_cluster_atr"], i, CONFIG["level_lookback"])
        resistances = [r for r in resistances if r["touches"] >= CONFIG["min_touches"] and r["level"] > row["Close"]]
        supports = [s for s in supports if s["touches"] >= CONFIG["min_touches"] and s["level"] < row["Close"]]
        if resistances or supports:
            diag["levels_found"] += 1

        avg_vol = df.iloc[i - 20:i]["Volume"].mean()
        traded = False

        # ---------------- LONG: شکست نزدیک‌ترین مقاومت معتبر ----------------
        if row["regime_long"] and resistances:
            nearest_res = min(resistances, key=lambda r: r["level"] - row["Close"])
            for k in range(i + 1, min(i + 1 + CONFIG["breakout_window"], n)):
                bos_c = df.iloc[k]
                if bos_c["Close"] > nearest_res["level"] and bos_c["Volume"] >= CONFIG["breakout_vol_mult"] * avg_vol:
                    diag["breakout_events"] += 1
                    for p in range(k + 1, min(k + 1 + CONFIG["retest_window"], n)):
                        rt_c = df.iloc[p]
                        dist_atr = abs(rt_c["Low"] - nearest_res["level"]) / atr_now
                        in_zone = rt_c["Low"] <= nearest_res["level"] + CONFIG["retest_atr_mult"] * atr_now
                        if in_zone and rt_c["Close"] > rt_c["Open"] and rt_c["Close"] > nearest_res["level"]:
                            diag["retest_events"] += 1
                            rsi_ok = 45 <= rt_c["RSI"] <= 68
                            sc = score_signal(nearest_res["touches"], row["ADX_1D"],
                                               bos_c["Volume"] / (avg_vol + 1e-9), dist_atr, rsi_ok)
                            if sc < CONFIG["min_score"]:
                                break
                            diag["score_passed"] += 1
                            # ورود در باز شدنِ کندل بعد از کندل تأیید رتست
                            # (بدون قاطی‌کردن Close/Open یک کندل — رفع باگ قبلی)
                            entry_idx = p + 1
                            if entry_idx >= n:
                                break
                            entry_c = df.iloc[entry_idx]
                            entry_price = entry_c["Open"] * (1 + CONFIG["slippage_pct"])
                            swing_low = df.iloc[k:p + 1]["Low"].min()
                            sl = swing_low - CONFIG["sl_atr_buffer"] * atr_now
                            risk = entry_price - sl
                            if risk <= 0:
                                break
                            risk_atr = risk / atr_now
                            if not (CONFIG["min_sl_atr"] <= risk_atr <= CONFIG["max_sl_atr"]):
                                break
                            diag["risk_bounds_passed"] += 1
                            tp = entry_price + CONFIG["rr"] * risk
                            outcome, exit_idx, bars_held = resolve_forward(df, entry_idx, entry_price, sl, tp, "LONG")
                            if outcome in ("WIN", "LOSS"):
                                trades.append(build_trade_record(
                                    symbol, "LONG", entry_price, sl, tp, outcome,
                                    entry_c["Date"], df.iloc[exit_idx]["Date"], bars_held, sc, nearest_res["touches"]
                                ))
                                locked_until = exit_idx
                                traded = True
                                diag["trades_taken"] += 1
                            break
                    break

        # ---------------- SHORT: شکست نزدیک‌ترین حمایت معتبر ----------------
        if not traded and row["regime_short"] and supports:
            nearest_sup = min(supports, key=lambda s: row["Close"] - s["level"])
            for k in range(i + 1, min(i + 1 + CONFIG["breakout_window"], n)):
                bos_c = df.iloc[k]
                if bos_c["Close"] < nearest_sup["level"] and bos_c["Volume"] >= CONFIG["breakout_vol_mult"] * avg_vol:
                    diag["breakout_events"] += 1
                    for p in range(k + 1, min(k + 1 + CONFIG["retest_window"], n)):
                        rt_c = df.iloc[p]
                        dist_atr = abs(rt_c["High"] - nearest_sup["level"]) / atr_now
                        in_zone = rt_c["High"] >= nearest_sup["level"] - CONFIG["retest_atr_mult"] * atr_now
                        if in_zone and rt_c["Close"] < rt_c["Open"] and rt_c["Close"] < nearest_sup["level"]:
                            diag["retest_events"] += 1
                            rsi_ok = 32 <= rt_c["RSI"] <= 55
                            sc = score_signal(nearest_sup["touches"], row["ADX_1D"],
                                               bos_c["Volume"] / (avg_vol + 1e-9), dist_atr, rsi_ok)
                            if sc < CONFIG["min_score"]:
                                break
                            diag["score_passed"] += 1
                            entry_idx = p + 1
                            if entry_idx >= n:
                                break
                            entry_c = df.iloc[entry_idx]
                            entry_price = entry_c["Open"] * (1 - CONFIG["slippage_pct"])
                            swing_high = df.iloc[k:p + 1]["High"].max()
                            sl = swing_high + CONFIG["sl_atr_buffer"] * atr_now
                            risk = sl - entry_price
                            if risk <= 0:
                                break
                            risk_atr = risk / atr_now
                            if not (CONFIG["min_sl_atr"] <= risk_atr <= CONFIG["max_sl_atr"]):
                                break
                            diag["risk_bounds_passed"] += 1
                            tp = entry_price - CONFIG["rr"] * risk
                            outcome, exit_idx, bars_held = resolve_forward(df, entry_idx, entry_price, sl, tp, "SHORT")
                            if outcome in ("WIN", "LOSS"):
                                trades.append(build_trade_record(
                                    symbol, "SHORT", entry_price, sl, tp, outcome,
                                    entry_c["Date"], df.iloc[exit_idx]["Date"], bars_held, sc, nearest_sup["touches"]
                                ))
                                locked_until = exit_idx
                                traded = True
                                diag["trades_taken"] += 1
                            break
                    break

    return trades, diag


def resolve_forward(df, entry_idx, entry_price, sl, tp, side):
    n = len(df)
    last = min(entry_idx + 1 + CONFIG["max_forward_bars"], n)
    for j in range(entry_idx + 1, last):
        c = df.iloc[j]
        if side == "LONG":
            hit_sl, hit_tp = c["Low"] <= sl, c["High"] >= tp
        else:
            hit_sl, hit_tp = c["High"] >= sl, c["Low"] <= tp
        if hit_sl:
            return "LOSS", j, j - entry_idx
        if hit_tp:
            return "WIN", j, j - entry_idx
    return "OPEN", last - 1, last - 1 - entry_idx


def build_trade_record(symbol, side, entry, sl, tp, outcome, entry_time, exit_time, bars_held, score, touches):
    risk_pct = abs(entry - sl) / entry
    cost_R = (2 * CONFIG["taker_fee"] + 2 * CONFIG["slippage_pct"]) / risk_pct
    gross_R = CONFIG["rr"] if outcome == "WIN" else -1.0
    net_R = gross_R - cost_R
    return {
        "Symbol": symbol, "Side": side, "Outcome": outcome,
        "Entry": round(entry, 5), "SL": round(sl, 5), "TP": round(tp, 5),
        "EntryTime": entry_time, "ExitTime": exit_time, "HoursHeld": bars_held * 4,
        "Score": round(score, 1), "LevelTouches": touches,
        "Gross_R": gross_R, "Net_R": round(net_R, 4),
    }


# ============================================================================
# ۷. متریک‌ها
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

    print(f"\n--- نتایج ({label}) ---")
    print(f"🔸 تعداد کل معاملات: {total}")
    print(f"🔸 برنده: {len(wins)}   بازنده: {len(losses)}")
    print(f"🎯 Win Rate: {win_rate:.2f}%")
    print(f"💰 Gross Profit: {gross_profit:.2f}R   Gross Loss: {gross_loss:.2f}R   Net: {net:.2f}R")
    print(f"📈 Profit Factor: {profit_factor:.2f}   Expectancy: {expectancy:.3f}R")
    print(f"📉 Max Drawdown: {max_dd:.2f}R")
    print(f"🔁 حداکثر برد متوالی: {max_w}   حداکثر باخت متوالی: {max_l}")
    print(f"📅 میانگین معامله در روز: {avg_trades_per_day:.2f}")
    print("\nتفکیک به هر نماد:")
    print(trades_df.groupby("Symbol")["Outcome"].value_counts().unstack(fill_value=0).to_string())


def print_diag_table(all_diag):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    diag_df = pd.DataFrame(all_diag).T
    diag_df.loc["TOTAL"] = diag_df.sum(numeric_only=True)
    cols = ["bars_scanned", "levels_found", "breakout_events", "retest_events",
            "score_passed", "risk_bounds_passed", "trades_taken"]
    print(diag_df[cols].to_string())


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
        df = df.dropna(subset=["ATR", "ADX", "EMA200_1D"]).reset_index(drop=True)

        trades, diag = run_symbol_backtest(key, df)
        print(f"  ✅ {key}: {len(trades)} معامله یافت شد.  "
              f"[Bars={diag['bars_scanned']} Levels={diag['levels_found']} "
              f"Breakout={diag['breakout_events']} Retest={diag['retest_events']} "
              f"Score-ok={diag['score_passed']} Risk-ok={diag['risk_bounds_passed']}]")
        all_trades.extend(trades)
        all_diag[key] = diag

    print("\n" + "=" * 70)
    print("🔬 قیف تشخیصی")
    print("=" * 70)
    print_diag_table(all_diag)

    print("\n" + "=" * 70)
    print("📊 گزارش تجمیعی نهایی پورتفوی — MTBR")
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

    print("\n✨ بک‌تست MTBR به اتمام رسید.")


if __name__ == "__main__":
    main()
