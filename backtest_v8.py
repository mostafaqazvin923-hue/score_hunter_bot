import json
import time
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (Smart Money Concept: BOS + FVG Reversal/Retest)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_CANDLES = 8000
PAGE_LIMIT = 1000

INITIAL_CAPITAL_PER_SYMBOL = 500.0  
FIXED_RISK_AMOUNT = 50.0            # ریسک ثابت ۵۰ دلار
TARGET_RR = 2.5                     # ریوارد ۱ به ۲.۵ برای جبران وین‌ریت ساختاری

# ============================================================
# HTTP & PAGINATION DATA DOWNLOADER
# ============================================================

def download_klines(symbol, timeframe, target_count):
    all_rows = {}
    end_time = None
    pages = 0
    max_pages = 15

    while len(all_rows) < target_count and pages < max_pages:
        pages += 1
        url = f"{BASE_URL}/spot/kline?market={symbol}&period={timeframe}&limit={PAGE_LIMIT}"
        if end_time:
            url += f"&end_time={end_time}"

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-SMC"}
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw)
        except Exception:
            break

        if not isinstance(payload, dict) or payload.get("code") != 0:
            break

        rows = payload.get("data", [])
        if not rows:
            break

        added = 0
        for row in rows:
            try:
                if isinstance(row, dict):
                    ts = int(float(row.get("created_at", row.get("time", 0))))
                    op = float(row.get("open", 0))
                    hi = float(row.get("high", 0))
                    lo = float(row.get("low", 0))
                    cl = float(row.get("close", 0))
                    vol = float(row.get("volume", 0))
                elif isinstance(row, list) and len(row) >= 6:
                    ts = int(float(row[0]))
                    op = float(row[1])
                    cl = float(row[2])
                    hi = float(row[3])
                    lo = float(row[4])
                    vol = float(row.get("volume", 0))
                else:
                    continue

                if ts > 100000000000:
                    ts = ts // 1000

                if hi <= 0 or lo <= 0 or op <= 0 or cl <= 0:
                    continue

                if ts not in all_rows:
                    all_rows[ts] = {
                        "timestamp": ts,
                        "open": op,
                        "high": hi,
                        "low": lo,
                        "close": cl,
                        "volume": vol,
                    }
                    added += 1
            except Exception:
                continue

        if added == 0:
            break

        oldest_ts = min(all_rows.keys())
        end_time = oldest_ts * 1000
        time.sleep(0.1)

    candles = list(all_rows.values())
    candles.sort(key=lambda x: x["timestamp"])

    if len(candles) > target_count:
        candles = candles[-target_count:]

    return candles

# ============================================================
# SMART MONEY ENGINE (BOS & FVG Detection)
# ============================================================

def run_backtest():
    total_portfolio_profit = 0.0
    total_trades_all = 0
    total_wins_all = 0
    total_losses_all = 0

    print("=" * 60)
    print("گزارش تست استراتژی اسمارت مانی (SMC: BOS + FVG Retest / R:R 1:2.5):")
    print("=" * 60)

    for symbol in SYMBOLS:
        candles = download_klines(symbol, TIMEFRAME, TARGET_CANDLES)
        if len(candles) < 200:
            continue

        trades = []
        active_position = None

        # اسکن ساختار بازار و گپ‌های ارزش منصفانه (FVG)
        for i in range(10, len(candles) - 1):
            c = candles[i]
            prev_c = candles[i-1]
            prev2_c = candles[i-2]

            if active_position is not None:
                if c["low"] <= active_position["sl"]:
                    trades.append("LOSS")
                    active_position = None
                elif c["high"] >= active_position["tp"]:
                    trades.append("WIN")
                    active_position = None
                else:
                    continue

            # 1. تشخیص Break of Structure (شکست سقف قبلی با کندل قدرتمند صعودی)
            recent_highs = max(x["high"] for x in candles[i-10:i])
            is_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.5

            # 2. تشخیص Fair Value Gap (گپ بین کندل i-2 و کندل i)
            # در روند صعودی: Low کندل بعد از گپ بالاتر از High کندل قبل از گپ است
            has_bullish_fvg = prev2_c["high"] < c["low"]

            if is_bos and has_bullish_fvg:
                # ناحیه ورود روی ترید فیر ولیو گپ (FVG Zone)
                entry_price = (prev2_c["high"] + c["low"]) / 2.0
                
                # اگر قیمت فعلی از ناحیه رد شده، منتظر پولبک به FVG می‌شویم یا استاپ را زیر کف اخیر قرار می‌دهیم
                stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.003)
                risk_distance = entry_price - stop_loss

                if risk_distance <= 0 or risk_distance / entry_price > 0.03:
                    continue  # ریسک غیرعادی است

                take_profit = entry_price + (risk_distance * TARGET_RR)

                # شبیه‌سازی آینده قیمت برای این ستاپ ساختاری
                trade_result = None
                for future_c in candles[i+1:]:
                    if future_c["low"] <= stop_loss:
                        trade_result = "LOSS"
                        break
                    elif future_c["high"] >= take_profit:
                        trade_result = "WIN"
                        break

                if trade_result:
                    trades.append(trade_result)
                else:
                    active_position = {"tp": take_profit, "sl": stop_loss}

        wins = trades.count("WIN")
        losses = trades.count("LOSS")
        symbol_total_trades = wins + losses
        symbol_win_rate = (wins / symbol_total_trades * 100) if symbol_total_trades > 0 else 0.0
        
        symbol_profit = (wins * FIXED_RISK_AMOUNT * TARGET_RR) - (losses * FIXED_RISK_AMOUNT)
        
        total_trades_all += symbol_total_trades
        total_wins_all += wins
        total_losses_all += losses
        total_portfolio_profit += symbol_profit

        print(f"\nارز: {symbol}")
        print(f"  - تعداد کل معاملات: {symbol_total_trades}")
        print(f"  - موفق (Win): {wins} | ناموفق (Loss): {losses}")
        print(f"  - وین‌ریت: {symbol_win_rate:.2f}%")
        print(f"  - سود/زیان خالص: {symbol_profit:+.2f}$")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("برآیند نهایی سبد اسمارت مانی (SMC):")
    print("=" * 60)
    print(f"کل معاملات کل سبد: {total_trades_all}")
    print(f"معاملات موفق (Win): {total_wins_all}")
    print(f"معاملات ناموفق (Loss): {total_losses_all}")
    
    portfolio_wr = (total_wins_all / total_trades_all * 100) if total_trades_all > 0 else 0.0
    initial_total_capital = len(SYMBOLS) * INITIAL_CAPITAL_PER_SYMBOL
    final_capital = initial_total_capital + total_portfolio_profit
    portfolio_roi = (total_portfolio_profit / initial_total_capital) * 100

    print(f"وین‌ریت کل سبد: {portfolio_wr:.2f}%")
    print(f"سرمایه نهایی کل سبد: {final_capital:.2f}$")
    print(f"سود خالص کل سبد: {total_portfolio_profit:+.2f}$ ({portfolio_roi:+.2f}%)")

if __name__ == "__main__":
    run_backtest()
