import requests

# --- تنظیمات صرافی LBank ---
SYMBOL = "sol_usdt"
LBANK_API_URL = f"https://api.lbank.info/v2/kline.do?symbol={SYMBOL}&size=500&type=15min"

def fetch_historical_candles():
    """دریافت کندل‌های تاریخی از LBank با هدر استاندارد برای جلوگیری از مسدود شدن"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(LBANK_API_URL, headers=headers, timeout=10)
        print(f"وضعیت پاسخ صرافی LBank: {response.status_code}")
        
        data = response.json()
        print(f"محتوای پاسخ: {str(data)[:200]}")
        
        if data.get("result") == "true" or data.get("result") is True or "data" in data:
            raw_candles = data.get("data", [])
            if not raw_candles:
                print("داده‌ای داخل کلید data وجود ندارد.")
                return []
            
            candles = []
            for c in raw_candles:
                candles.append({
                    "time": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                })
            return candles
        else:
            print(f"خطا در نتیجه API ال‌بنک: {data}")
            return []
    except Exception as e:
        print(f"خطای ارتباطی یا پارس JSON: {e}")
        return []

def run_backtest():
    candles = fetch_historical_candles()
    if len(candles) < 50:
        print("تعداد کندل‌های دریافتی برای بک‌تست کافی نیست.")
        return

    total_trades = 0
    wins = 0
    losses = 0
    
    print(f"شروع بک‌تست روی {len(candles)} کندل آخرِ {SYMBOL.upper()}...")
    print("-" * 50)

    for i in range(20, len(candles) - 1):
        prev_candle = candles[i - 1]
        current_candle = candles[i]
        
        close_price = current_candle["close"]
        volume = current_candle["volume"]
        
        avg_volume = sum(c["volume"] for c in candles[i-20:i]) / 20

        is_bullish_momentum = close_price > prev_candle["high"]
        is_volume_confirmed = volume > (avg_volume * 1.5)

        if is_bullish_momentum and is_volume_confirmed:
            entry_price = close_price
            stop_loss = entry_price * 0.985
            take_profit = entry_price * 1.03
            
            total_trades += 1
            
            trade_result = None
            for future_candle in candles[i+1:]:
                if future_candle["low"] <= stop_loss:
                    trade_result = "LOSS"
                    break
                elif future_candle["high"] >= take_profit:
                    trade_result = "WIN"
                    break
            
            if trade_result == "WIN":
                wins += 1
            elif trade_result == "LOSS":
                losses += 1
            else:
                total_trades -= 1

    print("-" * 50)
    print(f"📊 **نتایج نهایی بک‌تست:**")
    print(f"🔹 کل معاملات انجام شده: {total_trades}")
    print(f"✅ معاملات موفق (Win): {wins}")
    print(f"❌ معاملات ناموفق (Loss): {losses}")
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        print(f"🎯 وین‌ریت استراتژی: {win_rate:.2f}%")
    else:
        print("هیچ معامله‌ای با این شرایط در این بازه فعال نشد.")

if __name__ == "__main__":
    run_backtest()
