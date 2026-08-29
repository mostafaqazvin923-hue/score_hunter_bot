import time
import requests
import ccxt

# ==============================================================================
# تنظیمات ربات تلگرام و آیدی چت
# ==============================================================================
# دقت کنید: بخش بعد از : را حتما با توکن واقعی BotFather جایگزین کنید
TELEGRAM_BOT_TOKEN = "7543298101:AAH8j-XXXXXXXXXXXXXXX" 
TELEGRAM_CHAT_ID = "2090120004"                         # آیدی چت مصطفی

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        print(f"ارور در ارسال پیام به تلگرام: {e}")
        return None

# ==============================================================================
# توابع محاسباتی اندیکاتورها
# ==============================================================================

def calculate_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(price * alpha + ema[-1] * (1 - alpha))
    return ema

def calculate_atr(highs, lows, closes, period=14):
    tr_list = []
    for i in range(len(closes)):
        if i == 0:
            tr_list.append(highs[i] - lows[i])
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
    
    atr = [sum(tr_list[:period]) / period]
    alpha = 1.0 / period
    for tr in tr_list[period:]:
        atr.append(tr * alpha + atr[-1] * (1 - alpha))
    return [atr[0]] * (period - 1) + atr

# ==============================================================================
# موتور اصلی ربات (Score Hunter v8.6 - Real Market Data)
# ==============================================================================

class ScoreHunterBot:
    def __init__(self, assets, initial_capital=1000.0, rr_ratio=2.0, risk_per_trade_pct=1.5):
        self.assets = assets
        self.capital = initial_capital
        self.rr_ratio = rr_ratio
        self.risk_per_trade_pct = risk_per_trade_pct
        self.active_trades = {}  # جلوگیری از سیگنال تکراری

    def process_candles(self, asset_name, candles):
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        opens = [c['open'] for c in candles]
        volumes = [c['volume'] for c in candles]

        i = len(candles) - 1
        current = candles[i]
        prev = candles[i-1]

        ema10 = calculate_ema(closes, 10)
        ema30 = calculate_ema(closes, 30)
        ema100 = calculate_ema(closes, 100)
        atr = calculate_atr(highs, lows, closes, 14)

        # ۱. بررسی وضعیت پوزیشن‌های فعال (حد سود یا حد ضرر)
        if asset_name in self.active_trades:
            t = self.active_trades[asset_name]
            
            if t['type'] == 'LONG':
                if current['low'] <= t['sl']:
                    pnl = -t['risk_amount']
                    self.capital += pnl
                    msg = (
                        f"❌ **نتیجه معامله: حد ضرر (SL)**\n\n"
                        f"📌 نماد: #{asset_name.replace('/', '')}\n"
                        f"نوع: LONG\n"
                        f"قیمت خروج: `{t['sl']:.4f}`\n"
                        f"سود/زیان: `{pnl:.2f}$`\n"
                        f"موجودی جدید: `{self.capital:.2f}$`"
                    )
                    send_telegram_message(msg)
                    del self.active_trades[asset_name]

                elif current['high'] >= t['tp']:
                    pnl = t['risk_amount'] * self.rr_ratio
                    self.capital += pnl
                    msg = (
                        f"🎯 **نتیجه معامله: تارگت (TP)**\n\n"
                        f"📌 نماد: #{asset_name.replace('/', '')}\n"
                        f"نوع: LONG\n"
                        f"قیمت خروج: `{t['tp']:.4f}`\n"
                        f"سود/زیان: `+{pnl:.2f}$`\n"
                        f"موجودی جدید: `{self.capital:.2f}$`"
                    )
                    send_telegram_message(msg)
                    del self.active_trades[asset_name]

            elif t['type'] == 'SHORT':
                if current['high'] >= t['sl']:
                    pnl = -t['risk_amount']
                    self.capital += pnl
                    msg = (
                        f"❌ **نتیجه معامله: حد ضرر (SL)**\n\n"
                        f"📌 نماد: #{asset_name.replace('/', '')}\n"
                        f"نوع: SHORT\n"
                        f"قیمت خروج: `{t['sl']:.4f}`\n"
                        f"سود/زیان: `{pnl:.2f}$`\n"
                        f"موجودی جدید: `{self.capital:.2f}$`"
                    )
                    send_telegram_message(msg)
                    del self.active_trades[asset_name]

                elif current['low'] <= t['tp']:
                    pnl = t['risk_amount'] * self.rr_ratio
                    self.capital += pnl
                    msg = (
                        f"🎯 **نتیجه معامله: تارگت (TP)**\n\n"
                        f"📌 نماد: #{asset_name.replace('/', '')}\n"
                        f"نوع: SHORT\n"
                        f"قیمت خروج: `{t['tp']:.4f}`\n"
                        f"سود/زیان: `+{pnl:.2f}$`\n"
                        f"موجودی جدید: `{self.capital:.2f}$`"
                    )
                    send_telegram_message(msg)
                    del self.active_trades[asset_name]

        # ۲. بررسی ورود به معامله جدید
        if asset_name not in self.active_trades:
            c_close = prev['close']
            c_open = prev['open']
            
            bull_trend = (ema10[i-1] > ema30[i-1] > ema100[i-1]) and (ema10[i-1] > ema10[i-3])
            bear_trend = (ema10[i-1] < ema30[i-1] < ema100[i-1]) and (ema10[i-1] < ema10[i-3])

            strong_bull = (c_close > c_open) and ((c_close - c_open) / (prev['high'] - prev['low']) > 0.70) if (prev['high'] - prev['low']) > 0 else False
            strong_bear = (c_open > c_close) and ((c_open - c_close) / (prev['high'] - prev['low']) > 0.70) if (prev['high'] - prev['low']) > 0 else False

            avg_vol = sum(volumes[i-15:i-1]) / 14.0
            high_vol = prev['volume'] > (1.8 * avg_vol)

            long_cond = bull_trend and strong_bull and high_vol
            short_cond = bear_trend and strong_bear and high_vol

            risk_amt = self.capital * (self.risk_per_trade_pct / 100.0)

            if long_cond:
                entry = current['open']
                swing_low = min([candles[j]['low'] for j in range(i-10, i)])
                sl = min(swing_low, entry - (atr[i-1] * 1.2))
                tp = entry + ((entry - sl) * self.rr_ratio)
                
                self.active_trades[asset_name] = {
                    'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt
                }

                msg = (
                    f"🚀 **سیگنال جدید ورود (LONG)**\n\n"
                    f"📌 نماد: #{asset_name.replace('/', '')}\n"
                    f"قیمت ورود (Entry): `{entry:.4f}`\n"
                    f"حد ضرر (SL): `{sl:.4f}`\n"
                    f"حد سود (TP): `{tp:.4f}`\n"
                    f"ریسک معامله: `1.5% ({risk_amt:.2f}$)`\n"
                    f"نسبت R:R برابر 1:2"
                )
                send_telegram_message(msg)

            elif short_cond:
                entry = current['open']
                swing_high = max([candles[j]['high'] for j in range(i-10, i)])
                sl = max(swing_high, entry + (atr[i-1] * 1.2))
                tp = entry - ((sl - entry) * self.rr_ratio)
                
                self.active_trades[asset_name] = {
                    'type': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt
                }

                msg = (
                    f"🔻 **سیگنال جدید ورود (SHORT)**\n\n"
                    f"📌 نماد: #{asset_name.replace('/', '')}\n"
                    f"قیمت ورود (Entry): `{entry:.4f}`\n"
                    f"حد ضرر (SL): `{sl:.4f}`\n"
                    f"حد سود (TP): `{tp:.4f}`\n"
                    f"ریسک معامله: `1.5% ({risk_amt:.2f}$)`\n"
                    f"نسبت R:R برابر 1:2"
                )
                send_telegram_message(msg)

# ==============================================================================
# دریافت قیمت‌های واقعی بازار کریپتو از صرافی
# ==============================================================================
def fetch_real_candles(exchange, symbol, timeframe='15m', limit=150):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        candles = []
        for c in ohlcv:
            candles.append({
                'timestamp': c[0],
                'open': c[1],
                'high': c[2],
                'low': c[3],
                'close': c[4],
                'volume': c[5]
            })
        return candles
    except Exception as e:
        print(f"خطا در دریافت داده برای {symbol}: {e}")
        return None

if __name__ == "__main__":
    # اتصال به صرافی (کوینکس یا بایننس بدون نیاز به API Key برای قیمت عمومی)
    exchange = ccxt.coinex()
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT']
    bot = ScoreHunterBot(assets=symbols)
    
    print("ربات Score Hunter v8.6 فعال شد.")
    print("در حال پایش لایو بازار کریپتو...")

    while True:
        for symbol in symbols:
            candles = fetch_real_candles(exchange, symbol, timeframe='15m', limit=150)
            if candles:
                bot.process_candles(symbol, candles)
            time.sleep(1)  # رعایت محدودیت نرخ درخواست API
            
        time.sleep(60)  # چک کردن بازار در فواصل ۶۰ ثانیه‌ای
