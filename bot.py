import time
import requests

# ==============================================================================
# تنظیمات ربات تلگرام و آیدی چت (ثبت شده برای Mostafa Signal Bot)
# ==============================================================================
TELEGRAM_BOT_TOKEN = "8937303392:AAGXDckoHV61vY6G0B4VFcHMi90YbhY-jiY"
TELEGRAM_CHAT_ID = "2090120004"

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
# توابع محاسباتی اندیکاتورها (منطبق بر v8.6)
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
# موتور اصلی ربات (Score Hunter v8.6)
# ==============================================================================

class ScoreHunterBot:
    def __init__(self, assets, initial_capital=1000.0, rr_ratio=2.0, risk_per_trade_pct=1.5):
        self.assets = assets
        self.capital = initial_capital
        self.rr_ratio = rr_ratio
        self.risk_per_trade_pct = risk_per_trade_pct
        self.active_trades = {}  # جلوگیری قطعی از ارسال سیگنال تکراری

    def process_candles(self, asset_name, candles):
        if len(candles) < 100:
            return

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

        # ۱. بررسی پوزیشن‌های فعال (خروج با TP یا SL)
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

        # ۲. بررسی شرایط سیگنال جدید (فقط در صورت عدم وجود پوزیشن باز روی نماد)
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
                
                sl_pct = ((entry - sl) / entry) * 100
                tp_pct = ((tp - entry) / entry) * 100

                self.active_trades[asset_name] = {
                    'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt
                }

                msg = (
                    f"🚀 **سیگنال جدید ورود (LONG)**\n\n"
                    f"📌 نماد: #{asset_name.replace('/', '')}\n"
                    f"قیمت ورود (Entry): `{entry:.4f}`\n"
                    f"حد ضرر (SL): `{sl:.4f}` `(-{sl_pct:.2f}%)`\n"
                    f"حد سود (TP): `{tp:.4f}` `(+{tp_pct:.2f}%)`\n"
                    f"ریسک معامله: `1.5% ({risk_amt:.2f}$)`\n"
                    f"نسبت R:R برابر 1:2"
                )
                send_telegram_message(msg)

            elif short_cond:
                entry = current['open']
                swing_high = max([candles[j]['high'] for j in range(i-10, i)])
                sl = max(swing_high, entry + (atr[i-1] * 1.2))
                tp = entry - ((sl - entry) * self.rr_ratio)
                
                sl_pct = ((sl - entry) / entry) * 100
                tp_pct = ((entry - tp) / entry) * 100

                self.active_trades[asset_name] = {
                    'type': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt
                }

                msg = (
                    f"🔻 **سیگنال جدید ورود (SHORT)**\n\n"
                    f"📌 نماد: #{asset_name.replace('/', '')}\n"
                    f"قیمت ورود (Entry): `{entry:.4f}`\n"
                    f"حد ضرر (SL): `{sl:.4f}` `(-{sl_pct:.2f}%)`\n"
                    f"حد سود (TP): `{tp:.4f}` `(+{tp_pct:.2f}%)`\n"
                    f"ریسک معامله: `1.5% ({risk_amt:.2f}$)`\n"
                    f"نسبت R:R برابر 1:2"
                )
                send_telegram_message(msg)

# ==============================================================================
# دریافت داده‌های واقعی بازار از API عمومی CoinEx
# ==============================================================================

def fetch_real_candles_coinex(symbol, timeframe='15m', limit=150):
    market = symbol.replace('/', '')
    url = f"https://api.coinex.com/v2/spot/kline?market={market}&limit={limit}&interval={timeframe}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('code') == 0:
            candles = []
            for c in data['data']:
                candles.append({
                    'timestamp': c['created_at'],
                    'open': float(c['open']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'close': float(c['close']),
                    'volume': float(c['volume'])
                })
            return candles
        return None
    except Exception as e:
        print(f"خطا در دریافت داده برای {symbol}: {e}")
        return None

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT']
    bot = ScoreHunterBot(assets=symbols)
    
    print("ربات Mostafa Signal Bot فعال شد.")
    print("در حال دریافت لایو داده‌های بازار کریپتو...")

    for symbol in symbols:
        candles = fetch_real_candles_coinex(symbol, timeframe='15m', limit=150)
        if candles:
            bot.process_candles(symbol, candles)
            print(f"پردازش {symbol} با موفقیت انجام شد.")
        time.sleep(1)
