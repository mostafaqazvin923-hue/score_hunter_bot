import numpy as np
import pandas as pd


class FuturesBacktester:

  def __init__(
      self,
      df,
      initial_capital=1000.0,
      margin_per_trade=100.0,
      leverage=50.0,
      risk_reward=2.0,
  ):
    self.df = df.copy()
    self.capital = initial_capital
    self.initial_capital = initial_capital
    self.margin = margin_per_trade
    self.leverage = leverage
    self.position_size = margin_per_trade * leverage
    self.rr = risk_reward

  def prepare_indicators(self):
    # محاسبه اندیکاتورها بدون نگاه به آینده (استفاده از داده‌های گذشته)
    self.df['ema_50'] = self.df['close'].ewm(span=50, adjust=False).mean()
    self.df['ema_200'] = self.df['close'].ewm(span=200, adjust=False).mean()

    # محاسبه ATR برای حد ضرر پویا
    high_low = self.df['high'] - self.df['low']
    high_close = np.abs(self.df['high'] - self.df['shift_close']())  # safe check
    low_close = np.abs(self.df['low'] - self.df['shift_close']())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    self.df['atr'] = true_range.rolling(14).mean()

  def run_backtest(self):
    trades = []
    in_position = False
    entry_price = 0
    sl_price = 0
    tp_price = 0
    position_type = None  # 'LONG' or 'SHORT'

    # شبیه‌سازی گام به گام (جلوگیری از Look-ahead Bias)
    for i in range(200, len(self.df)):
      current_candle = self.df.iloc[i]
      prev_candle = self.df.iloc[i - 1]  # تکیه بر اطلاعات کندل بسته شده

      # اگر در پوزیشن نیستیم، به دنبال سیگنال باشیم
      if not in_position:
        # شرایط لانگ: قیمت بالاای EMA 200 و کراس صعودی EMA 50
        if (
            prev_candle['close'] > prev_candle['ema_200']
            and prev_candle['ema_50'] > prev_candle['ema_200']
        ):
          in_position = True
          position_type = 'LONG'
          entry_price = current_candle['open']  # ورود در Open کندل جدید
          atr_val = prev_candle['atr']
          sl_price = entry_price - (atr_val * 1.5)
          tp_price = entry_price + (atr_val * 1.5 * self.rr)

        # شرایط شورت: قیمت پایین EMA 200 و کراس نزولی EMA 50
        elif (
            prev_candle['close'] < prev_candle['ema_200']
            and prev_candle['ema_50'] < prev_candle['ema_200']
        ):
          in_position = True
          position_type = 'SHORT'
          entry_price = current_candle['open']
          atr_val = prev_candle['atr']
          sl_price = entry_price + (atr_val * 1.5)
          tp_price = entry_price - (atr_val * 1.5 * self.rr)

      # اگر در پوزیشن هستیم، بررسی برخورد با TP یا SL
      else:
        if position_type == 'LONG':
          if current_candle['low'] <= sl_price:
            # ضرر
            loss_amount = (
                self.position_size
                * ((entry_price - sl_price) / entry_price)
            )
            self.capital -= loss_amount
            trades.append(
                {'type': 'LONG', 'result': 'LOSS', 'pnl': -loss_amount}
            )
            in_position = False
          elif current_candle['high'] >= tp_price:
            # سود
            profit_amount = (
                self.position_size
                * ((tp_price - entry_price) / entry_price)
            )
            self.capital += profit_amount
            trades.append(
                {'type': 'LONG', 'result': 'WIN', 'pnl': profit_amount}
            )
            in_position = False

        elif position_type == 'SHORT':
          if current_candle['high'] >= sl_price:
            # ضرر
            loss_amount = (
                self.position_size
                * ((sl_price - entry_price) / entry_price)
            )
            self.capital -= loss_amount
            trades.append(
                {'type': 'SHORT', 'result': 'LOSS', 'pnl': -loss_amount}
            )
            in_position = False
          elif current_candle['low'] <= tp_price:
            # سود
            profit_amount = (
                self.position_size
                * ((entry_price - tp_price) / entry_price)
            )
            self.capital += profit_amount
            trades.append(
                {'type': 'SHORT', 'result': 'WIN', 'pnl': profit_amount}
            )
            in_position = False

    return pd.DataFrame(trades)


# نمونه خروجی گزارش‌گیری متریک‌ها
