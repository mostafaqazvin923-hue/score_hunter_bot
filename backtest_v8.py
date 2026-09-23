from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd
import numpy as np


# ============================================================
# 1. RISK MANAGER & CONFIG (مدیریت ریسک ضد باگ)
# ============================================================

@dataclass
class RiskConfig:
    risk_reward: float = 2.0
    max_consecutive_losses: int = 3
    min_trades_for_stats: int = 30
    trade_margin: float = 100.0
    leverage: float = 50.0
    fee_rate: float = 0.0007  # کارمزد رفت و برگشت فیوچرز


class RiskManager:
    def __init__(self, initial_equity: float, cfg: RiskConfig = None):
        self.cfg = cfg or RiskConfig()
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.open_trade = None
        self.consecutive_losses = 0
        self.trading_paused = False
        self.pause_timer = 0
        self.trade_history = []

    def can_open_new_trade(self) -> bool:
        if self.trading_paused or self.open_trade is not None:
            return False
        return True

    def position_size(self, entry_price: float, stop_price: float) -> float:
        # محاسبه دقیق تعداد واحد دارایی بر اساس مارجین و لوریج
        position_notional = self.cfg.trade_margin * self.cfg.leverage
        quantity = position_notional / entry_price
        return quantity

    def register_result(self, pnl_r: float, pnl_amount: float):
        # کسر هزینه کارمزد فیوچرز از سود/زیان خالص
        notional = self.cfg.trade_margin * self.cfg.leverage
        fee_cost = notional * self.cfg.fee_rate * 2.0
        net_pnl = pnl_amount - fee_cost
        
        self.equity += net_pnl
        self.trade_history.append({"pnl_r": pnl_r, "pnl_amount": net_pnl})

        if pnl_r < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.trading_paused = True
                self.pause_timer = 24  # استراحت اجباری جهت جلوگیری از دراوداون‌های متوالی
        else:
            self.consecutive_losses = 0

    def tick_pause_timer(self):
        if self.trading_paused:
            self.pause_timer -= 1
            if self.pause_timer <= 0:
                self.trading_paused = False
                self.consecutive_losses = 0

    def stats(self):
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "valid": False}
        
        wins = [t for t in self.trade_history if t["pnl_r"] > 0]
        win_rate = (len(wins) / total_trades) * 100.0
        
        total_r = sum([t["pnl_r"] for t in self.trade_history])
        expectancy_r = total_r / total_trades
        
        valid = total_trades >= self.cfg.min_trades_for_stats
        return {
            "trades": total_trades,
            "win_rate": round(win_rate, 2),
            "expectancy_r": round(expectancy_r, 4),
            "valid": valid
        }


# ============================================================
# 2. STRATEGY CORE (هسته اجرایی ضد نگاه به آینده و قفل همپوشانی)
# ============================================================

@dataclass
class Trade:
    direction: str
    entry_time: object
    entry_price: float
    stop_price: float
    target_price: float
    size: float


class StrategyCore:
    def __init__(self, risk_manager: RiskManager, signal_fn: Callable):
        self.rm = risk_manager
        self.signal_fn = signal_fn
        self._pending_entry = None

    def on_new_closed_candle(self, index: int, candle: dict, history_up_to_here) -> Optional[dict]:
        event = None

        if self.rm.open_trade is not None:
            event = self._check_exit(candle)
        elif self._pending_entry is not None:
            event = self._execute_entry(candle)

        if self.rm.can_open_new_trade() and self._pending_entry is None and self.rm.open_trade is None:
            sig = self.signal_fn(history_up_to_here)
            if sig is not None:
                self._pending_entry = sig

        self.rm.tick_pause_timer()
        return event

    def _execute_entry(self, candle: dict) -> dict:
        sig = self._pending_entry
        self._pending_entry = None
        entry_price = candle["open"]
        stop_price = sig["stop_price"]
        risk_distance = abs(entry_price - stop_price)
        
        if sig["direction"] == "long":
            target_price = entry_price + self.rm.cfg.risk_reward * risk_distance
        else:
            target_price = entry_price - self.rm.cfg.risk_reward * risk_distance
            
        size = self.rm.position_size(entry_price, stop_price)
        self.rm.open_trade = Trade(
            sig["direction"], candle.get("time"), entry_price,
            stop_price, target_price, size
        )
        return {"type": "entry", "trade": self.rm.open_trade}

    def _check_exit(self, candle: dict) -> Optional[dict]:
        t = self.rm.open_trade
        if t.direction == "long":
            hit_stop = candle["low"] <= t.stop_price
            hit_target = candle["high"] >= t.target_price
        else:
            hit_stop = candle["high"] >= t.stop_price
            hit_target = candle["low"] <= t.target_price

        if not hit_stop and not hit_target:
            return None

        if hit_stop:
            pnl_r = -1.0
            pnl_amount = -t.size * abs(t.entry_price - t.stop_price)
            result = "loss"
        else:
            pnl_r = self.rm.cfg.risk_reward
            pnl_amount = t.size * abs(t.target_price - t.entry_price)
            result = "win"

        self.rm.register_result(pnl_r, pnl_amount)
        self.rm.open_trade = None
        return {"type": "exit", "result": result, "pnl_r": pnl_r}


# ============================================================
# 3. HIGH-FREQUENCY PULLBACK SIGNAL FUNCTION (سیگنال پربازدد و دقیق)
# ============================================================

def high_frequency_pullback_signal(history_list):
    if len(history_list) < 30:
        return None

    df = pd.DataFrame(history_list)
    
    # میانگین‌های متحرک سریع برای تشخیص جهت روند کوتاه‌مدت
    df["EMA_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
    
    # محاسبه ATR جهت تعیین حد ضرر پویا
    high_low = df["High"] - df["Low"] if "High" in df else df["high"] - df["low"]
    high_close = np.abs(df["High"] - df["close"].shift()) if "High" in df else np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["Low"] - df["close"].shift()) if "High" in df else np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df["ATR"] = ranges.max(axis=1).rolling(14).mean()

    last_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    ema_9 = df["EMA_9"].iloc[-1]
    ema_21 = df["EMA_21"].iloc[-1]
    current_atr = df["ATR"].iloc[-1]

    if not np.isfinite(current_atr) or current_atr <= 0:
        return None

    is_uptrend = ema_9 > ema_21
    is_downtrend = ema_9 < ema_21

    # پولبک لانگ: روند صعودی، اصلاح به زیر EMA 9 و بازگشت با قدرت در کندل بسته شده
    if is_uptrend and prev_close <= ema_9 and last_close > ema_9 and last_close > prev_close:
        stop_price = last_close - (current_atr * 1.2)
        return {"direction": "long", "stop_price": stop_price}
    
    # پولبک شورت: روند نزولی، اصلاح به بالای EMA 9 و ریزش مجدد در کندل بسته شده
    elif is_downtrend and prev_close >= ema_9 and last_close < ema_9 and last_close < prev_close:
        stop_price = last_close + (current_atr * 1.2)
        return {"direction": "short", "stop_price": stop_price}

    return None


# ============================================================
# 4. BACKTEST RUNNER
# ============================================================

def run_backtest(df, signal_fn, initial_equity: float = 1000.0, risk_cfg: RiskConfig = None):
    risk_cfg = risk_cfg or RiskConfig()
    rm = RiskManager(initial_equity, risk_cfg)
    engine = StrategyCore(rm, signal_fn)

    records = df.to_dict("records")
    log = []

    for i, candle in enumerate(records):
        history = records[: i + 1]
        event = engine.on_new_closed_candle(i, candle, history)
        if event:
            log.append({"index": i, "time": candle.get("time"), **event})

    stats = rm.stats()
    print("=" * 60)
    print("== HIGH-FREQUENCY PULLBACK BACKTEST RESULTS ==")
    print("=" * 60)
    print(f"تعداد معاملات کل:           {stats['trades']}")
    print(f"وین‌ریت (Win Rate):          {stats['win_rate']}%")
    print(f"اکسپکتانسی (واحد R):         {stats['expectancy_r']}")
    print(f"سرمایه‌ی نهایی (USD):        ${round(rm.equity, 2)}")
    print(f"اعتبار آماری:               {'تایید شد' if stats['valid'] else 'نیاز به داده بیشتر'}")
    if rm.trading_paused:
        print(f"⚠️ محافظ سرمایه فعال شد (توقف اضطراری)")
    print("=" * 60)

    return {"log": log, "stats": stats, "final_equity": rm.equity, "risk_manager": rm}


if __name__ == "__main__":
    # تست اولیه با داده‌های شبیه‌سازی‌شده
    np.random.seed(42)
    n = 2000
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    test_df = pd.DataFrame({
        "time": range(n),
        "open": price,
        "high": price + np.random.rand(n) * 1.2,
        "low": price - np.random.rand(n) * 1.2,
        "close": price + np.random.randn(n) * 0.2,
    })

    run_backtest(test_df, high_frequency_pullback_signal)
