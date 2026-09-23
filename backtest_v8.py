from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd
import numpy as np


# ============================================================
# 1. RISK MANAGER & CONFIG
# ============================================================

@dataclass
class RiskConfig:
    max_consecutive_losses: int = 4
    min_trades_for_stats: int = 20
    trade_margin: float = 100.0
    leverage: float = 50.0
    fee_rate: float = 0.0007


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
        position_notional = self.cfg.trade_margin * self.cfg.leverage
        return position_notional / entry_price

    def register_result(self, pnl_amount: float, is_win: bool):
        notional = self.cfg.trade_margin * self.cfg.leverage
        fee_cost = notional * self.cfg.fee_rate * 2.0
        net_pnl = pnl_amount - fee_cost
        
        self.equity += net_pnl
        self.trade_history.append({"win": is_win, "pnl_amount": net_pnl})

        if not is_win:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.trading_paused = True
                self.pause_timer = 15
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
            return {"trades": 0, "win_rate": 0.0, "valid": False}
        
        wins = [t for t in self.trade_history if t["win"]]
        win_rate = (len(wins) / total_trades) * 100.0
        
        valid = total_trades >= self.cfg.min_trades_for_stats
        return {
            "trades": total_trades,
            "win_rate": round(win_rate, 2),
            "valid": valid
        }


# ============================================================
# 2. STRATEGY CORE
# ============================================================

@dataclass
class Trade:
    direction: str
    entry_time: object
    entry_price: float
    stop_price: float
    target_ema_span: int
    size: float


class StrategyCore:
    def __init__(self, risk_manager: RiskManager, signal_fn: Callable):
        self.rm = risk_manager
        self.signal_fn = signal_fn
        self._pending_entry = None

    def on_new_closed_candle(self, index: int, candle: dict, history_up_to_here) -> Optional[dict]:
        event = None

        if self.rm.open_trade is not None:
            event = self._check_exit(candle, history_up_to_here)
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
        
        size = self.rm.position_size(entry_price, stop_price)
        self.rm.open_trade = Trade(
            direction=sig["direction"],
            entry_time=candle.get("time"),
            entry_price=entry_price,
            stop_price=stop_price,
            target_ema_span=sig["target_ema_span"],
            size=size
        )
        return {"type": "entry", "trade": self.rm.open_trade}

    def _check_exit(self, candle: dict, history_list) -> Optional[dict]:
        t = self.rm.open_trade
        df = pd.DataFrame(history_list)
        current_ema = df["close"].ewm(span=t.target_ema_span, adjust=False).mean().iloc[-1]

        hit_stop = False
        hit_target_mean = False

        if t.direction == "long":
            hit_stop = candle["low"] <= t.stop_price
            hit_target_mean = candle["high"] >= current_ema
        else:
            hit_stop = candle["high"] >= t.stop_price
            hit_target_mean = candle["low"] <= current_ema

        if not hit_stop and not hit_target_mean:
            return None

        if hit_stop:
            pnl_amount = -t.size * abs(t.entry_price - t.stop_price)
            is_win = False
            result = "loss"
        else:
            pnl_amount = t.size * abs(current_ema - t.entry_price) if t.direction == "long" else t.size * abs(t.entry_price - current_ema)
            is_win = True
            result = "win"

        self.rm.register_result(pnl_amount, is_win)
        self.rm.open_trade = None
        return {"type": "exit", "result": result}


# ============================================================
# 3. OPTIMIZED Z-SCORE SIGNAL (فریاد حجم و فرکانس بالا)
# ============================================================

def zscore_mean_reversion_signal(history_list):
    if len(history_list) < 30:
        return None

    df = pd.DataFrame(history_list)
    span = 14  # کاهش دوره برای حساسیت و تعداد سیگنال بیشتر
    df["EMA"] = df["close"].ewm(span=span, adjust=False).mean()
    rolling_std = df["close"].rolling(span).std()
    
    df["Z_Score"] = (df["close"] - df["EMA"]) / rolling_std

    high_low = df["High"] - df["Low"] if "High" in df else df["high"] - df["low"]
    df["ATR"] = high_low.rolling(14).mean()

    last_z = df["Z_Score"].iloc[-1]
    last_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2] if len(df) > 1 else last_close
    current_atr = df["ATR"].iloc[-1]

    if not np.isfinite(last_z) or not np.isfinite(current_atr) or current_atr <= 0:
        return None

    # کاهش آستانه به 1.5 برای افزایش تعداد معاملات + تاییدیه بازگشت کندل
    if last_z < -1.5 and last_close > prev_close:
        stop_price = last_close - (current_atr * 1.5)
        return {"direction": "long", "stop_price": stop_price, "target_ema_span": span}
    
    elif last_z > 1.5 and last_close < prev_close:
        stop_price = last_close + (current_atr * 1.5)
        return {"direction": "short", "stop_price": stop_price, "target_ema_span": span}

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
    print("== HIGH-FREQUENCY Z-SCORE BACKTEST RESULTS ==")
    print("=" * 60)
    print(f"تعداد معاملات کل:           {stats['trades']}")
    print(f"وین‌ریت (Win Rate):          {stats['win_rate']}%")
    print(f"سرمایه‌ی نهایی (USD):        ${round(rm.equity, 2)}")
    print(f"اعتبار آماری:               {'تایید شد' if stats['valid'] else 'نیاز به داده بیشتر'}")
    if rm.trading_paused:
        print(f"⚠️ محافظ سرمایه فعال شد (توقف اضطراری)")
    print("=" * 60)

    return {"log": log, "stats": stats, "final_equity": rm.equity, "risk_manager": rm}


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    price = 100 + np.cumsum(np.random.randn(n) * 0.4)
    test_df = pd.DataFrame({
        "time": range(n),
        "open": price,
        "high": price + np.random.rand(n) * 1.0,
        "low": price - np.random.rand(n) * 1.0,
        "close": price + np.random.randn(n) * 0.2,
    })

    run_backtest(test_df, zscore_mean_reversion_signal)
