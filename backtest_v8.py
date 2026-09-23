from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd
import numpy as np


# ============================================================
# 1. RISK MANAGER & CONFIG (مارجین ۱۰۰، اهرم ۵۰، ریوارد ۱ به ۲)
# ============================================================

@dataclass
class RiskConfig:
    risk_reward: float = 2.0  # اصلاح دقیق به ریوارد ۱ به ۲
    max_consecutive_losses: int = 6
    min_trades_for_stats: int = 50
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
        if self.trading_paused or self.open_trade is not None or self.equity < self.cfg.trade_margin:
            return False
        return True

    def position_size(self, entry_price: float) -> float:
        position_notional = self.cfg.trade_margin * self.cfg.leverage
        return position_notional / entry_price

    def register_result(self, pnl_amount: float, is_win: bool):
        notional = self.cfg.trade_margin * self.cfg.leverage
        fee_cost = notional * self.cfg.fee_rate * 2.0  # کارمزد رفت و برگشت
        net_pnl = pnl_amount - fee_cost
        
        self.equity += net_pnl
        if self.equity < 0:
            self.equity = 0.0

        self.trade_history.append({"win": is_win, "net_pnl": net_pnl})

        if not is_win:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.trading_paused = True
                self.pause_timer = 5
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
        
        # اعمال دقیق ضریب ۲.۰ برای حد سود بر اساس فاصله تا استاپ‌لاوس
        if sig["direction"] == "long":
            target_price = entry_price + (self.rm.cfg.risk_reward * abs(entry_price - stop_price))
        else:
            target_price = entry_price - (self.rm.cfg.risk_reward * abs(entry_price - stop_price))

        size = self.rm.position_size(entry_price)
        self.rm.open_trade = Trade(
            direction=sig["direction"],
            entry_time=candle.get("time"),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            size=size
        )
        return {"type": "entry", "trade": self.rm.open_trade}

    def _check_exit(self, candle: dict) -> Optional[dict]:
        t = self.rm.open_trade
        hit_stop = False
        hit_target = False

        if t.direction == "long":
            hit_stop = candle["low"] <= t.stop_price
            hit_target = candle["high"] >= t.target_price
        else:
            hit_stop = candle["high"] >= t.stop_price
            hit_target = candle["low"] <= t.target_price

        if not hit_stop and not hit_target:
            return None

        if hit_stop:
            pnl_amount = -t.size * abs(t.entry_price - t.stop_price)
            is_win = False
        else:
            pnl_amount = t.size * abs(t.target_price - t.entry_price)
            is_win = True

        self.rm.register_result(pnl_amount, is_win)
        self.rm.open_trade = None
        return {"type": "exit", "result": "win" if is_win else "loss"}


# ============================================================
# 3. HIGH-FREQUENCY ULTRA-FAST RSI MEAN REVERSION SIGNAL
# ============================================================

def ultra_fast_rsi_signal(history_list):
    if len(history_list) < 15:
        return None

    df = pd.DataFrame(history_list)
    
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(2).mean()
    avg_loss = loss.rolling(2).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI2"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"] if "High" in df else df["high"] - df["low"]
    df["ATR"] = high_low.rolling(10).mean()

    last_rsi = df["RSI2"].iloc[-1]
    last_close = df["close"].iloc[-1]
    current_atr = df["ATR"].iloc[-1]

    if not np.isfinite(last_rsi) or not np.isfinite(current_atr) or current_atr <= 0:
        return None

    if last_rsi < 10:
        stop_price = last_close - (current_atr * 0.8)
        return {"direction": "long", "stop_price": stop_price}
    
    elif last_rsi > 90:
        stop_price = last_close + (current_atr * 0.8)
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
        if rm.equity <= 0:
            break
        if event:
            log.append({"index": i, "time": candle.get("time"), **event})

    stats = rm.stats()
    print("=" * 60)
    print("== ULTRA HIGH-FREQUENCY RSI(2) BACKTEST RESULTS (RR 1:2) ==")
    print("=" * 60)
    print(f"تعداد معاملات کل:           {stats['trades']}")
    print(f"وین‌ریت (Win Rate):          {stats['win_rate']}%")
    print(f"سرمایه‌ی نهایی (USD):        ${round(rm.equity, 2)}")
    print(f"اعتبار آماری:               {'تایید شد' if stats['valid'] else 'نیاز به داده بیشتر'}")
    if rm.equity <= 0:
        print(f"⚠️ حساب لیکویید / صفر شد")
    print("=" * 60)

    return {"log": log, "stats": stats, "final_equity": rm.equity, "risk_manager": rm}


if __name__ == "__main__":
    np.random.seed(42)
    n = 2500
    price = 100 + np.cumsum(np.random.randn(n) * 0.3)
    test_df = pd.DataFrame({
        "time": range(n),
        "open": price,
        "high": price + np.random.rand(n) * 0.8,
        "low": price - np.random.rand(n) * 0.8,
        "close": price + np.random.randn(n) * 0.15,
    })

    run_backtest(test_df, ultra_fast_rsi_signal)
