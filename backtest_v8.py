import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np

# اتصال امن به صرافی LBank از طریق CCXT
exchange = ccxt.lbank({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# لیست ۳۰ ارز نهایی و کاملاً پاکسازی‌شده (بدون RUNE، AR و UNI، با جایگزینی STX و JUP)
SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT",
    "DOT": "DOT/USDT",
    "SHIB": "SHIB/USDT",
    "PEPE": "PEPE/USDT",
    "ARB": "ARB/USDT",
    "OP": "OP/USDT",
    "POL": "POL/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "INJ": "INJ/USDT",
    "FET": "FET/USDT",
    "APT": "APT/USDT",
    "TIA": "TIA/USDT",
    "ICP": "ICP/USDT",
    "BCH": "BCH/USDT",
    "LTC": "LTC/USDT",
    "CRV": "CRV/USDT",
    "PENDLE": "PENDLE/USDT",
    "AAVE": "AAVE/USDT",
    "STX": "STX/USDT",  # جایگزین مطمئن برای RUNE
    "JUP": "JUP/USDT",  # جایگزین ارزهای ضعیف‌شده
    "NEAR": "NEAR/USDT" # (اگر خواستید تعداد ۳۰ تا کامل بماند یا یکی دیگر مثل GRT)
}
# نکته: برای اینکه دقیقاً ۳۰ کلید یکتا داشته باشیم (چون NEAR تکرار نشود)، یک ارز دیگر مثل GRT جایگزین می‌کنیم:
