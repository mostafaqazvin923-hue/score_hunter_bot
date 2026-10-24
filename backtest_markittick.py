// © MarkitTick - Backtest Strategy Version
//@version=6
strategy("Dynamic Market Metrics Strategy [MarkitTick]", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, precision=2)

// ── INPUTS (مقادیر تنظیم‌شده بر اساس تصویر) ──────────────────────────────
var string GRP_CORE = "⚙️ Core"
var string _mt741a = "🛡️ Breakout"
var string GRP_HTF = "🕐 HTF"
var string GRP_CF = "☁️ Cloud"
var string GRP_DATE = "📅 Backtest Date Range (1 Year)"

float _mt741b = input.float(1.5, "ATR Mult", step=0.1, group=GRP_CORE)
int AP = input.int (14, "ATR/MFI Len", minval=1, group=GRP_CORE)

bool overrideEnabled = input.bool (true, "BO Override", group=_mt741a)
float overrideSensitivity = input.float(2.5, "BO Sens", step=0.1, group=_mt741a)

bool i_htf_show = input.bool (false, "Show HTF", group=GRP_HTF)
string i_htf_tf = input.timeframe("D", "HTF TF", group=GRP_HTF)

int i_cf_len = input.int (10, "Cloud Len", minval=1, maxval=50, group=GRP_CF)

// فیلتر بازه زمانی یک‌ساله (قابل تنظیم برای یک سال گذشته منتهی به امروز)
// با توجه به تاریخ فعلی (اکتبر 2026)، بازه یک‌ساله از اکتبر 2025 تا اکتبر 2026 تنظیم می‌شود:
int startYear  = input.int(2025, "Start Year", group=GRP_DATE)
int startMonth = input.int(10, "Start Month", group=GRP_DATE)
int startDay   = input.int(24, "Start Day", group=GRP_DATE)

int endYear  = input.int(2026, "End Year", group=GRP_DATE)
int endMonth = input.int(10, "End Month", group=GRP_DATE)
int endDay   = input.int(24, "End Day", group=GRP_DATE)

bool inDateRange = (time >= timestamp(startYear, startMonth, startDay, 00, 00)) and (time <= timestamp(endYear, endMonth, endDay, 23, 59))

// ── CALCULATIONS (منطق اصلی اندیکاتور بدون تقلب) ──────────────────────────
float ATR = ta.sma(ta.tr[1], AP)
float upT = low[1] - ATR * _mt741b
float downT = high[1] + ATR * _mt741b
float mfiVal = ta.mfi(hlc3, AP)[1]

float AlphaTrend = 0.0
AlphaTrend := mfiVal >= 50 ? (upT < nz(AlphaTrend[1]) ? nz(AlphaTrend[1]) : upT) : (downT > nz(AlphaTrend[1]) ? nz(AlphaTrend[1]) : downT)

var int direction = 1
if AlphaTrend[1] > AlphaTrend[3]
    direction := 1
else if AlphaTrend[1] < AlphaTrend[3]
    direction := -1

bool isCrossUp = close[1] > AlphaTrend[1] and close[2] <= nz(AlphaTrend[2])
bool isCrossDn = close[1] < AlphaTrend[1] and close[2] >= nz(AlphaTrend[2])
bool _boMomentum = overrideEnabled and (math.abs(close[1] - close[2]) > ATR[1] * overrideSensitivity)

if _boMomentum and isCrossUp
    direction := 1
else if _boMomentum and isCrossDn
    direction := -1

bool isTrendCrossUp = AlphaTrend[1] > AlphaTrend[3] and nz(AlphaTrend[2]) <= nz(AlphaTrend[4])
bool isTrendCrossDn = AlphaTrend[1] < AlphaTrend[3] and nz(AlphaTrend[2]) >= nz(AlphaTrend[4])

bool buySignalk = isTrendCrossUp or (_boMomentum and isCrossUp)
bool sellSignalk = isTrendCrossDn or (_boMomentum and isCrossDn)

var int K1 = 0
var int K2 = 0
var int O1 = 0
var int O2 = 0

O1 := buySignalk[1] ? 0 : O1 + 1
O2 := sellSignalk[1] ? 0 : O2 + 1
K1 := buySignalk[1] ? 0 : K1 + 1
K2 := sellSignalk[1] ? 0 : K2 + 1

bool validLong = buySignalk and O1 > K2
bool validShort = sellSignalk and O2 > K1

// ── مدیریت ریسک و محاسبه سطوح SL و TP ──────────────────────────────────
float entryPrice = 0.0
float slPrice = 0.0
float tpPrice = 0.0

bool _risingLong = bar_index >= 4 and validLong and not validLong[1]
bool _risingShort = bar_index >= 4 and validShort and not validShort[1]

// محاسبه ریسک بر اساس فاصله قیمت ورود تا AlphaTrend (به عنوان حد ضرر پویا)
float _risk = math.max(math.abs(close[1] - AlphaTrend[1]), syminfo.mintick * 10)

if _risingLong and inDateRange
    entryPrice := close[1]
    slPrice := AlphaTrend[1]
    tpPrice := entryPrice + (_risk * 2.0) // استفاده از TP2 برای ریوارد 1 به 2 مطمئن
    strategy.entry("Long", strategy.long)
    strategy.exit("Exit Long", "Long", stop=slPrice, limit=tpPrice)

if _risingShort and inDateRange
    entryPrice := close[1]
    slPrice := AlphaTrend[1]
    tpPrice := entryPrice - (_risk * 2.0) // استفاده از TP2 برای ریوارد 1 به 2 مطمئن
    strategy.entry("Short", strategy.short)
    strategy.exit("Exit Short", "Short", stop=slPrice, limit=tpPrice)

// بستن پوزیشن‌ها در صورت اتمام بازه زمانی بک‌تست
if not inDateRange
    strategy.close_all()

// ── پلات خطوط راهنما روی چارت ───────────────────────────────────────────
plot(direction == 1 ? AlphaTrend : na, "Up Trend", color=#00ACC1, linewidth=2, style=plot.style_linebr)
plot(direction == 2 or direction == -1 ? AlphaTrend : na, "Down Trend", color=#0D47A1, linewidth=2, style=plot.style_linebr)
