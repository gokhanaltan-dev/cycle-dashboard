"""Universe, data series and model settings.

Everything tunable lives here so the engine code stays generic.
"""

BENCHMARK = "SPY"

# ticker: (display name, keyword expected in the instrument's Yahoo name, used for identity checks)
SECTORS = {
    "XLK": ("Technology", "Technology"),
    "XLC": ("Communication Services", "Communication"),
    "XLY": ("Consumer Discretionary", "Discretionary"),
    "XLP": ("Consumer Staples", "Staples"),
    "XLE": ("Energy", "Energy"),
    "XLF": ("Financials", "Financial"),
    "XLV": ("Health Care", "Health"),
    "XLI": ("Industrials", "Industrial"),
    "XLB": ("Materials", "Materials"),
    "XLRE": ("Real Estate", "Real Estate"),
    "XLU": ("Utilities", "Utilities"),
}

INDUSTRIES = {
    "SMH": ("Semiconductors", "Semiconductor"),
    "IGV": ("Software", "Software"),
    "XBI": ("Biotech", "Biotech"),
    "IHI": ("Medical Devices", "Medical Devices"),
    "KRE": ("Regional Banks", "Regional"),
    "IAI": ("Brokers & Exchanges", "Broker"),
    "KIE": ("Insurance", "Insurance"),
    "ITB": ("Homebuilders", "Home Construction"),
    "XRT": ("Retail", "Retail"),
    "XME": ("Metals & Mining", "Metals"),
    "XOP": ("Oil & Gas E&P", "Oil"),
    "OIH": ("Oil Services", "Oil Services"),
    "ITA": ("Aerospace & Defense", "Aerospace"),
    "IYT": ("Transports", "Transportation"),
    "JETS": ("Airlines", "Jets"),
    "TAN": ("Solar", "Solar"),
    "GDX": ("Gold Miners", "Gold Miners"),
    "COPX": ("Copper Miners", "Copper"),
    "PAVE": ("Infrastructure", "Infrastructure"),
    "URA": ("Uranium", "Uranium"),
}

FACTORS = {
    "IWM": ("Small Caps", "Russell 2000"),
    "RSP": ("Equal-Weight S&P", "Equal Weight"),
    "MTUM": ("Momentum", "Momentum"),
    "QUAL": ("Quality", "Quality"),
    "USMV": ("Min Volatility", "Min Vol"),
    "VLUE": ("Value", "Value"),
    "IWF": ("Large Growth", "Growth"),
    "IWD": ("Large Value", "Value"),
    "SPHB": ("High Beta", "High Beta"),
    "SPLV": ("Low Volatility", "Low Vol"),
}

CROSS_ASSET = {
    "TLT": ("20+Y Treasuries", "20+"),
    "IEF": ("7-10Y Treasuries", "7-10"),
    "SHY": ("1-3Y Treasuries", "1-3"),
    "HYG": ("High-Yield Credit", "High Yield"),
    "LQD": ("IG Credit", "Investment Grade"),
    "GLD": ("Gold (GLD)", "Gold"),
    "DBC": ("Commodities", "Commodity"),
    "HG=F": ("Copper", "Copper"),
    "GC=F": ("Gold", "Gold"),
    "CL=F": ("WTI Crude", "Crude"),
    "DX-Y.NYB": ("US Dollar Index", "Dollar"),
    "^VIX": ("VIX", "Volatility"),
    "BTC-USD": ("Bitcoin", "Bitcoin"),
}

GROUPS = {"sectors": SECTORS, "industries": INDUSTRIES, "factors": FACTORS}

ALL_TICKERS = (
    [BENCHMARK] + list(SECTORS) + list(INDUSTRIES) + list(FACTORS) + list(CROSS_ASSET)
)

# Tickers whose prices are not USD-denominated equities/ETFs: skip currency check
# Note: ^MOVE is not used — on Yahoo it resolves to an unrelated bond ETF, not the ICE MOVE index.
NON_EQUITY = {"HG=F", "GC=F", "CL=F", "DX-Y.NYB", "^VIX", "BTC-USD"}

# ---------------------------------------------------------------------------
# Macro (FRED). freq: D/W/M. transform: level | yoy | level3m. sign: +1 good when up.
# ---------------------------------------------------------------------------
GROWTH_SERIES = {
    "CFNAIMA3": dict(name="Chicago Fed activity index (3m)", freq="M", transform="level", sign=1),
    "ICSA": dict(name="Initial jobless claims (YoY)", freq="W", transform="yoy", sign=-1),
    "PERMIT": dict(name="Building permits (YoY)", freq="M", transform="yoy", sign=1),
    "INDPRO": dict(name="Industrial production (YoY)", freq="M", transform="yoy", sign=1),
    "UMCSENT": dict(name="Consumer sentiment (UMich)", freq="M", transform="level", sign=1),
    "SAHMREALTIME": dict(name="Sahm rule indicator", freq="M", transform="level", sign=-1),
    "GACDFSA066MSFRBPHI": dict(name="Philly Fed manufacturing", freq="M", transform="level3m", sign=1),
    "GACDISA066MSFRBNY": dict(name="Empire State manufacturing", freq="M", transform="level3m", sign=1),
}

# Context overlays (not in the growth composite). unit is for display only.
OVERLAY_SERIES = {
    "T10Y2Y": dict(name="10Y-2Y curve", freq="D", unit="%", sign=1, group="Curve & rates"),
    "T10Y3M": dict(name="10Y-3M curve", freq="D", unit="%", sign=1, group="Curve & rates"),
    "DFII10": dict(name="10Y real yield", freq="D", unit="%", sign=-1, group="Curve & rates"),
    "T10YIE": dict(name="10Y breakeven inflation", freq="D", unit="%", sign=0, group="Inflation"),
    "CPIAUCSL": dict(name="CPI inflation (YoY)", freq="M", unit="%", sign=-1, group="Inflation", yoy=True),
    "BAMLH0A0HYM2": dict(name="High-yield spread (OAS)", freq="D", unit="%", sign=-1, group="Credit & conditions"),
    "NFCI": dict(name="Chicago Fed financial conditions", freq="W", unit="", sign=-1, group="Credit & conditions"),
}

LIQUIDITY_SERIES = ["WALCL", "WTREGEN", "RRPONTSYD"]  # WALCL in $m; the other two in $bn
RECESSION_SERIES = "USREC"

# Rolling window for macro z-scores (months)
Z_WINDOW_M = 120
Z_MIN_M = 60
MOMENTUM_LAG_M = 3
MIN_SERIES_FOR_COMPOSITE = 4

# ---------------------------------------------------------------------------
# Market-implied phase: classic rotation-model leadership templates.
# +1 = tends to lead in that phase, -1 = tends to lag, 0 = neutral. A prior, not a law.
# ---------------------------------------------------------------------------
PHASES = ["Early", "Mid", "Late", "Recession"]
PHASE_TEMPLATES = {
    "Early": {"XLF": 1, "XLRE": 1, "XLY": 1, "XLI": 1, "XLK": 1, "XLB": 1,
              "XLU": -1, "XLP": -1, "XLV": -1, "XLC": -0.5, "XLE": -1},
    "Mid": {"XLK": 1, "XLC": 1, "XLI": 0.5, "XLF": 0, "XLY": 0, "XLRE": 0,
            "XLB": -1, "XLU": -1, "XLP": -0.5, "XLV": 0, "XLE": -0.5},
    "Late": {"XLE": 1, "XLB": 1, "XLP": 1, "XLV": 1, "XLU": 1,
             "XLK": -1, "XLY": -1, "XLC": -0.5, "XLRE": -0.5, "XLF": 0, "XLI": 0},
    "Recession": {"XLP": 1, "XLU": 1, "XLV": 1, "XLI": -1, "XLK": -1, "XLB": -1,
                  "XLRE": -1, "XLF": -0.5, "XLY": -0.5, "XLE": -0.5, "XLC": 0},
}
# Clock angle (degrees) for each phase centre. x = momentum, y = level.
PHASE_ANGLE = {"Early": -45, "Mid": 45, "Late": 135, "Recession": 225}

# ---------------------------------------------------------------------------
# Rotation settings
# ---------------------------------------------------------------------------
HORIZONS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "12M": 252}
SCORE_WEIGHTS = {"1M": 0.2, "3M": 0.4, "6M": 0.25, "12M": 0.15}

RRG = {
    "weekly": dict(smooth=5, window=20, roc=4, tail=8),
    "daily": dict(smooth=10, window=63, roc=10, tail=10, step=2),
}

# (numerator, denominator, label, risk sign: +1 rising = risk-on, 0 = informational)
RATIOS = [
    ("XLY", "XLP", "Discretionary / Staples", 1),
    ("XLK", "XLU", "Tech / Utilities", 1),
    ("SPHB", "SPLV", "High beta / Low vol", 1),
    ("IWM", "SPY", "Small caps / S&P 500", 1),
    ("RSP", "SPY", "Equal weight / Cap weight", 1),
    ("XLF", "XLU", "Financials / Utilities", 1),
    ("XLI", "XLU", "Industrials / Utilities", 1),
    ("HYG", "IEF", "Junk bonds / Treasuries", 1),
    ("HG=F", "GC=F", "Copper / Gold", 1),
    ("ITB", "SPY", "Homebuilders / S&P 500", 1),
    ("KRE", "SPY", "Regional banks / S&P 500", 1),
    ("SMH", "SPY", "Semis / S&P 500", 1),
    ("CYC", "DEF", "Cyclicals / Defensives", 1),
    ("XLE", "SPY", "Energy / S&P 500", 0),
]
CYCLICALS = ["XLY", "XLF", "XLI", "XLB", "XLK"]
DEFENSIVES = ["XLP", "XLU", "XLV"]

PRICE_HISTORY_YEARS = 4
ALERT_LOOKBACK_SESSIONS = 60
