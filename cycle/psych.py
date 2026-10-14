"""Market psychology: an estimate of where the S&P 500 sits on the classic
"cycle of market emotions" (optimism → thrill → euphoria → anxiety → denial →
fear → panic → capitulation → despondency → hope → relief → optimism).

Method
  Mood level (y) = average of three z-scores, each against its own rolling
    10-year history:
      - S&P 500 distance from its 200-day average
      - S&P 500 position in its 52-week range
      - VIX (inverted: low volatility = complacent mood)
    smoothed with a 30-day EMA.
  Mood momentum (x) = 3-month change in mood level, z-scored the same way.
  Position on the emotion wave = the angle of (x, y): bottom = low mood that has
    stopped falling (despondency), top = high mood that has stopped rising
    (euphoria), rising half = momentum up, falling half = momentum down.
  A separate fear-and-greed gauge (0–100) cross-checks the read: each of six
  market measures is ranked against its own past three years and averaged.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C

# angle on the wave: 0 = trough, 90 = mid-rise, 180 = peak, 270 = mid-fall
STAGES = [
    ("Despondency", 0, "Prices near the low and few want to own stocks. Historically the point of maximum opportunity."),
    ("Depression", 20, "Selling has run out of steam, but the mood is still gloomy."),
    ("Hope", 45, "First real rallies; most investors doubt them."),
    ("Relief", 70, "The recovery is gaining ground and losses are being made back."),
    ("Optimism", 100, "The uptrend is confirmed and buyers are returning."),
    ("Excitement", 130, "Gains are accelerating and confidence is building."),
    ("Thrill", 155, "Strong gains feel easy; risk is being taken freely."),
    ("Euphoria", 180, "Peak confidence. Historically the point of maximum financial risk."),
    ("Anxiety", 205, "The first meaningful pullback after the peak."),
    ("Denial", 230, "Dips are still bought on the view that it is just a correction."),
    ("Fear", 260, "The trend has broken and losses are mounting."),
    ("Desperation", 290, "Selling broadens and rallies fail."),
    ("Panic", 315, "Volatility spikes and selling turns forced."),
    ("Capitulation", 340, "Investors give up and sell regardless of price."),
]
NAMES = [s[0] for s in STAGES]
RISK_ZONE = {"Thrill", "Euphoria", "Anxiety"}
OPPORTUNITY_ZONE = {"Capitulation", "Despondency", "Depression"}
KEY_STAGES = {"Euphoria", "Anxiety", "Panic", "Capitulation", "Despondency", "Hope"}

Z_WIN, Z_MIN = 2520, 504


def stage_of(phi: float) -> str:
    best, bd = None, 999
    for name, a, _ in STAGES:
        d = abs(((phi - a + 180) % 360) - 180)
        if d < bd:
            best, bd = name, d
    return best


def _z(s: pd.Series, win: int = Z_WIN, minp: int = Z_MIN) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return ((s - mu) / sd).clip(-4, 4)


LEVEL_SPAN, MOM_LAG, MOM_SPAN, R_MIN, STAGE_MARGIN = 30, 63, 15, 0.4, 10
CENTRE = {n: a for n, a, _ in STAGES}


def _dist(p: float, a: float) -> float:
    return abs(((p - a + 180) % 360) - 180)


def mood_frame(spy: pd.Series, vix: pd.Series, win: int = Z_WIN, minp: int = Z_MIN) -> pd.DataFrame:
    df = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
    _zz = lambda x: _z(x, win, minp)
    ext = df["spy"] / df["spy"].rolling(200).mean() - 1
    lo, hi = df["spy"].rolling(252).min(), df["spy"].rolling(252).max()
    rng = (df["spy"] - lo) / (hi - lo)
    vz = -_zz(np.log(df["vix"]))
    level = pd.concat([_zz(ext), _zz(rng), vz], axis=1).mean(axis=1, skipna=False)
    level = level.ewm(span=LEVEL_SPAN, adjust=False).mean()
    mom = _zz(level - level.shift(MOM_LAG)).ewm(span=MOM_SPAN, adjust=False).mean()
    out = pd.DataFrame({"spy": df["spy"], "y": level, "x": mom}).dropna()
    # angle on the wave; when the signal is weak (close to the origin) hold the last angle
    phis, prev = [], None
    for x, y in zip(out["x"], out["y"]):
        p = (math.degrees(math.atan2(x, -y)) + 360) % 360
        if prev is not None and math.hypot(x, y) < R_MIN:
            p = prev
        phis.append(p)
        prev = p
    out["phi"] = phis
    # stage with hysteresis: only switch when clearly closer to the new stage's centre
    labels, cur = [], None
    for p in out["phi"]:
        c = stage_of(p)
        if cur is None or (c != cur and _dist(p, CENTRE[c]) + STAGE_MARGIN < _dist(p, CENTRE[cur])):
            cur = c
        labels.append(cur)
    out["stage"] = labels
    return out


def _pct_rank(s: pd.Series, window: int = 756) -> float | None:
    s = s.dropna()
    if len(s) < 120:
        return None
    w = s.iloc[-window:]
    return float((w <= w.iloc[-1]).mean() * 100)


def gauge(prices: pd.DataFrame) -> tuple[int | None, list[dict]]:
    comps = []
    spy = prices.get(C.BENCHMARK)

    def add(name, series, note):
        v = _pct_rank(series)
        comps.append(dict(name=name, score=None if v is None else int(round(v)), note=note))

    if spy is not None:
        add("Momentum", spy / spy.rolling(125).mean() - 1, "S&P 500 vs its 125-day average")
        lo, hi = spy.rolling(252).min(), spy.rolling(252).max()
        add("Price strength", (spy - lo) / (hi - lo), "Position in its 52-week range")
    if "^VIX" in prices:
        v = prices["^VIX"]
        add("Volatility", -(v / v.rolling(50).mean() - 1), "VIX vs its 50-day average (low = greed)")
    if {"HYG", "IEF"} <= set(prices.columns):
        add("Junk bond demand", prices["HYG"].pct_change(20, fill_method=None) - prices["IEF"].pct_change(20, fill_method=None),
            "High-yield vs Treasuries, 20-day return gap")
    if spy is not None and "TLT" in prices:
        add("Safe-haven demand", spy.pct_change(20, fill_method=None) - prices["TLT"].pct_change(20, fill_method=None),
            "Stocks vs long bonds, 20-day return gap")
    etfs = [t for t in list(C.SECTORS) + list(C.INDUSTRIES) if t in prices]
    if etfs:
        px = prices[etfs]
        add("Breadth", (px > px.rolling(200).mean()).mean(axis=1).where(px.rolling(200).mean().notna().any(axis=1)),
            "Share of sector & industry ETFs above their 200-day")
    scores = [c["score"] for c in comps if c["score"] is not None]
    return (int(round(np.mean(scores))) if scores else None), comps


def gauge_label(g: int | None) -> str:
    if g is None:
        return "unavailable"
    return ("Extreme fear" if g < 25 else "Fear" if g < 45 else "Neutral" if g <= 55
            else "Greed" if g <= 75 else "Extreme greed")


def turning_points(df: pd.DataFrame, threshold: float = 0.15, since: str = "1998-01-01") -> list[dict]:
    """Peaks before ≥15% S&P 500 drawdowns and the troughs that followed, with the model's stage at each."""
    s = df["spy"].loc[since:]
    events, peak_d, peak_v, in_dd, trough_d, trough_v = [], s.index[0], s.iloc[0], False, None, None
    for d, v in s.items():
        if not in_dd:
            if v > peak_v:
                peak_d, peak_v = d, v
            elif v / peak_v - 1 <= -threshold:
                in_dd, trough_d, trough_v = True, d, v
        else:
            if v < trough_v:
                trough_d, trough_v = d, v
            if v >= peak_v:
                events.append((peak_d, peak_v, trough_d, trough_v))
                in_dd, peak_d, peak_v = False, d, v
    if in_dd:
        events.append((peak_d, peak_v, trough_d, trough_v))
    out = []
    for pd_, pv, td, tv in events:
        out.append(dict(peak=pd_.strftime("%Y-%m-%d"), peak_stage=df.loc[pd_, "stage"],
                        trough=td.strftime("%Y-%m-%d"), trough_stage=df.loc[td, "stage"],
                        drawdown=round((tv / pv - 1) * 100, 1)))
    return out


def psych(prices: pd.DataFrame, long: pd.DataFrame | None) -> tuple[dict | None, pd.Series]:
    """Returns (payload, stage series). Falls back to the shorter price history with
    3-year z-scores if the long download failed; returns (None, empty) if even that is too short."""
    if long is not None and {"SPY", "^VIX"} <= set(long.columns) and len(long) > 1500:
        df = mood_frame(long["SPY"], long["^VIX"])
    elif {"SPY", "^VIX"} <= set(prices.columns):
        df = mood_frame(prices["SPY"], prices["^VIX"], win=756, minp=252)
    else:
        df = pd.DataFrame()
    if df.empty:
        return None, pd.Series(dtype=object)
    last = df.iloc[-1]
    phi, stage = float(last["phi"]), last["stage"]
    i = NAMES.index(stage)
    g, comps = gauge(prices)

    upper = 90 < phi < 270
    r = math.hypot(last["x"], last["y"])
    conf_r = 1 - math.exp(-r / 0.8)
    if g is None:
        agree, agree_txt = 0.5, "The fear-and-greed gauge is unavailable today."
    elif (g > 50) == upper:
        agree = min(1.0, abs(g - 50) / 25)
        agree_txt = f"The fear-and-greed gauge ({g}, {gauge_label(g).lower()}) agrees."
    else:
        agree = 0.0
        agree_txt = (f"The fear-and-greed gauge ({g}, {gauge_label(g).lower()}) disagrees — "
                     "treat this stage as low conviction.")
    conf = int(round(100 * (0.6 * conf_r + 0.4 * agree)))

    trail = df.iloc[-126::10]
    weekly = df.resample("W-FRI").last().loc["2015-01-01":]
    hist = dict(dates=[d.strftime("%Y-%m-%d") for d in weekly.index],
                spy=[round(float(v), 2) for v in weekly["spy"]],
                stage=list(weekly["stage"]))
    payload = dict(
        stage=stage, phi=round(phi, 1), confidence=conf,
        description=STAGES[i][2],
        prev_stage=NAMES[i - 1], next_stage=NAMES[(i + 1) % len(NAMES)],
        rising=bool(0 < phi < 180),
        x=round(float(last["x"]), 2), y=round(float(last["y"]), 2),
        gauge=g, gauge_label=gauge_label(g), components=comps, agreement=agree_txt,
        stages=[dict(name=n, phi=a, text=t) for n, a, t in STAGES],
        trail=[dict(d=d.strftime("%Y-%m-%d"), phi=round(float(p), 1), stage=s)
               for d, p, s in zip(trail.index, trail["phi"], trail["stage"])],
        history=hist,
        turning_points=turning_points(df),
        risk_zone=sorted(RISK_ZONE), opportunity_zone=sorted(OPPORTUNITY_ZONE),
        history_start=df.index[0].strftime("%Y-%m-%d"),
        switches_per_year=round(float(df["stage"].iloc[-1260:].ne(df["stage"].iloc[-1260:].shift(1)).sum()) / 5, 1),
    )
    return payload, df["stage"]


def psych_alerts(stages: pd.Series, lookback: int) -> list[dict]:
    s = stages.iloc[-(lookback + 1):]
    out = []
    for d, new, old in zip(s.index[1:], s.iloc[1:], s.shift(1).iloc[1:]):
        if new != old and new in KEY_STAGES:
            sev = "warning" if new in ("Euphoria", "Panic", "Capitulation") else "watch"
            out.append(dict(date=d, kind="psych", severity=sev,
                            text=f"Market mood moved {old} → {new} on the emotion cycle."))
    return out
