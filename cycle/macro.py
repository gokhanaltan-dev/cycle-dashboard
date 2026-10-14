"""Macro lens: growth composite, cycle-clock phase, overlays and a backtest vs NBER.

Method (documented on the page too):
  1. Each growth series is converted to monthly, transformed (level / YoY / 3m avg),
     z-scored against its own rolling 10-year history and sign-adjusted so that
     "higher = stronger growth".
  2. Composite = equal-weight average of the available z-scores (>= 4 needed).
     Series released with a lag are carried forward up to 2 months.
  3. Level = 3-month average of the composite (above/below trend).
     Momentum = 3-month change in Level, z-scored against its own 10-year history.
  4. Phase from the quadrant: Early = below trend & improving, Mid = above & improving,
     Late = above & deteriorating, Recession = below & deteriorating.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C

ORDER = C.PHASES  # Early -> Mid -> Late -> Recession -> Early


def next_phase(p: str) -> str:
    return ORDER[(ORDER.index(p) + 1) % 4]


def prev_phase(p: str) -> str:
    return ORDER[(ORDER.index(p) - 1) % 4]


def quadrant(x: float, y: float) -> str:
    if y >= 0:
        return "Mid" if x >= 0 else "Late"
    return "Early" if x >= 0 else "Recession"


def to_monthly(s: pd.Series, freq: str) -> pd.Series:
    s = s.sort_index()
    if freq == "M":
        return s.resample("ME").last()
    return s.resample("ME").mean()


def transform(m: pd.Series, how: str) -> pd.Series:
    if how == "yoy":
        return m.pct_change(12, fill_method=None) * 100
    if how == "level3m":
        return m.rolling(3, min_periods=2).mean()
    return m


def rolling_z(x: pd.Series, window=C.Z_WINDOW_M, min_periods=C.Z_MIN_M) -> pd.Series:
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std()
    return ((x - mu) / sd).clip(-4, 4)


def _fmt(sid: str, v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    spec = C.GROWTH_SERIES.get(sid, {})
    if spec.get("transform") == "yoy":
        return f"{v:+.1f}% YoY"
    if sid == "SAHMREALTIME":
        return f"{v:.2f}"
    if sid in ("UMCSENT",):
        return f"{v:.1f}"
    return f"{v:+.2f}" if abs(v) < 10 else f"{v:+.1f}"


def growth_composite(fred: dict[str, pd.Series], today: pd.Timestamp):
    """Return (frame of monthly z-scores incl. composite/level/momentum, per-series info)."""
    end = (today + pd.offsets.MonthEnd(0)).normalize()
    idx = pd.date_range("1985-01-31", end, freq="ME")
    zs, info = {}, []
    for sid, spec in C.GROWTH_SERIES.items():
        raw = fred.get(sid)
        if raw is None or raw.empty:
            info.append(dict(id=sid, name=spec["name"], status="missing"))
            continue
        m = to_monthly(raw, spec["freq"]).reindex(idx)
        t = transform(m, spec["transform"])
        z = rolling_z(t) * spec["sign"]
        last_obs = raw.index.max()
        zs[sid] = z.ffill(limit=2)
        t_valid = t.dropna()
        z_valid = z.dropna()
        info.append(dict(
            id=sid,
            name=spec["name"],
            status="ok",
            last_obs=last_obs.strftime("%Y-%m-%d"),
            freq=spec["freq"],
            value=_fmt(sid, t_valid.iloc[-1]) if len(t_valid) else "–",
            z=round(float(z_valid.iloc[-1]), 2) if len(z_valid) else None,
            z_3m_change=round(float(z_valid.iloc[-1] - z_valid.iloc[-4]), 2) if len(z_valid) > 4 else None,
            spark=[round(float(v), 2) for v in z_valid.iloc[-36:]],
        ))
    Z = pd.DataFrame(zs, index=idx)
    count = Z.notna().sum(axis=1)
    comp = Z.mean(axis=1).where(count >= C.MIN_SERIES_FOR_COMPOSITE)
    level = comp.rolling(3, min_periods=2).mean()
    mom = level - level.shift(C.MOMENTUM_LAG_M)
    mom_sd = mom.rolling(C.Z_WINDOW_M, min_periods=C.Z_MIN_M).std()
    mom_z = (mom / mom_sd).clip(-4, 4)
    Z["_count"] = count
    Z["_composite"] = comp
    Z["_level"] = level
    Z["_momentum"] = mom_z
    return Z, info


def classify(Z: pd.DataFrame, deadband: float = 0.1) -> pd.Series:
    """Phase per month with a small dead-band to avoid flip-flopping near the axes."""
    phases, prev = [], None
    for x, y in zip(Z["_momentum"], Z["_level"]):
        if pd.isna(x) or pd.isna(y):
            phases.append(None)
            continue
        p = quadrant(x, y)
        if prev and p != prev and (abs(x) < deadband or abs(y) < deadband):
            p = prev
        phases.append(p)
        prev = p
    return pd.Series(phases, index=Z.index)


def macro_call(Z: pd.DataFrame, phase: pd.Series) -> dict:
    valid = Z.dropna(subset=["_level", "_momentum"])
    last = valid.index[-1]
    x, y = float(valid.loc[last, "_momentum"]), float(valid.loc[last, "_level"])
    p = phase.loc[last]

    # months in current phase
    run = 0
    for v in reversed(phase.loc[:last].dropna().tolist()):
        if v != p:
            break
        run += 1

    # confidence: distance from the axes + breadth of agreement across indicators
    cols = [c for c in Z.columns if not c.startswith("_")]
    now = Z.loc[last, cols].dropna()
    before = Z[cols].shift(3).loc[last].reindex(now.index)
    agree_level = float((np.sign(now) == np.sign(y)).mean()) if len(now) else 0.5
    d = (now - before).dropna()
    agree_mom = float((np.sign(d) == np.sign(x)).mean()) if len(d) else 0.5
    dist = math.hypot(x, y)
    conf = 0.5 * (1 - math.exp(-dist / 0.6)) + 0.5 * (agree_level + agree_mom) / 2

    # direction of travel: extrapolate the last 3 months of movement
    prev3 = valid.iloc[-4] if len(valid) > 4 else valid.iloc[0]
    dx, dy = x - float(prev3["_momentum"]), y - float(prev3["_level"])
    proj = quadrant(x + dx, y + dy)
    if proj == p:
        heading = f"Holding in {p}"
    elif proj == next_phase(p):
        heading = f"Heading toward {proj}"
    elif proj == prev_phase(p):
        heading = f"Drifting back toward {proj}"
    else:
        heading = f"Unstable — projection points to {proj}"

    trail = valid.iloc[-13:]
    return dict(
        phase=p,
        confidence=int(round(100 * conf)),
        heading=heading,
        months_in_phase=run,
        as_of=last.strftime("%Y-%m"),
        x=round(x, 2),
        y=round(y, 2),
        level_change_3m=round(dy, 2),
        momentum_change_3m=round(dx, 2),
        agree_level=round(agree_level, 2),
        agree_momentum=round(agree_mom, 2),
        n_series=int(Z.loc[last, "_count"]),
        trail=[dict(d=i.strftime("%Y-%m"), x=round(float(r["_momentum"]), 2), y=round(float(r["_level"]), 2))
               for i, r in trail.iterrows()],
        phase_1m_ago=phase.loc[:last].dropna().iloc[-2] if phase.loc[:last].dropna().size > 1 else None,
    )


# ---------------------------------------------------------------------------
# Backtest vs NBER
# ---------------------------------------------------------------------------
def _episodes(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    eps, start = [], None
    for d, v in mask.items():
        if v and start is None:
            start = d
        if not v and start is not None:
            eps.append((start, prev_d))
            start = None
        prev_d = d
    if start is not None:
        eps.append((start, prev_d))
    return eps


def backtest(phase: pd.Series, usrec: pd.Series | None, start="2000-01-01") -> dict:
    if usrec is None or usrec.empty:
        return {"available": False}
    rec = to_monthly(usrec, "M").reindex(phase.index).ffill().loc[start:]
    ph = phase.loc[start:].reindex(rec.index)
    valid = rec.notna() & ph.notna()
    rec, ph = rec[valid].astype(int), ph[valid]
    is_rec = ph == "Recession"

    episodes = []
    for s, e in _episodes(rec == 1):
        window = ph.loc[s - pd.DateOffset(months=12): e + pd.DateOffset(months=3)]
        flagged = window[window == "Recession"]
        first = flagged.index[0] if len(flagged) else None
        lag = None
        if first is not None:
            lag = (first.year - s.year) * 12 + (first.month - s.month)
        months = rec.loc[s:e]
        episodes.append(dict(
            start=s.strftime("%Y-%m"), end=e.strftime("%Y-%m"),
            first_flag=first.strftime("%Y-%m") if first is not None else None,
            lag_months=lag,
            share_flagged=round(float(is_rec.loc[months.index].mean()), 2),
        ))

    # false alarms: Recession runs (>=2 months) with no NBER month within +/- 6 months
    near = rec.rolling(13, center=True, min_periods=1).max().fillna(0) == 1
    false_alarms = []
    for s, e in _episodes(is_rec):
        if (e.year - s.year) * 12 + e.month - s.month < 1:
            continue
        if not near.loc[s:e].any():
            false_alarms.append(dict(start=s.strftime("%Y-%m"), end=e.strftime("%Y-%m")))

    rec_months = rec == 1
    return dict(
        available=True,
        period=f"{rec.index[0]:%Y-%m} to {rec.index[-1]:%Y-%m}",
        hit_rate=round(float(is_rec[rec_months].mean()), 2) if rec_months.any() else None,
        late_or_rec_rate=round(float(ph[rec_months].isin(["Late", "Recession"]).mean()), 2) if rec_months.any() else None,
        precision=round(float(near[is_rec].mean()), 2) if is_rec.any() else None,
        episodes=episodes,
        false_alarms=false_alarms,
    )


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------
def _light(z: float | None, sign: int) -> str:
    if z is None or sign == 0 or math.isnan(z):
        return "neutral"
    v = z * sign
    return "green" if v > 0.5 else "red" if v < -0.5 else "amber"


def overlays(fred: dict[str, pd.Series], prices: pd.DataFrame) -> list[dict]:
    rows = []
    for sid, spec in C.OVERLAY_SERIES.items():
        s = fred.get(sid)
        if s is None or s.empty:
            rows.append(dict(id=sid, name=spec["name"], group=spec["group"], status="missing"))
            continue
        if spec.get("yoy"):
            s = s.pct_change(12, fill_method=None).mul(100).dropna()
        rows.append(_overlay_row(sid, spec["name"], spec["group"], s, spec["unit"], spec["sign"]))

    # net liquidity = Fed balance sheet - Treasury General Account - reverse repo ($bn)
    if all(k in fred for k in C.LIQUIDITY_SERIES):
        walcl = fred["WALCL"].resample("W-WED").last() / 1000
        tga = fred["WTREGEN"].resample("W-WED").last()
        rrp = fred["RRPONTSYD"].resample("W-WED").last()
        net = (walcl - tga - rrp).dropna() / 1000  # $tn
        rows.append(_overlay_row("NETLIQ", "Net liquidity (Fed BS − TGA − RRP)", "Liquidity",
                                 net, "$tn", 1, change_periods=13))
    else:
        rows.append(dict(id="NETLIQ", name="Net liquidity", group="Liquidity", status="missing"))

    # market stress gauges from prices
    for t, name, sign in (("^VIX", "VIX (equity vol)", -1), ("DX-Y.NYB", "US dollar index", 0)):
        if t in prices and prices[t].notna().sum() > 260:
            rows.append(_overlay_row(t, name, "Market stress", prices[t].dropna(), "", sign, change_periods=63,
                                     z_window=36, z_min=24))
        else:
            rows.append(dict(id=t, name=name, group="Market stress", status="missing"))
    return rows


def _overlay_row(sid, name, group, s: pd.Series, unit, sign, change_periods=None,
                 z_window=C.Z_WINDOW_M, z_min=C.Z_MIN_M) -> dict:
    s = s.dropna().sort_index()
    m = s.resample("ME").mean()
    z = rolling_z(m, z_window, z_min)
    zl = float(z.dropna().iloc[-1]) if z.notna().any() else None
    if change_periods is None:
        base = s.loc[: s.index[-1] - pd.DateOffset(months=3)]
        chg = float(s.iloc[-1] - base.iloc[-1]) if len(base) else None
    else:
        chg = float(s.iloc[-1] - s.iloc[-1 - change_periods]) if len(s) > change_periods else None
    wk = s.resample("W-FRI").last().dropna().iloc[-104:]
    return dict(
        id=sid, name=name, group=group, status="ok", unit=unit,
        value=round(float(s.iloc[-1]), 2), last_obs=s.index[-1].strftime("%Y-%m-%d"),
        change_3m=round(chg, 2) if chg is not None else None,
        z=round(zl, 2) if zl is not None else None,
        light=_light(zl, sign),
        spark=[round(float(v), 3) for v in wk],
    )


def macro_alerts(fred: dict[str, pd.Series], lookback_days: int = 90) -> list[dict]:
    """Threshold events on macro series within the lookback window."""
    out = []
    today = max(s.index.max() for s in fred.values()) if fred else pd.Timestamp.today()
    since = today - pd.Timedelta(days=lookback_days)

    s = fred.get("SAHMREALTIME")
    if s is not None and len(s) and s.iloc[-1] >= 0.5:
        out.append(dict(date=s.index[-1], kind="macro", severity="warning",
                        text=f"Sahm rule triggered: {s.iloc[-1]:.2f} (≥ 0.50 has marked every US recession since 1970)."))

    s = fred.get("T10Y3M")
    if s is not None and len(s) > 300:
        recent = s.loc[since:]
        prior = s.loc[since - pd.Timedelta(days=365): since]
        if len(recent) and recent.iloc[-1] > 0 and (prior < 0).sum() >= 60:
            cross = recent[(recent > 0) & (recent.shift(1) <= 0)]
            if len(cross):
                out.append(dict(date=cross.index[-1], kind="macro", severity="warning",
                                text="10Y-3M curve un-inverted after a long inversion. Past recessions have often started after the un-inversion, not during it."))

    s = fred.get("BAMLH0A0HYM2")
    if s is not None and len(s) > 70:
        chg = s.iloc[-1] - s.iloc[-64]
        if chg >= 1.0:
            out.append(dict(date=s.index[-1], kind="macro", severity="warning",
                            text=f"High-yield spreads widened {chg:+.2f} pts over 3 months — credit stress building."))
        elif chg <= -0.75:
            out.append(dict(date=s.index[-1], kind="macro", severity="info",
                            text=f"High-yield spreads tightened {chg:+.2f} pts over 3 months — credit easing."))

    s = fred.get("NFCI")
    if s is not None and len(s) > 2 and s.iloc[-1] > 0 >= s.iloc[-2]:
        out.append(dict(date=s.index[-1], kind="macro", severity="watch",
                        text="Financial conditions moved to tighter than average (NFCI > 0)."))

    s = fred.get("ICSA")
    if s is not None and len(s) > 60:
        ma = s.rolling(4).mean()
        low = ma.iloc[-52:].min()
        rise = ma.iloc[-1] / low - 1
        if rise >= 0.15:
            out.append(dict(date=s.index[-1], kind="macro", severity="watch",
                            text=f"Jobless claims 4-wk average is {rise:.0%} above its 52-week low — labour market softening."))
    return [a for a in out if a["date"] >= since]
