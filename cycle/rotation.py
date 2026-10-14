"""Rotation lens: relative performance across horizons, RRG, intermarket ratios,
breadth, price-based alerts and trading implications.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .market import relative_returns

QUADS = ("Leading", "Weakening", "Lagging", "Improving")


def _quad(r: float, m: float) -> str | None:
    if pd.isna(r) or pd.isna(m):
        return None
    if r >= 100:
        return "Leading" if m >= 100 else "Weakening"
    return "Improving" if m >= 100 else "Lagging"


# ---------------------------------------------------------------------------
# Relative Rotation Graph (JdK-style approximation)
# ---------------------------------------------------------------------------
def rrg(prices: pd.DataFrame, tickers: list[str], mode: str = "weekly"):
    """RS-Ratio: where relative strength sits vs its own recent trend (z-score, centred 100).
    RS-Momentum: rate of change of RS-Ratio, scaled by its own volatility (centred 100).
    """
    p = C.RRG[mode]
    px = prices[tickers + [C.BENCHMARK]].dropna(how="all")
    if mode == "weekly":
        last_date = px.index[-1]
        px = px.resample("W-FRI").last()
        px.index = px.index[:-1].append(pd.DatetimeIndex([last_date]))
    rs = 100 * px[tickers].div(px[C.BENCHMARK], axis=0)
    rs_s = rs.ewm(span=p["smooth"], adjust=False).mean()
    mu = rs_s.rolling(p["window"]).mean()
    sd = rs_s.rolling(p["window"]).std()
    ratio = (100 + (rs_s - mu) / sd).clip(96, 104)
    roc = ratio - ratio.shift(p["roc"])
    mom = (100 + roc / roc.rolling(p["window"]).std()).clip(96, 104)
    return ratio, mom


def rrg_payload(ratio: pd.DataFrame, mom: pd.DataFrame, tail: int, step: int = 1) -> dict:
    out = {}
    for t in ratio.columns:
        r, m = ratio[t].dropna(), mom[t].dropna()
        idx = r.index.intersection(m.index)
        if len(idx) < tail * step:
            continue
        pts = idx[-(tail - 1) * step - 1::step]
        out[t] = [dict(d=d.strftime("%Y-%m-%d"), r=round(float(r[d]), 2), m=round(float(m[d]), 2)) for d in pts]
    return out


def quadrant_frame(ratio: pd.DataFrame, mom: pd.DataFrame, band: float = 0.25) -> pd.DataFrame:
    """Quadrant labels with a dead-band: a side of the 100 line only flips once the
    value is `band` beyond it, so ETFs hugging an axis don't flicker between quadrants."""
    q = pd.DataFrame(index=ratio.index, columns=ratio.columns, dtype=object)
    for t in ratio.columns:
        labels, sr, sm = [], None, None
        for a, b in zip(ratio[t], mom[t]):
            if pd.isna(a) or pd.isna(b):
                labels.append(None)
                continue
            if sr is None:
                sr, sm = a >= 100, b >= 100
            if a >= 100 + band:
                sr = True
            elif a <= 100 - band:
                sr = False
            if b >= 100 + band:
                sm = True
            elif b <= 100 - band:
                sm = False
            labels.append(_quad(100.1 if sr else 99.9, 100.1 if sm else 99.9))
        q[t] = labels
    return q


def dedupe_alerts(alerts: list[dict], cooldown_days: int = 21) -> list[dict]:
    """Keep one alert per (ticker, kind, direction) within the cooldown window."""
    kept, last = [], {}
    for a in sorted(alerts, key=lambda a: a["date"]):
        key = (a.get("ticker"), a["kind"], a.get("up"), None if a.get("ticker") else a["text"][:40])
        prev = last.get(key)
        if prev is not None and (a["date"] - prev).days < cooldown_days:
            continue
        last[key] = a["date"]
        kept.append(a)
    return kept


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------
def leaderboard(prices: pd.DataFrame, group: dict, qw: pd.DataFrame, qd: pd.DataFrame):
    tickers = [t for t in group if t in prices]
    sub = prices[tickers + [C.BENCHMARK]]
    rel = {h: relative_returns(sub, n)[tickers] for h, n in C.HORIZONS.items()}
    ranks = {h: r.rank(axis=1, ascending=False) for h, r in rel.items()}
    pct = {h: r.rank(axis=1, pct=True) for h, r in rel.items()}
    score = sum(w * pct[h] for h, w in C.SCORE_WEIGHTS.items()) * 100
    rs = sub[tickers].div(sub[C.BENCHMARK], axis=0)
    rs_hi = rs >= rs.rolling(252, min_periods=200).max()
    rs_lo = rs <= rs.rolling(252, min_periods=200).min()
    ma50 = sub[tickers].rolling(50).mean()
    ma200 = sub[tickers].rolling(200).mean()
    hi52 = sub[tickers].rolling(252, min_periods=200).max()
    ret1d = sub[tickers].pct_change(fill_method=None)
    bench1d = sub[C.BENCHMARK].pct_change(fill_method=None)
    n = len(tickers)

    rows = []
    for t in tickers:
        last = lambda df: df[t].iloc[-1]
        r3 = ranks["3M"][t]
        rows.append(dict(
            ticker=t,
            name=group[t][0],
            ret={h: round(float(rel[h][t].iloc[-1]) * 100, 2) if pd.notna(rel[h][t].iloc[-1]) else None for h in C.HORIZONS},
            rank={h: int(ranks[h][t].iloc[-1]) if pd.notna(ranks[h][t].iloc[-1]) else None for h in C.HORIZONS},
            rank_chg_5d=int(r3.iloc[-6] - r3.iloc[-1]) if len(r3.dropna()) > 6 else 0,
            rank_chg_20d=int(r3.iloc[-21] - r3.iloc[-1]) if len(r3.dropna()) > 21 else 0,
            score=int(round(float(score[t].iloc[-1]))) if pd.notna(score[t].iloc[-1]) else None,
            above50=bool(last(sub) > last(ma50)),
            above200=bool(last(sub) > last(ma200)),
            from_high=round(float(last(sub) / last(hi52) - 1) * 100, 1),
            rs_high=bool(rs_hi[t].iloc[-1]),
            rs_low=bool(rs_lo[t].iloc[-1]),
            rel_1d=round(float(ret1d[t].iloc[-1] - bench1d.iloc[-1]) * 100, 2),
            quad_w=qw[t].iloc[-1] if t in qw else None,
            quad_d=qd[t].iloc[-1] if t in qd else None,
            conflict=_conflict(ranks["12M"][t].iloc[-1], ranks["1M"][t].iloc[-1], n),
        ))
    rows.sort(key=lambda r: -(r["score"] or 0))
    state = dict(rank3m=ranks["3M"], rank1m=ranks["1M"], rank12m=ranks["12M"], rs_hi=rs_hi, rs_lo=rs_lo, n=n)
    return rows, state


def _conflict(rank_long, rank_short, n) -> str | None:
    if pd.isna(rank_long) or pd.isna(rank_short):
        return None
    third = n / 3
    if rank_long > n - third and rank_short <= third:
        return "rotating_in"
    if rank_long <= third and rank_short > n - third:
        return "rotating_out"
    return None


# ---------------------------------------------------------------------------
# Intermarket ratios & breadth
# ---------------------------------------------------------------------------
def _basket(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    rets = prices[[t for t in tickers if t in prices]].pct_change(fill_method=None).mean(axis=1).fillna(0)
    return (1 + rets).cumprod()


def ratios(prices: pd.DataFrame) -> tuple[list[dict], float, list[dict]]:
    px = prices.copy()
    px["CYC"] = _basket(prices, C.CYCLICALS)
    px["DEF"] = _basket(prices, C.DEFENSIVES)
    out, alerts = [], []
    for num, den, label, sign in C.RATIOS:
        if num not in px or den not in px:
            out.append(dict(id=f"{num}/{den}", label=label, status="missing"))
            continue
        r = (px[num] / px[den]).dropna()
        if len(r) < 260:
            out.append(dict(id=f"{num}/{den}", label=label, status="missing"))
            continue
        m50, m200 = r.rolling(50).mean(), r.rolling(200).mean()
        last, a50, a200 = r.iloc[-1], m50.iloc[-1], m200.iloc[-1]
        if last > a50 > a200:
            trend = "up"
        elif last < a50 < a200:
            trend = "down"
        else:
            trend = "mixed"
        above = r > m200
        cross = above.ne(above.shift(1)) & m200.notna() & m200.shift(1).notna()
        recent = cross.iloc[-C.ALERT_LOOKBACK_SESSIONS:]
        if recent.any():
            d = recent[recent].index[-1]
            direction = "above" if above.loc[d] else "below"
            n_cross = int(recent.sum())
            note = " (choppy — several crosses recently)" if n_cross >= 3 else ""
            alerts.append(dict(date=d, kind="ratio", severity="watch" if sign else "info",
                               text=f"{label} crossed {direction} its 200-day average{note}."))
        wk = r.resample("W-FRI").last().iloc[-53:]
        w50, w200 = m50.resample("W-FRI").last().iloc[-53:], m200.resample("W-FRI").last().iloc[-53:]
        base = r.iloc[0]
        out.append(dict(
            id=f"{num}/{den}", label=label, status="ok", sign=sign, trend=trend,
            chg_3m=round(float(r.iloc[-1] / r.iloc[-64] - 1) * 100, 1),
            spark=[round(float(v / base), 4) for v in wk],
            spark50=[round(float(v / base), 4) if pd.notna(v) else None for v in w50],
            spark200=[round(float(v / base), 4) if pd.notna(v) else None for v in w200],
        ))
    scored = [(1 if o["trend"] == "up" else -1 if o["trend"] == "down" else 0)
              for o in out if o.get("status") == "ok" and o["sign"]]
    risk = float(np.mean(scored)) if scored else 0.0
    return out, round(risk, 2), alerts


def breadth(prices: pd.DataFrame) -> dict:
    tickers = [t for t in list(C.SECTORS) + list(C.INDUSTRIES) if t in prices]
    px = prices[tickers]
    a50 = (px > px.rolling(50).mean()).mean(axis=1) * 100
    a200 = (px > px.rolling(200).mean()).mean(axis=1) * 100
    w50, w200 = a50.resample("W-FRI").last().iloc[-53:], a200.resample("W-FRI").last().iloc[-53:]
    return dict(
        above50=round(float(a50.iloc[-1])), above200=round(float(a200.iloc[-1])),
        above50_1m=round(float(a50.iloc[-22])), above200_1m=round(float(a200.iloc[-22])),
        dates=[d.strftime("%Y-%m-%d") for d in w50.index],
        s50=[round(float(v)) for v in w50], s200=[round(float(v)) for v in w200],
        n=len(tickers),
    )


# ---------------------------------------------------------------------------
# Alerts from prices (recomputed every run over the lookback window — no state needed)
# ---------------------------------------------------------------------------
def _first_true(mask: pd.Series) -> pd.Series:
    return mask & ~mask.shift(1, fill_value=False)


def price_alerts(prices, names, qw: pd.DataFrame, states: dict, market_labels: pd.Series) -> list[dict]:
    out = []
    since = prices.index[-min(C.ALERT_LOOKBACK_SESSIONS, len(prices) - 1)]

    # RRG weekly quadrant changes — completed weeks only (a partial week's bar moves daily)
    if qw.index[-1].weekday() != 4:
        qw = qw.iloc[:-1]
    for t in qw.columns:
        q = qw[t].dropna()
        ch = q[q.ne(q.shift(1))].iloc[1:]
        for d, new in ch.loc[since:].items():
            old = q.shift(1).loc[d]
            sev = {"Leading": "info", "Improving": "watch", "Weakening": "watch", "Lagging": "info"}[new]
            arrow = "↑" if new in ("Leading", "Improving") else "↓"
            out.append(dict(date=d, kind="rrg", severity=sev, ticker=t, up=new in ("Leading", "Improving"),
                            text=f"{arrow} {names[t]} ({t}) moved {old} → {new} on the weekly RRG."))

    for gname, st in states.items():
        thr = 3 if st["n"] <= 12 else 5
        r3 = st["rank3m"]
        for t in r3.columns:
            # rank jumps on 3M relative strength over 20 sessions
            chg = r3[t].shift(20) - r3[t]
            up, dn = _first_true(chg >= thr), _first_true(chg <= -thr)
            for d in up.loc[since:][up.loc[since:]].index:
                out.append(dict(date=d, kind="rank", severity="info", ticker=t, up=True,
                                text=f"↑ {names[t]} ({t}) jumped {int(chg[d])} places to #{int(r3[t][d])} on 3-month relative strength."))
            for d in dn.loc[since:][dn.loc[since:]].index:
                out.append(dict(date=d, kind="rank", severity="watch", ticker=t, up=False,
                                text=f"↓ {names[t]} ({t}) dropped {int(-chg[d])} places to #{int(r3[t][d])} on 3-month relative strength."))
            # RS line 52-week highs / lows (first in 20 sessions)
            hi = st["rs_hi"][t] & ~st["rs_hi"][t].shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
            lo = st["rs_lo"][t] & ~st["rs_lo"][t].shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
            for d in hi.loc[since:][hi.loc[since:]].index:
                out.append(dict(date=d, kind="rs", severity="info", ticker=t, up=True,
                                text=f"↑ {names[t]} ({t}) relative-strength line made a new 52-week high vs SPY."))
            for d in lo.loc[since:][lo.loc[since:]].index:
                out.append(dict(date=d, kind="rs", severity="watch", ticker=t, up=False,
                                text=f"↓ {names[t]} ({t}) relative-strength line made a new 52-week low vs SPY."))
            # horizon conflicts (first day)
            n = st["n"]
            third = n / 3
            rin = _first_true((st["rank12m"][t] > n - third) & (st["rank1m"][t] <= third))
            rout = _first_true((st["rank12m"][t] <= third) & (st["rank1m"][t] > n - third))
            for d in rin.loc[since:][rin.loc[since:]].index:
                out.append(dict(date=d, kind="conflict", severity="watch", ticker=t, up=True,
                                text=f"↻ {names[t]} ({t}): 12-month laggard now a 1-month leader — possible early rotation in."))
            for d in rout.loc[since:][rout.loc[since:]].index:
                out.append(dict(date=d, kind="conflict", severity="watch", ticker=t, up=False,
                                text=f"↻ {names[t]} ({t}): 12-month leader now a 1-month laggard — possible rotation out."))

    ml = market_labels
    ch = ml[ml.ne(ml.shift(1))].iloc[1:]
    for d, new in ch.loc[since:].items():
        out.append(dict(date=d, kind="phase", severity="warning",
                        text=f"Market-implied phase changed: {ml.shift(1).loc[d]} → {new}."))
    return out


# ---------------------------------------------------------------------------
# Trading implications
# ---------------------------------------------------------------------------
def implications(rows_all: list[dict], prices: pd.DataFrame, risk: float, fred: dict,
                 macro_phase: str, market_phase: str) -> dict:
    fav = [r for r in rows_all if r["quad_w"] in ("Leading", "Improving") and r["above50"] and (r["score"] or 0) >= 60]
    avoid = [r for r in rows_all if r["quad_w"] in ("Lagging", "Weakening") and not r["above50"] and (r["score"] or 100) <= 40]
    fav.sort(key=lambda r: -r["score"])
    avoid.sort(key=lambda r: r["score"])

    spy = prices[C.BENCHMARK].dropna()
    m50, m200 = spy.rolling(50).mean().iloc[-1], spy.rolling(200).mean().iloc[-1]
    checks = [
        dict(label="S&P 500 above its 200-day average", ok=bool(spy.iloc[-1] > m200)),
        dict(label="S&P 500 above its 50-day average", ok=bool(spy.iloc[-1] > m50)),
        dict(label="50-day average above 200-day", ok=bool(m50 > m200)),
        dict(label="Risk appetite ratios net positive", ok=risk > 0),
    ]
    hy = fred.get("BAMLH0A0HYM2")
    if hy is not None and len(hy) > 70:
        checks.append(dict(label="High-yield spreads not widening (3m)", ok=bool(hy.iloc[-1] - hy.iloc[-64] <= 0.25)))
    if "^VIX" in prices and prices["^VIX"].notna().any():
        checks.append(dict(label="VIX below 20", ok=bool(prices["^VIX"].dropna().iloc[-1] < 20)))
    passed = sum(c["ok"] for c in checks)
    penalty = 1 if macro_phase == "Recession" else 0
    pts = passed - penalty
    total = len(checks)
    if pts >= total - 1:
        verdict, level = "Full exposure — trend and risk appetite support breakouts.", "full"
    elif pts >= total - 3:
        verdict, level = "Normal exposure — be selective, favour the leading groups.", "normal"
    elif pts >= 1:
        verdict, level = "Reduced exposure — smaller size, A+ setups only.", "reduced"
    else:
        verdict, level = "Defensive — mostly cash; protect capital.", "defensive"

    tmpl = C.PHASE_TEMPLATES.get(macro_phase, {})
    favoured_by_phase = [s for s, w in tmpl.items() if w >= 1]
    top_sectors = [r["ticker"] for r in rows_all if r["ticker"] in C.SECTORS][:5]
    overlap = [s for s in top_sectors if s in favoured_by_phase]
    art = "an" if macro_phase[0] in "AEIOU" else "a"
    if len(overlap) >= 3:
        consistency = f"Leadership confirms the {macro_phase}-cycle read ({', '.join(overlap)} are in the top 5)."
    elif len(overlap) >= 1:
        consistency = f"Leadership partly fits {art} {macro_phase} cycle ({', '.join(overlap)} in the top 5)."
    else:
        consistency = f"Leadership does not fit {art} {macro_phase} cycle — none of {', '.join(favoured_by_phase)} are in the top 5."

    slim = lambda r: dict(ticker=r["ticker"], name=r["name"], score=r["score"], quad=r["quad_w"],
                          ret3m=r["ret"]["3M"], from_high=r["from_high"])
    return dict(
        favoured=[slim(r) for r in fav[:8]],
        avoid=[slim(r) for r in avoid[:8]],
        checks=checks, passed=passed, total=total, penalty=penalty,
        verdict=verdict, level=level, consistency=consistency,
        phase_favoured=favoured_by_phase,
    )


# ---------------------------------------------------------------------------
# "What changed" bullets
# ---------------------------------------------------------------------------
def what_changed(sector_rows, industry_rows, sector_state, qw: pd.DataFrame, names: dict,
                 market: dict, macro: dict, alerts_today: int) -> list[dict]:
    out = []
    srt = sorted(sector_rows, key=lambda r: r["rel_1d"])
    lo, hi = srt[0], srt[-1]
    out.append(dict(when="Since yesterday", text=(
        f"Best sector vs SPY: {hi['name']} {hi['rel_1d']:+.1f}%; worst: {lo['name']} {lo['rel_1d']:+.1f}%. "
        f"{alerts_today} new alert{'s' if alerts_today != 1 else ''} today.")))

    r3 = sector_state["rank3m"]
    d5 = (r3.iloc[-6] - r3.iloc[-1]).dropna()
    riser, faller = d5.idxmax(), d5.idxmin()
    wk = []
    if d5[riser] > 0:
        wk.append(f"{names[riser]} climbed #{int(r3[riser].iloc[-6])} → #{int(r3[riser].iloc[-1])}")
    if d5[faller] < 0:
        wk.append(f"{names[faller]} slipped #{int(r3[faller].iloc[-6])} → #{int(r3[faller].iloc[-1])}")
    nq = 0
    if len(qw) > 1:
        nq = int((qw.iloc[-1] != qw.iloc[-2]).sum())
    out.append(dict(when="Past week", text=(
        ("On 3-month strength, " + "; ".join(wk) + ". " if wk else "Sector ranks were stable on 3-month strength. ")
        + f"{nq} ETF{'s' if nq != 1 else ''} changed RRG quadrant.")))

    mp_now, mp_then = market["phase"], market["phase_1m_ago"]
    mm_now, mm_then = macro["phase"], macro.get("phase_1m_ago")
    month = []
    month.append(f"market-implied phase {mp_then} → {mp_now}" if mp_now != mp_then else f"market-implied phase steady at {mp_now}")
    month.append(f"macro phase {mm_then} → {mm_now}" if mm_then and mm_now != mm_then
                 else f"macro phase {mm_now} for {macro['months_in_phase']} month{'s' if macro['months_in_phase'] != 1 else ''}")
    d20 = (r3.iloc[-21] - r3.iloc[-1]).dropna()
    best = d20.idxmax()
    tail = f" Biggest sector riser: {names[best]} (+{int(d20[best])} places)." if d20[best] > 0 else ""
    out.append(dict(when="Past month", text=month[0][0].upper() + month[0][1:] + "; " + month[1] + "." + tail))
    return out
