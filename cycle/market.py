"""Market lens: which cycle phase does current sector leadership look like?

Each session, sectors are ranked on blended 3M + 6M performance relative to SPY.
That ranking is correlated (Spearman) with the classic leadership template of each
phase. The best-matching template is the market-implied phase. Scores are smoothed
(10-day EMA) and a new phase must lead for 5 straight sessions before the call flips.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C
from .macro import next_phase, prev_phase


def relative_returns(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    bench = prices[C.BENCHMARK]
    return (prices / prices.shift(n)).div(bench / bench.shift(n), axis=0) - 1


def _rank_rows(df: pd.DataFrame) -> np.ndarray:
    return df.rank(axis=1).to_numpy()


def phase_scores(prices: pd.DataFrame, sessions: int = 520) -> pd.DataFrame:
    sectors = [t for t in C.SECTORS if t in prices]
    sub = prices[sectors + [C.BENCHMARK]]
    r63 = relative_returns(sub, 63)[sectors]
    r126 = relative_returns(sub, 126)[sectors]
    z = lambda d: d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1), axis=0)
    blend = (z(r63) + z(r126)) / 2
    blend = blend.dropna(how="any").iloc[-sessions:]
    R = _rank_rows(blend)
    Rc = R - R.mean(axis=1, keepdims=True)

    out = {}
    for phase, tmpl in C.PHASE_TEMPLATES.items():
        t = pd.Series({s: tmpl.get(s, 0) for s in sectors}).rank().to_numpy()
        tc = t - t.mean()
        num = Rc @ tc
        den = np.sqrt((Rc ** 2).sum(axis=1) * (tc ** 2).sum())
        out[phase] = num / den
    scores = pd.DataFrame(out, index=blend.index)
    return scores.ewm(span=10, adjust=False).mean()


def label_with_persistence(scores: pd.DataFrame, confirm: int = 5) -> pd.Series:
    labels, current, challenger, streak = [], None, None, 0
    for _, row in scores.iterrows():
        top = row.idxmax()
        if current is None:
            current = top
        elif top != current:
            if top == challenger:
                streak += 1
            else:
                challenger, streak = top, 1
            if streak >= confirm:
                current, challenger, streak = top, None, 0
        else:
            challenger, streak = None, 0
        labels.append(current)
    return pd.Series(labels, index=scores.index)


def clock_point(row: pd.Series, scale: float = 2.0) -> tuple[float, float]:
    x = y = 0.0
    for p, s in row.items():
        w = max(float(s), 0.0)
        a = math.radians(C.PHASE_ANGLE[p])
        x += w * math.cos(a)
        y += w * math.sin(a)
    return round(x * scale, 2), round(y * scale, 2)


def market_call(prices: pd.DataFrame) -> tuple[dict, pd.Series]:
    scores = phase_scores(prices)
    labels = label_with_persistence(scores)
    last = scores.iloc[-1].sort_values(ascending=False)
    phase = labels.iloc[-1]
    top, second = float(last.iloc[0]), float(last.iloc[1])
    s_phase = float(scores.iloc[-1][phase])
    conf = max(0.0, min(1.0, s_phase)) * (0.5 + 0.5 * max(0.0, min(1.0, (s_phase - second if phase == last.index[0] else 0) / 0.4)))

    run = 0
    for v in reversed(labels.tolist()):
        if v != phase:
            break
        run += 1

    weekly = scores.resample("W-FRI").last().iloc[-53:]
    trail = []
    for d, row in weekly.iterrows():
        x, y = clock_point(row)
        trail.append(dict(d=d.strftime("%Y-%m-%d"), x=x, y=y))
    x, y = clock_point(scores.iloc[-1])

    lab_21 = labels.iloc[-22] if len(labels) > 21 else labels.iloc[0]
    hist = scores.iloc[-260:].resample("W-FRI").last()
    return dict(
        phase=phase,
        confidence=int(round(100 * conf)),
        sessions_in_phase=run,
        scores={k: round(float(v), 2) for k, v in scores.iloc[-1].items()},
        x=x, y=y,
        trail=trail,
        phase_1m_ago=lab_21,
        history={"dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                 **{p: [round(float(v), 3) for v in hist[p]] for p in C.PHASES}},
    ), labels


def compare(macro_phase: str, market_phase: str) -> dict:
    if macro_phase == market_phase:
        return dict(state="agree", text=f"Macro data and sector leadership agree: {macro_phase} cycle.")
    if market_phase == next_phase(macro_phase):
        return dict(state="market_leads",
                    text=f"Markets are pricing the next phase ({market_phase}) while macro data still reads {macro_phase}. "
                         "Markets usually move first — watch for the data to follow.")
    if market_phase == prev_phase(macro_phase):
        return dict(state="market_lags",
                    text=f"Sector leadership still looks {market_phase} while macro data has moved on to {macro_phase}. "
                         "Leadership may be stale or the macro read may be early.")
    return dict(state="conflict",
                text=f"Macro reads {macro_phase} but leadership looks {market_phase} — opposite sides of the cycle. "
                     "Treat both calls with low conviction until they converge.")
