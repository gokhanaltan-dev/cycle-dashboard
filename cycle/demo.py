"""Synthetic data for offline testing (`python -m cycle.build --demo`).

Produces prices and FRED-like series with plausible shapes so the whole pipeline
and page can be exercised without network access. Never used in the daily run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

NBER = [("1990-08", "1991-03"), ("2001-04", "2001-11"), ("2008-01", "2009-06"), ("2020-03", "2020-04")]


def demo_prices(seed: int = 7, years: int = C.PRICE_HISTORY_YEARS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end - pd.DateOffset(years=years), end)
    n = len(idx)
    t = np.arange(n)
    spy = 400 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n)))
    out = {C.BENCHMARK: spy}
    tickers = [x for x in C.ALL_TICKERS if x != C.BENCHMARK]
    for i, tk in enumerate(tickers):
        beta = rng.uniform(0.6, 1.4)
        phase = rng.uniform(0, 2 * np.pi)
        period = rng.uniform(180, 420)
        drift = 0.0006 * np.sin(2 * np.pi * t / period + phase)
        idio = np.cumsum(drift + rng.normal(0, 0.009, n))
        level = {"^VIX": 18, "^MOVE": 100, "DX-Y.NYB": 103, "HG=F": 4.2, "GC=F": 2400, "CL=F": 75,
                 "BTC-USD": 60000}.get(tk, rng.uniform(30, 250))
        if tk in ("^VIX", "^MOVE"):
            s = level * np.exp(0.25 * np.sin(2 * np.pi * t / 300 + phase) + rng.normal(0, 0.05, n))
        else:
            s = level * (spy / spy[0]) ** beta * np.exp(idio)
        out[tk] = s
    return pd.DataFrame(out, index=idx)


def demo_fred(seed: int = 11) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    m = pd.date_range("1980-01-01", end - pd.DateOffset(months=1), freq="MS")
    n = len(m)
    cycle = np.sin(2 * np.pi * np.arange(n) / 84) + rng.normal(0, 0.25, n).cumsum() * 0.05
    rec = pd.Series(0, index=m)
    for s, e in NBER:
        rec.loc[s:e] = 1
        cycle[(m >= s) & (m <= e)] -= 1.8
    cyc = pd.Series(cycle, index=m)
    noise = lambda k=0.3: rng.normal(0, k, n)
    fred = {
        "CFNAIMA3": cyc * 0.5 + noise(0.15),
        "PERMIT": 1300 * np.exp(np.cumsum(0.002 + 0.01 * cyc.diff().fillna(0) + noise(0.02))),
        "INDPRO": 100 * np.exp(np.cumsum(0.0015 + 0.004 * cyc + noise(0.004))),
        "UMCSENT": 85 + 10 * cyc + noise(3),
        "SAHMREALTIME": np.clip(0.2 - 0.25 * cyc + noise(0.05), 0, None),
        "GACDFSA066MSFRBPHI": 5 + 12 * cyc + noise(6),
        "GACDISA066MSFRBNY": 3 + 10 * cyc + noise(7),
        "CPIAUCSL": 100 * np.exp(np.cumsum(0.0022 + noise(0.001))),
        "USREC": rec,
    }
    fred = {k: pd.Series(np.asarray(v, dtype=float), index=m) for k, v in fred.items()}
    # weekly & daily series
    w = pd.date_range("1980-01-05", end, freq="W-SAT")
    cw = cyc.reindex(w, method="ffill").bfill()
    fred["ICSA"] = pd.Series(np.asarray(330000 - 60000 * cw + rng.normal(0, 8000, len(w))), index=w)
    fred["NFCI"] = pd.Series(np.asarray(-0.3 - 0.3 * cw + rng.normal(0, 0.05, len(w))), index=w)
    d = pd.bdate_range("1990-01-01", end)
    cd = cyc.reindex(d, method="ffill").bfill()
    fred["T10Y2Y"] = pd.Series(np.asarray(0.8 - 0.6 * cd + rng.normal(0, 0.05, len(d))), index=d)
    fred["T10Y3M"] = pd.Series(np.asarray(1.0 - 0.8 * cd + rng.normal(0, 0.05, len(d))), index=d)
    fred["DFII10"] = pd.Series(np.asarray(1.5 + 0.3 * cd + rng.normal(0, 0.03, len(d))), index=d)
    fred["T10YIE"] = pd.Series(np.asarray(2.2 + 0.2 * cd + rng.normal(0, 0.03, len(d))), index=d)
    fred["BAMLH0A0HYM2"] = pd.Series(np.asarray(4.0 - 1.2 * cd + rng.normal(0, 0.1, len(d))), index=d)
    fred["RRPONTSYD"] = pd.Series(np.asarray(300 + rng.normal(0, 20, len(d))), index=d)
    wd = pd.date_range("2003-01-01", end, freq="W-WED")
    fred["WALCL"] = pd.Series(np.linspace(900_000, 6_800_000, len(wd)), index=wd)
    fred["WTREGEN"] = pd.Series(np.asarray(700 + rng.normal(0, 40, len(wd))), index=wd)
    return fred


def demo_long(seed: int = 5) -> pd.DataFrame:
    """30 years of SPY and VIX with bull/bear cycles, for the mood model."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end - pd.DateOffset(years=30), end)
    n = len(idx)
    t = np.arange(n)
    drift = 0.0004 + 0.0012 * np.sin(2 * np.pi * t / 1300)
    rets = drift + rng.normal(0, 0.011, n)
    spy = 100 * np.exp(np.cumsum(rets))
    trend = pd.Series(rets).rolling(60, min_periods=1).mean().to_numpy()
    vix = np.exp(np.log(18) - 120 * trend + rng.normal(0, 0.06, n))
    return pd.DataFrame({"SPY": spy, "^VIX": vix}, index=idx)
