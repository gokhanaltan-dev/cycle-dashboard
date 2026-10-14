"""Calibrate the emotion-cycle model on real S&P 500 / VIX history.

Runs a grid of model variants and scores each on how it labels the peaks and
troughs of every ≥15% S&P 500 fall since 2000. Writes data/calib_psych.json.

    python tools/calibrate_psych.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cycle import fetch as F  # noqa: E402
from cycle import psych as PS  # noqa: E402

# NB: the live model's settings live in cycle/psych.py (LEVEL_SPAN, MOM_LAG, MOM_SPAN, USE_VIX).

PEAK_OK = set(PS.PEAK_OK)
TROUGH_OK = set(PS.TROUGH_OK)


def variant_frame(spy, vix, vix_mode, lvl_span, mom_lag, mom_span):
    df = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
    z = PS._z
    ext = df["spy"] / df["spy"].rolling(200).mean() - 1
    lo, hi = df["spy"].rolling(252).min(), df["spy"].rolling(252).max()
    rng = (df["spy"] - lo) / (hi - lo)
    parts = [z(ext), z(rng)]
    if vix_mode == "z10y":
        parts.append(-z(np.log(df["vix"])))
    elif vix_mode == "rel1y":
        parts.append(-z(np.log(df["vix"] / df["vix"].rolling(252).median())))
    level = pd.concat(parts, axis=1).mean(axis=1, skipna=False).ewm(span=lvl_span, adjust=False).mean()
    mom = z(level - level.shift(mom_lag)).ewm(span=mom_span, adjust=False).mean()
    out = pd.DataFrame({"spy": df["spy"], "y": level, "x": mom}).dropna()
    phis, prev = [], None
    for x, y in zip(out["x"], out["y"]):
        p = (math.degrees(math.atan2(x, -y)) + 360) % 360
        if prev is not None and math.hypot(x, y) < PS.R_MIN:
            p = prev
        phis.append(p)
        prev = p
    out["phi"] = phis
    labels, cur = [], None
    for p in out["phi"]:
        c = PS.stage_of(p)
        if cur is None or (c != cur and PS._dist(p, PS.CENTRE[c]) + PS.STAGE_MARGIN < PS._dist(p, PS.CENTRE[cur])):
            cur = c
        labels.append(cur)
    out["stage"] = labels
    return out


def main():
    long = F.fetch_long()
    if long.empty:
        raise SystemExit("could not download SPY/VIX history")
    grid = itertools.product(["z10y", "rel1y", "none"], [20, 30], [42, 63, 126], [10, 15])
    results = []
    for vix_mode, ls, lag, ms in grid:
        df = variant_frame(long["SPY"], long["^VIX"], vix_mode, ls, lag, ms)
        tps = [t for t in PS.turning_points(df, since="2000-01-01")]
        peak_hits = sum(t["peak_stage"] in PEAK_OK for t in tps)
        trough_hits = sum(t["trough_stage"] in TROUGH_OK for t in tps)
        sw = float(df["stage"].iloc[-2520:].ne(df["stage"].iloc[-2520:].shift(1)).sum()) / 10
        results.append(dict(vix=vix_mode, level_span=ls, mom_lag=lag, mom_span=ms,
                            peak_hits=peak_hits, trough_hits=trough_hits, events=len(tps),
                            score=round((peak_hits + trough_hits) / max(1, 2 * len(tps)), 3),
                            switches_per_year=round(sw, 1), now=df["stage"].iloc[-1],
                            detail=[(t["peak"], t["peak_stage"], t["trough"], t["trough_stage"]) for t in tps]))
    results.sort(key=lambda r: (-r["score"], r["switches_per_year"]))
    out = Path(__file__).resolve().parents[1] / "data" / "calib_psych.json"
    out.write_text(json.dumps(results, indent=1))
    for r in results[:10]:
        print(r["score"], r["vix"], r["level_span"], r["mom_lag"], r["mom_span"], r["switches_per_year"], r["now"])


if __name__ == "__main__":
    main()
