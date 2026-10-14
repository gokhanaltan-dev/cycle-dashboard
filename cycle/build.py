"""Daily pipeline: fetch -> compute -> data/latest.json + site/index.html.

    python -m cycle.build           # live run (needs internet)
    python -m cycle.build --demo    # synthetic data, offline test
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import macro as M
from . import market as MK
from . import rotation as R
from .render import render

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SITE = ROOT / "site"


def _clean(o):
    """Make the payload JSON-safe (NaN -> None, numpy -> python, timestamps -> str)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) or math.isinf(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def staleness(last: pd.Timestamp, freq: str, now: pd.Timestamp) -> bool:
    limit = {"D": 7, "W": 16, "M": 100}[freq]
    return (now - last).days > limit


def run(demo: bool = False) -> dict:
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    fred_ids = list(C.GROWTH_SERIES) + list(C.OVERLAY_SERIES) + C.LIQUIDITY_SERIES + [C.RECESSION_SERIES]

    if demo:
        from .demo import demo_fred, demo_prices
        prices, price_health = demo_prices(), {"source": {}, "missing": []}
        fred, fred_health = demo_fred(), {"failed": [], "via": "demo"}
        ticker_issues = []
    else:
        from . import fetch as F
        prices, price_health = F.fetch_prices()
        if prices.empty or C.BENCHMARK not in prices:
            raise SystemExit("No price data (benchmark missing) — aborting so the last good dashboard stays up.")
        fred, fred_health = F.fetch_fred(fred_ids)
        ticker_issues = F.validate_tickers(DATA / "ticker_meta.json", C.ALL_TICKERS)

    last_session = prices.index[-1]
    names = {C.BENCHMARK: "S&P 500"}
    for g in (C.SECTORS, C.INDUSTRIES, C.FACTORS, C.CROSS_ASSET):
        names.update({t: v[0] for t, v in g.items()})

    # ---- Cycle: macro lens
    Z, growth_info = M.growth_composite(fred, now)
    phase_m = M.classify(Z)
    macro_call = M.macro_call(Z, phase_m)
    bt = M.backtest(phase_m, fred.get(C.RECESSION_SERIES))
    comp = Z["_level"].loc["2000-01-01":].dropna()
    rec = fred.get(C.RECESSION_SERIES)
    rec_m = M.to_monthly(rec, "M").reindex(comp.index).fillna(0) if rec is not None else pd.Series(0, index=comp.index)
    history = dict(
        dates=[d.strftime("%Y-%m") for d in comp.index],
        level=[round(float(v), 2) for v in comp],
        momentum=[round(float(v), 2) if pd.notna(v) else None for v in Z["_momentum"].reindex(comp.index)],
        phase=[phase_m.get(d) for d in comp.index],
        recession=[int(v) for v in rec_m],
    )
    overlays = M.overlays(fred, prices)

    # ---- Cycle: market lens
    market_call, market_labels = MK.market_call(prices)
    agreement = MK.compare(macro_call["phase"], market_call["phase"])

    # ---- Rotation
    sec = [t for t in C.SECTORS if t in prices]
    ind = [t for t in C.INDUSTRIES if t in prices]
    fac = [t for t in C.FACTORS if t in prices]
    rw_r, rw_m = R.rrg(prices, sec + ind + fac, "weekly")
    rd_r, rd_m = R.rrg(prices, sec + ind + fac, "daily")
    qw, qd = R.quadrant_frame(rw_r, rw_m), R.quadrant_frame(rd_r, rd_m)
    rrg_payload = dict(
        weekly=R.rrg_payload(rw_r, rw_m, C.RRG["weekly"]["tail"]),
        daily=R.rrg_payload(rd_r, rd_m, C.RRG["daily"]["tail"], C.RRG["daily"]["step"]),
    )
    boards, states = {}, {}
    for gname, group in C.GROUPS.items():
        g = {t: v for t, v in group.items() if t in prices}
        boards[gname], st = R.leaderboard(prices, g, qw, qd)
        if gname != "factors":
            states[gname] = st
    ratio_rows, risk, ratio_alerts = R.ratios(prices)
    br = R.breadth(prices)

    # ---- Alerts
    alerts = R.price_alerts(prices, names, qw[sec + ind], states, market_labels)
    alerts += ratio_alerts + M.macro_alerts(fred)
    if agreement["state"] != "agree":
        alerts.append(dict(date=last_session, kind="divergence",
                           severity="warning" if agreement["state"] == "conflict" else "watch",
                           text=agreement["text"]))
    for a in alerts:
        a["date"] = pd.Timestamp(a["date"]).normalize()
    alerts = R.dedupe_alerts(alerts)
    for a in alerts:
        a["new"] = a["date"] >= last_session
    alerts.sort(key=lambda a: (a["date"], {"warning": 2, "watch": 1, "info": 0}[a["severity"]]), reverse=True)
    alerts = alerts[:120]
    n_new = sum(a["new"] for a in alerts)

    # ---- Implications & narrative
    rows_all = boards["sectors"] + boards["industries"]
    rows_all.sort(key=lambda r: -(r["score"] or 0))
    impl = R.implications(rows_all, prices, risk, fred, macro_call["phase"], market_call["phase"])
    changed = R.what_changed(boards["sectors"], boards["industries"], states["sectors"], qw[sec], names,
                             market_call, macro_call, n_new)

    # ---- Data health
    health_rows = []
    for row in growth_info:
        if row["status"] == "ok":
            row["stale"] = staleness(pd.Timestamp(row["last_obs"]), row["freq"], now)
    for sid in C.OVERLAY_SERIES:
        s = fred.get(sid)
        if s is not None and len(s):
            freq = C.OVERLAY_SERIES[sid]["freq"]
            if staleness(s.index.max(), freq, now):
                health_rows.append(f"{sid} last updated {s.index.max():%Y-%m-%d} (stale)")
    price_stale = (now - last_session).days > 4
    stale_growth = [r["id"] for r in growth_info if r.get("stale")]
    health = dict(
        price_last=last_session.strftime("%Y-%m-%d"),
        price_stale=price_stale,
        price_missing=price_health["missing"],
        price_sources=sorted(set(price_health["source"].values())) or ["demo"],
        fred_failed=fred_health["failed"],
        fred_via=fred_health["via"],
        stale_series=stale_growth + health_rows,
        ticker_issues=ticker_issues,
        n_prices=int(prices.shape[1]),
    )

    payload = dict(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        as_of=last_session.strftime("%Y-%m-%d"),
        demo=demo,
        macro=dict(call=macro_call, indicators=growth_info, history=history, backtest=bt),
        overlays=overlays,
        market=market_call,
        agreement=agreement,
        changed=changed,
        boards=boards,
        rrg=rrg_payload,
        ratios=ratio_rows,
        risk_appetite=risk,
        breadth=br,
        alerts=[dict(a, date=a["date"].strftime("%Y-%m-%d")) for a in alerts],
        implications=impl,
        names=names,
        health=health,
        config=dict(horizons=list(C.HORIZONS), phases=C.PHASES, templates=C.PHASE_TEMPLATES,
                    groups={g: list(v) for g, v in C.GROUPS.items()}),
    )

    # ---- Snapshot log (real-time record of each day's call; immune to data revisions)
    snap_path = DATA / ("snapshots_demo.csv" if demo else "snapshots.csv")
    snap = dict(
        date=payload["as_of"], macro_phase=macro_call["phase"], macro_conf=macro_call["confidence"],
        market_phase=market_call["phase"], market_conf=market_call["confidence"],
        exposure=impl["level"], top3=" ".join(r["ticker"] for r in boards["sectors"][:3]),
        risk_appetite=risk,
    )
    snaps = pd.read_csv(snap_path) if snap_path.exists() else pd.DataFrame(columns=list(snap))
    snaps = snaps[snaps["date"] != snap["date"]]
    snaps = pd.concat([snaps, pd.DataFrame([snap])], ignore_index=True).sort_values("date")
    DATA.mkdir(exist_ok=True)
    snaps.to_csv(snap_path, index=False)
    payload["snapshots"] = snaps.tail(30).iloc[::-1].to_dict(orient="records")

    payload = _clean(payload)
    (DATA / ("latest_demo.json" if demo else "latest.json")).write_text(json.dumps(payload, separators=(",", ":")))
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(render(payload))
    print(f"[build] {payload['as_of']}: macro {macro_call['phase']} ({macro_call['confidence']}%), "
          f"market {market_call['phase']} ({market_call['confidence']}%), exposure {impl['level']}, "
          f"{len(alerts)} alerts ({n_new} new)")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline test)")
    args = ap.parse_args()
    run(demo=args.demo)


if __name__ == "__main__":
    main()
