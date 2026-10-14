"""Offline checks. Run with:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd

from cycle import config as C
from cycle import macro as M
from cycle import market as MK
from cycle import rotation as R
from cycle.fetch import parse_fred_csv


def test_fred_csv_both_header_styles():
    new = "observation_date,ICSA\n2026-01-03,210000\n2026-01-10,.\n2026-01-17,215000\n"
    old = "DATE,ICSA\n2026-01-03,210000\n2026-01-17,215000\n"
    for txt in (new, old):
        s = parse_fred_csv(txt, "ICSA")
        assert list(s.values) == [210000, 215000]
        assert s.index[0] == pd.Timestamp("2026-01-03")


def test_quadrants_follow_the_clock():
    assert M.quadrant(1, 1) == "Mid"
    assert M.quadrant(-1, 1) == "Late"
    assert M.quadrant(-1, -1) == "Recession"
    assert M.quadrant(1, -1) == "Early"
    for p in C.PHASES:
        assert M.prev_phase(M.next_phase(p)) == p


def _prices_from_ranks(order, n=400):
    """Build prices where sectors' relative performance follows `order` (best first)."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    data = {C.BENCHMARK: np.linspace(100, 120, n)}
    for i, t in enumerate(order):
        drift = (len(order) - i) * 0.0004
        data[t] = data[C.BENCHMARK] * np.exp(drift * np.arange(n))
    return pd.DataFrame(data, index=idx)


def test_market_phase_recognises_each_template():
    for phase, tmpl in C.PHASE_TEMPLATES.items():
        order = sorted(C.SECTORS, key=lambda s: -tmpl.get(s, 0))
        call, _ = MK.market_call(_prices_from_ranks(order))
        assert call["phase"] == phase, (phase, call["scores"])


def test_rrg_quadrant_deadband():
    idx = pd.date_range("2025-01-03", periods=6, freq="W-FRI")
    ratio = pd.DataFrame({"X": [101, 100.1, 99.9, 99.6, 99.6, 101]}, index=idx)
    mom = pd.DataFrame({"X": [101, 101, 101, 101, 99, 99]}, index=idx)
    q = R.quadrant_frame(ratio, mom)["X"].tolist()
    # 99.9 is inside the dead-band, so it stays Leading until 99.6
    assert q == ["Leading", "Leading", "Leading", "Improving", "Lagging", "Weakening"]


def test_full_pipeline_on_demo_data():
    from cycle.build import run

    p = run(demo=True)
    json.dumps(p)  # serialisable
    assert p["macro"]["call"]["phase"] in C.PHASES
    assert p["market"]["phase"] in C.PHASES
    assert len(p["boards"]["sectors"]) == len(C.SECTORS)
    assert p["implications"]["level"] in {"full", "normal", "reduced", "defensive"}
    assert len(p["changed"]) == 3


def test_live_code_path_with_mocked_network(monkeypatch, tmp_path):
    """Exercise the real fetch -> build path with yfinance and FRED responses faked."""
    import types

    import requests
    import yfinance as yf

    from cycle import build as B
    from cycle.demo import demo_fred, demo_prices

    px = demo_prices()
    fred = demo_fred()

    def fake_download(tickers, **kw):
        sub = px[[t for t in tickers if t in px]]
        cols = pd.MultiIndex.from_product([["Close", "Open"], sub.columns], names=["Price", "Ticker"])
        return pd.DataFrame(np.hstack([sub.values, sub.values]), index=sub.index.rename("Date"), columns=cols)

    class FakeTicker:
        def __init__(self, t):
            self.t = t
            self.history_metadata = {"longName": f"{t} fund", "currency": "USD"}

        def history(self, **kw):
            return pd.DataFrame()

    def fake_get(url, **kw):
        sid = url.split("id=")[-1]
        s = fred[sid]
        text = "observation_date," + sid + "\n" + "\n".join(f"{d:%Y-%m-%d},{v}" for d, v in s.items())
        return types.SimpleNamespace(status_code=200, text=text, raise_for_status=lambda: None)

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(B, "DATA", tmp_path / "data")
    monkeypatch.setattr(B, "SITE", tmp_path / "site")
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    p = B.run(demo=False)
    assert p["health"]["price_missing"] == []
    assert p["health"]["fred_failed"] == []
    assert p["health"]["price_sources"] == ["yahoo"]
    # names in fake metadata don't contain the expected keywords -> identity warnings surface
    assert len(p["health"]["ticker_issues"]) > 0
    assert (tmp_path / "site" / "index.html").exists()
    assert (tmp_path / "data" / "snapshots.csv").exists()


def test_emotion_cycle_tracks_a_clean_cycle():
    from cycle import psych as PS

    assert PS.stage_of(180) == "Euphoria" and PS.stage_of(0) == "Despondency" and PS.stage_of(359) == "Despondency"
    idx = pd.bdate_range("1995-01-02", periods=7000)
    t = np.arange(len(idx))
    wave = np.sin(2 * np.pi * t / 1200)
    spy = pd.Series(100 * np.exp(0.0002 * t + 0.25 * wave), index=idx)
    vix = pd.Series(np.exp(np.log(18) - 0.5 * np.gradient(wave) * 1200 / (2 * np.pi) * 0.3), index=idx)
    df = PS.mood_frame(spy, vix).loc["2005":]
    peaks = spy.loc["2005":].rolling(601, center=True).max() == spy.loc["2005":]
    troughs = spy.loc["2005":].rolling(601, center=True).min() == spy.loc["2005":]
    near_top = {"Excitement", "Thrill", "Euphoria", "Anxiety", "Denial"}
    near_bottom = {"Panic", "Capitulation", "Despondency", "Depression", "Hope"}
    assert all(df.loc[d, "stage"] in near_top for d in peaks[peaks].index if d in df.index)
    assert all(df.loc[d, "stage"] in near_bottom for d in troughs[troughs].index if d in df.index)
