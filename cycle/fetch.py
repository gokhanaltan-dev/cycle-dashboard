"""Data download: prices (Yahoo via yfinance, Stooq fallback) and macro (FRED).

Every fetch is defensive: a failure for one ticker or series is logged in the
returned `health` record and the pipeline carries on with what it has.
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from . import config as C

UA = {"User-Agent": "Mozilla/5.0 (cycle-dashboard; +https://github.com)"}


def _log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def _yf_batch(tickers: list[str], years: int) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        multi_level_index=True,
        timeout=30,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:  # single ticker
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.dropna(how="all", axis=1)


def _stooq(ticker: str, years: int) -> pd.Series | None:
    """Best-effort fallback for US ETFs. Returns None on any problem."""
    if ticker in C.NON_EQUITY:
        return None
    sym = ticker.lower().replace(".", "-") + ".us"
    d1 = (datetime.now(timezone.utc) - timedelta(days=365 * years + 10)).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&i=d"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200 or not r.text.startswith("Date"):
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
        return df["Close"].rename(ticker) if "Close" in df else None
    except Exception:
        return None


def fetch_prices(tickers: list[str] | None = None, years: int = C.PRICE_HISTORY_YEARS):
    """Return (prices DataFrame, health dict). Prices are adjusted daily closes."""
    tickers = tickers or C.ALL_TICKERS
    frames, source = [], {}
    missing = list(tickers)
    for attempt in range(3):
        if not missing:
            break
        try:
            got = _yf_batch(missing, years)
        except Exception as e:  # network / rate limit
            _log(f"yfinance attempt {attempt + 1} failed: {e}")
            got = pd.DataFrame()
        if not got.empty:
            ok = [t for t in got.columns if got[t].notna().sum() > 200]
            frames.append(got[ok])
            for t in ok:
                source[t] = "yahoo"
            missing = [t for t in missing if t not in ok]
        if missing:
            time.sleep(5 * (attempt + 1))

    for t in list(missing):
        s = _stooq(t, years)
        if s is not None and s.notna().sum() > 200:
            frames.append(s.to_frame())
            source[t] = "stooq"
            missing.remove(t)

    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    # Futures / indices trade on different calendars: align to the benchmark's sessions
    if C.BENCHMARK in prices:
        sessions = prices[C.BENCHMARK].dropna().index
        prices = prices.reindex(sessions).ffill(limit=3)

    health = {"source": source, "missing": missing}
    if missing:
        _log(f"missing tickers: {missing}")
    return prices, health


# ---------------------------------------------------------------------------
# Ticker identity checks (cached; refreshed every 30 days)
# ---------------------------------------------------------------------------
def _expected():
    out = {C.BENCHMARK: "S&P 500"}
    for group in (C.SECTORS, C.INDUSTRIES, C.FACTORS, C.CROSS_ASSET):
        out.update({t: kw for t, (_, kw) in group.items()})
    return out


def validate_tickers(cache_path: Path, tickers: list[str]) -> list[dict]:
    """Check each ticker resolves to the intended instrument and currency.

    Returns a list of issues. Metadata is cached so Yahoo is hit at most monthly.
    """
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    fresh_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    stale = [t for t in tickers if t not in cache or cache[t].get("checked", "") < fresh_cutoff]
    if stale:
        try:
            import yfinance as yf

            for t in stale:
                try:
                    tk = yf.Ticker(t)
                    tk.history(period="5d")
                    md = tk.history_metadata or {}
                    cache[t] = {
                        "name": md.get("longName") or md.get("shortName") or "",
                        "currency": md.get("currency", ""),
                        "exchange": md.get("fullExchangeName") or md.get("exchangeName", ""),
                        "type": md.get("instrumentType", ""),
                        "checked": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as e:
                    _log(f"metadata failed for {t}: {e}")
                time.sleep(0.3)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
        except Exception as e:
            _log(f"metadata refresh skipped: {e}")

    issues, expected = [], _expected()
    norm = lambda s: s.lower().replace("–", "-").replace("-", " ")
    for t in tickers:
        md = cache.get(t)
        if not md:
            issues.append({"ticker": t, "issue": "no metadata"})
            continue
        kw = expected.get(t, "")
        if kw and norm(kw) not in norm(md.get("name", "")):
            issues.append({"ticker": t, "issue": f"name '{md.get('name')}' does not mention '{kw}'"})
        if t not in C.NON_EQUITY and md.get("currency") and md["currency"] != "USD":
            issues.append({"ticker": t, "issue": f"currency {md['currency']} (expected USD)"})
    return issues


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def _fred_csv(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return parse_fred_csv(r.text, series_id)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]  # 'observation_date' (current) or 'DATE' (older)
    val_col = series_id if series_id in df.columns else df.columns[1]
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna().rename(series_id)


def _fred_api(series_id: str, key: str) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={"series_id": series_id, "api_key": key, "file_type": "json"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
    return pd.to_numeric(s, errors="coerce").dropna().rename(series_id)


def fetch_fred(series_ids: list[str]):
    """Return ({id: Series}, health). Uses the official API when FRED_API_KEY is set."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    out, failed = {}, []
    for sid in series_ids:
        s = None
        for attempt in range(3):
            try:
                s = _fred_api(sid, key) if key else _fred_csv(sid)
                break
            except Exception as e:
                _log(f"FRED {sid} attempt {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))
        if s is not None and len(s):
            out[sid] = s
        else:
            failed.append(sid)
    return out, {"failed": failed, "via": "api" if key else "csv"}
