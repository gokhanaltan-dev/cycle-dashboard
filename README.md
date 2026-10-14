# Cycle & Rotation Dashboard

A one-page dashboard that answers three questions every weekday after the US close:

1. **Where are we in the business cycle?** Two independent reads, shown on a cycle clock:
   - the **macro read**, from FRED growth data;
   - the **market read**, from which sectors are leading.
   When the two disagree, the page says so, because that divergence is often the earliest signal.
2. **Which sectors and industries are rotating in or out?** Relative strength vs SPY over 1W/1M/3M/6M/12M, rank changes, a Relative Rotation Graph, intermarket ratios and breadth.
3. **What does that mean for trading?** An exposure verdict from a transparent checklist, plus which groups to focus breakout scans on and which to avoid.

It runs on GitHub Actions for free, publishes to GitHub Pages, and needs no manual input.

---

## Setup (about 10 minutes, once)

1. **Create a public repository** on GitHub, e.g. `cycle-dashboard`.
   - It must be public because free GitHub Pages requires it.
   - It only contains public market data and code.
2. **Upload these files** to the repository, keeping the folder structure.
   - The `.github/workflows/daily.yml` file is essential.
   - If your computer hides the `.github` folder when you drag and drop, add the file by hand instead: on GitHub, use **Add file → Create new file**, type `.github/workflows/daily.yml` as the name, and paste in its contents.
3. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
4. **Run it for the first time:** go to the **Actions** tab, open **Daily update**, and click **Run workflow**. It takes 2–4 minutes.
5. **Open your dashboard** at `https://<your-username>.github.io/<repo-name>/` and bookmark it.

From then on it rebuilds itself at 22:15 UTC, Monday to Friday. That's 23:15 UK time in summer and 22:15 in winter.

### Optional

- **FRED API key.** The pipeline uses FRED's public CSV download by default, which needs no key. For more reliable access, get a free key at <https://fred.stlouisfed.org/docs/api/api_key.html> and add it under **Settings → Secrets and variables → Actions → New repository secret**, named `FRED_API_KEY`.
- **Failure emails.** GitHub emails you when a scheduled run fails. If a day's run fails, the last good dashboard stays online, and its header turns red once the data is more than 4 days old.

---

## What's on the page

| Section | What it shows |
|---|---|
| Where we stand | Macro phase and market-implied phase, each with a confidence score. Also the exposure verdict and what changed since yesterday, last week and last month. |
| Cycle clock | Growth level vs momentum, with 12-month trails for both reads. Next to it, how closely current leadership matches each phase over time. |
| Macro dashboard | Growth composite since 2000, with NBER recessions shaded. Each input gets a z-score, a status and a sparkline. Context cards cover the curve, real yields, inflation, credit spreads, financial conditions, net liquidity, VIX and the dollar. |
| Rotation | Sortable heatmap for sectors, industries and factors: relative performance, rank changes, RRG quadrant, trend, distance from high and signals. Plus a weekly or daily RRG with tails. |
| Risk appetite & breadth | 14 intermarket ratios (Discretionary/Staples, Copper/Gold, HYG/IEF, IWM/SPY and others), each with its trend. Also the % of ETFs above their 50/200-day averages. |
| Rotation alerts | Quadrant changes, rank jumps, relative-strength highs and lows, horizon conflicts, ratio crosses, macro thresholds and phase changes. |
| Trading implications | Exposure checklist, groups to focus on, groups to avoid, and whether leadership fits the macro phase. |
| Track record & data health | Backtest of the macro phase against NBER recessions, a daily call log, data freshness and ticker identity checks. |

The **Method** section at the bottom of the page explains every calculation.

---

## Run it on your own machine

```bash
pip install -r requirements.txt
python -m cycle.build --demo    # synthetic data, no internet needed
python -m cycle.build           # live data
open site/index.html            # or double-click the file
python -m pytest -q             # offline tests
```

## Customise

Everything tunable lives in `cycle/config.py`:

- **Universe:** add or remove sector, industry, factor or cross-asset ETFs.
- **Macro inputs:** FRED series IDs, transforms and signs.
- **Phase templates:** which sectors lead in each phase.
- **Horizons and score weights.**
- **RRG settings:** smoothing, window, tail length.
- **Intermarket ratios.**

Push a change and the next run picks it up. You can also trigger a run by hand from the Actions tab.

## Files

```
cycle/config.py     universe, series, templates, settings
cycle/fetch.py      Yahoo (yfinance) + Stooq fallback, FRED, ticker identity checks
cycle/macro.py      growth composite, phase clock, overlays, NBER backtest, macro alerts
cycle/market.py     market-implied phase (Spearman match vs leadership templates)
cycle/rotation.py   leaderboards, RRG, ratios, breadth, alerts, implications
cycle/build.py      orchestrates a run → data/latest.json + site/index.html
web/template.html   the page (single file, chart library inlined, no external calls)
data/snapshots.csv  the daily call log, committed by the workflow
```

## Troubleshooting

- **Header says data is stale.** Open the Actions tab and read the latest run's log. Yahoo occasionally rate-limits GitHub's servers. The pipeline retries and falls back to Stooq, and the next day's run usually recovers.
- **A ticker or series is missing.** Data health at the bottom of the page lists what failed. The rest of the page still renders without it.
- **Scheduled runs stopped.** GitHub pauses schedules in repositories with no activity for 60 days. The daily commit of `data/snapshots.csv` normally prevents this. If it happens anyway, re-enable the schedule from the Actions tab.

Not investment advice. This is a decision aid for your own process.
