# The Fragility of Shadow Dollar Settlement

Code for my MIT Sloan MSMS thesis, *"The Fragility of Shadow Dollar Settlement: Tail Risk and Peg Instability in Tether Under Macro-Financial Stress"* (advised by Prof. Hong Ru, May 2026, Grade A).

## Research question

USDT is the largest dollar-pegged stablecoin and works as a dollar substitute for many users in emerging markets, outside central-bank backstops and deposit insurance. The thesis asks whether USDT's peg becomes less stable under crypto-market stress and under macro-financial stress, and whether the two kinds of stress act through the same channel.

## Approach

- **Risk measures.** From daily USDT close prices (2017–2026), build rolling peg-deviation volatility, 5% Expected Shortfall (downside and upside) and depeg frequency.
- **Stress regimes.** Classify each day as CryptoStress (BTC daily return < −5%), MacroStress (VIX > 30), both, or neither.
- **Regressions.** OLS with HAC standard errors on Expected Shortfall, and logits on depeg events, with SOFR and extended macro controls (dollar index, EM FX, credit spreads).
- **Event study.** Six historical episodes: Bitfinex/Tether 2018, Turkey 2018, COVID 2020, Russia sanctions 2022, Luna/UST 2022, SVB/USDC 2023.
- **Robustness.** HAC lag sweeps (5–50), window-length and threshold sensitivity, TED-spread sub-sample.

## Main findings

- Crypto and macro stress affect USDT's downside risk through different channels. CryptoStress mainly raises how often the peg breaks (logit odds ×5.9).
- MacroStress has no independent effect on depeg frequency. Its negative effect on downside severity is the net of two opposing forces: wider credit spreads increase downside risk, while EM dollar pressure (USDT as a refuge) reduces it, and the EM channel is about 2.25× larger.
- The twenty most extreme deviations all occur outside the stress regimes. They come from issuer-side events (e.g. Bitfinex 2018) that price-based market indicators cannot see. The thesis treats this as a scope limit of the method.

## Repository layout

```
run_analysis.py      # main entry point: risk metrics, regressions, figures, tables
src/                 # risk metrics, regressions, bilateral analysis, event study, plots
scripts/             # data fetching, sensitivity and lag sweeps, table cleanup and export
data/                # daily USDT, BTC and macro series used in the thesis
output/              # generated figures (PDF/PNG) and tables (CSV/TXT)
```

## Reproducing the results

```bash
pip install -r requirements.txt
make all
```

`make all` runs the full analysis, the event study and the table export from the CSVs in `data/`. To refresh the data, set `FRED_API_KEY` and run `make fetch` (USDT and BTC prices come from CryptoCompare, macro series from FRED).
