# ARIS500

Systematic equity model by Alexander Villax.

This project builds a point-in-time US equity research panel, evaluates
cross-sectional stock-selection signals, combines five accepted ideas into a
composite score, and constructs a long-only portfolio. I built it as a
personal research project, and the supported workflows run from notebooks
without needing a terminal, Git, or prior knowledge of the codebase.

## Start here

Open [`notebooks/START_HERE.ipynb`](notebooks/START_HERE.ipynb) in JupyterLab.
It explains setup, the notebook order, expected run times, who to ask for help,
and what each common error means. It contains no code.

The whole interface is six notebooks, including the orientation page:

| Order | Notebook | Use it for |
|---:|---|---|
| — | [`START_HERE.ipynb`](notebooks/START_HERE.ipynb) | Orientation, first-time setup, and troubleshooting |
| 01 | [`01_build_data.ipynb`](notebooks/01_build_data.ipynb) | First build, major data refreshes, or universe/data-rule changes |
| 02 | [`02_monthly_update.ipynb`](notebooks/02_monthly_update.ipynb) | The normal monthly price refresh, panel rebuild, target, and trade review |
| 03 | [`03_signal_research.ipynb`](notebooks/03_signal_research.ipynb) | Alignment tests and individual signal-block research |
| 04 | [`04_backtest.ipynb`](notebooks/04_backtest.ipynb) | Composite research, portfolio construction, performance, and attribution |
| 05 | [`05_stock_lookup.ipynb`](notebooks/05_stock_lookup.ipynb) | Type a ticker and review its signal, block, and composite rankings |

Every workflow notebook says what must be run first, what it writes, how long it
usually takes, what successful output looks like, and what to do next.

## First-time setup without a terminal

1. Install Anaconda Distribution or Miniforge and open **Anaconda Navigator**.
2. Use Python 3.11 or newer. Install the packages in `requirements.txt`. If you
   do not manage Python environments, set up one environment from
   `requirements.txt` once rather than troubleshooting packages individually.
3. Clone the repository. If you do not use a terminal, use GitHub Desktop and
   choose **Clone repository**.
4. Launch JupyterLab from Navigator, browse to the project folder, and open
   `notebooks/START_HERE.ipynb`.
5. Confirm the WRDS username in `config.py`. Never store a WRDS password in the
   project. You must have permission for the required CRSP and Compustat data.
6. Open notebook 01, set its WRDS refresh switch to `True`, and run one cell at
   a time.

The first WRDS build usually takes 30 minutes to two hours, plus 10–20 minutes
to build the panel. Cached runs are much faster. A normal monthly update usually
takes 5–20 minutes.

Brokerage trading lives in the separate private `ARIS500_Trading_Alpaca`
repository. This repository sends it a dated target-portfolio CSV. After fills,
that repository reads actual brokerage positions and, only after explicit human
confirmation, reconciles them back to this repository's
`data/current_portfolio.csv`. This model repository contains no credentials or
brokerage code.

## Five supported workflows

### 1. Build data from scratch

**Prerequisites:** a cloned repository, the packages in `requirements.txt`, a
WRDS username, and CRSP/Compustat access.

1. Open `notebooks/01_build_data.ipynb`.
2. Set `REQUERY_ALL_WRDS_SOURCES = True` for the first build.
3. Run the cells in order: prices and short interest, factors and benchmarks,
   fundamentals and reference tables, live prices, then the configured panel.
4. Stop if a blocking panel check fails.

**Produces:** the source Parquet files under `data/` and the one panel selected
by `UNIVERSE`, normally `data/panel_russell3000.parquet`. A fresh pull usually
takes 30 minutes to two hours, followed by roughly 10–20 minutes for the panel.
Cache-based rebuilds are commonly 5–20 minutes.

### 2. Run the monthly update

**Prerequisites:** workflow 1 has succeeded at least once, and
`data/current_portfolio.csv` contains the actual reconciled holdings from the
previous rebalance. Only the explicit first launch may use
`FIRST_REBALANCE = True`.

1. Open `notebooks/02_monthly_update.ipynb`.
2. Run the mandatory fresh WRDS live-price pull and confirm its date.
3. Rebuild the panel and require zero blocking failures.
4. Generate and manually review the target, quarantine table, trade list,
   warnings, constraints, and current-portfolio summary.
5. Continue in the private trading repository to review and submit orders.
6. After all orders reach a terminal status, run its reconciliation step. That
   step reads actual brokerage positions and updates
   `data/current_portfolio.csv` here after a separate confirmation.

**Produces:** refreshed live prices and panel data, a dated target portfolio and
trade list, and—after trading—the reconciled current-portfolio file. The model
notebook usually takes 5–20 minutes; human review and execution time are
additional.

### 3. Research a signal

**Prerequisites:** workflow 1 and a valid Russell 3000 panel. If the signal
needs a new source field, add and rebuild that field before evaluating it.

1. Follow the complete contributor path in **Adding a signal** below.
2. Add the registered block to the extension cell in
   `notebooks/03_signal_research.ipynb`.
3. Run the alignment harness first, then the registered block evaluation.
4. Record the result and keep/reject decision in `docs/signal_log.md`.

**Produces:** notebook tables and charts plus a permanent research decision;
it writes no production portfolio. Reproducing all existing blocks usually
takes 3–8 minutes; a new block's time depends on its rolling calculations.

### 4. Run the backtest

**Prerequisites:** workflow 1, and workflow 3 whenever a signal definition or
live block has changed.

1. Open `notebooks/04_backtest.ipynb`.
2. Run the composite comparison, configured portfolio backtest, and portfolio
   analysis in order.
3. Review structural diagnostics and limitations rather than selecting settings
   from the best historical Sharpe.

**Produces:** in-notebook composite, performance, attribution, concentration,
and cost diagnostics; it writes no portfolio or data files. Allow roughly
5–15 minutes.

### 5. Look up a stock

**Prerequisites:** workflow 1; run workflow 2 first if the latest provisional
month is required.

1. Open `notebooks/05_stock_lookup.ipynb` and run the setup cell once.
2. Enter a ticker in `TICKER` and run the lookup cell.
3. Change only `TICKER`, `LOOKUP_DATE`, or `HISTORY_MONTHS` for another view.

**Produces:** current individual-signal, block, and composite ranks plus history
charts; it writes no files. Initial scoring takes about one minute, and later
lookups take seconds.

## The monthly process has human checkpoints

Notebook 02 is designed to be rerun each month without guessing what happened
last time. It walks through:

1. refresh provisional live prices;
2. rebuild and validate the panel;
3. generate the proposed target and trade list;
4. print the current target's holdings, active positions, sectors, size, and
   signal tilts;
5. manually review every quarantined live return;
6. manually review warnings, buys, sells, constraints, and sector weights; and
7. after execution, run the private trading repository's reconciliation and
   confirm its update to `data/current_portfolio.csv`.

The quarantine review, trade approval, and reconciliation confirmation are
deliberate human checkpoints. The model does not approve trades or repair
corporate actions. The target and trade files are proposals; only the separate
private trading repository reads actual positions and writes reconciled holdings
back after confirmation.

A missing or empty `current_portfolio.csv` stops the rebalance. Starting from
cash is permitted only when `FIRST_REBALANCE = True` is explicitly set for the
portfolio's first launch. Dated target and trade files also refuse silent
replacement; an operator must deliberately enable the overwrite argument after
reviewing the existing files.

## What the project produces

The main local outputs are:

- `data/prices.parquet`: monthly CRSP common-stock prices and returns;
- `data/live_prices.parquet`: provisional post-CRSP monthly extension;
- `data/short_interest.parquet`: point-in-time security short interest;
- `data/fundamentals.parquet`: annual and quarterly Compustat statements;
- `data/link.parquet` and `data/sectors.parquet`: date-valid company links and
  GICS history;
- `data/factors.parquet`: factor and benchmark total returns;
- one configured monthly point-in-time panel: `data/panel_sp500.parquet`,
  `data/panel_russell3000.parquet`, or `data/panel_all.parquet`, according to
  `UNIVERSE`—one run builds only the selected panel; and
- `data/target_portfolio_YYYYMMDD.csv` and
  `data/trade_list_YYYYMMDD.csv`: proposed live holdings and trades.

Generated data are ignored by Git. Slow WRDS queries are cached outside the
repository so an ordinary rerun does not download everything again.

## Five rules that protect the research

1. A signal is a function that takes the panel and returns one row-aligned
   column.
2. Signals use the row as-is. Fundamentals are already point-in-time and are
   never shifted again.
3. Signal modules never look forward. Evaluation and backtest modules obtain
   next-period returns only through `src/utils.forward_returns`.
4. Nothing the model depends on lives in a notebook. Notebooks run workflows
   and present results; Python modules under `src/` produce them.
5. Every user-adjustable data, signal, composite, portfolio, and analysis
   parameter lives in `config.py`. Notebook variables are operating controls
   such as refresh, overwrite confirmation, lookup ticker, and display horizon;
   they do not define the model.

These are correctness rules, not coding preferences. Breaking them can create
look-ahead bias, double-lag accounting data, or inconsistent signal ranks.

## If you change X, rerun Y

| If you change... | Rerun... | What changes |
|---|---|---|
| WRDS columns, query filters, or sample dates | Notebook 01 with WRDS refresh enabled; then 03 and 04; run 02 before issuing a new live proposal | Source files, panel, signal evidence, backtest, and live target |
| Universe, Russell reconstitution, company-link, reporting-lag, or staleness rules | Notebook 01 from the panel-build step; run full notebook if source fields changed; then 03 and 04, plus 02 for a new live proposal | Configured panel and every downstream result |
| Live-price dates or quarantine rule | Notebook 02 from the beginning | Live prices, panel, target, and trades |
| Price, market-cap, or liquidity thresholds | Notebook 01 or 02, then notebooks 03 and 04 if evaluating impact | Investability, signal sample, and portfolio |
| Signal definition, window, sign, exclusion, or shared standardization | Notebook 03, then 04; if accepted into `MODEL_BLOCKS`, run 02 for a new live proposal | Signal research, composite, backtest, target, and trades; panel stays unchanged unless inputs changed |
| Model blocks, composite start, or composite weighting | Notebook 04, then notebook 02 for a new live proposal | Composite diagnostics, portfolio research, target, and trades |
| Buy/sell thresholds, holdings band, weighting, position floor, caps, beta band, or costs | Notebook 04, then notebook 02 for a new live proposal | Backtest, analysis, target, and trades |
| `data/current_portfolio.csv` only | Notebook 02 from the rebalance step | Target and trades versus actual holdings |
| Notebook prose or presentation only | That notebook | Display only; data and model outputs should not change |

## Troubleshooting

The full table is in `START_HERE.ipynb`; every workflow notebook also ends with
a short version.

- **WRDS login fails or times out:** confirm the username and WRDS website
  access. Retry once, then contact the WRDS account administrator. Code changes
  do not fix missing permissions.
- **A required file is missing:** the named prerequisite has not run. Run the
  earlier numbered notebook or earlier step that produces it.
- **The live-price staleness guard raises:** refresh live prices in notebook 02.
  Do not increase the limit merely to make the panel build.
- **A panel check fails:** stop. Save the complete error and check table and
  investigate the failing check. Never bypass a blocking check in a notebook.
- **A notebook was run out of order:** restart the kernel and run all cells from
  the top. If the file itself is missing, run the earlier numbered notebook.
Questions about the model and notebooks can go to the repository owner,
Alexander Villax. WRDS access and entitlement problems go to your institution's
WRDS administrator or WRDS support. Include the notebook, step name, and
complete error message when asking for help.

## Current model

The composite equally weights five standardized blocks:

- value;
- quality;
- capital deployment;
- 12-1 price momentum; and
- short-interest sentiment.

Risk and fundamental momentum were tested and excluded. The low-risk block had
inverted quintiles and a negative spread. Fundamental momentum decayed toward
zero as its predeclared lookback lengthened. Their modules remain in the project
as a research record, not as live model inputs.

The portfolio buys and retains the top decile, with one identical 10% entry and
retention threshold rather than a separate hysteresis rule. The latest live
selection has about 267 names before weighting; the 240–280 guardrail applies
at that stage. Benchmark-proportional positions below 0.02% are then removed,
leaving 160 names in the September 2026 target and about 190 on average in the
backtest. Names are capped at 3%, sector deviation is limited to ten percentage
points, and beta is constrained to 0.95–1.10.

Read [`docs/methodology.md`](docs/methodology.md) for the complete method,
results, and limitations. Read [`docs/signal_log.md`](docs/signal_log.md) for
every signal tested and the keep, combine, or reject decision. The
[`research-panel data dictionary`](docs/data_dictionary.csv) defines every
column in the configured final panel, including units and point-in-time traps.

## Project structure

```text
config.py                              All settings and paths
README.md                              This operating overview
notebooks/
  START_HERE.ipynb                     Orientation; no code
  01_build_data.ipynb                  Historical data and panel build
  02_monthly_update.ipynb              Live monthly workflow and checkpoints
  03_signal_research.ipynb             Signal tests and research
  04_backtest.ipynb                    Composite, portfolio, and analysis
  05_stock_lookup.ipynb                Interactive ticker lookup
src/
  native_core.py                       Builds and loads the C++ weight solver
  notebook_support.py                  Shared notebook confirmations and displays
  signal_evaluation.py                 Generic and block-specific research evaluation
  utils.py                             Neutral calendar-return alignment helpers
  data/                                Pulls, joins, and panel construction
  signals/
    registry.py                        One registry for every block and signal
    _template.py                       Copyable starting point for a new block
    beta.py                            Live trailing beta estimator
    archive/                           Tested and rejected signal blocks
    *.py                               Live signal definitions and standardization
  model/
    composite.py                       Block combination and composite research
    portfolio.py                       Historical construction and backtest
    rebalance.py                       One-date target and trade-list generation
    portfolio_summary.py               Current target meeting view
    portfolio_analysis.py              Attribution and stability analysis
    stock_explainer.py                 Ticker-level rankings and history
    portfolio_sensitivity.py           Historical weighting/beta sensitivity runner
    name_cap_sensitivity.py            Historical name-cap sensitivity runner
docs/
  data_dictionary.csv                   Every final research-panel column
  methodology.md                       Model method, results, and limitations
  signal_log.md                        Complete signal research record
native/
  aris_core.cpp                        C++ name/sector/beta weight solver
  build/                               Compiled library; Git-ignored
tests/
  test_native_parity.py                C++ versus Python bitwise parity tests
data/                                  Generated local files; Git-ignored
```

## Adding a signal

Follow every step below; registration makes discovery consistent, but it cannot
create missing data or document the research decision for you.

1. Define the economic idea, sign, exclusions, data needs, and expected failure
   modes before looking at results.
2. Copy `src/signals/_template.py` to `src/signals/<name>.py`. Rename every
   example symbol, keep the row-as-is point-in-time contract, and call the shared
   standardization functions rather than copying their implementation.
3. Check every required panel field. If one does not exist, add it to the
   appropriate pull-column configuration (and source query or panel merge if
   necessary), then run notebook 01 before evaluating the signal.
4. Add one `SignalBlockSpec` to `src/signals/registry.py`, including its builder,
   constituent functions, and required panel columns. Set `live=False` while it
   is only a research candidate; `live=True` means the registry permits it in
   `MODEL_BLOCKS`. `display_as_percent=True` affects raw-value presentation in
   the stock explainer only—it does not rescale or change the signal.
5. In notebook 03's contributor-extension cell, set
   `REGISTERED_BLOCK_TO_EVALUATE = "<name>"`. It calls
   `evaluate.run_registered_block_research("<name>")`; this generic path runs
   equal- and value-weighted spreads, size splits, IC, coverage, turnover, and a
   spread chart for every registered constituent and the block. The seven older
   `run_*_research()` functions remain because they include block-specific
   sensitivity and correlation studies; a new block does not need a custom
   runner unless its predeclared research design requires extra tests.
6. Run the random-signal and perfect-foresight harness before interpreting the
   new evaluation. Record the definition, evidence, result, and keep/reject
   decision in `docs/signal_log.md`, including failed results.
7. If accepted, set the registry entry `live=True`, add the name to
   `config.MODEL_BLOCKS`, rerun notebooks 04 and 02, and update
   `docs/methodology.md`. Current portfolio-summary labels also come from the
   registry, so no second label map needs editing.
8. If rejected, keep the record in `docs/signal_log.md`; move the module under
   `src/signals/archive/` when it is useful to preserve as a reproducible failed
   experiment, and mark the registry entry `live=False, archived=True`.

Notebook outputs must not be committed. They can expose local paths and user
names, and displayed results become stale when configuration changes. The
development requirements include `nbstripout`; run `nbstripout` on changed
notebooks before committing them.

The research bar is an absolute IC t-statistic of 3, together with coherent
equal- and value-weighted spreads, size-group evidence, economic reasoning, and
diversification against existing blocks. Failed tests stay in the research
record.

## C++ weight solver

Portfolio construction is split between Python and C++. Python handles data,
signals, ranking, the backtest, and the notebooks. The inner weight solver,
which fits each month's name caps, sector bands, and beta band, runs in C++
(`native/aris_core.cpp`) because it accounted for about 97% of construction
time.

- **No setup step.** The first portfolio run compiles the library into
  `native/build/` with the system C++ compiler (Xcode Command Line Tools on
  macOS, `g++` or `clang++` on Linux).
- **Identical results.** The C++ code reproduces numpy's and pandas' summation
  order, so weights, diagnostics, and returns match the Python reference bit for
  bit. `tests/test_native_parity.py` checks this.
- **Automatic fallback.** If no compiler is available, or `ARIS500_NATIVE=0` is
  set, the same Python reference in `src/model/portfolio.py` runs instead.
- **Speed.** On a synthetic 3,000-stock, 2007–2025 panel, construction took
  2.5 seconds instead of 48.5 seconds, about 20 times faster. The gain is largest
  in the sensitivity runners, which rebuild the portfolio many times.

Run the parity tests after changing either implementation:

```bash
python3 tests/test_native_parity.py
```

## Optional command-line entry points for maintainers

The notebooks are the supported interface. Maintainers can run the
same functions from the project root:

```bash
python3 -m src.data.pull_prices
python3 -m src.data.pull_live_prices --refresh-from-wrds
python3 -m src.data.pull_factors
python3 -m src.data.pull_fundamentals
python3 -m src.data.build_panel
python3 -m src.model.portfolio
python3 -m src.model.portfolio_analysis
python3 -m src.model.rebalance
python3 -m src.model.portfolio_summary
```

These commands call the same modules used by the notebooks. There is no separate
notebook version of the model.
