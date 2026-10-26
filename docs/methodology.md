# Model methodology

## What the model does

ARIS500 is a systematic US equity model to support stock selection inside a
long-only portfolio. The model ranks companies on valuation, business quality,
capital deployment, price momentum, and short-interest sentiment, then builds a
diversified portfolio from the highest-ranked investable names. It is an
overlay for investment judgment, not a claim that accounting data can replace
fundamental work. Its purpose is to apply the same evidence and risk controls to
every company, make trade-offs visible, and give a disciplined starting point
for judgment.

## Universe and benchmark

The research universe is a proxy for the Russell 3000. Actual historical
Russell constituent data were not available on the WRDS subscription, so the
model reconstructs membership from CRSP securities. At each June month-end it
ranks eligible US common stocks by market capitalization and selects the largest
3,000. That membership applies from July through the following June. A company
keeps its history before entry and after exit so new members have enough signal
history and departed firms are not erased from the backtest.

This is a proxy, not the Russell 3000 itself. It uses full market capitalization
rather than Russell's complete float-adjusted eligibility process and cannot
reproduce every corporate-action or membership judgment. Over the overlapping
sample, the proxy's capitalization-weighted return had 0.83% annualized tracking
error against the real Russell 3000 total return. That is close enough for
research, but not exact enough to label the reconstructed membership as official
Russell data. Portfolio performance is compared with the real Russell 3000 total
return series, not with the proxy return.

The membership set is broader than the portfolio's investable set. The portfolio
keeps one primary security per company and requires a price of at least $5,
market capitalization of at least $50 million, and three-month average daily
dollar volume of at least $500,000. A security with a quarantined live return is
also temporarily non-investable. These rules prevent an apparent small-stock
effect from depending on securities that could not be traded responsibly.

## Data and point-in-time treatment

### Sources

- CRSP monthly stock data provide security prices, total returns, shares,
  volume, identifiers, exchange history, and historical S&P 500 membership.
- CRSP/Compustat links connect each security to the correct company only during
  the dates when that link was valid.
- Compustat annual and quarterly statements provide accounting data in US
  dollars, subject to the industrial, standardized, domestic, and consolidated
  reporting filters.
- Compustat's historical GICS table supplies the sector classification effective
  on each date.
- Compustat short-interest history supplies security-level reported short shares.
- WRDS factor data provide the market, size, value, profitability, investment,
  momentum, and risk-free returns. Compustat index history supplies the actual
  Russell 3000 total return.
- Compustat Security Daily extends prices beyond the latest annual CRSP release.
  Those rows are explicitly marked provisional.

The final panel is monthly, with one row per security and trading month-end.
Market capitalization and accounting values are in US dollar millions, shares
outstanding are in millions of shares, price is dollars per share, and volume is
individual shares.

### When information becomes available

Fiscal period-end dates are not treated as publication dates. Quarterly data
become available on the reported earnings date. If that date is missing or
impossibly precedes the fiscal period end, the model uses three months after the
period end. A genuinely late report keeps its actual late date; it is never
pulled forward merely because the delay looks unusual. Annual data become
available six months after fiscal year-end.

The actual availability day is compared directly with the CRSP trading
month-end. It is not rounded to calendar month-end, which would unnecessarily
delay a report when a month ends on a weekend. Each stock-month receives the
latest report that was public by that date. Quarterly and annual statements are
attached separately and are never blended. Accounting observations older than
18 months are removed rather than carried forward indefinitely.

Trailing accounting measures use only consecutive fiscal quarters whose own
availability dates are no later than the stock-month being scored. Missing
quarters break the history. Signal calculations do not add another reporting
lag because the panel has already imposed the point-in-time rule.

Short interest is handled in the same direction: each month receives the latest
security-level report dated on or before month-end. Reports after month-end are
never used. Historical GICS sectors are joined over their effective date ranges,
so today's sector classification is not applied backward.

### Returns, delistings, and live prices

CRSP's monthly CIZ return already includes recorded delisting returns. Adding a
second delisting return would double count them. The exception is a terminal row
that CRSP marks “Missing — Return Excluded.” The model compounds a -100% outcome
only into those rows and flags every such return as imputed.

CRSP on this subscription currently ends in December 2025 and updates annually.
For later months, daily Compustat security prices are adjusted for splits and
dividends and reduced to the last priced day of each calendar month. Returns
beyond plus or minus 50% are quarantined only after the normal liquidity screen;
the row remains in the data, but a quarantined security cannot be selected until
the event is reviewed. CRSP takes precedence whenever the two sources overlap.

## How signals are made comparable

Each raw signal is calculated from information available on the row's date. It
does not see next month's return or shift already point-in-time fundamentals.
Within each month, only investable names are standardized:

1. Extreme observations are limited at the 1st and 99th percentiles.
2. The signal receives a z-score relative to the full investable universe.
3. It receives a second z-score relative to its GICS sector. A sector with
   fewer than ten valid companies falls back to the universe z-score.
4. The final signal is 70% universe z-score and 30% sector z-score, then
   standardized again to mean zero and standard deviation one.

The 70/30 split was taken from prior practice, not selected by trying different
weights in this sample. It leaves most of the cross-industry information intact
while reducing the chance that the portfolio is simply making sector bets.

## The five model blocks

### Value

Value asks how much operating cash flow, earnings, and EBITDA an investor
receives for the price paid. The block combines cash-flow yield, earnings yield,
and EBITDA to enterprise value. Yields are used instead of their inverse so a
company with negative fundamentals ranks as expensive rather than disappearing.
EBITDA to enterprise value is omitted for Financials and Real Estate because
debt is operating raw material in those sectors; the remaining value signals
still score those companies.

### Quality

Quality asks whether the business earns attractive returns on the capital it
uses and whether operating margins are stable. It combines a returns-on-capital
input, return on equity, and EBIT-margin stability. Cash flow on invested
capital and EBITDA return on invested capital are first combined into one input
because they correlate 0.86; treating them separately would double-weight one
idea. Asset-turnover stability and EPS stability were tested and removed. The
former had the wrong sign and the latter had no signal.

### Capital deployment

Capital deployment measures whether management is shrinking or expanding the
share count over three, twelve, and thirty-six months. Repurchases score well;
issuance scores poorly. Share counts are adjusted for splits before calculating
changes, because an untreated stock split would otherwise look like extreme
issuance. The three horizons are intentionally retained to distinguish recent
actions from persistent policy.

### Momentum

Momentum measures the compounded stock return over the eleven months ending one
month before formation. It skips the current month and requires a complete,
consecutive history. The block captures persistence in market information that
is not contained in the accounting statements. Its standalone result was weak
in this sample, but it has low correlation with the four more fundamental
blocks and rests on evidence broader than this backtest.

### Sentiment

Sentiment measures short interest as a share of shares outstanding. Lower short
interest scores better. Both numerator and denominator are adjusted consistently
for splits, and a missing short-interest report remains missing rather than
being interpreted as zero. This is currently a one-signal block; independent
analyst-revision or dispersion data could broaden it later.

The complete component-level evidence, including the rejected risk and
fundamental-momentum blocks, is recorded in the [signal research log](signal_log.md).

## Composite score

The five blocks receive equal 20% weights. If a block is unavailable for one
company, the remaining weights are rescaled rather than treating missing data as
a neutral score. The combined score is then standardized once more within the
month.

The composite has zero coefficients fitted to realized returns. Its equal
weights, 70/30 standardization split, and signal directions are explicit rules,
not outputs from an optimizer. This matters because optimized weights can make a
historical portfolio look precise while encoding noise from one sample. Equal
weighting is not claimed to be mathematically optimal. It is used because it is
auditable, stable, and honest about how limited the evidence is.

Inverse-volatility and correlation-group weights were tested only for structural
diagnosis. They were not adopted because one happened to show a better backtest
Sharpe ratio. Changing the production weights requires evidence of a structural
imbalance, not a higher historical return.

## Portfolio construction

The backtest begins in February 2007, after short-interest history becomes
usable, and rebalances monthly.

1. Investable names are ranked by composite score and also reported in deciles.
2. A position is bought and retained only while it ranks in the top 10%. Entry
   and retention use the same threshold, so there is no fitted hysteresis
   parameter.
3. The latest live top-decile selection has about 267 names before weighting.
   Over the historical backtest, the pre-weighting selection averaged 259 names,
   with a range of 203 to 283. The guardrail is centered at 260 and permits 240
   to 280 names at this pre-weighting stage. It
   intervened in 14 of 227 realized backtest months, or 6.2%, so it protects
   against unusual counts rather than normally selecting the portfolio.
   Paired tests found no measurable active-return difference among the feasible
   entry and exit rules: Newey-West t-statistics ranged from -0.64 to 1.20 and
   every twelve-month block-bootstrap confidence interval crossed zero. The
   common-threshold rule was therefore chosen for having no tuned parameters,
   not for its backtest return. The former hard-275 rule admitted 612 names on
   average before truncation, making its cap the selector rather than a guardrail.
4. Starting weights are proportional to each selected company's weight in the
   full benchmark universe, then normalized. This keeps the main bet focused on
   stock selection rather than an accidental small-cap tilt.
5. Positions below 0.02% are removed after initial weighting and the remaining
   weights are redistributed. Benchmark-proportional weighting otherwise creates
   positions too small to matter operationally. This floor leaves about 190 names
   on average in the backtest—189.6 before rounding—and 159 names in the
   September 2026 live target. It is intentionally applied after the 240–280
   pre-weighting guardrail, so that range is not the final holdings count.
6. No company may exceed 3%. The 3% cap materially reduced top-ten
   concentration while keeping the information ratio within 0.05 of the 4%
   baseline. A 2.5% cap failed that predeclared trade-off.
7. A sector may deviate by no more than ten percentage points from its benchmark
   weight.
8. Weighted trailing market beta must remain between 0.95 and 1.10. The
   constraint was adopted because it lowered tracking error and volatility while
   improving the information ratio in the predeclared test.
9. The portfolio is long only, fully invested, and unlevered. A security whose
   current row is known to be terminal is not selected for the following month.

Backtest costs are 10 basis points each way for the largest 500 stocks, 20 for
the next 1,000, and 40 for the rest. These are estimates, not executed costs.

## Backtest results

The configured backtest covers 227 monthly returns from February 2007 through
December 2025. Results are hypothetical and include no management fee or tax.

| Measure | Gross strategy | Net strategy | Russell 3000 |
|---|---:|---:|---:|
| Annualized return | 14.20% | 13.27% | 10.45% |
| Annualized excess return | 3.75% | 2.82% | — |
| Annualized volatility | 16.76% | 16.74% | 15.97% |
| Sharpe ratio | 0.80 | 0.75 | 0.62 |
| Tracking error | 5.16% | 5.15% | — |
| Information ratio | 0.68 | 0.52 | — |
| Average active share | 81.54% | 81.54% | — |
| Average monthly drift-adjusted turnover | 27.64% | 27.64% | — |
| Average monthly name turnover | 24.56% | 24.56% | — |

Modeled trading costs were 0.82% a year before compounding effects. The net
maximum drawdown was 47.3%. These figures describe the historical simulation;
they do not establish that the model will earn the same return in live trading.

## Known limitations

### The excess return is concentrated

Six of the 227 months produced half of the cumulative log excess return. The
result is not a steady stream of small wins. A handful of months determines a
large share of the advantage, so missing those episodes or changing the
portfolio around them could materially alter the outcome.

### The estimated alpha is not statistically significant

Annualized active-return factor-model alpha is 2.05% with a Newey-West
t-statistic of 1.91.
That does not clear a conventional significance test and is far below the
research bar of 3 used for signal selection. The backtest does not prove
persistent alpha.

### Factor tilts explain a material share of active return

At sample-average factor returns, conventional factor tilts contribute 0.61% a
year, or 22.9% of the 2.66% arithmetic annual active return attributed by the
regression. The regression explains 15.3% of monthly active-return variation.
Momentum is the only factor loading with an absolute Newey-West t-statistic above 2. That
factor contribution is not unique stock-selection alpha and could be purchased
more cheaply in a transparent factor product. The unexplained remainder is not
automatically alpha; it also contains omitted exposures and noise.

### The portfolio is concentrated in its largest positions and technology

The top ten holdings average 29.4% of the portfolio and reach the 30.0% maximum.
Technology frequently sits at the permitted sector cap. The 3%
name limit and ten-percentage-point sector limit reduce concentration but do not
eliminate it. Results remain meaningfully dependent on a small group of large
technology companies.

### The beta constraint is doing substantial work

The beta constraint binds in 67% of backtest months. Unconstrained stock
selection often produces more market exposure than intended. The reported risk
and return depend on repeated reweighting back toward the 0.95–1.10 band; this
is not a rarely used emergency control.

### Live prices are not CRSP prices

CRSP on this subscription updates annually. Months after December 2025 use
Compustat security prices and are marked provisional. The two sources agree in
most ordinary months, but splits, reorganizations, and delistings can create
large differences. Quarantine catches obvious extremes; it does not make the
live extension identical to CRSP.

### The accounting history is not a true vintage database

Compustat preserves report dates but can restate historical values. The model
delays each observation until it could have been public, yet it may still use a
later-restated version of that statement. The backtest is point-in-time with
respect to availability, not a perfect reconstruction of exactly what every
number looked like on the original day.

### New post-2025 IPOs cannot enter until CRSP refreshes

The live extension updates prices for securities already linked into the CRSP
panel. A company that first listed after the latest CRSP cutoff has no CRSP
security identifier in the base universe and cannot enter the Russell proxy or
portfolio until the next CRSP release. The live portfolio is therefore missing
new IPOs between annual refreshes.

### The benchmark universe remains an approximation

The 0.83% proxy tracking error is small but real. The reconstruction does not
fully reproduce float adjustments, official eligibility screens, intra-year
corporate-action decisions, or every share-class treatment. Universe mismatch
can affect measured signal returns as well as benchmark comparison.

### Trading costs are modeled, not observed

The cost schedule is a broad size-based estimate. Actual costs depend on order
size, volatility, spreads, market conditions, and implementation skill. The
portfolio has meaningful small- and mid-cap exposure, where live costs can be
higher than the assumptions. Net performance is therefore more uncertain than
the table's precision suggests.
