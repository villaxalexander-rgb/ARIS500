# Signal research log

This document records every signal tested for the Russell 3000 research model,
including the ideas that failed. Results are for the point-in-time research
sample through December 2025. A higher score is always intended to predict a
higher next-month return.

Mean IC is the average monthly Spearman rank correlation between a signal and
the next month's return. The IC t-statistic tests whether that monthly average
differs from zero. Spread Sharpe is the annualized Sharpe ratio of the top-minus-
bottom quintile return. Equal-weighted spreads give every stock the same weight;
value-weighted spreads use market capitalization at portfolio formation.
Turnover is average one-way turnover in the top quintile. Size results report
mean IC and equal-weighted spread Sharpe for the largest 500, next 1,000, and
smallest 1,500 stocks each month.

## Summary of all tests

| Block | Signal or configuration | Mean IC | IC t-stat | EW spread Sharpe | VW spread Sharpe | Decision |
|---|---|---:|---:|---:|---:|---|
| Momentum | 12-1 price momentum | 0.0117 | 1.39 | 0.21 | 0.29 | Keep as a block |
| Value | Cash-flow yield | 0.0225 | 3.49 | 0.41 | 0.15 | Keep in value |
| Value | Earnings yield | 0.0211 | 3.55 | 0.12 | 0.27 | Keep in value |
| Value | EBITDA to enterprise value | 0.0208 | 2.71 | 0.16 | -0.03 | Keep in value |
| Value | Value block | 0.0241 | 3.61 | 0.30 | 0.09 | Keep in model |
| Quality | Cash flow on invested capital | 0.0266 | 5.62 | 0.65 | 0.75 | Combine with EBITDA ROIC |
| Quality | EBITDA return on invested capital | 0.0245 | 5.02 | 0.45 | 0.64 | Combine with CFO/IC |
| Quality | Return on equity | 0.0247 | 4.28 | 0.21 | 0.36 | Keep in quality |
| Quality | Asset-turnover stability | -0.0047 | -0.91 | -0.50 | -0.58 | Drop; wrong sign |
| Quality | EBIT-margin stability | 0.0180 | 4.27 | 0.14 | -0.03 | Keep in quality |
| Quality | EPS stability | -0.0033 | -0.73 | -0.08 | -0.19 | Drop; no signal |
| Quality | Original six-signal block | 0.0227 | 3.98 | 0.17 | 0.20 | Replace |
| Quality | Revised three-input block | 0.0282 | 5.03 | 0.35 | 0.43 | Keep in model |
| Capital deployment | Three-month share reduction | 0.0152 | 3.78 | 0.28 | 0.08 | Keep in block |
| Capital deployment | Twelve-month share reduction | 0.0220 | 4.56 | 0.49 | 0.21 | Keep in block |
| Capital deployment | Thirty-six-month share reduction | 0.0229 | 4.91 | 0.61 | 0.52 | Keep in block |
| Capital deployment | Capital-deployment block | 0.0249 | 4.86 | 0.51 | 0.42 | Keep in model |
| Risk | Low total volatility | 0.0241 | 2.38 | -0.05 | -0.30 | Reject with risk block |
| Risk | Low market beta | 0.0006 | 0.06 | -0.28 | -0.31 | Reject |
| Risk | Low idiosyncratic volatility | 0.0290 | 3.28 | 0.07 | -0.14 | Reject with risk block |
| Risk | Low-risk block | 0.0208 | 1.95 | -0.08 | -0.32 | Exclude from model |
| Sentiment | Low short interest | 0.0246 | 4.20 | 0.39 | 0.40 | Keep in model |
| Fundamental momentum | Cash-flow trend | 0.0114 | 3.30 | 0.39 | 0.58 | Record, but do not use alone |
| Fundamental momentum | Revenue trend | -0.0026 | -0.48 | -0.11 | 0.30 | Reject |
| Fundamental momentum | Twelve-quarter block | 0.0032 | 0.74 | 0.03 | 0.39 | Exclude from model |

The table reports the audited post-fix replication through 2025. The fixed
fundamental-momentum sensitivity run used for the original decision recorded
mean ICs of 0.0125, 0.0039, and 0.0022 at four, eight, and twelve quarters. The
audited replication is slightly higher at each window but reaches the same
decision: the effect decays as the horizon lengthens and is negligible at the
preselected twelve-quarter horizon.

## Decisions about the research method

### A higher score always means better

Every component is put into the same direction before ranking. High valuation
yields, profitability, improving fundamentals, past returns, share-count
reduction, stability, low risk, and low short interest are all expressed as
positive scores. A failed sign is not reversed after seeing the result. Asset-
turnover stability and the low-risk block therefore remain failed tests rather
than being relabeled as pro-cyclical signals.

### The universe and sector blend is 70/30

Each signal is winsorized and converted to a z-score within the investable
universe each month. It is also converted to a z-score within its GICS sector.
The final signal is 70% universe z-score and 30% sector-relative z-score, then
standardized again. The 70/30
split came from prior practice, not a search over this backtest. It preserves
most of the economic information that exists across sectors while reducing the
chance that a signal is only an industry allocation.

### The evidence bar is a t-statistic of 3

The research bar is an absolute IC t-statistic of 3, not the conventional 2.
Many related signals and design choices were examined, and a stricter bar helps
offset false discoveries. Economic logic, consistency across weighting methods
and size groups, and correlation with existing blocks still matter; the bar is
not an automatic inclusion rule. Momentum remains despite a lower standalone
t-statistic because it is a long-established effect and diversifies the four
more correlated fundamental blocks.

### Blocks are equally weighted

The five accepted blocks receive equal weight. The weights were not optimized
on realized returns. Inverse-volatility and correlation-group alternatives were
tested as structural diagnostics, not as a menu from which to select the best
Sharpe ratio. Equal weights are transparent, stable, and avoid converting a
stock-ranking model into an in-sample allocation exercise.

### Fundamental-momentum horizons were a robustness test

Four, eight, and twelve quarters were specified to ask whether the same idea
survived reasonable horizons. They were not candidates in a parameter search.
The model would have remained at twelve quarters because that horizon matched
the quality block's three-year stability window. The falling IC across horizons
was evidence against the block; the strong four-quarter result was not selected.

## Momentum

### Twelve-minus-one price momentum

**Definition and sign.** The signal compounds the eleven monthly returns ending
one month before portfolio formation. The current month is skipped. Higher past
return is better. All eleven returns must exist in consecutive calendar months.

**Sector treatment.** No sector is excluded. The standard 70/30 universe/sector
blend removes some, but not all, industry momentum.

**Results.** Mean IC was 0.0117 with a t-statistic of 1.39. Equal- and value-
weighted spread Sharpes were 0.21 and 0.29. Top-quintile turnover was 25.1% per
month. Size results were: largest 500, IC 0.0133 and Sharpe 0.24; next 1,000, IC
0.0078 and Sharpe 0.17; smallest 1,500, IC 0.0122 and Sharpe 0.18.

**Decision.** Keep. The result is positive but weak in this period and universe.
Momentum earns its place because its economic evidence predates this sample and
its returns are unusually independent of the fundamental blocks. It was not
kept because this backtest cleared the normal statistical bar—it did not.

## Value

### Cash-flow yield

**Definition and sign.** Trailing twelve-month operating cash flow divided by
current market capitalization. Higher yield is cheaper and therefore better.
Negative cash flow remains in the ranking as expensive rather than being
discarded.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0225 with a t-statistic of 3.49. Equal- and value-
weighted spread Sharpes were 0.41 and 0.15. Turnover was 11.8%. Size results:
largest 500, IC 0.0122 and Sharpe 0.22; next 1,000, IC 0.0175 and Sharpe 0.34;
smallest 1,500, IC 0.0276 and Sharpe 0.41.

**Decision.** Keep. It clears the IC evidence bar and works most strongly below
the largest stocks, although its value-weighted spread is modest.

### Earnings yield

**Definition and sign.** Trailing twelve-month income before extraordinary
items divided by market capitalization. Higher is cheaper and better. Losses
remain valid low-ranked observations.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0211 with a t-statistic of 3.55. Equal- and value-
weighted spread Sharpes were 0.12 and 0.27. Turnover was 12.4%. Size results:
largest 500, IC 0.0104 and Sharpe 0.06; next 1,000, IC 0.0148 and Sharpe 0.08;
smallest 1,500, IC 0.0234 and Sharpe 0.04.

**Decision.** Keep. The rank relationship is reliable even though equal-weighted
quintile separation is weak. It contributes a different accounting definition
of cheapness from cash flow and enterprise value.

### EBITDA to enterprise value

**Definition and sign.** Trailing twelve-month EBITDA divided by market value
plus short- and long-term debt, less cash and short-term investments. Higher is
cheaper and better. Observations with nonpositive enterprise value are missing
rather than assigned a nonsensical yield.

**Sector treatment.** Financials and Real Estate are excluded because debt is an
operating input in those businesses and EBITDA is not comparable.

**Results.** Mean IC was 0.0208 with a t-statistic of 2.71. Equal- and value-
weighted spread Sharpes were 0.16 and -0.03. Turnover was 12.2%. Size results:
largest 500, IC 0.0084 and Sharpe 0.03; next 1,000, IC 0.0128 and Sharpe 0.09;
smallest 1,500, IC 0.0280 and Sharpe 0.18.

**Decision.** Keep inside the block, not as a standalone signal. It has the
expected rank direction and broad small-stock support, but it misses the t-stat
bar and has no value-weighted spread on its own.

### Value block

**Definition and sign.** Equal average of the available standardized cash-flow,
earnings, and EBITDA yields. Missing EBITDA/EV in excluded sectors does not count
as zero; the remaining signals are reweighted.

**Sector treatment.** The EBITDA component excludes Financials and Real Estate;
the other two components cover them.

**Results.** Mean IC was 0.0241 with a t-statistic of 3.61. Equal- and value-
weighted spread Sharpes were 0.30 and 0.09. Turnover was 12.3%. Size results:
largest 500, IC 0.0114 and Sharpe 0.11; next 1,000, IC 0.0185 and Sharpe 0.23;
smallest 1,500, IC 0.0291 and Sharpe 0.20.

**Decision.** Keep as one of the five model blocks. Its rank IC is stronger than
any single component and passes the research bar, but its weak value-weighted
spread is recorded as an important limitation.

## Quality

### Cash flow on invested capital

**Definition and sign.** Trailing operating cash flow divided by common equity
plus long- and short-term debt, less cash and short-term investments. Higher is
better. Nonpositive invested capital is treated as missing.

**Sector treatment.** Financials and Real Estate are excluded because debt is
operating raw material rather than ordinary financing.

**Results.** Mean IC was 0.0266 with a t-statistic of 5.62. Equal- and value-
weighted spread Sharpes were 0.65 and 0.75. Turnover was 9.0%. Size results:
largest 500, IC 0.0208 and Sharpe 0.71; next 1,000, IC 0.0165 and Sharpe 0.51;
smallest 1,500, IC 0.0307 and Sharpe 0.53.

**Decision.** Keep, but combine it with EBITDA return on invested capital before
forming the quality block. The two signals correlate 0.86, so counting each
separately would give one profitability idea twice the intended weight.

### EBITDA return on invested capital

**Definition and sign.** Trailing EBITDA divided by the same invested-capital
measure. Higher is better; nonpositive invested capital is missing.

**Sector treatment.** Financials and Real Estate are excluded for the same
reason as cash flow on invested capital.

**Results.** Mean IC was 0.0245 with a t-statistic of 5.02. Equal- and value-
weighted spread Sharpes were 0.45 and 0.64. Turnover was 7.3%. Size results:
largest 500, IC 0.0181 and Sharpe 0.60; next 1,000, IC 0.0153 and Sharpe 0.31;
smallest 1,500, IC 0.0276 and Sharpe 0.51.

**Decision.** Keep through the combined returns-on-capital input. It is strong
across size groups but too redundant with cash flow on invested capital to
receive a separate block weight.

### Return on equity

**Definition and sign.** Trailing income before extraordinary items divided by
average common equity over the same period. Higher is better. Nonpositive
average equity is missing.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0247 with a t-statistic of 4.28. Equal- and value-
weighted spread Sharpes were 0.21 and 0.36. Turnover was 6.2%. Size results:
largest 500, IC 0.0162 and Sharpe 0.34; next 1,000, IC 0.0173 and Sharpe 0.22;
smallest 1,500, IC 0.0246 and Sharpe 0.09.

**Decision.** Keep. It clears the IC bar and covers Financials and Real Estate,
where the invested-capital signals are intentionally absent.

### Asset-turnover stability

**Definition and sign.** Negative three-year volatility of quarterly sales
scaled by average assets. The sign was reversed before testing so stable asset
turnover scored higher.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was -0.0047 with a t-statistic of -0.91. Equal- and value-
weighted spread Sharpes were -0.50 and -0.58. Turnover was 5.9%. Size results:
largest 500, IC -0.0087 and Sharpe -0.35; next 1,000, IC -0.0077 and Sharpe
-0.52; smallest 1,500, IC -0.0041 and Sharpe -0.43. At the original decision
point, its equal-weighted spread was about -0.34% per month with a t-statistic
of -2.24.

**Decision.** Drop from the block. The intended sign was wrong in every size
group. The signal remains documented and is not reversed after seeing the data.

### EBIT-margin stability

**Definition and sign.** Negative three-year volatility of quarterly operating
income after depreciation divided by sales. Lower volatility is intended to be
better, so the raw volatility is reversed. Quarters with nonpositive sales are
not used.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0180 with a t-statistic of 4.27. Equal- and value-
weighted spread Sharpes were 0.14 and -0.03. Turnover was 10.7%. Size results:
largest 500, IC 0.0090 and Sharpe -0.06; next 1,000, IC 0.0151 and Sharpe 0.14;
smallest 1,500, IC 0.0182 and Sharpe 0.18.

**Decision.** Keep. The rank relationship clears the evidence bar, although the
flat value-weighted spread means it should remain only one input to quality.

### EPS stability

**Definition and sign.** Negative three-year volatility of quarterly basic EPS
excluding extraordinary items. Lower volatility was intended to be better, so
volatility was reversed.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was -0.0033 with a t-statistic of -0.73. Equal- and value-
weighted spread Sharpes were -0.08 and -0.19. Turnover was 7.2%. Size results:
largest 500, IC 0.0012 and Sharpe 0.07; next 1,000, IC -0.0009 and Sharpe -0.05;
smallest 1,500, IC 0.0015 and Sharpe -0.17. The original decision record rounded
this to IC -0.0031 and t-statistic -0.69.

**Decision.** Drop from the block. There is no reliable rank or spread signal,
and retaining it would add noise merely because it fits the quality story.

### Original six-signal quality block

**Definition and sign.** Equal average of all six standardized quality signals,
including both highly correlated returns-on-capital measures and both failed
stability measures.

**Sector treatment.** Financials and Real Estate lack the two invested-capital
signals; all other available inputs are averaged.

**Results.** Mean IC was 0.0227 with a t-statistic of 3.98. Equal- and value-
weighted spread Sharpes were 0.17 and 0.20. Turnover was 7.1%. Size results:
largest 500, IC 0.0122 and Sharpe 0.27; next 1,000, IC 0.0122 and Sharpe -0.04;
smallest 1,500, IC 0.0287 and Sharpe 0.14.

**Decision.** Replace after one predeclared revision. It double-counted one
profitability idea and diluted the block with two failed signals.

### Revised three-input quality block

**Definition and sign.** Equal average of three inputs: a combined returns-on-
capital score, return on equity, and EBIT-margin stability. CFO/IC and EBITDA
ROIC are averaged and re-standardized before entering as one input.

**Sector treatment.** Financials and Real Estate use return on equity and EBIT-
margin stability but not the invested-capital pair.

**Results.** Mean IC was 0.0282 with a t-statistic of 5.03. Equal- and value-
weighted spread Sharpes were 0.35 and 0.43. Turnover was 6.8%. Size results:
largest 500, IC 0.0201 and Sharpe 0.42; next 1,000, IC 0.0186 and Sharpe 0.26;
smallest 1,500, IC 0.0309 and Sharpe 0.31.

**Decision.** Keep as one of the five model blocks. The one allowed revision
improved both the evidence and the economic balance. No further composition
search was performed.

## Capital deployment

All three capital-deployment signals use split-adjusted shares. The score is the
negative log change in shares, so repurchases score high and issuance scores
low. The three horizons are nested and intentionally remain separate: here the
same economic action over different horizons is the definition of the block,
not accidental duplication.

### Three-month share reduction

**Definition and sign.** Negative log change in split-adjusted shares over three
months. A shrinking share count scores high and issuance scores low.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0152 with a t-statistic of 3.78. Equal- and value-
weighted spread Sharpes were 0.28 and 0.08. Turnover was 25.3%. Size results:
largest 500, IC 0.0087 and Sharpe 0.12; next 1,000, IC 0.0151 and Sharpe 0.36;
smallest 1,500, IC 0.0092 and Sharpe 0.09.

**Decision.** Keep. It captures recent issuance and repurchase activity, though
its high turnover and weak value-weighted result limit its standalone appeal.

### Twelve-month share reduction

**Definition and sign.** Negative log change in split-adjusted shares over twelve
months. A shrinking share count scores high and issuance scores low.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0220 with a t-statistic of 4.56. Equal- and value-
weighted spread Sharpes were 0.49 and 0.21. Turnover was 9.7%. Size results:
largest 500, IC 0.0121 and Sharpe 0.03; next 1,000, IC 0.0226 and Sharpe 0.52;
smallest 1,500, IC 0.0174 and Sharpe 0.45.

**Decision.** Keep. It clears the evidence bar and is especially useful outside
the largest 500 stocks.

### Thirty-six-month share reduction

**Definition and sign.** Negative log change in split-adjusted shares over
thirty-six months. A shrinking share count scores high and issuance scores low.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0229 with a t-statistic of 4.91. Equal- and value-
weighted spread Sharpes were 0.61 and 0.52. Turnover was 5.2%. Size results:
largest 500, IC 0.0208 and Sharpe 0.43; next 1,000, IC 0.0241 and Sharpe 0.65;
smallest 1,500, IC 0.0161 and Sharpe 0.41.

**Decision.** Keep. It is the strongest and lowest-turnover horizon, with useful
results in every size group and both weighting methods.

### Capital-deployment block

**Definition and sign.** Equal average of the available three-, twelve-, and
thirty-six-month share-reduction scores.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0249 with a t-statistic of 4.86. Equal- and value-
weighted spread Sharpes were 0.51 and 0.42. Turnover was 12.6%. Size results:
largest 500, IC 0.0179 and Sharpe 0.27; next 1,000, IC 0.0256 and Sharpe 0.50;
smallest 1,500, IC 0.0190 and Sharpe 0.37.

**Decision.** Keep as one of the five model blocks. The block is statistically
credible, works under both weighting methods, and captures a corporate action
not represented by value or quality.

## Risk — tested and excluded

All risk measures use 24 to 36 consecutive monthly observations ending in the
current month. Each raw measure is reversed so lower risk scores higher. No
sector is excluded. This sign convention was specified before evaluation and
was not changed after high-risk stocks outperformed.

### Low total volatility

**Definition and sign.** Negative annualized standard deviation of monthly stock
returns.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0241 with a t-statistic of 2.38, but equal- and value-
weighted spread Sharpes were -0.05 and -0.30. Turnover was 5.3%. Size results:
largest 500, IC -0.0024 and Sharpe -0.21; next 1,000, IC 0.0182 and Sharpe 0.01;
smallest 1,500, IC 0.0277 and Sharpe -0.08.

**Decision.** Reject with the block. A positive average rank IC did not translate
into the intended top-minus-bottom payoff, particularly when value weighted.

### Low market beta

**Definition and sign.** Negative trailing beta from a stock excess-return
regression on the market excess return.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0006 with a t-statistic of 0.06. Equal- and value-
weighted spread Sharpes were -0.28 and -0.31. Turnover was 10.3%. Size results:
largest 500, IC -0.0136 and Sharpe -0.30; next 1,000, IC -0.0018 and Sharpe
-0.28; smallest 1,500, IC 0.0037 and Sharpe -0.28.

**Decision.** Reject. There is no rank signal, and the quintiles are inverted in
every size group.

### Low idiosyncratic volatility

**Definition and sign.** Negative annualized residual volatility from the same
market regression used to estimate beta.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0290 with a t-statistic of 3.28, but equal- and value-
weighted spread Sharpes were only 0.07 and -0.14. Turnover was 5.5%. Size
results: largest 500, IC 0.0054 and Sharpe -0.08; next 1,000, IC 0.0235 and
Sharpe 0.15; smallest 1,500, IC 0.0311 and Sharpe 0.02.

**Decision.** Reject with the block rather than rescue one favorable statistic.
The value-weighted payoff is negative and the size results are economically
weak despite the full-sample IC t-statistic.

### Low-risk block

**Definition and sign.** Equal average of the three low-risk scores.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0208 with a t-statistic of 1.95. Equal- and value-
weighted spread Sharpes were -0.08 and -0.32. Turnover was 6.2%. Size results:
largest 500, IC -0.0048 and Sharpe -0.19; next 1,000, IC 0.0150 and Sharpe
-0.07; smallest 1,500, IC 0.0243 and Sharpe -0.09.

**Decision.** Exclude from the model. The ranked quintiles were inverted: the
stocks labeled high risk outperformed. The file remains in the research record
so the failed test is not quietly repeated or mistaken for a live block.

## Sentiment

### Low short interest

**Definition and sign.** Reported short-interest shares divided by shares
outstanding, with both sides adjusted consistently for splits. The ratio is
reversed before ranking, so lower short interest scores higher. The latest
report available by month-end is used; missing reports are not treated as zero.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0246 with a t-statistic of 4.20. Equal- and value-
weighted spread Sharpes were 0.39 and 0.40. Turnover was 22.9%. Size results:
largest 500, IC 0.0196 and Sharpe 0.48; next 1,000, IC 0.0234 and Sharpe 0.49;
smallest 1,500, IC 0.0243 and Sharpe 0.38.

**Decision.** Keep as the sentiment block. It clears the evidence bar, works
under both weighting methods and in all size groups, and adds information not
contained in accounting statements. It is a one-signal block until independent
analyst-revision or dispersion data become available.

## Fundamental momentum — tested and excluded

### Cash-flow trend

**Definition and sign.** The slope of quarterly operating cash flow over the
trailing twelve quarters, divided by the mean absolute cash-flow level. Higher
improvement is better. The scaling prevents large companies from dominating,
and the history resets after a missing fiscal quarter.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was 0.0114 with a t-statistic of 3.30. Equal- and value-
weighted spread Sharpes were 0.39 and 0.58. Turnover was 16.0%. Size results:
largest 500, IC 0.0142 and Sharpe 0.50; next 1,000, IC 0.0055 and Sharpe 0.22;
smallest 1,500, IC 0.0132 and Sharpe 0.30.

**Decision.** Record as promising but do not promote it alone after its paired
block failed the predeclared robustness test. Selecting only this component
after seeing the revenue result would create another layer of model selection.

### Revenue trend

**Definition and sign.** The slope of quarterly revenue over the trailing
twelve quarters, divided by the mean absolute revenue level. Higher improvement
is better, and histories reset after missing quarters.

**Sector treatment.** No sector is excluded.

**Results.** Mean IC was -0.0026 with a t-statistic of -0.48. Equal- and value-
weighted spread Sharpes were -0.11 and 0.30. Turnover was 8.3%. Size results:
largest 500, IC 0.0003 and Sharpe -0.01; next 1,000, IC -0.0041 and Sharpe
-0.10; smallest 1,500, IC -0.0024 and Sharpe -0.18.

**Decision.** Reject. Its rank and equal-weighted spread are negative, with no
support across size groups.

### Fundamental-momentum block

**Definition and sign.** Equal average of the available cash-flow and revenue
trend scores. Improving fundamentals score higher.

**Sector treatment.** No sector is excluded.

**Results.** In the audited twelve-quarter replication, mean IC was 0.0032 with
a t-statistic of 0.74. Equal- and value-weighted spread Sharpes were 0.03 and
0.39. Turnover was 12.5%. Size results: largest 500, IC 0.0082 and Sharpe 0.25;
next 1,000, IC -0.0006 and Sharpe 0.01; smallest 1,500, IC 0.0042 and Sharpe
-0.05.

The predeclared sensitivity record was:

| Lookback | Recorded mean IC | Audited IC t-stat | Audited EW Sharpe | Audited VW Sharpe | Audited turnover |
|---:|---:|---:|---:|---:|---:|
| 4 quarters | 0.0125 | 4.32 | 0.53 | 0.74 | 26.6% |
| 8 quarters | 0.0039 | 1.30 | 0.12 | 0.43 | 17.4% |
| 12 quarters | 0.0022 | 0.74 | 0.03 | 0.39 | 12.5% |

The audited size result for the four-quarter version was broad, but by eight
quarters the next 1,000 and smallest 1,500 groups were essentially flat. At
twelve quarters only the largest 500 retained a modest positive result.

**Decision.** Exclude from the model. The signal decayed toward zero as the
lookback lengthened and failed at the fixed twelve-quarter horizon. Choosing
four quarters because it produced the best backtest would turn a robustness
test into an undisclosed parameter search.

## Portfolio selection-threshold research

This is a portfolio-construction test rather than a new signal, but it is
recorded here because it used the same predeclared principle: examine robustness
without selecting the best historical return. All variants used the unchanged
five-block composite, benchmark weighting, 3% name cap, sector cap, beta band,
and modeled trading costs. These tests predated the 0.02% minimum-position
floor, so the return figures below are a historical decision record rather than
the current production backtest. “Natural names” is the pre-weighting average
before any holdings guardrail or minimum-position floor; turnover excludes the
initial formation month.

| Buy / sell rule | Natural names | Drift-adjusted turnover | Net return | Net Sharpe | Tracking error | Information ratio |
|---|---:|---:|---:|---:|---:|---:|
| Top 10% / top 10%, common threshold | 259 | 27.27% | 12.99% | 0.73 | 5.19% | 0.47 |
| Top 6% / top 8% | 175 | 24.22% | 12.09% | 0.67 | 5.82% | 0.30 |
| Top 6% / top 10% | 193 | 19.12% | 12.77% | 0.71 | 5.71% | 0.40 |
| Top 6% / top 12% | 209 | 15.84% | 12.87% | 0.72 | 5.30% | 0.44 |
| Top 6% / top 20% | 265 | 9.58% | 12.25% | 0.70 | 4.58% | 0.38 |
| Top 10% / hard 275 | 612 before truncation | 22.45% | 13.16% | 0.74 | 4.88% | 0.53 |

The top-6% / top-6% variant could not satisfy the production name, sector, and
beta constraints in every month, so it was not a feasible production rule.
Against the common-threshold top-decile baseline, paired monthly active-return
tests for every feasible alternative produced Newey-West t-statistics between
-0.64 and 1.20. Every twelve-month block-bootstrap 95% confidence interval
crossed zero. None of the apparent return differences was distinguishable from
noise.

**Decision.** Use top 10% for both purchase and retention. This removes the
entry/exit gap rather than tuning it after seeing returns. The historical
pre-weighting count averaged 259 names (203 to 283); the latest live selection
has about 267. A 240–280 guardrail is applied only to this pre-weighting count
and bound in 14 of 227 historical months, or 6.2%. The later 0.02% position
floor reduces the final portfolio to about 190 names on average in the current
backtest and 159 in the September 2026 target. The old hard-275 implementation
naturally admitted 612 names on average before truncation, so the cap—not the
score rule—was selecting the portfolio. The current band is explicitly a
guardrail, not a selector.
