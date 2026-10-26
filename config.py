"""Define every user-adjustable setting for the research and portfolio pipeline.

All modules import this file; it reads only environment variables and writes
nothing. Paths are anchored to the project root, and changing a setting can
alter cached queries or model outputs, so parameters include the reason for
their current production value wherever that choice is empirical.
"""

import getpass
import os
from pathlib import Path


# SAMPLE PERIOD

# Start three years before signal evaluation so momentum and risk histories are
# seasoned when the 2005 research sample begins. Moving this date changes pulls.
START_DATE = "2003-01-01"

# Stop all pulls at this date; None requests the latest observation available.
END_DATE = None


# UNIVERSE AND INVESTABILITY

# Select the panel membership rule; pulls remain wide so switching does not
# require another WRDS extraction. Valid values: sp500, all, russell3000.
UNIVERSE = "russell3000"

# Hold the largest 3,000 eligible securities at each Russell proxy reconstitution.
RUSSELL_N = 3000

# Reconstitute in June and apply membership from July through the next June.
RUSSELL_RECON_MONTH = 6

# Exclude sub-$5 shares from investable portfolios while retaining their rows.
MIN_PRICE = 5.0

# Exclude companies below $50 million market cap; panel market cap is USD millions.
MIN_MARKET_CAP = 50.0

# Require $500,000 of three-month average daily dollar volume for investability.
MIN_ADV_USD = 500_000


# DELISTING RETURNS

# Compound a total loss only where CIZ marks the terminal return as excluded
# (mthdelflg='M'); ordinary CIZ mthret already contains recorded delisting returns.
DELISTING_ASSUMPTION = -1.0

# Treat return comparisons within this absolute tolerance as numerically equal.
RETURN_TOLERANCE = 2e-6


# LIVE PRICE EXTENSION

# Append provisional Compustat Security Daily months after the CRSP cutoff.
USE_LIVE_EXTENSION = True

# Pull two overlap months so the first live monthly return has a prior endpoint.
LIVE_PRICES_START_DATE = "2025-11-01"

# End live pulls here; None means the current date.
LIVE_PRICES_END_DATE = None

# Flag absolute live returns above 50% after liquidity screening for manual review.
LIVE_RETURN_QUARANTINE = 0.50

# Exclude a security-month when its last price trails the market's final session
# by more than five observed trading days; this catches live-period delistings.
LIVE_PRICE_MAX_GAP_DAYS = 5

# Refuse to build the panel on a live-price extension older than this; a stale
# file usually means pull_live_prices was skipped, not that nothing changed.
LIVE_PRICE_MAX_STALENESS_DAYS = 45

# The monthly operating notebook is stricter than the panel builder: stop a
# rebalance when the freshly queried latest price is over seven calendar days old.
LIVE_PRICE_REBALANCE_MAX_AGE_DAYS = 7


# POINT-IN-TIME FUNDAMENTAL MERGE

# Make annual statements available six months after fiscal year end.
ANNUAL_LAG_MONTHS = 6

# Use this fallback only when quarterly rdq is missing or before datadate; valid
# late reports retain their actual rdq and are never pulled forward.
QUARTERLY_MAX_LAG_MONTHS = 3

# Clear attached fundamentals more than 18 months old rather than carry forever.
MAX_STALENESS_MONTHS = 18


# CRSP/COMPUSTAT LINKING

# Accept researched CCM security links; LS is needed for valid secondary issues.
ACCEPTED_LINK_TYPES = ("LC", "LU", "LS")

# Retain primary/current and join-only link markers so share classes stay visible.
ACCEPTED_LINKPRIM_VALUES = ("P", "C", "J")

# Prefer P/C links when choosing the one investable security per company-date.
PRIMARY_LINKPRIM_VALUES = ("P", "C")


# DATA VALIDATION THRESHOLDS

# Require near-perfect correlation in reconstructed benchmark-return checks.
BENCHMARK_MIN_CORRELATION = 0.995

# Flag annualized benchmark tracking error above 1%.
BENCHMARK_MAX_TRACKING_ERROR = 0.01

# Compare this many large-company fiscal years in quarterly/annual flow checks.
CASH_FLOW_CHECK_SAMPLE_SIZE = 20

# Allow a $1 million absolute cash-flow reconciliation difference.
CASH_FLOW_ABS_TOLERANCE = 1.0

# Allow a 0.1% relative cash-flow reconciliation difference.
CASH_FLOW_REL_TOLERANCE = 0.001


# SIGNAL STANDARDIZATION AND EVALUATION

# Begin evaluated forward returns in 2005 after the configured history runway.
EVAL_START = "2005-01-31"

# Winsorize each monthly investable cross-section at the 1st/99th percentiles.
WINSOR_PCT = 0.01

# Use the universe z-score when a sector has fewer than ten valid names.
MIN_SECTOR_SIZE = 10

# Weight the universe z-score 70% and sector z-score 30%. This split came from
# prior practice rather than a search over this backtest; it retains broad
# cross-industry information while limiting unintended sector bets.
UNIVERSE_Z_WEIGHT = 0.7

# Form quintiles for common signal evaluation reports.
N_QUANTILES = 5

# Use twelve fiscal quarters (three years) for quality stability measures.
QUALITY_VOL_QUARTERS = 12

# Permit a stability estimate with at least eight usable consecutive quarters.
QUALITY_MIN_VOL_OBSERVATIONS = 8

# Measure split-adjusted share change at short, medium, and long horizons.
CAPITAL_DEPLOYMENT_HORIZONS = (3, 12, 36)

# Show ten empirical split examples when auditing CIZ share adjustment.
SPLIT_AUDIT_SAMPLE_SIZE = 10

# Treat factor changes of at least 2x as split candidates in that audit.
SPLIT_MIN_RATIO = 2.0

# Estimate tested risk measures over at most 36 consecutive monthly returns.
RISK_VOL_MONTHS = 36

# Require 24 monthly observations for tested volatility/beta estimates.
RISK_MIN_MONTHS = 24

# Keep the rejected fundamental-momentum test at its preselected 12-quarter window.
FUNDMOM_QUARTERS = 12

# Require eight consecutive quarters for the rejected trend test.
FUNDMOM_MIN_QUARTERS = 8


# COMPOSITE MODEL

# Include the five blocks that survived research. Risk was excluded after
# inverted quintiles/negative spread; fundamental momentum decayed with lookback.
MODEL_BLOCKS = ["value", "quality", "capital_deployment", "momentum", "sentiment"]

# Equal weight blocks by default; alternatives exist only for structural tests.
COMPOSITE_WEIGHTING = "equal"

# Estimate inverse-volatility composite weights from a trailing five-year window.
# The estimate is lagged one month, and equal weights apply until the window fills.
INVERSE_VOL_MONTHS = 60

# Start the combined model in 2007 because short-interest coverage begins mid-2006.
COMPOSITE_START = "2007-01-31"


# PORTFOLIO CONSTRUCTION

# Report and use score deciles in portfolio selection and stock explanations.
N_DECILES = 10

# Center and half-width of the pre-weighting holdings guardrail. The latest live
# top-decile selection has about 267 names; 240-280 is an occasional guardrail.
# The 0.02% minimum-weight floor is applied later, leaving about 190 names on
# average in the backtest and 160 names in the September 2026 target.
TARGET_HOLDINGS = 260
HOLDINGS_BAND = 20

# Buy and retain only the top decile: there is no tuned entry/exit gap. Paired
# tests found no measurable active-return difference among feasible rules (all
# Newey-West t-stats -0.64 to 1.20 and every block-bootstrap interval crossed
# zero), so this was chosen for having no tuned parameters, not for performance.
# The former hard-275 rule naturally admitted 612 names on average before
# truncation, meaning its cap selected the portfolio instead of guarding it.
BUY_PERCENTILE = 0.10
SELL_PERCENTILE = 0.10

# Cap names at 3%. Sensitivity testing reduced top-ten concentration materially
# while keeping net information ratio within 0.05 of the 4% baseline; 2.5% did not.
MAX_NAME_WEIGHT = 0.03

# Drop benchmark-proportional targets below 0.02%: on an $800k account the
# median target was $395 and the smallest just $16, too small to matter.
# Communication services, utilities, and real estate sometimes fall below
# three names, but a 0.01% position adds no useful diversification. The final
# portfolio holds about 190 names on average in the backtest (160 in the latest
# September 2026 target); the 10pp sector cap and 3% name cap also constrain
# concentration. Performance improved at every tested floor (0.02%: +2.05bp/month;
# Newey-West t=2.14), but that was not the basis for this decision.
MIN_POSITION_WEIGHT = 0.0002

# Limit each sector to +/-10 percentage points versus benchmark weight.
MAX_SECTOR_DEVIATION = 0.10

# Use benchmark-proportional weights; alternatives remain research diagnostics.
WEIGHT_SCHEME = "benchmark"

# Apply the beta constraint adopted after it improved IR, tracking error, and
# volatility in the predeclared sensitivity comparison.
BETA_CONSTRAINT_ENABLED = True

# Allow modestly above-market beta without permitting an uncontrolled risk tilt.
MAX_BETA = 1.10

# Prevent the stock-selection portfolio from becoming a low-beta allocation bet.
MIN_BETA = 0.95

# Charge one-way trading costs by contemporaneous size bucket, in basis points.
TRADING_COST_BPS = {"large": 10, "mid": 20, "small": 40}


# PORTFOLIO ANALYSIS

# Use twelve Newey-West lags for monthly factor and active-return regressions.
HAC_LAGS = 12

# Use three-year rolling windows for stability charts and factor exposures.
ROLLING_MONTHS = 36

# Warn in live rebalances when the latest rdq is more than six months old.
REBALANCE_STALE_FUNDAMENTAL_MONTHS = 6

# Warn below $10 even though the formal investability floor remains $5.
REBALANCE_WARNING_MIN_PRICE = 10.0

# Warn below $100 million even though the formal floor remains $50 million.
REBALANCE_WARNING_MIN_MARKET_CAP = 100.0


# PROJECT AND CACHE PATHS

# Anchor every project-relative output to the directory containing config.py.
PROJECT_ROOT = Path(__file__).resolve().parent

# Store all reproducible local data products under data/.
DATA_DIR = PROJECT_ROOT / "data"

# Document every configured research-panel column; the panel hard checks require
# this file to contain exactly one nonempty description for every output field.
DATA_DICTIONARY_PATH = PROJECT_ROOT / "docs" / "data_dictionary.csv"

# Keep slow raw-query caches outside the repository by default.
CACHE_DIR = Path.home() / ".cache"

# Override the CRSP/short-interest cache with WRDS_PRICE_CACHE_DIR if needed.
PRICE_CACHE_DIR = Path(
    os.environ.get("WRDS_PRICE_CACHE_DIR", CACHE_DIR / "wrds_crsp_monthly_prices")
)

# Override the Compustat fundamentals/reference cache independently.
FUNDAMENTALS_CACHE_DIR = Path(
    os.environ.get(
        "WRDS_FUNDAMENTALS_CACHE_DIR",
        CACHE_DIR / "wrds_compustat_fundamentals",
    )
)

# Override the factor/index-return cache independently.
FACTOR_CACHE_DIR = Path(
    os.environ.get("WRDS_FACTOR_CACHE_DIR", CACHE_DIR / "wrds_monthly_factors")
)

# Override the provisional Security Daily cache independently.
LIVE_PRICE_CACHE_DIR = Path(
    os.environ.get("WRDS_LIVE_PRICE_CACHE_DIR", CACHE_DIR / "wrds_live_prices")
)

# Canonical monthly CRSP price output.
PRICES_OUTPUT_PATH = DATA_DIR / "prices.parquet"

# Provisional Compustat Security Daily monthly extension.
LIVE_PRICES_OUTPUT_PATH = DATA_DIR / "live_prices.parquet"

# Point-in-time Compustat security short-interest observations.
SHORT_INTEREST_OUTPUT_PATH = DATA_DIR / "short_interest.parquet"

# Combined annual and quarterly Compustat fundamentals.
FUNDAMENTALS_OUTPUT_PATH = DATA_DIR / "fundamentals.parquet"

# Filtered CCM link history used for date-valid security/company joins.
LINK_OUTPUT_PATH = DATA_DIR / "link.parquet"

# Point-in-time Compustat GICS history.
SECTORS_OUTPUT_PATH = DATA_DIR / "sectors.parquet"

# Monthly factors plus S&P 500 and Russell 3000 total returns.
FACTORS_OUTPUT_PATH = DATA_DIR / "factors.parquet"

# S&P 500 research-panel destination.
PANEL_SP500_OUTPUT_PATH = DATA_DIR / "panel_sp500.parquet"

# Russell 3000 proxy research-panel destination.
PANEL_RUSSELL3000_OUTPUT_PATH = DATA_DIR / "panel_russell3000.parquet"

# Agreed live holdings; automation reads but never overwrites this file.
CURRENT_PORTFOLIO_PATH = DATA_DIR / "current_portfolio.csv"

# Permit a missing or empty holdings file only for the explicit first launch.
# Keep False after launch so a lost file cannot create a full-cash rebalance.
FIRST_REBALANCE = False

# Map the selected universe to its output without scattering path conditions.
PANEL_OUTPUT_PATHS = {
    "sp500": PANEL_SP500_OUTPUT_PATH,
    "russell3000": PANEL_RUSSELL3000_OUTPUT_PATH,
    "all": DATA_DIR / "panel_all.parquet",
}

# Resolve the active panel path once for downstream imports.
PANEL_OUTPUT_PATH = PANEL_OUTPUT_PATHS[UNIVERSE]


# WRDS CONNECTION

# Use WRDS_USERNAME when supplied, otherwise default to the local login name.
WRDS_USERNAME = os.environ.get("WRDS_USERNAME", getpass.getuser())

# Select the annual Compustat fields written to fundamentals.parquet; removing
# one also removes its ``_a`` counterpart from downstream panels.
ANNUAL_COLUMNS = (
    "gvkey",
    "datadate",
    "fyear",
    "fyr",
    "pdate",
    "fdate",
    "final",
    "upd",
    "apdedate",
    "curcd",
    "acctstd",
    "compst",
    "scf",
    "src",
    "stalt",
    # Balance sheet.
    "at",
    "lt",
    "lse",
    "seq",
    "teq",
    "ceq",
    "re",
    "pstk",
    "pstkr",
    "pstkn",
    "txditc",
    "txdb",
    "mibt",
    "act",
    "lct",
    "che",
    "ch",
    "ivst",
    "rect",
    "invt",
    "aco",
    "ao",
    "dlc",
    "dd1",
    "dltt",
    "ap",
    "lco",
    "lo",
    "xacc",
    "np",
    "ppent",
    "ppegt",
    "intan",
    "gdwl",
    "ivao",
    "drc",
    "drlt",
    "tstk",
    "tstkn",
    "icapt",
    "wcap",
    "dcom",
    # Income statement.
    "sale",
    "revt",
    "cogs",
    "xsga",
    "xrd",
    "xopr",
    "oibdp",
    "oiadp",
    "dp",
    "xint",
    "nopi",
    "spi",
    "pi",
    "txt",
    "txdi",
    "ib",
    "ibcom",
    "ibmii",
    "ni",
    "mii",
    "xido",
    "xi",
    "do",
    "esub",
    "stkco",
    "gdwlip",
    "rcp",
    "wdp",
    "setp",
    "epspx",
    "epsfx",
    "epspi",
    "epsfi",
    "opeps",
    "oprepsx",
    # Shares and market data.
    "csho",
    "cshpri",
    "cshfd",
    "cshi",
    "cshtr_f",
    "ajex",
    "ajp",
    "adjex_f",
    "prcc_f",
    "prch_f",
    "prcl_f",
    "mkvalt",
    "dvpsx_f",
    "dvpsp_f",
    # Cash flow.
    "oancf",
    "ivncf",
    "fincf",
    "capx",
    "dpc",
    "txdc",
    "dv",
    "cdvc",
    "pdvc",
    "prstkc",
    "prstkcc",
    "sstk",
    "scstkc",
    "dltis",
    "dltr",
    "dlcch",
    "aqc",
    "sppe",
    "siv",
    "ivch",
    "recch",
    "invch",
    "apalch",
    "txach",
    "aoloch",
    "xidoc",
    "chech",
    "exre",
    "esubc",
    "fopo",
    "sppiv",
    "fiao",
    "ivaco",
    "txpd",
    "intpn",
    # Financials-specific.
    "nim",
    "tie",
    "tii",
    "pll",
    "nco",
)

# Select quarterly Compustat fields written to fundamentals.parquet and panels.
QUARTERLY_COLUMNS = (
    "gvkey",
    "datadate",
    "fyearq",
    "fqtr",
    "fyr",
    "datacqtr",
    "datafqtr",
    "rdq",
    "pdateq",
    "fdateq",
    "finalq",
    "updq",
    "apdedateq",
    "curcdq",
    "acctstdq",
    "compstq",
    "scfq",
    "srcq",
    "staltq",
    # Balance sheet.
    "atq",
    "ltq",
    "lseq",
    "seqq",
    "teqq",
    "ceqq",
    "req",
    "pstkq",
    "pstkrq",
    "pstknq",
    "txditcq",
    "txdbq",
    "mibtq",
    "actq",
    "lctq",
    "cheq",
    "chq",
    "ivstq",
    "rectq",
    "invtq",
    "acoq",
    "aoq",
    "dlcq",
    "dd1q",
    "dlttq",
    "apq",
    "lcoq",
    "loq",
    "xaccq",
    "npq",
    "ppentq",
    "ppegtq",
    "intanq",
    "gdwlq",
    "ivaoq",
    "ivltq",
    "altoq",
    "drcq",
    "drltq",
    "tstkq",
    "tstknq",
    "icaptq",
    "wcapq",
    "dcomq",
    # Income statement.
    "saleq",
    "revtq",
    "cogsq",
    "xsgaq",
    "xrdq",
    "xoprq",
    "oibdpq",
    "oiadpq",
    "dpq",
    "xintq",
    "nopiq",
    "spiq",
    "piq",
    "txtq",
    "txdiq",
    "ibq",
    "ibcomq",
    "ibmiiq",
    "niq",
    "miiq",
    "xidoq",
    "xiq",
    "doq",
    "esubq",
    "stkcoq",
    "gdwlipq",
    "rcpq",
    "wdpq",
    "setpq",
    "epspxq",
    "epsfxq",
    "epspiq",
    "epsfiq",
    "opepsq",
    "oepsxq",
    # Shares and market.
    "cshoq",
    "cshprq",
    "cshfdq",
    "cshiq",
    "cshopq",
    "csh12q",
    "cshtrq",
    "ajexq",
    "ajpq",
    "adjex",
    "prccq",
    "prchq",
    "prclq",
    "prcraq",
    "mkvaltq",
    "dvpsxq",
    "dvpspq",
    # Fiscal-year-to-date cash flow.
    "oancfy",
    "ivncfy",
    "fincfy",
    "capxy",
    "dpcy",
    "txdcy",
    "dvy",
    "cdvcy",
    "pdvcy",
    "prstkcy",
    "prstkccy",
    "sstky",
    "scstkcy",
    "dltisy",
    "dltry",
    "dlcchy",
    "aqcy",
    "sppey",
    "sivy",
    "ivchy",
    "recchy",
    "invchy",
    "apalchy",
    "txachy",
    "aolochy",
    "xidocy",
    "chechy",
    "exrey",
    "esubcy",
    "fopoy",
    "sppivy",
    "fiaoy",
    "ivacoy",
    "txpdy",
    "intpny",
    "stkcoy",
    # Financials-specific.
    "nimq",
    "tieq",
    "tiiq",
    "pllq",
    "ncoq",
)

# Identify fiscal-YTD cash-flow fields that must be de-cumulated by quarter.
QUARTERLY_YTD_COLUMNS = (
    "oancfy",
    "ivncfy",
    "fincfy",
    "capxy",
    "dpcy",
    "txdcy",
    "dvy",
    "cdvcy",
    "pdvcy",
    "prstkcy",
    "prstkccy",
    "sstky",
    "scstkcy",
    "dltisy",
    "dltry",
    "dlcchy",
    "aqcy",
    "sppey",
    "sivy",
    "ivchy",
    "recchy",
    "invchy",
    "apalchy",
    "txachy",
    "aolochy",
    "xidocy",
    "chechy",
    "exrey",
    "esubcy",
    "fopoy",
    "sppivy",
    "fiaoy",
    "ivacoy",
    "txpdy",
    "intpny",
    "stkcoy",
)

# Derive stable names for the de-cumulated quarterly flow outputs.
QUARTERLY_FLOW_COLUMNS = tuple(
    f"{column[:-1]}_qtr" for column in QUARTERLY_YTD_COLUMNS
)

# Use recognizable large firms for empirical Compustat units and scale checks.
UNIT_CHECK_COMPANIES = {
    "001690": "Apple",
    "012141": "Microsoft",
    "011259": "Walmart",
    "004503": "ExxonMobil",
    "002968": "JPMorgan",
}
