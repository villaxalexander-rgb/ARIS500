"""Define tested profitability and stability signals and the production quality block.

Reads point-in-time quarterly fields from an in-memory panel and writes nothing.
Signals are CFO/invested capital, EBITDA ROIC, ROE, and trailing volatility of
asset turnover, EBIT margin, and EPS; volatility measures are sign-flipped so
higher means more stable. Capital-return measures exclude Financials and Real
Estate because debt is an operating input. Evaluation dropped asset-turnover
stability (IC -0.0047; spread -0.34%/month, t=-2.24) and EPS stability (IC
-0.0031, t=-0.69), combined the 0.861-correlated capital-return pair, and kept
that combined input, ROE, and EBIT-margin stability in the production block.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import QUALITY_MIN_VOL_OBSERVATIONS, QUALITY_VOL_QUARTERS
from src.signals.standardize import restandardize_monthly, standardize_signal


QUALITY_SIGNAL_NAMES = (
    "cfo_to_ic",
    "ebitda_roic",
    "roe_ltm",
    "asset_turnover_vol",
    "ebit_margin_vol",
    "eps_vol",
)


@dataclass
class QualitySignalBlock:
    """Raw, standardized, and combined quality signals plus diagnostics."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    returns_on_capital: pd.Series
    quality_block: pd.Series
    legacy_six_signal_block: pd.Series
    diagnostics: pd.DataFrame


def _quarterly_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one point-in-time row per company fiscal quarter from the panel."""
    fields = [
        "gvkey",
        "datadate",
        "available_date",
        "fyearq",
        "fqtr",
        "oancf_qtr",
        "oibdpq",
        "ibq",
        "saleq",
        "oiadpq",
        "epspxq",
        "ceqq",
        "atq",
    ]
    missing = [column for column in fields if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing quarterly quality inputs: {missing}")
    quarters = panel.loc[:, fields].dropna(subset=["gvkey", "datadate"]).copy()
    quarters["gvkey"] = quarters["gvkey"].astype("string")
    quarters["datadate"] = pd.to_datetime(quarters["datadate"])
    quarters["available_date"] = pd.to_datetime(quarters["available_date"])
    for column in fields[3:]:
        quarters[column] = pd.to_numeric(quarters[column], errors="coerce")
    quarters = (
        quarters.sort_values(["gvkey", "datadate", "fyearq", "fqtr"])
        .drop_duplicates(["gvkey", "datadate"], keep="last")
        .reset_index(drop=True)
    )
    quarters["_fiscal_quarter"] = (
        quarters["fyearq"] * 4.0 + quarters["fqtr"] - 1.0
    )
    return quarters


def _consecutive_window(quarters: pd.DataFrame, window: int) -> pd.Series:
    """Flag rows ending a window of consecutive observed fiscal quarters."""
    if window < 2:
        return quarters["_fiscal_quarter"].notna()
    step_is_one = quarters.groupby("gvkey", sort=False)["_fiscal_quarter"].diff().eq(1)
    valid_steps = (
        step_is_one.fillna(False).astype("int8")
        .groupby(quarters["gvkey"], sort=False)
        .rolling(window - 1, min_periods=window - 1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return valid_steps.eq(window - 1)


def _quarterly_quality_fields(panel: pd.DataFrame) -> pd.DataFrame:
    """Build TTM, average-capital, and trailing-volatility quarterly fields."""
    if QUALITY_VOL_QUARTERS < QUALITY_MIN_VOL_OBSERVATIONS:
        raise ValueError(
            "QUALITY_VOL_QUARTERS must be at least "
            f"{QUALITY_MIN_VOL_OBSERVATIONS} so volatility measures can meet "
            "their minimum observation requirement."
        )
    quarters = _quarterly_history(panel)
    groups = quarters["gvkey"]
    consecutive_four = _consecutive_window(quarters, 4)

    for source, output in {
        "oancf_qtr": "oancf_ttm",
        "oibdpq": "oibdp_ttm",
        "ibq": "ib_ttm",
    }.items():
        rolling_sum = (
            quarters[source]
            .groupby(groups, sort=False)
            .rolling(4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )
        quarters[output] = rolling_sum.where(consecutive_four)

    average_common_equity = (
        quarters["ceqq"]
        .groupby(groups, sort=False)
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )
    quarters["average_common_equity"] = average_common_equity.where(
        consecutive_four
    )
    available_ns = quarters["available_date"].astype("int64")
    quarters["_ttm_available_date"] = pd.to_datetime(
        available_ns.groupby(groups, sort=False)
        .rolling(4, min_periods=4)
        .max()
        .reset_index(level=0, drop=True)
    ).where(consecutive_four)

    quarters["_ebit_margin"] = (
        quarters["oiadpq"] / quarters["saleq"]
    ).where(quarters["saleq"].gt(0))
    quarters["_eps"] = quarters["epspxq"]

    step_is_one = quarters.groupby("gvkey", sort=False)["_fiscal_quarter"].diff().eq(1)
    sequence_break = ~step_is_one.fillna(False)
    consecutive_run = sequence_break.groupby(quarters["gvkey"], sort=False).cumsum()
    volatility_groups = [quarters["gvkey"], consecutive_run]
    quarters["_vol_available_date"] = pd.to_datetime(
        available_ns.groupby(volatility_groups, sort=False)
        .rolling(QUALITY_VOL_QUARTERS, min_periods=1)
        .max()
        .reset_index(level=[0, 1], drop=True)
    )
    rolling_sales_volatility = (
        quarters["saleq"]
        .groupby(volatility_groups, sort=False)
        .rolling(
            QUALITY_VOL_QUARTERS,
            min_periods=QUALITY_MIN_VOL_OBSERVATIONS,
        )
        .std(ddof=1)
        .reset_index(level=[0, 1], drop=True)
    )
    rolling_average_assets = (
        quarters["atq"]
        .groupby(volatility_groups, sort=False)
        .rolling(
            QUALITY_VOL_QUARTERS,
            min_periods=QUALITY_MIN_VOL_OBSERVATIONS,
        )
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    # Scale quarterly sales volatility by average assets over the same window.
    quarters["asset_turnover_volatility"] = (
        rolling_sales_volatility / rolling_average_assets
    ).where(rolling_average_assets.gt(0))
    for source, output in {
        "_ebit_margin": "ebit_margin_volatility",
        "_eps": "eps_volatility",
    }.items():
        # Rolling within uninterrupted fiscal-quarter runs prevents observations
        # before a reporting gap from entering the 8-to-12-quarter window.
        quarters[output] = (
            quarters[source]
            .groupby(volatility_groups, sort=False)
            .rolling(
                QUALITY_VOL_QUARTERS,
                min_periods=QUALITY_MIN_VOL_OBSERVATIONS,
            )
            .std(ddof=1)
            .reset_index(level=[0, 1], drop=True)
        )

    return quarters.loc[
        :,
        [
            "gvkey",
            "datadate",
            "oancf_ttm",
            "oibdp_ttm",
            "ib_ttm",
            "average_common_equity",
            "_ttm_available_date",
            "asset_turnover_volatility",
            "ebit_margin_volatility",
            "eps_volatility",
            "_vol_available_date",
            "saleq",
        ],
    ]


def _quality_inputs(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach quarterly quality inputs to every panel row without shifting them."""
    fields = ["gvkey", "datadate", "ceqq", "dlttq", "dlcq", "cheq", "gsector"]
    missing = [column for column in fields if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing quality inputs: {missing}")

    quarters = _quarterly_quality_fields(panel)
    indexed_quarters = quarters.set_index(["gvkey", "datadate"])
    keys = pd.MultiIndex.from_arrays(
        [
            panel["gvkey"].astype("string"),
            pd.to_datetime(panel["datadate"]),
        ],
        names=["gvkey", "datadate"],
    )
    inputs = panel.loc[:, ["ceqq", "dlttq", "dlcq", "cheq", "gsector"]].copy()
    for column in inputs.columns:
        inputs[column] = pd.to_numeric(inputs[column], errors="coerce")
    derived = [
        "oancf_ttm",
        "oibdp_ttm",
        "ib_ttm",
        "average_common_equity",
        "asset_turnover_volatility",
        "ebit_margin_volatility",
        "eps_volatility",
    ]
    for column in derived:
        inputs[column] = indexed_quarters[column].reindex(keys).to_numpy()
    panel_date = pd.to_datetime(panel["date"])
    ttm_available = pd.Series(
        indexed_quarters["_ttm_available_date"].reindex(keys).to_numpy(),
        index=panel.index,
    )
    vol_available = pd.Series(
        indexed_quarters["_vol_available_date"].reindex(keys).to_numpy(),
        index=panel.index,
    )
    # Out-of-order filings can make an earlier source quarter public after the
    # current quarter. Derived histories remain unavailable until every source is public.
    inputs.loc[
        ~ttm_available.le(panel_date),
        ["oancf_ttm", "oibdp_ttm", "ib_ttm", "average_common_equity"],
    ] = np.nan
    inputs.loc[
        ~vol_available.le(panel_date),
        ["asset_turnover_volatility", "ebit_margin_volatility", "eps_volatility"],
    ] = np.nan

    # Common equity is the book-equity component of invested capital.
    inputs["invested_capital"] = (
        inputs["ceqq"] + inputs["dlttq"] + inputs["dlcq"] - inputs["cheq"]
    )
    return inputs, quarters


def _distinct_company_period_count(panel: pd.DataFrame, mask: pd.Series) -> int:
    """Count unique company-quarters for which a row-level condition is true."""
    periods = pd.DataFrame(
        {
            "gvkey": panel["gvkey"].astype("string"),
            "datadate": pd.to_datetime(panel["datadate"]),
            "condition": mask.fillna(False).to_numpy(),
        }
    ).dropna(subset=["gvkey", "datadate"])
    return int(periods.groupby(["gvkey", "datadate"])["condition"].any().sum())


def raw_quality_signals(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct all six raw quality signals and construction diagnostics.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame]``.
    """
    inputs, quarters = _quality_inputs(panel)
    financial_or_real_estate = inputs["gsector"].isin([40, 60])
    positive_invested_capital = inputs["invested_capital"].gt(0)
    positive_average_equity = inputs["average_common_equity"].gt(0)

    raw = pd.DataFrame(index=panel.index, dtype="float64")
    # Invested capital and EBITDA are not meaningful for Financials or Real
    # Estate, where debt is an operating input, so the block uses other signals.
    valid_return_on_capital = positive_invested_capital & ~financial_or_real_estate
    raw["cfo_to_ic"] = (
        inputs["oancf_ttm"] / inputs["invested_capital"]
    ).where(valid_return_on_capital)
    raw["ebitda_roic"] = (
        inputs["oibdp_ttm"] / inputs["invested_capital"]
    ).where(valid_return_on_capital)
    raw["roe_ltm"] = (
        inputs["ib_ttm"] / inputs["average_common_equity"]
    ).where(positive_average_equity)

    # Low volatility indicates higher quality, so reverse these measures before
    # standardization so every signal has the same higher-is-better direction.
    raw["asset_turnover_vol"] = -inputs["asset_turnover_volatility"]
    raw["ebit_margin_vol"] = -inputs["ebit_margin_volatility"]
    raw["eps_vol"] = -inputs["eps_volatility"]
    raw = raw.replace([np.inf, -np.inf], np.nan)

    computable_ic = inputs["invested_capital"].notna() & ~financial_or_real_estate
    nonpositive_ic = computable_ic & inputs["invested_capital"].le(0)
    computable_equity = inputs["average_common_equity"].notna()
    nonpositive_equity = computable_equity & inputs["average_common_equity"].le(0)
    investable = panel["investable"].fillna(False).astype(bool)
    populated_sales = quarters["saleq"].notna()
    nonpositive_sales = populated_sales & quarters["saleq"].le(0)
    diagnostic_rows: list[dict[str, object]] = [
        {
            "diagnostic": "nonfinancial rows with computable invested capital",
            "count": int(computable_ic.sum()),
            "share": 1.0,
        },
        {
            "diagnostic": "rows with invested capital <= 0",
            "count": int(nonpositive_ic.sum()),
            "share": float(nonpositive_ic.sum() / computable_ic.sum())
            if computable_ic.any()
            else np.nan,
        },
        {
            "diagnostic": "investable rows with invested capital <= 0",
            "count": int((nonpositive_ic & investable).sum()),
            "share": float(
                (nonpositive_ic & investable).sum()
                / (computable_ic & investable).sum()
            )
            if (computable_ic & investable).any()
            else np.nan,
        },
        {
            "diagnostic": "distinct company-quarters with invested capital <= 0",
            "count": _distinct_company_period_count(panel, nonpositive_ic),
            "share": np.nan,
        },
        {
            "diagnostic": "rows with average common equity <= 0",
            "count": int(nonpositive_equity.sum()),
            "share": float(nonpositive_equity.sum() / computable_equity.sum())
            if computable_equity.any()
            else np.nan,
        },
        {
            "diagnostic": "investable rows with average common equity <= 0",
            "count": int((nonpositive_equity & investable).sum()),
            "share": float(
                (nonpositive_equity & investable).sum()
                / (computable_equity & investable).sum()
            )
            if (computable_equity & investable).any()
            else np.nan,
        },
        {
            "diagnostic": "distinct company-quarters with common equity <= 0",
            "count": _distinct_company_period_count(panel, nonpositive_equity),
            "share": np.nan,
        },
        {
            "diagnostic": "quarters skipped for EBIT margin because sales <= 0",
            "count": int(nonpositive_sales.sum()),
            "share": float(nonpositive_sales.sum() / populated_sales.sum())
            if populated_sales.any()
            else np.nan,
        },
        {
            "diagnostic": "Financials/Real Estate rows excluded from capital returns",
            "count": int(financial_or_real_estate.sum()),
            "share": float(financial_or_real_estate.mean()),
        },
    ]
    investable_count = int(investable.sum())
    for name in QUALITY_SIGNAL_NAMES:
        available = investable & raw[name].notna()
        diagnostic_rows.append(
            {
                "diagnostic": f"investable coverage: {name}",
                "count": int(available.sum()),
                "share": float(available.sum() / investable_count)
                if investable_count
                else np.nan,
            }
        )
    return raw, pd.DataFrame(diagnostic_rows)


def cfo_to_ic(panel: pd.DataFrame) -> pd.Series:
    """Return trailing operating cash flow divided by invested capital.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_quality_signals(panel)
    return raw["cfo_to_ic"]


def ebitda_roic(panel: pd.DataFrame) -> pd.Series:
    """Return trailing EBITDA divided by invested capital.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_quality_signals(panel)
    return raw["ebitda_roic"]


def roe_ltm(panel: pd.DataFrame) -> pd.Series:
    """Return trailing income divided by average common equity.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_quality_signals(panel)
    return raw["roe_ltm"]


def asset_turnover_vol(panel: pd.DataFrame) -> pd.Series:
    """Return negative trailing sales volatility scaled by average assets.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_quality_signals(panel)
    return raw["asset_turnover_vol"]


def ebit_margin_vol(panel: pd.DataFrame) -> pd.Series:
    """Return negative trailing volatility of quarterly EBIT margins.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_quality_signals(panel)
    return raw["ebit_margin_vol"]


def eps_vol(panel: pd.DataFrame) -> pd.Series:
    """Return negative trailing volatility of quarterly basic EPS.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_quality_signals(panel)
    return raw["eps_vol"]


def build_quality_block(panel: pd.DataFrame) -> QualitySignalBlock:
    """Standardize quality signals and build the evidence-selected block score.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``QualitySignalBlock``.
    """
    raw, diagnostics = raw_quality_signals(panel)
    standardized = pd.DataFrame(index=panel.index, dtype="float64")
    for name in QUALITY_SIGNAL_NAMES:
        standardized[name] = standardize_signal(panel, raw[name])["signal"]

    # CFO/IC and EBITDA ROIC had 0.861 correlation, so average them into one
    # re-standardized returns-on-capital input rather than double-weight one idea.
    capital_components = standardized.loc[:, ["cfo_to_ic", "ebitda_roic"]]
    capital_average = capital_components.mean(axis=1, skipna=True).where(
        capital_components.notna().any(axis=1)
    )
    returns_on_capital = restandardize_monthly(
        panel,
        capital_average,
    ).rename("returns_on_capital")

    # Asset-turnover volatility is excluded because its IC was -0.0047 and its
    # EW spread was -0.34%/month (t=-2.24); EPS volatility is excluded because
    # its IC was -0.0031 (t=-0.69). Both remain available as tested signals.
    # ROE and sign-adjusted EBIT-margin stability remain because their ICs were
    # positive (0.0241 and 0.0176); these join returns on capital at equal weight.
    selected_inputs = pd.concat(
        [
            returns_on_capital,
            standardized["roe_ltm"],
            standardized["ebit_margin_vol"],
        ],
        axis=1,
    )
    selected_average = selected_inputs.mean(axis=1, skipna=True).where(
        selected_inputs.notna().any(axis=1)
    )
    block = restandardize_monthly(panel, selected_average).rename("quality_block")

    # Preserve the originally tested six-signal block for an explicit one-time
    # comparison; it is not the production quality score.
    legacy_average = standardized.mean(axis=1, skipna=True).where(
        standardized.notna().any(axis=1)
    )
    legacy_block = restandardize_monthly(panel, legacy_average).rename(
        "quality_block_old_six_signal"
    )
    return QualitySignalBlock(
        raw=raw,
        standardized=standardized,
        returns_on_capital=returns_on_capital,
        quality_block=block,
        legacy_six_signal_block=legacy_block,
        diagnostics=diagnostics,
    )
