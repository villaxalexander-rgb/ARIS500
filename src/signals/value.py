"""Define the production cash-flow, earnings, and EBITDA value signals.

Reads point-in-time quarterly fields from an in-memory panel and writes nothing.
CFO yield, earnings yield, and EBITDA/enterprise-value are expressed as yields
so negative fundamentals rank as expensive rather than being discarded; higher
is better and no sign flip is applied. EBITDA/EV excludes Financials and Real
Estate because debt is operating raw material there, and nonpositive EV is
invalid. Evaluation retained the equal-available-signal value block as one of
the five production blocks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals.standardize import restandardize_monthly, standardize_signal


VALUE_SIGNAL_NAMES = ("cfo_yield", "earnings_yield", "ebitda_to_ev")


@dataclass
class ValueSignalBlock:
    """Raw, standardized, and combined value signals plus construction diagnostics."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    value_block: pd.Series
    diagnostics: pd.DataFrame


def _quarterly_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one point-in-time row per company fiscal quarter from the panel."""
    required = [
        "gvkey",
        "datadate",
        "available_date",
        "fyearq",
        "fqtr",
        "oancf_qtr",
        "ibq",
        "oibdpq",
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing quarterly value inputs: {missing}")
    quarters = panel.loc[:, required].dropna(subset=["gvkey", "datadate"]).copy()
    quarters["gvkey"] = quarters["gvkey"].astype("string")
    quarters["datadate"] = pd.to_datetime(quarters["datadate"])
    quarters["available_date"] = pd.to_datetime(quarters["available_date"])
    for column in ("fyearq", "fqtr", "oancf_qtr", "ibq", "oibdpq"):
        quarters[column] = pd.to_numeric(quarters[column], errors="coerce")
    return (
        quarters.sort_values(["gvkey", "datadate", "fyearq", "fqtr"])
        .drop_duplicates(["gvkey", "datadate"], keep="last")
        .reset_index(drop=True)
    )


def _trailing_four_quarters(panel: pd.DataFrame) -> pd.DataFrame:
    """Build TTM fields from four consecutive fiscal quarters without future data."""
    quarters = _quarterly_history(panel)
    quarters["_fiscal_quarter"] = quarters["fyearq"] * 4.0 + quarters["fqtr"] - 1.0
    groups = quarters.groupby("gvkey", sort=False)
    step = groups["_fiscal_quarter"].diff()
    consecutive = step.eq(1.0)
    consecutive &= consecutive.groupby(quarters["gvkey"], sort=False).shift(1).eq(True)
    consecutive &= consecutive.groupby(quarters["gvkey"], sort=False).shift(2).eq(True)

    # The current panel has no stored TTM columns, so derive them from the
    # point-in-time quarterly observations attached by build_panel.py.
    source_fields = {
        "oancf_qtr": "oancf_ttm",
        "ibq": "ib_ttm",
        "oibdpq": "oibdp_ttm",
    }
    for source, output in source_fields.items():
        rolling_sum = (
            groups[source]
            .rolling(4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )
        quarters[output] = rolling_sum.where(consecutive)
    available_ns = quarters["available_date"].astype("int64")
    quarters["_ttm_available_date"] = pd.to_datetime(
        available_ns.groupby(quarters["gvkey"], sort=False)
        .rolling(4, min_periods=4)
        .max()
        .reset_index(level=0, drop=True)
    ).where(consecutive)
    return quarters.loc[
        :,
        [
            "gvkey",
            "datadate",
            "oancf_ttm",
            "ib_ttm",
            "oibdp_ttm",
            "_ttm_available_date",
        ],
    ]


def _value_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach TTM accounting inputs and enterprise value to each panel row."""
    required = ["gvkey", "datadate", "market_cap", "dlcq", "dlttq", "cheq", "gsector"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing value inputs: {missing}")

    ttm = _trailing_four_quarters(panel).set_index(["gvkey", "datadate"])
    keys = pd.MultiIndex.from_arrays(
        [
            panel["gvkey"].astype("string"),
            pd.to_datetime(panel["datadate"]),
        ],
        names=["gvkey", "datadate"],
    )
    inputs = panel.loc[:, ["market_cap", "dlcq", "dlttq", "cheq", "gsector"]].copy()
    for column in ("market_cap", "dlcq", "dlttq", "cheq", "gsector"):
        inputs[column] = pd.to_numeric(inputs[column], errors="coerce")
    for column in ("oancf_ttm", "ib_ttm", "oibdp_ttm"):
        inputs[column] = ttm[column].reindex(keys).to_numpy()
    history_available = pd.Series(
        ttm["_ttm_available_date"].reindex(keys).to_numpy(), index=panel.index
    )
    history_is_public = history_available.le(pd.to_datetime(panel["date"]))
    # A late prior quarter can otherwise leak into a TTM value keyed by a newer
    # fiscal quarter. Mask the derived value until every source quarter is public.
    inputs.loc[~history_is_public, ["oancf_ttm", "ib_ttm", "oibdp_ttm"]] = np.nan
    inputs["enterprise_value"] = (
        inputs["market_cap"]
        + inputs["dlcq"]
        + inputs["dlttq"]
        - inputs["cheq"]
    )
    return inputs


def raw_value_signals(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct all three raw value yields and return construction diagnostics.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame]``.
    """
    inputs = _value_inputs(panel)
    positive_market_cap = inputs["market_cap"].gt(0)
    raw = pd.DataFrame(index=panel.index, dtype="float64")
    raw["cfo_yield"] = (
        inputs["oancf_ttm"] / inputs["market_cap"]
    ).where(positive_market_cap)
    raw["earnings_yield"] = (
        inputs["ib_ttm"] / inputs["market_cap"]
    ).where(positive_market_cap)

    # EBITDA/EV is not meaningful for Financials or Real Estate, where debt is
    # operating raw material and banks generally do not report EBITDA.
    financial_or_real_estate = inputs["gsector"].isin([40, 60])
    valid_ev = inputs["enterprise_value"].gt(0) & ~financial_or_real_estate
    raw["ebitda_to_ev"] = (
        inputs["oibdp_ttm"] / inputs["enterprise_value"]
    ).where(valid_ev)
    raw = raw.replace([np.inf, -np.inf], np.nan)

    computable_ev = inputs["enterprise_value"].notna() & ~financial_or_real_estate
    nonpositive_ev = computable_ev & inputs["enterprise_value"].le(0)
    investable = panel["investable"].fillna(False).astype(bool)
    distinct_company_period = pd.DataFrame(
        {
            "gvkey": panel["gvkey"].astype("string"),
            "datadate": pd.to_datetime(panel["datadate"]),
            "nonpositive_ev": nonpositive_ev,
        }
    ).dropna(subset=["gvkey", "datadate"])
    distinct_company_period = distinct_company_period.drop_duplicates(
        ["gvkey", "datadate"], keep="last"
    )
    diagnostic_rows = [
        {
            "diagnostic": "nonfinancial rows with computable EV",
            "count": int(computable_ev.sum()),
            "share": 1.0,
        },
        {
            "diagnostic": "rows with EV <= 0 (EBITDA/EV set to NaN)",
            "count": int(nonpositive_ev.sum()),
            "share": float(nonpositive_ev.sum() / computable_ev.sum())
            if computable_ev.any()
            else np.nan,
        },
        {
            "diagnostic": "investable rows with EV <= 0",
            "count": int((nonpositive_ev & investable).sum()),
            "share": float(
                (nonpositive_ev & investable).sum()
                / (computable_ev & investable).sum()
            )
            if (computable_ev & investable).any()
            else np.nan,
        },
        {
            "diagnostic": "distinct company-quarters with EV <= 0",
            "count": int(distinct_company_period["nonpositive_ev"].sum()),
            "share": float(distinct_company_period["nonpositive_ev"].mean())
            if len(distinct_company_period)
            else np.nan,
        },
        {
            "diagnostic": "Financials/Real Estate rows excluded from EBITDA/EV",
            "count": int(financial_or_real_estate.sum()),
            "share": float(financial_or_real_estate.mean()),
        },
    ]
    return raw, pd.DataFrame(diagnostic_rows)


def cfo_yield(panel: pd.DataFrame) -> pd.Series:
    """Return trailing operating cash flow divided by current market cap.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_value_signals(panel)
    return raw["cfo_yield"]


def earnings_yield(panel: pd.DataFrame) -> pd.Series:
    """Return trailing income before extraordinary items divided by market cap.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_value_signals(panel)
    return raw["earnings_yield"]


def ebitda_to_ev(panel: pd.DataFrame) -> pd.Series:
    """Return trailing EBITDA divided by positive enterprise value.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_value_signals(panel)
    return raw["ebitda_to_ev"]


def build_value_block(panel: pd.DataFrame) -> ValueSignalBlock:
    """Standardize the value signals and combine available components equally.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``ValueSignalBlock``.
    """
    raw, diagnostics = raw_value_signals(panel)
    standardized = pd.DataFrame(index=panel.index, dtype="float64")
    for name in VALUE_SIGNAL_NAMES:
        standardized[name] = standardize_signal(panel, raw[name])["signal"]
    available_average = standardized.mean(axis=1, skipna=True)
    available_average = available_average.where(standardized.notna().any(axis=1))
    block = restandardize_monthly(panel, available_average).rename("value_block")
    return ValueSignalBlock(
        raw=raw,
        standardized=standardized,
        value_block=block,
        diagnostics=diagnostics,
    )
