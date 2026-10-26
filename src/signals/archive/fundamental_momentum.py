"""Define tested fundamental-trend signals; this block is NOT in the model.

Reads point-in-time quarterly CFO and revenue from an in-memory panel and writes
nothing. Each signal is the time-trend slope over consecutive quarters divided
by the mean absolute level; improving fundamentals score higher, with no sign
flip and no sector exclusion. Four-, eight-, and twelve-quarter sensitivity
tests showed the effect decayed to essentially nothing as the window lengthened,
so the block was rejected rather than tuned to the best historical window.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FUNDMOM_MIN_QUARTERS, FUNDMOM_QUARTERS
from src.signals.standardize import restandardize_monthly, standardize_signal


FUNDMOM_SIGNAL_NAMES = ("cfo_trend", "revenue_trend")


@dataclass
class FundamentalMomentumSignalBlock:
    """Raw, standardized, and combined fundamental-momentum signals."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    fundamental_momentum_block: pd.Series
    diagnostics: pd.DataFrame
    quarters: int
    minimum_observations: int


def _quarterly_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one point-in-time row per company fiscal quarter from the panel."""
    fields = [
        "gvkey",
        "datadate",
        "available_date",
        "fyearq",
        "fqtr",
        "oancf_qtr",
        "revtq",
    ]
    missing = [column for column in fields if column not in panel.columns]
    if missing:
        raise KeyError(
            f"Panel is missing quarterly fundamental-momentum inputs: {missing}"
        )
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
    step_is_one = quarters.groupby("gvkey", sort=False)["_fiscal_quarter"].diff().eq(1)
    sequence_break = ~step_is_one.fillna(False)
    quarters["_consecutive_run"] = sequence_break.groupby(
        quarters["gvkey"], sort=False
    ).cumsum()
    return quarters


def _rolling_scaled_trend(
    quarters: pd.DataFrame,
    source: str,
    window: int,
    minimum_observations: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return slopes, counts, zero-scale flags, and latest source availability."""
    values = pd.to_numeric(quarters[source], errors="coerce")
    fiscal_quarter = pd.to_numeric(
        quarters["_fiscal_quarter"], errors="coerce"
    )
    valid = values.notna() & fiscal_quarter.notna()
    sufficient_statistics = pd.DataFrame(
        {
            "n": valid.astype("float64"),
            "x": fiscal_quarter.where(valid, 0.0),
            "y": values.where(valid, 0.0),
            "xx": fiscal_quarter.pow(2).where(valid, 0.0),
            "xy": fiscal_quarter.mul(values).where(valid, 0.0),
            "abs_y": values.abs().where(valid, 0.0),
        },
        index=quarters.index,
    )
    groups = [quarters["gvkey"], quarters["_consecutive_run"]]
    rolling = (
        sufficient_statistics.groupby(groups, sort=False)
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .reindex(quarters.index)
    )

    observations = rolling["n"]
    safe_count = observations.where(observations.gt(0), 1.0)
    centered_xx = rolling["xx"] - rolling["x"].pow(2) / safe_count
    centered_xy = rolling["xy"] - rolling["x"] * rolling["y"] / safe_count
    mean_absolute_level = rolling["abs_y"] / safe_count
    sufficient = observations.ge(minimum_observations)
    positive_scale = mean_absolute_level.gt(0)
    estimable = (
        sufficient
        & positive_scale
        & centered_xx.gt(np.finfo("float64").eps)
    )
    slope = centered_xy / centered_xx
    trend = (slope / mean_absolute_level).where(estimable)
    zero_scale = sufficient & ~positive_scale
    available_ns = quarters["available_date"].astype("int64").where(valid)
    history_available = pd.to_datetime(
        available_ns.groupby(groups, sort=False)
        .rolling(window, min_periods=1)
        .max()
        .reset_index(level=[0, 1], drop=True)
        .reindex(quarters.index)
    )
    return trend, observations, zero_scale, history_available


def _quarterly_trends(
    panel: pd.DataFrame,
    quarters: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """Build scaled CFO and revenue trends within consecutive-quarter runs."""
    if quarters < 2:
        raise ValueError("A trend window must contain at least two quarters.")
    if minimum_observations < 2 or minimum_observations > quarters:
        raise ValueError(
            "minimum_observations must be between 2 and the trend window."
        )
    history = _quarterly_history(panel)
    for source, output in {
        "oancf_qtr": "cfo_trend",
        "revtq": "revenue_trend",
    }.items():
        trend, observations, zero_scale, history_available = _rolling_scaled_trend(
            history,
            source,
            quarters,
            minimum_observations,
        )
        history[output] = trend
        history[f"_{output}_observations"] = observations
        history[f"_{output}_zero_scale"] = zero_scale
        history[f"_{output}_available_date"] = history_available
    return history


def raw_fundamental_momentum_signals(
    panel: pd.DataFrame,
    quarters: int = FUNDMOM_QUARTERS,
    minimum_observations: int = FUNDMOM_MIN_QUARTERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct both raw trends and report missing-window diagnostics.

    Args:
        panel: Monthly point-in-time research panel.
        quarters: Quarters used by this operation.
        minimum_observations: Minimum observations used by this operation.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame]``.
    """
    history = _quarterly_trends(panel, quarters, minimum_observations)
    indexed = history.set_index(["gvkey", "datadate"])
    keys = pd.MultiIndex.from_arrays(
        [
            panel["gvkey"].astype("string"),
            pd.to_datetime(panel["datadate"]),
        ],
        names=["gvkey", "datadate"],
    )
    raw = pd.DataFrame(index=panel.index, dtype="float64")
    panel_date = pd.to_datetime(panel["date"])
    for name in FUNDMOM_SIGNAL_NAMES:
        raw[name] = indexed[name].reindex(keys).to_numpy()
        history_available = pd.Series(
            indexed[f"_{name}_available_date"].reindex(keys).to_numpy(),
            index=panel.index,
        )
        # Do not expose a trend until every quarterly observation used by its
        # rolling regression was public, including unusually late prior filings.
        raw[name] = raw[name].where(history_available.le(panel_date))
    raw = raw.replace([np.inf, -np.inf], np.nan)

    diagnostic_rows: list[dict[str, object]] = []
    quarter_count = len(history)
    for name in FUNDMOM_SIGNAL_NAMES:
        observation_count = history[f"_{name}_observations"]
        insufficient = observation_count.lt(minimum_observations)
        zero_scale = history[f"_{name}_zero_scale"].fillna(False)
        for label, condition in (
            (f"company-quarters with < {minimum_observations} observations: {name}", insufficient),
            (f"company-quarters with zero mean absolute level: {name}", zero_scale),
        ):
            diagnostic_rows.append(
                {
                    "diagnostic": label,
                    "count": int(condition.sum()),
                    "share": float(condition.sum() / quarter_count)
                    if quarter_count
                    else np.nan,
                }
            )

    investable = panel["investable"].fillna(False).astype(bool)
    investable_count = int(investable.sum())
    for name in FUNDMOM_SIGNAL_NAMES:
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


def cfo_trend(panel: pd.DataFrame) -> pd.Series:
    """Return the scaled trend in point-in-time quarterly operating cash flow.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_fundamental_momentum_signals(panel)
    return raw["cfo_trend"]


def revenue_trend(panel: pd.DataFrame) -> pd.Series:
    """Return the scaled trend in point-in-time quarterly revenue.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _ = raw_fundamental_momentum_signals(panel)
    return raw["revenue_trend"]


def build_fundamental_momentum_block(
    panel: pd.DataFrame,
    quarters: int = FUNDMOM_QUARTERS,
    minimum_observations: int = FUNDMOM_MIN_QUARTERS,
) -> FundamentalMomentumSignalBlock:
    """Standardize improving fundamentals and combine available trends equally.

    Args:
        panel: Monthly point-in-time research panel.
        quarters: Quarters used by this operation.
        minimum_observations: Minimum observations used by this operation.

    Returns:
        Computed result as ``FundamentalMomentumSignalBlock``.
    """
    raw, diagnostics = raw_fundamental_momentum_signals(
        panel,
        quarters=quarters,
        minimum_observations=minimum_observations,
    )
    standardized = pd.DataFrame(index=panel.index, dtype="float64")
    for name in FUNDMOM_SIGNAL_NAMES:
        # Improving fundamentals are positive, so no sign reversal is applied.
        standardized[name] = standardize_signal(panel, raw[name])["signal"]
    available_average = standardized.mean(axis=1, skipna=True).where(
        standardized.notna().any(axis=1)
    )
    block = restandardize_monthly(panel, available_average).rename(
        "fundamental_momentum_block"
    )
    return FundamentalMomentumSignalBlock(
        raw=raw,
        standardized=standardized,
        fundamental_momentum_block=block,
        diagnostics=diagnostics,
        quarters=quarters,
        minimum_observations=minimum_observations,
    )


def build_sensitivity_blocks(
    panel: pd.DataFrame,
) -> dict[int, FundamentalMomentumSignalBlock]:
    """Build fixed robustness windows without changing the production block.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``dict[int, FundamentalMomentumSignalBlock]``.
    """
    # The 4- and 8-quarter results are robustness checks, not candidates for
    # selection. Production remains the configured 12-quarter window even if a
    # shorter window scores better: a common 12-quarter horizon with quality is
    # preferable to tuning each block to this sample. Because an 8-observation
    # minimum is impossible in four quarters, each shorter diagnostic requires
    # every quarter in its window (4 of 4 and 8 of 8).
    return {
        window: build_fundamental_momentum_block(
            panel,
            quarters=window,
            minimum_observations=min(FUNDMOM_MIN_QUARTERS, window),
        )
        for window in (4, 8)
    }
