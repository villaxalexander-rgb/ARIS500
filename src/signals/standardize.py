"""Apply the single shared cross-sectional transformation used by all signals.

Reads an in-memory panel and raw signal and writes nothing. Each month it
winsorizes investable names, computes universe and point-in-time-sector z-scores,
falls back to the universe score for small sectors, blends them with the
configured 70/30 weights, and re-standardizes. Keeping this implementation in
one module prevents blocks from drifting into incomparable conventions.
"""

import warnings

import numpy as np
import pandas as pd

from config import MIN_SECTOR_SIZE, UNIVERSE_Z_WEIGHT, WINSOR_PCT

# pandas 2.x can emit this warning inside the safe DataFrame.assign method.
warnings.filterwarnings("ignore", message=r"ChainedAssignmentError.*")


def _population_zscore(
    values: pd.Series,
    groups: pd.Series | list[pd.Series],
) -> pd.Series:
    """Return within-group population z-scores, leaving zero-variance groups null."""
    grouped = values.groupby(groups, dropna=False)
    mean = grouped.transform("mean")
    standard_deviation = grouped.transform(lambda group: group.std(ddof=0))
    return ((values - mean) / standard_deviation).where(standard_deviation.gt(0))


def standardize_signal(
    panel: pd.DataFrame,
    raw_signal: pd.Series,
) -> pd.DataFrame:
    """Winsorize and blend universe and sector z-scores by month.

    Args:
        panel: Monthly point-in-time research panel.
        raw_signal: Row-aligned raw signal values.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if len(raw_signal) != len(panel):
        raise ValueError(
            f"Signal has {len(raw_signal):,} rows but panel has {len(panel):,}."
        )
    raw = pd.Series(raw_signal, index=panel.index, dtype="float64")
    work = panel.loc[:, ["date", "gsector", "investable"]].copy()
    work = work.assign(
        date=pd.to_datetime(work["date"]),
        _raw=raw,
    )
    eligible = work["investable"] & work["_raw"].notna()
    eligible_rows = work.loc[eligible].copy()

    monthly = eligible_rows.groupby("date")["_raw"]
    lower = monthly.transform("quantile", q=WINSOR_PCT)
    upper = monthly.transform("quantile", q=1.0 - WINSOR_PCT)
    eligible_rows = eligible_rows.assign(
        _winsorized=eligible_rows["_raw"].clip(lower=lower, upper=upper)
    )
    eligible_rows = eligible_rows.assign(
        _universe_z=_population_zscore(
            eligible_rows["_winsorized"],
            eligible_rows["date"],
        )
    )

    sector_groups = [eligible_rows["date"], eligible_rows["gsector"]]
    sector_count = eligible_rows["_winsorized"].groupby(
        sector_groups,
        dropna=False,
    ).transform("count")
    sector_z = _population_zscore(
        eligible_rows["_winsorized"],
        sector_groups,
    )
    use_sector = sector_count.ge(MIN_SECTOR_SIZE) & sector_z.notna()
    eligible_rows = eligible_rows.assign(
        _sector_z=sector_z.where(use_sector, eligible_rows["_universe_z"])
    )
    preliminary = (
        UNIVERSE_Z_WEIGHT * eligible_rows["_universe_z"]
        + (1.0 - UNIVERSE_Z_WEIGHT) * eligible_rows["_sector_z"]
    )
    eligible_rows = eligible_rows.assign(
        _signal=_population_zscore(preliminary, eligible_rows["date"])
    )

    output = pd.DataFrame(
        {
            "raw": np.nan,
            "winsorized": np.nan,
            "universe_z": np.nan,
            "sector_z": np.nan,
            "signal": np.nan,
        },
        index=panel.index,
        dtype="float64",
    )
    output.loc[eligible_rows.index, "raw"] = eligible_rows["_raw"]
    output.loc[eligible_rows.index, "winsorized"] = eligible_rows["_winsorized"]
    output.loc[eligible_rows.index, "universe_z"] = eligible_rows["_universe_z"]
    output.loc[eligible_rows.index, "sector_z"] = eligible_rows["_sector_z"]
    output.loc[eligible_rows.index, "signal"] = eligible_rows["_signal"]
    return output


def restandardize_monthly(
    panel: pd.DataFrame,
    score: pd.Series,
) -> pd.Series:
    """Re-standardize a pre-combined score within month among investable names.

    Args:
        panel: Monthly point-in-time research panel.
        score: Row-aligned score values.

    Returns:
        Row-aligned Series containing the values described above.
    """
    if len(score) != len(panel):
        raise ValueError(f"Score has {len(score):,} rows but panel has {len(panel):,}.")
    values = pd.Series(score, index=panel.index, dtype="float64")
    dates = pd.to_datetime(panel["date"])
    eligible = panel["investable"] & values.notna()
    output = pd.Series(np.nan, index=panel.index, dtype="float64", name=score.name)
    output.loc[eligible] = _population_zscore(
        values.loc[eligible],
        dates.loc[eligible],
    )
    return output
