"""Define the production share issuance/buyback signals over three horizons.

Reads CRSP shares and CIZ ``mthcumfacshr`` from an in-memory panel and writes
nothing. It computes negative log changes in split-adjusted shares over 3, 12,
and 36 consecutive months, so buybacks score high and issuance scores low; no
sector is excluded. CIZ requires multiplication by the cumulative share factor,
not the legacy-style division—a wrong orientation turns splits into fake
issuance. All three nested horizons were intentionally retained after evaluation
as the production capital-deployment block.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    CAPITAL_DEPLOYMENT_HORIZONS,
    SPLIT_AUDIT_SAMPLE_SIZE,
    SPLIT_MIN_RATIO,
)
from src.signals.standardize import restandardize_monthly, standardize_signal


CAPITAL_DEPLOYMENT_SIGNAL_NAMES = tuple(
    f"share_chg_{months}m" for months in CAPITAL_DEPLOYMENT_HORIZONS
)


@dataclass
class CapitalDeploymentSignalBlock:
    """Raw, standardized, and combined capital deployment signals."""

    adjusted_shares: pd.Series
    raw: pd.DataFrame
    standardized: pd.DataFrame
    capital_deployment_block: pd.Series
    diagnostics: pd.DataFrame


def split_adjusted_shares(panel: pd.DataFrame) -> pd.Series:
    """Put CRSP shares on a comparable pre-split share-count basis.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    required = ["shares_out", "mthcumfacshr"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing CRSP share inputs: {missing}")
    shares = pd.to_numeric(panel["shares_out"], errors="coerce")
    factor = pd.to_numeric(panel["mthcumfacshr"], errors="coerce")

    # CIZ mthcumfacshr is oriented so the historical raw share count must be
    # multiplied by the factor. Empirically, Apple's Aug-2020 4-for-1 split
    # changed shares 4x while the factor fell from 4 to 1; multiplication stays
    # flat, whereas the often-quoted shares/factor formula creates a false 16x
    # issuance. The latter convention is therefore inverted for msf_v2 CIZ.
    adjusted = (shares * factor).where(shares.gt(0) & factor.gt(0))
    return adjusted.rename("adjusted_shares")


def _ordered_share_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Return row-aligned adjusted shares ordered by security and month."""
    required = ["permno", "date", "shares_out", "mthcumfacshr"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing capital deployment inputs: {missing}")
    history = panel.loc[:, required].copy()
    history = history.assign(
        _source_row=np.arange(len(history), dtype=np.int64),
        date=pd.to_datetime(history["date"]),
        adjusted_shares=split_adjusted_shares(panel),
    ).sort_values(["permno", "date"], ignore_index=True)
    history["_month_number"] = history["date"].dt.year * 12 + history["date"].dt.month
    return history


def raw_capital_deployment_signals(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construct split-adjusted share changes and coverage diagnostics.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.Series, pd.DataFrame]``.
    """
    history = _ordered_share_history(panel)
    groups = history.groupby("permno", sort=False)
    ordered_raw = pd.DataFrame(index=history.index, dtype="float64")

    for months, name in zip(
        CAPITAL_DEPLOYMENT_HORIZONS,
        CAPITAL_DEPLOYMENT_SIGNAL_NAMES,
        strict=True,
    ):
        lagged_shares = groups["adjusted_shares"].shift(months)
        lagged_month = groups["_month_number"].shift(months)
        consecutive = history["_month_number"].sub(lagged_month).eq(months)
        valid = consecutive & history["adjusted_shares"].gt(0) & lagged_shares.gt(0)

        # Issuance predicts lower returns and buybacks higher returns, so flip
        # every log share change before standardization: high means share-count
        # shrinkage. The analogous quality asset-turnover-volatility sign flip
        # failed empirically, so individual evaluations must validate this one.
        share_ratio = pd.Series(np.nan, index=history.index, dtype="float64")
        share_ratio.loc[valid] = (
            history.loc[valid, "adjusted_shares"] / lagged_shares.loc[valid]
        )
        ordered_raw[name] = -np.log(share_ratio)

    raw = pd.DataFrame(
        np.nan,
        index=panel.index,
        columns=CAPITAL_DEPLOYMENT_SIGNAL_NAMES,
        dtype="float64",
    )
    source_rows = history["_source_row"].to_numpy()
    for name in CAPITAL_DEPLOYMENT_SIGNAL_NAMES:
        raw.iloc[source_rows, raw.columns.get_loc(name)] = ordered_raw[name].to_numpy()

    adjusted = pd.Series(np.nan, index=panel.index, dtype="float64")
    adjusted.iloc[source_rows] = history["adjusted_shares"].to_numpy()
    adjusted = adjusted.rename("adjusted_shares")

    investable = panel["investable"].fillna(False).astype(bool)
    diagnostic_rows = [
        {
            "diagnostic": "rows with missing/nonpositive split-adjusted shares",
            "count": int(adjusted.isna().sum()),
            "share": float(adjusted.isna().mean()),
        }
    ]
    investable_count = int(investable.sum())
    for name in CAPITAL_DEPLOYMENT_SIGNAL_NAMES:
        available = investable & raw[name].notna()
        diagnostic_rows.append(
            {
                "diagnostic": f"investable coverage: {name}",
                "count": int(available.sum()),
                "share": (
                    float(available.sum() / investable_count)
                    if investable_count
                    else np.nan
                ),
            }
        )
    return raw, adjusted, pd.DataFrame(diagnostic_rows)


def _share_change(panel: pd.DataFrame, months: int) -> pd.Series:
    """Return one configured signed log share-change horizon."""
    if months not in CAPITAL_DEPLOYMENT_HORIZONS:
        raise ValueError(
            f"Unsupported horizon {months}; configured horizons are "
            f"{CAPITAL_DEPLOYMENT_HORIZONS}."
        )
    raw, _, _ = raw_capital_deployment_signals(panel)
    return raw[f"share_chg_{months}m"]


def share_chg_3m(panel: pd.DataFrame) -> pd.Series:
    """Return negative three-month log change in split-adjusted shares.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    return _share_change(panel, 3)


def share_chg_12m(panel: pd.DataFrame) -> pd.Series:
    """Return negative twelve-month log change in split-adjusted shares.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    return _share_change(panel, 12)


def share_chg_36m(panel: pd.DataFrame) -> pd.Series:
    """Return negative thirty-six-month log change in split-adjusted shares.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    return _share_change(panel, 36)


def split_adjustment_examples(panel: pd.DataFrame) -> pd.DataFrame:
    """Return three-month windows for ten clean 2-for-1-or-larger splits.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    required = [
        "permno",
        "date",
        "ticker",
        "company_name",
        "shares_out",
        "mthcumfacshr",
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing split-audit inputs: {missing}")
    history = panel.loc[:, required].copy()
    history = history.assign(
        date=pd.to_datetime(history["date"]),
        shares_out=pd.to_numeric(history["shares_out"], errors="coerce"),
        mthcumfacshr=pd.to_numeric(history["mthcumfacshr"], errors="coerce"),
    ).sort_values(["permno", "date"], ignore_index=True)
    history["adjusted_shares"] = (
        history["shares_out"] * history["mthcumfacshr"]
    ).where(history["shares_out"].gt(0) & history["mthcumfacshr"].gt(0))
    groups = history.groupby("permno", sort=False)
    previous_date = groups["date"].shift(1)
    previous_factor = groups["mthcumfacshr"].shift(1)
    previous_raw = groups["shares_out"].shift(1)
    previous_adjusted = groups["adjusted_shares"].shift(1)
    month_number = history["date"].dt.year * 12 + history["date"].dt.month
    previous_month = previous_date.dt.year * 12 + previous_date.dt.month
    history["split_ratio"] = previous_factor / history["mthcumfacshr"]
    history["raw_share_ratio"] = history["shares_out"] / previous_raw
    history["adjusted_share_ratio"] = (
        history["adjusted_shares"] / previous_adjusted
    )
    history["_consecutive"] = month_number.sub(previous_month).eq(1)

    candidates = history.loc[
        history["_consecutive"]
        & history["split_ratio"].ge(SPLIT_MIN_RATIO)
        & history["raw_share_ratio"].between(
            history["split_ratio"] * 0.70,
            history["split_ratio"] * 1.30,
        )
        & history["adjusted_share_ratio"].between(0.70, 1.30)
    ].copy()
    candidates["_adjustment_error"] = candidates["adjusted_share_ratio"].sub(1).abs()
    events = (
        candidates.sort_values(
            ["split_ratio", "_adjustment_error", "date", "permno"],
            ascending=[False, True, True, True],
        )
        .drop_duplicates("permno", keep="first")
        .drop_duplicates("company_name", keep="first")
        .head(SPLIT_AUDIT_SAMPLE_SIZE)
    )
    if len(events) < SPLIT_AUDIT_SAMPLE_SIZE:
        raise ValueError(
            f"Found only {len(events)} clean splits; expected "
            f"{SPLIT_AUDIT_SAMPLE_SIZE}."
        )

    windows: list[pd.DataFrame] = []
    for event in events.itertuples(index=True):
        start = max(0, event.Index - 1)
        stop = min(len(history), event.Index + 2)
        window = history.iloc[start:stop].copy()
        window = window.loc[window["permno"].eq(event.permno)]
        window["split_event_date"] = event.date
        window["event_split_ratio"] = event.split_ratio
        window["month_relative_to_split"] = (
            window["date"].dt.year * 12
            + window["date"].dt.month
            - (event.date.year * 12 + event.date.month)
        )
        windows.append(window)
    return pd.concat(windows, ignore_index=True).loc[
        :,
        [
            "permno",
            "ticker",
            "company_name",
            "split_event_date",
            "event_split_ratio",
            "month_relative_to_split",
            "date",
            "shares_out",
            "mthcumfacshr",
            "adjusted_shares",
        ],
    ]


def build_capital_deployment_block(
    panel: pd.DataFrame,
) -> CapitalDeploymentSignalBlock:
    """Standardize share changes and combine the three horizons equally.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``CapitalDeploymentSignalBlock``.
    """
    raw, adjusted, diagnostics = raw_capital_deployment_signals(panel)
    standardized = pd.DataFrame(index=panel.index, dtype="float64")
    for name in CAPITAL_DEPLOYMENT_SIGNAL_NAMES:
        standardized[name] = standardize_signal(panel, raw[name])["signal"]

    # These nested horizons intentionally express the same issuance/buyback
    # theme at different speeds. Their expected correlation is not collapsed.
    average = standardized.mean(axis=1, skipna=True).where(
        standardized.notna().any(axis=1)
    )
    block = restandardize_monthly(panel, average).rename(
        "capital_deployment_block"
    )
    return CapitalDeploymentSignalBlock(
        adjusted_shares=adjusted,
        raw=raw,
        standardized=standardized,
        capital_deployment_block=block,
        diagnostics=diagnostics,
    )
