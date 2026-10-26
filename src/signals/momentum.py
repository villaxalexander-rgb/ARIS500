"""Define the production 12-1 price-momentum signal.

Reads an in-memory monthly panel and writes nothing. The signal compounds the
11 returns from t-11 through t-1, excludes month t, and requires consecutive
months; higher past return is positive, with no sector exclusion. Evaluation
found a positive but period/universe-sensitive effect weaker than standard UMD;
it remains one of five model blocks, blended 70/30 universe/sector to retain
some industry momentum without letting it dominate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals.standardize import standardize_signal

MOMENTUM_SIGNAL_NAMES = ("momentum_12_1",)


@dataclass
class MomentumSignalBlock:
    """Raw, standardized, and combined momentum signal values."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    momentum_block: pd.Series


def momentum_12_1(panel: pd.DataFrame) -> pd.Series:
    """Return compounded returns from t-11 through t-1 for each security.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    work = panel.loc[:, ["permno", "date", "ret"]].copy()
    work = work.assign(
        _source_row=np.arange(len(work), dtype=np.int64),
        date=pd.to_datetime(work["date"]),
        ret=pd.to_numeric(work["ret"], errors="coerce"),
    ).sort_values(["permno", "date"], ignore_index=True)
    work = work.assign(
        _month_number=work["date"].dt.year * 12 + work["date"].dt.month
    )
    groups = work["permno"]
    lagged_return = work.groupby("permno", sort=False)["ret"].shift(1)
    lagged_month = work.groupby("permno", sort=False)["_month_number"].shift(1)

    compounded = (
        (1.0 + lagged_return)
        .groupby(groups, sort=False)
        .rolling(11, min_periods=11)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    first_month = (
        lagged_month.groupby(groups, sort=False)
        .rolling(11, min_periods=11)
        .min()
        .reset_index(level=0, drop=True)
    )
    last_month = (
        lagged_month.groupby(groups, sort=False)
        .rolling(11, min_periods=11)
        .max()
        .reset_index(level=0, drop=True)
    )
    consecutive = first_month.eq(work["_month_number"] - 11) & last_month.eq(
        work["_month_number"] - 1
    )
    work = work.assign(_momentum=compounded.where(consecutive))

    output = pd.Series(np.nan, index=np.arange(len(panel)), dtype="float64")
    output.iloc[work["_source_row"].to_numpy()] = work["_momentum"].to_numpy()
    output.index = panel.index
    output.name = "momentum_12_1"
    return output


def build_momentum_block(panel: pd.DataFrame) -> MomentumSignalBlock:
    """Build the standardized one-signal production momentum block.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Raw and standardized momentum plus the production block score.
    """
    raw_signal = momentum_12_1(panel)
    standardized_signal = standardize_signal(panel, raw_signal)["signal"].rename(
        "momentum_12_1"
    )
    return MomentumSignalBlock(
        raw=raw_signal.to_frame(),
        standardized=standardized_signal.to_frame(),
        # A one-signal block is already monthly standardized.  Do not apply a
        # second transformation: this preserves the exact production result.
        momentum_block=standardized_signal.rename("momentum_block"),
    )


__all__ = [
    "MOMENTUM_SIGNAL_NAMES",
    "MomentumSignalBlock",
    "build_momentum_block",
    "momentum_12_1",
    "standardize_signal",
]
