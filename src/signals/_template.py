"""Template for a new point-in-time signal block; this is not a live signal.

Copy this file, rename every ``example`` symbol, and document the real panel
columns it reads; signal modules must never write files. The skeleton shows the
required row-as-is contract, higher-is-better sign convention, optional sector
exclusion, shared standardization, missing-component handling, and monthly block
re-standardization that keep new research comparable with existing blocks.
After implementing the file, add one ``SignalBlockSpec`` to
``src/signals/registry.py`` with its builder, constituents, and required panel
columns. Add its name to ``config.MODEL_BLOCKS`` only after a documented keep
decision; no composite or explainer edit should be necessary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals.standardize import restandardize_monthly, standardize_signal


# Give every raw component a stable, descriptive snake_case name. One file owns
# one coherent economic idea; do not mix unrelated signals merely to save files.
EXAMPLE_SIGNAL_NAMES = ("example_signal",)


@dataclass
class ExampleSignalBlock:
    """Hold raw components, standardized components, and the combined block."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    example_block: pd.Series


def example_signal(panel: pd.DataFrame) -> pd.Series:
    """Calculate one raw point-in-time signal.

    Args:
        panel: Monthly research panel whose fundamentals are already point-in-time.

    Returns:
        Row-aligned raw signal with higher values representing the preferred trait.
    """
    required = {"gsector", "example_numerator", "example_denominator"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Panel is missing example inputs: {sorted(missing)}")

    numerator = pd.to_numeric(panel["example_numerator"], errors="coerce")
    denominator = pd.to_numeric(panel["example_denominator"], errors="coerce")
    raw = (numerator / denominator).where(denominator.gt(0))

    # OPTIONAL SECTOR EXCLUSION PATTERN: exclude only when the accounting ratio
    # is economically incomparable, and explain that reason in the module docs.
    excluded_sector = pd.to_numeric(panel["gsector"], errors="coerce").isin(
        [40, 60]
    )
    raw = raw.mask(excluded_sector)

    # OPTIONAL SIGN-FLIP PATTERN: every standardized signal must point in the
    # same direction. Uncomment only when a lower raw value is economically
    # better (for example, volatility or short interest), and validate the sign
    # in the individual evaluation before trusting the block.
    lower_is_better = False
    if lower_is_better:
        raw = -raw

    # Never shift fundamentals here: build_panel.py already made each row
    # point-in-time. Never use shift(-n), future returns, backward fill, or a
    # full-sample statistic. Preserve unavailable observations as NaN.
    return raw.replace([np.inf, -np.inf], np.nan).rename("example_signal")


def build_example_block(panel: pd.DataFrame) -> ExampleSignalBlock:
    """Standardize raw components and combine them into one monthly block.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Raw signals, standardized signals, and a row-aligned block score.
    """
    raw = example_signal(panel).to_frame()
    standardized = pd.DataFrame(index=panel.index, dtype="float64")
    for name in EXAMPLE_SIGNAL_NAMES:
        # Always call the shared function. Do not copy winsorization or z-score
        # logic into a signal file; one implementation prevents silent drift.
        standardized[name] = standardize_signal(panel, raw[name])["signal"]

    # Average only the components available on each row. A missing component is
    # unknown, not neutral, so never replace it with zero before averaging.
    available_average = standardized.mean(axis=1, skipna=True).where(
        standardized.notna().any(axis=1)
    )
    block = restandardize_monthly(panel, available_average).rename(
        "example_block"
    )
    return ExampleSignalBlock(
        raw=raw,
        standardized=standardized,
        example_block=block,
    )


__all__ = ["EXAMPLE_SIGNAL_NAMES", "ExampleSignalBlock", "build_example_block"]
