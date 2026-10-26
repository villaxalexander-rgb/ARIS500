"""Define the production short-interest sentiment signal.

Reads point-in-time short-interest and CIZ share data from an in-memory panel
and writes nothing. The raw ratio is split-adjusted short shares divided by
split-adjusted shares outstanding; it is sign-flipped because lower short
interest is the positive characteristic. No sector is excluded, but missing
reports remain missing rather than being treated as zero. Evaluation retained
this one-signal sentiment block in the five-block production composite.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signals.standardize import restandardize_monthly, standardize_signal


SENTIMENT_SIGNAL_NAMES = ("short_interest_ratio",)


@dataclass
class SentimentSignalBlock:
    """Raw, standardized, and combined sentiment signals plus diagnostics."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    sentiment_block: pd.Series
    diagnostics: pd.DataFrame


def short_interest_ratio(panel: pd.DataFrame) -> pd.Series:
    """Return split-adjusted short interest divided by shares outstanding.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    required = [
        "short_interest_adjusted_shares",
        "shares_out",
        "mthcumfacshr",
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing short-interest inputs: {missing}")
    short_shares = pd.to_numeric(
        panel["short_interest_adjusted_shares"], errors="coerce"
    )
    crsp_shares = pd.to_numeric(panel["shares_out"], errors="coerce")
    share_factor = pd.to_numeric(panel["mthcumfacshr"], errors="coerce")

    # Both sides are in millions of shares in panel_russell3000.parquet. CIZ's
    # cumulative share factor is inverted relative to legacy CRSP conventions:
    # shares_out * mthcumfacshr is split-adjusted, not shares_out / factor.
    adjusted_shares_outstanding = crsp_shares * share_factor
    valid = short_shares.ge(0) & adjusted_shares_outstanding.gt(0)
    ratio = (short_shares / adjusted_shares_outstanding).where(valid)
    return ratio.replace([np.inf, -np.inf], np.nan).rename("short_interest_ratio")


def build_sentiment_block(panel: pd.DataFrame) -> SentimentSignalBlock:
    """Standardize signed short interest and form the one-signal block.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``SentimentSignalBlock``.
    """
    ratio = short_interest_ratio(panel)
    raw = ratio.to_frame()

    # High short interest is expected to predict lower returns, so reverse the
    # ratio before standardization: high means low short interest. Two earlier
    # sign choices failed in this universe—quality's asset_turnover_vol and the
    # entire risk block—so this individual evaluation must be checked before
    # trusting the sign or adding the signal to a production composite.
    standardized = pd.DataFrame(index=panel.index, dtype="float64")
    standardized["short_interest_ratio"] = standardize_signal(
        panel,
        -ratio,
    )["signal"]
    # Re-standardize like every other block, even though a single-signal
    # block leaves this mathematically equivalent to the input: consistency
    # with the shared convention matters more than skipping one operation.
    block = restandardize_monthly(
        panel,
        standardized["short_interest_ratio"],
    ).rename("sentiment_block")

    investable = panel["investable"].fillna(False).astype(bool)
    covered = investable & ratio.notna()
    investable_count = int(investable.sum())
    diagnostics = pd.DataFrame(
        [
            {
                "diagnostic": "investable coverage: short_interest_ratio",
                "count": int(covered.sum()),
                "share": (
                    float(covered.sum() / investable_count)
                    if investable_count
                    else np.nan
                ),
            },
            {
                "diagnostic": "rows with short interest but invalid adjusted shares",
                "count": int(
                    (
                        panel["short_interest_adjusted_shares"].notna()
                        & ratio.isna()
                    ).sum()
                ),
                "share": float(
                    (
                        panel["short_interest_adjusted_shares"].notna()
                        & ratio.isna()
                    ).mean()
                ),
            },
        ]
    )
    return SentimentSignalBlock(
        raw=raw,
        standardized=standardized,
        sentiment_block=block,
        diagnostics=diagnostics,
    )
