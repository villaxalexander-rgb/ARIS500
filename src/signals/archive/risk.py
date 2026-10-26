"""Define tested trailing risk signals; this block is NOT in the model.

Reads monthly stock returns from an in-memory panel and the neutral beta
estimator in ``src/signals/beta.py``; writes nothing. Annualized total volatility,
market beta, and annualized idiosyncratic volatility use 24-36 consecutive
months and are sign-flipped so lower risk scores higher; no sector is excluded.
Evaluation produced inverted quintiles and a negative spread, so the entire
risk block was rejected and remains only as a documented research record.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import RISK_MIN_MONTHS
from src.signals.beta import rolling_risk_measures
from src.signals.standardize import restandardize_monthly, standardize_signal


RISK_SIGNAL_NAMES = ("return_volatility", "market_beta", "idio_volatility")


@dataclass
class RiskSignalBlock:
    """Raw, standardized, and combined risk signals plus diagnostics."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    risk_block: pd.Series
    observations: pd.DataFrame
    diagnostics: pd.DataFrame


def raw_risk_signals(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct signed risk signals, observation counts, and diagnostics.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]``.
    """
    measures, observations = rolling_risk_measures(panel)

    # Low risk is the positive signal, so reverse all three measures before
    # standardization: a high score means low volatility or beta. The analogous
    # asset-turnover-volatility flip failed in the quality sample, and low-vol
    # has been weak since roughly 2020, so inspect each evaluation before using
    # the block.
    raw = -measures
    raw = raw.replace([np.inf, -np.inf], np.nan)

    investable = panel["investable"].fillna(False).astype(bool)
    investable_count = int(investable.sum())
    diagnostic_rows: list[dict[str, object]] = []
    for name in RISK_SIGNAL_NAMES:
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
    insufficient = observations["regression_observations"].lt(RISK_MIN_MONTHS)
    diagnostic_rows.append(
        {
            "diagnostic": "rows with fewer than minimum regression observations",
            "count": int(insufficient.sum()),
            "share": float(insufficient.mean()),
        }
    )
    return raw, observations, pd.DataFrame(diagnostic_rows)


def return_volatility(panel: pd.DataFrame) -> pd.Series:
    """Return negative annualized trailing realized return volatility.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _, _ = raw_risk_signals(panel)
    return raw["return_volatility"]


def market_beta(panel: pd.DataFrame) -> pd.Series:
    """Return negative trailing market beta from excess-return regressions.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _, _ = raw_risk_signals(panel)
    return raw["market_beta"]


def idio_volatility(panel: pd.DataFrame) -> pd.Series:
    """Return negative annualized residual volatility from the beta regression.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    raw, _, _ = raw_risk_signals(panel)
    return raw["idio_volatility"]


def build_risk_block(panel: pd.DataFrame) -> RiskSignalBlock:
    """Standardize the three risk signals and combine available inputs equally.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``RiskSignalBlock``.
    """
    raw, observations, diagnostics = raw_risk_signals(panel)
    standardized = pd.DataFrame(index=panel.index, dtype="float64")
    for name in RISK_SIGNAL_NAMES:
        standardized[name] = standardize_signal(panel, raw[name])["signal"]

    # Total and idiosyncratic volatility remain separate pending the requested
    # empirical redundancy check, even though high correlation is expected.
    average = standardized.mean(axis=1, skipna=True).where(
        standardized.notna().any(axis=1)
    )
    block = restandardize_monthly(panel, average).rename("risk_block")
    return RiskSignalBlock(
        raw=raw,
        standardized=standardized,
        risk_block=block,
        observations=observations,
        diagnostics=diagnostics,
    )
