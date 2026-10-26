"""Estimate trailing stock market beta for the live portfolio constraint.

Reads monthly stock returns from an in-memory panel and market/risk-free
returns from ``data/factors.parquet``; writes nothing.  Windows contain 24-36
consecutive calendar months, end at month t, and reset after gaps so stale
pre-gap observations cannot influence the live beta constraint.  This neutral
estimator is live even though the broader low-risk signal block was rejected.
"""

import numpy as np
import pandas as pd

from config import FACTORS_OUTPUT_PATH, RISK_MIN_MONTHS, RISK_VOL_MONTHS


RISK_MEASURE_NAMES = ("return_volatility", "market_beta", "idio_volatility")


def _load_monthly_factors() -> pd.DataFrame:
    """Load decimal market and risk-free returns keyed by calendar month."""
    if not FACTORS_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing local factor file: {FACTORS_OUTPUT_PATH}")
    factors = pd.read_parquet(
        FACTORS_OUTPUT_PATH,
        columns=["date", "mkt_excess", "rf"],
    )
    factors = factors.assign(
        date=pd.to_datetime(factors["date"]),
        mkt_excess=pd.to_numeric(factors["mkt_excess"], errors="coerce"),
        rf=pd.to_numeric(factors["rf"], errors="coerce"),
    )
    factors["_month"] = factors["date"].dt.to_period("M")
    duplicate_months = int(factors.duplicated("_month", keep=False).sum())
    if duplicate_months:
        raise ValueError(
            f"factors.parquet contains {duplicate_months:,} duplicate month rows."
        )
    return factors.loc[:, ["_month", "mkt_excess", "rf"]]


def _window_sum(
    values: np.ndarray,
    valid: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    """Return inclusive rolling sums using a possibly different start per row."""
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.where(valid, values, 0.0), dtype="float64"))
    )
    ends = np.arange(1, len(values) + 1)
    return cumulative[ends] - cumulative[starts]


def _window_count(valid: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Return inclusive rolling valid-observation counts."""
    cumulative = np.concatenate(([0], np.cumsum(valid, dtype="int32")))
    ends = np.arange(1, len(valid) + 1)
    return cumulative[ends] - cumulative[starts]


def _stock_rolling_statistics(group: pd.DataFrame) -> pd.DataFrame:
    """Estimate inclusive calendar-window volatility, beta, and residual risk."""
    months = group["_month_number"].to_numpy(dtype="int32", copy=False)
    returns = group["ret"].to_numpy(dtype="float64", copy=False)
    market = group["mkt_excess"].to_numpy(dtype="float64", copy=False)
    risk_free = group["rf"].to_numpy(dtype="float64", copy=False)
    # Month t's realized return is known at the month-t formation date.
    calendar_starts = np.searchsorted(
        months,
        months - RISK_VOL_MONTHS + 1,
        side="left",
    )
    # A missing calendar month starts a new run; never bridge a return gap.
    sequence_break = np.concatenate(([True], np.diff(months) != 1))
    run_starts = np.maximum.accumulate(
        np.where(sequence_break, np.arange(len(months)), 0)
    )
    starts = np.maximum(calendar_starts, run_starts)

    valid_return = np.isfinite(returns)
    return_count = _window_count(valid_return, starts)
    return_sum = _window_sum(returns, valid_return, starts)
    return_square_sum = _window_sum(returns * returns, valid_return, starts)
    centered_return_squares = return_square_sum - np.divide(
        return_sum * return_sum,
        return_count,
        out=np.zeros_like(return_sum),
        where=return_count > 0,
    )
    return_variance = np.full(len(group), np.nan, dtype="float64")
    valid_return_window = return_count >= RISK_MIN_MONTHS
    return_variance[valid_return_window] = np.maximum(
        centered_return_squares[valid_return_window]
        / (return_count[valid_return_window] - 1),
        0.0,
    )

    excess_return = returns - risk_free
    valid_regression = np.isfinite(excess_return) & np.isfinite(market)
    regression_count = _window_count(valid_regression, starts)
    sum_x = _window_sum(market, valid_regression, starts)
    sum_y = _window_sum(excess_return, valid_regression, starts)
    sum_xx = _window_sum(market * market, valid_regression, starts)
    sum_yy = _window_sum(excess_return * excess_return, valid_regression, starts)
    sum_xy = _window_sum(market * excess_return, valid_regression, starts)

    safe_count = np.where(regression_count > 0, regression_count, 1)
    centered_xx = sum_xx - sum_x * sum_x / safe_count
    centered_yy = sum_yy - sum_y * sum_y / safe_count
    centered_xy = sum_xy - sum_x * sum_y / safe_count
    valid_regression_window = (
        (regression_count >= RISK_MIN_MONTHS)
        & np.isfinite(centered_xx)
        & (centered_xx > np.finfo("float64").eps)
    )
    beta = np.full(len(group), np.nan, dtype="float64")
    beta[valid_regression_window] = (
        centered_xy[valid_regression_window]
        / centered_xx[valid_regression_window]
    )
    residual_sum_squares = centered_yy - beta * centered_xy
    residual_variance = np.full(len(group), np.nan, dtype="float64")
    residual_variance[valid_regression_window] = np.maximum(
        residual_sum_squares[valid_regression_window]
        / (regression_count[valid_regression_window] - 1),
        0.0,
    )

    return pd.DataFrame(
        {
            "return_observations": return_count,
            "regression_observations": regression_count,
            "return_volatility": np.sqrt(return_variance) * np.sqrt(12.0),
            "market_beta": beta,
            "idio_volatility": np.sqrt(residual_variance) * np.sqrt(12.0),
        },
        index=group.index,
    )


def rolling_risk_measures(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate trailing risk measures once and align them to panel rows.

    Args:
        panel: Monthly panel with ``permno``, ``date``, and ``ret``.

    Returns:
        A measures DataFrame and a matching observation-count DataFrame.
    """
    if RISK_VOL_MONTHS < RISK_MIN_MONTHS:
        raise ValueError("RISK_VOL_MONTHS must be at least RISK_MIN_MONTHS.")
    if RISK_MIN_MONTHS <= 2:
        raise ValueError("RISK_MIN_MONTHS must exceed two regression parameters.")
    required = ["permno", "date", "ret"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing risk inputs: {missing}")

    factors = _load_monthly_factors()
    work = panel.loc[:, required].copy()
    work = work.assign(
        _source_row=np.arange(len(work), dtype=np.int64),
        date=pd.to_datetime(work["date"]),
        ret=pd.to_numeric(work["ret"], errors="coerce"),
    )
    work["_month"] = work["date"].dt.to_period("M")
    work = work.merge(
        factors,
        on="_month",
        how="left",
        validate="many_to_one",
    ).sort_values(["permno", "_month"], ignore_index=True)
    if work.duplicated(["permno", "_month"]).any():
        raise ValueError("Panel contains duplicate (permno, calendar month) rows.")
    work["_month_number"] = work["_month"].astype("int64")

    pieces = [
        _stock_rolling_statistics(group)
        for _, group in work.groupby("permno", sort=False)
    ]
    ordered = pd.concat(pieces).sort_index()
    columns = [
        "return_volatility",
        "market_beta",
        "idio_volatility",
        "return_observations",
        "regression_observations",
    ]
    aligned = pd.DataFrame(np.nan, index=panel.index, columns=columns, dtype="float64")
    source_rows = work["_source_row"].to_numpy()
    aligned.iloc[source_rows] = ordered.loc[work.index, columns].to_numpy()
    return aligned.loc[:, list(RISK_MEASURE_NAMES)], aligned.loc[
        :, ["return_observations", "regression_observations"]
    ]


def market_beta(panel: pd.DataFrame) -> pd.Series:
    """Return the unflipped trailing market beta used by portfolio constraints.

    Args:
        panel: Monthly point-in-time panel.

    Returns:
        Row-aligned trailing beta Series.
    """
    measures, _ = rolling_risk_measures(panel)
    return measures["market_beta"].rename("market_beta")


__all__ = ["RISK_MEASURE_NAMES", "market_beta", "rolling_risk_measures"]
