"""Evaluate standardized signals against next-calendar-month stock returns.

Reads an in-memory panel (or the configured panel path) and writes no data
files; it returns tables and Matplotlib figures used by research notebooks.
Signal modules never look forward. This research module deliberately aligns
each signal with the same security's next calendar month through the shared
``forward_returns`` helper; portfolio backtest modules use that same helper.
The random/perfect-foresight harness protects the alignment from leakage.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from config import EVAL_START, N_QUANTILES, PANEL_OUTPUT_PATH
from src.signals.archive.fundamental_momentum import build_sensitivity_blocks
from src.signals.capital_deployment import split_adjustment_examples
from src.signals.registry import get_block_spec
from src.signals.standardize import standardize_signal
from src.utils import forward_returns


# Research resolves the same builders and constituent lists used by production.
_VALUE_SPEC = get_block_spec("value")
_QUALITY_SPEC = get_block_spec("quality")
_CAPITAL_SPEC = get_block_spec("capital_deployment")
_MOMENTUM_SPEC = get_block_spec("momentum")
_SENTIMENT_SPEC = get_block_spec("sentiment")
_RISK_SPEC = get_block_spec("risk")
_FUNDMOM_SPEC = get_block_spec("fundamental_momentum")

VALUE_SIGNAL_NAMES = _VALUE_SPEC.signal_names
QUALITY_SIGNAL_NAMES = _QUALITY_SPEC.signal_names
CAPITAL_DEPLOYMENT_SIGNAL_NAMES = _CAPITAL_SPEC.signal_names
RISK_SIGNAL_NAMES = _RISK_SPEC.signal_names
FUNDMOM_SIGNAL_NAMES = _FUNDMOM_SPEC.signal_names

build_value_block = _VALUE_SPEC.builder
build_quality_block = _QUALITY_SPEC.builder
build_capital_deployment_block = _CAPITAL_SPEC.builder
build_sentiment_block = _SENTIMENT_SPEC.builder
build_risk_block = _RISK_SPEC.builder
build_fundamental_momentum_block = _FUNDMOM_SPEC.builder
momentum_12_1 = _MOMENTUM_SPEC.signals[0].function


@dataclass
class EvaluationResult:
    """Tables and summary statistics from one signal evaluation."""

    name: str
    coverage: pd.DataFrame
    monthly_ic: pd.DataFrame
    ic_summary: pd.DataFrame
    ic_by_year: pd.DataFrame
    quantile_returns: pd.DataFrame
    quantile_average: pd.DataFrame
    spread_returns: pd.DataFrame
    spread_summary: pd.DataFrame
    turnover: pd.DataFrame


@dataclass
class ExtendedEvaluation:
    """Base evaluation plus value-weighted and size-segmented results."""

    base: EvaluationResult
    value_weighted_quantile_returns: pd.DataFrame
    value_weighted_quantile_average: pd.DataFrame
    value_weighted_spread_returns: pd.DataFrame
    value_weighted_spread_summary: pd.DataFrame
    size_summary: pd.DataFrame


def load_panel() -> pd.DataFrame:
    """Load the completed local research panel without modifying it.

    Args:
        None.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if not PANEL_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing research panel: {PANEL_OUTPUT_PATH}")
    return pd.read_parquet(PANEL_OUTPUT_PATH)


def _rank_correlation(group: pd.DataFrame) -> float:
    """Return Spearman correlation without relying on a future-looking join."""
    if len(group) < 3:
        return np.nan
    signal_rank = group["signal"].rank(method="average")
    return_rank = group["forward_return"].rank(method="average")
    if signal_rank.nunique() < 2 or return_rank.nunique() < 2:
        return np.nan
    return float(signal_rank.corr(return_rank))


def _summary_statistics(values: pd.Series) -> dict[str, float | int]:
    """Return mean, dispersion, t-statistic, and annualized information ratio."""
    clean = values.dropna().astype(float)
    count = len(clean)
    mean = float(clean.mean()) if count else np.nan
    standard_deviation = float(clean.std(ddof=1)) if count > 1 else np.nan
    t_statistic = (
        mean / (standard_deviation / np.sqrt(count))
        if count > 1 and standard_deviation > 0
        else np.nan
    )
    information_ratio = (
        mean / standard_deviation * np.sqrt(12.0)
        if count > 1 and standard_deviation > 0
        else np.nan
    )
    return {
        "months": count,
        "mean": mean,
        "std": standard_deviation,
        "t_stat": t_statistic,
        "annualized_ir": information_ratio,
    }


def _assign_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign equal-count monthly quantiles using deterministic rank tie-breaks."""
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < N_QUANTILES:
            continue
        ranks = group["signal"].rank(method="first")
        quantile = np.floor((ranks - 1.0) * N_QUANTILES / len(group)) + 1.0
        piece = group.copy()
        piece["quantile"] = quantile.astype("int8")
        pieces.append(piece)
    if not pieces:
        return frame.assign(quantile=pd.Series(dtype="int8")).iloc[0:0]
    return pd.concat(pieces, ignore_index=True)


def _maximum_drawdown(returns: pd.Series) -> float:
    """Return maximum drawdown from a monthly return series."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min()) if len(drawdown) else np.nan


def _top_quantile_turnover(assignments: pd.DataFrame) -> pd.DataFrame:
    """Return equal-weight target turnover for the top quantile each month."""
    top = assignments.loc[assignments["quantile"].eq(N_QUANTILES)]
    holdings = {
        date: set(group["permno"].astype(int))
        for date, group in top.groupby("date", sort=True)
    }
    rows: list[dict[str, object]] = []
    previous: set[int] | None = None
    for date in sorted(holdings):
        current = holdings[date]
        if previous is None or not previous or not current:
            turnover = np.nan
        else:
            overlap = len(previous & current)
            previous_weight = 1.0 / len(previous)
            current_weight = 1.0 / len(current)
            turnover = 0.5 * (
                overlap * abs(current_weight - previous_weight)
                + (len(previous) - overlap) * previous_weight
                + (len(current) - overlap) * current_weight
            )
        rows.append(
            {
                "date": date,
                "top_quantile_names": len(current),
                "turnover": turnover,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def evaluate_signal(
    panel: pd.DataFrame,
    standardized_signal: pd.Series,
    name: str,
) -> EvaluationResult:
    """Evaluate one standardized signal against next-calendar-month returns.

    Args:
        panel: Monthly point-in-time research panel.
        standardized_signal: Row-aligned standardized signal values.
        name: Stable cache or series name.

    Returns:
        Computed result as ``EvaluationResult``.
    """
    if len(standardized_signal) != len(panel):
        raise ValueError(
            f"Signal has {len(standardized_signal):,} rows but panel has "
            f"{len(panel):,}."
        )
    signal = pd.Series(standardized_signal, index=panel.index, dtype="float64")
    evaluation = panel.loc[:, ["permno", "date", "investable"]].copy()
    evaluation = evaluation.assign(
        date=pd.to_datetime(evaluation["date"]),
        signal=signal,
        forward_return=forward_returns(panel),
    )
    evaluation = evaluation.loc[
        evaluation["investable"]
        & evaluation["date"].ge(pd.Timestamp(EVAL_START))
    ].copy()

    coverage = evaluation.groupby("date", as_index=False).agg(
        investable_names=("permno", "size"),
        signal_names=("signal", "count"),
    )
    coverage = coverage.assign(
        coverage=coverage["signal_names"] / coverage["investable_names"]
    )

    valid = evaluation.dropna(subset=["signal", "forward_return"]).copy()
    monthly_ic = pd.DataFrame(
        [
            {"date": date, "rank_ic": _rank_correlation(group), "names": len(group)}
            for date, group in valid.groupby("date", sort=True)
        ]
    )
    ic_statistics = _summary_statistics(monthly_ic["rank_ic"])
    ic_summary = pd.DataFrame([ic_statistics])

    yearly_rows: list[dict[str, float | int]] = []
    for year, group in monthly_ic.assign(year=monthly_ic["date"].dt.year).groupby(
        "year", sort=True
    ):
        yearly_rows.append({"year": int(year), **_summary_statistics(group["rank_ic"])})
    ic_by_year = pd.DataFrame(yearly_rows)

    assignments = _assign_quantiles(valid)
    quantile_returns = (
        assignments.groupby(["date", "quantile"])["forward_return"]
        .mean()
        .unstack("quantile")
        .reindex(columns=range(1, N_QUANTILES + 1))
        .rename(columns=lambda quantile: f"Q{quantile}")
        .reset_index()
    )
    quantile_average = pd.DataFrame(
        {
            "quantile": [f"Q{quantile}" for quantile in range(1, N_QUANTILES + 1)],
            "average_monthly_return": [
                quantile_returns[f"Q{quantile}"].mean()
                for quantile in range(1, N_QUANTILES + 1)
            ],
        }
    )
    spread_returns = quantile_returns.loc[:, ["date"]].assign(
        spread=(quantile_returns[f"Q{N_QUANTILES}"] - quantile_returns["Q1"])
    )
    spread_returns = spread_returns.dropna(subset=["spread"]).assign(
        cumulative_spread=lambda frame: (1.0 + frame["spread"]).cumprod() - 1.0
    )
    spread = spread_returns["spread"]
    spread_mean = float(spread.mean()) if len(spread) else np.nan
    spread_volatility = (
        float(spread.std(ddof=1) * np.sqrt(12.0)) if len(spread) > 1 else np.nan
    )
    spread_sharpe = (
        float(spread.mean() / spread.std(ddof=1) * np.sqrt(12.0))
        if len(spread) > 1 and spread.std(ddof=1) > 0
        else np.nan
    )
    spread_t_stat = (
        float(spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread))))
        if len(spread) > 1 and spread.std(ddof=1) > 0
        else np.nan
    )
    spread_summary = pd.DataFrame(
        [
            {
                "months": len(spread),
                "average_monthly_spread": spread_mean,
                "annualized_volatility": spread_volatility,
                "annualized_sharpe": spread_sharpe,
                "t_stat": spread_t_stat,
                "maximum_drawdown": _maximum_drawdown(spread),
            }
        ]
    )
    turnover = _top_quantile_turnover(assignments)
    return EvaluationResult(
        name=name,
        coverage=coverage,
        monthly_ic=monthly_ic,
        ic_summary=ic_summary,
        ic_by_year=ic_by_year,
        quantile_returns=quantile_returns,
        quantile_average=quantile_average,
        spread_returns=spread_returns,
        spread_summary=spread_summary,
        turnover=turnover,
    )


def _spread_tables(quantile_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly and summary top-minus-bottom spread tables."""
    spread_returns = quantile_returns.loc[:, ["date"]].assign(
        spread=quantile_returns[f"Q{N_QUANTILES}"] - quantile_returns["Q1"]
    )
    spread_returns = spread_returns.dropna(subset=["spread"]).assign(
        cumulative_spread=lambda frame: (1.0 + frame["spread"]).cumprod() - 1.0
    )
    spread = spread_returns["spread"]
    standard_deviation = float(spread.std(ddof=1)) if len(spread) > 1 else np.nan
    summary = pd.DataFrame(
        [
            {
                "months": len(spread),
                "average_monthly_spread": float(spread.mean()) if len(spread) else np.nan,
                "annualized_volatility": standard_deviation * np.sqrt(12.0),
                "annualized_sharpe": (
                    float(spread.mean() / standard_deviation * np.sqrt(12.0))
                    if standard_deviation > 0
                    else np.nan
                ),
                "t_stat": (
                    float(spread.mean() / (standard_deviation / np.sqrt(len(spread))))
                    if standard_deviation > 0
                    else np.nan
                ),
                "maximum_drawdown": _maximum_drawdown(spread),
            }
        ]
    )
    return spread_returns, summary


def _evaluation_rows(
    panel: pd.DataFrame,
    standardized_signal: pd.Series,
) -> pd.DataFrame:
    """Return investable signal rows with next-month returns and current weights."""
    signal = pd.Series(standardized_signal, index=panel.index, dtype="float64")
    rows = panel.loc[:, ["permno", "date", "investable", "market_cap"]].copy()
    rows = rows.assign(
        date=pd.to_datetime(rows["date"]),
        market_cap=pd.to_numeric(rows["market_cap"], errors="coerce"),
        signal=signal,
        forward_return=forward_returns(panel),
    )
    return rows.loc[
        rows["investable"]
        & rows["date"].ge(pd.Timestamp(EVAL_START))
    ].copy()


def _value_weighted_evaluation(
    panel: pd.DataFrame,
    standardized_signal: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate quintiles using signal-month cap as the next month's return weight."""
    valid = _evaluation_rows(panel, standardized_signal).dropna(
        subset=["signal", "forward_return", "market_cap"]
    )
    valid = valid.loc[valid["market_cap"].gt(0)].copy()
    assignments = _assign_quantiles(valid)
    assignments["_weighted_return"] = (
        assignments["forward_return"] * assignments["market_cap"]
    )
    grouped = assignments.groupby(["date", "quantile"])
    weighted = grouped["_weighted_return"].sum() / grouped["market_cap"].sum()
    quantile_returns = (
        weighted.unstack("quantile")
        .reindex(columns=range(1, N_QUANTILES + 1))
        .rename(columns=lambda quantile: f"Q{quantile}")
        .reset_index()
    )
    quantile_average = pd.DataFrame(
        {
            "quantile": [f"Q{quantile}" for quantile in range(1, N_QUANTILES + 1)],
            "average_monthly_return": [
                quantile_returns[f"Q{quantile}"].mean()
                for quantile in range(1, N_QUANTILES + 1)
            ],
        }
    )
    spread_returns, spread_summary = _spread_tables(quantile_returns)
    return quantile_returns, quantile_average, spread_returns, spread_summary


def _size_group_evaluation(
    panel: pd.DataFrame,
    standardized_signal: pd.Series,
) -> pd.DataFrame:
    """Evaluate IC and equal-weight spreads within monthly prior-cap size groups."""
    rows = _evaluation_rows(panel, standardized_signal)
    eligible_cap = rows["market_cap"].gt(0)
    rows.loc[eligible_cap, "_size_rank"] = rows.loc[eligible_cap].groupby("date")[
        "market_cap"
    ].rank(method="first", ascending=False)
    rows["size_group"] = pd.NA
    rows.loc[rows["_size_rank"].le(500), "size_group"] = "largest_500"
    rows.loc[
        rows["_size_rank"].gt(500) & rows["_size_rank"].le(1500),
        "size_group",
    ] = "next_1000"
    rows.loc[rows["_size_rank"].gt(1500), "size_group"] = "smallest_1500"

    output: list[dict[str, object]] = []
    order = ("largest_500", "next_1000", "smallest_1500")
    for size_group in order:
        segment = rows.loc[rows["size_group"].eq(size_group)].copy()
        valid = segment.dropna(subset=["signal", "forward_return"])
        monthly_ic = pd.Series(
            {
                date: _rank_correlation(group)
                for date, group in valid.groupby("date", sort=True)
            },
            dtype="float64",
        )
        ic_statistics = _summary_statistics(monthly_ic)
        assignments = _assign_quantiles(valid)
        quantile_returns = (
            assignments.groupby(["date", "quantile"])["forward_return"]
            .mean()
            .unstack("quantile")
            .reindex(columns=range(1, N_QUANTILES + 1))
            .rename(columns=lambda quantile: f"Q{quantile}")
            .reset_index()
        )
        _, spread_summary = _spread_tables(quantile_returns)
        spread = spread_summary.iloc[0]
        output.append(
            {
                "size_group": size_group,
                "average_names_per_month": segment.groupby("date")["permno"]
                .size()
                .mean(),
                "average_signal_coverage": segment["signal"]
                .notna()
                .groupby(segment["date"])
                .mean()
                .mean(),
                "mean_rank_ic": ic_statistics["mean"],
                "ic_t_stat": ic_statistics["t_stat"],
                "ic_annualized_ir": ic_statistics["annualized_ir"],
                "average_monthly_spread": spread["average_monthly_spread"],
                "annualized_volatility": spread["annualized_volatility"],
                "annualized_sharpe": spread["annualized_sharpe"],
                "spread_t_stat": spread["t_stat"],
                "maximum_drawdown": spread["maximum_drawdown"],
            }
        )
    return pd.DataFrame(output)


def evaluate_signal_extended(
    panel: pd.DataFrame,
    standardized_signal: pd.Series,
    name: str,
) -> ExtendedEvaluation:
    """Run equal-weight, value-weight, and size-segmented signal evaluation.

    Args:
        panel: Monthly point-in-time research panel.
        standardized_signal: Row-aligned standardized signal values.
        name: Stable cache or series name.

    Returns:
        Computed result as ``ExtendedEvaluation``.
    """
    base = evaluate_signal(panel, standardized_signal, name)
    value_weighted = _value_weighted_evaluation(panel, standardized_signal)
    return ExtendedEvaluation(
        base=base,
        value_weighted_quantile_returns=value_weighted[0],
        value_weighted_quantile_average=value_weighted[1],
        value_weighted_spread_returns=value_weighted[2],
        value_weighted_spread_summary=value_weighted[3],
        size_summary=_size_group_evaluation(panel, standardized_signal),
    )


def _mean_monthly_correlation(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Return average monthly cross-sectional Pearson correlations."""
    work = signals.copy()
    work["date"] = pd.to_datetime(panel["date"])
    work["investable"] = panel["investable"].to_numpy()
    work = work.loc[
        work["investable"] & work["date"].ge(pd.Timestamp(EVAL_START))
    ].drop(columns="investable")
    columns = list(signals.columns)
    sums = pd.DataFrame(0.0, index=columns, columns=columns)
    counts = pd.DataFrame(0, index=columns, columns=columns)
    for _, group in work.groupby("date", sort=True):
        correlation = group[columns].corr(min_periods=3)
        valid = correlation.notna()
        sums = sums.add(correlation.fillna(0.0), fill_value=0.0)
        counts = counts.add(valid.astype(int), fill_value=0)
    return sums.divide(counts.where(counts.gt(0)))


def _format_percent_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a display copy with selected decimal columns formatted as percentages."""
    display = frame.copy()
    for column in columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    return display


def print_full_evaluation(result: EvaluationResult) -> None:
    """Print every requested diagnostic table for one evaluated signal.

    Args:
        result: Previously constructed result object.

    Returns:
        None.
    """
    print(f"\nFULL EVALUATION — {result.name}")
    print("Coverage by month:")
    print(
        _format_percent_table(result.coverage, ["coverage"]).to_string(index=False)
    )
    print("Rank IC summary:")
    print(result.ic_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Rank IC by year:")
    print(result.ic_by_year.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Average forward return by quantile:")
    print(
        _format_percent_table(
            result.quantile_average,
            ["average_monthly_return"],
        ).to_string(index=False)
    )
    print("Top-minus-bottom spread summary:")
    print(
        _format_percent_table(
            result.spread_summary,
            [
                "average_monthly_spread",
                "annualized_volatility",
                "maximum_drawdown",
            ],
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("Top-quintile turnover by month:")
    print(
        _format_percent_table(result.turnover, ["turnover"]).to_string(index=False)
    )


def print_extended_evaluation(result: ExtendedEvaluation) -> None:
    """Print base, value-weighted, and size-segmented diagnostics.

    Args:
        result: Previously constructed result object.

    Returns:
        None.
    """
    print_full_evaluation(result.base)
    print("Value-weighted average forward return by quantile (signal-month cap weights):")
    print(
        _format_percent_table(
            result.value_weighted_quantile_average,
            ["average_monthly_return"],
        ).to_string(index=False)
    )
    print("Value-weighted top-minus-bottom spread summary:")
    print(
        _format_percent_table(
            result.value_weighted_spread_summary,
            [
                "average_monthly_spread",
                "annualized_volatility",
                "maximum_drawdown",
            ],
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("Results by signal-month market-cap size group:")
    print(
        _format_percent_table(
            result.size_summary,
            [
                "average_signal_coverage",
                "average_monthly_spread",
                "annualized_volatility",
                "maximum_drawdown",
            ],
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def plot_cumulative_spread(result: EvaluationResult) -> plt.Figure:
    """Plot the cumulative equal-weighted top-minus-bottom spread.

    Args:
        result: Previously constructed result object.

    Returns:
        Matplotlib figure containing the requested chart.
    """
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        result.spread_returns["date"],
        result.spread_returns["cumulative_spread"],
        linewidth=1.5,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"{result.name}: cumulative Q{N_QUANTILES} minus Q1 return",
        xlabel="",
        ylabel="Cumulative return",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_value_spreads(
    results: dict[str, ExtendedEvaluation],
) -> plt.Figure:
    """Plot cumulative equal-weighted spreads for all value signals.

    Args:
        results: Results used by this operation.

    Returns:
        Matplotlib figure containing the requested chart.
    """
    figure, axis = plt.subplots(figsize=(10, 5))
    for name, result in results.items():
        spread = result.base.spread_returns
        axis.plot(spread["date"], spread["cumulative_spread"], label=name, linewidth=1.3)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"Value signals: cumulative Q{N_QUANTILES} minus Q1 return",
        xlabel="",
        ylabel="Cumulative return",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_quality_spreads(
    results: dict[str, ExtendedEvaluation],
) -> plt.Figure:
    """Plot cumulative equal-weighted spreads for all quality signals.

    Args:
        results: Results used by this operation.

    Returns:
        Matplotlib figure containing the requested chart.
    """
    figure, axis = plt.subplots(figsize=(11, 5))
    for name, result in results.items():
        spread = result.base.spread_returns
        axis.plot(spread["date"], spread["cumulative_spread"], label=name, linewidth=1.2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"Quality signals: cumulative Q{N_QUANTILES} minus Q1 return",
        xlabel="",
        ylabel="Cumulative return",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure


def plot_capital_deployment_spreads(
    results: dict[str, ExtendedEvaluation],
) -> plt.Figure:
    """Plot cumulative equal-weighted spreads for capital deployment signals.

    Args:
        results: Results used by this operation.

    Returns:
        Matplotlib figure containing the requested chart.
    """
    figure, axis = plt.subplots(figsize=(11, 5))
    for name, result in results.items():
        spread = result.base.spread_returns
        axis.plot(spread["date"], spread["cumulative_spread"], label=name, linewidth=1.2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"Capital deployment: cumulative Q{N_QUANTILES} minus Q1 return",
        xlabel="",
        ylabel="Cumulative return",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure


def plot_risk_spreads(
    results: dict[str, ExtendedEvaluation],
) -> plt.Figure:
    """Plot cumulative equal-weighted spreads for the risk signals.

    Args:
        results: Results used by this operation.

    Returns:
        Matplotlib figure containing the requested chart.
    """
    figure, axis = plt.subplots(figsize=(11, 5))
    for name, result in results.items():
        spread = result.base.spread_returns
        axis.plot(spread["date"], spread["cumulative_spread"], label=name, linewidth=1.2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"Risk signals: cumulative Q{N_QUANTILES} minus Q1 return",
        xlabel="",
        ylabel="Cumulative return",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure


def plot_fundamental_momentum_spreads(
    results: dict[str, ExtendedEvaluation],
) -> plt.Figure:
    """Plot cumulative equal-weighted spreads for fundamental momentum.

    Args:
        results: Results used by this operation.

    Returns:
        Matplotlib figure containing the requested chart.
    """
    figure, axis = plt.subplots(figsize=(11, 5))
    for name, result in results.items():
        spread = result.base.spread_returns
        axis.plot(spread["date"], spread["cumulative_spread"], label=name, linewidth=1.2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"Fundamental momentum: cumulative Q{N_QUANTILES} minus Q1 return",
        xlabel="",
        ylabel="Cumulative return",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def run_registered_block_research(
    block_name: str,
    panel: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Run the standard evaluation for any block registered in one place.

    Args:
        block_name: Name of a live or archived entry in ``BLOCK_REGISTRY``.
        panel: Optional in-memory research panel; the configured panel is loaded
            when omitted.

    Returns:
        The registered specification, built block, component and block
        evaluations, optional construction diagnostics, and spread figure.
    """

    spec = get_block_spec(block_name)
    research_panel = load_panel() if panel is None else panel.copy()
    missing = sorted(set(spec.panel_columns).difference(research_panel.columns))
    if missing:
        raise KeyError(
            f"Registered block {block_name!r} requires missing panel columns: {missing}"
        )

    built = spec.build(research_panel)
    standardized = getattr(built, "standardized", None)
    if not isinstance(standardized, pd.DataFrame):
        raise TypeError(
            f"Registered builder for {block_name!r} must return a result with "
            "a standardized DataFrame."
        )
    missing_signals = sorted(set(spec.signal_names).difference(standardized.columns))
    if missing_signals:
        raise KeyError(
            f"Registered block {block_name!r} did not build standardized signals: "
            f"{missing_signals}"
        )

    score = spec.score(built)
    evaluations: dict[str, ExtendedEvaluation] = {}
    for name in spec.signal_names:
        evaluation = evaluate_signal_extended(research_panel, standardized[name], name)
        evaluations[name] = evaluation
        print_extended_evaluation(evaluation)
    block_evaluation = evaluate_signal_extended(
        research_panel,
        score,
        spec.block_attribute,
    )
    evaluations[spec.block_attribute] = block_evaluation
    print_extended_evaluation(block_evaluation)

    diagnostics = getattr(built, "diagnostics", pd.DataFrame())
    if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
        print(f"\n{spec.label.upper()} CONSTRUCTION DIAGNOSTICS")
        print(diagnostics.to_string(index=False))
    figure = plot_cumulative_spread(block_evaluation.base)
    plt.show()
    return {
        "spec": spec,
        "block": built,
        "evaluations": evaluations,
        "diagnostics": diagnostics,
        "figure": figure,
    }


def run_harness_tests(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, EvaluationResult]]:
    """Run three random signals and one perfect-foresight alignment test.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``tuple[pd.DataFrame, dict[str, EvaluationResult]]``.
    """
    results: dict[str, EvaluationResult] = {}
    rows: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        raw = pd.Series(
            np.random.default_rng(seed).standard_normal(len(panel)),
            index=panel.index,
            name=f"random_seed_{seed}",
        )
        standardized = standardize_signal(panel, raw)
        result = evaluate_signal(panel, standardized["signal"], raw.name)
        results[raw.name] = result
        ic_t_stat = float(result.ic_summary.loc[0, "t_stat"])
        spread_sharpe = float(result.spread_summary.loc[0, "annualized_sharpe"])
        passed = abs(ic_t_stat) < 2.0 and abs(spread_sharpe) < 0.5
        rows.append(
            {
                "test": raw.name,
                "mean_rank_ic": result.ic_summary.loc[0, "mean"],
                "ic_t_stat": ic_t_stat,
                "spread_sharpe": spread_sharpe,
                "expected": "|IC t-stat| < 2 and |spread Sharpe| < 0.5",
                "pass/fail": "PASS" if passed else "FAIL",
            }
        )

    perfect_raw = forward_returns(panel).rename("perfect_foresight")
    perfect_standardized = standardize_signal(panel, perfect_raw)
    perfect = evaluate_signal(
        panel,
        perfect_standardized["signal"],
        perfect_raw.name,
    )
    results[perfect_raw.name] = perfect
    perfect_ic = float(perfect.ic_summary.loc[0, "mean"])
    rows.append(
        {
            "test": perfect_raw.name,
            "mean_rank_ic": perfect_ic,
            "ic_t_stat": perfect.ic_summary.loc[0, "t_stat"],
            "spread_sharpe": perfect.spread_summary.loc[0, "annualized_sharpe"],
            "expected": "mean rank IC >= 0.95",
            "pass/fail": "PASS" if perfect_ic >= 0.95 else "FAIL",
        }
    )
    return pd.DataFrame(rows), results


def run_signal_research() -> dict[str, object]:
    """Run harness tests first, then the complete momentum evaluation.

    Args:
        None.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    panel = load_panel()
    harness, harness_results = run_harness_tests(panel)
    print("HARNESS TESTS")
    print(harness.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    raw_momentum = momentum_12_1(panel)
    standardized_momentum = standardize_signal(panel, raw_momentum)
    momentum = evaluate_signal(
        panel,
        standardized_momentum["signal"],
        "momentum_12_1",
    )
    print_full_evaluation(momentum)
    figure = plot_cumulative_spread(momentum)
    plt.show()
    return {
        "harness": harness,
        "harness_results": harness_results,
        "momentum": momentum,
        "momentum_standardized": standardized_momentum,
        "figure": figure,
    }


def run_value_research() -> dict[str, object]:
    """Build and fully evaluate the three value signals and their block score.

    Args:
        None.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    panel = load_panel()
    value = build_value_block(panel)
    print("VALUE CONSTRUCTION DIAGNOSTICS")
    print(
        _format_percent_table(value.diagnostics, ["share"]).to_string(index=False)
    )

    signals = value.standardized.copy()
    signals["value_block"] = value.value_block
    evaluations: dict[str, ExtendedEvaluation] = {}
    for name in (*VALUE_SIGNAL_NAMES, "value_block"):
        result = evaluate_signal_extended(panel, signals[name], name)
        evaluations[name] = result
        print_extended_evaluation(result)

    raw_momentum = momentum_12_1(panel)
    momentum_standardized = standardize_signal(panel, raw_momentum)["signal"].rename(
        "momentum_12_1"
    )
    momentum = evaluate_signal(panel, momentum_standardized, "momentum_12_1")

    value_correlations = _mean_monthly_correlation(panel, value.standardized)
    block_momentum_correlation = _mean_monthly_correlation(
        panel,
        pd.concat([value.value_block, momentum_standardized], axis=1),
    )
    spread_comparison = evaluations["value_block"].base.spread_returns.loc[
        :, ["date", "spread"]
    ].merge(
        momentum.spread_returns.loc[:, ["date", "spread"]],
        on="date",
        how="inner",
        suffixes=("_value", "_momentum"),
        validate="one_to_one",
    )
    spread_correlation = float(
        spread_comparison["spread_value"].corr(spread_comparison["spread_momentum"])
    )

    print("\nCORRELATIONS")
    print("Average monthly cross-sectional correlation of standardized value signals:")
    print(value_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Average monthly cross-sectional correlation: value block vs momentum:")
    print(
        block_momentum_correlation.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print(
        "Equal-weight Q5-Q1 monthly spread correlation "
        f"(value block vs momentum): {spread_correlation:.4f} "
        f"over {len(spread_comparison):,} months"
    )

    figure = plot_value_spreads(evaluations)
    plt.show()
    return {
        "value": value,
        "evaluations": evaluations,
        "momentum": momentum,
        "momentum_standardized": momentum_standardized,
        "value_correlations": value_correlations,
        "block_momentum_correlation": block_momentum_correlation,
        "spread_comparison": spread_comparison,
        "spread_correlation": spread_correlation,
        "figure": figure,
    }


def run_quality_research() -> dict[str, object]:
    """Build and fully evaluate the six quality signals and their block score.

    Args:
        None.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    panel = load_panel()
    quality = build_quality_block(panel)
    print("QUALITY CONSTRUCTION DIAGNOSTICS")
    print(
        _format_percent_table(quality.diagnostics, ["share"]).to_string(index=False)
    )

    signals = quality.standardized.copy()
    signals["quality_block"] = quality.quality_block
    evaluations: dict[str, ExtendedEvaluation] = {}
    for name in (*QUALITY_SIGNAL_NAMES, "quality_block"):
        result = evaluate_signal_extended(panel, signals[name], name)
        evaluations[name] = result
        print_extended_evaluation(result)

    legacy_evaluation = evaluate_signal_extended(
        panel,
        quality.legacy_six_signal_block,
        "quality_block_old_six_signal",
    )
    block_comparison_rows: list[dict[str, object]] = []
    for label, result in (
        ("old_six_signal", legacy_evaluation),
        ("new_three_input", evaluations["quality_block"]),
    ):
        base = result.base
        block_comparison_rows.append(
            {
                "block": label,
                "mean_coverage": base.coverage["coverage"].mean(),
                "minimum_coverage": base.coverage["coverage"].min(),
                "mean_rank_ic": base.ic_summary.loc[0, "mean"],
                "ic_t_stat": base.ic_summary.loc[0, "t_stat"],
                "ic_annualized_ir": base.ic_summary.loc[0, "annualized_ir"],
                "ew_monthly_spread": base.spread_summary.loc[
                    0, "average_monthly_spread"
                ],
                "ew_annualized_sharpe": base.spread_summary.loc[
                    0, "annualized_sharpe"
                ],
                "vw_monthly_spread": result.value_weighted_spread_summary.loc[
                    0, "average_monthly_spread"
                ],
                "vw_annualized_sharpe": result.value_weighted_spread_summary.loc[
                    0, "annualized_sharpe"
                ],
                "top_quintile_turnover": base.turnover["turnover"].mean(),
            }
        )
    block_comparison = pd.DataFrame(block_comparison_rows)
    block_ic_by_year = legacy_evaluation.base.ic_by_year.loc[
        :, ["year", "mean"]
    ].rename(columns={"mean": "old_six_signal"}).merge(
        evaluations["quality_block"].base.ic_by_year.loc[:, ["year", "mean"]].rename(
            columns={"mean": "new_three_input"}
        ),
        on="year",
        how="outer",
        validate="one_to_one",
    )
    block_quintiles = legacy_evaluation.base.quantile_average.rename(
        columns={"average_monthly_return": "old_ew"}
    ).merge(
        evaluations["quality_block"].base.quantile_average.rename(
            columns={"average_monthly_return": "new_ew"}
        ),
        on="quantile",
        validate="one_to_one",
    ).merge(
        legacy_evaluation.value_weighted_quantile_average.rename(
            columns={"average_monthly_return": "old_vw"}
        ),
        on="quantile",
        validate="one_to_one",
    ).merge(
        evaluations["quality_block"].value_weighted_quantile_average.rename(
            columns={"average_monthly_return": "new_vw"}
        ),
        on="quantile",
        validate="one_to_one",
    )
    block_size_comparison = legacy_evaluation.size_summary.loc[
        :, ["size_group", "mean_rank_ic", "average_monthly_spread"]
    ].rename(
        columns={
            "mean_rank_ic": "old_ic",
            "average_monthly_spread": "old_spread",
        }
    ).merge(
        evaluations["quality_block"].size_summary.loc[
            :, ["size_group", "mean_rank_ic", "average_monthly_spread"]
        ].rename(
            columns={
                "mean_rank_ic": "new_ic",
                "average_monthly_spread": "new_spread",
            }
        ),
        on="size_group",
        validate="one_to_one",
    )
    print("\nOLD VERSUS REVISED QUALITY BLOCK")
    print(
        _format_percent_table(
            block_comparison,
            [
                "mean_coverage",
                "minimum_coverage",
                "ew_monthly_spread",
                "vw_monthly_spread",
                "top_quintile_turnover",
            ],
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("Block IC by year:")
    print(block_ic_by_year.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Average monthly block quintile returns:")
    print(
        _format_percent_table(
            block_quintiles,
            ["old_ew", "new_ew", "old_vw", "new_vw"],
        ).to_string(index=False)
    )
    print("Block results by size group:")
    print(
        _format_percent_table(
            block_size_comparison,
            ["old_spread", "new_spread"],
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    value = build_value_block(panel)
    raw_momentum = momentum_12_1(panel)
    momentum_standardized = standardize_signal(panel, raw_momentum)["signal"].rename(
        "momentum_12_1"
    )
    value_block = value.value_block.rename("value_block")
    block_correlations = _mean_monthly_correlation(
        panel,
        pd.concat(
            [quality.quality_block, value_block, momentum_standardized],
            axis=1,
        ),
    )
    quality_correlations = _mean_monthly_correlation(panel, quality.standardized)

    value_evaluation = evaluate_signal(panel, value_block, "value_block")
    momentum_evaluation = evaluate_signal(
        panel,
        momentum_standardized,
        "momentum_12_1",
    )
    spread_comparison = evaluations["quality_block"].base.spread_returns.loc[
        :, ["date", "spread"]
    ].rename(columns={"spread": "quality_block"})
    spread_comparison = spread_comparison.merge(
        value_evaluation.spread_returns.loc[:, ["date", "spread"]].rename(
            columns={"spread": "value_block"}
        ),
        on="date",
        how="inner",
        validate="one_to_one",
    ).merge(
        momentum_evaluation.spread_returns.loc[:, ["date", "spread"]].rename(
            columns={"spread": "momentum_12_1"}
        ),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    spread_correlations = spread_comparison.loc[
        :, ["quality_block", "value_block", "momentum_12_1"]
    ].corr()

    print("\nCORRELATIONS")
    print("Average monthly cross-sectional correlation of quality signals:")
    print(quality_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Average monthly cross-sectional block correlations:")
    print(block_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Equal-weight Q5-Q1 monthly spread correlations:")
    print(spread_correlations.to_string(float_format=lambda value: f"{value:.4f}"))

    figure = plot_quality_spreads(evaluations)
    plt.show()
    return {
        "quality": quality,
        "evaluations": evaluations,
        "legacy_evaluation": legacy_evaluation,
        "block_comparison": block_comparison,
        "block_ic_by_year": block_ic_by_year,
        "block_quintiles": block_quintiles,
        "block_size_comparison": block_size_comparison,
        "value": value,
        "momentum_standardized": momentum_standardized,
        "value_evaluation": value_evaluation,
        "momentum_evaluation": momentum_evaluation,
        "quality_correlations": quality_correlations,
        "block_correlations": block_correlations,
        "spread_comparison": spread_comparison,
        "spread_correlations": spread_correlations,
        "figure": figure,
    }


def run_capital_deployment_research() -> dict[str, object]:
    """Audit split adjustment, evaluate share changes, and compare all blocks.

    Args:
        None.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    panel = load_panel()

    split_examples = split_adjustment_examples(panel)
    split_event_rows = split_examples.loc[
        split_examples["month_relative_to_split"].eq(0)
    ]
    split_ratios = split_examples.sort_values(
        ["permno", "date"]
    ).groupby("permno", sort=False).apply(
        lambda group: pd.Series(
            {
                "raw_share_ratio": (
                    group.loc[group["month_relative_to_split"].eq(0), "shares_out"].iloc[0]
                    / group.loc[group["month_relative_to_split"].eq(-1), "shares_out"].iloc[0]
                ),
                "adjusted_share_ratio": (
                    group.loc[group["month_relative_to_split"].eq(0), "adjusted_shares"].iloc[0]
                    / group.loc[group["month_relative_to_split"].eq(-1), "adjusted_shares"].iloc[0]
                ),
            }
        ),
        include_groups=False,
    )
    print("SPLIT-ADJUSTMENT AUDIT — THREE MONTHS AROUND TEN EVENTS")
    print(split_examples.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "Split-month adjusted/raw summary: "
        f"{len(split_event_rows):,} events; "
        f"median split ratio {split_event_rows['event_split_ratio'].median():.2f}x; "
        f"median raw-share change {split_ratios['raw_share_ratio'].median():.4f}x; "
        f"median adjusted-share change "
        f"{split_ratios['adjusted_share_ratio'].median():.4f}x "
        f"(range {split_ratios['adjusted_share_ratio'].min():.4f}x to "
        f"{split_ratios['adjusted_share_ratio'].max():.4f}x). "
        "Adjusted shares use shares_out * mthcumfacshr because that is the "
        "empirically correct CIZ orientation."
    )

    capital = build_capital_deployment_block(panel)
    print("\nCAPITAL DEPLOYMENT CONSTRUCTION DIAGNOSTICS")
    print(_format_percent_table(capital.diagnostics, ["share"]).to_string(index=False))

    signals = capital.standardized.copy()
    signals["capital_deployment_block"] = capital.capital_deployment_block
    evaluations: dict[str, ExtendedEvaluation] = {}
    for name in (*CAPITAL_DEPLOYMENT_SIGNAL_NAMES, "capital_deployment_block"):
        result = evaluate_signal_extended(panel, signals[name], name)
        evaluations[name] = result
        print_extended_evaluation(result)

    value = build_value_block(panel)
    quality = build_quality_block(panel)
    momentum_standardized = standardize_signal(
        panel,
        momentum_12_1(panel),
    )["signal"].rename("momentum_12_1")
    comparison_blocks = pd.concat(
        [
            capital.capital_deployment_block,
            value.value_block,
            quality.quality_block,
            momentum_standardized,
        ],
        axis=1,
    )
    capital_correlations = _mean_monthly_correlation(
        panel,
        capital.standardized,
    )
    block_correlations = _mean_monthly_correlation(panel, comparison_blocks)

    comparison_evaluations = {
        "capital_deployment_block": evaluations["capital_deployment_block"].base,
        "value_block": evaluate_signal(panel, value.value_block, "value_block"),
        "quality_block": evaluate_signal(panel, quality.quality_block, "quality_block"),
        "momentum_12_1": evaluate_signal(
            panel,
            momentum_standardized,
            "momentum_12_1",
        ),
    }
    spread_comparison: pd.DataFrame | None = None
    for name, result in comparison_evaluations.items():
        spread = result.spread_returns.loc[:, ["date", "spread"]].rename(
            columns={"spread": name}
        )
        spread_comparison = (
            spread
            if spread_comparison is None
            else spread_comparison.merge(
                spread,
                on="date",
                how="inner",
                validate="one_to_one",
            )
        )
    if spread_comparison is None:
        raise RuntimeError("No block spread returns were available for correlation.")
    spread_correlations = spread_comparison.drop(columns="date").corr()

    print("\nCORRELATIONS")
    print("Average monthly cross-sectional correlations of capital deployment signals:")
    print(capital_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Average monthly cross-sectional block correlations:")
    print(block_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Equal-weight Q5-Q1 monthly spread correlations:")
    print(spread_correlations.to_string(float_format=lambda value: f"{value:.4f}"))

    figure = plot_capital_deployment_spreads(evaluations)
    plt.show()
    return {
        "split_examples": split_examples,
        "capital_deployment": capital,
        "evaluations": evaluations,
        "value": value,
        "quality": quality,
        "momentum_standardized": momentum_standardized,
        "comparison_evaluations": comparison_evaluations,
        "capital_correlations": capital_correlations,
        "block_correlations": block_correlations,
        "spread_comparison": spread_comparison,
        "spread_correlations": spread_correlations,
        "figure": figure,
    }


def run_risk_research() -> dict[str, object]:
    """Build, evaluate, and compare the trailing realized-risk signal block.

    Args:
        None.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    panel = load_panel()
    risk = build_risk_block(panel)
    print("RISK CONSTRUCTION DIAGNOSTICS")
    print(_format_percent_table(risk.diagnostics, ["share"]).to_string(index=False))

    signals = risk.standardized.copy()
    signals["risk_block"] = risk.risk_block
    evaluations: dict[str, ExtendedEvaluation] = {}
    for name in (*RISK_SIGNAL_NAMES, "risk_block"):
        result = evaluate_signal_extended(panel, signals[name], name)
        evaluations[name] = result
        print_extended_evaluation(result)

    value = build_value_block(panel)
    quality = build_quality_block(panel)
    capital = build_capital_deployment_block(panel)
    momentum_standardized = standardize_signal(
        panel,
        momentum_12_1(panel),
    )["signal"].rename("momentum_12_1")
    comparison_blocks = pd.concat(
        [
            risk.risk_block,
            value.value_block,
            quality.quality_block,
            momentum_standardized,
            capital.capital_deployment_block,
        ],
        axis=1,
    )
    risk_correlations = _mean_monthly_correlation(panel, risk.standardized)
    block_correlations = _mean_monthly_correlation(panel, comparison_blocks)

    comparison_evaluations = {
        "risk_block": evaluations["risk_block"].base,
        "value_block": evaluate_signal(panel, value.value_block, "value_block"),
        "quality_block": evaluate_signal(panel, quality.quality_block, "quality_block"),
        "momentum_12_1": evaluate_signal(
            panel,
            momentum_standardized,
            "momentum_12_1",
        ),
        "capital_deployment_block": evaluate_signal(
            panel,
            capital.capital_deployment_block,
            "capital_deployment_block",
        ),
    }
    spread_comparison: pd.DataFrame | None = None
    for name, result in comparison_evaluations.items():
        spread = result.spread_returns.loc[:, ["date", "spread"]].rename(
            columns={"spread": name}
        )
        spread_comparison = (
            spread
            if spread_comparison is None
            else spread_comparison.merge(
                spread,
                on="date",
                how="inner",
                validate="one_to_one",
            )
        )
    if spread_comparison is None:
        raise RuntimeError("No block spread returns were available for correlation.")
    spread_correlations = spread_comparison.drop(columns="date").corr()

    print("\nCORRELATIONS")
    print("Average monthly cross-sectional correlations of standardized risk signals:")
    print(risk_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Average monthly cross-sectional block correlations:")
    print(block_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Equal-weight Q5-Q1 monthly spread correlations:")
    print(spread_correlations.to_string(float_format=lambda value: f"{value:.4f}"))

    figure = plot_risk_spreads(evaluations)
    plt.show()
    return {
        "risk": risk,
        "evaluations": evaluations,
        "value": value,
        "quality": quality,
        "capital_deployment": capital,
        "momentum_standardized": momentum_standardized,
        "comparison_evaluations": comparison_evaluations,
        "risk_correlations": risk_correlations,
        "block_correlations": block_correlations,
        "spread_comparison": spread_comparison,
        "spread_correlations": spread_correlations,
        "figure": figure,
    }


def run_sentiment_research() -> dict[str, object]:
    """Evaluate short-interest sentiment and compare it with existing blocks.

    Args:
        None.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    panel = load_panel()
    sentiment = build_sentiment_block(panel)
    print("SENTIMENT CONSTRUCTION DIAGNOSTICS")
    print(
        _format_percent_table(sentiment.diagnostics, ["share"]).to_string(index=False)
    )
    evaluation = evaluate_signal_extended(
        panel,
        sentiment.sentiment_block,
        "short_interest_ratio",
    )
    print_extended_evaluation(evaluation)

    value = build_value_block(panel)
    quality = build_quality_block(panel)
    capital = build_capital_deployment_block(panel)
    momentum_standardized = standardize_signal(
        panel,
        momentum_12_1(panel),
    )["signal"].rename("momentum_12_1")
    comparison_blocks = pd.concat(
        [
            sentiment.sentiment_block,
            value.value_block,
            quality.quality_block,
            momentum_standardized,
            capital.capital_deployment_block,
        ],
        axis=1,
    )
    block_correlations = _mean_monthly_correlation(panel, comparison_blocks)

    comparison_evaluations = {
        "sentiment_block": evaluation.base,
        "value_block": evaluate_signal(panel, value.value_block, "value_block"),
        "quality_block": evaluate_signal(panel, quality.quality_block, "quality_block"),
        "momentum_12_1": evaluate_signal(
            panel,
            momentum_standardized,
            "momentum_12_1",
        ),
        "capital_deployment_block": evaluate_signal(
            panel,
            capital.capital_deployment_block,
            "capital_deployment_block",
        ),
    }
    spread_comparison: pd.DataFrame | None = None
    for name, result in comparison_evaluations.items():
        spread = result.spread_returns.loc[:, ["date", "spread"]].rename(
            columns={"spread": name}
        )
        spread_comparison = (
            spread
            if spread_comparison is None
            else spread_comparison.merge(
                spread,
                on="date",
                how="inner",
                validate="one_to_one",
            )
        )
    if spread_comparison is None:
        raise RuntimeError("No block spread returns were available for correlation.")
    spread_correlations = spread_comparison.drop(columns="date").corr()

    print("\nCORRELATIONS")
    print("Average monthly cross-sectional block correlations:")
    print(block_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Equal-weight Q5-Q1 monthly spread correlations:")
    print(spread_correlations.to_string(float_format=lambda value: f"{value:.4f}"))

    figure = plot_cumulative_spread(evaluation.base)
    plt.show()
    return {
        "sentiment": sentiment,
        "evaluation": evaluation,
        "value": value,
        "quality": quality,
        "capital_deployment": capital,
        "momentum_standardized": momentum_standardized,
        "comparison_evaluations": comparison_evaluations,
        "block_correlations": block_correlations,
        "spread_comparison": spread_comparison,
        "spread_correlations": spread_correlations,
        "figure": figure,
    }


def run_fundamental_momentum_research() -> dict[str, object]:
    """Evaluate fundamental trends, fixed sensitivities, and block correlations.

    Args:
        None.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    panel = load_panel()
    fundamental_momentum = build_fundamental_momentum_block(panel)
    print("FUNDAMENTAL MOMENTUM CONSTRUCTION DIAGNOSTICS")
    print(
        _format_percent_table(
            fundamental_momentum.diagnostics,
            ["share"],
        ).to_string(index=False)
    )

    signals = fundamental_momentum.standardized.copy()
    signals["fundamental_momentum_block"] = (
        fundamental_momentum.fundamental_momentum_block
    )
    evaluations: dict[str, ExtendedEvaluation] = {}
    for name in (*FUNDMOM_SIGNAL_NAMES, "fundamental_momentum_block"):
        result = evaluate_signal_extended(panel, signals[name], name)
        evaluations[name] = result
        print_extended_evaluation(result)

    sensitivity_blocks = build_sensitivity_blocks(panel)
    sensitivity_evaluations: dict[int, ExtendedEvaluation] = {
        fundamental_momentum.quarters: evaluations["fundamental_momentum_block"]
    }
    for window, block in sensitivity_blocks.items():
        sensitivity_evaluations[window] = evaluate_signal_extended(
            panel,
            block.fundamental_momentum_block,
            f"fundamental_momentum_block_{window}q_sensitivity",
        )

    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_block_lookup = {
        fundamental_momentum.quarters: fundamental_momentum,
        **sensitivity_blocks,
    }
    for window in sorted(sensitivity_evaluations):
        result = sensitivity_evaluations[window]
        block = sensitivity_block_lookup[window]
        sensitivity_rows.append(
            {
                "quarters": window,
                "minimum_observations": block.minimum_observations,
                "mean_coverage": result.base.coverage["coverage"].mean(),
                "mean_rank_ic": result.base.ic_summary.loc[0, "mean"],
                "ic_t_stat": result.base.ic_summary.loc[0, "t_stat"],
                "ew_monthly_spread": result.base.spread_summary.loc[
                    0, "average_monthly_spread"
                ],
                "ew_annualized_sharpe": result.base.spread_summary.loc[
                    0, "annualized_sharpe"
                ],
                "vw_monthly_spread": result.value_weighted_spread_summary.loc[
                    0, "average_monthly_spread"
                ],
                "vw_annualized_sharpe": result.value_weighted_spread_summary.loc[
                    0, "annualized_sharpe"
                ],
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    positive_at_all_windows = bool(
        sensitivity[
            ["mean_rank_ic", "ew_monthly_spread", "vw_monthly_spread"]
        ].gt(0).all().all()
    )
    print("\nFIXED LOOKBACK SENSITIVITY — ROBUSTNESS CHECK, NOT MODEL SELECTION")
    print(
        _format_percent_table(
            sensitivity,
            ["mean_coverage", "ew_monthly_spread", "vw_monthly_spread"],
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(
        "Positive IC and equal-/value-weighted spreads at every fixed window: "
        f"{'YES' if positive_at_all_windows else 'NO'}"
    )

    value = build_value_block(panel)
    quality = build_quality_block(panel)
    capital = build_capital_deployment_block(panel)
    sentiment = build_sentiment_block(panel)
    momentum_standardized = standardize_signal(
        panel,
        momentum_12_1(panel),
    )["signal"].rename("momentum_12_1")
    comparison_blocks = pd.concat(
        [
            fundamental_momentum.fundamental_momentum_block,
            value.value_block,
            quality.quality_block,
            momentum_standardized,
            capital.capital_deployment_block,
            sentiment.sentiment_block,
        ],
        axis=1,
    )
    signal_correlations = _mean_monthly_correlation(
        panel,
        fundamental_momentum.standardized,
    )
    block_correlations = _mean_monthly_correlation(panel, comparison_blocks)

    comparison_evaluations = {
        "fundamental_momentum_block": evaluations[
            "fundamental_momentum_block"
        ].base,
        "value_block": evaluate_signal(panel, value.value_block, "value_block"),
        "quality_block": evaluate_signal(panel, quality.quality_block, "quality_block"),
        "momentum_12_1": evaluate_signal(
            panel,
            momentum_standardized,
            "momentum_12_1",
        ),
        "capital_deployment_block": evaluate_signal(
            panel,
            capital.capital_deployment_block,
            "capital_deployment_block",
        ),
        "sentiment_block": evaluate_signal(
            panel,
            sentiment.sentiment_block,
            "sentiment_block",
        ),
    }
    spread_comparison: pd.DataFrame | None = None
    for name, result in comparison_evaluations.items():
        spread = result.spread_returns.loc[:, ["date", "spread"]].rename(
            columns={"spread": name}
        )
        spread_comparison = (
            spread
            if spread_comparison is None
            else spread_comparison.merge(
                spread,
                on="date",
                how="inner",
                validate="one_to_one",
            )
        )
    if spread_comparison is None:
        raise RuntimeError("No block spread returns were available for correlation.")
    spread_correlations = spread_comparison.drop(columns="date").corr()

    print("\nCORRELATIONS")
    print("Average monthly cross-sectional correlations of trend signals:")
    print(signal_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Average monthly cross-sectional block correlations:")
    print(block_correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Equal-weight Q5-Q1 monthly spread correlations:")
    print(spread_correlations.to_string(float_format=lambda value: f"{value:.4f}"))

    figure = plot_fundamental_momentum_spreads(evaluations)
    plt.show()
    return {
        "fundamental_momentum": fundamental_momentum,
        "evaluations": evaluations,
        "sensitivity_blocks": sensitivity_blocks,
        "sensitivity_evaluations": sensitivity_evaluations,
        "sensitivity": sensitivity,
        "positive_at_all_windows": positive_at_all_windows,
        "value": value,
        "quality": quality,
        "capital_deployment": capital,
        "sentiment": sentiment,
        "momentum_standardized": momentum_standardized,
        "comparison_evaluations": comparison_evaluations,
        "signal_correlations": signal_correlations,
        "block_correlations": block_correlations,
        "spread_comparison": spread_comparison,
        "spread_correlations": spread_correlations,
        "figure": figure,
    }
