"""Build and compare point-in-time composites from the configured signal blocks.

Reads ``data/panel_russell3000.parquet`` and writes no files; research functions
return score tables, diagnostics, and figures. Missing block scores renormalize
the remaining weights, inverse-volatility weights use only trailing realized
spreads, and equal weighting remains the default unless a predeclared structural
risk rule—not the best historical Sharpe—justifies changing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from config import (
    COMPOSITE_START,
    COMPOSITE_WEIGHTING,
    INVERSE_VOL_MONTHS,
    MODEL_BLOCKS,
    N_QUANTILES,
    PANEL_RUSSELL3000_OUTPUT_PATH,
)
from src.signals.registry import required_panel_columns, validate_live_blocks
from src.signals.standardize import restandardize_monthly
from src.utils import forward_returns

if TYPE_CHECKING:
    from src.signal_evaluation import ExtendedEvaluation, EvaluationResult


WEIGHTING_SCHEMES = ("equal", "inverse_vol", "correlation_group")
COMPOSITE_PANEL_COLUMNS = required_panel_columns(
    MODEL_BLOCKS,
    base=("permno", "date", "ret", "investable", "gsector", "market_cap"),
)


@dataclass
class CompositeScores:
    """Block scores, spread histories, weights, and resulting composites."""

    blocks: pd.DataFrame
    block_spreads: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    scores: pd.DataFrame
    selected_scheme: str
    selected_score: pd.Series


@dataclass
class CompositeResearch:
    """Evaluations and diagnostics returned to the research notebook."""

    construction: CompositeScores
    evaluations: dict[str, ExtendedEvaluation]
    risk_contribution: pd.DataFrame
    leave_one_out: pd.DataFrame
    versus_quality: pd.DataFrame
    scheme_spread_correlations: pd.DataFrame
    individual_block_summary: pd.DataFrame
    figure: plt.Figure


def load_panel() -> pd.DataFrame:
    """Load the local Russell 3000 research panel.

    Args:
        None.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if not PANEL_RUSSELL3000_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Russell 3000 panel: {PANEL_RUSSELL3000_OUTPUT_PATH}"
        )
    return pd.read_parquet(
        PANEL_RUSSELL3000_OUTPUT_PATH,
        columns=list(COMPOSITE_PANEL_COLUMNS),
    )


def _validate_model_blocks() -> None:
    """Reject unsupported or duplicated configured block names."""
    validate_live_blocks(MODEL_BLOCKS)


def build_block_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the configured standardized block scores without future data.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    _validate_model_blocks()
    specs = validate_live_blocks(MODEL_BLOCKS)
    # Risk and fundamental momentum intentionally remain outside the model:
    # risk produced inverted quintiles and a negative spread, while fundamental
    # momentum decayed to essentially no signal as its lookback lengthened.
    return pd.DataFrame(
        {spec.name: spec.score(spec.build(panel)) for spec in specs},
        index=panel.index,
        dtype="float64",
    )


def _monthly_block_spreads(
    panel: pd.DataFrame,
    blocks: pd.DataFrame,
) -> pd.DataFrame:
    """Return equal-weighted monthly Q5-minus-Q1 returns for every block."""
    rows = panel.loc[:, ["permno", "date", "investable"]].copy()
    rows = rows.assign(
        date=pd.to_datetime(rows["date"]),
        forward_return=forward_returns(panel),
    )
    all_dates = pd.DatetimeIndex(rows["date"].drop_duplicates().sort_values())
    output = pd.DataFrame(index=all_dates, columns=blocks.columns, dtype="float64")
    for name in blocks.columns:
        work = rows.assign(signal=blocks[name]).loc[
            rows["investable"]
        ].dropna(subset=["signal", "forward_return"])
        spreads: dict[pd.Timestamp, float] = {}
        for date, group in work.groupby("date", sort=True):
            if len(group) < N_QUANTILES:
                continue
            ranks = group["signal"].rank(method="first")
            quantile = (
                np.floor((ranks - 1.0) * N_QUANTILES / len(group)) + 1.0
            ).astype("int8")
            returns = group.assign(_quantile=quantile).groupby(
                "_quantile"
            )["forward_return"].mean()
            if 1 in returns.index and N_QUANTILES in returns.index:
                spreads[date] = float(returns.loc[N_QUANTILES] - returns.loc[1])
        output[name] = pd.Series(spreads).reindex(all_dates)
    output.index.name = "date"
    return output


def _equal_weights(dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Return constant equal block weights for every month."""
    return pd.DataFrame(
        1.0 / len(columns),
        index=dates,
        columns=columns,
        dtype="float64",
    )


def _inverse_vol_weights(
    spreads: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return lagged trailing-60-month inverse spread-volatility weights."""
    history = spreads.reindex(dates)
    # A spread dated t uses the return from t+1. Shifting the trailing estimate
    # one formation month makes the t score depend only on realized spreads.
    volatility = history.rolling(
        INVERSE_VOL_MONTHS,
        min_periods=INVERSE_VOL_MONTHS,
    ).std(ddof=1).shift(1)
    inverse = 1.0 / volatility.where(volatility.gt(0))
    weights = inverse.div(inverse.sum(axis=1), axis=0)
    complete = weights.notna().all(axis=1)
    fallback = _equal_weights(dates, history.columns)
    fallback.loc[complete, :] = weights.loc[complete, :]
    return fallback


def _correlation_group_weights(
    dates: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    """Give momentum half the weight and the correlated group the other half."""
    if "momentum" not in columns:
        raise ValueError("correlation_group weighting requires the momentum block.")
    correlated = [column for column in columns if column != "momentum"]
    if not correlated:
        raise ValueError("correlation_group weighting requires non-momentum blocks.")
    # This allocation follows the correlation structure, not realized returns:
    # value, quality, capital deployment, and sentiment share 0.5 in total,
    # while the nearly orthogonal momentum block receives the other 0.5.
    weights = pd.DataFrame(0.0, index=dates, columns=columns)
    weights.loc[:, correlated] = 0.5 / len(correlated)
    weights.loc[:, "momentum"] = 0.5
    return weights


def weighting_table(
    dates: pd.DatetimeIndex,
    spreads: pd.DataFrame,
    scheme: str,
) -> pd.DataFrame:
    """Return point-in-time block weights for the requested scheme.

    Args:
        dates: Formation dates to cover.
        spreads: Monthly block long-short spread returns.
        scheme: Composite weighting scheme.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(
            f"Unknown composite weighting {scheme!r}; expected {WEIGHTING_SCHEMES}."
        )
    columns = spreads.columns
    if scheme == "equal":
        return _equal_weights(dates, columns)
    if scheme == "inverse_vol":
        return _inverse_vol_weights(spreads, dates)
    return _correlation_group_weights(dates, columns)


def _weighted_score(
    panel: pd.DataFrame,
    blocks: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    name: str,
) -> pd.Series:
    """Average available block scores and re-standardize the monthly result."""
    dates = pd.to_datetime(panel["date"])
    row_weights = monthly_weights.reindex(dates).set_axis(panel.index)
    available_weights = row_weights.where(blocks.notna())
    denominator = available_weights.sum(axis=1, min_count=1)
    numerator = (blocks * available_weights).sum(axis=1, min_count=1)
    weighted_average = (numerator / denominator).where(denominator.gt(0))
    return restandardize_monthly(panel, weighted_average).rename(name)


def build_composite_scores(
    panel: pd.DataFrame,
    blocks: pd.DataFrame | None = None,
) -> CompositeScores:
    """Build all three composite scores and their point-in-time weights.

    Args:
        panel: Monthly point-in-time research panel.
        blocks: Row-aligned standardized block scores.

    Returns:
        Computed result as ``CompositeScores``.
    """
    if COMPOSITE_WEIGHTING not in WEIGHTING_SCHEMES:
        raise ValueError(
            "COMPOSITE_WEIGHTING must be one of " f"{WEIGHTING_SCHEMES}."
        )
    block_scores = build_block_scores(panel) if blocks is None else blocks.copy()
    spreads = _monthly_block_spreads(panel, block_scores)
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"]).drop_duplicates().sort_values())
    weights: dict[str, pd.DataFrame] = {}
    scores = pd.DataFrame(index=panel.index, dtype="float64")
    for scheme in WEIGHTING_SCHEMES:
        weights[scheme] = weighting_table(dates, spreads, scheme)
        scores[scheme] = _weighted_score(
            panel,
            block_scores,
            weights[scheme],
            f"composite_{scheme}",
        )
    return CompositeScores(
        blocks=block_scores,
        block_spreads=spreads,
        weights=weights,
        scores=scores,
        selected_scheme=COMPOSITE_WEIGHTING,
        selected_score=scores[COMPOSITE_WEIGHTING].rename("composite_score"),
    )


def build_model_score(panel: pd.DataFrame) -> pd.Series:
    """Return the composite selected by COMPOSITE_WEIGHTING in config.py.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    return build_composite_scores(panel).selected_score


def _composite_evaluation_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Restrict evaluation membership without truncating signal histories."""
    output = panel.copy()
    output["investable"] = output["investable"].fillna(False).astype(bool) & pd.to_datetime(
        output["date"]
    ).ge(pd.Timestamp(COMPOSITE_START))
    return output


def _risk_contribution(spreads: pd.DataFrame) -> pd.DataFrame:
    """Attribute equal-weighted block-spread variance with Euler contributions."""
    sample = spreads.loc[spreads.index >= pd.Timestamp(COMPOSITE_START)].dropna()
    covariance = sample.cov().to_numpy()
    weights = np.full(len(sample.columns), 1.0 / len(sample.columns))
    portfolio_variance = float(weights @ covariance @ weights)
    component_variance = weights * (covariance @ weights)
    contribution = component_variance / portfolio_variance
    return pd.DataFrame(
        {
            "block": sample.columns,
            "weight": weights,
            "variance_contribution": component_variance,
            "risk_share": contribution,
            "over_40_percent": contribution > 0.40,
        }
    ).sort_values("risk_share", ascending=False, ignore_index=True)


def _leave_one_out(
    panel: pd.DataFrame,
    blocks: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate equal composites after dropping each configured block once."""
    from src.signal_evaluation import evaluate_signal

    evaluation_panel = _composite_evaluation_panel(panel)
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"]).drop_duplicates().sort_values())
    rows: list[dict[str, object]] = []
    for omitted in blocks.columns:
        remaining = blocks.drop(columns=omitted)
        weights = _equal_weights(dates, remaining.columns)
        score = _weighted_score(
            panel,
            remaining,
            weights,
            f"equal_without_{omitted}",
        )
        result = evaluate_signal(evaluation_panel, score, score.name)
        rows.append(
            {
                "omitted_block": omitted,
                "mean_rank_ic": result.ic_summary.loc[0, "mean"],
                "ic_t_stat": result.ic_summary.loc[0, "t_stat"],
                "spread_sharpe": result.spread_summary.loc[
                    0, "annualized_sharpe"
                ],
            }
        )
    return pd.DataFrame(rows)


def _mean_turnover(result: EvaluationResult) -> float:
    """Return average monthly top-quintile turnover, excluding the first month."""
    return float(result.turnover["turnover"].mean())


def _versus_quality(
    equal_result: ExtendedEvaluation,
    quality_result: EvaluationResult,
) -> pd.DataFrame:
    """Compare the equal composite directly with the quality block."""
    rows = []
    for name, result in (
        ("equal_composite", equal_result.base),
        ("quality", quality_result),
    ):
        rows.append(
            {
                "model": name,
                "mean_rank_ic": result.ic_summary.loc[0, "mean"],
                "ic_t_stat": result.ic_summary.loc[0, "t_stat"],
                "spread_sharpe": result.spread_summary.loc[
                    0, "annualized_sharpe"
                ],
                "maximum_drawdown": result.spread_summary.loc[
                    0, "maximum_drawdown"
                ],
                "mean_top_quintile_turnover": _mean_turnover(result),
            }
        )
    return pd.DataFrame(rows)


def _evaluation_summary(
    evaluations: dict[str, EvaluationResult],
) -> pd.DataFrame:
    """Return compact IC and spread statistics for evaluated scores."""
    rows = []
    for name, result in evaluations.items():
        rows.append(
            {
                "name": name,
                "mean_rank_ic": result.ic_summary.loc[0, "mean"],
                "ic_t_stat": result.ic_summary.loc[0, "t_stat"],
                "spread_sharpe": result.spread_summary.loc[
                    0, "annualized_sharpe"
                ],
            }
        )
    return pd.DataFrame(rows)


def plot_scheme_spreads(
    evaluations: dict[str, ExtendedEvaluation],
) -> plt.Figure:
    """Plot cumulative equal-weighted spread returns for all three schemes.

    Args:
        evaluations: Evaluations used by this operation.

    Returns:
        Matplotlib figure containing the requested chart.
    """
    figure, axis = plt.subplots(figsize=(10, 5))
    for scheme, result in evaluations.items():
        spread = result.base.spread_returns
        axis.plot(
            spread["date"],
            spread["cumulative_spread"],
            label=scheme,
            linewidth=1.4,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Composite Q5 minus Q1 cumulative returns",
        xlabel="",
        ylabel="Cumulative return",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def run_composite_research(
    panel: pd.DataFrame | None = None,
) -> CompositeResearch:
    """Build, evaluate, print, and return the complete composite study.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``CompositeResearch``.
    """
    from src.signal_evaluation import (
        evaluate_signal,
        evaluate_signal_extended,
        print_extended_evaluation,
    )

    research_panel = load_panel() if panel is None else panel
    construction = build_composite_scores(research_panel)
    evaluation_panel = _composite_evaluation_panel(research_panel)

    evaluations = {
        scheme: evaluate_signal_extended(
            evaluation_panel,
            construction.scores[scheme],
            f"composite_{scheme}",
        )
        for scheme in WEIGHTING_SCHEMES
    }
    for scheme, result in evaluations.items():
        print(f"\nCOMPOSITE WEIGHTING — {scheme}")
        print_extended_evaluation(result)

    risk_contribution = _risk_contribution(construction.block_spreads)
    leave_one_out = _leave_one_out(research_panel, construction.blocks)
    quality_result = evaluate_signal(
        evaluation_panel,
        construction.blocks["quality"],
        "quality_same_composite_period",
    )
    versus_quality = _versus_quality(evaluations["equal"], quality_result)
    spread_frame = pd.concat(
        {
            scheme: result.base.spread_returns.set_index("date")["spread"]
            for scheme, result in evaluations.items()
        },
        axis=1,
    )
    scheme_spread_correlations = spread_frame.corr()

    individual_results = {
        name: evaluate_signal(research_panel, construction.blocks[name], name)
        for name in construction.blocks
    }
    individual_block_summary = _evaluation_summary(individual_results)

    print("\nEQUAL COMPOSITE — BLOCK RISK CONTRIBUTION")
    print(
        risk_contribution.to_string(
            index=False,
            formatters={
                "weight": "{:.1%}".format,
                "variance_contribution": "{:.8f}".format,
                "risk_share": "{:.1%}".format,
            },
        )
    )
    print("\nEQUAL COMPOSITE — LEAVE ONE OUT")
    print(leave_one_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nEQUAL COMPOSITE VERSUS QUALITY")
    print(versus_quality.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCOMPOSITE SPREAD-RETURN CORRELATIONS")
    print(scheme_spread_correlations.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nINDIVIDUAL BLOCK REGRESSION CHECK")
    print(
        individual_block_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Decision rule: equal weighting remains the default. Switch to inverse-vol
    # only if one block contributes more than about 40% of variance, or switch
    # to correlation-group only if leave-one-out shows the correlated group
    # crowds out momentum. Never choose a scheme because it has the best
    # backtest Sharpe; selecting the winner among three would be dishonest.
    figure = plot_scheme_spreads(evaluations)
    return CompositeResearch(
        construction=construction,
        evaluations=evaluations,
        risk_contribution=risk_contribution,
        leave_one_out=leave_one_out,
        versus_quality=versus_quality,
        scheme_spread_correlations=scheme_spread_correlations,
        individual_block_summary=individual_block_summary,
        figure=figure,
    )
