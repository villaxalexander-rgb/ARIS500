"""Describe the configured target portfolio at one rebalance date.

Reads ``data/panel_russell3000.parquet`` and the current holdings used by the
shared rebalance process; it writes no files. The report consumes the existing
rebalance, composite, benchmark-weight, beta, constraint, and size/style logic
so this meeting view cannot quietly diverge from the portfolio actually built.
Pass an existing ``RebalanceResult`` after a monthly run to avoid rescoring the
panel; otherwise the module builds the configured target for the requested date.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from config import MAX_NAME_WEIGHT, MAX_SECTOR_DEVIATION, MODEL_BLOCKS
from src.model.portfolio import _benchmark_weights
from src.model.portfolio_analysis import (
    SECTOR_NAMES,
    assign_benchmark_size_buckets,
    size_and_style_profile,
)
from src.model.rebalance import (
    RebalanceResult,
    build_rebalance,
    load_current_portfolio,
    load_panel,
)
from src.signals.registry import validate_live_blocks


BLOCK_LABELS = {
    spec.name: spec.label for spec in validate_live_blocks(MODEL_BLOCKS)
}
SIZE_LABELS = {
    "large": "Largest 500",
    "mid": "Next 1,000",
    "small": "Rest",
}


@dataclass
class PortfolioSummary:
    """Tables and meeting-ready charts for one target portfolio."""

    date: pd.Timestamp
    headline: pd.DataFrame
    top_holdings: pd.DataFrame
    sector_positioning: pd.DataFrame
    largest_overweights: pd.DataFrame
    largest_underweights: pd.DataFrame
    size_profile: pd.DataFrame
    block_tilts: pd.DataFrame
    sector_figure: Figure
    block_figure: Figure


def _summary_metric(result: RebalanceResult, metric: str) -> object:
    """Read one uniquely named value from the shared rebalance summary."""
    matches = result.summary.loc[result.summary["metric"].eq(metric), "value"]
    if len(matches) != 1:
        raise ValueError(f"Expected one rebalance summary value for {metric!r}.")
    return matches.iloc[0]


def _sector_name(sector: object) -> str:
    """Map a stable GICS sector code to its existing display label."""
    return SECTOR_NAMES.get(str(sector), str(sector))


def _month_with_scores(
    panel: pd.DataFrame,
    result: RebalanceResult,
) -> pd.DataFrame:
    """Attach the already-computed block scores to the requested panel month."""
    panel_dates = pd.to_datetime(panel["date"])
    month_mask = panel_dates.eq(result.date)
    month = panel.loc[month_mask].copy()
    if month.empty:
        raise ValueError(f"Panel has no rows for target date {result.date.date()}.")
    month["date"] = panel_dates.loc[month_mask]
    for block in MODEL_BLOCKS:
        month[block] = result.composite.blocks.loc[month_mask, block].to_numpy()
    return month


def _effective_block_contributions(
    target: pd.DataFrame,
    result: RebalanceResult,
) -> pd.DataFrame:
    """Return each block's normalized contribution to the composite score."""
    scheme = result.composite.selected_scheme
    weights = result.composite.weights[scheme]
    matching = weights.loc[pd.DatetimeIndex(weights.index) == result.date]
    if len(matching) != 1:
        raise ValueError(
            f"Composite weights contain {len(matching)} rows for {result.date.date()}."
        )
    block_weights = matching.iloc[0].reindex(MODEL_BLOCKS).astype(float)
    scores = target.loc[:, MODEL_BLOCKS].apply(pd.to_numeric, errors="coerce")
    available_weights = pd.DataFrame(
        np.broadcast_to(block_weights.to_numpy(), scores.shape),
        index=scores.index,
        columns=scores.columns,
    ).where(scores.notna())
    denominator = available_weights.sum(axis=1, min_count=1)
    return scores.mul(available_weights).div(denominator, axis=0)


def _top_holdings(
    result: RebalanceResult,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Build the top-15 holdings table with benchmark and block context."""
    target = result.target.copy()
    target = target.merge(
        benchmark[["permno", "benchmark_weight"]],
        on="permno",
        how="left",
        validate="1:1",
    )
    target["benchmark_weight"] = target["benchmark_weight"].fillna(0.0)
    target["active_weight"] = target["weight"] - target["benchmark_weight"]
    contributions = _effective_block_contributions(target, result)
    target["driving_block"] = contributions.idxmax(axis=1).map(BLOCK_LABELS)
    target["driver_contribution"] = contributions.max(axis=1)
    target["sector"] = target["sector"].map(_sector_name)
    target = target.rename(columns={"decile": "composite_decile"})
    return target.nlargest(15, "weight").loc[
        :,
        [
            "ticker",
            "company_name",
            "sector",
            "weight",
            "benchmark_weight",
            "active_weight",
            "composite_decile",
            "driving_block",
            "driver_contribution",
        ],
    ].reset_index(drop=True)


def _sector_positioning(result: RebalanceResult) -> pd.DataFrame:
    """Return portfolio and benchmark sector weights ordered by active weight."""
    sectors = result.sector_weights.copy().rename(
        columns={
            "benchmark_sector_weight": "benchmark_weight",
            "portfolio_sector_weight": "portfolio_weight",
            "sector_deviation": "active_weight",
        }
    )
    sectors["sector_code"] = sectors["sector"]
    sectors["sector"] = sectors["sector"].map(_sector_name)
    sectors["at_sector_cap"] = np.isclose(
        sectors["active_weight"].abs(),
        MAX_SECTOR_DEVIATION,
        atol=1e-6,
        rtol=0.0,
    )
    return sectors.loc[
        :,
        [
            "sector_code",
            "sector",
            "portfolio_weight",
            "benchmark_weight",
            "active_weight",
            "at_sector_cap",
        ],
    ].sort_values("active_weight", ascending=False, ignore_index=True)


def _active_positions(
    month: pd.DataFrame,
    result: RebalanceResult,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ten largest security overweights and underweights."""
    metadata = month.loc[
        :, ["permno", "ticker", "company_name", "gsector"]
    ].drop_duplicates("permno")
    positions = benchmark.loc[:, ["permno", "benchmark_weight"]].merge(
        metadata,
        on="permno",
        how="left",
        validate="1:1",
    )
    positions = positions.merge(
        result.target[["permno", "weight"]],
        on="permno",
        how="outer",
        validate="1:1",
    )
    positions[["benchmark_weight", "weight"]] = positions[
        ["benchmark_weight", "weight"]
    ].fillna(0.0)
    positions["active_weight"] = positions["weight"] - positions["benchmark_weight"]
    positions["held"] = positions["weight"].gt(0)
    positions["sector"] = positions["gsector"].map(
        lambda value: _sector_name(
            str(int(value)) if pd.notna(value) else "Unknown"
        )
    )
    columns = [
        "ticker",
        "company_name",
        "sector",
        "weight",
        "benchmark_weight",
        "active_weight",
        "held",
    ]
    overweights = positions.nlargest(10, "active_weight").loc[:, columns]
    underweights = positions.nsmallest(10, "active_weight").loc[:, columns]
    return overweights.reset_index(drop=True), underweights.reset_index(drop=True)


def _size_and_block_profiles(
    month: pd.DataFrame,
    result: RebalanceResult,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the shared backtest profiler for current size and block exposures."""
    benchmark_profile = benchmark.copy().assign(date=result.date)
    score_lookup = month.set_index("permno")[list(MODEL_BLOCKS)]
    for block in MODEL_BLOCKS:
        benchmark_profile[block] = benchmark_profile["permno"].map(
            score_lookup[block]
        )
    benchmark_profile = assign_benchmark_size_buckets(benchmark_profile)

    holdings_profile = result.target.loc[
        :, ["permno", "weight", *MODEL_BLOCKS]
    ].merge(
        benchmark_profile[["permno", "date", "size_bucket", "market_cap"]],
        on="permno",
        how="left",
        validate="1:1",
    )
    size_weights, _, style = size_and_style_profile(
        benchmark_profile,
        holdings_profile,
        list(MODEL_BLOCKS),
    )
    size_weights["active_weight"] = (
        size_weights["portfolio_weight"] - size_weights["benchmark_weight"]
    )
    size_weights["size_group"] = size_weights["size_bucket"].map(SIZE_LABELS)
    size_weights["_order"] = size_weights["size_bucket"].map(
        {"large": 0, "mid": 1, "small": 2}
    )
    size_profile = size_weights.sort_values("_order").loc[
        :, ["size_group", "portfolio_weight", "benchmark_weight", "active_weight"]
    ].reset_index(drop=True)

    block_tilts = style.rename(
        columns={
            "score": "block",
            "portfolio_average": "portfolio_score",
            "benchmark_average": "benchmark_score",
            "portfolio_coverage": "portfolio_weight_coverage",
            "benchmark_coverage": "benchmark_weight_coverage",
        }
    )
    block_tilts["tilt"] = (
        block_tilts["portfolio_score"] - block_tilts["benchmark_score"]
    )
    block_tilts["block"] = block_tilts["block"].map(BLOCK_LABELS)
    return size_profile, block_tilts.loc[
        :,
        [
            "block",
            "portfolio_score",
            "benchmark_score",
            "tilt",
            "portfolio_weight_coverage",
            "benchmark_weight_coverage",
        ],
    ].reset_index(drop=True)


def _headline(result: RebalanceResult) -> pd.DataFrame:
    """Return the requested one-date portfolio headline metrics."""
    target = result.target
    return pd.DataFrame(
        [
            {"metric": "Underlying data date", "value": result.date.date()},
            {"metric": "Holdings", "value": len(target)},
            {
                "metric": "Top-ten weight",
                "value": float(target.nlargest(10, "weight")["weight"].sum()),
            },
            {
                "metric": "Active share",
                "value": float(_summary_metric(result, "active share")),
            },
            {
                "metric": "Portfolio beta",
                "value": float(_summary_metric(result, "post-constraint beta")),
            },
            {
                "metric": "Weighted average market cap ($m)",
                "value": float(
                    _summary_metric(result, "weighted average market cap ($m)")
                ),
            },
            {
                "metric": f"Names at {MAX_NAME_WEIGHT:.0%} cap",
                "value": int(
                    np.isclose(
                        target["weight"],
                        MAX_NAME_WEIGHT,
                        atol=1e-8,
                        rtol=0.0,
                    ).sum()
                ),
            },
        ]
    )


def plot_sector_positioning(sectors: pd.DataFrame, date: pd.Timestamp) -> Figure:
    """Chart current portfolio and benchmark weights by sector.

    Args:
        sectors: Current sector positioning table.
        date: Underlying data date.

    Returns:
        Meeting-ready grouped horizontal bar chart.
    """
    chart = sectors.sort_values("active_weight", ascending=True).copy()
    chart["display_sector"] = chart["sector"] + np.where(
        chart["at_sector_cap"], " *", ""
    )
    y = np.arange(len(chart))
    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.barh(
        y - 0.19,
        chart["benchmark_weight"],
        height=0.36,
        label="Benchmark",
        color="#9aa4b2",
    )
    axis.barh(
        y + 0.19,
        chart["portfolio_weight"],
        height=0.36,
        label="Portfolio",
        color="#2457a6",
    )
    axis.set_yticks(y, chart["display_sector"])
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Weight")
    axis.set_title(f"Sector positioning — {date:%Y-%m-%d}")
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    if chart["at_sector_cap"].any():
        axis.text(
            0.0,
            -0.12,
            f"* active weight at the {MAX_SECTOR_DEVIATION:.0%} sector cap",
            transform=axis.transAxes,
            fontsize=9,
        )
    figure.tight_layout()
    return figure


def plot_block_tilts(blocks: pd.DataFrame, date: pd.Timestamp) -> Figure:
    """Chart weighted block scores for the portfolio and benchmark.

    Args:
        blocks: Current block-score exposure table.
        date: Underlying data date.

    Returns:
        Meeting-ready grouped block-score chart.
    """
    x = np.arange(len(blocks))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        x - 0.19,
        blocks["benchmark_score"],
        width=0.36,
        label="Benchmark",
        color="#9aa4b2",
    )
    axis.bar(
        x + 0.19,
        blocks["portfolio_score"],
        width=0.36,
        label="Portfolio",
        color="#2457a6",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, blocks["block"], rotation=20, ha="right")
    axis.set_ylabel("Weighted average standardized score")
    axis.set_title(f"Signals expressed by the portfolio — {date:%Y-%m-%d}")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def build_portfolio_summary(
    panel: pd.DataFrame,
    rebalance: RebalanceResult,
) -> PortfolioSummary:
    """Build all current-portfolio tables and charts from an existing target.

    Args:
        panel: Monthly point-in-time Russell 3000 panel.
        rebalance: Target portfolio produced by the shared rebalance process.

    Returns:
        ``PortfolioSummary`` containing copyable tables and two figures.
    """
    if rebalance.target.empty:
        raise ValueError("Cannot summarize an empty target portfolio.")
    month = _month_with_scores(panel, rebalance)
    benchmark = _benchmark_weights(month)
    top_holdings = _top_holdings(rebalance, benchmark)
    sectors = _sector_positioning(rebalance)
    overweights, underweights = _active_positions(
        month,
        rebalance,
        benchmark,
    )
    size_profile, block_tilts = _size_and_block_profiles(
        month,
        rebalance,
        benchmark,
    )
    return PortfolioSummary(
        date=rebalance.date,
        headline=_headline(rebalance),
        top_holdings=top_holdings,
        sector_positioning=sectors,
        largest_overweights=overweights,
        largest_underweights=underweights,
        size_profile=size_profile,
        block_tilts=block_tilts,
        sector_figure=plot_sector_positioning(sectors, rebalance.date),
        block_figure=plot_block_tilts(block_tilts, rebalance.date),
    )


def _print_weight_table(frame: pd.DataFrame) -> None:
    """Print a security table with slide-ready percentage formatting."""
    formatters = {
        column: "{:.2%}".format
        for column in ("weight", "benchmark_weight", "active_weight")
        if column in frame
    }
    if "driver_contribution" in frame:
        formatters["driver_contribution"] = "{:+.3f}".format
    print(frame.to_string(index=False, formatters=formatters))


def print_portfolio_summary(summary: PortfolioSummary) -> None:
    """Print all current-portfolio tables in a copyable text layout.

    Args:
        summary: Previously built current-portfolio summary.

    Returns:
        None.
    """
    headline = summary.headline.copy()
    headline.loc[headline["metric"].isin(["Top-ten weight", "Active share"]), "value"] = (
        headline.loc[
            headline["metric"].isin(["Top-ten weight", "Active share"]), "value"
        ].map(lambda value: f"{float(value):.2%}")
    )
    headline.loc[headline["metric"].eq("Portfolio beta"), "value"] = headline.loc[
        headline["metric"].eq("Portfolio beta"), "value"
    ].map(lambda value: f"{float(value):.3f}")
    headline.loc[
        headline["metric"].eq("Weighted average market cap ($m)"), "value"
    ] = headline.loc[
        headline["metric"].eq("Weighted average market cap ($m)"), "value"
    ].map(lambda value: f"${float(value):,.0f}m")
    print(f"\nCURRENT TARGET PORTFOLIO — {summary.date.date()}")
    print("\nHEADLINE")
    print(headline.to_string(index=False))

    print("\nTOP 15 HOLDINGS")
    _print_weight_table(summary.top_holdings)

    print("\nSECTOR POSITIONING")
    sectors = summary.sector_positioning.copy()
    print(
        sectors.to_string(
            index=False,
            formatters={
                "portfolio_weight": "{:.2%}".format,
                "benchmark_weight": "{:.2%}".format,
                "active_weight": "{:+.2%}".format,
            },
        )
    )

    print("\nTOP 10 ACTIVE OVERWEIGHTS")
    _print_weight_table(summary.largest_overweights)
    print("\nTOP 10 ACTIVE UNDERWEIGHTS")
    _print_weight_table(summary.largest_underweights)

    print("\nSIZE PROFILE")
    print(
        summary.size_profile.to_string(
            index=False,
            formatters={
                "portfolio_weight": "{:.2%}".format,
                "benchmark_weight": "{:.2%}".format,
                "active_weight": "{:+.2%}".format,
            },
        )
    )

    print("\nBLOCK SCORE TILTS")
    print(
        summary.block_tilts.to_string(
            index=False,
            formatters={
                "portfolio_score": "{:+.3f}".format,
                "benchmark_score": "{:+.3f}".format,
                "tilt": "{:+.3f}".format,
                "portfolio_weight_coverage": "{:.1%}".format,
                "benchmark_weight_coverage": "{:.1%}".format,
            },
        )
    )


def run_portfolio_summary(
    rebalance: RebalanceResult | None = None,
    date: str | pd.Timestamp | None = None,
    panel: pd.DataFrame | None = None,
    show_charts: bool = True,
) -> PortfolioSummary:
    """Print and return the latest or requested-date target portfolio view.

    Args:
        rebalance: Existing target result; pass this after ``run_rebalance``.
        date: Optional panel month used when a rebalance is not supplied.
        panel: Optional preloaded panel to avoid another disk read.
        show_charts: Whether to display the meeting-ready charts.

    Returns:
        ``PortfolioSummary`` for the target date.
    """
    local_panel = load_panel() if panel is None else panel
    if rebalance is None:
        current = load_current_portfolio()
        rebalance, _ = build_rebalance(local_panel, current, date)
    elif date is not None and pd.Timestamp(date).to_period("M") != rebalance.date.to_period(
        "M"
    ):
        raise ValueError(
            f"Requested {pd.Timestamp(date):%Y-%m}, but the supplied rebalance is "
            f"for {rebalance.date:%Y-%m}."
        )
    summary = build_portfolio_summary(local_panel, rebalance)
    print_portfolio_summary(summary)
    if show_charts:
        plt.show()
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the current portfolio summary from the command line.

    Args:
        argv: Optional command-line arguments.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        help="Panel month/date to summarize; defaults to the latest available.",
    )
    args = parser.parse_args(argv)
    run_portfolio_summary(date=args.date)


if __name__ == "__main__":
    main()
