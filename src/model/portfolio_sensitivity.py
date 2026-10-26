"""Compare portfolio weighting schemes with and without the beta constraint.

Reads the configured Russell panel/factors and writes sensitivity CSVs under
``data/``. The decision rule was fixed before results: reduce concentration,
keep beta in band, and avoid a statistically meaningful return loss; Sharpe is
reported but is never used to select a variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DATA_DIR, MAX_BETA, MIN_BETA
from src.model.composite import build_model_score
from src.model.portfolio import (
    PortfolioBacktest,
    WEIGHT_SCHEMES,
    backtest_portfolio,
    compute_market_betas,
    construct_portfolio,
    load_inputs,
)


# Decision rule fixed before inspecting results: a candidate must reduce average
# top-10 concentration by at least five percentage points, keep every monthly
# weighted beta inside the configured band, and not have a significantly lower
# paired net return than the benchmark-weight/no-beta baseline (5% two-sided
# threshold). Among candidates, choose lowest concentration, then turnover.
# Sharpe is reported but never enters the choice.
MATERIAL_CONCENTRATION_REDUCTION = 0.05
PAIRED_NOISE_TSTAT = -1.96
BASELINE_VARIANT = "benchmark_without_beta"


@dataclass
class PortfolioSensitivity:
    """Backtests and comparison tables for all six construction variants."""

    summary: pd.DataFrame
    monthly_beta: pd.DataFrame
    sector_weights: pd.DataFrame
    backtests: dict[str, PortfolioBacktest]
    selected_variant: str | None


def _maximum_drawdown(returns: pd.Series) -> float:
    """Return the maximum drawdown of a monthly return stream."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _months_to_half_log_excess(monthly: pd.DataFrame) -> int:
    """Count the largest positive months needed for half total log excess."""
    log_excess = np.log1p(monthly["net_return"]) - np.log1p(
        monthly["benchmark_return"]
    )
    total = float(log_excess.sum())
    positive = log_excess.loc[log_excess.gt(0)].sort_values(ascending=False)
    if total <= 0 or positive.empty:
        return 0
    return int(
        np.searchsorted(positive.cumsum().to_numpy(), total / 2.0) + 1
    )


def _variant_summary(
    variant: str,
    scheme: str,
    constrained: bool,
    backtest: PortfolioBacktest,
) -> dict[str, object]:
    """Return all requested performance and construction diagnostics."""
    monthly = backtest.monthly_returns.dropna(
        subset=["net_return", "benchmark_return", "rf"]
    ).copy()
    returns = monthly["net_return"].astype(float)
    excess = returns - monthly["rf"].astype(float)
    active = returns - monthly["benchmark_return"].astype(float)
    annualized_volatility = returns.std(ddof=1) * np.sqrt(12.0)
    excess_volatility = excess.std(ddof=1) * np.sqrt(12.0)
    tracking_error = active.std(ddof=1) * np.sqrt(12.0)
    steady_turnover = monthly["turnover"].iloc[1:]
    return {
        "variant": variant,
        "weight_scheme": scheme,
        "beta_constraint": constrained,
        "months": len(monthly),
        "annualized_net_return": (
            (1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0
        ),
        "annualized_volatility": annualized_volatility,
        "net_sharpe": (
            excess.mean() * 12.0 / excess_volatility
            if excess_volatility > 0
            else np.nan
        ),
        "tracking_error": tracking_error,
        "net_information_ratio": (
            active.mean() * 12.0 / tracking_error
            if tracking_error > 0
            else np.nan
        ),
        "maximum_drawdown": _maximum_drawdown(returns),
        "average_top_10_weight": monthly["top_10_weight"].mean(),
        "maximum_top_10_weight": monthly["top_10_weight"].max(),
        "average_names_at_cap": monthly["names_at_name_cap"].mean(),
        "maximum_names_at_cap": int(monthly["names_at_name_cap"].max()),
        "average_portfolio_beta": monthly["post_constraint_beta"].mean(),
        "minimum_portfolio_beta": monthly["post_constraint_beta"].min(),
        "maximum_portfolio_beta": monthly["post_constraint_beta"].max(),
        "beta_binding_months": int(monthly["beta_constraint_bound"].sum()),
        "beta_binding_frequency": monthly["beta_constraint_bound"].mean(),
        "average_beta_observed_weight": monthly["beta_observed_weight"].mean(),
        "average_max_sector_deviation": monthly[
            "max_abs_sector_deviation"
        ].mean(),
        "maximum_sector_deviation": monthly["max_abs_sector_deviation"].max(),
        "average_turnover": steady_turnover.mean(),
        "months_to_half_cumulative_log_excess": (
            _months_to_half_log_excess(monthly)
        ),
    }


def _monthly_beta_table(
    variant: str,
    scheme: str,
    constrained: bool,
    backtest: PortfolioBacktest,
) -> pd.DataFrame:
    """Return pre/post beta and binding status for every backtest month."""
    columns = [
        "formation_date",
        "return_date",
        "pre_constraint_beta",
        "post_constraint_beta",
        "beta_constraint_bound",
        "beta_observed_weight",
        "beta_missing_names",
    ]
    return backtest.monthly_returns.loc[:, columns].assign(
        variant=variant,
        weight_scheme=scheme,
        beta_constraint=constrained,
    )[
        [
            "variant",
            "weight_scheme",
            "beta_constraint",
            *columns,
        ]
    ]


def _sector_table(
    variant: str,
    scheme: str,
    constrained: bool,
    backtest: PortfolioBacktest,
) -> pd.DataFrame:
    """Return average and extreme sector weights over matched backtest months."""
    formation_dates = set(backtest.monthly_returns["formation_date"])
    sectors = backtest.construction.sector_weights.loc[
        backtest.construction.sector_weights["date"].isin(formation_dates)
    ].copy()
    sectors["abs_deviation"] = sectors["sector_deviation"].abs()
    output = sectors.groupby("sector", as_index=False).agg(
        average_portfolio_weight=("portfolio_sector_weight", "mean"),
        average_benchmark_weight=("benchmark_sector_weight", "mean"),
        average_sector_deviation=("sector_deviation", "mean"),
        maximum_absolute_deviation=("abs_deviation", "max"),
    )
    return output.assign(
        variant=variant,
        weight_scheme=scheme,
        beta_constraint=constrained,
    )[
        [
            "variant",
            "weight_scheme",
            "beta_constraint",
            "sector",
            "average_portfolio_weight",
            "average_benchmark_weight",
            "average_sector_deviation",
            "maximum_absolute_deviation",
        ]
    ]


def _apply_decision_rule(
    summary: pd.DataFrame,
    backtests: dict[str, PortfolioBacktest],
) -> tuple[pd.DataFrame, str | None]:
    """Apply the precommitted concentration/beta/noise decision rule."""
    baseline = summary.loc[summary["variant"].eq(BASELINE_VARIANT)].iloc[0]
    baseline_returns = backtests[BASELINE_VARIANT].monthly_returns.set_index(
        "return_date"
    )["net_return"]
    output = summary.copy()
    paired_t_stats: list[float] = []
    for variant in output["variant"]:
        returns = backtests[variant].monthly_returns.set_index("return_date")[
            "net_return"
        ]
        difference = returns.sub(baseline_returns, fill_value=np.nan).dropna()
        standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
        paired_t_stats.append(
            float(difference.mean() / standard_error)
            if standard_error > 0
            else np.nan
        )
    output["paired_net_return_tstat_vs_baseline"] = paired_t_stats
    output["material_concentration_reduction"] = output[
        "average_top_10_weight"
    ].le(
        baseline["average_top_10_weight"]
        - MATERIAL_CONCENTRATION_REDUCTION
    )
    output["beta_controlled"] = (
        output["minimum_portfolio_beta"].ge(MIN_BETA - 1e-7)
        & output["maximum_portfolio_beta"].le(MAX_BETA + 1e-7)
    )
    output["information_ratio_within_noise"] = output[
        "paired_net_return_tstat_vs_baseline"
    ].ge(PAIRED_NOISE_TSTAT) | output["variant"].eq(BASELINE_VARIANT)
    output["qualifies_under_decision_rule"] = (
        output["material_concentration_reduction"]
        & output["beta_controlled"]
        & output["information_ratio_within_noise"]
    )
    candidates = output.loc[output["qualifies_under_decision_rule"]].sort_values(
        ["average_top_10_weight", "average_turnover", "variant"]
    )
    selected = str(candidates.iloc[0]["variant"]) if len(candidates) else None
    output["selected"] = output["variant"].eq(selected) if selected else False
    return output, selected


def run_portfolio_sensitivity(
    panel: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    write: bool = True,
) -> PortfolioSensitivity:
    """Run and report all three weighting schemes with and without beta bounds.

    Args:
        panel: Monthly point-in-time research panel.
        factors: Monthly factor and benchmark-return rows.
        write: Whether to write date-stamped output files.

    Returns:
        Computed result as ``PortfolioSensitivity``.
    """
    if panel is None or factors is None:
        local_panel, local_factors = load_inputs()
        panel = local_panel if panel is None else panel
        factors = local_factors if factors is None else factors
    print("DECISION RULE (fixed before results)")
    print(
        "Require >=5 percentage points lower average top-10 weight, every monthly "
        f"beta in [{MIN_BETA:.2f}, {MAX_BETA:.2f}], and no significantly lower "
        "paired net return versus benchmark weighting without beta at the 5% "
        "level. Choose lowest concentration, then turnover; never Sharpe."
    )
    composite_score = build_model_score(panel)
    market_beta = compute_market_betas(panel)
    backtests: dict[str, PortfolioBacktest] = {}
    summary_rows: list[dict[str, object]] = []
    beta_pieces: list[pd.DataFrame] = []
    sector_pieces: list[pd.DataFrame] = []
    for scheme in WEIGHT_SCHEMES:
        for constrained in (False, True):
            suffix = "with_beta" if constrained else "without_beta"
            variant = f"{scheme}_{suffix}"
            print(f"Running {variant}...")
            construction = construct_portfolio(
                panel,
                composite_score=composite_score,
                market_beta=market_beta,
                weight_scheme=scheme,
                apply_beta_constraint=constrained,
            )
            monthly, performance, annual = backtest_portfolio(
                panel, factors, construction
            )
            backtest = PortfolioBacktest(
                construction=construction,
                monthly_returns=monthly,
                performance=performance,
                annual_returns=annual,
            )
            backtests[variant] = backtest
            summary_rows.append(
                _variant_summary(
                    variant, scheme, constrained, backtest
                )
            )
            beta_pieces.append(
                _monthly_beta_table(
                    variant, scheme, constrained, backtest
                )
            )
            sector_pieces.append(
                _sector_table(variant, scheme, constrained, backtest)
            )
    summary, selected = _apply_decision_rule(
        pd.DataFrame(summary_rows), backtests
    )
    monthly_beta = pd.concat(beta_pieces, ignore_index=True)
    sector_weights = pd.concat(sector_pieces, ignore_index=True)
    result = PortfolioSensitivity(
        summary=summary,
        monthly_beta=monthly_beta,
        sector_weights=sector_weights,
        backtests=backtests,
        selected_variant=selected,
    )
    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(DATA_DIR / "portfolio_weighting_sensitivity.csv", index=False)
        monthly_beta.to_csv(DATA_DIR / "portfolio_beta_monthly.csv", index=False)
        sector_weights.to_csv(
            DATA_DIR / "portfolio_sector_sensitivity.csv", index=False
        )
    print("\nSIX-VARIANT SUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBETA CONSTRAINT BINDING FREQUENCY")
    print(
        summary[
            [
                "variant",
                "beta_binding_months",
                "beta_binding_frequency",
                "average_portfolio_beta",
                "minimum_portfolio_beta",
                "maximum_portfolio_beta",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nDECISION")
    print(selected if selected is not None else "No variant qualified.")
    return result


def main() -> None:
    """Run the complete six-variant portfolio construction test.

    Args:
        None.

    Returns:
        None.
    """
    run_portfolio_sensitivity()


if __name__ == "__main__":
    main()
