"""Compare name-weight caps under benchmark weighting and controlled beta.

Reads the configured Russell panel/factors and writes name-cap sensitivity CSVs
under ``data/``. The predeclared rule selects the lowest cap whose information
ratio stays within 0.05 of the 4% baseline, rather than selecting on Sharpe;
that rule produced the current 3% cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import DATA_DIR
from src.model.composite import build_model_score
from src.model.portfolio import (
    PortfolioBacktest,
    backtest_portfolio,
    compute_market_betas,
    construct_portfolio,
    load_inputs,
)
from src.model.portfolio_sensitivity import (
    _monthly_beta_table,
    _sector_table,
    _variant_summary,
)


# Decision rule fixed before inspecting results: take the lowest name cap whose
# net information ratio is no more than 0.05 below the 4% baseline. If no lower
# cap qualifies, retain 4%. Sharpe is reported but never enters the decision.
NAME_CAPS = (0.040, 0.035, 0.030, 0.025)
BASELINE_CAP = 0.040
MAX_INFORMATION_RATIO_COST = 0.05


@dataclass
class NameCapSensitivity:
    """Backtests and comparison tables for the four name-cap variants."""

    summary: pd.DataFrame
    monthly_beta: pd.DataFrame
    sector_weights: pd.DataFrame
    backtests: dict[str, PortfolioBacktest]
    selected_cap: float


def _variant_name(cap: float) -> str:
    """Return a stable label for one percentage cap."""
    return f"benchmark_beta_cap_{cap * 100:.1f}pct"


def _apply_decision_rule(summary: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Select the lowest cap within 0.05 net IR of the 4% baseline."""
    baseline_ir = float(
        summary.loc[
            summary["name_cap"].eq(BASELINE_CAP),
            "net_information_ratio",
        ].iloc[0]
    )
    output = summary.copy()
    output["information_ratio_cost_vs_4pct"] = (
        baseline_ir - output["net_information_ratio"]
    )
    output["within_0.05_ir_of_4pct"] = output[
        "information_ratio_cost_vs_4pct"
    ].le(MAX_INFORMATION_RATIO_COST + 1e-12)
    qualifying = output.loc[output["within_0.05_ir_of_4pct"]]
    selected_cap = (
        float(qualifying["name_cap"].min())
        if len(qualifying)
        else BASELINE_CAP
    )
    output["selected"] = output["name_cap"].eq(selected_cap)
    return output, selected_cap


def run_name_cap_sensitivity(
    panel: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    write: bool = True,
) -> NameCapSensitivity:
    """Run and report the four precommitted name-cap variants.

    Args:
        panel: Monthly point-in-time research panel.
        factors: Monthly factor and benchmark-return rows.
        write: Whether to write date-stamped output files.

    Returns:
        Computed result as ``NameCapSensitivity``.
    """
    if panel is None or factors is None:
        local_panel, local_factors = load_inputs()
        panel = local_panel if panel is None else panel
        factors = local_factors if factors is None else factors
    print("DECISION RULE (fixed before results)")
    print(
        "Choose the lowest name cap whose net information ratio is within "
        "0.05 of the 4% benchmark-weight/beta-constrained baseline. If none "
        "qualifies, retain 4%. Sharpe does not enter the decision."
    )
    composite_score = build_model_score(panel)
    market_beta = compute_market_betas(panel)
    backtests: dict[str, PortfolioBacktest] = {}
    summary_rows: list[dict[str, object]] = []
    beta_pieces: list[pd.DataFrame] = []
    sector_pieces: list[pd.DataFrame] = []
    for cap in NAME_CAPS:
        variant = _variant_name(cap)
        print(f"Running {variant}...")
        construction = construct_portfolio(
            panel,
            composite_score=composite_score,
            market_beta=market_beta,
            weight_scheme="benchmark",
            apply_beta_constraint=True,
            max_name_weight=cap,
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
        summary = _variant_summary(
            variant,
            "benchmark",
            True,
            backtest,
        )
        summary["name_cap"] = cap
        summary_rows.append(summary)
        beta_pieces.append(
            _monthly_beta_table(variant, "benchmark", True, backtest).assign(
                name_cap=cap
            )
        )
        sector_pieces.append(
            _sector_table(variant, "benchmark", True, backtest).assign(
                name_cap=cap
            )
        )
    summary, selected_cap = _apply_decision_rule(pd.DataFrame(summary_rows))
    monthly_beta = pd.concat(beta_pieces, ignore_index=True)
    sector_weights = pd.concat(sector_pieces, ignore_index=True)
    result = NameCapSensitivity(
        summary=summary,
        monthly_beta=monthly_beta,
        sector_weights=sector_weights,
        backtests=backtests,
        selected_cap=selected_cap,
    )
    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(DATA_DIR / "portfolio_name_cap_sensitivity.csv", index=False)
        monthly_beta.to_csv(
            DATA_DIR / "portfolio_name_cap_monthly_beta.csv", index=False
        )
        sector_weights.to_csv(
            DATA_DIR / "portfolio_name_cap_sector_weights.csv", index=False
        )
    print("\nNAME-CAP SUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDECISION")
    print(f"Selected MAX_NAME_WEIGHT = {selected_cap:.1%}")
    return result


def main() -> None:
    """Run the four-cap benchmark-weight/beta-constrained comparison.

    Args:
        None.

    Returns:
        None.
    """
    run_name_cap_sensitivity()


if __name__ == "__main__":
    main()
