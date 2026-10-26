"""Produce allocator-style diagnostics for the configured portfolio backtest.

Reads ``data/panel_russell3000.parquet`` and ``data/factors.parquet`` and writes
no files; it returns performance, stability, factor/sector/style attribution,
cost sensitivity, findings, and figures for ``notebooks/04_backtest.ipynb``.
Newey-West inference, lagged portfolio weights, and explicit concentration
diagnostics are used so apparent alpha is not confused with factor exposure or
a handful of unusually strong months.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from config import (
    BUY_PERCENTILE,
    COMPOSITE_START,
    COMPOSITE_WEIGHTING,
    HAC_LAGS,
    HOLDINGS_BAND,
    MAX_NAME_WEIGHT,
    MAX_SECTOR_DEVIATION,
    MODEL_BLOCKS,
    PANEL_RUSSELL3000_OUTPUT_PATH,
    ROLLING_MONTHS,
    SELL_PERCENTILE,
    TARGET_HOLDINGS,
    TRADING_COST_BPS,
)
from src.utils import forward_returns
from src.model.composite import CompositeScores, build_composite_scores
from src.model.portfolio import (
    PortfolioBacktest,
    PortfolioConstruction,
    _sector_label,
    backtest_portfolio,
    construct_portfolio,
    load_inputs,
)


FACTOR_COLUMNS = ("mkt_excess", "smb", "hml", "rmw", "cma", "umd")
SECTOR_NAMES = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer discretionary",
    "30": "Consumer staples",
    "35": "Health care",
    "40": "Financials",
    "45": "Information technology",
    "50": "Communication services",
    "55": "Utilities",
    "60": "Real estate",
    "Unknown": "Unknown",
}


@dataclass
class RegressionResult:
    """Coefficient table and fit statistics from one HAC regression."""

    coefficients: pd.DataFrame
    observations: int
    r_squared: float
    annualized_alpha: float
    alpha_t_stat: float


@dataclass
class PortfolioAnalysis:
    """Tables, findings, and figures from the complete portfolio analysis."""

    portfolio: PortfolioBacktest
    composite: CompositeScores
    headline: pd.DataFrame
    subperiod_headline: pd.DataFrame
    yearly_headline: pd.DataFrame
    rolling_stability: pd.DataFrame
    factor_regressions: dict[str, RegressionResult]
    rolling_factor_betas: pd.DataFrame
    sector_full_period: pd.DataFrame
    sector_by_year: pd.DataFrame
    sector_by_sector: pd.DataFrame
    sector_weights: pd.DataFrame
    size_weights: pd.DataFrame
    market_cap_profile: pd.DataFrame
    style_profile: pd.DataFrame
    block_contribution: pd.DataFrame
    block_regression_summary: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    findings: list[str]
    figures: dict[str, plt.Figure]


def _maximum_drawdown(returns: pd.Series) -> float:
    """Return maximum drawdown for a monthly return series."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _annual_returns(dates: pd.Series, returns: pd.Series) -> pd.Series:
    """Compound monthly returns into calendar-year returns."""
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "return": returns})
    return frame.groupby(frame["date"].dt.year)["return"].apply(
        lambda values: (1.0 + values).prod() - 1.0
    )


def _performance_metrics(
    dates: pd.Series,
    returns: pd.Series,
    benchmark: pd.Series,
    rf: pd.Series,
    active_share: pd.Series,
) -> dict[str, object]:
    """Calculate headline performance and benchmark-relative diagnostics."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "return": pd.to_numeric(returns, errors="coerce"),
            "benchmark": pd.to_numeric(benchmark, errors="coerce"),
            "rf": pd.to_numeric(rf, errors="coerce"),
            "active_share": pd.to_numeric(active_share, errors="coerce"),
        }
    ).dropna(subset=["return", "benchmark", "rf"])
    count = len(frame)
    excess = frame["return"] - frame["rf"]
    active = frame["return"] - frame["benchmark"]
    volatility = frame["return"].std(ddof=1) * np.sqrt(12.0)
    excess_volatility = excess.std(ddof=1) * np.sqrt(12.0)
    tracking_error = active.std(ddof=1) * np.sqrt(12.0)
    rolling_year = (1.0 + frame["return"]).rolling(12).apply(np.prod, raw=True) - 1.0
    annual = _annual_returns(frame["date"], frame["return"])
    worst_month_index = frame["return"].idxmin()
    worst_rolling_index = rolling_year.idxmin() if rolling_year.notna().any() else None
    up = frame["benchmark"].gt(0)
    down = frame["benchmark"].lt(0)
    return {
        "months": count,
        "annualized_return": (1.0 + frame["return"]).prod() ** (12.0 / count) - 1.0,
        "annualized_volatility": volatility,
        "sharpe": excess.mean() * 12.0 / excess_volatility
        if excess_volatility > 0
        else np.nan,
        "tracking_error": tracking_error,
        "information_ratio": active.mean() * 12.0 / tracking_error
        if tracking_error > 0
        else np.nan,
        "average_active_share": frame["active_share"].mean(),
        "maximum_drawdown": _maximum_drawdown(frame["return"]),
        "worst_month": frame.loc[worst_month_index, "return"],
        "worst_month_date": frame.loc[worst_month_index, "date"],
        "worst_rolling_12_months": rolling_year.loc[worst_rolling_index]
        if worst_rolling_index is not None
        else np.nan,
        "worst_rolling_12_end": frame.loc[worst_rolling_index, "date"]
        if worst_rolling_index is not None
        else pd.NaT,
        "best_calendar_year": annual.max(),
        "best_calendar_year_date": int(annual.idxmax()),
        "worst_calendar_year": annual.min(),
        "worst_calendar_year_date": int(annual.idxmin()),
        "benchmark_hit_rate": frame["return"].gt(frame["benchmark"]).mean(),
        "up_capture": frame.loc[up, "return"].mean()
        / frame.loc[up, "benchmark"].mean()
        if up.any()
        else np.nan,
        "down_capture": frame.loc[down, "return"].mean()
        / frame.loc[down, "benchmark"].mean()
        if down.any()
        else np.nan,
    }


def headline_performance(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return gross, net, and benchmark headline performance.

    Args:
        monthly: Monthly strategy and benchmark return history.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    rows: list[dict[str, object]] = []
    for label, column in (("strategy_gross", "gross_return"), ("strategy_net", "net_return")):
        rows.append(
            {
                "series": label,
                **_performance_metrics(
                    monthly["return_date"],
                    monthly[column],
                    monthly["benchmark_return"],
                    monthly["rf"],
                    monthly["active_share"],
                ),
            }
        )
    benchmark_metrics = _performance_metrics(
        monthly["return_date"],
        monthly["benchmark_return"],
        monthly["benchmark_return"],
        monthly["rf"],
        pd.Series(0.0, index=monthly.index),
    )
    benchmark_metrics.update(
        {
            "tracking_error": 0.0,
            "information_ratio": np.nan,
            "average_active_share": 0.0,
            "benchmark_hit_rate": np.nan,
            "up_capture": 1.0,
            "down_capture": 1.0,
        }
    )
    rows.append({"series": "russell3000", **benchmark_metrics})
    return pd.DataFrame(rows)


def subperiod_stability(
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | bool]]:
    """Return split-sample, yearly, rolling, and concentration diagnostics.

    Args:
        monthly: Monthly strategy and benchmark return history.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | bool]]``.
    """
    dates = pd.to_datetime(monthly["return_date"])
    subperiod_rows: list[pd.DataFrame] = []
    for label, start, end in (
        ("2007-2015", "2007-01-01", "2015-12-31"),
        ("2016-2025", "2016-01-01", "2025-12-31"),
    ):
        subset = monthly.loc[dates.between(start, end)].copy()
        table = headline_performance(subset)
        table.insert(0, "period", label)
        subperiod_rows.append(table)
    subperiod = pd.concat(subperiod_rows, ignore_index=True)

    yearly_rows: list[dict[str, object]] = []
    for year, group in monthly.groupby(dates.dt.year, sort=True):
        metrics = _performance_metrics(
            group["return_date"],
            group["net_return"],
            group["benchmark_return"],
            group["rf"],
            group["active_share"],
        )
        yearly_rows.append({"year": int(year), **metrics})
    yearly = pd.DataFrame(yearly_rows)

    active = monthly["net_return"] - monthly["benchmark_return"]
    rolling = pd.DataFrame(
        {
            "date": dates,
            "annualized_excess_return": active.rolling(ROLLING_MONTHS).mean() * 12.0,
            "information_ratio": active.rolling(ROLLING_MONTHS).mean()
            / active.rolling(ROLLING_MONTHS).std(ddof=1)
            * np.sqrt(12.0),
        }
    )

    log_excess = np.log1p(monthly["net_return"]) - np.log1p(
        monthly["benchmark_return"]
    )
    total_log_excess = float(log_excess.sum())
    positive = log_excess.loc[log_excess.gt(0)].sort_values(ascending=False)
    if total_log_excess > 0 and not positive.empty:
        months_to_half = int(
            np.searchsorted(positive.cumsum().to_numpy(), total_log_excess / 2.0) + 1
        )
    else:
        months_to_half = 0
    concentration = {
        "months": len(monthly),
        "positive_active_months": int(active.gt(0).sum()),
        "positive_active_month_share": float(active.gt(0).mean()),
        "months_to_half_cumulative_log_excess": months_to_half,
        "months_to_half_share": months_to_half / len(monthly),
        "more_than_half_from_fewer_than_quarter_months": bool(
            months_to_half > 0 and months_to_half < len(monthly) / 4.0
        ),
        "positive_rolling_36_month_share": float(
            rolling["annualized_excess_return"].dropna().gt(0).mean()
        ),
    }
    return subperiod, yearly, rolling, concentration


def _newey_west_regression(
    dependent: pd.Series,
    factors: pd.DataFrame,
    lags: int = HAC_LAGS,
) -> RegressionResult:
    """Fit OLS and calculate Bartlett-kernel Newey-West standard errors."""
    data = pd.concat([dependent.rename("dependent"), factors], axis=1).dropna()
    y = data["dependent"].to_numpy(dtype=float)
    factor_names = list(factors.columns)
    x = np.column_stack([np.ones(len(data)), data[factor_names].to_numpy(dtype=float)])
    inverse_xx = np.linalg.pinv(x.T @ x)
    beta = inverse_xx @ x.T @ y
    residual = y - x @ beta
    score = x * residual[:, None]
    meat = score.T @ score
    usable_lags = min(lags, len(data) - 1)
    for lag in range(1, usable_lags + 1):
        weight = 1.0 - lag / (usable_lags + 1.0)
        gamma = score[lag:].T @ score[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = inverse_xx @ meat @ inverse_xx
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_stat = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan),
        where=standard_error > 0,
    )
    centered = y - y.mean()
    r_squared = 1.0 - float(residual @ residual) / float(centered @ centered)
    terms = ["alpha", *factor_names]
    coefficients = pd.DataFrame(
        {
            "term": terms,
            "coefficient": beta,
            "newey_west_t_stat": t_stat,
            "annualized_effect": [
                beta[0] * 12.0,
                *[
                    beta[index + 1] * data[name].mean() * 12.0
                    for index, name in enumerate(factor_names)
                ],
            ],
        }
    )
    return RegressionResult(
        coefficients=coefficients,
        observations=len(data),
        r_squared=r_squared,
        annualized_alpha=float(beta[0] * 12.0),
        alpha_t_stat=float(t_stat[0]),
    )


def factor_attribution(
    monthly: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[dict[str, RegressionResult], pd.DataFrame, pd.DataFrame]:
    """Run total and active factor regressions plus rolling factor betas.

    Args:
        monthly: Monthly strategy and benchmark return history.
        factors: Monthly factor and benchmark-return rows.

    Returns:
        Computed result as ``tuple[dict[str, RegressionResult], pd.DataFrame, pd.DataFrame]``.
    """
    factor_frame = factors.copy().assign(date=pd.to_datetime(factors["date"]))
    factor_frame["return_month"] = factor_frame["date"].dt.to_period("M")
    work = monthly.copy()
    work["return_month"] = pd.to_datetime(work["return_date"]).dt.to_period("M")
    work = work.merge(
        factor_frame[["return_month", *FACTOR_COLUMNS]],
        on="return_month",
        how="left",
        validate="1:1",
    )
    x = work.loc[:, FACTOR_COLUMNS]
    regressions = {
        "strategy_excess_return": _newey_west_regression(
            work["net_return"] - work["rf"], x
        ),
        "active_return": _newey_west_regression(
            work["net_return"] - work["benchmark_return"], x
        ),
    }
    summary = pd.DataFrame(
        [
            {
                "model": name,
                "observations": result.observations,
                "r_squared": result.r_squared,
                "annualized_alpha": result.annualized_alpha,
                "alpha_t_stat": result.alpha_t_stat,
            }
            for name, result in regressions.items()
        ]
    )

    rolling_rows: list[dict[str, object]] = []
    complete = work.dropna(subset=["net_return", "benchmark_return", *FACTOR_COLUMNS])
    for end in range(ROLLING_MONTHS - 1, len(complete)):
        window = complete.iloc[end - ROLLING_MONTHS + 1 : end + 1]
        y = (window["net_return"] - window["benchmark_return"]).to_numpy(float)
        x_window = np.column_stack(
            [np.ones(len(window)), window.loc[:, FACTOR_COLUMNS].to_numpy(float)]
        )
        beta = np.linalg.pinv(x_window.T @ x_window) @ x_window.T @ y
        rolling_rows.append(
            {
                "date": window["return_date"].iloc[-1],
                **{name: beta[index + 1] for index, name in enumerate(FACTOR_COLUMNS)},
            }
        )
    return regressions, summary, pd.DataFrame(rolling_rows)


def _security_profile(
    panel: pd.DataFrame,
    portfolio: PortfolioBacktest,
    composite: CompositeScores,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build formation-month benchmark securities and held-security profiles."""
    scores = composite.blocks.copy()
    scores["composite"] = composite.selected_score
    security = panel.loc[
        :,
        ["permno", "date", "in_universe", "market_cap", "gsector"],
    ].copy()
    security = security.assign(
        date=pd.to_datetime(security["date"]),
        market_cap=pd.to_numeric(security["market_cap"], errors="coerce"),
        forward_return=forward_returns(panel),
        sector=_sector_label(security["gsector"]),
    )
    for column in scores:
        security[column] = scores[column].to_numpy()
    formation_dates = pd.DatetimeIndex(
        portfolio.construction.holdings["date"].drop_duplicates()
    )
    benchmark = security.loc[
        security["date"].isin(formation_dates)
        & security["in_universe"].fillna(False)
        & security["market_cap"].gt(0)
    ].copy()
    benchmark["benchmark_weight"] = benchmark["market_cap"] / benchmark.groupby(
        "date"
    )["market_cap"].transform("sum")
    benchmark = assign_benchmark_size_buckets(benchmark)

    profile_columns = [
        "permno",
        "date",
        "forward_return",
        "sector",
        "size_bucket",
        "market_cap",
        *scores.columns,
    ]
    holdings = portfolio.construction.holdings.merge(
        benchmark[profile_columns],
        on=["permno", "date"],
        how="left",
        validate="1:1",
        suffixes=("", "_panel"),
    )
    if "market_cap_panel" in holdings:
        holdings = holdings.drop(columns="market_cap_panel")
    return benchmark, holdings


def assign_benchmark_size_buckets(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Assign largest-500, next-1,000, and remaining benchmark size buckets.

    Args:
        benchmark: Benchmark securities with date, market cap, and PERMNO.

    Returns:
        A copy with deterministic ``size_rank`` and ``size_bucket`` columns.
    """
    required = {"date", "permno", "market_cap"}
    missing = required.difference(benchmark.columns)
    if missing:
        raise KeyError(f"Benchmark is missing size-bucket columns: {sorted(missing)}")
    output = benchmark.copy()
    output["size_rank"] = output.groupby("date")["market_cap"].rank(
        method="first", ascending=False
    )
    output["size_bucket"] = "small"
    output.loc[output["size_rank"].le(1500), "size_bucket"] = "mid"
    output.loc[output["size_rank"].le(500), "size_bucket"] = "large"
    return output


def sector_attribution(
    monthly: pd.DataFrame,
    benchmark: pd.DataFrame,
    holdings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform monthly Brinson-Fachler attribution on net portfolio returns.

    Args:
        monthly: Monthly strategy and benchmark return history.
        benchmark: Monthly benchmark return values.
        holdings: Holdings used by this operation.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]``.
    """
    monthly_map = monthly.set_index("formation_date")
    held = holdings.copy()
    held["trading_cost"] = held["date"].map(monthly_map["trading_cost"])
    held["net_security_return"] = (
        (1.0 - held["trading_cost"])
        * (1.0 + held["forward_return"].fillna(0.0))
        - 1.0
    )
    held["portfolio_contribution"] = held["weight"] * held["net_security_return"]
    portfolio_sector = held.groupby(["date", "sector"], as_index=False).agg(
        portfolio_weight=("weight", "sum"),
        portfolio_contribution=("portfolio_contribution", "sum"),
    )
    portfolio_sector["portfolio_sector_return"] = (
        portfolio_sector["portfolio_contribution"]
        / portfolio_sector["portfolio_weight"]
    )

    bmk = benchmark.copy()
    bmk["benchmark_contribution"] = (
        bmk["benchmark_weight"] * bmk["forward_return"].fillna(0.0)
    )
    benchmark_sector = bmk.groupby(["date", "sector"], as_index=False).agg(
        benchmark_weight=("benchmark_weight", "sum"),
        benchmark_contribution=("benchmark_contribution", "sum"),
    )
    benchmark_sector["benchmark_sector_return"] = (
        benchmark_sector["benchmark_contribution"]
        / benchmark_sector["benchmark_weight"]
    )
    benchmark_total = benchmark_sector.groupby("date")[
        "benchmark_contribution"
    ].sum()

    detail = benchmark_sector.merge(
        portfolio_sector,
        on=["date", "sector"],
        how="outer",
        validate="1:1",
    ).fillna(
        {
            "benchmark_weight": 0.0,
            "benchmark_contribution": 0.0,
            "benchmark_sector_return": 0.0,
            "portfolio_weight": 0.0,
            "portfolio_contribution": 0.0,
            "portfolio_sector_return": 0.0,
        }
    )
    detail["proxy_benchmark_return"] = detail["date"].map(benchmark_total)
    active_weight = detail["portfolio_weight"] - detail["benchmark_weight"]
    active_sector_return = (
        detail["portfolio_sector_return"] - detail["benchmark_sector_return"]
    )
    detail["allocation"] = active_weight * (
        detail["benchmark_sector_return"] - detail["proxy_benchmark_return"]
    )
    detail["selection"] = detail["benchmark_weight"] * active_sector_return
    detail["interaction"] = active_weight * active_sector_return

    monthly_effects = detail.groupby("date", as_index=False)[
        ["allocation", "selection", "interaction"]
    ].sum()
    monthly_effects["return_date"] = monthly_effects["date"].map(
        monthly_map["return_date"]
    )
    monthly_effects["year"] = pd.to_datetime(monthly_effects["return_date"]).dt.year
    monthly_effects["proxy_benchmark_return"] = monthly_effects["date"].map(
        benchmark_total
    )
    monthly_effects["published_benchmark_return"] = monthly_effects["date"].map(
        monthly_map["benchmark_return"]
    )
    monthly_effects["benchmark_residual"] = (
        monthly_effects["proxy_benchmark_return"]
        - monthly_effects["published_benchmark_return"]
    )
    monthly_effects["active_return"] = monthly_effects["date"].map(
        monthly_map["net_return"] - monthly_map["benchmark_return"]
    )
    effect_columns = ["allocation", "selection", "interaction", "benchmark_residual"]
    monthly_effects["reconciliation_error"] = (
        monthly_effects[effect_columns].sum(axis=1) - monthly_effects["active_return"]
    )
    full = pd.DataFrame(
        [
            {
                **{
                    f"annualized_{column}": monthly_effects[column].mean() * 12.0
                    for column in effect_columns
                },
                "annualized_active_return": monthly_effects["active_return"].mean()
                * 12.0,
                "maximum_monthly_reconciliation_error": monthly_effects[
                    "reconciliation_error"
                ].abs().max(),
            }
        ]
    )
    by_year = monthly_effects.groupby("year", as_index=False)[
        [*effect_columns, "active_return"]
    ].sum()

    sector_rows: list[dict[str, object]] = []
    month_count = len(monthly_effects)
    for sector, group in detail.groupby("sector", sort=True):
        sector_rows.append(
            {
                "sector": sector,
                "sector_name": SECTOR_NAMES.get(str(sector), str(sector)),
                "average_portfolio_weight": group["portfolio_weight"].sum()
                / month_count,
                "average_benchmark_weight": group["benchmark_weight"].sum()
                / month_count,
                "annualized_allocation": group["allocation"].sum()
                / month_count
                * 12.0,
                "annualized_selection": group["selection"].sum()
                / month_count
                * 12.0,
                "annualized_interaction": group["interaction"].sum()
                / month_count
                * 12.0,
            }
        )
    by_sector = pd.DataFrame(sector_rows)
    sector_weights = detail.loc[
        :,
        ["date", "sector", "portfolio_weight", "benchmark_weight"],
    ].copy()
    sector_weights["sector_name"] = sector_weights["sector"].map(
        lambda value: SECTOR_NAMES.get(str(value), str(value))
    )
    return full, by_year, by_sector, sector_weights, monthly_effects


def size_and_style_profile(
    benchmark: pd.DataFrame,
    holdings: pd.DataFrame,
    score_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare portfolio and benchmark size, market cap, and signal exposures.

    Args:
        benchmark: Monthly benchmark return values.
        holdings: Holdings used by this operation.
        score_columns: Score columns used by this operation.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]``.
    """
    benchmark_size = benchmark.groupby(["date", "size_bucket"], as_index=False)[
        "benchmark_weight"
    ].sum()
    portfolio_size = holdings.groupby(["date", "size_bucket"], as_index=False)[
        "weight"
    ].sum()
    size_weights = benchmark_size.merge(
        portfolio_size,
        on=["date", "size_bucket"],
        how="outer",
        validate="1:1",
    ).fillna(0.0).rename(columns={"weight": "portfolio_weight"})

    benchmark_cap = (
        benchmark.assign(
            _weighted_market_cap=benchmark["benchmark_weight"] * benchmark["market_cap"]
        )
        .groupby("date", as_index=False)["_weighted_market_cap"]
        .sum()
        .rename(columns={"_weighted_market_cap": "benchmark_weighted_market_cap"})
    )
    portfolio_cap = (
        holdings.assign(_weighted_market_cap=holdings["weight"] * holdings["market_cap"])
        .groupby("date", as_index=False)["_weighted_market_cap"]
        .sum()
        .rename(columns={"_weighted_market_cap": "portfolio_weighted_market_cap"})
    )
    market_cap = benchmark_cap.merge(
        portfolio_cap, on="date", how="outer", validate="1:1"
    )

    style_rows: list[pd.DataFrame] = []
    for score in score_columns:
        benchmark_valid = benchmark.loc[benchmark[score].notna()].copy()
        benchmark_valid["_weighted_score"] = (
            benchmark_valid["benchmark_weight"] * benchmark_valid[score]
        )
        bmk = benchmark_valid.groupby("date").agg(
            benchmark_numerator=("_weighted_score", "sum"),
            benchmark_coverage=("benchmark_weight", "sum"),
        )
        bmk["benchmark_average"] = (
            bmk["benchmark_numerator"] / bmk["benchmark_coverage"]
        )

        portfolio_valid = holdings.loc[holdings[score].notna()].copy()
        portfolio_valid["_weighted_score"] = (
            portfolio_valid["weight"] * portfolio_valid[score]
        )
        port = portfolio_valid.groupby("date").agg(
            portfolio_numerator=("_weighted_score", "sum"),
            portfolio_coverage=("weight", "sum"),
        )
        port["portfolio_average"] = (
            port["portfolio_numerator"] / port["portfolio_coverage"]
        )
        style_rows.append(
            bmk[["benchmark_average", "benchmark_coverage"]]
            .join(port[["portfolio_average", "portfolio_coverage"]], how="outer")
            .reset_index()
            .assign(score=score)
        )
    return size_weights, market_cap, pd.concat(style_rows, ignore_index=True)


def block_contribution(
    monthly: pd.DataFrame,
    spreads: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regress net active returns on the five block Q5-minus-Q1 spreads.

    Args:
        monthly: Monthly strategy and benchmark return history.
        spreads: Monthly block long-short spread returns.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame]``.
    """
    spread_frame = spreads.copy().reset_index().rename(columns={"date": "formation_date"})
    work = monthly.merge(
        spread_frame,
        on="formation_date",
        how="left",
        validate="1:1",
    )
    active = work["net_return"] - work["benchmark_return"]
    regression = _newey_west_regression(active, work.loc[:, MODEL_BLOCKS])
    coefficients = regression.coefficients.copy()
    spread_means = work.loc[:, MODEL_BLOCKS].mean()
    coefficients["annualized_contribution"] = np.nan
    coefficients.loc[coefficients["term"].eq("alpha"), "annualized_contribution"] = (
        regression.annualized_alpha
    )
    for block in MODEL_BLOCKS:
        beta = coefficients.loc[coefficients["term"].eq(block), "coefficient"].iloc[0]
        coefficients.loc[
            coefficients["term"].eq(block), "annualized_contribution"
        ] = beta * spread_means[block] * 12.0
    annualized_active = active.mean() * 12.0
    coefficients["share_of_average_active_return"] = (
        coefficients["annualized_contribution"] / annualized_active
    )
    summary = pd.DataFrame(
        [
            {
                "observations": regression.observations,
                "r_squared": regression.r_squared,
                "annualized_active_return": annualized_active,
                "annualized_intercept": regression.annualized_alpha,
                "intercept_t_stat": regression.alpha_t_stat,
            }
        ]
    )
    return coefficients, summary


def cost_sensitivity(monthly: pd.DataFrame) -> pd.DataFrame:
    """Reprice the same monthly holdings at multiples of configured costs.

    Args:
        monthly: Monthly strategy and benchmark return history.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    rows: list[dict[str, object]] = []
    for multiplier in (1.0, 1.5, 2.0, 3.0):
        net = (
            (1.0 - multiplier * monthly["trading_cost"])
            * (1.0 + monthly["gross_return"])
            - 1.0
        )
        metrics = _performance_metrics(
            monthly["return_date"],
            net,
            monthly["benchmark_return"],
            monthly["rf"],
            monthly["active_share"],
        )
        rows.append(
            {
                "cost_multiplier": multiplier,
                "annualized_return": metrics["annualized_return"],
                "sharpe": metrics["sharpe"],
                "tracking_error": metrics["tracking_error"],
                "information_ratio": metrics["information_ratio"],
                "annualized_cost_estimate": monthly["trading_cost"].mean()
                * 12.0
                * multiplier,
            }
        )
    return pd.DataFrame(rows)


def _analysis_findings(
    headline: pd.DataFrame,
    subperiod: pd.DataFrame,
    rolling: pd.DataFrame,
    concentration: dict[str, float | int | bool],
    regressions: dict[str, RegressionResult],
    sector_full: pd.DataFrame,
    block_summary: pd.DataFrame,
    costs: pd.DataFrame,
) -> list[str]:
    """Translate the main diagnostics into direct investment conclusions."""
    net = headline.set_index("series").loc["strategy_net"]
    benchmark = headline.set_index("series").loc["russell3000"]
    findings = [
        "Net annualized return was "
        f"{net['annualized_return']:.1%} versus {benchmark['annualized_return']:.1%} "
        f"for Russell 3000, with an information ratio of {net['information_ratio']:.2f}.",
    ]
    period_net = subperiod.loc[subperiod["series"].eq("strategy_net")].set_index("period")
    period_benchmark = subperiod.loc[subperiod["series"].eq("russell3000")].set_index(
        "period"
    )
    first_active = (
        period_net.loc["2007-2015", "annualized_return"]
        - period_benchmark.loc["2007-2015", "annualized_return"]
    )
    second_active = (
        period_net.loc["2016-2025", "annualized_return"]
        - period_benchmark.loc["2016-2025", "annualized_return"]
    )
    findings.append(
        "Annualized outperformance was "
        f"{first_active:.1%} in 2007-2015 and {second_active:.1%} in 2016-2025."
    )
    concentration_flag = concentration[
        "more_than_half_from_fewer_than_quarter_months"
    ]
    findings.append(
        f"Half of cumulative log excess return came from "
        f"{concentration['months_to_half_cumulative_log_excess']} of "
        f"{concentration['months']} months "
        f"({concentration['months_to_half_share']:.1%})."
    )
    if concentration_flag:
        findings.append(
            "FRAGILITY FLAG: more than half of cumulative excess return came "
            "from fewer than one quarter of the months."
        )
    else:
        findings.append(
            "Outperformance is not concentrated enough to trigger the stated "
            "quarter-of-months fragility rule."
        )
    findings.append(
        f"Rolling 36-month excess return was positive in "
        f"{concentration['positive_rolling_36_month_share']:.1%} of available windows."
    )

    active_regression = regressions["active_return"]
    active_coefficients = active_regression.coefficients.set_index("term")
    significant = active_coefficients.loc[
        (active_coefficients.index != "alpha")
        & active_coefficients["newey_west_t_stat"].abs().ge(2.0)
    ]
    findings.append(
        f"Factor-adjusted active alpha was {active_regression.annualized_alpha:.1%} "
        f"with Newey-West t-stat {active_regression.alpha_t_stat:.2f}; "
        f"the active factor regression R-squared was {active_regression.r_squared:.1%}."
    )
    factor_effect = active_coefficients.loc[
        active_coefficients.index != "alpha", "annualized_effect"
    ].sum()
    regression_mean = active_regression.annualized_alpha + factor_effect
    alpha_share = (
        active_regression.annualized_alpha / regression_mean
        if abs(regression_mean) > np.finfo(float).eps
        else np.nan
    )
    findings.append(
        f"At sample-average factor returns, the active regression attributes "
        f"{active_regression.annualized_alpha:.1%} to intercept alpha and "
        f"{factor_effect:.1%} to conventional factor tilts "
        f"(alpha share {alpha_share:.1%}); the alpha t-stat is "
        f"{'below' if abs(active_regression.alpha_t_stat) < 2.0 else 'above'} 2."
    )
    if significant.empty:
        findings.append(
            "No active factor tilt had an absolute Newey-West t-stat above 2; "
            "the return is not obviously a repackaged conventional factor exposure."
        )
    else:
        terms = ", ".join(
            f"{term} ({row['coefficient']:+.2f}, t={row['newey_west_t_stat']:.2f})"
            for term, row in significant.iterrows()
        )
        findings.append(f"Material active factor tilts: {terms}.")

    sector = sector_full.iloc[0]
    findings.append(
        "Annualized Brinson effects were "
        f"allocation {sector['annualized_allocation']:.2%}, "
        f"selection {sector['annualized_selection']:.2%}, and "
        f"interaction {sector['annualized_interaction']:.2%}; the proxy-to-published "
        f"benchmark residual was {sector['annualized_benchmark_residual']:.2%}."
    )
    block = block_summary.iloc[0]
    findings.append(
        f"The five block spreads explain {block['r_squared']:.1%} of monthly active-return "
        f"variance; the regression intercept is {block['annualized_intercept']:.1%} "
        f"(t={block['intercept_t_stat']:.2f})."
    )
    triple = costs.set_index("cost_multiplier").loc[3.0]
    findings.append(
        f"At three times modeled trading costs, annualized return is still "
        f"{triple['annualized_return']:.1%}, Sharpe {triple['sharpe']:.2f}, and "
        f"information ratio {triple['information_ratio']:.2f}."
    )
    if net["information_ratio"] > 1.0 or net["annualized_return"] - benchmark[
        "annualized_return"
    ] > 0.06:
        findings.append(
            "TOO-GOOD FLAG: the headline edge is unusually large and should be "
            "treated as a research result, not an allocator-ready expectation."
        )
    else:
        findings.append(
            "The headline edge is economically meaningful but not implausibly large; "
            "subperiod, factor, concentration, and cost diagnostics remain the key caveats."
        )
    return findings


def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the full drawdown path for plotting."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def build_charts(
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    rolling: pd.DataFrame,
    rolling_betas: pd.DataFrame,
    sector_weights: pd.DataFrame,
    size_weights: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> dict[str, plt.Figure]:
    """Build plain allocator-facing performance and exposure charts.

    Args:
        monthly: Monthly strategy and benchmark return history.
        yearly: Yearly used by this operation.
        rolling: Rolling used by this operation.
        rolling_betas: Rolling betas used by this operation.
        sector_weights: Sector weights used by this operation.
        size_weights: Size weights used by this operation.
        market_cap: Market cap used by this operation.

    Returns:
        Computed result as ``dict[str, plt.Figure]``.
    """
    dates = pd.to_datetime(monthly["return_date"])
    figures: dict[str, plt.Figure] = {}

    figure, axis = plt.subplots(figsize=(10, 5))
    for label, column in (
        ("Strategy gross", "gross_return"),
        ("Strategy net", "net_return"),
        ("Russell 3000", "benchmark_return"),
    ):
        axis.plot(dates, (1.0 + monthly[column]).cumprod(), label=label)
    axis.set_yscale("log")
    axis.set(title="Cumulative growth of $1", ylabel="Growth of $1", xlabel="")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figures["growth"] = figure

    figure, axis = plt.subplots(figsize=(10, 4))
    relative = (1.0 + monthly["net_return"]).cumprod() / (
        1.0 + monthly["benchmark_return"]
    ).cumprod() - 1.0
    axis.plot(dates, relative)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Cumulative net excess wealth versus Russell 3000", xlabel="")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figures["cumulative_excess"] = figure

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(dates, _drawdown_series(monthly["net_return"]), label="Strategy net")
    axis.plot(
        dates,
        _drawdown_series(monthly["benchmark_return"]),
        label="Russell 3000",
    )
    axis.set(title="Drawdowns", xlabel="")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figures["drawdown"] = figure

    figure, axis = plt.subplots(figsize=(10, 4))
    strategy_12 = (1.0 + monthly["net_return"]).rolling(12).apply(np.prod, raw=True) - 1.0
    benchmark_12 = (1.0 + monthly["benchmark_return"]).rolling(12).apply(
        np.prod, raw=True
    ) - 1.0
    axis.plot(dates, strategy_12 - benchmark_12)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Rolling 12-month excess return", xlabel="")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figures["rolling_12_excess"] = figure

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(rolling["date"], rolling["information_ratio"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Rolling 36-month information ratio", xlabel="")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figures["rolling_36_information_ratio"] = figure

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(rolling["date"], rolling["annualized_excess_return"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Rolling 36-month annualized excess return", xlabel="")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figures["rolling_36_excess_return"] = figure

    figure, axis = plt.subplots(figsize=(11, 5))
    annual = yearly.set_index("year")
    positions = np.arange(len(annual))
    width = 0.25
    axis.bar(positions - width, annual["gross_return"], width, label="Gross")
    axis.bar(positions, annual["net_return"], width, label="Net")
    axis.bar(
        positions + width,
        annual["russell3000_return"],
        width,
        label="Russell 3000",
    )
    axis.set_xticks(positions, annual.index, rotation=45)
    axis.set(title="Calendar-year returns", xlabel="")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    figures["annual_returns"] = figure

    figure, axis = plt.subplots(figsize=(10, 5))
    for factor in FACTOR_COLUMNS:
        axis.plot(rolling_betas["date"], rolling_betas[factor], label=factor)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Rolling 36-month active factor betas", xlabel="")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figures["rolling_factor_betas"] = figure

    sectors = sorted(sector_weights["sector_name"].unique())
    rows = int(np.ceil(len(sectors) / 3.0))
    figure, axes = plt.subplots(rows, 3, figsize=(13, rows * 2.6), sharex=True)
    flat_axes = np.asarray(axes).reshape(-1)
    for axis, sector in zip(flat_axes, sectors):
        frame = sector_weights.loc[sector_weights["sector_name"].eq(sector)]
        axis.plot(frame["date"], frame["portfolio_weight"], label="Portfolio")
        axis.plot(frame["date"], frame["benchmark_weight"], label="Benchmark")
        axis.set_title(sector, fontsize=9)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.2)
    for axis in flat_axes[len(sectors) :]:
        axis.set_visible(False)
    flat_axes[0].legend(fontsize=8)
    figure.suptitle("Sector weights over time", y=1.01)
    figure.tight_layout()
    figures["sector_weights"] = figure

    figure, axis = plt.subplots(figsize=(10, 4))
    second = axis.twinx()
    axis.plot(dates, monthly["holdings"], label="Holdings", color="tab:blue")
    second.plot(
        dates,
        monthly["turnover"],
        label="Turnover",
        color="tab:orange",
        alpha=0.8,
    )
    axis.set(title="Holdings count and monthly turnover", ylabel="Holdings", xlabel="")
    second.set_ylabel("Turnover")
    second.yaxis.set_major_formatter(PercentFormatter(1.0))
    lines = axis.lines + second.lines
    axis.legend(lines, [line.get_label() for line in lines], loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figures["holdings_turnover"] = figure

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, bucket in zip(axes, ("large", "mid", "small")):
        frame = size_weights.loc[size_weights["size_bucket"].eq(bucket)]
        axis.plot(frame["date"], frame["portfolio_weight"], label="Portfolio")
        axis.plot(frame["date"], frame["benchmark_weight"], label="Benchmark")
        axis.set_title(bucket.capitalize())
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Size-bucket weights")
    figure.tight_layout()
    figures["size_weights"] = figure

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        market_cap["date"],
        market_cap["portfolio_weighted_market_cap"],
        label="Portfolio",
    )
    axis.plot(
        market_cap["date"],
        market_cap["benchmark_weighted_market_cap"],
        label="Benchmark",
    )
    axis.set_yscale("log")
    axis.set(title="Weighted-average market capitalization", ylabel="USD millions")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figures["market_cap_profile"] = figure
    return figures


def _format_table(frame: pd.DataFrame) -> str:
    """Return a readable representation for printed notebook output."""
    return frame.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def _print_analysis(result: PortfolioAnalysis, factor_summary: pd.DataFrame) -> None:
    """Print all tables with direct findings for notebook and terminal users."""
    print("\nHEADLINE PERFORMANCE")
    print(_format_table(result.headline))
    print("\nSUBPERIOD STABILITY")
    print(_format_table(result.subperiod_headline))
    print("\nCALENDAR-YEAR STABILITY — NET STRATEGY")
    print(_format_table(result.yearly_headline))
    print("\nFACTOR REGRESSION SUMMARY")
    print(_format_table(factor_summary))
    for name, regression in result.factor_regressions.items():
        print(f"\nFACTOR COEFFICIENTS — {name}")
        print(_format_table(regression.coefficients))
    print("\nBRINSON ATTRIBUTION — FULL PERIOD")
    print(_format_table(result.sector_full_period))
    print("\nBRINSON ATTRIBUTION — BY YEAR")
    print(_format_table(result.sector_by_year))
    print("\nBRINSON ATTRIBUTION — BY SECTOR")
    print(_format_table(result.sector_by_sector))
    print("\nAVERAGE SIZE WEIGHTS")
    print(
        _format_table(
            result.size_weights.groupby("size_bucket", as_index=False)[
                ["portfolio_weight", "benchmark_weight"]
            ].mean()
        )
    )
    print("\nAVERAGE MARKET-CAP PROFILE — USD MILLIONS")
    print(_format_table(result.market_cap_profile.mean(numeric_only=True).to_frame().T))
    print("\nAVERAGE STYLE PROFILE")
    print(
        _format_table(
            result.style_profile.groupby("score", as_index=False)[
                [
                    "portfolio_average",
                    "benchmark_average",
                    "portfolio_coverage",
                    "benchmark_coverage",
                ]
            ].mean()
        )
    )
    print("\nBLOCK CONTRIBUTION REGRESSION")
    print(_format_table(result.block_regression_summary))
    print(_format_table(result.block_contribution))
    print("\nCOST SENSITIVITY")
    print(_format_table(result.cost_sensitivity))
    print("\nFINDINGS")
    for finding in result.findings:
        print(f"- {finding}")


def run_portfolio_analysis(
    panel: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
) -> PortfolioAnalysis:
    """Build the configured backtest and return the complete portfolio analysis.

    Args:
        panel: Monthly point-in-time research panel.
        factors: Monthly factor and benchmark-return rows.

    Returns:
        Computed result as ``PortfolioAnalysis``.
    """
    if panel is None or factors is None:
        local_panel, local_factors = load_inputs()
        panel = local_panel if panel is None else panel
        factors = local_factors if factors is None else factors

    composite = build_composite_scores(panel)
    construction: PortfolioConstruction = construct_portfolio(
        panel, composite.selected_score
    )
    monthly, performance, annual = backtest_portfolio(panel, factors, construction)
    portfolio = PortfolioBacktest(
        construction=construction,
        monthly_returns=monthly,
        performance=performance,
        annual_returns=annual,
    )

    headline = headline_performance(monthly)
    subperiod, yearly, rolling, concentration = subperiod_stability(monthly)
    regressions, factor_summary, rolling_betas = factor_attribution(monthly, factors)
    benchmark, holdings = _security_profile(panel, portfolio, composite)
    sector_full, sector_year, sector_by_sector, sector_weights, _ = sector_attribution(
        monthly, benchmark, holdings
    )
    score_columns = [*MODEL_BLOCKS, "composite"]
    size_weights, market_cap, style = size_and_style_profile(
        benchmark, holdings, score_columns
    )
    block_coefficients, block_summary = block_contribution(
        monthly, composite.block_spreads
    )
    costs = cost_sensitivity(monthly)
    findings = _analysis_findings(
        headline,
        subperiod,
        rolling,
        concentration,
        regressions,
        sector_full,
        block_summary,
        costs,
    )
    figures = build_charts(
        monthly,
        annual,
        rolling,
        rolling_betas,
        sector_weights,
        size_weights,
        market_cap,
    )
    result = PortfolioAnalysis(
        portfolio=portfolio,
        composite=composite,
        headline=headline,
        subperiod_headline=subperiod,
        yearly_headline=yearly,
        rolling_stability=rolling,
        factor_regressions=regressions,
        rolling_factor_betas=rolling_betas,
        sector_full_period=sector_full,
        sector_by_year=sector_year,
        sector_by_sector=sector_by_sector,
        sector_weights=sector_weights,
        size_weights=size_weights,
        market_cap_profile=market_cap,
        style_profile=style,
        block_contribution=block_coefficients,
        block_regression_summary=block_summary,
        cost_sensitivity=costs,
        findings=findings,
        figures=figures,
    )
    _print_analysis(result, factor_summary)
    return result


def main() -> None:
    """Run the complete configured portfolio analysis.

    Args:
        None.

    Returns:
        None.
    """
    print(
        "Configured portfolio: "
        f"{COMPOSITE_WEIGHTING} composite, buy top {BUY_PERCENTILE:.0%}, "
        f"sell below top {SELL_PERCENTILE:.0%}, target {TARGET_HOLDINGS} +/- {HOLDINGS_BAND}, "
        f"name cap {MAX_NAME_WEIGHT:.1%}, sector deviation "
        f"{MAX_SECTOR_DEVIATION:.1%}, costs {TRADING_COST_BPS}."
    )
    print(f"Panel: {PANEL_RUSSELL3000_OUTPUT_PATH}; start: {COMPOSITE_START}.")
    run_portfolio_analysis()


if __name__ == "__main__":
    main()
