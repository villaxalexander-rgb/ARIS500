"""Provide plain-language confirmations and review tables for workflow notebooks.

This module reads only objects returned by the production functions in ``src``
and writes no files. It keeps presentation checks out of notebooks without
duplicating data pulls, signal definitions, portfolio rules, or backtest logic;
all confirmations describe existing results rather than recomputing the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

import config


def settings_table() -> pd.DataFrame:
    """Return the settings a notebook operator should verify before running.

    Args:
        None.

    Returns:
        A two-column table of current workflow settings.
    """
    settings = {
        "sample start": config.START_DATE,
        "sample end": config.END_DATE or "latest available",
        "universe": config.UNIVERSE,
        "use live extension": config.USE_LIVE_EXTENSION,
        "live-price staleness limit": f"{config.LIVE_PRICE_MAX_STALENESS_DAYS} days",
        "monthly fresh-price limit": (
            f"{config.LIVE_PRICE_REBALANCE_MAX_AGE_DAYS} days"
        ),
        "model blocks": ", ".join(config.MODEL_BLOCKS),
        "composite weighting": config.COMPOSITE_WEIGHTING,
        "portfolio buy threshold": f"top {config.BUY_PERCENTILE:.0%}",
        "portfolio sell threshold": f"top {config.SELL_PERCENTILE:.0%}",
        "portfolio target": f"{config.TARGET_HOLDINGS} +/- {config.HOLDINGS_BAND}",
        "name cap": f"{config.MAX_NAME_WEIGHT:.1%}",
        "minimum position weight": f"{config.MIN_POSITION_WEIGHT:.2%}",
        "explicit first rebalance": config.FIRST_REBALANCE,
        "beta band": f"{config.MIN_BETA:.2f}–{config.MAX_BETA:.2f}",
        "WRDS username": config.WRDS_USERNAME,
        "data folder": str(config.DATA_DIR),
    }
    return pd.DataFrame(settings.items(), columns=["setting", "current value"])


def confirm_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Print and return the checks that confirm the historical price pull worked.

    Args:
        prices: Output returned by the historical price pull.

    Returns:
        A table of checks, observed values, expectations, and statuses.
    """
    dates = pd.to_datetime(prices["date"])
    duplicates = int(prices.duplicated(["permno", "date"]).sum())
    checks = pd.DataFrame(
        [
            ("rows", f"{len(prices):,}", "more than 1,000,000", len(prices) > 1_000_000),
            (
                "date range",
                f"{dates.min():%Y-%m} to {dates.max():%Y-%m}",
                "begins by 2003 and includes 2025",
                dates.min() <= pd.Timestamp("2003-01-31")
                and dates.max() >= pd.Timestamp("2025-12-01"),
            ),
            (
                "unique securities",
                f"{prices['permno'].nunique():,}",
                "roughly 10,000–20,000",
                10_000 <= prices["permno"].nunique() <= 20_000,
            ),
            ("duplicate security-months", f"{duplicates:,}", "0", duplicates == 0),
        ],
        columns=["check", "observed", "expected", "passed"],
    )
    checks = checks.assign(status=np.where(checks["passed"], "PASS", "CHECK"))
    imputed = int(prices.get("delret_imputed", pd.Series(False, index=prices.index)).sum())
    print(
        f"Prices ready: {len(prices):,} rows, {dates.min():%Y-%m} to "
        f"{dates.max():%Y-%m}, {prices['permno'].nunique():,} securities, "
        f"{imputed:,} flagged delisting imputations."
    )
    print(f"Confirmation: {(checks['status'] == 'PASS').sum()}/{len(checks)} checks passed.")
    return checks.drop(columns="passed")


def confirm_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Print and return coverage checks for the Compustat pull.

    Args:
        fundamentals: Combined annual and quarterly Compustat output.

    Returns:
        Counts by reporting frequency.
    """
    dates = pd.to_datetime(fundamentals["datadate"])
    counts = (
        fundamentals.groupby("freq", as_index=False)
        .agg(rows=("gvkey", "size"), unique_companies=("gvkey", "nunique"))
        .sort_values("freq", ignore_index=True)
    )
    duplicates = int(fundamentals.duplicated(["gvkey", "datadate", "freq"]).sum())
    status = (
        "PASS"
        if len(fundamentals) > 1_000_000
        and fundamentals["gvkey"].nunique() > 20_000
        and set(counts["freq"]) == {"A", "Q"}
        and duplicates == 0
        else "CHECK"
    )
    print(
        f"Fundamentals ready: {len(fundamentals):,} rows, "
        f"{fundamentals['gvkey'].nunique():,} companies, {len(fundamentals.columns):,} "
        f"columns, {dates.min():%Y-%m} to {dates.max():%Y-%m}."
    )
    print(
        f"Confirmation: {status}. Expected both annual and quarterly rows, "
        f">20,000 companies, and 0 duplicate company-date-frequency rows; observed "
        f"{duplicates:,} duplicates."
    )
    return counts


def confirm_factors(factors: pd.DataFrame) -> pd.DataFrame:
    """Print and return coverage checks for factor and benchmark returns.

    Args:
        factors: Monthly factor and benchmark return output.

    Returns:
        A one-row factor-file summary.
    """
    dates = pd.to_datetime(factors["date"])
    expected = {"mkt_excess", "smb", "hml", "rmw", "cma", "umd", "rf"}
    missing = sorted(expected.difference(factors.columns))
    duplicates = int(dates.duplicated().sum())
    summary = pd.DataFrame(
        [
            {
                "rows": len(factors),
                "first month": dates.min().strftime("%Y-%m"),
                "last month": dates.max().strftime("%Y-%m"),
                "missing required columns": ", ".join(missing) or "none",
                "duplicate months": duplicates,
            }
        ]
    )
    status = "PASS" if not missing and duplicates == 0 else "CHECK"
    print(
        f"Factors ready: {len(factors):,} months, {dates.min():%Y-%m} to "
        f"{dates.max():%Y-%m}. Confirmation: {status}; all seven required factor "
        f"series should be present and duplicate months should be 0."
    )
    return summary


def confirm_live_prices(live_prices: pd.DataFrame) -> pd.DataFrame:
    """Print and return monthly coverage for the provisional live-price pull.

    Args:
        live_prices: Output returned by the live-price pull.

    Returns:
        Monthly row, security, and quarantine counts.
    """
    if live_prices.empty:
        raise ValueError("The live-price pull returned no rows.")
    dates = pd.to_datetime(live_prices["date"])
    monthly = (
        live_prices.assign(_month=dates.dt.to_period("M"))
        .groupby("_month", as_index=False)
        .agg(
            latest_price_date=("date", "max"),
            rows=("permno", "size"),
            names=("permno", "nunique"),
            quarantined=("quarantined", "sum"),
        )
        .rename(columns={"_month": "month"})
    )
    latest_age = (pd.Timestamp.now().normalize() - dates.max().normalize()).days
    status = "PASS" if latest_age <= config.LIVE_PRICE_MAX_STALENESS_DAYS else "CHECK"
    print(
        f"Live prices ready: {len(live_prices):,} rows across {len(monthly):,} months, "
        f"latest {dates.max():%Y-%m-%d} ({latest_age} days old), "
        f"{int(live_prices['quarantined'].sum()):,} quarantined rows."
    )
    print(
        f"Confirmation: {status}. Latest data must be no more than "
        f"{config.LIVE_PRICE_MAX_STALENESS_DAYS} days old before the panel can build."
    )
    return monthly


def enforce_live_price_recency(
    live_prices: pd.DataFrame,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Print and enforce the monthly rebalance's live-price recency limit.

    Args:
        live_prices: Output returned by a fresh live-price pull.
        max_age_days: Maximum permitted calendar-day age of the newest price;
            defaults to ``config.LIVE_PRICE_REBALANCE_MAX_AGE_DAYS``.

    Returns:
        A one-row table containing the latest price date, today's date, and age.
    """
    if max_age_days is None:
        max_age_days = config.LIVE_PRICE_REBALANCE_MAX_AGE_DAYS
    if live_prices.empty:
        raise RuntimeError("The fresh live-price pull returned no rows.")
    if max_age_days < 0:
        raise ValueError("max_age_days must be nonnegative.")
    latest = pd.to_datetime(live_prices["date"]).max().normalize()
    today = pd.Timestamp.now().normalize()
    age_days = int((today - latest).days)
    print("\n" + "=" * 72)
    print(f"LATEST AVAILABLE PRICE DATE: {latest:%Y-%m-%d}")
    print(f"TODAY: {today:%Y-%m-%d} | PRICE AGE: {age_days} CALENDAR DAY(S)")
    print(f"MONTHLY REBALANCE LIMIT: {max_age_days} CALENDAR DAYS")
    print("=" * 72)
    if age_days > max_age_days:
        raise RuntimeError(
            "STOP: the freshly queried live prices are "
            f"{age_days} calendar days old, beyond the monthly rebalance limit "
            f"of {max_age_days}. Do not build the panel or generate trades until "
            "the WRDS source is current."
        )
    return pd.DataFrame(
        [
            {
                "latest available price date": latest.date(),
                "today": today.date(),
                "age in calendar days": age_days,
                "maximum permitted age": max_age_days,
                "status": "PASS",
            }
        ]
    )


def announce_rebalance_signal_date(panel: pd.DataFrame) -> pd.Timestamp:
    """Print and return the panel date that an undated rebalance will use.

    Args:
        panel: Completed point-in-time research panel.

    Returns:
        Latest panel date, which is the default rebalance signal date.
    """
    if panel.empty:
        raise RuntimeError("The panel is empty, so no rebalance date is available.")
    signal_date = pd.to_datetime(panel["date"]).max()
    print("\n" + "=" * 72)
    print(f"REBALANCE SIGNAL DATE: {signal_date:%Y-%m-%d}")
    print("The target portfolio will use information available on this date.")
    print("=" * 72)
    return signal_date


def quarantine_review(
    live_prices: pd.DataFrame,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Return every quarantined live-price row for mandatory human review.

    Args:
        live_prices: Output returned by the live-price pull.
        latest_only: Whether to restrict the review to the newest live month.

    Returns:
        Review columns for quarantined security-months.
    """
    columns = [
        column
        for column in (
            "permno",
            "ticker",
            "company_name",
            "date",
            "ret",
            "price",
            "market_cap",
            "quarantined",
        )
        if column in live_prices.columns
    ]
    candidates = live_prices
    if latest_only and not live_prices.empty:
        dates = pd.to_datetime(live_prices["date"])
        months = dates.dt.to_period("M")
        candidates = live_prices.loc[months.eq(months.max())]
    review = candidates.loc[candidates["quarantined"].fillna(False), columns].copy()
    return review.sort_values(["date", "permno"], ignore_index=True)


def confirm_panel(panel: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    """Print the panel dimensions, latest investability, and validation outcome.

    Args:
        panel: Completed point-in-time research panel.
        checks: Check table returned by the panel builder.

    Returns:
        The supplied validation table, ordered with failures first.
    """
    dates = pd.to_datetime(panel["date"])
    latest = dates.max()
    latest_investable = int(panel.loc[dates.eq(latest), "investable"].sum())
    status = checks["pass/fail"].astype(str)
    tiers = checks.get("tier", pd.Series("WARN", index=checks.index)).astype(str)
    passed = int(status.eq("PASS").sum())
    blocking = int(status.eq("FAIL").mul(tiers.eq("FAIL")).sum())
    warnings = int(status.eq("FAIL").mul(tiers.eq("WARN")).sum())
    duplicates = int(panel.duplicated(["permno", "date"]).sum())
    print(
        f"Panel built: {len(panel):,} rows, {dates.min():%Y-%m} to "
        f"{latest:%Y-%m}, {panel['permno'].nunique():,} securities."
    )
    print(f"Latest month: {latest_investable:,} investable names; expected roughly 2,000–3,000.")
    print(
        f"Checks: {passed} passed, {blocking} blocking failures, {warnings} warnings; "
        f"duplicate security-months: {duplicates}."
    )
    if blocking:
        print("STOP: do not continue to research or rebalance until blocking failures are resolved.")
    elif warnings:
        print("Panel is usable, but read every warning before continuing.")
    else:
        print("Confirmation: PASS. The panel is ready for downstream notebooks.")
    order = status.map({"FAIL": 0, "PASS": 1}).fillna(2)
    return checks.assign(_order=order).sort_values(["_order", "tier"]).drop(columns="_order")


def confirm_harness(harness: pd.DataFrame) -> pd.DataFrame:
    """Print whether the random and perfect-foresight alignment tests passed.

    Args:
        harness: Harness table returned by signal research.

    Returns:
        The unchanged harness table.
    """
    failures = int(harness["pass/fail"].ne("PASS").sum())
    print(
        f"Alignment harness: {len(harness) - failures}/{len(harness)} tests passed, "
        f"{failures} failed. Expected: random signals look random and perfect "
        f"foresight has rank IC at least 0.95."
    )
    if failures:
        print("STOP: a harness failure means signal/return alignment may be wrong.")
    return harness


def confirm_evaluation(name: str, evaluation: Any) -> pd.DataFrame:
    """Print headline coverage, IC, spread, and turnover for one evaluation.

    Args:
        name: Plain-language label for the signal or block.
        evaluation: EvaluationResult or ExtendedEvaluation returned by research.

    Returns:
        A one-row headline table.
    """
    base = getattr(evaluation, "base", evaluation)
    ic = base.ic_summary.iloc[0]
    spread = base.spread_summary.iloc[0]
    row: dict[str, Any] = {
        "signal or block": name,
        "mean coverage": base.coverage["coverage"].mean(),
        "mean rank IC": ic["mean"],
        "IC t-stat": ic["t_stat"],
        "EW spread Sharpe": spread["annualized_sharpe"],
        "top-quintile turnover": base.turnover["turnover"].mean(),
    }
    if hasattr(evaluation, "value_weighted_spread_summary"):
        row["VW spread Sharpe"] = evaluation.value_weighted_spread_summary.iloc[0][
            "annualized_sharpe"
        ]
    table = pd.DataFrame([row])
    print(
        f"{name} evaluated: coverage {row['mean coverage']:.1%}, mean IC "
        f"{row['mean rank IC']:.4f}, IC t-stat {row['IC t-stat']:.2f}, "
        f"EW spread Sharpe {row['EW spread Sharpe']:.2f}. A research-strength "
        f"IC t-stat is about 3 or higher; weaker results require independent "
        f"economic evidence and must not be hidden."
    )
    return table


def confirm_composite(research: Any) -> pd.DataFrame:
    """Print the selected composite's evidence and structural risk diagnostic.

    Args:
        research: CompositeResearch returned by composite research.

    Returns:
        A one-row summary for each tested weighting scheme.
    """
    rows = []
    for scheme, result in research.evaluations.items():
        base = result.base
        rows.append(
            {
                "scheme": scheme,
                "mean coverage": base.coverage["coverage"].mean(),
                "mean rank IC": base.ic_summary.iloc[0]["mean"],
                "IC t-stat": base.ic_summary.iloc[0]["t_stat"],
                "EW spread Sharpe": base.spread_summary.iloc[0]["annualized_sharpe"],
            }
        )
    table = pd.DataFrame(rows)
    maximum_risk_share = float(research.risk_contribution["risk_share"].max())
    print(
        f"Composite research complete: {len(table)} weighting schemes tested; "
        f"configured scheme is {research.construction.selected_scheme!r}. Largest "
        f"single-block risk contribution is {maximum_risk_share:.1%}; the structural "
        f"warning threshold is about 40%."
    )
    return table


def confirm_portfolio(backtest: Any) -> pd.DataFrame:
    """Print the current gross/net backtest confirmation and implementation ranges.

    Args:
        backtest: PortfolioBacktest returned by portfolio research.

    Returns:
        The gross and net performance table.
    """
    performance = backtest.performance.copy()
    net = performance.set_index("return_basis").loc["net"]
    monthly = backtest.monthly_returns
    holdings_mean = float(monthly["holdings"].mean())
    beta_min = float(monthly["post_constraint_beta"].min())
    beta_max = float(monthly["post_constraint_beta"].max())
    band_bind_rate = float(monthly["holdings_band_bound"].mean())
    print(
        f"Backtest complete: {int(net['months'])} monthly returns; net annualized "
        f"return {net['annualized_return']:.2%}, volatility "
        f"{net['annualized_volatility']:.2%}, Sharpe {net['sharpe']:.2f}, "
        f"information ratio {net['information_ratio']:.2f}."
    )
    print(
        f"Implementation check: average {holdings_mean:.1f} holdings; realized beta "
        f"range {beta_min:.2f}–{beta_max:.2f}, expected inside "
        f"{config.MIN_BETA:.2f}–{config.MAX_BETA:.2f}."
    )
    print(
        f"Turnover: {net['average_monthly_drift_adjusted_turnover']:.2%} "
        f"drift-adjusted and "
        f"{net['average_monthly_name_turnover']:.2%} by names; the holdings "
        f"guardrail bound in {band_bind_rate:.1%} of months."
    )
    return performance


def confirm_analysis(analysis: Any) -> pd.DataFrame:
    """Print active attribution, concentration, and top-ten diagnostics.

    Args:
        analysis: PortfolioAnalysis returned by the analysis module.

    Returns:
        Active-factor coefficient table.
    """
    from src.model.portfolio_analysis import subperiod_stability

    monthly = analysis.portfolio.monthly_returns
    _, _, _, concentration = subperiod_stability(monthly)
    active = analysis.factor_regressions["active_return"]
    top_ten = pd.to_numeric(monthly["top_10_weight"], errors="coerce")
    print(
        f"Analysis complete: active alpha {active.annualized_alpha:.2%} with "
        f"Newey-West t-stat {active.alpha_t_stat:.2f}; active-regression R-squared "
        f"{active.r_squared:.1%}."
    )
    print(
        f"Fragility check: {concentration['months_to_half_cumulative_log_excess']} "
        f"of {concentration['months']} months produced half the cumulative log "
        f"excess return. Top-ten weight averaged {top_ten.mean():.1%} and peaked "
        f"at {top_ten.max():.1%}."
    )
    return active.coefficients


def confirm_rebalance(result: Any) -> pd.DataFrame:
    """Print output paths, holdings, trade counts, and safety-check statuses.

    Args:
        result: RebalanceResult returned by the monthly rebalance.

    Returns:
        The unchanged rebalance check table.
    """
    counts = result.trades.groupby("trade_type").size().to_dict()
    failures = int(result.checks["status"].eq("FAIL").sum())
    warnings = int(result.checks["status"].eq("WARN").sum())
    print(
        f"Rebalance produced: {len(result.target):,} target holdings and "
        f"{len(result.trades):,} trade rows for {result.date:%Y-%m-%d}; "
        f"trade counts {counts}."
    )
    print(
        f"Safety checks: {failures} failures, {warnings} warnings. Target: "
        f"{result.target_path}; trades: {result.trade_path}."
    )
    print(f"Current holdings were NOT changed: {config.CURRENT_PORTFOLIO_PATH}.")
    if failures:
        print("STOP: do not approve trades while any safety check is FAIL.")
    return result.checks


def trade_review(result: Any) -> Mapping[str, pd.DataFrame]:
    """Return the target, trades, sectors, and checks for manual review.

    Args:
        result: RebalanceResult returned by the monthly rebalance.

    Returns:
        Named review tables for display in a notebook.
    """
    return {
        "checks": result.checks,
        "trades": result.trades,
        "target portfolio": result.target,
        "sector weights": result.sector_weights,
    }


def display_stock_report(report: Any) -> None:
    """Display the latest stock rankings and history figures once.

    Args:
        report: StockRankingReport returned by the stock lookup module.

    Returns:
        None.
    """
    from IPython.display import display

    print("CURRENT SUMMARY")
    display(report.summary)
    print("CURRENT FACTOR RANKINGS")
    display(report.latest_factors)
    print("CURRENT INDIVIDUAL SIGNAL RANKINGS")
    display(report.latest_signals)
    for figure in report.figures.values():
        display(figure)
    ticker = report.ticker
    date = pd.to_datetime(report.date)
    print(
        f"Lookup complete: {ticker} through {date:%Y-%m-%d}; "
        f"{len(report.latest_factors)} factor ranks, "
        f"{len(report.latest_signals)} individual-signal ranks, and "
        f"{len(report.figures)} history charts."
    )
