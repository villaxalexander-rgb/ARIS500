"""Build one configured monthly target portfolio and an explained trade list.

Reads ``data/panel_russell3000.parquet`` and ``data/current_portfolio.csv``;
writes date-stamped ``data/target_portfolio_YYYYMMDD.csv`` and
``data/trade_list_YYYYMMDD.csv`` without overwriting current holdings. It uses
the same scores, selection threshold, holdings band, weighting, caps, and beta
constraint as the backtest, while surfacing quarantine, stale-fundamental,
price, and size warnings rather than silently removing operational exceptions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BUY_PERCENTILE,
    CURRENT_PORTFOLIO_PATH,
    DATA_DIR,
    FIRST_REBALANCE,
    HOLDINGS_BAND,
    MODEL_BLOCKS,
    PANEL_RUSSELL3000_OUTPUT_PATH,
    REBALANCE_STALE_FUNDAMENTAL_MONTHS,
    REBALANCE_WARNING_MIN_MARKET_CAP,
    REBALANCE_WARNING_MIN_PRICE,
    SELL_PERCENTILE,
    TARGET_HOLDINGS,
)
from src.model.composite import CompositeScores, build_composite_scores
from src.model.portfolio import (
    _active_share,
    _benchmark_weights,
    _natural_holdings,
    _buy_mask,
    _sell_mask,
    compute_market_betas,
    _constrained_weights,
    _rank_deciles,
    _sector_label,
    _select_holdings,
    _terminal_rows,
    _validate_settings,
)


_WEIGHT_TOLERANCE = 1e-10


@dataclass
class RebalanceResult:
    """Target, trades, diagnostics, and written paths for one rebalance date."""

    date: pd.Timestamp
    target: pd.DataFrame
    trades: pd.DataFrame
    sector_weights: pd.DataFrame
    summary: pd.DataFrame
    checks: pd.DataFrame
    composite: CompositeScores
    target_path: Path | None = None
    trade_path: Path | None = None


def load_panel(path: Path = PANEL_RUSSELL3000_OUTPUT_PATH) -> pd.DataFrame:
    """Load the extended Russell 3000 research panel from local storage.

    Args:
        path: Input or output filesystem path.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing Russell 3000 panel: {path}")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def load_current_portfolio(
    path: Path = CURRENT_PORTFOLIO_PATH,
) -> pd.DataFrame:
    """Load current weights, allowing cash only for an explicit first launch.

    Args:
        path: Input or output filesystem path.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if not path.exists():
        if not FIRST_REBALANCE:
            raise FileNotFoundError(
                f"Current portfolio is missing at {path}. Replace it with actual "
                "filled positions before rebalancing. Set FIRST_REBALANCE = True "
                "only for the portfolio's explicit first launch."
            )
        print(f"FIRST_REBALANCE=True: starting from cash because {path} is missing.")
        return pd.DataFrame(
            {
                "permno": pd.Series(dtype="int64"),
                "weight": pd.Series(dtype="float64"),
            }
        )
    current = pd.read_csv(path)
    missing = {"permno", "weight"} - set(current.columns)
    if missing:
        raise KeyError(
            f"Current portfolio is missing required columns: {sorted(missing)}"
        )
    current = current.loc[:, ["permno", "weight"]].copy()
    if current.empty:
        if not FIRST_REBALANCE:
            raise ValueError(
                f"Current portfolio at {path} is empty. Replace it with actual "
                "filled positions before rebalancing. Set FIRST_REBALANCE = True "
                "only for the portfolio's explicit first launch."
            )
        print(f"FIRST_REBALANCE=True: starting from cash because {path} is empty.")
        return current.astype({"permno": "int64", "weight": "float64"})
    current["permno"] = pd.to_numeric(current["permno"], errors="raise").astype(
        "int64"
    )
    current["weight"] = pd.to_numeric(current["weight"], errors="raise").astype(
        "float64"
    )
    if current["permno"].duplicated().any():
        duplicates = current.loc[current["permno"].duplicated(False), "permno"]
        raise ValueError(
            "Current portfolio contains duplicate PERMNOs: "
            f"{sorted(duplicates.unique().tolist())}"
        )
    if (~np.isfinite(current["weight"])).any() or current["weight"].lt(0).any():
        raise ValueError("Current portfolio weights must be finite and nonnegative.")
    current = current.loc[current["weight"].gt(_WEIGHT_TOLERANCE)].copy()
    if current["weight"].sum() > 1.0 + 1e-6:
        raise ValueError(
            "Current portfolio weights exceed 100%: "
            f"{current['weight'].sum():.6%}."
        )
    print(
        f"Loaded {len(current):,} current holdings from {path}; "
        f"security weights sum to {current['weight'].sum():.2%}."
    )
    return current.sort_values("permno", ignore_index=True)


def resolve_rebalance_date(
    panel: pd.DataFrame,
    requested_date: str | pd.Timestamp | None,
) -> pd.Timestamp:
    """Resolve an optional requested calendar month to its latest panel date.

    Args:
        panel: Monthly point-in-time research panel.
        requested_date: Requested calendar date or month.

    Returns:
        Computed result as ``pd.Timestamp``.
    """
    dates = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    if dates.empty:
        raise ValueError("The panel contains no dates.")
    if requested_date is None:
        return pd.Timestamp(dates.max())
    requested = pd.Timestamp(requested_date)
    same_month = dates[dates.to_period("M") == requested.to_period("M")]
    if same_month.empty:
        raise ValueError(
            f"No panel observation exists in requested month {requested:%Y-%m}."
        )
    return pd.Timestamp(same_month.max())


def _selection_with_reasons(
    ranked: pd.DataFrame,
    current_permnos: set[int],
) -> tuple[pd.DataFrame, set[int], set[int]]:
    """Run the shared selector and identify guardrail additions and drops."""
    natural = _natural_holdings(ranked, current_permnos)
    selected = _select_holdings(ranked, current_permnos)
    natural_permnos = set(natural["permno"].astype(int))
    selected_permnos = set(selected["permno"].astype(int))
    return (
        selected,
        selected_permnos - natural_permnos,
        natural_permnos - selected_permnos,
    )


def _latest_metadata(
    panel: pd.DataFrame,
    date: pd.Timestamp,
    permnos: set[int],
) -> pd.DataFrame:
    """Return the latest available security labels on or before the rebalance."""
    if not permnos:
        return pd.DataFrame(
            columns=["permno", "ticker", "company_name", "sector"]
        )
    history = panel.loc[
        panel["permno"].isin(permnos) & panel["date"].le(date),
        ["permno", "date", "ticker", "company_name", "gsector"],
    ]
    latest = history.sort_values(["permno", "date"]).groupby(
        "permno", as_index=False
    ).tail(1)
    return latest.assign(sector=_sector_label(latest["gsector"]))[
        ["permno", "ticker", "company_name", "sector"]
    ]


def _turnover(current: pd.DataFrame, target: pd.DataFrame) -> float:
    """Calculate one-way turnover, including any current cash allocation."""
    current_weights = current.set_index("permno")["weight"]
    target_weights = target.set_index("permno")["weight"]
    names = current_weights.index.union(target_weights.index)
    security_trades = target_weights.reindex(names, fill_value=0.0) - current_weights.reindex(
        names, fill_value=0.0
    )
    current_cash = max(0.0, 1.0 - float(current_weights.sum()))
    target_cash = max(0.0, 1.0 - float(target_weights.sum()))
    return float(
        0.5 * (security_trades.abs().sum() + abs(target_cash - current_cash))
    )


def _build_trade_list(
    panel: pd.DataFrame,
    date: pd.Timestamp,
    month: pd.DataFrame,
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    weighted: pd.DataFrame,
    current: pd.DataFrame,
    resize_additions: set[int],
    resize_drops: set[int],
) -> pd.DataFrame:
    """Compare current and target weights and explain every nonzero trade."""
    current_weights = current.set_index("permno")["weight"]
    target_weights = weighted.set_index("permno")["weight"]
    names = current_weights.index.union(target_weights.index)
    comparison = pd.DataFrame(index=names)
    comparison.index.name = "permno"
    comparison["current_weight"] = current_weights.reindex(names, fill_value=0.0)
    comparison["target_weight"] = target_weights.reindex(names, fill_value=0.0)
    comparison["weight_change"] = (
        comparison["target_weight"] - comparison["current_weight"]
    )
    comparison = comparison.loc[
        comparison["weight_change"].abs().gt(_WEIGHT_TOLERANCE)
    ].reset_index()
    output_columns = [
        "permno",
        "ticker",
        "company_name",
        "sector",
        "trade_type",
        "current_weight",
        "target_weight",
        "weight_change",
        "reason",
        "composite_score",
        "decile",
        *MODEL_BLOCKS,
    ]
    if comparison.empty:
        return pd.DataFrame(columns=output_columns)

    target_ids = set(weighted["permno"].astype(int))
    current_ids = set(current["permno"].astype(int))
    selected_ids = set(selected["permno"].astype(int))
    buy_ids = set(ranked.loc[_buy_mask(ranked), "permno"].astype(int))
    sell_ids = set(ranked.loc[_sell_mask(ranked), "permno"].astype(int))
    ranked_lookup = ranked.set_index("permno")
    month_lookup = month.set_index("permno")
    raw_weight = weighted.set_index("permno")["raw_target_weight"]
    constrained = weighted.set_index("permno")["weight"]
    cap_adjusted = set(
        constrained.index[(constrained - raw_weight.reindex(constrained.index)).abs().gt(1e-8)]
        .astype(int)
        .tolist()
    )

    def trade_type(row: pd.Series) -> str:
        """Classify a target-current weight difference."""
        if row["current_weight"] <= _WEIGHT_TOLERANCE:
            return "buy"
        if row["target_weight"] <= _WEIGHT_TOLERANCE:
            return "sell"
        return "weight_increase" if row["weight_change"] > 0 else "weight_decrease"

    def reason(row: pd.Series) -> str:
        """Explain the selection or weighting event behind a trade."""
        permno = int(row["permno"])
        reasons: list[str] = []
        if permno in target_ids and permno not in current_ids:
            if permno in resize_additions:
                reasons.append("resize")
            elif permno in buy_ids:
                reasons.append(f"entered top {BUY_PERCENTILE:.0%}")
            else:
                reasons.append("selected")
        elif permno in current_ids and permno not in target_ids:
            if permno in selected_ids:
                reasons.append("below minimum position weight")
            elif permno in resize_drops:
                reasons.append("resize")
            elif permno in ranked_lookup.index and permno not in sell_ids:
                reasons.append(f"fell below top {SELL_PERCENTILE:.0%}")
            elif permno not in month_lookup.index:
                reasons.append("not in current panel month")
            elif not bool(month_lookup.at[permno, "investable"]):
                if bool(month_lookup.at[permno, "quarantined"]):
                    reasons.append("quarantined")
                else:
                    reasons.append("no longer investable")
            elif pd.isna(month_lookup.at[permno, "composite_score"]):
                reasons.append("missing composite score")
            elif bool(month_lookup.at[permno, "terminal_row"]):
                reasons.append("terminal observation")
            else:
                reasons.append("not selected")
        else:
            reasons.append(
                "cap" if permno in cap_adjusted else "benchmark-weight rebalance"
            )
        if permno in target_ids and permno in cap_adjusted and "cap" not in reasons:
            reasons.append("cap")
        return "; ".join(reasons)

    comparison["trade_type"] = comparison.apply(trade_type, axis=1)
    comparison["reason"] = comparison.apply(reason, axis=1)
    metadata = _latest_metadata(panel, date, set(comparison["permno"].astype(int)))
    score_columns = ["permno", "composite_score", "decile", *MODEL_BLOCKS]
    scores = ranked.loc[:, score_columns]
    comparison = comparison.merge(metadata, on="permno", how="left", validate="1:1")
    comparison = comparison.merge(scores, on="permno", how="left", validate="1:1")
    return comparison.loc[:, output_columns].sort_values(
        ["trade_type", "weight_change", "permno"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def _build_checks(
    panel: pd.DataFrame,
    month: pd.DataFrame,
    weighted: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build live-data safety checks and their affected-security details."""
    target = weighted.copy()
    quarantined = target.loc[target["quarantined"].fillna(False)].copy()
    report_date = pd.to_datetime(target["rdq"], errors="coerce")
    stale_cutoff = date - pd.DateOffset(
        months=REBALANCE_STALE_FUNDAMENTAL_MONTHS
    )
    stale = target.loc[report_date.isna() | report_date.lt(stale_cutoff)].copy()
    stale = stale.assign(fundamental_rdq=report_date.loc[stale.index])
    warning_thresholds = target.loc[
        pd.to_numeric(target["price"], errors="coerce").lt(
            REBALANCE_WARNING_MIN_PRICE
        )
        | pd.to_numeric(target["market_cap"], errors="coerce").lt(
            REBALANCE_WARNING_MIN_MARKET_CAP
        )
    ].copy()
    latest_panel_date = pd.to_datetime(panel["date"]).max()
    universe_rdq = pd.to_datetime(
        month.loc[month["in_universe"].fillna(False), "rdq"], errors="coerce"
    ).max()
    provisional = target.loc[target["provisional"].fillna(False)].copy()
    checks = pd.DataFrame(
        [
            {
                "check": "target contains quarantined prices",
                "status": "PASS" if quarantined.empty else "FAIL",
                "value": len(quarantined),
                "note": "expected 0",
            },
            {
                "check": (
                    "fundamental RDQ missing or more than "
                    f"{REBALANCE_STALE_FUNDAMENTAL_MONTHS} months stale"
                ),
                "status": "PASS" if stale.empty else "WARN",
                "value": len(stale),
                "note": f"RDQ cutoff {stale_cutoff.date()}",
            },
            {
                "check": (
                    f"price below ${REBALANCE_WARNING_MIN_PRICE:g} or market cap "
                    f"below ${REBALANCE_WARNING_MIN_MARKET_CAP:g}m"
                ),
                "status": "PASS" if warning_thresholds.empty else "WARN",
                "value": len(warning_thresholds),
                "note": "warning thresholds above the investability floor",
            },
            {
                "check": "latest panel price date",
                "status": "INFO",
                "value": latest_panel_date.date(),
                "note": f"rebalance uses {date.date()}",
            },
            {
                "check": "latest attached fundamental RDQ in universe",
                "status": "INFO",
                "value": universe_rdq.date() if pd.notna(universe_rdq) else "missing",
                "note": "point-in-time at rebalance date",
            },
            {
                "check": "target names using provisional prices",
                "status": "INFO",
                "value": len(provisional),
                "note": f"of {len(target)} holdings",
            },
        ]
    )
    detail_columns = [
        "permno",
        "ticker",
        "company_name",
        "price",
        "market_cap",
        "rdq",
        "provisional",
        "quarantined",
    ]
    details = {
        "quarantined": quarantined.loc[:, detail_columns],
        "stale_fundamentals": stale.loc[
            :, [*detail_columns, "fundamental_rdq"]
        ],
        "low_price_or_cap": warning_thresholds.loc[:, detail_columns],
    }
    return checks, details


def build_rebalance(
    panel: pd.DataFrame,
    current: pd.DataFrame,
    rebalance_date: str | pd.Timestamp | None = None,
) -> tuple[RebalanceResult, dict[str, pd.DataFrame]]:
    """Compute one target portfolio and trade list from the configured model.

    Args:
        panel: Monthly point-in-time research panel.
        current: Current portfolio weights.
        rebalance_date: Formation date; None selects the latest panel month.

    Returns:
        Computed result as ``tuple[RebalanceResult, dict[str, pd.DataFrame]]``.
    """
    _validate_settings()
    required = {
        "permno",
        "date",
        "ticker",
        "company_name",
        "gsector",
        "market_cap",
        "price",
        "in_universe",
        "investable",
        "provisional",
        "quarantined",
        "rdq",
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Panel is missing rebalance columns: {sorted(missing)}")
    date = resolve_rebalance_date(panel, rebalance_date)
    composite = build_composite_scores(panel)
    market_beta = compute_market_betas(panel)
    scored = panel.copy()
    scored["terminal_row"] = _terminal_rows(panel)
    scored["composite_score"] = composite.selected_score
    scored["market_beta"] = market_beta
    for block in MODEL_BLOCKS:
        scored[block] = composite.blocks[block]
    month = scored.loc[scored["date"].eq(date)].copy()
    if month.empty:
        raise ValueError(f"No panel rows are available at {date.date()}.")
    candidates = month.loc[
        month["investable"].fillna(False)
        & month["composite_score"].notna()
        & pd.to_numeric(month["market_cap"], errors="coerce").gt(0)
        & ~month["terminal_row"]
    ].copy()
    if candidates.empty:
        raise ValueError(f"No investable scored names are available at {date.date()}.")
    ranked = _rank_deciles(candidates)
    current_permnos = set(current["permno"].astype(int))
    selected, resize_additions, resize_drops = _selection_with_reasons(
        ranked, current_permnos
    )
    benchmark = _benchmark_weights(month)
    weighted, sector_weights, weight_diagnostics = _constrained_weights(
        selected, benchmark
    )
    weighted = weighted.assign(date=date)
    target = weighted.assign(sector=_sector_label(weighted["gsector"]))
    target_columns = [
        "permno",
        "ticker",
        "company_name",
        "sector",
        "weight",
        "composite_score",
        "decile",
        "market_beta",
        *MODEL_BLOCKS,
    ]
    target_output = target.loc[:, target_columns].sort_values(
        ["weight", "composite_score", "permno"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    trades = _build_trade_list(
        panel,
        date,
        month,
        ranked,
        selected,
        weighted,
        current,
        resize_additions,
        resize_drops,
    )
    turnover = _turnover(current, weighted)
    active_share = _active_share(weighted, benchmark)
    weighted_average_market_cap = float(
        (weighted["weight"] * weighted["market_cap"]).sum()
    )
    summary = pd.DataFrame(
        [
            {"metric": "rebalance date", "value": date.date()},
            {"metric": "investable scored names", "value": len(ranked)},
            {"metric": "current holdings", "value": len(current)},
            {"metric": "target holdings", "value": len(weighted)},
            {"metric": "target weight", "value": float(weighted["weight"].sum())},
            {"metric": "turnover vs current", "value": turnover},
            {"metric": "active share", "value": active_share},
            {
                "metric": "pre-constraint beta",
                "value": weight_diagnostics.pre_constraint_beta,
            },
            {
                "metric": "post-constraint beta",
                "value": weight_diagnostics.post_constraint_beta,
            },
            {
                "metric": "beta constraint bound",
                "value": weight_diagnostics.beta_constraint_bound,
            },
            {
                "metric": "weighted average market cap ($m)",
                "value": weighted_average_market_cap,
            },
            {"metric": "resize additions", "value": len(resize_additions)},
            {"metric": "resize drops", "value": len(resize_drops)},
        ]
    )
    checks, details = _build_checks(panel, month, weighted, date)
    result = RebalanceResult(
        date=date,
        target=target_output,
        trades=trades,
        sector_weights=sector_weights.sort_values("sector", ignore_index=True),
        summary=summary,
        checks=checks,
        composite=composite,
    )
    return result, details


def write_outputs(
    result: RebalanceResult,
    overwrite: bool = False,
) -> RebalanceResult:
    """Write validated date-stamped outputs without silent replacement.

    Args:
        result: Previously constructed result object.
        overwrite: Explicitly replace dated files that already exist.

    Returns:
        The result with its written target and trade paths attached.
    """
    stamp = result.date.strftime("%Y%m%d")
    target_path = DATA_DIR / f"target_portfolio_{stamp}.csv"
    trade_path = DATA_DIR / f"trade_list_{stamp}.csv"
    hard_failures = result.checks.loc[result.checks["status"].eq("FAIL")]
    if not hard_failures.empty:
        names = ", ".join(hard_failures["check"].astype(str))
        raise RuntimeError(
            "Rebalance has hard check failures; no target or trade list was "
            f"written. Failed checks: {names}"
        )
    existing = [path for path in (target_path, trade_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite approved dated output(s): "
            f"{existing}. Pass overwrite=True (or --overwrite) only after "
            "explicitly deciding to replace them."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.target.to_csv(target_path, index=False)
    result.trades.to_csv(trade_path, index=False)
    result.target_path = target_path
    result.trade_path = trade_path
    return result


def _print_result(
    result: RebalanceResult,
    details: dict[str, pd.DataFrame],
) -> None:
    """Print the live target summary, checks, exposures, and largest positions."""
    print(f"\nMONTHLY REBALANCE — {result.date.date()}")
    print(result.summary.to_string(index=False))
    print("\nCHECKS")
    print(result.checks.to_string(index=False))
    for name, detail in details.items():
        if detail.empty:
            continue
        print(f"\n{name.upper().replace('_', ' ')}")
        print(detail.to_string(index=False))
    print("\nSECTOR WEIGHTS VERSUS BENCHMARK")
    sector_display = result.sector_weights.copy()
    for column in (
        "benchmark_sector_weight",
        "portfolio_sector_weight",
        "sector_deviation",
    ):
        sector_display[column] = pd.to_numeric(
            sector_display[column], errors="coerce"
        ).astype("float64")
    print(
        sector_display.to_string(
            index=False,
            formatters={
                "benchmark_sector_weight": "{:.2%}".format,
                "portfolio_sector_weight": "{:.2%}".format,
                "sector_deviation": "{:+.2%}".format,
            },
        )
    )
    print("\nTOP 10 POSITIONS")
    print(
        result.target.head(10).to_string(
            index=False,
            formatters={"weight": "{:.2%}".format},
        )
    )
    trade_counts = result.trades.groupby("trade_type").size().rename("trades")
    print("\nTRADE COUNTS")
    print(trade_counts.to_string())
    if result.target_path is not None:
        print(f"\nWrote target: {result.target_path}")
    if result.trade_path is not None:
        print(f"Wrote trades: {result.trade_path}")
    print(f"Current portfolio was not modified: {CURRENT_PORTFOLIO_PATH}")


def run_rebalance(
    rebalance_date: str | pd.Timestamp | None = None,
    current_path: Path = CURRENT_PORTFOLIO_PATH,
    write: bool = True,
    overwrite: bool = False,
) -> RebalanceResult:
    """Run, print, and optionally write the configured one-date rebalance.

    Args:
        rebalance_date: Formation date; None selects the latest panel month.
        current_path: CSV path containing current portfolio weights.
        write: Whether to write date-stamped output files.
        overwrite: Explicitly replace existing dated outputs when writing.

    Returns:
        Computed result as ``RebalanceResult``.
    """
    panel = load_panel()
    current = load_current_portfolio(current_path)
    result, details = build_rebalance(panel, current, rebalance_date)
    if write:
        result = write_outputs(result, overwrite=overwrite)
    _print_result(result, details)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the latest month, or the month supplied with ``--date``.

    Args:
        argv: Optional command-line arguments.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        help="Rebalance month/date; defaults to the latest panel month.",
    )
    parser.add_argument(
        "--current-portfolio",
        type=Path,
        default=CURRENT_PORTFOLIO_PATH,
        help="CSV containing permno and weight.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing dated target and trade files.",
    )
    args = parser.parse_args(argv)
    run_rebalance(args.date, args.current_portfolio, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
