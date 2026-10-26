"""Explain one stock's raw signals, factor ranks, composite, and portfolio status.

Reads ``data/panel_russell3000.parquet`` and writes no files; it returns tables
and figures used by ``notebooks/05_stock_lookup.ipynb``. Scores reuse the exact
production signal/composite functions, percentiles are recomputed within each
historical month, ticker ambiguity requires a PERMNO, and figures are closed
before return so Jupyter does not render them twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BUY_PERCENTILE,
    COMPOSITE_WEIGHTING,
    MODEL_BLOCKS,
    PANEL_RUSSELL3000_OUTPUT_PATH,
    SELL_PERCENTILE,
)
from src.model.composite import CompositeScores, build_composite_scores
from src.model.portfolio import (
    N_DECILES,
    PortfolioConstruction,
    _buy_mask,
    _sell_mask,
    _rank_deciles,
    _terminal_rows,
    compute_market_betas,
    construct_portfolio,
)
from src.signals.registry import required_panel_columns, validate_live_blocks


SECTOR_NAMES = {
    10: "Energy",
    15: "Materials",
    20: "Industrials",
    25: "Consumer Discretionary",
    30: "Consumer Staples",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology",
    50: "Communication Services",
    55: "Utilities",
    60: "Real Estate",
}

_LIVE_BLOCK_SPECS = validate_live_blocks(MODEL_BLOCKS)
BLOCK_NAMES = {spec.name: spec.label for spec in _LIVE_BLOCK_SPECS}
SIGNAL_NAMES = {
    signal.name: signal.label
    for spec in _LIVE_BLOCK_SPECS
    for signal in spec.signals
}
PERCENT_SIGNALS = {
    signal.name
    for spec in _LIVE_BLOCK_SPECS
    for signal in spec.signals
    if signal.display_as_percent
}

EXPLAIN_PANEL_COLUMNS = required_panel_columns(
    MODEL_BLOCKS,
    base=(
        "permno",
        "date",
        "ret",
        "price",
        "market_cap",
        "ticker",
        "company_name",
        "gsector",
        "in_universe",
        "investable",
        "primary_security",
        "secondary_share_class",
        "price_screen_pass",
        "market_cap_screen_pass",
        "adv_screen_pass",
        "quarantined",
        "provisional",
        "delret_imputed",
    ),
)


@dataclass
class RankingContext:
    """Reusable signal and composite state behind stock ranking reports."""

    panel: pd.DataFrame
    composite: CompositeScores
    raw_signals: pd.DataFrame
    standardized_signals: pd.DataFrame


@dataclass
class ExplainerContext(RankingContext):
    """Ranking state plus configured portfolio history for decision explanations."""

    construction: PortfolioConstruction


@dataclass
class StockExplanation:
    """Tables, narrative, and history chart for one stock at one date."""

    permno: int
    date: pd.Timestamp
    summary: pd.DataFrame
    block_breakdown: pd.DataFrame
    signal_detail: pd.DataFrame
    history: pd.DataFrame
    what_would_change: str
    figure: plt.Figure


@dataclass
class StockRankingReport:
    """Current rankings, long-form histories, and charts for one stock."""

    permno: int
    ticker: str
    date: pd.Timestamp
    summary: pd.DataFrame
    latest_factors: pd.DataFrame
    latest_signals: pd.DataFrame
    composite_history: pd.DataFrame
    factor_history: pd.DataFrame
    signal_history: pd.DataFrame
    figures: dict[str, plt.Figure]


def _sector_name(value: object) -> str:
    """Return a readable GICS sector label."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "Unknown"
    code = int(numeric)
    return SECTOR_NAMES.get(code, f"GICS {code}")


def load_panel() -> pd.DataFrame:
    """Load only columns needed by the explainer and existing model functions.

    Args:
        None.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if not PANEL_RUSSELL3000_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Russell 3000 panel: {PANEL_RUSSELL3000_OUTPUT_PATH}"
        )
    import pyarrow.parquet as pq

    schema_columns = set(
        pq.ParquetFile(PANEL_RUSSELL3000_OUTPUT_PATH).schema.names
    )
    missing = set(EXPLAIN_PANEL_COLUMNS) - schema_columns
    if missing:
        raise KeyError(f"Panel is missing explainer inputs: {sorted(missing)}")
    panel = pd.read_parquet(
        PANEL_RUSSELL3000_OUTPUT_PATH,
        columns=list(EXPLAIN_PANEL_COLUMNS),
    )
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def build_ranking_context(panel: pd.DataFrame | None = None) -> RankingContext:
    """Build all raw, standardized, factor, and composite scores once.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``RankingContext``.
    """
    source = load_panel() if panel is None else panel.copy()
    source["date"] = pd.to_datetime(source["date"])

    built = {spec.name: spec.build(source) for spec in _LIVE_BLOCK_SPECS}
    blocks = pd.DataFrame(
        {spec.name: spec.score(built[spec.name]) for spec in _LIVE_BLOCK_SPECS},
        index=source.index,
    )
    composite = build_composite_scores(source, blocks=blocks)
    raw = pd.concat(
        {spec.name: built[spec.name].raw for spec in _LIVE_BLOCK_SPECS}, axis=1
    )
    standardized = pd.concat(
        {
            spec.name: built[spec.name].standardized for spec in _LIVE_BLOCK_SPECS
        },
        axis=1,
    )
    return RankingContext(
        panel=source,
        composite=composite,
        raw_signals=raw,
        standardized_signals=standardized,
    )


def build_explainer_context(panel: pd.DataFrame | None = None) -> ExplainerContext:
    """Add configured portfolio history to the reusable ranking state.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Computed result as ``ExplainerContext``.
    """
    ranking = build_ranking_context(panel)
    source = ranking.panel
    source["_terminal_row"] = _terminal_rows(source)
    market_beta = compute_market_betas(source)
    construction = construct_portfolio(
        source,
        composite_score=ranking.composite.selected_score,
        market_beta=market_beta,
    )
    return ExplainerContext(
        panel=source,
        composite=ranking.composite,
        raw_signals=ranking.raw_signals,
        standardized_signals=ranking.standardized_signals,
        construction=construction,
    )


@lru_cache(maxsize=1)
def get_ranking_context() -> RankingContext:
    """Return cached scores without building unnecessary portfolio history.

    Args:
        None.

    Returns:
        Computed result as ``RankingContext``.
    """
    return build_ranking_context()


@lru_cache(maxsize=1)
def get_context() -> ExplainerContext:
    """Return the process-local cached explainer context.

    Args:
        None.

    Returns:
        Computed result as ``ExplainerContext``.
    """
    return build_explainer_context()


def clear_context_cache() -> None:
    """Clear cached model state after local data or settings change.

    Args:
        None.

    Returns:
        None.
    """
    get_ranking_context.cache_clear()
    get_context.cache_clear()


def _resolve_date(panel: pd.DataFrame, date: object | None) -> pd.Timestamp:
    """Resolve a requested calendar month to its latest available panel date."""
    dates = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    if date is None:
        return pd.Timestamp(dates.max())
    requested = pd.Timestamp(date)
    same_month = dates[dates.to_period("M") == requested.to_period("M")]
    if same_month.empty:
        raise ValueError(f"No panel rows exist in {requested:%Y-%m}.")
    return pd.Timestamp(same_month.max())


def _resolve_permno(
    panel: pd.DataFrame,
    ticker_or_permno: str | int,
    date: pd.Timestamp,
) -> int:
    """Resolve a PERMNO or unambiguous date-specific ticker."""
    if isinstance(ticker_or_permno, (int, np.integer)) or str(
        ticker_or_permno
    ).strip().isdigit():
        permno = int(ticker_or_permno)
        if not ((panel["permno"].eq(permno)) & panel["date"].eq(date)).any():
            raise ValueError(f"PERMNO {permno} has no row at {date.date()}.")
        return permno

    ticker = str(ticker_or_permno).strip().upper()
    month = panel.loc[panel["date"].eq(date)]
    matches = month.loc[
        month["ticker"].astype("string").str.upper().eq(ticker),
        ["permno", "ticker", "company_name", "investable"],
    ].drop_duplicates("permno")
    if matches.empty:
        raise ValueError(f"Ticker {ticker!r} has no row at {date.date()}.")
    if len(matches) > 1:
        choices = matches.to_dict("records")
        raise ValueError(
            f"Ticker {ticker!r} is ambiguous at {date.date()}; use PERMNO. "
            f"Matches: {choices}"
        )
    return int(matches["permno"].iloc[0])


def _percentiles(values: pd.Series, eligible: pd.Series) -> pd.Series:
    """Return high-score-is-good cross-sectional percentiles on a 0-100 scale."""
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = eligible.fillna(False) & values.notna()
    output.loc[valid] = values.loc[valid].rank(method="average", pct=True) * 100.0
    return output


def _deciles(values: pd.Series, eligible: pd.Series) -> pd.Series:
    """Return deterministic high-score-is-decile-one ranks."""
    output = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = eligible.fillna(False) & values.notna()
    if valid.any():
        ordered = values.loc[valid].sort_values(
            ascending=False,
            kind="mergesort",
        )
        ranks = pd.Series(
            np.arange(1, len(ordered) + 1),
            index=ordered.index,
            dtype="int64",
        )
        output.loc[ordered.index] = np.ceil(
            ranks * N_DECILES / len(ordered)
        ).astype("int64")
    return output


def _held_lookup(context: ExplainerContext, date: pd.Timestamp) -> pd.DataFrame:
    """Return configured model holdings for one formation date."""
    return context.construction.holdings.loc[
        context.construction.holdings["date"].eq(date),
        ["permno", "weight"],
    ].drop_duplicates("permno")


def _month_scores(
    context: ExplainerContext,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """Attach composite/block ranks and configured holdings to one panel month."""
    panel = context.panel
    month = panel.loc[panel["date"].eq(date)].copy()
    if month.empty:
        raise ValueError(f"No panel rows exist at {date.date()}.")
    month["composite_score"] = context.composite.selected_score.loc[month.index]
    for block in MODEL_BLOCKS:
        month[block] = context.composite.blocks.loc[month.index, block]

    investable = month["investable"].fillna(False).astype(bool)
    month["composite_percentile"] = _percentiles(
        month["composite_score"], investable
    )
    for block in MODEL_BLOCKS:
        month[f"{block}_percentile"] = _percentiles(month[block], investable)
        month[f"{block}_decile"] = _deciles(month[block], investable)

    terminal = panel.loc[month.index, "_terminal_row"]
    candidates = month.loc[
        investable
        & month["composite_score"].notna()
        & pd.to_numeric(month["market_cap"], errors="coerce").gt(0)
        & ~terminal
    ].copy()
    month["composite_decile"] = pd.Series(
        pd.NA,
        index=month.index,
        dtype="Int64",
    )
    month["score_rank"] = pd.Series(pd.NA, index=month.index, dtype="Int64")
    month["buy_eligible"] = False
    month["sell_eligible"] = False
    if not candidates.empty:
        ranked = _rank_deciles(candidates)
        month.loc[ranked.index, "composite_decile"] = ranked["decile"].astype(
            "Int64"
        )
        month.loc[ranked.index, "score_rank"] = ranked["score_rank"].astype(
            "Int64"
        )
        month.loc[ranked.index, "buy_eligible"] = _buy_mask(ranked).to_numpy()
        month.loc[ranked.index, "sell_eligible"] = _sell_mask(ranked).to_numpy()
    holdings = _held_lookup(context, date).set_index("permno")
    month["held"] = month["permno"].isin(holdings.index)
    month["weight"] = month["permno"].map(holdings["weight"])
    month["sector_name"] = month["gsector"].map(_sector_name)
    month["terminal_row"] = terminal
    return month


def _effective_block_weights(
    context: ExplainerContext,
    date: pd.Timestamp,
    block_values: pd.Series,
) -> pd.Series:
    """Renormalize configured block weights over a stock's available scores."""
    monthly = context.composite.weights[COMPOSITE_WEIGHTING].loc[date, MODEL_BLOCKS]
    effective = monthly.where(block_values.notna(), 0.0).astype("float64")
    if effective.sum() <= 0:
        return effective
    return effective / effective.sum()


def _not_held_reason(row: pd.Series, previous_held: bool) -> str:
    """Explain the first binding reason a stock is absent from the target."""
    if not bool(row.get("in_universe", False)):
        return "Not in the Russell 3000 proxy at this date."
    failed_screens: list[str] = []
    for column, label in (
        ("price_screen_pass", "price"),
        ("market_cap_screen_pass", "market cap"),
        ("adv_screen_pass", "trading liquidity"),
    ):
        if not bool(row.get(column, False)):
            failed_screens.append(label)
    if failed_screens:
        return "Failed the liquidity screen: " + ", ".join(failed_screens) + "."
    if bool(row.get("quarantined", False)):
        return "The live return is quarantined, so the stock is not investable."
    if bool(row.get("secondary_share_class", False)) or not bool(
        row.get("primary_security", True)
    ):
        return "Excluded by the one-security-per-company constraint."
    if not bool(row.get("investable", False)):
        return "Failed an investability rule at this date."
    if pd.isna(row.get("composite_score")):
        return "Missing enough signal data to form a composite score."
    if bool(row.get("terminal_row", False)):
        return "Excluded because this is the security's terminal CRSP observation."
    if not bool(row.get("buy_eligible", False)):
        if previous_held and not bool(row.get("sell_eligible", False)):
            return f"Fell below the top {SELL_PERCENTILE:.0%} sell threshold."
        return f"Not in the top {BUY_PERCENTILE:.0%} buy group."
    return "Dropped by the holdings-band or portfolio constraints."


def _stock_summary(
    row: pd.Series,
    previous_held: bool,
) -> pd.DataFrame:
    """Return the summary section as a compact metric table."""
    held = bool(row["held"])
    decision = "Held in the configured target." if held else _not_held_reason(
        row, previous_held
    )
    metrics = [
        ("Date", row["date"].date()),
        ("PERMNO", int(row["permno"])),
        ("Ticker", row["ticker"]),
        ("Company", row["company_name"]),
        ("Sector", row["sector_name"]),
        ("Price (USD/share)", row["price"]),
        ("Market cap (USD millions)", row["market_cap"]),
        ("Held", held),
        ("Target weight", row["weight"] if held else 0.0),
        ("Composite score", row["composite_score"]),
        ("Composite percentile", row["composite_percentile"]),
        ("Composite decile", row["composite_decile"]),
        ("Decision", decision),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def _block_breakdown(
    context: ExplainerContext,
    month: pd.DataFrame,
    row: pd.Series,
) -> pd.DataFrame:
    """Return block scores, ranks, and weighted contributions for one stock."""
    block_values = row.loc[MODEL_BLOCKS].astype("float64")
    effective = _effective_block_weights(context, row["date"], block_values)
    records: list[dict[str, object]] = []
    sector = row["gsector"]
    sector_mask = month["investable"].fillna(False) & month["gsector"].eq(sector)
    for block in MODEL_BLOCKS:
        sector_percentile = _percentiles(month[block], sector_mask).loc[row.name]
        contribution = (
            float(effective[block] * block_values[block])
            if pd.notna(block_values[block])
            else np.nan
        )
        records.append(
            {
                "block": block,
                "block_name": BLOCK_NAMES[block],
                "standardized_score": block_values[block],
                "universe_percentile": row[f"{block}_percentile"],
                "sector_percentile": sector_percentile,
                "decile": row[f"{block}_decile"],
                "effective_weight": effective[block],
                "score_contribution": contribution,
                "effect": (
                    "pulling up"
                    if contribution > 0
                    else "dragging down"
                    if contribution < 0
                    else "neutral / unavailable"
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["score_contribution", "block"],
        ascending=[False, True],
        na_position="last",
        ignore_index=True,
    )


def _missing_signal_reason(
    block: str,
    signal: str,
    row: pd.Series,
) -> str:
    """Give an analyst-readable reason for an unavailable individual signal."""
    sector = pd.to_numeric(pd.Series([row.get("gsector")]), errors="coerce").iloc[0]
    financial = pd.notna(sector) and int(sector) in (40, 60)
    if signal == "ebitda_to_ev":
        if financial:
            return "Financials/Real Estate exclusion: enterprise value is not comparable."
        enterprise_value = sum(
            pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            * sign
            for column, sign in (
                ("market_cap", 1.0),
                ("dlcq", 1.0),
                ("dlttq", 1.0),
                ("cheq", -1.0),
            )
        )
        if pd.notna(enterprise_value) and enterprise_value <= 0:
            return "Enterprise value is nonpositive."
    if signal in {"cfo_to_ic", "ebitda_roic"}:
        if financial:
            return "Financials/Real Estate exclusion: invested capital is not comparable."
        invested_capital = (
            pd.to_numeric(pd.Series([row.get("ceqq")]), errors="coerce").iloc[0]
            + pd.to_numeric(pd.Series([row.get("dlttq")]), errors="coerce").iloc[0]
            + pd.to_numeric(pd.Series([row.get("dlcq")]), errors="coerce").iloc[0]
            - pd.to_numeric(pd.Series([row.get("cheq")]), errors="coerce").iloc[0]
        )
        if pd.notna(invested_capital) and invested_capital <= 0:
            return "Invested capital is nonpositive."
    if signal == "roe_ltm":
        equity = pd.to_numeric(pd.Series([row.get("ceqq")]), errors="coerce").iloc[0]
        if pd.notna(equity) and equity <= 0:
            return "Common equity is nonpositive."
    if signal in {"asset_turnover_vol", "ebit_margin_vol", "eps_vol"}:
        return "Fewer than 8 usable consecutive quarters or a required field is missing."
    if signal.startswith("share_chg_"):
        horizon = signal.removeprefix("share_chg_").removesuffix("m")
        return f"No valid consecutive {horizon}-month split-adjusted share history."
    if signal == "momentum_12_1":
        return "Fewer than 11 consecutive prior monthly returns."
    if signal == "short_interest_ratio":
        return "No usable short-interest report or split-adjusted share count."
    if block in {"value", "quality"}:
        return "Insufficient consecutive quarterly history or a required field is missing."
    return "Required source data is unavailable."


def _format_raw(signal: str, value: float) -> str:
    """Format an individual raw signal without discarding its numeric value."""
    if pd.isna(value):
        return "NaN"
    if signal in PERCENT_SIGNALS:
        return f"{value:.2%}"
    if signal == "eps_vol":
        return f"{value:.4f} EPS units"
    return f"{value:.4f}"


def _signal_detail(
    context: ExplainerContext,
    month: pd.DataFrame,
    row: pd.Series,
) -> pd.DataFrame:
    """Return every raw and standardized individual signal for one stock."""
    records: list[dict[str, object]] = []
    investable = month["investable"].fillna(False)
    for block in MODEL_BLOCKS:
        block_columns = context.raw_signals[block].columns
        for signal in block_columns:
            raw_value = context.raw_signals.loc[row.name, (block, signal)]
            standardized = context.standardized_signals.loc[
                row.name, (block, signal)
            ]
            month_standardized = context.standardized_signals.loc[
                month.index, (block, signal)
            ]
            percentile = _percentiles(month_standardized, investable).loc[row.name]
            records.append(
                {
                    "block": BLOCK_NAMES[block],
                    "signal": signal,
                    "signal_name": SIGNAL_NAMES[signal],
                    "raw_value": raw_value,
                    "raw_display": _format_raw(signal, raw_value),
                    "standardized_score": standardized,
                    "universe_percentile": percentile,
                    "status": (
                        "available"
                        if pd.notna(raw_value) and pd.notna(standardized)
                        else _missing_signal_reason(block, signal, row)
                    ),
                }
            )
    return pd.DataFrame(records)


def _history(
    context: ExplainerContext,
    permno: int,
    date: pd.Timestamp,
    months: int = 24,
) -> pd.DataFrame:
    """Return score, decile, and holding history through the requested month."""
    stock_rows = context.panel.index[
        context.panel["permno"].eq(permno) & context.panel["date"].le(date)
    ]
    stock_rows = stock_rows[-months:]
    records: list[pd.DataFrame] = []
    holdings = context.construction.holdings.loc[
        context.construction.holdings["permno"].eq(permno),
        ["date", "weight"],
    ].set_index("date")
    for index in stock_rows:
        row_date = pd.Timestamp(context.panel.at[index, "date"])
        month_index = context.panel.index[context.panel["date"].eq(row_date)]
        month_investable = context.panel.loc[month_index, "investable"].fillna(False)
        record: dict[str, object] = {
            "date": row_date,
            "composite_score": context.composite.selected_score.loc[index],
        }
        record["composite_decile"] = _deciles(
            context.composite.selected_score.loc[month_index],
            month_investable,
        ).loc[index]
        for block in MODEL_BLOCKS:
            record[f"{block}_decile"] = _deciles(
                context.composite.blocks.loc[month_index, block],
                month_investable,
            ).loc[index]
        record["held"] = row_date in holdings.index
        record["weight"] = (
            float(holdings.at[row_date, "weight"])
            if row_date in holdings.index
            else 0.0
        )
        records.append(pd.DataFrame([record]))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _history_figure(history: pd.DataFrame, ticker: str) -> plt.Figure:
    """Chart composite level and block deciles, marking held months."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    held = history["held"].fillna(False)
    axes[0].plot(
        history["date"],
        history["composite_score"],
        color="#1f4e79",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Composite score",
    )
    axes[0].scatter(
        history.loc[held, "date"],
        history.loc[held, "composite_score"],
        color="#2ca02c",
        marker="s",
        s=42,
        zorder=3,
        label="Held",
    )
    axes[0].axhline(0.0, color="0.7", linewidth=0.8)
    axes[0].set_ylabel("Standardized score")
    axes[0].set_title(f"{ticker}: score and ranking history")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.2)

    decile_columns = ["composite_decile", *[f"{b}_decile" for b in MODEL_BLOCKS]]
    labels = ["Composite", *[BLOCK_NAMES[b] for b in MODEL_BLOCKS]]
    for column, label in zip(decile_columns, labels, strict=True):
        axes[1].plot(
            history["date"],
            pd.to_numeric(history[column], errors="coerce"),
            marker="o" if column == "composite_decile" else None,
            linewidth=2 if column == "composite_decile" else 1,
            alpha=1.0 if column == "composite_decile" else 0.75,
            label=label,
        )
    axes[1].scatter(
        history.loc[held, "date"],
        np.full(int(held.sum()), 10.45),
        color="#2ca02c",
        marker="s",
        s=34,
        label="Held month",
        clip_on=False,
    )
    axes[1].set_yticks(range(1, 11))
    axes[1].set_ylim(10.7, 0.7)
    axes[1].set_ylabel("Decile (1 = best)")
    axes[1].set_xlabel("Formation month")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    return figure


def _pre_composite_scores(
    context: ExplainerContext,
    month: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return weighted pre-standardization composite inputs and row weights."""
    monthly_weights = context.composite.weights[COMPOSITE_WEIGHTING].loc[
        date, MODEL_BLOCKS
    ]
    values = month.loc[:, MODEL_BLOCKS].astype("float64")
    row_weights = pd.DataFrame(
        np.broadcast_to(monthly_weights.to_numpy(), values.shape),
        index=values.index,
        columns=values.columns,
    ).where(values.notna(), 0.0)
    denominator = row_weights.sum(axis=1)
    row_weights = row_weights.div(denominator.where(denominator.gt(0)), axis=0)
    pre_score = (values * row_weights).sum(axis=1, min_count=1)
    return pre_score, row_weights


def _what_would_change(
    context: ExplainerContext,
    month: pd.DataFrame,
    row: pd.Series,
) -> str:
    """Estimate the marginal block move needed to cross the selection cutoff."""
    if not bool(row["investable"]) or pd.isna(row["composite_score"]):
        return (
            "A score threshold is not meaningful until the stock passes all "
            "investability rules and has enough data for a composite score."
        )
    pre_score, row_weights = _pre_composite_scores(context, month, row["date"])
    block_values = row.loc[MODEL_BLOCKS].astype("float64")
    contributions = block_values * row_weights.loc[row.name]
    valid = contributions.dropna()
    if valid.empty:
        return "No block contribution is available."

    if not bool(row["held"]):
        drag = valid.idxmin()
        buy_group = month["buy_eligible"].fillna(False)
        threshold = pre_score.loc[buy_group].min()
        needed_pre = max(0.0, float(threshold - pre_score.loc[row.name]))
        weight = float(row_weights.loc[row.name, drag])
        needed_block = needed_pre / weight if weight > 0 else np.nan
        score_gap = month.loc[buy_group, "composite_score"].min() - row[
            "composite_score"
        ]
        drag_description = (
            "the largest drag" if block_values[drag] < 0 else "the weakest block"
        )
        return (
            f"{BLOCK_NAMES[drag]} is {drag_description} "
            f"({block_values[drag]:+.2f}z). "
            f"Holding the other blocks and the cross-section fixed, it would need "
            f"to improve by roughly {needed_block:.2f} block-score z-points to "
            f"reach the top {BUY_PERCENTILE:.0%} buy group. The current "
            f"composite gap to that "
            f"cutoff is about {score_gap:.2f}z."
        )

    driver = valid.idxmax()
    retained = month["sell_eligible"].fillna(False)
    cutoff = pre_score.loc[retained].min()
    cushion_pre = max(0.0, float(pre_score.loc[row.name] - cutoff))
    weight = float(row_weights.loc[row.name, driver])
    block_drop = cushion_pre / weight if weight > 0 else np.nan
    composite_cutoff = month.loc[retained, "composite_score"].min()
    composite_cushion = float(row["composite_score"] - composite_cutoff)
    return (
        f"{BLOCK_NAMES[driver]} is doing the most work "
        f"({block_values[driver]:+.2f}z). The stock is approximately "
        f"{composite_cushion:.2f} composite z-points above the top "
        f"{SELL_PERCENTILE:.0%} sell cutoff; that leading block could fall roughly "
        f"{block_drop:.2f} z-points, holding everything else fixed, before the "
        "stock crosses the sell cutoff."
    )


def _previous_held(
    context: ExplainerContext,
    permno: int,
    date: pd.Timestamp,
) -> bool:
    """Return whether the stock was held at the preceding formation date."""
    dates = context.construction.monthly_diagnostics["date"]
    preceding = dates.loc[dates.lt(date)]
    if preceding.empty:
        return False
    previous_date = preceding.max()
    holdings = context.construction.holdings
    return bool(
        (
            holdings["date"].eq(previous_date)
            & holdings["permno"].eq(permno)
        ).any()
    )


def _print_explanation(result: StockExplanation) -> None:
    """Print the five requested report sections."""
    print("\n1. SUMMARY")
    print(result.summary.to_string(index=False))
    print("\n2. BLOCK BREAKDOWN")
    print(
        result.block_breakdown.to_string(
            index=False,
            formatters={
                "standardized_score": lambda value: f"{value:+.3f}",
                "universe_percentile": lambda value: f"{value:.1f}",
                "sector_percentile": lambda value: f"{value:.1f}",
                "effective_weight": lambda value: f"{value:.1%}",
                "score_contribution": lambda value: f"{value:+.3f}",
            },
        )
    )
    print("\n3. SIGNAL DETAIL")
    print(
        result.signal_detail.loc[
            :,
            [
                "block",
                "signal_name",
                "raw_display",
                "standardized_score",
                "universe_percentile",
                "status",
            ],
        ].to_string(
            index=False,
            formatters={
                "standardized_score": lambda value: f"{value:+.3f}",
                "universe_percentile": lambda value: f"{value:.1f}",
            },
        )
    )
    print("\n4. HISTORY — LAST 24 MONTHS")
    print(result.history.to_string(index=False))
    print("\n5. WHAT WOULD CHANGE IT")
    print(result.what_would_change)


def explain_stock(
    ticker_or_permno: str | int,
    date: object | None = None,
    *,
    context: ExplainerContext | None = None,
) -> StockExplanation:
    """Print and return a five-part explanation for one stock and month.

    Args:
        ticker_or_permno: Ticker symbol or CRSP PERMNO to resolve.
        date: Requested formation date; None selects the latest month.
        context: Optional precomputed context reused across calls.

    Returns:
        Computed result as ``StockExplanation``.
    """
    state = get_context() if context is None else context
    resolved_date = _resolve_date(state.panel, date)
    permno = _resolve_permno(state.panel, ticker_or_permno, resolved_date)
    month = _month_scores(state, resolved_date)
    row = month.loc[month["permno"].eq(permno)].iloc[0]
    previous_held = _previous_held(state, permno, resolved_date)
    summary = _stock_summary(row, previous_held)
    blocks = _block_breakdown(state, month, row)
    signals = _signal_detail(state, month, row)
    history = _history(state, permno, resolved_date)
    what_changes = _what_would_change(state, month, row)
    ticker = str(row["ticker"]) if pd.notna(row["ticker"]) else str(permno)
    figure = _history_figure(history, ticker)
    result = StockExplanation(
        permno=permno,
        date=resolved_date,
        summary=summary,
        block_breakdown=blocks,
        signal_detail=signals,
        history=history,
        what_would_change=what_changes,
        figure=figure,
    )
    _print_explanation(result)
    return result


def compare_stocks(
    list_of_tickers: Sequence[str | int],
    date: object | None = None,
    *,
    context: ExplainerContext | None = None,
) -> pd.DataFrame:
    """Print and return block breakdowns for several stocks side by side.

    Args:
        list_of_tickers: Ticker symbols or PERMNOs to compare.
        date: Requested formation date; None selects the latest month.
        context: Optional precomputed context reused across calls.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if not list_of_tickers:
        raise ValueError("list_of_tickers cannot be empty.")
    state = get_context() if context is None else context
    resolved_date = _resolve_date(state.panel, date)
    month = _month_scores(state, resolved_date)
    records: list[pd.DataFrame] = []
    for identifier in list_of_tickers:
        permno = _resolve_permno(state.panel, identifier, resolved_date)
        row = month.loc[month["permno"].eq(permno)].iloc[0]
        block_table = _block_breakdown(state, month, row).assign(
            permno=permno,
            ticker=row["ticker"],
            company_name=row["company_name"],
            held=bool(row["held"]),
            weight=float(row["weight"]) if pd.notna(row["weight"]) else 0.0,
            composite_score=row["composite_score"],
            composite_decile=row["composite_decile"],
        )
        records.append(block_table)
    output = pd.concat(records, ignore_index=True)
    wide = output.pivot(
        index="block_name",
        columns="ticker",
        values=["standardized_score", "universe_percentile", "decile"],
    )
    print(f"\nSTOCK COMPARISON — {resolved_date.date()}")
    print(wide.round(2).to_string())
    return output


def _normalize_deciles(value: object) -> set[int]:
    """Normalize a scalar or collection of requested deciles."""
    if isinstance(value, (int, np.integer)):
        deciles = {int(value)}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        deciles = {int(item) for item in value}
    else:
        raise TypeError("Decile criteria must be an integer or a sequence of integers.")
    invalid = deciles - set(range(1, N_DECILES + 1))
    if invalid:
        raise ValueError(f"Deciles must be 1-{N_DECILES}; got {sorted(invalid)}.")
    return deciles


def _screen_criteria(
    block: str | Mapping[str, object] | None,
    decile: int | Sequence[int] | None,
) -> dict[str, set[int]]:
    """Translate simple or multi-block screen arguments into decile criteria."""
    if isinstance(block, Mapping):
        if decile is not None:
            raise ValueError("When block is a mapping, put each decile in the mapping.")
        criteria = {
            str(name): _normalize_deciles(value) for name, value in block.items()
        }
    else:
        name = "composite" if block is None else str(block)
        criteria = {} if decile is None else {name: _normalize_deciles(decile)}
    allowed = {"composite", *MODEL_BLOCKS}
    unknown = set(criteria) - allowed
    if unknown:
        raise ValueError(f"Unknown screen block(s): {sorted(unknown)}")
    return criteria


def _sector_filter(values: pd.Series, sector: str | int | None) -> pd.Series:
    """Return a boolean mask for a GICS code or readable sector name."""
    if sector is None:
        return pd.Series(True, index=values.index)
    if isinstance(sector, (int, np.integer)) or str(sector).strip().isdigit():
        return pd.to_numeric(values, errors="coerce").eq(int(sector))
    wanted = str(sector).strip().lower()
    aliases = {
        "consumer discretionary": 25,
        "consumer staples": 30,
        "healthcare": 35,
        "health care": 35,
        "information technology": 45,
        "technology": 45,
        "communication services": 50,
        "communications": 50,
        "real estate": 60,
    }
    matching = [code for code, name in SECTOR_NAMES.items() if name.lower() == wanted]
    code = aliases.get(wanted, matching[0] if matching else None)
    if code is None:
        raise ValueError(
            f"Unknown sector {sector!r}; use a GICS code or one of "
            f"{list(SECTOR_NAMES.values())}."
        )
    return pd.to_numeric(values, errors="coerce").eq(code)


def screen(
    block: str | Mapping[str, object] | None = None,
    sector: str | int | None = None,
    decile: int | Sequence[int] | None = None,
    date: object | None = None,
    *,
    context: ExplainerContext | None = None,
) -> pd.DataFrame:
    """List investable names matching composite or multi-block decile criteria.

        A mapping enables compound questions, for example
        ``block={"quality": (1, 2), "value": (8, 9, 10)}``.

    Args:
        block: Optional block name or block-to-decile criteria.
        sector: Optional GICS sector name or code.
        decile: Optional decile or collection of deciles.
        date: Requested formation date; None selects the latest month.
        context: Optional precomputed context reused across calls.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    state = get_context() if context is None else context
    resolved_date = _resolve_date(state.panel, date)
    month = _month_scores(state, resolved_date)
    criteria = _screen_criteria(block, decile)
    mask = month["investable"].fillna(False) & _sector_filter(
        month["gsector"], sector
    )
    for name, requested in criteria.items():
        column = "composite_decile" if name == "composite" else f"{name}_decile"
        mask &= month[column].isin(requested)

    columns = [
        "permno",
        "ticker",
        "company_name",
        "sector_name",
        "price",
        "market_cap",
        "held",
        "weight",
        "composite_score",
        "composite_decile",
    ]
    for name in MODEL_BLOCKS:
        columns.extend([name, f"{name}_decile"])
    sort_block = next(iter(criteria), "composite")
    sort_column = "composite_score" if sort_block == "composite" else sort_block
    output = month.loc[mask, columns].sort_values(
        [sort_column, "market_cap", "permno"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    print(
        f"\nSCREEN — {resolved_date.date()} | sector={sector or 'all'} | "
        f"criteria={criteria or 'none'} | {len(output):,} names"
    )
    if output.empty:
        print("No names matched.")
    else:
        print(output.to_string(index=False))
    return output


def _ranking_series_history(
    panel: pd.DataFrame,
    values: pd.Series,
    sample_mask: pd.Series,
    stock_index: pd.Index,
) -> pd.DataFrame:
    """Return score, universe percentile, and decile for selected stock rows."""
    numeric = pd.Series(values, index=panel.index, dtype="float64")
    eligible = (
        sample_mask
        & panel["investable"].fillna(False).astype(bool)
        & numeric.notna()
    )
    eligible_values = numeric.loc[eligible]
    eligible_dates = panel.loc[eligible, "date"]
    percentile = (
        eligible_values.groupby(eligible_dates, sort=False).rank(
            method="average",
            pct=True,
        )
        * 100.0
    )
    descending_rank = eligible_values.groupby(eligible_dates, sort=False).rank(
        method="first",
        ascending=False,
    )
    group_size = eligible_values.groupby(eligible_dates, sort=False).transform(
        "size"
    )
    decile = np.ceil(descending_rank * N_DECILES / group_size).astype("int8")
    return pd.DataFrame(
        {
            "score": numeric.reindex(stock_index).to_numpy(),
            "percentile": percentile.reindex(stock_index).to_numpy(),
            "decile": pd.array(decile.reindex(stock_index), dtype="Int64"),
        },
        index=stock_index,
    )


def _ranking_figures(
    ticker: str,
    composite_history: pd.DataFrame,
    factor_history: pd.DataFrame,
    signal_history: pd.DataFrame,
) -> dict[str, plt.Figure]:
    """Chart composite level and factor and individual-signal percentile histories."""
    composite_figure, score_axis = plt.subplots(figsize=(11, 4.5))
    percentile_axis = score_axis.twinx()
    score_axis.plot(
        composite_history["date"],
        composite_history["composite_score"],
        color="#1f4e79",
        linewidth=2,
        label="Composite score",
    )
    percentile_axis.plot(
        composite_history["date"],
        composite_history["composite_percentile"],
        color="#d97706",
        linewidth=1.5,
        linestyle="--",
        label="Composite percentile",
    )
    score_axis.axhline(0.0, color="0.7", linewidth=0.8)
    percentile_axis.axhline(90.0, color="#d97706", linewidth=0.8, alpha=0.35)
    score_axis.set_ylabel("Standardized composite score")
    percentile_axis.set_ylabel("Universe percentile (higher is better)")
    percentile_axis.set_ylim(0, 100)
    score_axis.set_title(f"{ticker}: composite score and ranking")
    lines = score_axis.lines[:1] + percentile_axis.lines[:1]
    score_axis.legend(lines, [line.get_label() for line in lines], frameon=False)
    score_axis.grid(alpha=0.2)
    composite_figure.tight_layout()

    factor_figure, factor_axis = plt.subplots(figsize=(11, 5))
    factor_pivot = factor_history.pivot(
        index="date",
        columns="factor_name",
        values="percentile",
    )
    factor_pivot.plot(ax=factor_axis, linewidth=1.6)
    factor_axis.axhline(90.0, color="0.5", linewidth=0.8, linestyle="--")
    factor_axis.set_ylim(0, 100)
    factor_axis.set_ylabel("Universe percentile (higher is better)")
    factor_axis.set_title(f"{ticker}: model factor rankings")
    factor_axis.legend(frameon=False, ncol=3)
    factor_axis.grid(alpha=0.2)
    factor_figure.tight_layout()

    blocks = [block for block in MODEL_BLOCKS if block in signal_history["factor"].unique()]
    signal_figure, axes = plt.subplots(
        len(blocks),
        1,
        figsize=(11, 3.0 * len(blocks)),
        sharex=True,
        squeeze=False,
    )
    for axis, block in zip(axes[:, 0], blocks, strict=True):
        block_history = signal_history.loc[signal_history["factor"].eq(block)]
        pivot = block_history.pivot(
            index="date",
            columns="signal_name",
            values="percentile",
        )
        pivot.plot(ax=axis, linewidth=1.3)
        axis.axhline(90.0, color="0.5", linewidth=0.7, linestyle="--")
        axis.set_ylim(0, 100)
        axis.set_ylabel("Percentile")
        axis.set_title(BLOCK_NAMES[block])
        axis.legend(frameon=False, ncol=3, fontsize=8)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Month")
    signal_figure.suptitle(
        f"{ticker}: individual signal rankings (higher is better)",
        y=1.0,
    )
    signal_figure.tight_layout()
    figures = {
        "composite": composite_figure,
        "factors": factor_figure,
        "signals": signal_figure,
    }
    # The notebook displays these returned figures explicitly. Closing their
    # pyplot handles prevents the inline backend from rendering them a second time.
    for figure in figures.values():
        plt.close(figure)
    return figures


def stock_rank_history(
    ticker_or_permno: str | int,
    date: object | None = None,
    months: int | None = 60,
    *,
    context: RankingContext | None = None,
) -> StockRankingReport:
    """Return current and historical signal, factor, and composite rankings.

    Args:
        ticker_or_permno: Ticker symbol or CRSP PERMNO to resolve.
        date: Requested formation date; None selects the latest month.
        months: Number of historical months; None keeps all available history.
        context: Optional precomputed context reused across calls.

    Returns:
        Computed result as ``StockRankingReport``.
    """
    state = get_ranking_context() if context is None else context
    resolved_date = _resolve_date(state.panel, date)
    permno = _resolve_permno(state.panel, ticker_or_permno, resolved_date)
    stock_mask = state.panel["permno"].eq(permno) & state.panel["date"].le(
        resolved_date
    )
    stock_index = state.panel.index[stock_mask]
    if months is not None:
        if months <= 0:
            raise ValueError("months must be positive or None.")
        stock_index = stock_index[-int(months) :]
    if stock_index.empty:
        raise ValueError(f"No history is available for PERMNO {permno}.")
    dates = state.panel.loc[stock_index, "date"]
    sample_mask = state.panel["date"].isin(pd.DatetimeIndex(dates.unique()))
    latest_index = stock_index[-1]
    latest_row = state.panel.loc[latest_index]

    composite_rank = _ranking_series_history(
        state.panel,
        state.composite.selected_score,
        sample_mask,
        stock_index,
    )
    composite_history = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "composite_score": composite_rank["score"].to_numpy(),
            "composite_percentile": composite_rank["percentile"].to_numpy(),
            "composite_decile": composite_rank["decile"].array,
            "in_universe": state.panel.loc[stock_index, "in_universe"].to_numpy(),
            "investable": state.panel.loc[stock_index, "investable"].to_numpy(),
        }
    )

    factor_pieces: list[pd.DataFrame] = []
    for factor in MODEL_BLOCKS:
        ranking = _ranking_series_history(
            state.panel,
            state.composite.blocks[factor],
            sample_mask,
            stock_index,
        )
        factor_pieces.append(
            pd.DataFrame(
                {
                    "date": dates.to_numpy(),
                    "factor": factor,
                    "factor_name": BLOCK_NAMES[factor],
                    "score": ranking["score"].to_numpy(),
                    "percentile": ranking["percentile"].to_numpy(),
                    "decile": ranking["decile"].array,
                }
            )
        )
    factor_history = pd.concat(factor_pieces, ignore_index=True)

    signal_pieces: list[pd.DataFrame] = []
    for factor in MODEL_BLOCKS:
        for signal in state.standardized_signals[factor].columns:
            ranking = _ranking_series_history(
                state.panel,
                state.standardized_signals[(factor, signal)],
                sample_mask,
                stock_index,
            )
            raw = state.raw_signals.loc[stock_index, (factor, signal)]
            signal_pieces.append(
                pd.DataFrame(
                    {
                        "date": dates.to_numpy(),
                        "factor": factor,
                        "factor_name": BLOCK_NAMES[factor],
                        "signal": signal,
                        "signal_name": SIGNAL_NAMES[signal],
                        "raw_value": raw.to_numpy(),
                        "score": ranking["score"].to_numpy(),
                        "percentile": ranking["percentile"].to_numpy(),
                        "decile": ranking["decile"].array,
                    }
                )
            )
    signal_history = pd.concat(signal_pieces, ignore_index=True)

    latest_factors = factor_history.loc[
        factor_history["date"].eq(resolved_date)
    ].copy()
    month_mask = state.panel["date"].eq(resolved_date)
    sector_mask = (
        month_mask
        & state.panel["investable"].fillna(False)
        & state.panel["gsector"].eq(latest_row["gsector"])
    )
    sector_percentiles: dict[str, float] = {}
    for factor in MODEL_BLOCKS:
        values = state.composite.blocks[factor]
        sector_percentiles[factor] = _percentiles(
            values.loc[month_mask],
            sector_mask.loc[month_mask],
        ).loc[latest_index]
    latest_factors["sector_percentile"] = latest_factors["factor"].map(
        sector_percentiles
    )
    latest_factors = latest_factors.loc[
        :,
        [
            "factor",
            "factor_name",
            "score",
            "percentile",
            "sector_percentile",
            "decile",
        ],
    ].sort_values("score", ascending=False, na_position="last", ignore_index=True)

    latest_signals = signal_history.loc[
        signal_history["date"].eq(resolved_date)
    ].copy()
    latest_signals["raw_display"] = [
        _format_raw(signal, value)
        for signal, value in latest_signals[["signal", "raw_value"]].itertuples(
            index=False,
            name=None,
        )
    ]
    latest_signals["status"] = [
        "available"
        if pd.notna(raw_value) and pd.notna(score)
        else _missing_signal_reason(factor, signal, latest_row)
        for factor, signal, raw_value, score in latest_signals[
            ["factor", "signal", "raw_value", "score"]
        ].itertuples(index=False, name=None)
    ]
    latest_signals = latest_signals.loc[
        :,
        [
            "factor_name",
            "signal",
            "signal_name",
            "raw_value",
            "raw_display",
            "score",
            "percentile",
            "decile",
            "status",
        ],
    ]

    summary = pd.DataFrame(
        [
            ("Date", resolved_date.date()),
            ("PERMNO", permno),
            ("Ticker", latest_row["ticker"]),
            ("Company", latest_row["company_name"]),
            ("Sector", _sector_name(latest_row["gsector"])),
            ("Price (USD/share)", latest_row["price"]),
            ("Market cap (USD millions)", latest_row["market_cap"]),
            ("In universe", bool(latest_row["in_universe"])),
            ("Investable", bool(latest_row["investable"])),
            ("Composite score", composite_rank.at[latest_index, "score"]),
            (
                "Composite percentile",
                composite_rank.at[latest_index, "percentile"],
            ),
            ("Composite decile", composite_rank.at[latest_index, "decile"]),
        ],
        columns=["metric", "value"],
    )
    ticker = (
        str(latest_row["ticker"])
        if pd.notna(latest_row["ticker"])
        else str(permno)
    )
    figures = _ranking_figures(
        ticker,
        composite_history,
        factor_history,
        signal_history,
    )
    print(
        f"{ticker} rankings through {resolved_date.date()} | "
        f"{len(composite_history):,} months shown | "
        "percentile 100 = best"
    )
    return StockRankingReport(
        permno=permno,
        ticker=ticker,
        date=resolved_date,
        summary=summary,
        latest_factors=latest_factors,
        latest_signals=latest_signals,
        composite_history=composite_history,
        factor_history=factor_history,
        signal_history=signal_history,
        figures=figures,
    )


def worked_example_names(
    date: object | None = None,
    *,
    context: ExplainerContext | None = None,
) -> tuple[str | int, str | int]:
    """Choose deterministic held and non-held names for the demonstration notebook.

    Args:
        date: Requested formation date; None selects the latest month.
        context: Optional precomputed context reused across calls.

    Returns:
        Computed result as ``tuple[str | int, str | int]``.
    """
    state = get_context() if context is None else context
    resolved_date = _resolve_date(state.panel, date)
    month = _month_scores(state, resolved_date)
    held = month.loc[month["held"]].sort_values(
        ["weight", "market_cap"], ascending=[False, False]
    )
    not_held = month.loc[
        month["investable"].fillna(False)
        & ~month["held"]
        & month["composite_score"].notna()
    ].sort_values(["composite_score", "market_cap"], ascending=[False, False])
    if held.empty or not_held.empty:
        raise ValueError("Could not find both a held and a non-held worked example.")

    def identifier(row: pd.Series) -> str | int:
        return str(row["ticker"]) if pd.notna(row["ticker"]) else int(row["permno"])

    return identifier(held.iloc[0]), identifier(not_held.iloc[0])


__all__ = [
    "ExplainerContext",
    "RankingContext",
    "StockExplanation",
    "StockRankingReport",
    "build_explainer_context",
    "build_ranking_context",
    "clear_context_cache",
    "compare_stocks",
    "explain_stock",
    "get_context",
    "get_ranking_context",
    "screen",
    "stock_rank_history",
    "worked_example_names",
]
