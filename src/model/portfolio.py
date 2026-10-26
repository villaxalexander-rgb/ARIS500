"""Construct and backtest the configured long-only composite portfolio.

Reads ``data/panel_russell3000.parquet`` and ``data/factors.parquet`` and writes
no files; it returns holdings, diagnostics, and gross/net return histories.
Selection holds the configured top-ranked fraction with a rarely binding count
guardrail; weights follow the configured benchmark scheme subject to
the optional minimum-position floor and name/sector/beta constraints, terminal
rows are excluded, and transaction costs vary by contemporaneous size bucket.
The name/sector/beta weight solver runs in C++ (``native/aris_core.cpp``) when
it can be built, and in the identical Python reference below otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BETA_CONSTRAINT_ENABLED,
    BUY_PERCENTILE,
    COMPOSITE_START,
    FACTORS_OUTPUT_PATH,
    HOLDINGS_BAND,
    MAX_BETA,
    MAX_NAME_WEIGHT,
    MAX_SECTOR_DEVIATION,
    MIN_BETA,
    MIN_POSITION_WEIGHT,
    N_DECILES,
    PANEL_RUSSELL3000_OUTPUT_PATH,
    SELL_PERCENTILE,
    TARGET_HOLDINGS,
    TRADING_COST_BPS,
    WEIGHT_SCHEME,
)
from src import native_core
from src.model.composite import build_model_score
from src.signals.beta import market_beta
from src.utils import forward_returns


_TOLERANCE = 1e-10
WEIGHT_SCHEMES = ("benchmark", "sqrt_benchmark", "blend")


@dataclass
class PortfolioConstruction:
    """Security weights and monthly diagnostics from portfolio construction."""

    holdings: pd.DataFrame
    monthly_diagnostics: pd.DataFrame
    sector_weights: pd.DataFrame


@dataclass
class PortfolioBacktest:
    """Portfolio construction, return history, and performance summaries."""

    construction: PortfolioConstruction
    monthly_returns: pd.DataFrame
    performance: pd.DataFrame
    annual_returns: pd.DataFrame


@dataclass
class WeightConstraintDiagnostics:
    """Beta coverage and constraint effect for one formation month."""

    pre_constraint_beta: float
    post_constraint_beta: float
    beta_constraint_bound: bool
    beta_observed_weight: float
    beta_missing_names: int
    minimum_weight_drops: int


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Russell research panel and monthly factor returns.

    Args:
        None.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame]``.
    """
    missing = [
        path
        for path in (PANEL_RUSSELL3000_OUTPUT_PATH, FACTORS_OUTPUT_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing portfolio inputs: {missing}")
    return (
        pd.read_parquet(PANEL_RUSSELL3000_OUTPUT_PATH),
        pd.read_parquet(FACTORS_OUTPUT_PATH),
    )


def _validate_settings(
    max_name_weight: float = MAX_NAME_WEIGHT,
    min_position_weight: float = MIN_POSITION_WEIGHT,
) -> None:
    """Reject portfolio settings that cannot define the requested portfolio."""
    if TARGET_HOLDINGS <= 0:
        raise ValueError("TARGET_HOLDINGS must be positive.")
    if not 0 <= HOLDINGS_BAND < TARGET_HOLDINGS:
        raise ValueError("HOLDINGS_BAND must be nonnegative and below TARGET_HOLDINGS.")
    if not 0 < BUY_PERCENTILE <= 1:
        raise ValueError("BUY_PERCENTILE must be in (0, 1].")
    if not 0 < SELL_PERCENTILE <= 1:
        raise ValueError("SELL_PERCENTILE must be in (0, 1].")
    if BUY_PERCENTILE > SELL_PERCENTILE:
        raise ValueError(
            "The buy percentile cannot be wider than the retention threshold."
        )
    if not 0 < max_name_weight <= 1:
        raise ValueError("The name-weight cap must be in (0, 1].")
    if not 0 <= min_position_weight <= max_name_weight:
        raise ValueError(
            "The minimum position weight must be nonnegative and no greater "
            "than the name-weight cap."
        )
    minimum_holdings = TARGET_HOLDINGS - HOLDINGS_BAND
    if minimum_holdings * max_name_weight < 1.0 - _TOLERANCE:
        raise ValueError("The minimum holdings count and name cap cannot reach 100%.")
    if not 0 <= MAX_SECTOR_DEVIATION <= 1:
        raise ValueError("MAX_SECTOR_DEVIATION must be in [0, 1].")
    if WEIGHT_SCHEME not in WEIGHT_SCHEMES:
        raise ValueError(
            f"WEIGHT_SCHEME must be one of {WEIGHT_SCHEMES}; got {WEIGHT_SCHEME!r}."
        )
    if not 0 < MIN_BETA <= MAX_BETA:
        raise ValueError("Beta limits must satisfy 0 < MIN_BETA <= MAX_BETA.")
    if set(TRADING_COST_BPS) != {"large", "mid", "small"}:
        raise ValueError(
            "TRADING_COST_BPS must contain exactly large, mid, and small."
        )
    if any(value < 0 for value in TRADING_COST_BPS.values()):
        raise ValueError("Trading-cost assumptions cannot be negative.")


def _sector_label(values: pd.Series) -> pd.Series:
    """Return stable sector labels, retaining missing classifications visibly."""
    numeric = pd.to_numeric(values, errors="coerce")
    labels = numeric.round().astype("Int64").astype("string")
    return labels.fillna("Unknown")


def _bounded_scale(
    target: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    total: float,
) -> pd.Series:
    """Scale positive targets to a total while respecting lower and upper bounds."""
    target = target.astype(float)
    lower = lower.reindex(target.index).astype(float)
    upper = upper.reindex(target.index).astype(float)
    target_values = target.to_numpy(dtype="float64")
    lower_values = lower.to_numpy(dtype="float64")
    upper_values = upper.to_numpy(dtype="float64")
    if np.any(lower_values > upper_values + _TOLERANCE):
        raise ValueError("Portfolio constraints have a lower bound above an upper bound.")
    if lower_values.sum() > total + _TOLERANCE or upper_values.sum() < total - _TOLERANCE:
        raise ValueError(
            "Portfolio constraints are infeasible: bounded weights cannot sum "
            f"to {total:.6f}."
        )

    positive_target = np.clip(target_values, np.finfo(float).eps, None)
    low = 0.0
    high = 1.0
    while (
        np.clip(high * positive_target, lower_values, upper_values).sum()
        < total - _TOLERANCE
    ):
        high *= 2.0
    for _ in range(60):
        midpoint = (low + high) / 2.0
        candidate = np.clip(
            midpoint * positive_target,
            lower_values,
            upper_values,
        )
        if candidate.sum() < total:
            low = midpoint
        else:
            high = midpoint
    result_values = np.clip(
        high * positive_target,
        lower_values,
        upper_values,
    )
    residual = total - float(result_values.sum())
    if abs(residual) > _TOLERANCE:
        room = (
            upper_values - result_values
            if residual > 0
            else result_values - lower_values
        )
        eligible = room > _TOLERANCE
        if np.any(eligible):
            result_values[eligible] += residual * (
                room[eligible] / room[eligible].sum()
            )
    return pd.Series(result_values, index=target.index, dtype="float64")


def _rank_deciles(candidates: pd.DataFrame) -> pd.DataFrame:
    """Rank eligible names into descending score deciles deterministically."""
    ranked = candidates.sort_values(
        ["composite_score", "market_cap", "permno"],
        ascending=[False, False, True],
        kind="mergesort",
    ).copy()
    count = len(ranked)
    ranked["score_rank"] = np.arange(1, count + 1, dtype=np.int64)
    ranked["decile"] = np.ceil(ranked["score_rank"] * N_DECILES / count).astype(
        "int8"
    )
    return ranked


def _buy_mask(ranked: pd.DataFrame) -> pd.Series:
    """Identify the configured top fraction using deterministic score ranks."""
    buy_count = int(np.floor(len(ranked) * BUY_PERCENTILE))
    return ranked["score_rank"].le(buy_count)


def _sell_mask(ranked: pd.DataFrame) -> pd.Series:
    """Identify names at or above the configured incumbent retention cutoff."""
    sell_count = int(np.floor(len(ranked) * SELL_PERCENTILE))
    return ranked["score_rank"].le(sell_count)


def _select_holdings(ranked: pd.DataFrame, previous: set[int]) -> pd.DataFrame:
    """Apply entry/exit thresholds and resize only outside the guardrail."""
    eligible = _natural_holdings(ranked, previous)
    minimum = min(TARGET_HOLDINGS - HOLDINGS_BAND, len(ranked))
    maximum = min(TARGET_HOLDINGS + HOLDINGS_BAND, len(ranked))
    if len(eligible) < minimum:
        additions = ranked.loc[~ranked["permno"].isin(eligible["permno"])].head(
            minimum - len(eligible)
        )
        eligible = pd.concat([eligible, additions], ignore_index=True)
    eligible = eligible.sort_values(["score_rank", "permno"], kind="mergesort")
    if len(eligible) > maximum:
        eligible = eligible.head(maximum)
    return eligible.copy()


def _natural_holdings(ranked: pd.DataFrame, previous: set[int]) -> pd.DataFrame:
    """Return names qualifying under the entry and retention thresholds."""
    return ranked.loc[
        _buy_mask(ranked)
        | (
            ranked["permno"].isin(previous)
            & _sell_mask(ranked)
        )
    ].copy()


def _benchmark_weights(month: pd.DataFrame) -> pd.DataFrame:
    """Compute full-universe security and sector market-cap weights."""
    benchmark = month.loc[
        month["in_universe"].fillna(False)
        & pd.to_numeric(month["market_cap"], errors="coerce").gt(0),
        ["permno", "market_cap", "gsector"],
    ].copy()
    if benchmark.empty:
        raise ValueError(f"No benchmark members with market cap at {month['date'].iloc[0]}.")
    benchmark["sector"] = _sector_label(benchmark["gsector"])
    benchmark["benchmark_weight"] = (
        benchmark["market_cap"] / benchmark["market_cap"].sum()
    )
    return benchmark


def compute_market_betas(panel: pd.DataFrame) -> pd.Series:
    """Return the unflipped trailing beta used by portfolio constraints.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        Row-aligned Series containing the values described above.
    """
    return market_beta(panel)


def _selected_weight_target(
    selected: pd.DataFrame,
    weight_scheme: str,
) -> pd.Series:
    """Return uncapped selected-name weights for one configured scheme."""
    if weight_scheme not in WEIGHT_SCHEMES:
        raise ValueError(
            f"Unknown weight scheme {weight_scheme!r}; expected {WEIGHT_SCHEMES}."
        )
    benchmark = selected.set_index("permno")["benchmark_weight"].astype(float)
    benchmark /= benchmark.sum()
    if weight_scheme == "benchmark":
        target = benchmark
    elif weight_scheme == "sqrt_benchmark":
        target = np.sqrt(benchmark)
        target /= target.sum()
    else:
        equal = pd.Series(1.0 / len(benchmark), index=benchmark.index)
        target = 0.5 * benchmark + 0.5 * equal
    return target / target.sum()


def _apply_minimum_position_weight(
    weighted: pd.DataFrame,
    raw_target: pd.Series,
    min_position_weight: float,
    max_name_weight: float,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Drop sub-floor raw targets and redistribute their weight proportionally."""
    normalized = raw_target.astype(float) / raw_target.astype(float).sum()
    keep = normalized.ge(min_position_weight - _TOLERANCE)
    drops = int((~keep).sum())
    retained_target = normalized.loc[keep]
    if retained_target.empty:
        raise ValueError("The minimum position weight removes every selected name.")
    if len(retained_target) * max_name_weight < 1.0 - _TOLERANCE:
        raise ValueError(
            "The minimum position weight leaves too few names to reach 100% "
            "under the configured name cap."
        )
    retained_target = retained_target / retained_target.sum()
    retained = weighted.loc[weighted["permno"].isin(retained_target.index)].copy()
    return retained, retained_target, drops


def _sector_layout(
    indexed: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[pd.Index, np.ndarray, pd.Series]:
    """Return sorted sectors, each name's sector position, and benchmark weights."""
    benchmark_sector = benchmark.groupby("sector")["benchmark_weight"].sum()
    selected_sectors = indexed.groupby("sector")["permno"].size().index
    sectors = benchmark_sector.index.union(selected_sectors)
    codes = sectors.get_indexer(indexed["sector"]).astype(np.int32)
    return sectors, codes, benchmark_sector.reindex(sectors, fill_value=0.0)


def _sector_table(
    sectors: pd.Index,
    benchmark_sector: pd.Series,
    portfolio_sector: np.ndarray,
) -> pd.DataFrame:
    """Tabulate benchmark, portfolio, and active weight by sector."""
    sector_table = pd.DataFrame(
        {
            "benchmark_sector_weight": benchmark_sector,
            "portfolio_sector_weight": pd.Series(portfolio_sector, index=sectors),
        }
    ).reset_index(names="sector")
    sector_table["sector_deviation"] = (
        sector_table["portfolio_sector_weight"]
        - sector_table["benchmark_sector_weight"]
    )
    return sector_table


def _raise_native_status(status: int) -> None:
    """Translate a native solver failure into the Python reference's error."""
    if status == native_core.LOWER_ABOVE_UPPER:
        raise ValueError("Portfolio constraints have a lower bound above an upper bound.")
    if status == native_core.INFEASIBLE_TOTAL:
        raise ValueError(
            "Portfolio constraints are infeasible: bounded weights cannot sum "
            "to the required total."
        )
    if status != native_core.OK:
        raise RuntimeError(f"Native weight solver failed with status {status}.")


def _native_name_sector_caps(
    weighted: pd.DataFrame,
    benchmark: pd.DataFrame,
    raw_target: pd.Series,
    max_name_weight: float,
    min_position_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run ``_apply_name_sector_caps`` through the C++ solver."""
    indexed = weighted.set_index("permno", drop=False)
    sectors, codes, benchmark_sector = _sector_layout(indexed, benchmark)
    status, weights, portfolio_sector = native_core.name_sector_caps(
        raw_target.reindex(indexed.index).astype(float).to_numpy(),
        codes,
        benchmark_sector.to_numpy(),
        max_name_weight,
        min_position_weight,
        MAX_SECTOR_DEVIATION,
    )
    _raise_native_status(status)
    return (
        pd.Series(weights, index=indexed.index, dtype="float64"),
        _sector_table(sectors, benchmark_sector, portfolio_sector),
    )


def _apply_name_sector_caps(
    weighted: pd.DataFrame,
    benchmark: pd.DataFrame,
    raw_target: pd.Series,
    max_name_weight: float,
    min_position_weight: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Project raw weights onto full-investment, position, and sector bounds."""
    if native_core.available():
        return _native_name_sector_caps(
            weighted, benchmark, raw_target, max_name_weight, min_position_weight
        )
    indexed = weighted.set_index("permno", drop=False)
    raw_target = raw_target.reindex(indexed.index).astype(float)
    raw_target = raw_target / raw_target.sum()
    benchmark_sector = benchmark.groupby("sector")["benchmark_weight"].sum()
    raw_sector = raw_target.groupby(indexed["sector"]).sum()
    selected_counts = indexed.groupby("sector")["permno"].size().astype(float)
    sectors = benchmark_sector.index.union(raw_sector.index)
    benchmark_sector = benchmark_sector.reindex(sectors, fill_value=0.0)
    raw_sector = raw_sector.reindex(sectors, fill_value=0.0)
    capacity = selected_counts.reindex(sectors, fill_value=0.0) * max_name_weight
    minimum_capacity = (
        selected_counts.reindex(sectors, fill_value=0.0) * min_position_weight
    )
    lower = (benchmark_sector - MAX_SECTOR_DEVIATION).clip(lower=0.0)
    lower = pd.concat([lower, minimum_capacity], axis=1).max(axis=1)
    upper = (benchmark_sector + MAX_SECTOR_DEVIATION).clip(upper=1.0)
    upper = pd.concat([upper, capacity], axis=1).min(axis=1)
    absent = selected_counts.reindex(sectors, fill_value=0.0).eq(0)
    lower.loc[absent] = 0.0
    upper.loc[absent] = 0.0
    sector_target = _bounded_scale(raw_sector, lower, upper, 1.0)

    name_weights = pd.Series(0.0, index=indexed.index, dtype="float64")
    for sector, group in indexed.groupby("sector", sort=True):
        raw = raw_target.reindex(group.index)
        name_weights.loc[group.index] = _bounded_scale(
            raw,
            pd.Series(min_position_weight, index=raw.index),
            pd.Series(max_name_weight, index=raw.index),
            float(sector_target.loc[sector]),
        )
    portfolio_sector = name_weights.groupby(indexed["sector"]).sum()
    sector_table = pd.DataFrame(
        {
            "benchmark_sector_weight": benchmark_sector,
            "portfolio_sector_weight": portfolio_sector.reindex(
                sectors, fill_value=0.0
            ),
        }
    ).reset_index(names="sector")
    sector_table["sector_deviation"] = (
        sector_table["portfolio_sector_weight"]
        - sector_table["benchmark_sector_weight"]
    )
    return name_weights, sector_table


def _portfolio_beta(weights: pd.Series, betas: pd.Series) -> float:
    """Return the fully invested portfolio's weighted trailing market beta."""
    return float((weights * betas.reindex(weights.index)).sum())


def _beta_constrained_weights(
    weighted: pd.DataFrame,
    benchmark: pd.DataFrame,
    initial_weights: pd.Series,
    constraint_betas: pd.Series,
    target_beta: float,
    max_name_weight: float,
    min_position_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Tilt minimally along beta until the nearest configured bound is reached."""
    if native_core.available():
        indexed = weighted.set_index("permno", drop=False)
        sectors, codes, benchmark_sector = _sector_layout(indexed, benchmark)
        status, weights, portfolio_sector = native_core.beta_constrained_weights(
            initial_weights.reindex(indexed.index).to_numpy(dtype="float64"),
            constraint_betas.reindex(indexed.index).to_numpy(dtype="float64"),
            codes,
            benchmark_sector.to_numpy(),
            target_beta,
            max_name_weight,
            min_position_weight,
            MAX_SECTOR_DEVIATION,
        )
        if status == native_core.BETA_INFEASIBLE:
            direction = (
                "below"
                if _portfolio_beta(initial_weights, constraint_betas) > target_beta
                else "above"
            )
            raise ValueError(
                f"Beta constraint is infeasible: could not move beta {direction} "
                f"{target_beta:.4f} while respecting name and sector caps."
            )
        _raise_native_status(status)
        return (
            pd.Series(weights, index=indexed.index, dtype="float64"),
            _sector_table(sectors, benchmark_sector, portfolio_sector),
        )
    center = _portfolio_beta(initial_weights, constraint_betas)

    def candidate(tilt: float) -> tuple[pd.Series, pd.DataFrame, float]:
        exponent = np.clip(
            tilt * (constraint_betas.reindex(initial_weights.index) - center),
            -50.0,
            50.0,
        )
        raw = initial_weights * np.exp(exponent)
        weights, sectors = _apply_name_sector_caps(
            weighted,
            benchmark,
            raw,
            max_name_weight,
            min_position_weight,
        )
        return weights, sectors, _portfolio_beta(weights, constraint_betas)

    lowering = center > target_beta
    step = -1.0 if lowering else 1.0
    endpoint = candidate(step)
    for _ in range(20):
        reached = endpoint[2] <= target_beta if lowering else endpoint[2] >= target_beta
        if reached:
            break
        step *= 2.0
        endpoint = candidate(step)
    else:
        direction = "below" if lowering else "above"
        raise ValueError(
            f"Beta constraint is infeasible: could not move beta {direction} "
            f"{target_beta:.4f} while respecting name and sector caps."
        )

    zero_side = 0.0
    reached_side = step
    best = endpoint
    # Thirty-five rounds leave beta error far below a basis point while avoiding
    # meaningless machine-precision work in a monthly portfolio rebalance.
    for _ in range(35):
        midpoint = (zero_side + reached_side) / 2.0
        trial = candidate(midpoint)
        still_breaching = trial[2] > target_beta if lowering else trial[2] < target_beta
        if still_breaching:
            zero_side = midpoint
        else:
            reached_side = midpoint
            best = trial
    return best[0], best[1]


def _constrained_weights(
    selected: pd.DataFrame,
    benchmark: pd.DataFrame,
    weight_scheme: str = WEIGHT_SCHEME,
    apply_beta_constraint: bool = BETA_CONSTRAINT_ENABLED,
    max_name_weight: float = MAX_NAME_WEIGHT,
    min_position_weight: float = MIN_POSITION_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, WeightConstraintDiagnostics]:
    """Apply the position floor, caps, and optional trailing-beta bound."""
    weighted = selected.copy()
    weighted["sector"] = _sector_label(weighted["gsector"])
    weighted = weighted.merge(
        benchmark[["permno", "benchmark_weight"]],
        on="permno",
        how="left",
        validate="1:1",
    )
    raw_target = _selected_weight_target(weighted, weight_scheme)
    weighted, raw_target, minimum_weight_drops = _apply_minimum_position_weight(
        weighted,
        raw_target,
        min_position_weight,
        max_name_weight,
    )
    initial_weights, sector_table = _apply_name_sector_caps(
        weighted,
        benchmark,
        raw_target,
        max_name_weight,
        min_position_weight,
    )
    observed_beta = pd.to_numeric(weighted["market_beta"], errors="coerce")
    observed_beta.index = weighted["permno"]
    # A newly listed name can lack the 24 observations required by risk.py.
    # Keep it investable but use the selected cross-section's median beta only
    # for the constraint calculation; observed-weight coverage stays explicit.
    beta_imputation = observed_beta.median()
    if pd.isna(beta_imputation):
        beta_imputation = 1.0
    constraint_betas = observed_beta.fillna(float(beta_imputation))
    pre_beta = _portfolio_beta(initial_weights, constraint_betas)
    final_weights = initial_weights
    bound = False
    if apply_beta_constraint and pre_beta > MAX_BETA + 1e-10:
        final_weights, sector_table = _beta_constrained_weights(
            weighted,
            benchmark,
            initial_weights,
            constraint_betas,
            MAX_BETA,
            max_name_weight,
            min_position_weight,
        )
        bound = True
    elif apply_beta_constraint and pre_beta < MIN_BETA - 1e-10:
        final_weights, sector_table = _beta_constrained_weights(
            weighted,
            benchmark,
            initial_weights,
            constraint_betas,
            MIN_BETA,
            max_name_weight,
            min_position_weight,
        )
        bound = True
    post_beta = _portfolio_beta(final_weights, constraint_betas)
    weighted["weight"] = weighted["permno"].map(final_weights)
    weighted["raw_target_weight"] = weighted["permno"].map(raw_target)
    weighted["market_beta"] = weighted["permno"].map(observed_beta)
    weighted["constraint_beta"] = weighted["permno"].map(constraint_betas)

    if abs(weighted["weight"].sum() - 1.0) > 1e-8:
        raise AssertionError("Portfolio is not fully invested.")
    if weighted["weight"].max() > max_name_weight + 1e-8:
        raise AssertionError("Portfolio breaches its name-weight cap.")
    if weighted["weight"].min() < min_position_weight - 1e-8:
        raise AssertionError("Portfolio breaches its minimum position weight.")
    if sector_table["sector_deviation"].abs().max() > MAX_SECTOR_DEVIATION + 1e-8:
        raise AssertionError("Portfolio breaches MAX_SECTOR_DEVIATION.")
    if apply_beta_constraint and not MIN_BETA - 1e-7 <= post_beta <= MAX_BETA + 1e-7:
        raise AssertionError(f"Portfolio beta remains outside its bounds: {post_beta}.")
    beta_observed_weight = float(
        weighted.loc[weighted["market_beta"].notna(), "weight"].sum()
    )
    diagnostics = WeightConstraintDiagnostics(
        pre_constraint_beta=pre_beta,
        post_constraint_beta=post_beta,
        beta_constraint_bound=bound,
        beta_observed_weight=beta_observed_weight,
        beta_missing_names=int(weighted["market_beta"].isna().sum()),
        minimum_weight_drops=minimum_weight_drops,
    )
    return weighted, sector_table, diagnostics


def _active_share(weighted: pd.DataFrame, benchmark: pd.DataFrame) -> float:
    """Return active share against the full market-cap-weighted universe."""
    comparison = benchmark.set_index("permno")[["benchmark_weight"]].join(
        weighted.set_index("permno")[["weight"]], how="outer"
    ).fillna(0.0)
    return float(
        0.5 * (comparison["weight"] - comparison["benchmark_weight"]).abs().sum()
    )


def _terminal_rows(panel: pd.DataFrame) -> pd.Series:
    """Flag each security's terminal panel observation before the sample end.

    A completed CRSP history makes a security's true final observation
    identifiable. This proxies for contemporaneously known delisting status
    without labeling the panel-wide sample endpoint as terminal. Shared by
    the backtest, the live rebalance, and the stock explainer so all three
    exclude the same rows from selection.
    """
    dates = pd.to_datetime(panel["date"])
    security_last = dates.groupby(panel["permno"]).transform("max")
    terminal = dates.eq(security_last) & security_last.lt(dates.max())
    if "delret_imputed" in panel.columns:
        terminal |= panel["delret_imputed"].fillna(False).astype(bool)
    return terminal


def _size_buckets(month: pd.DataFrame) -> pd.Series:
    """Assign trading-cost buckets from formation-month investable market caps."""
    eligible = month.loc[
        month["investable"].fillna(False)
        & pd.to_numeric(month["market_cap"], errors="coerce").gt(0),
        ["permno", "market_cap"],
    ].sort_values(["market_cap", "permno"], ascending=[False, True])
    rank = pd.Series(np.arange(1, len(eligible) + 1), index=eligible["permno"])
    bucket = pd.Series("small", index=rank.index, dtype="object")
    bucket.loc[rank.le(1500)] = "mid"
    bucket.loc[rank.le(500)] = "large"
    return bucket


def construct_portfolio(
    panel: pd.DataFrame,
    composite_score: pd.Series | None = None,
    market_beta: pd.Series | None = None,
    weight_scheme: str = WEIGHT_SCHEME,
    apply_beta_constraint: bool = BETA_CONSTRAINT_ENABLED,
    max_name_weight: float = MAX_NAME_WEIGHT,
    min_position_weight: float = MIN_POSITION_WEIGHT,
) -> PortfolioConstruction:
    """Build selected, constrained monthly target weights from composite scores.

    Args:
        panel: Monthly point-in-time research panel.
        composite_score: Optional precomputed row-aligned composite score.
        market_beta: Optional precomputed row-aligned market beta.
        weight_scheme: Portfolio weighting scheme to apply.
        apply_beta_constraint: Whether to enforce the configured beta band.
        max_name_weight: Maximum allowed security weight.
        min_position_weight: Raw target floor applied before portfolio caps.

    Returns:
        Computed result as ``PortfolioConstruction``.
    """
    _validate_settings(max_name_weight, min_position_weight)
    required = {
        "permno",
        "date",
        "investable",
        "in_universe",
        "market_cap",
        "gsector",
        "ret",
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Panel is missing portfolio columns: {sorted(missing)}")
    score = build_model_score(panel) if composite_score is None else composite_score
    if len(score) != len(panel):
        raise ValueError("Composite score and panel must have the same row count.")
    beta = compute_market_betas(panel) if market_beta is None else market_beta
    if len(beta) != len(panel):
        raise ValueError("Market beta and panel must have the same row count.")
    if weight_scheme not in WEIGHT_SCHEMES:
        raise ValueError(
            f"Unknown weight scheme {weight_scheme!r}; expected {WEIGHT_SCHEMES}."
        )

    work = panel.loc[:, sorted(required)].copy()
    terminal_row = _terminal_rows(panel)
    work = work.assign(
        date=pd.to_datetime(work["date"]),
        composite_score=pd.Series(score, index=panel.index, dtype="float64"),
        market_beta=pd.Series(beta, index=panel.index, dtype="float64"),
        market_cap=pd.to_numeric(work["market_cap"], errors="coerce"),
        terminal_row=terminal_row.to_numpy(),
    )
    work = work.loc[work["date"].ge(pd.Timestamp(COMPOSITE_START))]

    holding_pieces: list[pd.DataFrame] = []
    sector_pieces: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    previous: set[int] = set()
    for date, month in work.groupby("date", sort=True):
        candidates_before_terminal_filter = month.loc[
            month["investable"].fillna(False)
            & month["composite_score"].notna()
            & month["market_cap"].gt(0)
        ].copy()
        if candidates_before_terminal_filter.empty:
            continue
        prefilter_ranked = _rank_deciles(candidates_before_terminal_filter)
        terminal_selection = prefilter_ranked["terminal_row"] & (
            _buy_mask(prefilter_ranked)
            | (
                prefilter_ranked["permno"].isin(previous)
                & _sell_mask(prefilter_ranked)
            )
        )
        terminal_rows_excluded = int(
            candidates_before_terminal_filter["terminal_row"].sum()
        )
        terminal_selections_prevented = int(terminal_selection.sum())
        candidates = candidates_before_terminal_filter.loc[
            ~candidates_before_terminal_filter["terminal_row"]
        ].copy()
        if candidates.empty:
            continue
        ranked = _rank_deciles(candidates)
        natural_selection = _natural_holdings(ranked, previous)
        minimum_holdings = min(TARGET_HOLDINGS - HOLDINGS_BAND, len(ranked))
        maximum_holdings = min(TARGET_HOLDINGS + HOLDINGS_BAND, len(ranked))
        holdings_band_bound = bool(
            len(natural_selection) < minimum_holdings
            or len(natural_selection) > maximum_holdings
        )
        selected = _select_holdings(ranked, previous)
        benchmark = _benchmark_weights(month)
        weighted, sectors, weight_diagnostics = _constrained_weights(
            selected,
            benchmark,
            weight_scheme=weight_scheme,
            apply_beta_constraint=apply_beta_constraint,
            max_name_weight=max_name_weight,
            min_position_weight=min_position_weight,
        )
        weighted = weighted.assign(
            date=date,
            size_bucket=weighted["permno"].map(_size_buckets(month)),
        )
        active_share = _active_share(weighted, benchmark)
        max_sector_row = sectors.loc[sectors["sector_deviation"].abs().idxmax()]
        diagnostic_rows.append(
            {
                "date": date,
                "investable_names": len(candidates),
                "terminal_rows_excluded": terminal_rows_excluded,
                "terminal_selections_prevented": terminal_selections_prevented,
                "natural_holdings": len(natural_selection),
                "holdings": len(weighted),
                "minimum_weight_drops": weight_diagnostics.minimum_weight_drops,
                "minimum_final_weight": weighted["weight"].min(),
                "holdings_band_bound": holdings_band_bound,
                "max_name_weight": weighted["weight"].max(),
                "top_10_weight": weighted.nlargest(10, "weight")["weight"].sum(),
                "names_at_name_cap": int(
                    np.isclose(
                        weighted["weight"],
                        max_name_weight,
                        atol=1e-8,
                        rtol=0.0,
                    ).sum()
                ),
                "herfindahl": np.square(weighted["weight"]).sum(),
                "max_abs_sector_deviation": abs(max_sector_row["sector_deviation"]),
                "largest_sector_deviation": max_sector_row["sector_deviation"],
                "largest_deviation_sector": max_sector_row["sector"],
                "active_share": active_share,
                "pre_constraint_beta": (
                    weight_diagnostics.pre_constraint_beta
                ),
                "post_constraint_beta": (
                    weight_diagnostics.post_constraint_beta
                ),
                "beta_constraint_bound": (
                    weight_diagnostics.beta_constraint_bound
                ),
                "beta_observed_weight": (
                    weight_diagnostics.beta_observed_weight
                ),
                "beta_missing_names": (
                    weight_diagnostics.beta_missing_names
                ),
                "weight_scheme": weight_scheme,
                "beta_constraint_enabled": apply_beta_constraint,
            }
        )
        holding_pieces.append(
            weighted.loc[
                :,
                [
                    "date",
                    "permno",
                    "composite_score",
                    "score_rank",
                    "decile",
                    "sector",
                    "size_bucket",
                    "market_cap",
                    "benchmark_weight",
                    "market_beta",
                    "constraint_beta",
                    "weight",
                ],
            ]
        )
        sector_pieces.append(sectors.assign(date=date))
        previous = set(weighted["permno"].astype(int))

    if not holding_pieces:
        raise ValueError("No portfolio could be formed over the configured period.")
    return PortfolioConstruction(
        holdings=pd.concat(holding_pieces, ignore_index=True),
        monthly_diagnostics=pd.DataFrame(diagnostic_rows),
        sector_weights=pd.concat(sector_pieces, ignore_index=True),
    )


def _turnover_and_costs(
    panel: pd.DataFrame,
    construction: PortfolioConstruction,
) -> pd.DataFrame:
    """Compute weight turnover, name turnover, and size-bucket trading costs."""
    returns = panel.assign(date=pd.to_datetime(panel["date"])).set_index(
        ["date", "permno"]
    )["ret"]
    rows: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    previous_buckets: pd.Series | None = None
    previous_date: pd.Timestamp | None = None
    for date, group in construction.holdings.groupby("date", sort=True):
        target = group.set_index("permno")["weight"].astype(float)
        current_buckets = group.set_index("permno")["size_bucket"]
        if previous_weights is None:
            pretrade = pd.Series(dtype="float64")
            delta = target.copy()
            turnover = 1.0
            name_entries = len(target)
            name_exits = 0
            name_turnover = 1.0
        else:
            realized = returns.reindex(
                pd.MultiIndex.from_arrays(
                    [[date] * len(previous_weights), previous_weights.index],
                    names=["date", "permno"],
                )
            )
            realized.index = previous_weights.index
            drifted = previous_weights * (1.0 + realized.fillna(0.0))
            drifted = drifted.clip(lower=0.0)
            pretrade = drifted / drifted.sum() if drifted.sum() > 0 else drifted
            names = target.index.union(pretrade.index)
            delta = target.reindex(names, fill_value=0.0) - pretrade.reindex(
                names, fill_value=0.0
            )
            turnover = 0.5 * float(delta.abs().sum())
            current_names = set(target.index)
            previous_names = set(previous_weights.index)
            name_entries = len(current_names - previous_names)
            name_exits = len(previous_names - current_names)
            # Symmetric name turnover: a complete replacement equals 100%.
            name_turnover = (name_entries + name_exits) / (
                len(current_names) + len(previous_names)
            )

        trade_buckets = current_buckets.reindex(delta.index)
        if previous_buckets is not None:
            trade_buckets = trade_buckets.fillna(previous_buckets.reindex(delta.index))
        if trade_buckets.isna().any():
            raise AssertionError("A traded security is missing its size-cost bucket.")
        rates = trade_buckets.map(TRADING_COST_BPS).astype(float) / 10_000.0
        trading_cost = float((delta.abs() * rates).sum())
        rows.append(
            {
                "date": date,
                "turnover": turnover,
                "drift_adjusted_turnover": turnover,
                "name_turnover": name_turnover,
                "name_entries": name_entries,
                "name_exits": name_exits,
                "gross_traded_weight": float(delta.abs().sum()),
                "trading_cost": trading_cost,
                "previous_formation_date": previous_date,
            }
        )
        previous_weights = target
        previous_buckets = current_buckets
        previous_date = date
    return pd.DataFrame(rows)


def backtest_portfolio(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    construction: PortfolioConstruction,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply next-month returns, costs, and Russell benchmark comparisons.

    Args:
        panel: Monthly point-in-time research panel.
        factors: Monthly factor and benchmark-return rows.
        construction: Portfolio holdings and construction diagnostics.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]``.
    """
    panel_dates = pd.to_datetime(panel["date"])
    forward = forward_returns(panel)
    forward_map = pd.Series(
        forward.to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [panel_dates, panel["permno"]], names=["date", "permno"]
        ),
    )
    turnover = _turnover_and_costs(panel, construction)
    diagnostics = construction.monthly_diagnostics.merge(
        turnover, on="date", how="left", validate="1:1"
    )
    construction.monthly_diagnostics = diagnostics

    factor_data = factors.copy().assign(date=pd.to_datetime(factors["date"]))
    factor_data["return_month"] = factor_data["date"].dt.to_period("M")
    factor_data = factor_data.set_index("return_month")
    rows: list[dict[str, object]] = []
    for date, group in construction.holdings.groupby("date", sort=True):
        keys = pd.MultiIndex.from_arrays(
            [[date] * len(group), group["permno"]], names=["date", "permno"]
        )
        security_returns = forward_map.reindex(keys)
        security_returns.index = group.index
        missing_weight = float(group.loc[security_returns.isna(), "weight"].sum())
        # Missing next-month returns never affect selection—that would use future
        # information. They remain visible below and earn zero in the backtest.
        gross_return = float((group["weight"] * security_returns.fillna(0.0)).sum())
        return_month = date.to_period("M") + 1
        if return_month not in factor_data.index:
            continue
        factor_row = factor_data.loc[return_month]
        if isinstance(factor_row, pd.DataFrame):
            raise ValueError(f"Duplicate factor month: {return_month}")
        cost = float(
            diagnostics.loc[diagnostics["date"].eq(date), "trading_cost"].iloc[0]
        )
        rows.append(
            {
                "formation_date": date,
                "return_date": factor_row["date"],
                "gross_return": gross_return,
                "net_return": (1.0 - cost) * (1.0 + gross_return) - 1.0,
                "trading_cost": cost,
                "benchmark_return": factor_row["russell3000_total_return"],
                "rf": factor_row["rf"],
                "missing_return_weight": missing_weight,
            }
        )
    monthly = pd.DataFrame(rows).merge(
        diagnostics,
        left_on="formation_date",
        right_on="date",
        how="left",
        validate="1:1",
        suffixes=("", "_diagnostic"),
    ).drop(columns=["date", "trading_cost_diagnostic"])

    performance_rows: list[dict[str, object]] = []
    for label, column in (("gross", "gross_return"), ("net", "net_return")):
        sample = monthly.dropna(subset=[column, "benchmark_return", "rf"])
        returns = sample[column].astype(float)
        benchmark = sample["benchmark_return"].astype(float)
        excess = returns - sample["rf"].astype(float)
        active = returns - sample["benchmark_return"].astype(float)
        volatility = returns.std(ddof=1) * np.sqrt(12.0)
        excess_volatility = excess.std(ddof=1) * np.sqrt(12.0)
        tracking_error = active.std(ddof=1) * np.sqrt(12.0)
        performance_rows.append(
            {
                "return_basis": label,
                "months": len(sample),
                "annualized_return": (1.0 + returns).prod() ** (12.0 / len(sample)) - 1.0,
                "benchmark_annualized_return": (
                    (1.0 + benchmark).prod() ** (12.0 / len(sample)) - 1.0
                ),
                "annualized_volatility": volatility,
                "sharpe": (
                    excess.mean() * 12.0 / excess_volatility
                    if excess_volatility > 0
                    else np.nan
                ),
                "tracking_error": tracking_error,
                "information_ratio": (
                    active.mean() * 12.0 / tracking_error
                    if tracking_error > 0
                    else np.nan
                ),
                "average_active_share": sample["active_share"].mean(),
                "average_monthly_turnover": sample["turnover"].mean(),
                "average_monthly_drift_adjusted_turnover": sample[
                    "drift_adjusted_turnover"
                ].mean(),
                "average_monthly_name_turnover": sample["name_turnover"].mean(),
                "annualized_trading_cost": sample["trading_cost"].mean() * 12.0,
            }
        )
    performance = pd.DataFrame(performance_rows)

    annual = monthly.assign(year=monthly["return_date"].dt.year).groupby("year").agg(
        gross_return=("gross_return", lambda x: (1.0 + x).prod() - 1.0),
        net_return=("net_return", lambda x: (1.0 + x).prod() - 1.0),
        russell3000_return=(
            "benchmark_return", lambda x: (1.0 + x).prod() - 1.0
        ),
    ).reset_index()
    return monthly, performance, annual


def _print_results(result: PortfolioBacktest) -> None:
    """Print monthly construction diagnostics and backtest results."""
    monthly_diagnostics = result.construction.monthly_diagnostics.merge(
        result.monthly_returns[
            ["formation_date", "missing_return_weight"]
        ],
        left_on="date",
        right_on="formation_date",
        how="left",
        validate="1:1",
    ).drop(columns="formation_date")
    monthly_columns = [
        "date",
        "holdings",
        "minimum_weight_drops",
        "minimum_final_weight",
        "terminal_rows_excluded",
        "terminal_selections_prevented",
        "natural_holdings",
        "holdings_band_bound",
        "drift_adjusted_turnover",
        "name_turnover",
        "name_entries",
        "name_exits",
        "max_name_weight",
        "top_10_weight",
        "names_at_name_cap",
        "herfindahl",
        "max_abs_sector_deviation",
        "largest_deviation_sector",
        "active_share",
        "pre_constraint_beta",
        "post_constraint_beta",
        "beta_constraint_bound",
        "beta_observed_weight",
        "trading_cost",
        "missing_return_weight",
    ]
    print("\nMONTHLY PORTFOLIO DIAGNOSTICS")
    print(
        monthly_diagnostics[monthly_columns].to_string(
            index=False,
            formatters={
                "drift_adjusted_turnover": "{:.2%}".format,
                "name_turnover": "{:.2%}".format,
                "max_name_weight": "{:.2%}".format,
                "minimum_final_weight": "{:.3%}".format,
                "top_10_weight": "{:.2%}".format,
                "herfindahl": "{:.4f}".format,
                "max_abs_sector_deviation": "{:.2%}".format,
                "active_share": "{:.2%}".format,
                "pre_constraint_beta": "{:.3f}".format,
                "post_constraint_beta": "{:.3f}".format,
                "beta_observed_weight": "{:.2%}".format,
                "trading_cost": "{:.3%}".format,
                "missing_return_weight": "{:.3%}".format,
            },
        )
    )
    print("\nPORTFOLIO PERFORMANCE — 2007 TO 2025")
    performance_display = result.performance.drop(
        columns=["average_monthly_turnover"], errors="ignore"
    )
    print(
        performance_display.to_string(
            index=False,
            formatters={
                "annualized_return": "{:.2%}".format,
                "benchmark_annualized_return": "{:.2%}".format,
                "annualized_volatility": "{:.2%}".format,
                "sharpe": "{:.3f}".format,
                "tracking_error": "{:.2%}".format,
                "information_ratio": "{:.3f}".format,
                "average_active_share": "{:.2%}".format,
                "average_monthly_turnover": "{:.2%}".format,
                "average_monthly_drift_adjusted_turnover": "{:.2%}".format,
                "average_monthly_name_turnover": "{:.2%}".format,
                "annualized_trading_cost": "{:.2%}".format,
            },
        )
    )
    print("\nANNUAL RETURNS")
    print(
        result.annual_returns.to_string(
            index=False,
            formatters={
                "gross_return": "{:.2%}".format,
                "net_return": "{:.2%}".format,
                "russell3000_return": "{:.2%}".format,
            },
        )
    )


def run_portfolio_research(
    panel: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    composite_score: pd.Series | None = None,
) -> PortfolioBacktest:
    """Construct, backtest, print, and return the configured portfolio.

    Args:
        panel: Monthly point-in-time research panel.
        factors: Monthly factor and benchmark-return rows.
        composite_score: Optional precomputed row-aligned composite score.

    Returns:
        Computed result as ``PortfolioBacktest``.
    """
    if panel is None or factors is None:
        local_panel, local_factors = load_inputs()
        panel = local_panel if panel is None else panel
        factors = local_factors if factors is None else factors
    construction = construct_portfolio(panel, composite_score)
    monthly, performance, annual = backtest_portfolio(
        panel, factors, construction
    )
    result = PortfolioBacktest(
        construction=construction,
        monthly_returns=monthly,
        performance=performance,
        annual_returns=annual,
    )
    _print_results(result)
    return result


def main() -> None:
    """Run portfolio construction and the 2007-2025 Russell backtest.

    Args:
        None.

    Returns:
        None.
    """
    run_portfolio_research()


if __name__ == "__main__":
    main()
