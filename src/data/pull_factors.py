#!/usr/bin/env python3
"""Pull and align monthly factor and benchmark returns from WRDS.

Reads WRDS factor/index tables plus ``data/prices.parquet`` for the required
date span, caches raw queries, and writes ``data/factors.parquet``. Returns are
stored as decimals at CRSP-compatible month ends; Russell 3000 returns use the
last Compustat daily index level in each month because the September 2011
monthly level is defective, while the CRSP S&P series is retained for
replication checks.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import wrds

from config import (
    END_DATE,
    FACTOR_CACHE_DIR,
    FACTORS_OUTPUT_PATH,
    PRICES_OUTPUT_PATH,
    START_DATE,
    WRDS_USERNAME,
)


def _cache_path(name: str, sql: str, params: dict[str, Any]) -> Path:
    """Return a content-addressed path for one raw WRDS result."""
    payload = json.dumps(
        {"name": name, "sql": sql, "params": params},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return FACTOR_CACHE_DIR / f"{name}-{digest}.parquet"


def cached_query(
    db: wrds.Connection | None,
    name: str,
    sql: str,
    params: dict[str, Any],
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Read a cached query or refresh it from WRDS when requested.

    Args:
        db: Open WRDS connection.
        name: Stable cache or series name.
        sql: Parameterized SQL query text.
        params: Values bound to the SQL query.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    path = _cache_path(name, sql, params)
    if path.exists() and not refresh_from_wrds:
        print(f"Cache hit: {path}")
        return pd.read_parquet(path)
    if db is None:
        raise RuntimeError(
            f"Cache miss: {path}. Rerun with refresh_from_wrds=True."
        )

    FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Querying WRDS: {name}")
    frame = db.raw_sql(sql, params=params)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"Cached {len(frame):,} raw rows: {path}")
    return frame


def panel_period() -> tuple[str, str]:
    """Return the configured dates bounded to the local price-panel period.

    Args:
        None.

    Returns:
        Computed result as ``tuple[str, str]``.
    """
    if not PRICES_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Run pull_prices.py before factors: {PRICES_OUTPUT_PATH} is missing."
        )
    dates = pd.read_parquet(PRICES_OUTPUT_PATH, columns=["date"])["date"]
    panel_start = max(pd.Timestamp(START_DATE), pd.Timestamp(dates.min()))
    configured_end = pd.Timestamp(END_DATE) if END_DATE is not None else dates.max()
    panel_end = min(configured_end, pd.Timestamp(dates.max()))
    return panel_start.date().isoformat(), panel_end.date().isoformat()


def pull_raw_returns(
    refresh_from_wrds: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load monthly factors and benchmark index levels or returns.

    Args:
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        Factors, S&P monthly returns/levels, S&P daily levels, Russell monthly
        levels, and Russell daily levels.
    """
    start_date, end_date = panel_period()
    params = {"start_date": start_date, "end_date": end_date}
    factor_sql = """
        SELECT
            dateff AS date,
            mktrf::double precision AS mkt_excess,
            smb::double precision AS smb,
            hml::double precision AS hml,
            rmw::double precision AS rmw,
            cma::double precision AS cma,
            umd::double precision AS umd,
            rf::double precision AS rf
        FROM ff.fivefactors_monthly
        WHERE dateff BETWEEN %(start_date)s::date AND %(end_date)s::date
        ORDER BY dateff
    """
    benchmark_sql = """
        SELECT
            caldt AS date,
            vwretd::double precision AS sp500_total_return,
            spindx::double precision AS index_level
        FROM crsp.msp500_v2
        WHERE caldt >= (%(start_date)s::date - INTERVAL '1 month')
          AND caldt <= %(end_date)s::date
        ORDER BY caldt
    """
    sp500_daily_sql = """
        SELECT
            caldt AS date,
            spindx::double precision AS index_level
        FROM crsp.dsp500_v2
        WHERE caldt >= (%(start_date)s::date - INTERVAL '1 month')
          AND caldt <= %(end_date)s::date
        ORDER BY caldt
    """
    russell_monthly_sql = """
        SELECT
            datadate AS date,
            prccm::double precision AS index_level
        FROM comp.idx_mth
        WHERE gvkeyx = '156778'
          AND datadate >= (%(start_date)s::date - INTERVAL '1 month')
          AND datadate <= %(end_date)s::date
        ORDER BY datadate
    """
    # WRDS calls this physical table comp.idx_daily (the data dictionary's
    # shorthand is idx_dly). The September 2011 monthly Russell level is wrong
    # and creates false September/October returns, so production uses the last
    # valid daily level in each calendar month instead.
    russell_daily_sql = """
        SELECT
            datadate AS date,
            prccd::double precision AS index_level
        FROM comp.idx_daily
        WHERE gvkeyx = '156778'
          AND datadate >= (%(start_date)s::date - INTERVAL '1 month')
          AND datadate <= %(end_date)s::date
        ORDER BY datadate
    """
    db = (
        wrds.Connection(wrds_username=WRDS_USERNAME)
        if refresh_from_wrds
        else None
    )
    try:
        factors = cached_query(
            db,
            "ff_fivefactors_monthly",
            factor_sql,
            params,
            refresh_from_wrds,
        )
        benchmark = cached_query(
            db,
            "crsp_msp500_v2",
            benchmark_sql,
            params,
            refresh_from_wrds,
        )
        sp500_daily = cached_query(
            db,
            "crsp_dsp500_v2_levels",
            sp500_daily_sql,
            params,
            refresh_from_wrds,
        )
        russell_monthly = cached_query(
            db,
            "comp_idx_mth_russell3000_level",
            russell_monthly_sql,
            params,
            refresh_from_wrds,
        )
        russell_daily = cached_query(
            db,
            "comp_idx_daily_russell3000_level",
            russell_daily_sql,
            params,
            refresh_from_wrds,
        )
    finally:
        if db is not None:
            db.close()
    return factors, benchmark, sp500_daily, russell_monthly, russell_daily


def _last_level_by_month(raw_levels: pd.DataFrame) -> pd.DataFrame:
    """Return the last non-null index level observed in each calendar month.

    Args:
        raw_levels: Rows containing ``date`` and ``index_level``.

    Returns:
        One row per month with the source date and numeric level.
    """
    levels = raw_levels.loc[:, ["date", "index_level"]].copy(deep=True)
    levels = levels.assign(
        date=pd.to_datetime(levels["date"]),
        index_level=pd.to_numeric(levels["index_level"], errors="coerce"),
    )
    levels = levels.dropna(subset=["date", "index_level"]).sort_values("date")
    levels = levels.assign(_month=levels["date"].dt.to_period("M"))
    return (
        levels.groupby("_month", as_index=False, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )


def compare_monthly_and_daily_levels(
    monthly_levels: pd.DataFrame,
    daily_levels: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Print every monthly index level that disagrees with its last daily row.

    Args:
        monthly_levels: One or more monthly index-level rows.
        daily_levels: Daily index-level rows covering the same period.
        label: Human-readable index name used in diagnostics.

    Returns:
        The merged monthly comparison, including level and return differences.
    """
    monthly = _last_level_by_month(monthly_levels).rename(
        columns={"date": "monthly_date", "index_level": "monthly_level"}
    )
    daily = _last_level_by_month(daily_levels).rename(
        columns={"date": "daily_date", "index_level": "daily_level"}
    )
    comparison = monthly.merge(daily, on="_month", how="inner", validate="1:1")
    comparison = comparison.assign(
        monthly_return=comparison["monthly_level"].pct_change(fill_method=None),
        daily_return=comparison["daily_level"].pct_change(fill_method=None),
        level_difference=(
            comparison["monthly_level"] - comparison["daily_level"]
        ),
    )
    comparison = comparison.assign(
        return_difference=(
            comparison["monthly_return"] - comparison["daily_return"]
        )
    )
    disagreements = comparison.loc[
        ~comparison["level_difference"].abs().le(1e-10),
        [
            "_month",
            "monthly_date",
            "daily_date",
            "monthly_level",
            "daily_level",
            "level_difference",
            "monthly_return",
            "daily_return",
            "return_difference",
        ],
    ]
    return_disagreements = comparison.loc[
        comparison["return_difference"].abs().gt(1e-12),
        ["_month", "monthly_return", "daily_return", "return_difference"],
    ]
    print(f"{label} monthly-level versus last-daily-level check:")
    print(f"  matched months: {len(comparison):,}")
    print(f"  level disagreements: {len(disagreements):,}")
    print(f"  return disagreements: {len(return_disagreements):,}")
    if disagreements.empty and return_disagreements.empty:
        print("  none")
    else:
        if not disagreements.empty:
            print("  differing levels:")
            print(disagreements.to_string(index=False))
        if not return_disagreements.empty:
            print("  differing returns:")
            print(return_disagreements.to_string(index=False))
    return comparison


def returns_to_decimals(
    frame: pd.DataFrame,
    columns: list[str],
    source_name: str,
) -> pd.DataFrame:
    """Detect percent versus decimal returns and standardize to decimals.

    Args:
        frame: Frame used by this operation.
        columns: Columns used by this operation.
        source_name: Source name used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    work = frame.copy()
    work[columns] = work[columns].apply(pd.to_numeric, errors="coerce")
    maximum = float(work[columns].abs().max().max())
    if maximum <= 1.0:
        detected = "decimal"
        scale = 1.0
    elif maximum <= 100.0:
        detected = "percent"
        scale = 100.0
    else:
        raise RuntimeError(
            f"Could not infer return units for {source_name}: max abs={maximum:.4f}."
        )
    print(
        f"{source_name} return units detected as {detected} "
        f"(maximum absolute value={maximum:.4f}); output stored as decimals."
    )
    work.loc[:, columns] = work[columns] / scale
    return work


def prepare_output(
    raw_factors: pd.DataFrame,
    raw_benchmark: pd.DataFrame,
    raw_sp500_daily: pd.DataFrame,
    raw_russell_monthly: pd.DataFrame,
    raw_russell_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Validate, combine, and standardize the monthly return series.

    Args:
        raw_factors: Raw factors used by this operation.
        raw_benchmark: Raw benchmark used by this operation.
        raw_sp500_daily: Daily S&P index levels used for source validation.
        raw_russell_monthly: Monthly Russell levels used only for diagnostics.
        raw_russell_daily: Daily Russell levels used for production returns.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    factor_columns = ["mkt_excess", "smb", "hml", "rmw", "cma", "umd", "rf"]
    factors = returns_to_decimals(raw_factors, factor_columns, "Fama-French")
    benchmark = returns_to_decimals(
        raw_benchmark,
        ["sp500_total_return"],
        "CRSP S&P 500",
    )
    factors = factors.assign(date=pd.to_datetime(factors["date"]))
    benchmark = benchmark.assign(date=pd.to_datetime(benchmark["date"]))
    compare_monthly_and_daily_levels(
        raw_benchmark[["date", "index_level"]],
        raw_sp500_daily,
        "S&P 500",
    )
    compare_monthly_and_daily_levels(
        raw_russell_monthly,
        raw_russell_daily,
        "Russell 3000",
    )
    benchmark = benchmark.loc[:, ["date", "sp500_total_return"]].copy()
    russell = _last_level_by_month(raw_russell_daily)
    russell = russell.assign(
        russell3000_total_return=russell["index_level"].pct_change(fill_method=None),
    )
    panel_start, panel_end = panel_period()
    russell = russell.loc[
        russell["date"].between(panel_start, panel_end),
        ["_month", "russell3000_total_return"],
    ]

    for name, frame in (("factors", factors), ("benchmark", benchmark)):
        duplicates = int(frame.duplicated("date").sum())
        if duplicates:
            raise RuntimeError(f"{name} has {duplicates:,} duplicate monthly dates.")

    output = factors.merge(benchmark, on="date", how="left", validate="1:1")
    missing_benchmark = int(output["sp500_total_return"].isna().sum())
    if missing_benchmark:
        raise RuntimeError(
            f"CRSP S&P 500 total return is missing for {missing_benchmark:,} months."
        )
    output = output.assign(_month=output["date"].dt.to_period("M")).merge(
        russell,
        on="_month",
        how="left",
        validate="1:1",
    ).drop(columns="_month")
    missing_russell = int(output["russell3000_total_return"].isna().sum())
    if missing_russell:
        raise RuntimeError(
            "Compustat Russell 3000 total return is missing for "
            f"{missing_russell:,} months."
        )
    expected = pd.period_range(
        output["date"].min(),
        output["date"].max(),
        freq="M",
    )
    observed = output["date"].dt.to_period("M")
    missing_months = expected.difference(observed)
    if len(missing_months):
        raise RuntimeError(f"Factors are missing months: {list(missing_months)}")
    return output.sort_values("date", ignore_index=True)


def write_factors(factors: pd.DataFrame) -> None:
    """Atomically write the monthly factor file.

    Args:
        factors: Monthly factor and benchmark-return rows.

    Returns:
        None.
    """
    FACTORS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FACTORS_OUTPUT_PATH.with_suffix(".tmp.parquet")
    factors.to_parquet(temporary, index=False)
    temporary.replace(FACTORS_OUTPUT_PATH)
    print(f"Wrote {FACTORS_OUTPUT_PATH}")


def print_summary(factors: pd.DataFrame) -> None:
    """Print the completed factor file's dimensions and coverage.

    Args:
        factors: Monthly factor and benchmark-return rows.

    Returns:
        None.
    """
    print("Factors summary:")
    print(f"  rows: {len(factors):,}")
    print(
        f"  date range: {factors['date'].min().date()} to "
        f"{factors['date'].max().date()}"
    )
    print(f"  columns: {', '.join(factors.columns)}")


def main(refresh_from_wrds: bool = False) -> pd.DataFrame:
    """Pull, validate, and write monthly factors and benchmark returns.

    Args:
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        None.
    """
    (
        raw_factors,
        raw_benchmark,
        raw_sp500_daily,
        raw_russell_monthly,
        raw_russell_daily,
    ) = pull_raw_returns(refresh_from_wrds)
    factors = prepare_output(
        raw_factors,
        raw_benchmark,
        raw_sp500_daily,
        raw_russell_monthly,
        raw_russell_daily,
    )
    write_factors(factors)
    print_summary(factors)
    return factors


if __name__ == "__main__":
    main()
