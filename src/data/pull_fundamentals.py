#!/usr/bin/env python3
"""Pull filtered annual/quarterly Compustat fundamentals and reference tables.

Reads ``comp.funda``, ``comp.fundq``, ``crsp.ccmxpf_linktable``, and
``comp.co_hgic`` through cached WRDS queries; writes ``data/fundamentals.parquet``,
``data/link.parquet``, and ``data/sectors.parquet``. The INDL/STD/D/C and USD
filters prevent duplicate presentations, later fiscal years resolve remaining
company-date duplicates, quarterly YTD flows are de-cumulated, and native
Compustat USD-millions units are intentionally preserved.
"""

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wrds

from config import (
    ACCEPTED_LINKPRIM_VALUES,
    ACCEPTED_LINK_TYPES,
    ANNUAL_COLUMNS,
    END_DATE,
    FUNDAMENTALS_CACHE_DIR,
    FUNDAMENTALS_OUTPUT_PATH,
    LINK_OUTPUT_PATH,
    QUARTERLY_COLUMNS,
    QUARTERLY_FLOW_COLUMNS,
    QUARTERLY_YTD_COLUMNS,
    SECTORS_OUTPUT_PATH,
    START_DATE,
    UNIT_CHECK_COMPANIES,
    WRDS_USERNAME,
)

# pandas 2.x can emit this warning inside the safe DataFrame.assign method.
warnings.filterwarnings("ignore", message=r"ChainedAssignmentError.*")


def _date_params() -> dict[str, Any]:
    """Return the configured WRDS date parameters."""
    params: dict[str, Any] = {"start_date": START_DATE}
    if END_DATE is not None:
        params["end_date"] = END_DATE
    return params


def _end_date_clause() -> str:
    """Build the optional configured SQL end-date predicate."""
    return "" if END_DATE is None else "AND datadate <= %(end_date)s::date"


def build_query(table: str, columns: tuple[str, ...], currency: str) -> str:
    """Build a filtered Compustat query for one reporting frequency.

    Args:
        table: Table used by this operation.
        columns: Columns used by this operation.
        currency: Currency used by this operation.

    Returns:
        Computed result as ``str``.
    """
    selected = ",\n            ".join(f'"{column}"' for column in columns)
    return f"""
        SELECT
            {selected}
        FROM comp.{table}
        WHERE datadate >= %(start_date)s::date
          {_end_date_clause()}
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
          AND "{currency}" = 'USD'
        ORDER BY gvkey, datadate
    """


def _cache_path(name: str, sql: str, params: dict[str, Any]) -> Path:
    """Return the content-addressed cache path for one WRDS query."""
    payload = json.dumps(
        {"name": name, "sql": sql, "params": params},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return FUNDAMENTALS_CACHE_DIR / f"{name}-{digest}.parquet"


def cached_query(
    db: wrds.Connection | None,
    name: str,
    sql: str,
    params: dict[str, Any],
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Load a raw WRDS result from cache or query and cache it.

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

    FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Querying WRDS: {name}")
    frame = db.raw_sql(sql, params=params)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"Cached {len(frame):,} raw rows: {path}")
    return frame


def resolve_duplicates(
    frame: pd.DataFrame,
    freq: str,
    fiscal_year: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Resolve duplicate company-dates with deterministic filing priorities.

    Args:
        frame: Frame used by this operation.
        freq: Freq used by this operation.
        fiscal_year: Fiscal year used by this operation.

    Returns:
        Computed result as ``tuple[pd.DataFrame, dict[str, int]]``.
    """
    gvkey = frame["gvkey"].astype("string").reset_index(drop=True)
    datadate = pd.to_datetime(frame["datadate"]).reset_index(drop=True)
    work = frame.drop(columns=["gvkey", "datadate"]).reset_index(drop=True)
    work.insert(0, "datadate", datadate)
    work.insert(0, "gvkey", gvkey)
    keys = ["gvkey", "datadate"]
    duplicate_mask = work.duplicated(keys, keep=False)
    groups_before = int(work.loc[duplicate_mask, keys].drop_duplicates().shape[0])
    rows_before = int(duplicate_mask.sum())
    excess_before = int(len(work) - work[keys].drop_duplicates().shape[0])

    if freq == "A":
        secondary = ["pdate", "fdate", "apdedate", "upd"]
    else:
        secondary = ["fqtr", "pdateq", "fdateq", "apdedateq", "updq"]

    # Fiscal-year-end changes can duplicate gvkey/datadate; keep the later
    # fiscal year, then the latest filing metadata as a deterministic tie-break.
    work.insert(len(work.columns), "_source_order", np.arange(len(work)))
    work = (
        work.sort_values(
            keys + [fiscal_year] + secondary + ["_source_order"],
            na_position="first",
        )
        .drop_duplicates(keys, keep="last")
        .drop(columns="_source_order")
        .sort_values(keys, ignore_index=True)
    )
    excess_after = int(len(work) - work[keys].drop_duplicates().shape[0])
    return work, {
        "duplicate_groups_before": groups_before,
        "duplicate_rows_before": rows_before,
        "excess_rows_before": excess_before,
        "excess_rows_after": excess_after,
    }


def add_decumulated_quarterly_cashflows(frame: pd.DataFrame) -> pd.DataFrame:
    """Add standalone-quarter flows while retaining Compustat's raw YTD data.

    Args:
        frame: Frame used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    work = frame.sort_values(
        ["gvkey", "fyearq", "fqtr", "datadate"],
        na_position="first",
    ).copy()
    fiscal_quarter = pd.to_numeric(work["fqtr"], errors="coerce")
    groups = [work["gvkey"], work["fyearq"]]
    previous_quarter = fiscal_quarter.groupby(groups, dropna=False).shift(1)
    is_first_quarter = fiscal_quarter.eq(1)
    is_consecutive = previous_quarter.eq(fiscal_quarter - 1)

    derived: dict[str, pd.Series] = {}
    for source, output in zip(QUARTERLY_YTD_COLUMNS, QUARTERLY_FLOW_COLUMNS):
        numeric = pd.to_numeric(work[source], errors="coerce")
        current = pd.Series(
            numeric.to_numpy(dtype="float64", na_value=np.nan),
            index=work.index,
        )
        previous = current.groupby(groups, dropna=False).shift(1)
        standalone = current.where(is_first_quarter, current - previous)
        derived[output] = standalone.where(is_first_quarter | is_consecutive)

    work = pd.concat([work, pd.DataFrame(derived, index=work.index)], axis=1)
    return work.sort_values(["gvkey", "datadate"], ignore_index=True)


def coverage_by_decade(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Calculate non-null coverage percentages by decade.

    Args:
        frame: Frame used by this operation.
        fields: Fields used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    decade = (frame["datadate"].dt.year // 10 * 10).astype("Int64")
    available = [field for field in fields if field in frame.columns]
    coverage = frame[available].notna().groupby(decade).mean().mul(100.0)
    coverage.index = coverage.index.map(lambda value: f"{value}s")
    return coverage


def print_unit_checks(annual: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    """Print empirical Compustat unit checks for known companies.

    Args:
        annual: Annual Compustat rows.
        quarterly: Quarterly Compustat rows.

    Returns:
        None.
    """
    print("Unit checks using the latest complete observation for known companies:")
    configurations = [
        ("A", annual, "at", "sale", "ni", "csho", "prcc_f", "mkvalt"),
        ("Q", quarterly, "atq", "saleq", "niq", "cshoq", "prccq", "mkvaltq"),
    ]
    for freq, frame, assets, sales, income, shares, price, market_value in configurations:
        candidates = frame.loc[
            frame["gvkey"].isin(UNIT_CHECK_COMPANIES),
            ["gvkey", "datadate", assets, sales, income, shares, price, market_value],
        ].dropna(subset=[assets, sales, income, shares, price, market_value])
        sample = (
            candidates.sort_values("datadate")
            .groupby("gvkey")
            .tail(1)
            .copy(deep=True)
            .reset_index(drop=True)
        )
        sample.insert(
            len(sample.columns),
            "company",
            sample["gvkey"].map(UNIT_CHECK_COMPANIES),
        )
        sample.insert(
            len(sample.columns),
            "implied_mkvalt",
            sample[shares] * sample[price],
        )
        sample.insert(
            len(sample.columns),
            "implied/reported",
            sample["implied_mkvalt"] / sample[market_value]
        )
        display = sample[
            [
                "company",
                "datadate",
                assets,
                sales,
                income,
                shares,
                price,
                market_value,
                "implied/reported",
            ]
        ]
        print(f"  {freq} observations:")
        if display.empty:
            print("    no fully populated validation rows")
        else:
            print(
                display.to_string(
                    index=False,
                    float_format=lambda value: f"{value:,.3f}",
                )
            )
    print(
        "  convention: monetary fundamentals and market value are USD millions; "
        "share counts are millions of shares; prices/EPS/DPS are USD per share."
    )
    print("  no unit conversion was applied to the output.")


def print_summary(
    panel: pd.DataFrame,
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    duplicate_stats: dict[str, dict[str, int]],
) -> None:
    """Print dimensions, duplicate statistics, and field coverage.

    Args:
        panel: Monthly point-in-time research panel.
        annual: Annual Compustat rows.
        quarterly: Quarterly Compustat rows.
        duplicate_stats: Duplicate stats used by this operation.

    Returns:
        None.
    """
    print("Output summary:")
    for freq in ("A", "Q"):
        subset = panel.loc[panel["freq"].eq(freq)]
        print(f"  {freq} rows: {len(subset):,}")
        print(
            f"  {freq} date range: {subset['datadate'].min().date()} to "
            f"{subset['datadate'].max().date()}"
        )
        print(f"  {freq} unique gvkeys: {subset['gvkey'].nunique():,}")
        stats = duplicate_stats[freq]
        print(
            f"  {freq} duplicates before: {stats['duplicate_groups_before']:,} groups, "
            f"{stats['duplicate_rows_before']:,} rows "
            f"({stats['excess_rows_before']:,} excess)"
        )
        print(f"  {freq} duplicate excess rows after: {stats['excess_rows_after']:,}")
    print(f"  combined rows: {len(panel):,}")
    print(f"  combined unique gvkeys: {panel['gvkey'].nunique():,}")

    print("Quarterly rdq coverage by decade:")
    rdq_coverage = coverage_by_decade(quarterly, ["rdq"])
    print(rdq_coverage.to_string(float_format=lambda value: f"{value:.1f}%"))

    print("Annual main-field coverage by decade:")
    annual_coverage = coverage_by_decade(
        annual,
        ["at", "lt", "sale", "oibdp", "ni", "oancf", "capx", "csho", "prcc_f", "mkvalt"],
    )
    print(annual_coverage.to_string(float_format=lambda value: f"{value:.1f}%"))

    print("Quarterly main-field coverage by decade:")
    quarterly_coverage = coverage_by_decade(
        quarterly,
        [
            "atq",
            "ltq",
            "saleq",
            "oibdpq",
            "niq",
            "oancf_qtr",
            "capx_qtr",
            "cshoq",
            "prccq",
            "mkvaltq",
        ],
    )
    print(quarterly_coverage.to_string(float_format=lambda value: f"{value:.1f}%"))


def pull_raw_fundamentals(
    refresh_from_wrds: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load annual and quarterly Compustat source rows from cache or WRDS.

    Args:
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame]``.
    """
    params = _date_params()
    annual_sql = build_query("funda", ANNUAL_COLUMNS, "curcd")
    quarterly_sql = build_query("fundq", QUARTERLY_COLUMNS, "curcdq")
    db = (
        wrds.Connection(wrds_username=WRDS_USERNAME)
        if refresh_from_wrds
        else None
    )
    try:
        annual_raw = cached_query(
            db,
            "funda_selected",
            annual_sql,
            params,
            refresh_from_wrds,
        )
        quarterly_raw = cached_query(
            db,
            "fundq_selected",
            quarterly_sql,
            params,
            refresh_from_wrds,
        )
    finally:
        if db is not None:
            db.close()
    return annual_raw, quarterly_raw


def prepare_frequencies(
    annual_raw: pd.DataFrame,
    quarterly_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    """Resolve duplicates and prepare annual and quarterly observations.

    Args:
        annual_raw: Annual raw used by this operation.
        quarterly_raw: Quarterly raw used by this operation.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]``.
    """
    annual, annual_stats = resolve_duplicates(annual_raw, "A", "fyear")
    quarterly, quarterly_stats = resolve_duplicates(quarterly_raw, "Q", "fyearq")
    quarterly = add_decumulated_quarterly_cashflows(quarterly)
    annual = annual.copy(deep=True)
    quarterly = quarterly.copy(deep=True)
    annual.insert(2, "freq", "A")
    quarterly.insert(2, "freq", "Q")
    return annual, quarterly, {"A": annual_stats, "Q": quarterly_stats}


def combine_frequencies(
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> pd.DataFrame:
    """Combine annual and quarterly observations into the output schema.

    Args:
        annual: Annual Compustat rows.
        quarterly: Quarterly Compustat rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    ordered_columns = ["gvkey", "datadate", "freq"]
    ordered_columns.extend(
        column
        for column in ANNUAL_COLUMNS + QUARTERLY_COLUMNS
        if column not in ordered_columns
    )
    ordered_columns.extend(QUARTERLY_FLOW_COLUMNS)
    panel = pd.concat([annual, quarterly], ignore_index=True, sort=False)
    panel = panel.loc[:, ordered_columns].sort_values(
        ["gvkey", "datadate", "freq"],
        ignore_index=True,
    )
    duplicate_output = int(panel.duplicated(["gvkey", "datadate", "freq"]).sum())
    if duplicate_output:
        raise RuntimeError(
            f"Output still contains {duplicate_output:,} duplicate "
            "(gvkey, datadate, freq) rows."
        )
    return panel


def write_fundamentals(panel: pd.DataFrame) -> None:
    """Atomically write the completed fundamentals panel.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        None.
    """
    FUNDAMENTALS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FUNDAMENTALS_OUTPUT_PATH.with_suffix(".tmp.parquet")
    panel.to_parquet(temporary, index=False)
    temporary.replace(FUNDAMENTALS_OUTPUT_PATH)
    print(f"Wrote {FUNDAMENTALS_OUTPUT_PATH}")


def write_reference_table(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one processed reference table.

    Args:
        frame: Frame used by this operation.
        path: Input or output filesystem path.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"Wrote {len(frame):,} rows: {path}")


def pull_reference_tables(refresh_from_wrds: bool = False) -> None:
    """Pull, filter, and save the CCM link and historical GICS tables.

    Args:
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        None.
    """
    link_sql = """
        SELECT
            gvkey,
            linkprim,
            liid,
            linktype,
            lpermno,
            lpermco,
            usedflag,
            linkdt,
            linkenddt
        FROM crsp.ccmxpf_linktable
        ORDER BY linktype, linkprim, gvkey, linkdt
    """
    sector_sql = """
        SELECT
            gvkey,
            indtype,
            ggroup,
            gind,
            gsector,
            gsubind,
            indfrom,
            indthru
        FROM comp.co_hgic
        ORDER BY gvkey, indfrom, indthru
    """
    db = (
        wrds.Connection(wrds_username=WRDS_USERNAME)
        if refresh_from_wrds
        else None
    )
    try:
        raw_links = cached_query(
            db,
            "ccmxpf_linktable_raw",
            link_sql,
            {},
            refresh_from_wrds,
        )
        sectors = cached_query(
            db,
            "co_hgic_raw",
            sector_sql,
            {},
            refresh_from_wrds,
        )
    finally:
        if db is not None:
            db.close()

    link_counts = (
        raw_links.groupby(["linktype", "linkprim"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    print("CCM linktype/linkprim values before filtering:")
    print(link_counts.to_string(index=False))

    # LC/LU/LS are usable CCM links. Retain P/C primary markers plus J so the
    # panel can identify, link, and visibly flag non-primary share classes.
    usable = (
        raw_links["linktype"].isin(ACCEPTED_LINK_TYPES)
        & raw_links["linkprim"].isin(ACCEPTED_LINKPRIM_VALUES)
        & raw_links["gvkey"].notna()
        & raw_links["lpermno"].notna()
    )
    links = raw_links.loc[usable].assign(
        gvkey=lambda frame: frame["gvkey"].astype("string"),
        lpermno=lambda frame: pd.to_numeric(
            frame["lpermno"], errors="raise"
        ).astype("int32"),
        lpermco=lambda frame: pd.to_numeric(
            frame["lpermco"], errors="coerce"
        ).astype("Int32"),
        linkdt=lambda frame: pd.to_datetime(frame["linkdt"]),
        linkenddt=lambda frame: pd.to_datetime(frame["linkenddt"]),
    )

    sectors = sectors.assign(
        gvkey=lambda frame: frame["gvkey"].astype("string"),
        indfrom=lambda frame: pd.to_datetime(frame["indfrom"]),
        indthru=lambda frame: pd.to_datetime(frame["indthru"]),
    )
    write_reference_table(links, LINK_OUTPUT_PATH)
    write_reference_table(sectors, SECTORS_OUTPUT_PATH)


def main(refresh_from_wrds: bool = False) -> pd.DataFrame:
    """Pull fundamentals and reference tables, then prepare and write outputs.

    Args:
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        None.
    """
    annual_raw, quarterly_raw = pull_raw_fundamentals(refresh_from_wrds)
    annual, quarterly, duplicate_stats = prepare_frequencies(
        annual_raw,
        quarterly_raw,
    )
    panel = combine_frequencies(annual, quarterly)
    write_fundamentals(panel)
    print_unit_checks(annual, quarterly)
    print_summary(panel, annual, quarterly, duplicate_stats)
    pull_reference_tables(refresh_from_wrds)
    return panel


if __name__ == "__main__":
    main()
