#!/usr/bin/env python3
"""Pull survivorship-aware monthly CRSP prices and Compustat short interest.

Reads CIZ ``crsp.msf_v2``, CRSP index membership/delist tables, and
``comp.sec_shortint`` through cached WRDS queries; writes ``data/prices.parquet``
and ``data/short_interest.parquet``. The pull always keeps the full US common-
stock panel, date-ranges membership, and never overlays ordinary delisting
returns because CIZ already includes them; only ``mthdelflg='M'`` is compounded
with the configured terminal-loss assumption and explicitly flagged.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import wrds

from config import (
    DELISTING_ASSUMPTION,
    END_DATE,
    PRICE_CACHE_DIR,
    PRICES_OUTPUT_PATH,
    RETURN_TOLERANCE,
    SHORT_INTEREST_OUTPUT_PATH,
    START_DATE,
    UNIVERSE,
    WRDS_USERNAME,
)

# pandas 2.x can emit this warning inside safe DataFrame.assign operations.
warnings.filterwarnings("ignore", message=r"ChainedAssignmentError.*")


def _cache_path(name: str, sql: str, params: dict[str, Any]) -> Path:
    """Return a content-addressed path for one raw WRDS result."""
    payload = json.dumps(
        {"name": name, "sql": sql, "params": params},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return PRICE_CACHE_DIR / f"{name}-{digest}.parquet"


def cached_query(
    db: wrds.Connection | None,
    name: str,
    sql: str,
    params: dict[str, Any],
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Run a read-only WRDS query once, caching its unprocessed result.

    Args:
        db: Open WRDS connection.
        name: Stable cache or series name.
        sql: Parameterized SQL query text.
        params: Values bound to the SQL query.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name, sql, params)
    if path.exists() and not refresh_from_wrds:
        print(f"Cache hit: {path}")
        return pd.read_parquet(path)
    if db is None:
        raise RuntimeError(
            f"Cache miss: {path}. Rerun with refresh_from_wrds=True."
        )

    print(f"Querying WRDS: {name}")
    frame = db.raw_sql(sql, params=params)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"Cached {len(frame):,} raw rows: {path}")
    return frame


def _end_date_clause(column: str) -> str:
    """Build the optional upper-bound predicate without hiding configuration."""
    return "" if END_DATE is None else f"AND {column} <= %(end_date)s::date"


def _date_params(start: str = START_DATE, end: str | None = END_DATE) -> dict[str, Any]:
    """Return the configured WRDS date parameters."""
    params: dict[str, Any] = {"start_date": start}
    if end is not None:
        params["end_date"] = end
    return params


def pull_common_stock_panel(
    db: wrds.Connection | None,
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Pull the full CIZ common-stock panel regardless of the selected universe.

    Args:
        db: Open WRDS connection.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    sql = f"""
        SELECT
            permno,
            mthcaldt AS date,
            mthret::double precision AS ret,
            mthprc::double precision AS mthprc,
            shrout::double precision AS shrout,
            mthvol::double precision AS mthvol,
            mthcap::double precision AS mthcap,
            siccd,
            ticker,
            issuernm AS company_name,
            mthdelflg,
            mthretflg
        FROM crsp.msf_v2
        WHERE mthcaldt >= %(start_date)s::date
          {_end_date_clause('mthcaldt')}
          AND securitytype = 'EQTY'
          AND securitysubtype = 'COM'
          AND sharetype = 'NS'
          AND primaryexch IN ('N', 'A', 'Q')
          AND conditionaltype = 'RW'
          AND tradingstatusflg = 'A'
        ORDER BY permno, mthcaldt
    """
    return cached_query(
        db,
        "msf_v2_common_stock",
        sql,
        _date_params(),
        refresh_from_wrds,
    )


def pull_share_adjustment_factors(
    db: wrds.Connection | None,
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Pull the CIZ cumulative share factor needed for split-adjusted shares.

    Args:
        db: Open WRDS connection.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    sql = f"""
        SELECT
            permno,
            mthcaldt AS date,
            mthcumfacshr::double precision AS mthcumfacshr
        FROM crsp.msf_v2
        WHERE mthcaldt >= %(start_date)s::date
          {_end_date_clause('mthcaldt')}
          AND securitytype = 'EQTY'
          AND securitysubtype = 'COM'
          AND sharetype = 'NS'
          AND primaryexch IN ('N', 'A', 'Q')
          AND conditionaltype = 'RW'
          AND tradingstatusflg = 'A'
        ORDER BY permno, mthcaldt
    """
    return cached_query(
        db,
        "msf_v2_share_adjustment_factors",
        sql,
        _date_params(),
        refresh_from_wrds,
    )


def pull_short_interest(
    db: wrds.Connection | None,
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Pull Compustat security-level short interest settlement reports.

    Args:
        db: Open WRDS connection.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    sql = f"""
        SELECT
            gvkey,
            iid,
            shortint::double precision AS shortint,
            shortintadj::double precision AS shortintadj,
            datadate,
            splitadjdate
        FROM comp.sec_shortint
        WHERE datadate >= %(start_date)s::date
          {_end_date_clause('datadate')}
        ORDER BY gvkey, iid, datadate
    """
    return cached_query(
        db,
        "comp_sec_shortint",
        sql,
        _date_params(),
        refresh_from_wrds,
    )


def prepare_short_interest(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate source units, cadence, identifiers, and split adjustment.

    Args:
        raw: Raw signal component values.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    expected = [
        "gvkey",
        "iid",
        "shortint",
        "shortintadj",
        "datadate",
        "splitadjdate",
    ]
    missing = [column for column in expected if column not in raw.columns]
    if missing:
        raise KeyError(f"comp.sec_shortint is missing expected fields: {missing}")
    short_interest = raw.loc[:, expected].copy().assign(
        gvkey=lambda frame: frame["gvkey"].astype("string").str.strip(),
        iid=lambda frame: frame["iid"].astype("string").str.strip(),
        shortint=lambda frame: pd.to_numeric(frame["shortint"], errors="coerce"),
        shortintadj=lambda frame: pd.to_numeric(
            frame["shortintadj"], errors="coerce"
        ),
        datadate=lambda frame: pd.to_datetime(frame["datadate"]),
        splitadjdate=lambda frame: pd.to_datetime(frame["splitadjdate"]),
    )
    duplicates = int(short_interest.duplicated(["gvkey", "iid", "datadate"]).sum())
    if duplicates:
        raise RuntimeError(
            "comp.sec_shortint has "
            f"{duplicates:,} duplicate (gvkey, iid, datadate) rows."
        )
    if short_interest[["gvkey", "iid", "datadate"]].isna().any().any():
        raise RuntimeError("comp.sec_shortint contains null security keys or dates.")

    cadence = (
        short_interest.assign(month=short_interest["datadate"].dt.to_period("M"))
        .groupby(["gvkey", "iid", "month"])
        .size()
        .value_counts()
        .sort_index()
        .rename_axis("reports_per_security_month")
        .rename("security_months")
    )
    adjustment_ratio = (
        short_interest["shortintadj"] / short_interest["shortint"]
    ).replace([np.inf, -np.inf], np.nan)
    adjusted_rows = adjustment_ratio.notna() & ~np.isclose(adjustment_ratio, 1.0)
    print("comp.sec_shortint schema and conventions:")
    print("  gvkey string — Global Company Key")
    print("  iid string — Global Issue Key; identifier is (gvkey, iid)")
    print("  shortint float — unadjusted shares held short, individual shares")
    print("  shortintadj float — split-adjusted shares held short, individual shares")
    print("  datadate date — short-interest settlement date")
    print("  splitadjdate date — month-end split-adjustment date")
    print(
        f"  date range: {short_interest['datadate'].min().date()} to "
        f"{short_interest['datadate'].max().date()}"
    )
    print("  report cadence by security-month:")
    print(cadence.to_string())
    print(
        "  rows where shortintadj differs from shortint: "
        f"{int(adjusted_rows.sum()):,}/{len(short_interest):,}; "
        "shortintadj is used because raw shortint is not split-adjusted."
    )
    return short_interest.sort_values(["gvkey", "iid", "datadate"], ignore_index=True)


def write_short_interest(short_interest: pd.DataFrame) -> None:
    """Atomically save the security-level short-interest history.

    Args:
        short_interest: Short interest used by this operation.

    Returns:
        None.
    """
    SHORT_INTEREST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SHORT_INTEREST_OUTPUT_PATH.with_suffix(".tmp.parquet")
    short_interest.to_parquet(temporary, index=False)
    temporary.replace(SHORT_INTEREST_OUTPUT_PATH)
    print(
        f"Wrote {SHORT_INTEREST_OUTPUT_PATH}: {len(short_interest):,} rows, "
        f"{short_interest[['gvkey', 'iid']].drop_duplicates().shape[0]:,} securities."
    )


def diagnose_delisting_returns(
    db: wrds.Connection | None,
    refresh_from_wrds: bool = False,
) -> bool:
    """Determine empirically whether CIZ mthret already embeds delisting returns.

    Args:
        db: Open WRDS connection.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        Computed result as ``bool``.
    """
    sql = f"""
        SELECT
            c.permno,
            c.mthcaldt AS date,
            c.mthret::double precision AS ciz_ret,
            c.mthdelflg,
            c.mthretflg,
            d.delistingdt,
            d.delret::double precision AS delret,
            l.ret::double precision AS legacy_ret,
            ((1 + l.ret::double precision) *
             (1 + d.delret::double precision) - 1) AS compounded_ret
        FROM crsp.stkdelists d
        JOIN crsp.msf_v2 c
          ON c.permno = d.permno
         AND DATE_TRUNC('month', c.mthcaldt) =
             DATE_TRUNC('month', d.delistingdt)
        JOIN crsp.msf l
          ON l.permno = c.permno
         AND DATE_TRUNC('month', l.date) =
             DATE_TRUNC('month', c.mthcaldt)
        WHERE d.delistingdt >= %(start_date)s::date
          {_end_date_clause('d.delistingdt')}
          AND d.delret IS NOT NULL
          AND c.mthret IS NOT NULL
          AND l.ret IS NOT NULL
        ORDER BY d.delistingdt, c.permno
    """
    evidence = cached_query(
        db,
        "delisting_return_check",
        sql,
        _date_params(),
        refresh_from_wrds,
    )
    if evidence.empty:
        raise RuntimeError(
            "No matched delisting observations were available; refusing to "
            "guess whether CIZ mthret includes delisting returns."
        )

    error_compounded = (evidence["ciz_ret"] - evidence["compounded_ret"]).abs()
    error_legacy = (evidence["ciz_ret"] - evidence["legacy_ret"]).abs()
    compounded_matches = int((error_compounded < RETURN_TOLERANCE).sum())
    legacy_matches = int((error_legacy < RETURN_TOLERANCE).sum())
    match_rate = compounded_matches / len(evidence)
    compounded_mae = float(error_compounded.mean())
    legacy_mae = float(error_legacy.mean())
    flagged = int(evidence["mthdelflg"].notna().sum())

    print("Delisting-return diagnostic:")
    print(f"  matched delisting months: {len(evidence):,}")
    print(
        "  CIZ mthret matches compounded legacy_ret + delret: "
        f"{compounded_matches:,} ({match_rate:.1%})"
    )
    print(f"  CIZ mthret matches legacy_ret alone: {legacy_matches:,}")
    print(f"  mean absolute error vs compounded: {compounded_mae:.8f}")
    print(f"  mean absolute error vs legacy only: {legacy_mae:.8f}")
    print(f"  rows with mthdelflg populated: {flagged:,}")

    included = (
        match_rate >= 0.75
        and compounded_mae < 0.001
        and compounded_mae < legacy_mae * 0.10
    )
    excluded = (
        legacy_matches / len(evidence) >= 0.75
        and legacy_mae < 0.001
        and legacy_mae < compounded_mae * 0.10
    )
    if included:
        print("  conclusion: CIZ mthret already includes delisting returns.")
        return True
    if excluded:
        print("  conclusion: CIZ mthret excludes delisting returns.")
        return False
    raise RuntimeError(
        "Delisting-return evidence is inconclusive; refusing to choose a "
        "return convention automatically."
    )


def merge_delisting_returns(
    db: wrds.Connection | None,
    prices: pd.DataFrame,
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Fallback used only if the empirical check says mthret excludes delret.

    Args:
        db: Open WRDS connection.
        prices: Monthly security-price rows.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    sql = f"""
        SELECT permno, delistingdt,
               delret::double precision AS delret
        FROM crsp.stkdelists
        WHERE delistingdt >= %(start_date)s::date
          {_end_date_clause('delistingdt')}
          AND delret IS NOT NULL
        ORDER BY permno, delistingdt
    """
    delists = cached_query(
        db,
        "stkdelists_returns",
        sql,
        _date_params(),
        refresh_from_wrds,
    )
    if delists.empty:
        return prices

    work = prices.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.to_period("M")
    delists["month"] = pd.to_datetime(delists["delistingdt"]).dt.to_period("M")

    # Multiple event returns in one security-month, if present, compound.
    monthly_delists = (
        delists.groupby(["permno", "month"], as_index=False)["delret"]
        .agg(lambda values: np.prod(1.0 + values.dropna()) - 1.0)
    )
    work = work.merge(monthly_delists, on=["permno", "month"], how="left", validate="m:1")
    has_delret = work["delret"].notna()
    has_both = has_delret & work["ret"].notna()
    work.loc[has_both, "ret"] = (
        (1.0 + work.loc[has_both, "ret"])
        * (1.0 + work.loc[has_both, "delret"])
        - 1.0
    )
    work.loc[has_delret & work["ret"].isna(), "ret"] = work.loc[
        has_delret & work["ret"].isna(), "delret"
    ]
    return work.drop(columns=["month", "delret"])


def select_universe(
    universe: str,
    prices: pd.DataFrame,
    db: wrds.Connection | None,
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Return one date-aware universe flag for every price-panel row.

    Args:
        universe: Configured universe name.
        prices: Monthly security-price rows.
        db: Open WRDS connection.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    keys = prices[["permno", "date"]].copy()

    if universe == "all":
        keys["in_universe"] = True
        return keys

    if universe == "russell3000":
        # The Russell proxy depends on June cross-sectional ranks and CCM
        # primary-security information, so build_panel.py owns this flag.
        keys["in_universe"] = False
        print("Russell 3000 membership is constructed locally in build_panel.py.")
        return keys

    if universe == "sp500":
        membership_sql = """
            SELECT permno, mbrstartdt, mbrenddt
            FROM crsp.dsp500list_v2
            WHERE mbrenddt >= %(panel_start)s::date
              AND mbrstartdt <= %(panel_end)s::date
            ORDER BY permno, mbrstartdt, mbrenddt
        """
        membership_params = {
            "panel_start": pd.Timestamp(keys["date"].min()).date().isoformat(),
            "panel_end": pd.Timestamp(keys["date"].max()).date().isoformat(),
        }
        membership = cached_query(
            db,
            "dsp500list_v2_membership",
            membership_sql,
            membership_params,
            refresh_from_wrds,
        )
        membership = membership.assign(
            mbrstartdt=pd.to_datetime(membership["mbrstartdt"]),
            mbrenddt=pd.to_datetime(membership["mbrenddt"]),
        )
        reversed_intervals = membership["mbrstartdt"] > membership["mbrenddt"]
        if reversed_intervals.any():
            raise RuntimeError(
                "dsp500list_v2 contains membership intervals with start after "
                "end; inspect the raw cached membership before proceeding."
            )

        # Attach an internal row id, interval-join only candidate PERMNOs, then
        # flag rows whose actual CRSP month-end lies inside any membership span.
        candidates = keys.reset_index(names="_row_id").merge(
            membership,
            on="permno",
            how="inner",
            validate="m:m",
        )
        in_span = candidates["date"].between(
            candidates["mbrstartdt"],
            candidates["mbrenddt"],
            inclusive="both",
        )
        member_row_ids = candidates.loc[in_span, "_row_id"].unique()
        keys["in_universe"] = False
        keys.loc[member_row_ids, "in_universe"] = True
        return keys

    raise ValueError(
        f"Unknown UNIVERSE={universe!r}; choose 'sp500', 'russell3000', or 'all'."
    )


def prepare_output(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Rename CIZ fields, standardize units, and enforce the requested grain.

    Args:
        raw_prices: Raw prices used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    # Empirical cross-source checks show CIZ shrout and mthcap are scaled in
    # thousands; expose intuitive units (shares and currency units) downstream.
    prices = raw_prices.assign(
        date=pd.to_datetime(raw_prices["date"]),
        price=raw_prices["mthprc"].abs(),
        shares_out=raw_prices["shrout"] * 1_000.0,
        volume=raw_prices["mthvol"],
        market_cap=raw_prices["mthcap"] * 1_000.0,
        delret_imputed=raw_prices["mthdelflg"].eq("M"),
    )

    # mthdelflg='M' means the terminal return was excluded, so impute that outcome.
    imputed = prices["delret_imputed"]
    prices.loc[imputed, "ret"] = (
        (1.0 + prices.loc[imputed, "ret"])
        * (1.0 + DELISTING_ASSUMPTION)
        - 1.0
    )
    print(f"Rows with an imputed delisting return: {int(imputed.sum()):,}")

    output_columns = [
        "permno",
        "date",
        "ret",
        "delret_imputed",
        "price",
        "shares_out",
        "mthcumfacshr",
        "volume",
        "market_cap",
        "siccd",
        "ticker",
        "company_name",
    ]
    prices = prices.loc[:, output_columns].astype(
        {"permno": "int32", "siccd": "Int32"}
    )

    duplicate_count = int(prices.duplicated(["permno", "date"]).sum())
    if duplicate_count:
        raise RuntimeError(
            f"CIZ common-stock pull has {duplicate_count:,} duplicate "
            "(permno, date) rows; refusing to write an ambiguous panel."
        )
    return prices.sort_values(["date", "permno"], ignore_index=True)


def print_summary(prices: pd.DataFrame) -> None:
    """Print dimensions and universe coverage for the completed price panel.

    Args:
        prices: Monthly security-price rows.

    Returns:
        None.
    """
    average_per_month = prices.groupby("date")["permno"].nunique().mean()
    flagged_rows = int(prices["in_universe"].sum())
    flagged_permnos = prices.loc[prices["in_universe"], "permno"].nunique()
    print("Output summary:")
    print(f"  rows: {len(prices):,}")
    print(f"  date range: {prices['date'].min().date()} to {prices['date'].max().date()}")
    print(f"  unique permnos: {prices['permno'].nunique():,}")
    print(f"  average stocks per month: {average_per_month:,.1f}")
    print(f"  rows flagged in_universe: {flagged_rows:,}")
    print(f"  unique permnos ever flagged in_universe: {flagged_permnos:,}")


def main(refresh_from_wrds: bool = False) -> pd.DataFrame:
    """Pull, validate, transform, flag, and write the monthly price panel.

    Args:
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        None.
    """
    db = (
        wrds.Connection(wrds_username=WRDS_USERNAME)
        if refresh_from_wrds
        else None
    )
    try:
        short_interest = prepare_short_interest(
            pull_short_interest(db, refresh_from_wrds)
        )
        write_short_interest(short_interest)
        raw_prices = pull_common_stock_panel(db, refresh_from_wrds)
        share_factors = pull_share_adjustment_factors(db, refresh_from_wrds)
        raw_prices = raw_prices.merge(
            share_factors,
            on=["permno", "date"],
            how="left",
            validate="one_to_one",
        )
        missing_factors = int(raw_prices["mthcumfacshr"].isna().sum())
        print(f"Rows without mthcumfacshr: {missing_factors:,}")
        delisting_is_included = diagnose_delisting_returns(db, refresh_from_wrds)
        if not delisting_is_included:
            raw_prices = merge_delisting_returns(
                db,
                raw_prices,
                refresh_from_wrds,
            )

        prices = prepare_output(raw_prices)
        universe_flags = select_universe(
            UNIVERSE,
            prices,
            db,
            refresh_from_wrds,
        )
        prices = prices.merge(
            universe_flags,
            on=["permno", "date"],
            how="left",
            validate="1:1",
        )
        if prices["in_universe"].isna().any():
            raise RuntimeError("Universe function did not return one flag for every panel row.")
        prices = prices.assign(in_universe=prices["in_universe"].astype(bool))

        PRICES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(PRICES_OUTPUT_PATH, index=False)
        print(f"Wrote {PRICES_OUTPUT_PATH}")
        print_summary(prices)
        return prices
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    main()
