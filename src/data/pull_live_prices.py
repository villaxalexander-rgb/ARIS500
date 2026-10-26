#!/usr/bin/env python3
"""Build a provisional monthly price extension from Compustat Security Daily.

Reads cached/current ``comp.secd`` data and ``data/link.parquet`` and writes
``data/live_prices.parquet``. Total-return wealth is ``prccd / ajexdi * trfd``;
links use both gvkey and iid to avoid share-class confusion, and the +/-50%
quarantine applies only after the configured liquidity screen so genuine
microcap moves remain in the research history. A security whose last price is
more than the configured number of observed trading sessions before the market's
monthly close is retained as an auditable terminal row but excluded downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import wrds

from config import (
    ACCEPTED_LINKPRIM_VALUES,
    ACCEPTED_LINK_TYPES,
    LINK_OUTPUT_PATH,
    LIVE_PRICE_CACHE_DIR,
    LIVE_PRICE_MAX_GAP_DAYS,
    LIVE_PRICES_END_DATE,
    LIVE_PRICES_OUTPUT_PATH,
    LIVE_PRICES_START_DATE,
    LIVE_RETURN_QUARANTINE,
    MIN_ADV_USD,
    MIN_MARKET_CAP,
    MIN_PRICE,
    PRICES_OUTPUT_PATH,
    WRDS_USERNAME,
)

# pandas 2.x can emit this warning inside safe DataFrame.assign operations.
warnings.filterwarnings("ignore", message=r"ChainedAssignmentError.*")


DAILY_COLUMNS = (
    "gvkey",
    "iid",
    "datadate",
    "tic",
    "conm",
    "prccd",
    "ajexdi",
    "trfd",
    "cshoc",
    "cshtrd",
    "div",
    "divd",
    "divsp",
    "divrc",
    "capgn",
    "cheqv",
)
ACTION_LABELS = {
    "div": "dividend",
    "divd": "cash_dividend",
    "divsp": "special_dividend",
    "divrc": "return_of_capital",
    "capgn": "capital_gain",
    "cheqv": "cash_equivalent",
}


def _cache_path(name: str, sql: str, params: dict[str, Any]) -> Path:
    """Return a content-addressed path for one raw live-price query."""
    payload = json.dumps(
        {"name": name, "sql": sql, "params": params},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return LIVE_PRICE_CACHE_DIR / f"{name}-{digest}.parquet"


def cached_query(
    db: wrds.Connection | None,
    name: str,
    sql: str,
    params: dict[str, Any],
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Run a read-only WRDS query and cache its unprocessed result.

    Args:
        db: Open WRDS connection.
        name: Stable cache or series name.
        sql: Parameterized SQL query text.
        params: Values bound to the SQL query.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    LIVE_PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name, sql, params)
    if path.exists() and not refresh_from_wrds:
        print(f"Cache hit: {path}")
        return pd.read_parquet(path)
    if db is None:
        raise RuntimeError(
            f"Cache miss: {path}. Rerun with --refresh-from-wrds."
        )

    print(f"Querying WRDS: {name}")
    frame = db.raw_sql(sql, params=params)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"Cached {len(frame):,} raw rows: {path}")
    return frame


def pull_daily_prices(
    db: wrds.Connection | None,
    refresh_from_wrds: bool = False,
) -> pd.DataFrame:
    """Pull date-valid CCM-linked Compustat daily security observations.

    Args:
        db: Open WRDS connection.
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    linktypes = ", ".join(f"'{value}'" for value in ACCEPTED_LINK_TYPES)
    linkprims = ", ".join(f"'{value}'" for value in ACCEPTED_LINKPRIM_VALUES)
    end_predicate = (
        "s.datadate <= CURRENT_DATE"
        if LIVE_PRICES_END_DATE is None
        else "s.datadate <= %(end_date)s::date"
    )
    # EXISTS keeps the download to securities that can map through a date-valid
    # (gvkey, iid) CCM link without allowing a gvkey-only share-class match.
    sql = f"""
        SELECT
            s.gvkey,
            s.iid,
            s.datadate,
            s.tic,
            s.conm,
            s.prccd::double precision AS prccd,
            s.ajexdi::double precision AS ajexdi,
            s.trfd::double precision AS trfd,
            s.cshoc::double precision AS cshoc,
            s.cshtrd::double precision AS cshtrd,
            s.div::double precision AS div,
            s.divd::double precision AS divd,
            s.divsp::double precision AS divsp,
            s.divrc::double precision AS divrc,
            s.capgn::double precision AS capgn,
            s.cheqv::double precision AS cheqv
        FROM comp.secd s
        WHERE s.datadate >= %(start_date)s::date
          AND {end_predicate}
          AND EXISTS (
              SELECT 1
              FROM crsp.ccmxpf_linktable l
              WHERE l.gvkey = s.gvkey
                AND l.liid = s.iid
                AND l.lpermno IS NOT NULL
                AND l.linktype IN ({linktypes})
                AND l.linkprim IN ({linkprims})
                AND (l.linkdt IS NULL OR s.datadate >= l.linkdt)
                AND (
                    l.linkenddt IS NULL
                    OR s.datadate <=
                       (DATE_TRUNC('month', l.linkenddt)
                        + INTERVAL '1 month - 1 day')::date
                )
          )
        ORDER BY s.gvkey, s.iid, s.datadate
    """
    params: dict[str, Any] = {"start_date": LIVE_PRICES_START_DATE}
    if LIVE_PRICES_END_DATE is not None:
        params["end_date"] = LIVE_PRICES_END_DATE
    return cached_query(
        db,
        "comp_secd_live",
        sql,
        params,
        refresh_from_wrds,
    )


def prepare_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate keys and normalize Compustat daily field types.

    Args:
        raw: Raw signal component values.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    missing = [column for column in DAILY_COLUMNS if column not in raw]
    if missing:
        raise KeyError(f"comp.secd is missing expected fields: {missing}")
    daily = raw.loc[:, DAILY_COLUMNS].copy().assign(
        gvkey=lambda frame: frame["gvkey"].astype("string").str.strip(),
        iid=lambda frame: frame["iid"].astype("string").str.strip(),
        datadate=lambda frame: pd.to_datetime(frame["datadate"]),
        tic=lambda frame: frame["tic"].astype("string").str.strip(),
        conm=lambda frame: frame["conm"].astype("string").str.strip(),
    )
    numeric = [column for column in DAILY_COLUMNS if column not in {
        "gvkey", "iid", "datadate", "tic", "conm"
    }]
    daily[numeric] = daily[numeric].apply(pd.to_numeric, errors="coerce")
    null_keys = daily[["gvkey", "iid", "datadate"]].isna().any(axis=1)
    duplicates = daily.duplicated(["gvkey", "iid", "datadate"])
    if null_keys.any() or duplicates.any():
        raise RuntimeError(
            "comp.secd live pull has "
            f"{int(null_keys.sum()):,} null-key rows and "
            f"{int(duplicates.sum()):,} duplicate security-date rows."
        )
    return daily.sort_values(["gvkey", "iid", "datadate"], ignore_index=True)


def build_monthly_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Build monthly returns and flag observations stale versus the market close.

    Args:
        daily: Daily used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    work = daily.copy()
    work["cshoc"] = work.groupby(["gvkey", "iid"], sort=False)["cshoc"].ffill()
    missing_trfd = int(work["trfd"].isna().sum())
    # A missing Compustat trfd means no accumulated distribution adjustment;
    # its neutral wealth multiplier is one, not a missing price observation.
    work["trfd"] = work["trfd"].fillna(1.0)
    print(f"Daily rows with null trfd treated as 1.0: {missing_trfd:,}")
    valid_price = (
        work["prccd"].notna()
        & work["ajexdi"].gt(0)
        & work["trfd"].gt(0)
    )
    work = work.loc[valid_price].copy().assign(
        month=lambda frame: frame["datadate"].dt.to_period("M"),
        # Compustat's convention is adjusted price = prccd / ajexdi; trfd
        # already reinvests distributions, so dividends must not be added again.
        adjusted_wealth=lambda frame: frame["prccd"] / frame["ajexdi"]
        * frame["trfd"],
    )
    if work.empty:
        raise RuntimeError("No valid priced comp.secd rows were returned.")

    volume = (
        work.groupby(["gvkey", "iid", "month"], observed=True)["cshtrd"]
        .sum(min_count=1)
        .rename("volume")
        .reset_index()
    )
    trading_calendar = (
        work[["month", "datadate"]]
        .drop_duplicates()
        .sort_values(["month", "datadate"], ignore_index=True)
        .assign(
            _trading_day_number=lambda frame: frame.groupby(
                "month", observed=True
            ).cumcount()
        )
    )
    month_closes = (
        trading_calendar.groupby("month", observed=True, as_index=False)
        .tail(1)
        .rename(
            columns={
                "datadate": "month_end_date",
                "_trading_day_number": "_month_end_trading_day_number",
            }
        )
    )
    month_end = (
        work.sort_values(["gvkey", "iid", "datadate"])
        .groupby(["gvkey", "iid", "month"], observed=True, as_index=False)
        .tail(1)
        .merge(volume, on=["gvkey", "iid", "month"], how="left", validate="1:1")
        .merge(
            trading_calendar,
            on=["month", "datadate"],
            how="left",
            validate="m:1",
        )
        .merge(month_closes, on="month", how="left", validate="m:1")
        .sort_values(["gvkey", "iid", "month"], ignore_index=True)
    )
    month_end = month_end.assign(
        last_priced_date=month_end["datadate"],
        terminal_gap_trading_days=(
            month_end["_month_end_trading_day_number"]
            - month_end["_trading_day_number"]
        ).astype("int16"),
    )
    # The calendar is made from dates with valid prices across the whole daily
    # universe, so this gap counts observed trading sessions rather than weekdays.
    month_end["terminal"] = month_end["terminal_gap_trading_days"].gt(
        LIVE_PRICE_MAX_GAP_DAYS
    )
    month_end = month_end.drop(
        columns=["_trading_day_number", "_month_end_trading_day_number"]
    )
    groups = month_end.groupby(["gvkey", "iid"], sort=False)
    prior_wealth = groups["adjusted_wealth"].shift(1)
    month_end["prior_ajexdi"] = groups["ajexdi"].shift(1)
    month_end["prior_trfd"] = groups["trfd"].shift(1)
    prior_month_number = (
        groups["month"].shift(1).astype("int64", errors="ignore")
    )
    month_number = month_end["month"].astype("int64")
    consecutive = month_number.sub(prior_month_number).eq(1)
    month_end["ret"] = (
        month_end["adjusted_wealth"].div(prior_wealth).sub(1.0).where(consecutive)
    )
    return month_end


def _format_number(value: float) -> str:
    """Format a corporate-action amount compactly for review output."""
    return f"{value:.8g}"


def describe_corporate_actions(
    daily: pd.DataFrame,
    quarantined: pd.DataFrame,
) -> pd.DataFrame:
    """Describe distribution and adjustment-factor changes in quarantined months.

    Args:
        daily: Daily used by this operation.
        quarantined: Quarantined used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if quarantined.empty:
        return quarantined.assign(corporate_actions=pd.Series(dtype="string"))
    work = daily.copy().assign(month=daily["datadate"].dt.to_period("M"))
    groups = {
        key: frame.sort_values("datadate")
        for key, frame in work.groupby(["gvkey", "iid", "month"], observed=True)
    }
    descriptions: list[str] = []
    for row in quarantined.itertuples(index=False):
        frame = groups[(row.gvkey, row.iid, row.month)]
        actions: list[str] = []
        for column, label in ACTION_LABELS.items():
            values = frame[column].dropna()
            values = values.loc[~np.isclose(values, 0.0)]
            if len(values):
                actions.append(f"{label}={_format_number(float(values.sum()))}")
        for column, label in (("ajexdi", "ajexdi"), ("trfd", "trfd")):
            values = frame[column].dropna()
            if len(values) and not np.isclose(values.min(), values.max()):
                actions.append(
                    f"{label} {_format_number(float(values.iloc[0]))}"
                    f"->{_format_number(float(values.iloc[-1]))}"
                )
        for column, prior_column, label in (
            ("ajexdi", "prior_ajexdi", "month_end_ajexdi"),
            ("trfd", "prior_trfd", "month_end_trfd"),
        ):
            current = getattr(row, column)
            prior = getattr(row, prior_column)
            if pd.notna(prior) and pd.notna(current) and not np.isclose(prior, current):
                actions.append(
                    f"{label} {_format_number(float(prior))}"
                    f"->{_format_number(float(current))}"
                )
        descriptions.append("; ".join(actions) if actions else "none found in comp.secd")
    return quarantined.assign(corporate_actions=pd.array(descriptions, dtype="string"))


def attach_ccm_links(monthly: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Map security months to PERMNO through date-valid (gvkey, iid) CCM links.

    Args:
        monthly: Monthly strategy and benchmark return history.
        links: Date-ranged CCM link rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    usable = links.loc[
        links["linktype"].isin(ACCEPTED_LINK_TYPES)
        & links["linkprim"].isin(ACCEPTED_LINKPRIM_VALUES)
        & links["lpermno"].notna()
    ].copy().assign(
        gvkey=lambda frame: frame["gvkey"].astype("string").str.strip(),
        iid=lambda frame: frame["liid"].astype("string").str.strip(),
        linkdt=lambda frame: pd.to_datetime(frame["linkdt"]),
        linkenddt=lambda frame: pd.to_datetime(frame["linkenddt"])
        + pd.offsets.MonthEnd(0),
    )
    candidates = monthly.reset_index(names="_live_row").merge(
        usable,
        on=["gvkey", "iid"],
        how="left",
        validate="m:m",
    )
    valid = (
        candidates["lpermno"].notna()
        & (candidates["linkdt"].isna() | candidates["datadate"].ge(candidates["linkdt"]))
        & (
            candidates["linkenddt"].isna()
            | candidates["datadate"].le(candidates["linkenddt"])
        )
    )
    valid_links = candidates.loc[valid].copy().assign(
        _linktype_rank=lambda frame: frame["linktype"].map(
            {value: rank for rank, value in enumerate(ACCEPTED_LINK_TYPES)}
        ),
        _linkprim_rank=lambda frame: frame["linkprim"].map(
            {value: rank for rank, value in enumerate(ACCEPTED_LINKPRIM_VALUES)}
        ),
    )
    selected = (
        valid_links.sort_values(
            ["_live_row", "_linktype_rank", "_linkprim_rank", "linkdt"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates("_live_row", keep="first")
        [["_live_row", "lpermno", "linktype", "linkprim"]]
        .rename(columns={"lpermno": "permno"})
    )
    linked = monthly.reset_index(names="_live_row").merge(
        selected,
        on="_live_row",
        how="left",
        validate="1:1",
    )
    print(
        "CCM issue-key mapping: "
        f"{linked['permno'].notna().sum():,}/{len(linked):,} security-months linked; "
        f"{linked['permno'].isna().sum():,} unlinked."
    )
    return linked


def prepare_output(
    linked: pd.DataFrame,
    historical_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Match the CRSP price schema while preserving explicit live-data units.

    Args:
        linked: Linked used by this operation.
        historical_prices: Historical prices used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    history = historical_prices.assign(date=pd.to_datetime(historical_prices["date"]))
    reference_date = history["date"].max()
    # The extension can only identify securities known at the CRSP cutoff.
    # Restricting to that last active universe prevents stale CCM links from
    # resurrecting delisted issues or repurposed PERMNOs in later months.
    latest_reference = (
        history.loc[history["date"].eq(reference_date)]
        [["permno", "siccd", "ticker", "company_name", "in_universe"]]
        .rename(
            columns={
                "ticker": "_crsp_ticker",
                "company_name": "_crsp_company_name",
            }
        )
    )
    output = linked.loc[linked["permno"].notna()].copy().assign(
        permno=lambda frame: frame["permno"].astype("int32"),
        date=lambda frame: pd.to_datetime(frame["datadate"]),
        last_priced_date=lambda frame: pd.to_datetime(frame["last_priced_date"]),
        month_end_date=lambda frame: pd.to_datetime(frame["month_end_date"]),
        terminal_gap_trading_days=lambda frame: pd.to_numeric(
            frame["terminal_gap_trading_days"], errors="raise"
        ).astype("int16"),
        terminal=lambda frame: frame["terminal"].fillna(False).astype(bool),
        price=lambda frame: frame["prccd"].abs(),
        # comp.secd cshoc is individual shares. Store both shares and market
        # cap in millions, as required for direct use beside Compustat fields.
        shares_out=lambda frame: frame["cshoc"] / 1_000_000.0,
        mthcumfacshr=lambda frame: frame["ajexdi"],
        market_cap=lambda frame: frame["prccd"].abs()
        * frame["cshoc"]
        / 1_000_000.0,
        delret_imputed=False,
        provisional=True,
    )
    output = output.merge(latest_reference, on="permno", how="inner", validate="m:1")
    output = output.assign(
        ticker=output["tic"].fillna(output["_crsp_ticker"]),
        company_name=output["conm"].fillna(output["_crsp_company_name"]),
    )
    output["_month"] = output["date"].dt.to_period("M")
    duplicate_count = int(output.duplicated(["permno", "_month"]).sum())
    if duplicate_count:
        print(
            f"Resolving {duplicate_count:,} duplicate PERMNO-month issue mappings "
            "by primary link, latest priced date, then largest market cap."
        )
        primary_rank = output["linkprim"].map({"P": 0, "C": 0, "J": 1}).fillna(2)
        output = (
            output.assign(_primary_rank=primary_rank)
            .sort_values(
                ["permno", "_month", "_primary_rank", "date", "market_cap"],
                ascending=[True, True, True, False, False],
            )
            .drop_duplicates(["permno", "_month"], keep="first")
        )
    columns = [
        "permno",
        "date",
        "last_priced_date",
        "month_end_date",
        "terminal_gap_trading_days",
        "terminal",
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
        "in_universe",
        "provisional",
    ]
    output = output.loc[:, columns].sort_values(["date", "permno"], ignore_index=True)
    if output.duplicated(["permno", "date"]).any():
        raise RuntimeError("Live output still has duplicate (permno, date) rows.")
    return output


def add_live_adv(
    live: pd.DataFrame,
    historical_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Attach standard three-month ADV using CRSP history to seed live months.

    Args:
        live: Prepared provisional live-price rows.
        historical_prices: Historical prices used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    live_start_month = live["date"].min().to_period("M")
    history = historical_prices.loc[
        pd.to_datetime(historical_prices["date"])
        .dt.to_period("M")
        .lt(live_start_month),
        ["permno", "date", "price", "volume"],
    ].copy().assign(_live_row=-1)
    extension = live[["permno", "date", "price", "volume"]].copy().assign(
        _live_row=np.arange(len(live), dtype=np.int64)
    )
    work = pd.concat([history, extension], ignore_index=True).assign(
        date=lambda frame: pd.to_datetime(frame["date"]),
        price=lambda frame: pd.to_numeric(frame["price"], errors="coerce"),
        volume=lambda frame: pd.to_numeric(frame["volume"], errors="coerce"),
    )
    work["month"] = work["date"].dt.to_period("M")
    work = (
        work.sort_values(["permno", "month", "date"])
        .drop_duplicates(["permno", "month"], keep="last")
        .sort_values(["permno", "month"], ignore_index=True)
    )
    month_start = work["month"].dt.to_timestamp()
    next_month = month_start + pd.offsets.MonthBegin(1)
    work["_trading_days"] = np.busday_count(
        month_start.to_numpy(dtype="datetime64[D]"),
        next_month.to_numpy(dtype="datetime64[D]"),
    ).astype(float)
    work["_monthly_dollar_volume"] = work["price"] * work["volume"]
    groups = work.groupby("permno", sort=False)
    rolling_dollars = (
        groups["_monthly_dollar_volume"]
        .rolling(3, min_periods=3)
        .sum()
        .reset_index(level=0, drop=True)
    )
    rolling_days = (
        groups["_trading_days"]
        .rolling(3, min_periods=3)
        .sum()
        .reset_index(level=0, drop=True)
    )
    first_month = groups["month"].shift(2)
    consecutive = work["month"].astype("int64").sub(
        first_month.astype("int64", errors="ignore")
    ).eq(2)
    work["adv_usd_3m"] = (rolling_dollars / rolling_days).where(consecutive)
    live_adv = work.loc[work["_live_row"].ge(0), ["_live_row", "adv_usd_3m"]]
    if live_adv["_live_row"].duplicated().any():
        raise RuntimeError("ADV construction returned duplicate live rows.")
    output = live.copy().merge(
        live_adv,
        left_index=True,
        right_on="_live_row",
        how="left",
        validate="1:1",
    )
    return output.drop(columns="_live_row").sort_values(
        ["date", "permno"], ignore_index=True
    )


def apply_liquidity_quarantine(
    live: pd.DataFrame,
    historical_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Flag extreme returns only after applying the configured liquidity screen.

    Args:
        live: Prepared provisional live-price rows.
        historical_prices: Historical prices used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    work = add_live_adv(live, historical_prices)
    liquid = (
        work["price"].ge(MIN_PRICE)
        & work["market_cap"].ge(MIN_MARKET_CAP)
        & work["adv_usd_3m"].ge(MIN_ADV_USD)
    )
    work["quarantined"] = (
        liquid & work["ret"].abs().gt(LIVE_RETURN_QUARANTINE)
    ).fillna(False).astype(bool)
    return work.drop(columns="adv_usd_3m")


def print_terminal_exclusions(live: pd.DataFrame) -> None:
    """Print every security-month excluded for an early final price.

    Args:
        live: Prepared provisional live-price rows with terminal diagnostics.

    Returns:
        None.
    """
    terminal = live.loc[live["terminal"].fillna(False)].copy()
    if terminal.empty:
        print(
            "Live terminal-price exclusions: none "
            f"(gap must exceed {LIVE_PRICE_MAX_GAP_DAYS} trading days)."
        )
        return
    terminal["month"] = terminal["month_end_date"].dt.to_period("M")
    counts = terminal.groupby("month", observed=True).size().rename("excluded")
    print(
        "Live terminal-price exclusions by month "
        f"(gap > {LIVE_PRICE_MAX_GAP_DAYS} trading days):"
    )
    print(counts.to_string())
    print("Excluded security-months retained in live_prices.parquet for audit:")
    report = terminal[
        [
            "permno",
            "ticker",
            "company_name",
            "last_priced_date",
            "month_end_date",
            "terminal_gap_trading_days",
        ]
    ].sort_values(["month_end_date", "permno"])
    print(report.to_string(index=False))


def print_quarantine(live: pd.DataFrame) -> None:
    """Print every liquid extreme return and its adjustment-factor status.

    Args:
        live: Prepared provisional live-price rows.

    Returns:
        None.
    """
    ordered = live.sort_values(["permno", "date"]).copy()
    groups = ordered.groupby("permno", sort=False)
    prior_factor = groups["mthcumfacshr"].shift(1)
    prior_date = groups["date"].shift(1)
    consecutive = (
        ordered["date"].dt.to_period("M").astype("int64")
        .sub(prior_date.dt.to_period("M").astype("int64", errors="ignore"))
        .eq(1)
    )
    ordered["ajexdi_changed"] = (
        consecutive
        & prior_factor.notna()
        & ordered["mthcumfacshr"].notna()
        & ~np.isclose(
            prior_factor.to_numpy(dtype=float, na_value=np.nan),
            ordered["mthcumfacshr"].to_numpy(dtype=float, na_value=np.nan),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )
    )
    quarantined = ordered.loc[ordered["quarantined"]].copy()
    print(
        "Liquid monthly returns quarantined beyond "
        f"+/-{LIVE_RETURN_QUARANTINE:.0%}: {len(quarantined):,}"
    )
    if quarantined.empty:
        return
    report = quarantined[
        [
            "permno",
            "ticker",
            "date",
            "ret",
            "market_cap",
            "ajexdi_changed",
        ]
    ].sort_values(["date", "permno"])
    print(
        report.to_string(
            index=False,
            formatters={
                "ret": "{:.2%}".format,
                "market_cap": "{:,.2f}".format,
            },
        )
    )


def print_summary(
    live: pd.DataFrame,
    historical_prices: pd.DataFrame,
) -> None:
    """Print live month counts, quarantine totals, and latest-CRSP coverage.

    Args:
        live: Prepared provisional live-price rows.
        historical_prices: Historical prices used by this operation.

    Returns:
        None.
    """
    if live.empty:
        raise RuntimeError("No linked live price rows remain after quarantine.")
    terminal = live["terminal"].fillna(False)
    usable = live.loc[~terminal].copy()
    months = usable["date"].dt.to_period("M")
    reference_date = pd.to_datetime(historical_prices["date"]).max()
    reference_permnos = set(
        historical_prices.loc[
            pd.to_datetime(historical_prices["date"]).eq(reference_date), "permno"
        ].astype(int)
    )
    coverage = (
        usable.assign(
            month=months,
            _covered=usable["permno"].isin(reference_permnos),
        )
        .groupby("month")
        .agg(
            names=("permno", "nunique"),
            recent_crsp_names_covered=("_covered", "sum"),
        )
    )
    # The output is one PERMNO per month, so summing the coverage flag counts names.
    coverage["recent_crsp_universe"] = len(reference_permnos)
    coverage["coverage"] = (
        coverage["recent_crsp_names_covered"] / len(reference_permnos)
    )
    print("Live price summary:")
    print(f"  months covered: {months.min()} to {months.max()} ({months.nunique()} months)")
    print(
        f"  usable rows: {len(usable):,}; terminal audit rows: {int(terminal.sum()):,}; "
        f"unique usable permnos: {usable['permno'].nunique():,}"
    )
    print(f"  rows flagged quarantined: {int(live['quarantined'].sum()):,}")
    print("  names and coverage versus the most recent CRSP common-stock universe:")
    print(
        coverage.reset_index().to_string(
            index=False,
            formatters={"coverage": "{:.1%}".format},
        )
    )
    print(
        "  units: price=USD/share; shares_out=million shares; "
        "market_cap=USD millions; volume=individual shares."
    )


def main(refresh_from_wrds: bool = False) -> pd.DataFrame:
    """Pull, transform, quarantine, link, validate, and write live monthly prices.

    Args:
        refresh_from_wrds: Whether to bypass the local cache and query WRDS.

    Returns:
        None.
    """
    required = [LINK_OUTPUT_PATH, PRICES_OUTPUT_PATH]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing local inputs: {missing}")
    links = pd.read_parquet(LINK_OUTPUT_PATH)
    historical_prices = pd.read_parquet(PRICES_OUTPUT_PATH)
    db = (
        wrds.Connection(wrds_username=WRDS_USERNAME)
        if refresh_from_wrds
        else None
    )
    try:
        daily = prepare_daily(pull_daily_prices(db, refresh_from_wrds))
        monthly = build_monthly_prices(daily)
        linked = attach_ccm_links(monthly, links)
        live = prepare_output(linked, historical_prices)
        live = apply_liquidity_quarantine(live, historical_prices)
        print_terminal_exclusions(live)
        print_quarantine(live)
        output_columns = [
            "permno",
            "date",
            "last_priced_date",
            "month_end_date",
            "terminal_gap_trading_days",
            "terminal",
            "ret",
            "quarantined",
            "delret_imputed",
            "price",
            "shares_out",
            "mthcumfacshr",
            "volume",
            "market_cap",
            "siccd",
            "ticker",
            "company_name",
            "in_universe",
            "provisional",
        ]
        live = live.loc[:, output_columns]
        LIVE_PRICES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = LIVE_PRICES_OUTPUT_PATH.with_suffix(".tmp.parquet")
        live.to_parquet(temporary, index=False)
        temporary.replace(LIVE_PRICES_OUTPUT_PATH)
        print(f"Wrote {LIVE_PRICES_OUTPUT_PATH}")
        print_summary(live, historical_prices)
        return live
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-from-wrds",
        action="store_true",
        help="re-query WRDS instead of reading the content-addressed cache",
    )
    args = parser.parse_args()
    main(refresh_from_wrds=args.refresh_from_wrds)
