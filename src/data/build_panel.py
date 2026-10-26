#!/usr/bin/env python3
"""Merge local prices, links, sectors, and fundamentals into research panels.

Reads ``data/prices.parquet``, optional ``data/live_prices.parquet``,
``data/fundamentals.parquet``, ``data/link.parquet``, ``data/sectors.parquet``,
``data/short_interest.parquet``, ``data/factors.parquet``, and
``docs/data_dictionary.csv``; writes the configured
``data/panel_<universe>.parquet``. Range-valid links, actual filing dates,
fiscal staleness limits, one-security-per-company selection, dictionary coverage,
and unit conversion to USD millions are deliberate safeguards against
look-ahead, duplication, and unit errors; live rows remain visibly
provisional/quarantined.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from config import (
    ACCEPTED_LINKPRIM_VALUES,
    ACCEPTED_LINK_TYPES,
    ANNUAL_COLUMNS,
    ANNUAL_LAG_MONTHS,
    BENCHMARK_MAX_TRACKING_ERROR,
    BENCHMARK_MIN_CORRELATION,
    CASH_FLOW_ABS_TOLERANCE,
    CASH_FLOW_CHECK_SAMPLE_SIZE,
    CASH_FLOW_REL_TOLERANCE,
    DATA_DICTIONARY_PATH,
    FACTORS_OUTPUT_PATH,
    FUNDAMENTALS_OUTPUT_PATH,
    LINK_OUTPUT_PATH,
    LIVE_PRICES_OUTPUT_PATH,
    LIVE_PRICE_MAX_STALENESS_DAYS,
    MAX_STALENESS_MONTHS,
    MIN_ADV_USD,
    MIN_MARKET_CAP,
    MIN_PRICE,
    PANEL_OUTPUT_PATH,
    PRIMARY_LINKPRIM_VALUES,
    PRICES_OUTPUT_PATH,
    QUARTERLY_COLUMNS,
    QUARTERLY_FLOW_COLUMNS,
    QUARTERLY_MAX_LAG_MONTHS,
    RUSSELL_N,
    RUSSELL_RECON_MONTH,
    SECTORS_OUTPUT_PATH,
    SHORT_INTEREST_OUTPUT_PATH,
    UNIVERSE,
    USE_LIVE_EXTENSION,
)

# pandas 2.x can emit this warning inside the safe DataFrame.assign method.
warnings.filterwarnings("ignore", message=r"ChainedAssignmentError.*")


def load_inputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the required local parquet inputs, including live prices when enabled.

    Args:
        None.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]``.
    """
    paths = (
        PRICES_OUTPUT_PATH,
        FUNDAMENTALS_OUTPUT_PATH,
        LINK_OUTPUT_PATH,
        SECTORS_OUTPUT_PATH,
        FACTORS_OUTPUT_PATH,
        SHORT_INTEREST_OUTPUT_PATH,
    )
    if USE_LIVE_EXTENSION:
        paths = paths + (LIVE_PRICES_OUTPUT_PATH,)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing local panel inputs: {missing}")
    loaded = tuple(pd.read_parquet(path) for path in paths)
    if USE_LIVE_EXTENSION:
        return loaded
    return loaded + (pd.DataFrame(),)


def check_live_price_staleness(live_prices: pd.DataFrame) -> None:
    """Refuse to build on a live-price extension that looks abandoned.

    A missing or old live_prices.parquet usually means pull_live_prices was
    skipped this cycle, not that trading stopped. Building on it anyway would
    silently extend the panel with stale prices and stale quarantine flags.

    Args:
        live_prices: Prepared provisional live-price rows.

    Returns:
        None.
    """
    if not USE_LIVE_EXTENSION:
        return
    if live_prices.empty:
        raise RuntimeError(
            "USE_LIVE_EXTENSION is True but live_prices.parquet is empty. Run "
            "`python3 -m src.data.pull_live_prices --refresh-from-wrds` "
            "before building the panel."
        )
    latest = pd.to_datetime(live_prices["date"]).max()
    staleness_days = (pd.Timestamp.now().normalize() - latest).days
    if staleness_days > LIVE_PRICE_MAX_STALENESS_DAYS:
        raise RuntimeError(
            "live_prices.parquet's latest date "
            f"{latest.date()} is {staleness_days} days old, beyond the "
            f"configured {LIVE_PRICE_MAX_STALENESS_DAYS}-day limit "
            "(LIVE_PRICE_MAX_STALENESS_DAYS in config.py). Refresh it with "
            "`python3 -m src.data.pull_live_prices --refresh-from-wrds` "
            "before rebuilding the panel."
        )
    print(
        f"Live-price staleness check: latest date {latest.date()}, "
        f"{staleness_days} day(s) old (limit {LIVE_PRICE_MAX_STALENESS_DAYS})."
    )


def append_live_prices(
    prices: pd.DataFrame,
    live_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Append post-cutoff live months after normalizing them to raw CRSP units.

    Args:
        prices: Monthly security-price rows.
        live_prices: Live prices used by this operation.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.Timestamp]``.
    """
    historical = prices.copy().assign(
        date=lambda frame: pd.to_datetime(frame["date"]),
        provisional=False,
        quarantined=False,
    )
    cutoff = historical["date"].max()
    if not USE_LIVE_EXTENSION:
        print(f"Live extension disabled; CRSP cutoff remains {cutoff.date()}.")
        return historical, cutoff

    required = set(historical.columns).union(
        {
            "provisional",
            "quarantined",
            "terminal",
            "last_priced_date",
            "month_end_date",
            "terminal_gap_trading_days",
        }
    )
    missing = sorted(required.difference(live_prices.columns))
    if missing:
        raise KeyError(f"live_prices.parquet is missing required fields: {missing}")
    live = live_prices.copy().assign(
        date=lambda frame: pd.to_datetime(frame["date"]),
        provisional=lambda frame: frame["provisional"].fillna(True).astype(bool),
        quarantined=lambda frame: frame["quarantined"].fillna(False).astype(bool),
        terminal=lambda frame: frame["terminal"].fillna(False).astype(bool),
    )
    overlap_rows = int(live["date"].le(cutoff).sum())
    live = live.loc[live["date"].gt(cutoff)].copy()
    terminal_rows = live.loc[live["terminal"]].copy()
    live = live.loc[~live["terminal"]].copy()
    live_month = live["date"].dt.to_period("M")
    common_month_end = live.groupby(live_month)["date"].transform("max")
    relabeled_dates = int(live["date"].ne(common_month_end).sum())
    # Nonterminal securities may still trail the common close by up to the
    # configured limit; align those accepted observations to the panel month-end.
    live["date"] = common_month_end
    # Historical prices store individual shares and raw USD market cap. Live
    # prices arrive in millions, so restore raw units before the shared panel
    # logic; standardize_units() converts both sources back to millions once.
    live = live.assign(
        shares_out=pd.to_numeric(live["shares_out"], errors="coerce") * 1_000_000.0,
        market_cap=pd.to_numeric(live["market_cap"], errors="coerce") * 1_000_000.0,
    )
    # Terminal audit rows remain in live_prices.parquet, but only nonterminal
    # observations enter the research panel. Drop audit-only fields before the
    # shared CRSP-schema append.
    live = live.loc[:, historical.columns]
    combined = pd.concat(
        [historical.loc[:, live.columns], live],
        ignore_index=True,
    ).sort_values(["date", "permno"], ignore_index=True)
    duplicates = int(combined.duplicated(["permno", "date"]).sum())
    if duplicates:
        raise RuntimeError(
            f"Live append created {duplicates:,} duplicate (permno, date) rows."
        )
    print(
        "Live extension append: "
        f"CRSP rows={len(historical):,} through {cutoff.date()}, "
        f"overlap rows ignored={overlap_rows:,}, "
        f"post-cutoff terminal rows excluded={len(terminal_rows):,}, "
        f"post-cutoff rows appended={len(live):,}, "
        f"rows relabeled to common month-end={relabeled_dates:,}, "
        f"combined rows={len(combined):,}."
    )
    verify_live_unit_join(combined, cutoff)
    return combined, cutoff


def verify_live_unit_join(prices: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """Verify price, shares, and market-cap continuity across the source join.

    Args:
        prices: Monthly security-price rows.
        cutoff: Cutoff used by this operation.

    Returns:
        None.
    """
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    historical = prices.loc[
        prices["date"].le(cutoff) & prices["ticker"].isin(tickers)
    ].sort_values("date").groupby("ticker", as_index=False).tail(1)
    live = prices.loc[
        prices["date"].gt(cutoff) & prices["ticker"].isin(tickers)
    ].sort_values("date").groupby("ticker", as_index=False).head(1)
    sample = pd.concat(
        [historical.assign(source="CRSP"), live.assign(source="live")],
        ignore_index=True,
    ).assign(
        shares_m=lambda frame: frame["shares_out"] / 1_000_000.0,
        market_cap_m=lambda frame: frame["market_cap"] / 1_000_000.0,
        cap_identity_ratio=lambda frame: frame["market_cap"]
        / (frame["price"] * frame["shares_out"]),
    )
    ratio = sample["cap_identity_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 6 or not ratio.between(0.999, 1.001).all():
        raise RuntimeError("Live price units failed the source-join sanity check.")
    print("Large-name unit check across the CRSP/live join:")
    print(
        sample[
            [
                "source",
                "date",
                "ticker",
                "price",
                "shares_m",
                "market_cap_m",
                "cap_identity_ratio",
            ]
        ].sort_values(["ticker", "date"]).to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )


def add_trailing_adv(prices: pd.DataFrame) -> pd.DataFrame:
    """Estimate three-month average daily dollar volume from monthly CRSP data.

    Args:
        prices: Monthly security-price rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    work = prices.copy().assign(
        _source_row=np.arange(len(prices), dtype=np.int64),
        date=pd.to_datetime(prices["date"]),
        price=pd.to_numeric(prices["price"], errors="coerce"),
        volume=pd.to_numeric(prices["volume"], errors="coerce"),
    ).sort_values(["permno", "date"], ignore_index=True)
    month_start = work["date"].dt.to_period("M").dt.to_timestamp()
    next_month = month_start + pd.offsets.MonthBegin(1)
    work = work.assign(
        _month_number=work["date"].dt.year * 12 + work["date"].dt.month,
        _monthly_dollar_volume=work["price"] * work["volume"],
        # CRSP supplies monthly share volume, so trading days are approximated
        # with weekdays when converting it to average daily dollar volume.
        _trading_days=np.busday_count(
            month_start.to_numpy(dtype="datetime64[D]"),
            next_month.to_numpy(dtype="datetime64[D]"),
        ).astype(float),
    )
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
    first_month = groups["_month_number"].shift(2)
    consecutive = work["_month_number"].sub(first_month).eq(2)
    work = work.assign(
        adv_usd_3m=(rolling_dollars / rolling_days).where(consecutive)
    )
    adv = work.set_index("_source_row")["adv_usd_3m"].reindex(
        np.arange(len(prices))
    )
    output = prices.copy().assign(adv_usd_3m=adv.to_numpy())
    print(
        "Three-month ADV coverage: "
        f"{output['adv_usd_3m'].notna().sum():,}/{len(output):,} rows "
        f"({output['adv_usd_3m'].notna().mean():.1%})."
    )
    return output


def assign_universe_membership(prices: pd.DataFrame) -> pd.DataFrame:
    """Assign the configured date-aware universe membership locally.

    Args:
        prices: Monthly security-price rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    work = prices.copy().assign(date=pd.to_datetime(prices["date"]))
    if UNIVERSE == "sp500":
        if "in_universe" not in work:
            raise KeyError("prices.parquet has no S&P 500 in_universe column")
        return work.assign(in_universe=work["in_universe"].astype(bool))
    if UNIVERSE == "all":
        return work.assign(in_universe=True)
    if UNIVERSE != "russell3000":
        raise ValueError(f"Unsupported UNIVERSE={UNIVERSE!r}.")

    june = work.loc[
        work["date"].dt.month.eq(RUSSELL_RECON_MONTH)
        & pd.to_numeric(work["market_cap"], errors="coerce").gt(0),
        ["permno", "date", "market_cap"],
    ].assign(reconstitution_year=lambda frame: frame["date"].dt.year)
    duplicates = int(june.duplicated(["permno", "reconstitution_year"]).sum())
    if duplicates:
        raise RuntimeError(
            f"Russell June ranking has {duplicates:,} duplicate security-years."
        )
    candidate_counts = june.groupby("reconstitution_year").size()
    undersized = candidate_counts[candidate_counts.lt(RUSSELL_N)]
    if len(undersized):
        raise RuntimeError(
            "Fewer than the configured Russell membership count in June: "
            f"{undersized.to_dict()}"
        )
    selected = (
        june.sort_values(
            ["reconstitution_year", "market_cap", "permno"],
            ascending=[True, False, True],
        )
        .groupby("reconstitution_year", sort=True)
        .head(RUSSELL_N)
        [["permno", "reconstitution_year"]]
        .assign(in_universe=True)
    )
    membership_keys = work[["permno", "date"]].assign(
        reconstitution_year=lambda frame: np.where(
            frame["date"].dt.month.gt(RUSSELL_RECON_MONTH),
            frame["date"].dt.year,
            frame["date"].dt.year - 1,
        ).astype("int16")
    )
    membership = membership_keys.merge(
        selected,
        on=["permno", "reconstitution_year"],
        how="left",
        validate="m:1",
    )["in_universe"].eq(True)
    work = work.assign(in_universe=membership.to_numpy())
    recon_summary = pd.DataFrame(
        {
            "reconstitution_year": candidate_counts.index,
            "eligible_common_stocks": candidate_counts.to_numpy(),
            "selected": RUSSELL_N,
        }
    )
    print(
        "Russell 3000 proxy reconstitutions "
        f"(June rank, effective July through following June):\n"
        f"{recon_summary.to_string(index=False)}"
    )
    selected_sets = {
        int(year): set(group["permno"].astype(int))
        for year, group in selected.groupby("reconstitution_year")
    }
    changes = []
    for year in sorted(selected_sets):
        prior_year = year - 1
        if prior_year not in selected_sets:
            continue
        current = selected_sets[year]
        prior = selected_sets[prior_year]
        changes.append(
            {
                "reconstitution_year": year,
                "retained": len(current & prior),
                "added": len(current - prior),
                "dropped": len(prior - current),
            }
        )
    if changes:
        change_table = pd.DataFrame(changes)
        print("Russell proxy membership changes versus the prior June:")
        print(change_table.to_string(index=False))
        latest_year = int(change_table["reconstitution_year"].max())
        labels = (
            work.loc[work["date"].dt.year.eq(latest_year)]
            .sort_values("date")
            .drop_duplicates("permno", keep="last")
            .set_index("permno")[["ticker", "company_name"]]
        )
        latest_added = sorted(
            selected_sets[latest_year] - selected_sets[latest_year - 1]
        )
        latest_dropped = sorted(
            selected_sets[latest_year - 1] - selected_sets[latest_year]
        )

        def membership_labels(permnos: list[int]) -> str:
            """Format a compact sample of membership changes."""
            values = []
            for permno in permnos[:20]:
                if permno in labels.index:
                    ticker = labels.at[permno, "ticker"]
                    values.append(f"{permno}:{ticker}")
                else:
                    values.append(str(permno))
            return ", ".join(values)

        print(
            f"  {latest_year} additions (first 20): "
            f"{membership_labels(latest_added)}"
        )
        print(
            f"  {latest_year} drops (first 20): "
            f"{membership_labels(latest_dropped)}"
        )
    return work


def keep_ever_members(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep every history row for securities ever included in the universe.

    Args:
        prices: Monthly security-price rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    if "in_universe" not in prices:
        raise KeyError("prices.parquet has no in_universe column")
    member_permnos = prices.loc[prices["in_universe"], "permno"].unique()
    kept = prices.loc[prices["permno"].isin(member_permnos)].copy()
    kept = kept.assign(
        date=pd.to_datetime(kept["date"]),
        _panel_row=np.arange(len(kept), dtype=np.int64),
    )
    print(
        f"Universe '{UNIVERSE}': retaining {len(kept):,} rows for "
        f"{len(member_permnos):,} PERMNOs ever in the universe."
    )
    return kept


def attach_links(prices: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Range-join date-valid CCM links without changing the price-row count.

    Args:
        prices: Monthly security-price rows.
        links: Date-ranged CCM link rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    links = links.rename(columns={"lpermno": "permno"}).assign(
        permno=lambda frame: pd.to_numeric(frame["permno"], errors="raise").astype(
            "int32"
        ),
        gvkey=lambda frame: frame["gvkey"].astype("string"),
        linkdt=lambda frame: pd.to_datetime(frame["linkdt"]),
        # A mid-month link end still covers CRSP's observation for that month.
        # Adding MonthEnd leaves open-ended (NaT) links open.
        linkenddt=lambda frame: pd.to_datetime(frame["linkenddt"])
        + pd.offsets.MonthEnd(0),
    )
    candidates = prices[["_panel_row", "permno", "date"]].merge(
        links,
        on="permno",
        how="left",
        validate="m:m",
    )
    valid = (
        candidates["gvkey"].notna()
        & (candidates["linkdt"].isna() | candidates["date"].ge(candidates["linkdt"]))
        & (
            candidates["linkenddt"].isna()
            | candidates["date"].le(candidates["linkenddt"])
        )
    )
    valid_links = candidates.loc[valid].copy()
    counts = valid_links.groupby("_panel_row").size()
    duplicate_ids = counts[counts.gt(1)].index
    print(
        "CCM range join before resolution: "
        f"{len(duplicate_ids):,} price rows have multiple valid links "
        f"({int((counts - 1).clip(lower=0).sum()):,} excess rows)."
    )
    if len(duplicate_ids):
        overlaps = valid_links.loc[valid_links["_panel_row"].isin(duplicate_ids)]
        conflicting = overlaps.groupby("_panel_row")["gvkey"].nunique().gt(1).sum()
        print(f"  overlapping rows with conflicting gvkeys: {conflicting:,}")
        print("  overlap linktype/linkprim counts:")
        print(
            overlaps.groupby(["linktype", "linkprim"])
            .size()
            .rename("rows")
            .to_string()
        )
        print("  first 20 overlapping candidates:")
        print(
            overlaps[
                [
                    "permno",
                    "date",
                    "gvkey",
                    "liid",
                    "linktype",
                    "linkprim",
                    "linkdt",
                    "linkenddt",
                ]
            ]
            .sort_values(["permno", "date", "linktype", "linkprim"])
            .head(20)
            .to_string(index=False)
        )

    # Resolve overlapping usable links deterministically: researched LC first,
    # then LU/LS, and primary P/C ahead of joiner J.
    valid_links = valid_links.assign(
        _linktype_rank=valid_links["linktype"].map(
            {value: rank for rank, value in enumerate(ACCEPTED_LINK_TYPES)}
        ),
        _linkprim_rank=valid_links["linkprim"].map(
            {value: rank for rank, value in enumerate(ACCEPTED_LINKPRIM_VALUES)}
        ),
    )
    selected = (
        valid_links.sort_values(
            ["_panel_row", "_linktype_rank", "_linkprim_rank", "linkdt"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates("_panel_row", keep="first")
        .drop(columns=["permno", "date", "_linktype_rank", "_linkprim_rank"])
    )
    linked = prices.merge(selected, on="_panel_row", how="left", validate="1:1")
    if len(linked) != len(prices):
        raise RuntimeError(
            f"CCM join changed row count from {len(prices):,} to {len(linked):,}."
        )
    print(
        f"CCM range join: {linked['gvkey'].notna().sum():,} linked rows, "
        f"{linked['gvkey'].isna().sum():,} unlinked rows; row count unchanged."
    )
    ls_gains = linked.loc[
        linked["gvkey"].notna() & linked["linktype"].eq("LS"),
        ["permno", "gvkey"],
    ].drop_duplicates()
    print("PERMNOs gaining a GVKEY through newly accepted LS links:")
    if ls_gains.empty:
        print("  none")
    else:
        print(ls_gains.sort_values(["permno", "gvkey"]).to_string(index=False))
    return linked


def add_investability_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep one liquid in-universe primary security per company and month.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    candidates = panel["in_universe"] & panel["gvkey"].notna()
    ranked = panel.loc[
        candidates,
        ["date", "gvkey", "linkprim", "market_cap", "permno"],
    ].assign(
        _linkprim_rank=lambda frame: (~frame["linkprim"].isin(
            PRIMARY_LINKPRIM_VALUES
        )).astype("int8"),
        _market_cap_rank=lambda frame: pd.to_numeric(
            frame["market_cap"], errors="coerce"
        ),
    )
    # CCM primary markers win first, then larger capitalization; PERMNO is a
    # deterministic final tie-break so every date/GVKEY has exactly one winner.
    winner_index = (
        ranked.sort_values(
            [
                "date",
                "gvkey",
                "_linkprim_rank",
                "_market_cap_rank",
                "permno",
            ],
            ascending=[True, True, True, False, True],
            na_position="last",
        )
        .groupby(["date", "gvkey"], sort=False)
        .head(1)
        .index
    )
    primary_security = pd.Series(False, index=panel.index, dtype=bool)
    primary_security.loc[winner_index] = True
    price_screen = pd.to_numeric(panel["price"], errors="coerce").fillna(
        -np.inf
    ).ge(MIN_PRICE)
    market_cap_screen = pd.to_numeric(
        panel["market_cap"], errors="coerce"
    ).fillna(-np.inf).ge(MIN_MARKET_CAP * 1_000_000.0)
    adv_screen = pd.to_numeric(
        panel["adv_usd_3m"], errors="coerce"
    ).fillna(-np.inf).ge(MIN_ADV_USD)
    quarantine_screen = ~panel["quarantined"].fillna(False).astype(bool)
    investable = (
        primary_security
        & price_screen
        & market_cap_screen
        & adv_screen
        & quarantine_screen
    )
    secondary = candidates & ~primary_security
    panel = panel.assign(
        primary_security=primary_security,
        secondary_share_class=secondary.astype(bool),
        price_screen_pass=price_screen.astype(bool),
        market_cap_screen_pass=market_cap_screen.astype(bool),
        adv_screen_pass=adv_screen.astype(bool),
        investable=investable,
    )

    member_secondary = panel["in_universe"] & panel["secondary_share_class"]
    monthly_names = panel.loc[panel["investable"]].groupby("date")["permno"].nunique()
    print(
        "Investability flags: "
        f"in-universe rows={int(panel['in_universe'].sum()):,}, "
        f"secondary in-universe rows={int(member_secondary.sum()):,}, "
        f"investable rows={int(panel['investable'].sum()):,}."
    )
    print(
        "Investable names per month: "
        f"mean={monthly_names.mean():.1f}, min={int(monthly_names.min()):,}, "
        f"median={monthly_names.median():.1f}, max={int(monthly_names.max()):,}."
    )
    print_liquidity_screen_counts(panel)
    return panel


def short_interest_coverage_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """Return latest-report coverage by year among investable security-months.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    investable = panel.loc[panel["investable"]].copy()
    if investable.empty:
        return pd.DataFrame(columns=["year", "rows", "covered", "coverage"])
    return (
        investable.assign(
            year=investable["date"].dt.year,
            _covered=investable["short_interest_date"].notna(),
        )
        .groupby("year", as_index=False)
        .agg(rows=("permno", "size"), covered=("_covered", "sum"))
        .assign(coverage=lambda frame: frame["covered"] / frame["rows"])
    )


def attach_short_interest(
    panel: pd.DataFrame,
    short_interest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest non-future security-level short-interest report.

    Args:
        panel: Monthly point-in-time research panel.
        short_interest: Short interest used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    required = [
        "gvkey",
        "iid",
        "shortint",
        "shortintadj",
        "datadate",
        "splitadjdate",
    ]
    missing = [column for column in required if column not in short_interest.columns]
    if missing:
        raise KeyError(f"short_interest.parquet is missing fields: {missing}")
    reports = short_interest.loc[:, required].copy().rename(
        columns={
            "iid": "liid",
            "shortint": "short_interest_shares",
            "shortintadj": "short_interest_adjusted_shares",
            "datadate": "short_interest_date",
            "splitadjdate": "short_interest_split_adjustment_date",
        }
    )
    reports = reports.assign(
        gvkey=reports["gvkey"].astype("string").str.strip(),
        liid=reports["liid"].astype("string").str.strip(),
        short_interest_date=pd.to_datetime(reports["short_interest_date"]),
        short_interest_split_adjustment_date=pd.to_datetime(
            reports["short_interest_split_adjustment_date"]
        ),
        short_interest_shares=pd.to_numeric(
            reports["short_interest_shares"], errors="coerce"
        ),
        short_interest_adjusted_shares=pd.to_numeric(
            reports["short_interest_adjusted_shares"], errors="coerce"
        ),
    )
    duplicate_reports = int(
        reports.duplicated(["gvkey", "liid", "short_interest_date"]).sum()
    )
    if duplicate_reports:
        raise RuntimeError(
            "Short-interest history has "
            f"{duplicate_reports:,} duplicate security-date rows."
        )

    eligible = panel.loc[
        panel["gvkey"].notna() & panel["liid"].notna(),
        ["_panel_row", "date", "gvkey", "liid"],
    ].copy()
    eligible = eligible.assign(
        gvkey=eligible["gvkey"].astype("string").str.strip(),
        liid=eligible["liid"].astype("string").str.strip(),
    )
    matched = pd.merge_asof(
        eligible.sort_values(["date", "gvkey", "liid"]),
        reports.sort_values(["short_interest_date", "gvkey", "liid"]),
        left_on="date",
        right_on="short_interest_date",
        by=["gvkey", "liid"],
        direction="backward",
        allow_exact_matches=True,
    )
    payload = [
        "_panel_row",
        "short_interest_date",
        "short_interest_shares",
        "short_interest_adjusted_shares",
        "short_interest_split_adjustment_date",
    ]
    output = panel.merge(
        matched.loc[:, payload],
        on="_panel_row",
        how="left",
        validate="1:1",
    )
    if len(output) != len(panel):
        raise RuntimeError(
            f"Short-interest join changed rows from {len(panel):,} to {len(output):,}."
        )
    future = output["short_interest_date"].gt(output["date"])
    if future.any():
        raise RuntimeError(
            f"Short-interest join used {int(future.sum()):,} future reports."
        )

    coverage = short_interest_coverage_by_year(output)
    print("Short-interest coverage among investable rows by year:")
    print(
        coverage.assign(
            coverage=coverage["coverage"].map(lambda value: f"{value:.1%}")
        ).to_string(index=False)
    )
    return output


def liquidity_screen_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the sequential monthly investability-screen waterfall.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    work = panel.assign(
        _member=panel["in_universe"].astype(bool),
        _primary=panel["primary_security"].astype(bool),
    )
    work = work.assign(
        _removed_price=work["_primary"] & ~work["price_screen_pass"],
        _after_price=work["_primary"] & work["price_screen_pass"],
    )
    work = work.assign(
        _removed_market_cap=(
            work["_after_price"] & ~work["market_cap_screen_pass"]
        ),
        _after_market_cap=(
            work["_after_price"] & work["market_cap_screen_pass"]
        ),
    )
    work = work.assign(
        _removed_adv=work["_after_market_cap"] & ~work["adv_screen_pass"],
        _after_adv=work["_after_market_cap"] & work["adv_screen_pass"],
    )
    work = work.assign(
        _removed_quarantined=(
            work["_after_adv"] & work["quarantined"].fillna(False)
        )
    )
    return work.groupby("date", as_index=False).agg(
        member_securities=("_member", "sum"),
        primary_company_securities=("_primary", "sum"),
        removed_price=("_removed_price", "sum"),
        removed_market_cap=("_removed_market_cap", "sum"),
        removed_adv=("_removed_adv", "sum"),
        removed_quarantined=("_removed_quarantined", "sum"),
        investable_names=("investable", "sum"),
    )


def print_liquidity_screen_counts(panel: pd.DataFrame) -> None:
    """Print how many primary names each liquidity screen removes by month.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        None.
    """
    counts = liquidity_screen_counts(panel)
    print("Monthly liquidity-screen waterfall (screens applied sequentially):")
    print(counts.to_string(index=False))


def prepare_fundamentals(
    fundamentals: pd.DataFrame,
    gvkeys: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create point-in-time availability dates for both frequencies.

    Args:
        fundamentals: Compustat fundamental rows.
        gvkeys: Gvkeys used by this operation.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.DataFrame]``.
    """
    relevant = set(gvkeys.dropna().astype("string"))
    quarterly_fields = list(dict.fromkeys(QUARTERLY_COLUMNS + QUARTERLY_FLOW_COLUMNS))
    annual_fields = list(dict.fromkeys(ANNUAL_COLUMNS))

    quarterly = fundamentals.loc[
        fundamentals["freq"].eq("Q") & fundamentals["gvkey"].isin(relevant),
        quarterly_fields,
    ].assign(
        gvkey=lambda frame: frame["gvkey"].astype("string"),
        datadate=lambda frame: pd.to_datetime(frame["datadate"]),
    )
    reported = pd.to_datetime(quarterly["rdq"], errors="coerce")
    latest = quarterly["datadate"] + pd.DateOffset(
        months=QUARTERLY_MAX_LAG_MONTHS
    )
    missing_rdq = int(reported.isna().sum())
    early_rdq = int((reported < quarterly["datadate"]).fillna(False).sum())
    late_rdq = int((reported > latest).fillna(False).sum())
    invalid_rdq = reported.isna() | reported.lt(quarterly["datadate"])
    # Missing or pre-period RDQs are corrupt, so use the configured fallback.
    # There is deliberately no upper bound: a genuinely late report must not
    # become available before its actual RDQ, which would create look-ahead.
    available = reported.where(~invalid_rdq, latest)
    # Keep the actual availability day because CRSP dates are trading month-ends;
    # calendar month-end rounding would unnecessarily delay weekend month-ends.
    quarterly = quarterly.assign(available_date=available)

    annual = fundamentals.loc[
        fundamentals["freq"].eq("A") & fundamentals["gvkey"].isin(relevant),
        annual_fields,
    ].assign(
        gvkey=lambda frame: frame["gvkey"].astype("string"),
        datadate=lambda frame: pd.to_datetime(frame["datadate"]),
    )
    annual = annual.assign(
        available_date=(
            annual["datadate"] + pd.DateOffset(months=ANNUAL_LAG_MONTHS)
        )
    )

    quarterly_collisions = int(
        quarterly.duplicated(["gvkey", "available_date"], keep=False).sum()
    )
    annual_collisions = int(
        annual.duplicated(["gvkey", "available_date"], keep=False).sum()
    )
    if quarterly_collisions or annual_collisions:
        print(
            "Fundamental availability-date collisions before resolution: "
            f"quarterly={quarterly_collisions:,}, annual={annual_collisions:,}. "
            "Keeping the latest fiscal period on that availability date."
        )
    quarterly = (
        quarterly.sort_values(["gvkey", "available_date", "datadate"])
        .drop_duplicates(["gvkey", "available_date"], keep="last")
        .reset_index(drop=True)
    )
    annual = (
        annual.sort_values(["gvkey", "available_date", "datadate"])
        .drop_duplicates(["gvkey", "available_date"], keep="last")
        .reset_index(drop=True)
    )
    annual = annual.rename(
        columns={column: f"{column}_a" for column in annual if column != "gvkey"}
    )
    print(
        "Quarterly availability inputs: "
        f"missing rdq={missing_rdq:,}, bounded early={early_rdq:,}, "
        f"late but retained at actual rdq={late_rdq:,}."
    )
    return quarterly, annual


def merge_asof_by_gvkey(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_on: str,
) -> pd.DataFrame:
    """Attach the latest right-hand observation available for each gvkey.

    Args:
        left: Left used by this operation.
        right: Right used by this operation.
        right_on: Right on used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    left_work = left.assign(
        _join_gvkey=left["gvkey"].astype("string").fillna("__UNLINKED__")
    )
    right_work = right.assign(
        _join_gvkey=right["gvkey"].astype("string")
    ).drop(columns="gvkey")
    left_work = left_work.sort_values(["date", "_join_gvkey"])
    right_work = right_work.sort_values([right_on, "_join_gvkey"])
    merged = pd.merge_asof(
        left_work,
        right_work,
        left_on="date",
        right_on=right_on,
        by="_join_gvkey",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.drop(columns="_join_gvkey").sort_values(
        "_panel_row", ignore_index=True
    )


def month_age(current: pd.Series, observation: pd.Series) -> pd.Series:
    """Return the whole calendar-month distance between two date series.

    Args:
        current: Current portfolio weights.
        observation: Observation used by this operation.

    Returns:
        Row-aligned Series containing the values described above.
    """
    return (
        (current.dt.year - observation.dt.year) * 12
        + current.dt.month
        - observation.dt.month
    )


def attach_fundamentals(
    panel: pd.DataFrame,
    quarterly: pd.DataFrame,
    annual: pd.DataFrame,
) -> pd.DataFrame:
    """Attach quarterly and annual fundamentals separately and remove stale data.

    Args:
        panel: Monthly point-in-time research panel.
        quarterly: Quarterly Compustat rows.
        annual: Annual Compustat rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    quarterly_payload = [column for column in quarterly if column != "gvkey"]
    panel = merge_asof_by_gvkey(panel, quarterly, "available_date")
    quarterly_stale = panel["datadate"].notna() & month_age(
        panel["date"], panel["datadate"]
    ).gt(MAX_STALENESS_MONTHS)
    panel = panel.assign(
        **{
            column: panel[column].mask(quarterly_stale)
            for column in quarterly_payload
        }
    )

    annual_payload = [column for column in annual if column != "gvkey"]
    panel = merge_asof_by_gvkey(panel, annual, "available_date_a")
    annual_stale = panel["datadate_a"].notna() & month_age(
        panel["date"], panel["datadate_a"]
    ).gt(MAX_STALENESS_MONTHS)
    panel = panel.assign(
        **{column: panel[column].mask(annual_stale) for column in annual_payload}
    )
    print(
        f"Stale fundamentals removed at >{MAX_STALENESS_MONTHS} months: "
        f"quarterly={int(quarterly_stale.sum()):,}, "
        f"annual={int(annual_stale.sum()):,}."
    )
    return panel


def attach_sectors(panel: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """Attach the GICS classification whose effective range covers each month.

    Args:
        panel: Monthly point-in-time research panel.
        sectors: Point-in-time sector rows.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    sectors = sectors.assign(
        gvkey=lambda frame: frame["gvkey"].astype("string"),
        indfrom=lambda frame: pd.to_datetime(frame["indfrom"]),
        indthru=lambda frame: pd.to_datetime(frame["indthru"]),
    )
    duplicate_starts = int(
        sectors.duplicated(["gvkey", "indfrom"], keep=False).sum()
    )
    if duplicate_starts:
        raise RuntimeError(
            f"GICS history has {duplicate_starts:,} rows with duplicate gvkey/indfrom."
        )
    sector_payload = [column for column in sectors if column != "gvkey"]
    panel = merge_asof_by_gvkey(panel, sectors, "indfrom")
    outside_range = panel["indfrom"].notna() & panel["indthru"].notna() & panel[
        "date"
    ].gt(panel["indthru"])
    panel = panel.assign(
        **{column: panel[column].mask(outside_range) for column in sector_payload}
    )
    print(f"GICS candidates rejected outside their date range: {outside_range.sum():,}.")
    return panel


def standardize_units(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert price-panel capitalization and shares to Compustat-scale units.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    # Units: market_cap is USD millions; shares_out and short-interest fields
    # are millions of shares; price is USD/share; volume is individual shares.
    return panel.assign(
        market_cap=panel["market_cap"] / 1_000_000.0,
        shares_out=panel["shares_out"] / 1_000_000.0,
        short_interest_shares=panel["short_interest_shares"] / 1_000_000.0,
        short_interest_adjusted_shares=(
            panel["short_interest_adjusted_shares"] / 1_000_000.0
        ),
    )


def print_live_extension_checks(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    fundamentals: pd.DataFrame,
    sectors: pd.DataFrame,
    previous_panel_rows: int | None,
) -> None:
    """Report live row counts, investability, and point-in-time data coverage.

    Args:
        panel: Monthly point-in-time research panel.
        cutoff: Cutoff used by this operation.
        fundamentals: Compustat fundamental rows.
        sectors: Point-in-time sector rows.
        previous_panel_rows: Previous panel rows used by this operation.

    Returns:
        None.
    """
    live = panel.loc[panel["date"].gt(cutoff)].copy()
    if live.empty:
        print("Live extension checks: no post-cutoff panel rows.")
        return
    historical_rows = int(panel["date"].le(cutoff).sum())
    print("Live extension row counts:")
    print(
        f"  previous panel rows: "
        f"{previous_panel_rows if previous_panel_rows is not None else 'not available'}"
    )
    print(f"  rebuilt historical rows through {cutoff.date()}: {historical_rows:,}")
    print(f"  appended live panel rows: {len(live):,}")
    print(f"  extended panel rows: {len(panel):,}")
    print(
        "  duplicate (permno, date) rows: "
        f"{int(panel.duplicated(['permno', 'date']).sum()):,}"
    )

    monthly = panel.groupby("date", as_index=False).agg(
        rows=("permno", "size"),
        in_universe=("in_universe", "sum"),
        investable=("investable", "sum"),
        quarantined=("quarantined", "sum"),
    )
    comparison = pd.concat(
        [
            monthly.loc[monthly["date"].le(cutoff)].tail(6).assign(source="CRSP"),
            monthly.loc[monthly["date"].gt(cutoff)].assign(source="live"),
        ],
        ignore_index=True,
    )
    print("Investable names in trailing CRSP and new live months:")
    print(
        comparison[
            ["source", "date", "rows", "in_universe", "investable", "quarantined"]
        ].to_string(index=False)
    )

    members = live.loc[live["in_universe"]].copy()
    coverage = (
        members.assign(
            gvkey_covered=members["gvkey"].notna(),
            quarterly_covered=members["datadate"].notna(),
            annual_covered=members["datadate_a"].notna(),
            sector_covered=members["gsector"].notna(),
        )
        .groupby("date", as_index=False)
        .agg(
            member_rows=("permno", "size"),
            gvkey=("gvkey_covered", "mean"),
            quarterly=("quarterly_covered", "mean"),
            annual=("annual_covered", "mean"),
            sector=("sector_covered", "mean"),
        )
    )
    future_quarterly = int(
        (
            live["available_date"].notna()
            & live["available_date"].gt(live["date"])
        ).sum()
    )
    future_annual = int(
        (
            live["available_date_a"].notna()
            & live["available_date_a"].gt(live["date"])
        ).sum()
    )
    quarterly_source_max = pd.to_datetime(
        fundamentals.loc[fundamentals["freq"].eq("Q"), "datadate"]
    ).max()
    annual_source_max = pd.to_datetime(
        fundamentals.loc[fundamentals["freq"].eq("A"), "datadate"]
    ).max()
    sector_source_max = pd.to_datetime(sectors["indfrom"]).max()
    print(
        "Live point-in-time sources: "
        f"quarterly through {quarterly_source_max.date()}, "
        f"annual through {annual_source_max.date()}, "
        f"GICS updates through {sector_source_max.date()}."
    )
    print(
        "Live fundamentals used before availability: "
        f"quarterly={future_quarterly:,}, annual={future_annual:,}."
    )
    print("Live in-universe identifier/fundamental/sector coverage:")
    print(
        coverage.assign(
            **{
                column: coverage[column].map(lambda value: f"{value:.1%}")
                for column in ["gvkey", "quarterly", "annual", "sector"]
            }
        ).to_string(index=False)
    )


def coverage_by_year(
    panel: pd.DataFrame,
    population: str = "in_universe",
) -> pd.DataFrame:
    """Return identifier and data-source coverage by year for one population.

    Args:
        panel: Monthly point-in-time research panel.
        population: Population used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    observations = panel.loc[panel[population]].assign(
        year=panel.loc[panel[population], "date"].dt.year
    )
    coverage = observations.assign(
        gvkey_coverage=observations["gvkey"].notna(),
        quarterly_coverage=observations["datadate"].notna(),
        annual_coverage=observations["datadate_a"].notna(),
        sector_coverage=observations["gsector"].notna(),
    ).groupby("year").agg(
        rows=("permno", "size"),
        gvkey=("gvkey_coverage", "mean"),
        quarterly=("quarterly_coverage", "mean"),
        annual=("annual_coverage", "mean"),
        sector=("sector_coverage", "mean"),
    )
    coverage[["gvkey", "quarterly", "annual", "sector"]] *= 100.0
    return coverage


def print_coverage_by_year(panel: pd.DataFrame) -> None:
    """Print identifier and data-source coverage for in-universe rows by year.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        None.
    """
    coverage = coverage_by_year(panel, "in_universe")
    print("In-universe coverage by year:")
    print(coverage.to_string(float_format=lambda value: f"{value:.1f}%"))


def investable_names_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the number of investable securities in each month.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    return (
        panel.loc[panel["investable"]]
        .groupby("date")["permno"]
        .nunique()
        .rename("investable_names")
        .reset_index()
    )


def cumulative_investable_return(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the cumulative equal-weighted return of investable securities.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    monthly = (
        panel.loc[panel["investable"] & panel["ret"].notna()]
        .groupby("date")["ret"]
        .mean()
        .astype(float)
        .rename("monthly_return")
        .reset_index()
    )
    return monthly.assign(
        cumulative_return=(1.0 + monthly["monthly_return"]).cumprod() - 1.0
    )


def panel_preview(panel: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    """Return a compact set of columns from the finished research panel.

    Args:
        panel: Monthly point-in-time research panel.
        rows: Rows used by this operation.

    Returns:
        DataFrame containing the rows or diagnostics described above.
    """
    columns = [
        "date",
        "permno",
        "gvkey",
        "ticker",
        "company_name",
        "ret",
        "market_cap",
        "in_universe",
        "investable",
        "gsector",
    ]
    return panel.loc[:, columns].head(rows)


def print_unlinked_in_universe(panel: pd.DataFrame) -> int:
    """List in-universe securities that still lack a date-valid CCM link.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        None.
    """
    missing = panel.loc[panel["in_universe"] & panel["gvkey"].isna()].copy()
    print(f"In-universe rows without a GVKEY: {len(missing):,}")
    if missing.empty:
        print("  none")
        return 0

    def combine_labels(values: pd.Series) -> str:
        """Combine distinct non-null labels in order of appearance."""
        return " | ".join(dict.fromkeys(values.dropna().astype(str)))

    summary = missing.groupby("permno", as_index=False).agg(
        ticker=("ticker", combine_labels),
        company_name=("company_name", combine_labels),
        months=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        secondary_share_class=("secondary_share_class", "all"),
    )
    print(summary.to_string(index=False))
    return len(missing)


def benchmark_replication_check(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Compare lagged-cap-weighted members with the configured benchmark.

    Args:
        panel: Monthly point-in-time research panel.
        factors: Monthly factor and benchmark-return rows.

    Returns:
        Computed result as ``tuple[dict[str, object], pd.DataFrame, pd.DataFrame]``.
    """
    if UNIVERSE == "russell3000":
        benchmark_column = "russell3000_total_return"
        benchmark_label = "Russell 3000"
    elif UNIVERSE == "sp500":
        benchmark_column = "sp500_total_return"
        benchmark_label = "S&P 500"
    else:
        result = {
            "check name": "lagged-cap benchmark replication",
            "tier": "FAIL",
            "value": "not applicable to UNIVERSE='all'",
            "expected": "configured index universe",
            "pass/fail": "N/A",
        }
        return result, pd.DataFrame(), pd.DataFrame()

    work = panel.loc[
        :,
        ["permno", "date", "in_universe", "ret", "market_cap"],
    ].sort_values(["permno", "date"])
    grouped = work.groupby("permno", sort=False)
    work = work.assign(
        _lag_market_cap=grouped["market_cap"].shift(1),
        _lag_date=grouped["date"].shift(1),
    )
    current_month = work["date"].dt.year * 12 + work["date"].dt.month
    prior_month = work["_lag_date"].dt.year * 12 + work["_lag_date"].dt.month
    valid = (
        work["in_universe"]
        & work["ret"].notna()
        & work["_lag_market_cap"].gt(0)
        & current_month.sub(prior_month).eq(1)
    )
    weighted = work.loc[valid].copy()
    weighted = weighted.assign(
        _weighted_return=weighted["ret"] * weighted["_lag_market_cap"]
    )
    monthly = weighted.groupby("date", as_index=False).agg(
        weighted_return=("_weighted_return", "sum"),
        weight=("_lag_market_cap", "sum"),
        names=("permno", "nunique"),
    )
    monthly = monthly.assign(
        replicated_return=monthly["weighted_return"] / monthly["weight"]
    )
    comparison = monthly.merge(
        factors[["date", benchmark_column]],
        on="date",
        how="inner",
        validate="1:1",
    ).assign(
        difference=lambda frame: (
            frame["replicated_return"] - frame[benchmark_column]
        )
    )
    correlation = float(
        comparison["replicated_return"].corr(comparison[benchmark_column])
    )
    tracking_error = float(comparison["difference"].std(ddof=1) * np.sqrt(12.0))
    average_difference = float(comparison["difference"].mean())
    passed = bool(
        correlation >= BENCHMARK_MIN_CORRELATION
        and tracking_error < BENCHMARK_MAX_TRACKING_ERROR
    )

    annual = (
        comparison.assign(year=comparison["date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            months=("date", "size"),
            replicated_return=(
                "replicated_return",
                lambda values: np.prod(1.0 + values) - 1.0,
            ),
            benchmark_return=(
                benchmark_column,
                lambda values: np.prod(1.0 + values) - 1.0,
            ),
        )
        .assign(
            difference=lambda frame: (
                frame["replicated_return"] - frame["benchmark_return"]
            )
        )
    )
    largest = (
        comparison.assign(
            absolute_difference=lambda frame: frame["difference"].abs(),
            difference_bp=lambda frame: frame["difference"] * 10_000.0,
        )
        .nlargest(10, "absolute_difference")
        [[
            "date",
            "names",
            "replicated_return",
            benchmark_column,
            "difference",
            "difference_bp",
        ]]
    )

    print(f"{benchmark_label} benchmark replication:")
    print(f"  matched months: {len(comparison):,}")
    print(f"  correlation: {correlation:.6f}")
    print(f"  annualized tracking error: {tracking_error:.4%}")
    print(f"  average monthly difference: {average_difference:.4%}")
    print("  annual returns:")
    annual_display = annual.assign(
        replicated_return=annual["replicated_return"].map(lambda value: f"{value:.2%}"),
        benchmark_return=annual["benchmark_return"].map(
            lambda value: f"{value:.2%}"
        ),
        difference=annual["difference"].map(lambda value: f"{value:.2%}"),
    )
    print(annual_display.to_string(index=False))
    print("  10 largest monthly differences:")
    largest_display = largest.assign(
        replicated_return=largest["replicated_return"].map(
            lambda value: f"{value:.4%}"
        ),
        **{benchmark_column: largest[benchmark_column].map(
            lambda value: f"{value:.4%}"
        )},
        difference=largest["difference"].map(lambda value: f"{value:.4%}"),
        difference_bp=largest["difference_bp"].map(lambda value: f"{value:.2f}"),
    )
    print(largest_display.to_string(index=False))
    result = {
        "check name": f"lagged-cap {benchmark_label} benchmark replication",
        "tier": "FAIL",
        "value": (
            f"corr={correlation:.6f}, tracking error={tracking_error:.4%}, "
            f"mean monthly diff={average_difference:.4%}"
        ),
        "expected": (
            f"corr>={BENCHMARK_MIN_CORRELATION:.3f}, "
            f"tracking error<{BENCHMARK_MAX_TRACKING_ERROR:.1%}"
        ),
        "pass/fail": "PASS" if passed else "FAIL",
    }
    return result, annual, largest


def quarterly_annual_cashflow_check(
    panel: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare four standalone quarters with annual OCF and capex.

    Args:
        panel: Monthly point-in-time research panel.
        fundamentals: Compustat fundamental rows.

    Returns:
        Computed result as ``tuple[dict[str, object], pd.DataFrame]``.
    """
    quarterly = fundamentals.loc[
        fundamentals["freq"].eq("Q")
        & fundamentals["fqtr"].isin([1, 2, 3, 4]),
        ["gvkey", "fyearq", "fqtr", "oancf_qtr", "capx_qtr"],
    ].copy()
    quarterly = quarterly.assign(
        fyearq=pd.to_numeric(quarterly["fyearq"], errors="coerce")
    )
    quarterly_sums = quarterly.groupby(["gvkey", "fyearq"], as_index=False).agg(
        rows=("fqtr", "size"),
        quarters=("fqtr", "nunique"),
        oancf_quarters=("oancf_qtr", "count"),
        capx_quarters=("capx_qtr", "count"),
        oancf_qtr_sum=("oancf_qtr", "sum"),
        capx_qtr_sum=("capx_qtr", "sum"),
    )
    quarterly_sums = quarterly_sums.loc[
        quarterly_sums["rows"].eq(4)
        & quarterly_sums["quarters"].eq(4)
        & quarterly_sums["oancf_quarters"].eq(4)
        & quarterly_sums["capx_quarters"].eq(4)
    ]

    annual = fundamentals.loc[
        fundamentals["freq"].eq("A") & fundamentals["fyear"].notna(),
        ["gvkey", "datadate", "fyear", "oancf", "capx"],
    ].copy()
    annual = annual.assign(
        datadate=pd.to_datetime(annual["datadate"]),
        fyear=pd.to_numeric(annual["fyear"], errors="coerce"),
    )
    annual = annual.sort_values("datadate").drop_duplicates(
        ["gvkey", "fyear"], keep="last"
    )
    comparisons = quarterly_sums.merge(
        annual,
        left_on=["gvkey", "fyearq"],
        right_on=["gvkey", "fyear"],
        how="inner",
        validate="1:1",
    ).dropna(subset=["oancf", "capx"])
    comparisons = comparisons.assign(year=comparisons["datadate"].dt.year)

    company_sizes = (
        panel.loc[panel["gvkey"].notna() & panel["market_cap"].gt(0)]
        .assign(year=lambda frame: frame["date"].dt.year)
        .groupby(["gvkey", "year"], as_index=False)["market_cap"]
        .max()
    )
    comparisons = comparisons.merge(
        company_sizes,
        on=["gvkey", "year"],
        how="inner",
        validate="m:1",
    )
    available_years = np.array(sorted(comparisons["year"].unique()))
    target_positions = np.unique(
        np.rint(
            np.linspace(
                0,
                len(available_years) - 1,
                min(CASH_FLOW_CHECK_SAMPLE_SIZE, len(available_years)),
            )
        ).astype(int)
    )
    target_years = available_years[target_positions]
    chosen: list[pd.Series] = []
    used_gvkeys: set[str] = set()
    for year in target_years:
        choices = comparisons.loc[
            comparisons["year"].eq(year)
            & ~comparisons["gvkey"].isin(used_gvkeys)
        ].sort_values(["market_cap", "gvkey"], ascending=[False, True])
        if not choices.empty:
            observation = choices.iloc[0]
            chosen.append(observation)
            used_gvkeys.add(str(observation["gvkey"]))

    if len(chosen) < CASH_FLOW_CHECK_SAMPLE_SIZE:
        remaining = (
            comparisons.loc[~comparisons["gvkey"].isin(used_gvkeys)]
            .sort_values(["market_cap", "year"], ascending=[False, True])
            .drop_duplicates("gvkey", keep="first")
        )
        chosen.extend(
            observation
            for _, observation in remaining.head(
                CASH_FLOW_CHECK_SAMPLE_SIZE - len(chosen)
            ).iterrows()
        )
    sample = pd.DataFrame(chosen).reset_index(drop=True)

    names = (
        panel.loc[
            panel["gvkey"].notna(),
            ["gvkey", "date", "ticker", "company_name"],
        ]
        .sort_values("date")
        .drop_duplicates("gvkey", keep="last")
        .drop(columns="date")
    )
    sample = sample.merge(names, on="gvkey", how="left", validate="m:1")
    sample = sample.assign(
        oancf_difference=lambda frame: frame["oancf_qtr_sum"] - frame["oancf"],
        capx_difference=lambda frame: frame["capx_qtr_sum"] - frame["capx"],
    )
    oancf_tolerance = np.maximum(
        CASH_FLOW_ABS_TOLERANCE,
        sample["oancf"].abs() * CASH_FLOW_REL_TOLERANCE,
    )
    capx_tolerance = np.maximum(
        CASH_FLOW_ABS_TOLERANCE,
        sample["capx"].abs() * CASH_FLOW_REL_TOLERANCE,
    )
    sample = sample.assign(
        oancf_match=sample["oancf_difference"].abs().le(oancf_tolerance),
        capx_match=sample["capx_difference"].abs().le(capx_tolerance),
    )
    sample = sample[[
        "year",
        "gvkey",
        "ticker",
        "company_name",
        "oancf_qtr_sum",
        "oancf",
        "oancf_difference",
        "oancf_match",
        "capx_qtr_sum",
        "capx",
        "capx_difference",
        "capx_match",
    ]].sort_values("year", ignore_index=True)
    oancf_matches = int(sample["oancf_match"].sum())
    capx_matches = int(sample["capx_match"].sum())
    passed = bool(
        len(sample) == CASH_FLOW_CHECK_SAMPLE_SIZE
        and oancf_matches == len(sample)
        and capx_matches == len(sample)
    )

    print(
        "Quarterly versus annual cash-flow check "
        f"({len(sample)} large companies across fiscal years):"
    )
    print(
        sample.to_string(
            index=False,
            float_format=lambda value: f"{value:,.1f}",
        )
    )
    mismatches = sample.loc[~sample["oancf_match"] | ~sample["capx_match"]]
    print(f"  operating cash flow matches: {oancf_matches}/{len(sample)}")
    print(f"  capex matches: {capx_matches}/{len(sample)}")
    print(f"  rows with any difference beyond tolerance: {len(mismatches)}")
    result = {
        "check name": "four-quarter cash flow sums versus annual",
        "tier": "WARN",
        "value": (
            f"OCF={oancf_matches}/{len(sample)}, "
            f"capex={capx_matches}/{len(sample)}, mismatches={len(mismatches)}"
        ),
        "expected": (
            f"{CASH_FLOW_CHECK_SAMPLE_SIZE}/{CASH_FLOW_CHECK_SAMPLE_SIZE} each "
            f"within max(${CASH_FLOW_ABS_TOLERANCE:.0f}m, "
            f"{CASH_FLOW_REL_TOLERANCE:.1%})"
        ),
        "pass/fail": "PASS" if passed else "FAIL",
    }
    return result, sample


def check_market_cap_ratio(
    panel: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> dict[str, object]:
    """Return the CRSP/Compustat market-value comparison check.

    Args:
        panel: Monthly point-in-time research panel.
        quarterly: Quarterly Compustat rows.

    Returns:
        Computed result as ``dict[str, object]``.
    """
    prices = panel.loc[
        panel["gvkey"].notna() & panel["market_cap"].gt(0),
        ["gvkey", "date", "market_cap"],
    ].assign(month=lambda frame: frame["date"].dt.to_period("M"))
    prices = (
        prices.sort_values("market_cap")
        .drop_duplicates(["gvkey", "month"], keep="last")
    )
    compustat = quarterly.loc[
        quarterly["mkvaltq"].notna() & quarterly["mkvaltq"].gt(0),
        ["gvkey", "datadate", "mkvaltq"],
    ].assign(month=lambda frame: frame["datadate"].dt.to_period("M"))
    compustat = (
        compustat.sort_values("datadate")
        .drop_duplicates(["gvkey", "month"], keep="last")
    )
    comparison = prices.merge(
        compustat,
        on=["gvkey", "month"],
        how="inner",
        validate="1:1",
    )
    ratio = comparison["market_cap"] / comparison["mkvaltq"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    median = float(ratio.median()) if len(ratio) else np.nan
    passed = bool(len(ratio) and 0.8 <= median <= 1.2)
    return {
        "check name": "market_cap/mkvaltq at company quarter-ends",
        "tier": "FAIL",
        "value": (
            f"n={len(ratio):,}, median={median:.4f}, "
            f"p10={ratio.quantile(0.10):.4f}, "
            f"p90={ratio.quantile(0.90):.4f}"
        ),
        "expected": "median between 0.8 and 1.2",
        "pass/fail": "PASS" if passed else "FAIL",
    }


def check_data_dictionary(panel: pd.DataFrame) -> dict[str, object]:
    """Require one nonempty dictionary entry for every final panel column.

    Args:
        panel: Completed monthly research panel before it is written.

    Returns:
        Blocking check result covering schema, duplicates, and column coverage.
    """
    expected_headers = ["variable", "description"]
    if not DATA_DICTIONARY_PATH.exists():
        return {
            "check name": "data dictionary matches panel columns",
            "tier": "FAIL",
            "value": f"missing file: {DATA_DICTIONARY_PATH}",
            "expected": "exactly one nonempty description per panel column",
            "pass/fail": "FAIL",
        }

    try:
        dictionary = pd.read_csv(DATA_DICTIONARY_PATH, dtype="string")
    except Exception as exc:
        return {
            "check name": "data dictionary matches panel columns",
            "tier": "FAIL",
            "value": f"unreadable CSV: {type(exc).__name__}: {exc}",
            "expected": "exactly one nonempty description per panel column",
            "pass/fail": "FAIL",
        }

    headers_match = dictionary.columns.tolist() == expected_headers
    if headers_match:
        variables = dictionary["variable"].fillna("").str.strip()
        descriptions = dictionary["description"].fillna("").str.strip()
        duplicates = sorted(variables.loc[variables.duplicated(keep=False)].unique())
        missing = sorted(set(panel.columns).difference(variables))
        extra = sorted(set(variables).difference(panel.columns).difference({""}))
        blank_rows = int(variables.eq("").sum() + descriptions.eq("").sum())
    else:
        duplicates = []
        missing = list(panel.columns)
        extra = []
        blank_rows = len(dictionary)

    passed = bool(
        headers_match
        and not duplicates
        and not missing
        and not extra
        and blank_rows == 0
        and len(dictionary) == len(panel.columns)
    )
    value = (
        f"rows={len(dictionary):,}, panel_columns={len(panel.columns):,}, "
        f"missing={len(missing):,}, extra={len(extra):,}, "
        f"duplicates={len(duplicates):,}, blank_cells={blank_rows:,}, "
        f"headers={'correct' if headers_match else 'incorrect'}"
    )
    if missing:
        value += f"; missing sample={missing[:10]}"
    if extra:
        value += f"; extra sample={extra[:10]}"
    if duplicates:
        value += f"; duplicate sample={duplicates[:10]}"
    return {
        "check name": "data dictionary matches panel columns",
        "tier": "FAIL",
        "value": value,
        "expected": "exactly one nonempty description per panel column",
        "pass/fail": "PASS" if passed else "FAIL",
    }


def run_checks(
    panel: pd.DataFrame,
    quarterly: pd.DataFrame,
    fundamentals: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Print diagnostics and return their table plus any hard failures.

    Every check declares its own severity tier explicitly at the point it is
    built: ``FAIL`` means a failing result blocks ``write_panel`` and raises;
    ``WARN`` means a failing result is printed and summarized but does not
    stop the pipeline. The tier is never inferred from which list a check
    ends up in.

    Args:
        panel: Monthly point-in-time research panel.
        quarterly: Quarterly Compustat rows.
        fundamentals: Compustat fundamental rows.
        factors: Monthly factor and benchmark-return rows.

    Returns:
        Computed result as ``tuple[pd.DataFrame, list[str]]``.
    """
    results: list[dict[str, object]] = []
    results.append(check_data_dictionary(panel))
    duplicates = int(panel.duplicated(["permno", "date"]).sum())
    duplicate_pass = duplicates == 0
    results.append(
        {
            "check name": "duplicate (permno, date) rows",
            "tier": "FAIL",
            "value": f"{duplicates:,}",
            "expected": "0",
            "pass/fail": "PASS" if duplicate_pass else "FAIL",
        }
    )

    quarterly_future = int(
        (panel["available_date"].notna() & panel["available_date"].gt(panel["date"])).sum()
    )
    annual_future = int(
        (
            panel["available_date_a"].notna()
            & panel["available_date_a"].gt(panel["date"])
        ).sum()
    )
    future_pass = quarterly_future == 0 and annual_future == 0
    results.append(
        {
            "check name": "fundamentals used before available_date",
            "tier": "FAIL",
            "value": f"quarterly={quarterly_future:,}, annual={annual_future:,}",
            "expected": "quarterly=0, annual=0",
            "pass/fail": "PASS" if future_pass else "FAIL",
        }
    )

    unlinked = print_unlinked_in_universe(panel)
    results.append(
        {
            "check name": "in-universe rows without a GVKEY",
            "tier": "WARN",
            "value": f"{unlinked:,}",
            "expected": "0",
            "pass/fail": "PASS" if unlinked == 0 else "FAIL",
        }
    )

    winners = (
        panel.loc[panel["investable"]]
        .groupby(["date", "gvkey"])["permno"]
        .nunique()
        .rename("winners")
    )
    winner_violations = int(winners.gt(1).sum())
    max_winners = int(winners.max()) if len(winners) else 0
    winner_pass = winner_violations == 0
    results.append(
        {
            "check name": "at most one investable security per company-date",
            "tier": "FAIL",
            "value": f"violations={winner_violations:,}, max={max_winners:,}",
            "expected": "violations=0, max=1",
            "pass/fail": "PASS" if winner_pass else "FAIL",
        }
    )
    print_coverage_by_year(panel)

    members = panel.loc[panel["in_universe"] & panel["ret"].notna()]
    monthly_returns = members.groupby("date")["ret"].mean().astype(float)
    annual_return = float(
        np.prod(1.0 + monthly_returns) ** (12.0 / len(monthly_returns)) - 1.0
    )
    annual_volatility = float(monthly_returns.std(ddof=1) * np.sqrt(12.0))
    return_pass = 0.06 <= annual_return <= 0.12
    volatility_pass = 0.12 <= annual_volatility <= 0.20
    results.append(
        {
            "check name": "equal-weighted annualized in-universe return",
            "tier": "WARN",
            "value": f"{annual_return:.2%}",
            "expected": "roughly 6–12%",
            "pass/fail": "PASS" if return_pass else "FAIL",
        }
    )
    results.append(
        {
            "check name": "equal-weighted annualized in-universe volatility",
            "tier": "WARN",
            "value": f"{annual_volatility:.2%}",
            "expected": "roughly 12–20%",
            "pass/fail": "PASS" if volatility_pass else "FAIL",
        }
    )

    return_months = monthly_returns.rename_axis("date").reset_index()
    return_months["month"] = return_months["date"].dt.to_period("M")
    for month in ("2008-10", "2020-03"):
        value = return_months.loc[
            return_months["month"].eq(pd.Period(month, freq="M")), "ret"
        ]
        monthly_return = float(value.iloc[0]) if len(value) else np.nan
        passed = bool(len(value) and monthly_return < -0.10)
        results.append(
            {
                "check name": f"{month} equal-weighted in-universe return",
                "tier": "WARN",
                "value": f"{monthly_return:.2%}",
                "expected": "below -10%",
                "pass/fail": "PASS" if passed else "FAIL",
            }
        )

    # check_market_cap_ratio, benchmark_replication_check, and
    # quarterly_annual_cashflow_check each declare their own "tier" key.
    results.append(check_market_cap_ratio(panel, quarterly))
    benchmark_result, _, _ = benchmark_replication_check(panel, factors)
    results.append(benchmark_result)
    cashflow_result, _ = quarterly_annual_cashflow_check(panel, fundamentals)
    results.append(cashflow_result)

    checks = pd.DataFrame(
        results,
        columns=["check name", "tier", "value", "expected", "pass/fail"],
    )
    for row in checks.itertuples(index=False, name=None):
        name, tier, value, expected, status = row
        print(f"CHECK {status} ({tier}) — {name}: {value} (expected {expected})")

    hard_failures = checks.loc[
        checks["tier"].eq("FAIL") & checks["pass/fail"].eq("FAIL"),
        "check name",
    ].tolist()
    soft_warnings = checks.loc[
        checks["tier"].eq("WARN") & checks["pass/fail"].eq("FAIL"),
        "check name",
    ].tolist()

    print("\nCHECK SUMMARY")
    if hard_failures:
        print(f"  {len(hard_failures)} FAIL (blocking write_panel):")
        for name in hard_failures:
            print(f"    - {name}")
    else:
        print("  0 FAIL (blocking write_panel)")
    if soft_warnings:
        print(f"  {len(soft_warnings)} WARN (non-blocking):")
        for name in soft_warnings:
            print(f"    - {name}")
    else:
        print("  0 WARN (non-blocking)")

    return checks, hard_failures


def write_panel(panel: pd.DataFrame) -> None:
    """Atomically write the completed research panel.

    Args:
        panel: Monthly point-in-time research panel.

    Returns:
        None.
    """
    PANEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PANEL_OUTPUT_PATH.with_suffix(".tmp.parquet")
    panel.to_parquet(temporary, index=False)
    temporary.replace(PANEL_OUTPUT_PATH)


def print_summary(panel: pd.DataFrame, path: Path) -> None:
    """Print final panel dimensions and file size.

    Args:
        panel: Monthly point-in-time research panel.
        path: Input or output filesystem path.

    Returns:
        None.
    """
    print("Panel summary:")
    print(f"  rows: {len(panel):,}")
    print(f"  date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"  unique permnos: {panel['permno'].nunique():,}")
    print(f"  in_universe rows: {int(panel['in_universe'].sum()):,}")
    print(f"  investable rows: {int(panel['investable'].sum()):,}")
    print(f"  columns: {len(panel.columns):,}")
    print(f"  file size: {path.stat().st_size / (1024 ** 2):,.1f} MiB")


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, validate, and write the local point-in-time research panel.

    Args:
        None.

    Returns:
        None.
    """
    previous_panel_rows = None
    if PANEL_OUTPUT_PATH.exists():
        previous_panel_rows = len(
            pd.read_parquet(PANEL_OUTPUT_PATH, columns=["permno"])
        )
    (
        prices,
        fundamentals,
        links,
        sectors,
        factors,
        short_interest,
        live_prices,
    ) = load_inputs()
    check_live_price_staleness(live_prices)
    prices, crsp_cutoff = append_live_prices(prices, live_prices)
    prices = add_trailing_adv(prices)
    prices = assign_universe_membership(prices)
    panel = keep_ever_members(prices)
    panel = attach_links(panel, links)
    panel = add_investability_flags(panel)
    panel = attach_short_interest(panel, short_interest)
    quarterly, annual = prepare_fundamentals(fundamentals, panel["gvkey"])
    panel = attach_fundamentals(panel, quarterly, annual)
    panel = attach_sectors(panel, sectors)
    panel = standardize_units(panel)
    panel = panel.drop(columns="_panel_row").sort_values(
        ["date", "permno"], ignore_index=True
    )
    print_live_extension_checks(
        panel,
        crsp_cutoff,
        fundamentals,
        sectors,
        previous_panel_rows,
    )
    checks, hard_failures = run_checks(panel, quarterly, fundamentals, factors)
    if hard_failures:
        raise RuntimeError(f"Panel validation failed: {hard_failures}")
    write_panel(panel)
    print(f"Wrote {PANEL_OUTPUT_PATH}")
    print_summary(panel, PANEL_OUTPUT_PATH)
    return panel, checks


if __name__ == "__main__":
    main()
