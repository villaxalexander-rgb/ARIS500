"""Provide neutral, reusable time-alignment utilities for production and research.

Reads only in-memory DataFrames and writes nothing.  Forward returns live here
rather than in the research evaluator so production portfolio code does not
depend on research-only modules; the helper requires the next *calendar* month
and deliberately refuses to bridge a missing month.
"""

import numpy as np
import pandas as pd


def forward_returns(panel: pd.DataFrame) -> pd.Series:
    """Return each security's return in the immediately following calendar month.

    Args:
        panel: Monthly panel with ``permno``, ``date``, and ``ret`` columns.

    Returns:
        A row-aligned Series named ``forward_return``; gaps map to ``NaN``.
    """
    work = panel.loc[:, ["permno", "date", "ret"]].copy()
    work = work.assign(
        _source_row=np.arange(len(work), dtype=np.int64),
        date=pd.to_datetime(work["date"]),
        ret=pd.to_numeric(work["ret"], errors="coerce"),
    ).sort_values(["permno", "date"], ignore_index=True)
    work = work.assign(
        _month_number=work["date"].dt.year * 12 + work["date"].dt.month
    )
    groups = work.groupby("permno", sort=False)
    next_return = groups["ret"].shift(-1)
    next_month = groups["_month_number"].shift(-1)
    consecutive = next_month.sub(work["_month_number"]).eq(1)
    work = work.assign(_forward_return=next_return.where(consecutive))

    output = pd.Series(np.nan, index=np.arange(len(panel)), dtype="float64")
    output.iloc[work["_source_row"].to_numpy()] = work["_forward_return"].to_numpy()
    output.index = panel.index
    output.name = "forward_return"
    return output


__all__ = ["forward_returns"]
