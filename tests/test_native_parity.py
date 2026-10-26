"""Check the C++ weight solver against the Python reference, bit for bit.

Builds a synthetic panel (no WRDS data needed), constructs the portfolio once
with the native solver and once with the Python reference, and requires the
holdings, diagnostics, and sector weights to be exactly equal. Run with
``python -m pytest tests`` or ``python tests/test_native_parity.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import native_core  # noqa: E402
from src.model import portfolio  # noqa: E402

SECTORS = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], dtype=float)
SECTOR_SHARES = [0.05, 0.07, 0.13, 0.11, 0.07, 0.13, 0.14, 0.18, 0.05, 0.04, 0.03]


def synthetic_panel(
    n_stocks: int = 3000,
    start: str = "2006-12-31",
    end: str = "2009-12-31",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return a panel, composite score, and beta with realistic structure.

    Args:
        n_stocks: Number of securities per month.
        start: First month-end.
        end: Last month-end.
        seed: Random seed.

    Returns:
        Computed result as ``tuple[pd.DataFrame, pd.Series, pd.Series]``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="ME")
    permnos = np.arange(10001, 10001 + n_stocks)
    stock_sector = rng.choice(SECTORS, n_stocks, p=SECTOR_SHARES)
    log_cap = rng.normal(7.5, 1.6, n_stocks)
    beta_base = rng.normal(1.15, 0.35, n_stocks)
    score_base = rng.normal(0.0, 1.0, n_stocks)
    months = []
    for date in dates:
        log_cap = log_cap + rng.normal(0.006, 0.08, n_stocks)
        score_base = 0.9 * score_base + rng.normal(0.0, 0.45, n_stocks)
        months.append(
            pd.DataFrame(
                {
                    "permno": permnos,
                    "date": date,
                    "investable": rng.random(n_stocks) > 0.04,
                    "in_universe": True,
                    "market_cap": np.exp(log_cap),
                    "gsector": np.where(
                        rng.random(n_stocks) < 0.003, np.nan, stock_sector
                    ),
                    "ret": rng.normal(0.008, 0.09, n_stocks),
                    "score": np.where(
                        rng.random(n_stocks) < 0.02, np.nan, score_base
                    ),
                    "beta": np.where(
                        rng.random(n_stocks) < 0.03,
                        np.nan,
                        beta_base + rng.normal(0.0, 0.05, n_stocks),
                    ),
                }
            )
        )
    panel = pd.concat(months, ignore_index=True)
    return panel, panel.pop("score"), panel.pop("beta")


def _construct(panel: pd.DataFrame, score: pd.Series, beta: pd.Series, native: bool):
    """Construct the portfolio with the native solver switched on or off."""
    saved = (native_core._library, native_core._load_attempted)
    if not native:
        native_core._library, native_core._load_attempted = None, True
    try:
        return portfolio.construct_portfolio(panel, score, beta)
    finally:
        native_core._library, native_core._load_attempted = saved


def test_native_solver_builds() -> None:
    """The library compiles and loads on a machine with a C++ compiler."""
    assert native_core.available(), native_core.load_error


def test_bounded_scale_matches_python() -> None:
    """Random bounded-scaling problems give identical results."""
    rng = np.random.default_rng(11)
    for _ in range(500):
        n = int(rng.integers(1, 400))
        target = rng.random(n) * rng.choice([1e-5, 1e-2, 1.0], n)
        lower = np.full(n, 0.0002 if rng.random() < 0.5 else 0.0)
        upper = np.full(n, 0.03)
        total = float(rng.uniform(lower.sum(), min(upper.sum(), 1.0)))
        expected = portfolio._bounded_scale(
            pd.Series(target), pd.Series(lower), pd.Series(upper), total
        ).to_numpy()
        status, actual = native_core.bounded_scale(target, lower, upper, total)
        assert status == native_core.OK
        np.testing.assert_array_equal(actual, expected)


def test_portfolio_construction_matches_python() -> None:
    """Full monthly construction, including beta tilts, is bitwise identical."""
    panel, score, beta = synthetic_panel()
    native = _construct(panel, score, beta, native=True)
    python = _construct(panel, score, beta, native=False)
    assert native.monthly_diagnostics["beta_constraint_bound"].any()
    pd.testing.assert_frame_equal(native.holdings, python.holdings, check_exact=True)
    pd.testing.assert_frame_equal(
        native.monthly_diagnostics, python.monthly_diagnostics, check_exact=True
    )
    pd.testing.assert_frame_equal(
        native.sector_weights, python.sector_weights, check_exact=True
    )


if __name__ == "__main__":
    for test in (
        test_native_solver_builds,
        test_bounded_scale_matches_python,
        test_portfolio_construction_matches_python,
    ):
        test()
        print(f"passed: {test.__name__}")
