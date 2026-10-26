"""Expose composite scoring, portfolio construction, and model diagnostics.

Model modules read local panels and factors from ``data/``; research functions
are read-only except explicit sensitivity exports and monthly rebalance CSVs.
The production composite deliberately excludes tested risk and fundamental-
momentum blocks, as documented in ``config.py`` and the signal modules.
"""

from src.model.composite import (
    build_block_scores,
    build_composite_scores,
    build_model_score,
    run_composite_research,
)

__all__ = [
    "build_block_scores",
    "build_composite_scores",
    "build_model_score",
    "run_composite_research",
]
