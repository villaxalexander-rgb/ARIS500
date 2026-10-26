"""Define point-in-time stock signals and their common standardization.

Signal functions read an in-memory panel and write no files. Every signal uses
the row as already lagged by panel construction and never looks forward;
forward-return alignment is centralized in ``src.utils`` for evaluators and
backtests. New blocks should start from ``src/signals/_template.py``.
"""
