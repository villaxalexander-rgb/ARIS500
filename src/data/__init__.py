"""Contain WRDS extraction and point-in-time panel construction stages.

Pull modules read WRDS and caches and write documented Parquet files in
``data/``; ``build_panel`` reads those local files and writes universe-specific
panels. Imports do not query WRDS—the caller must invoke a module ``main``.
"""
