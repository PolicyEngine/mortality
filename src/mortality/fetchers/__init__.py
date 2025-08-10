"""Data fetchers for mortality tables from various sources."""

from .ssa import fetch_ssa_table, SSATableFetcher

__all__ = ["fetch_ssa_table", "SSATableFetcher"]