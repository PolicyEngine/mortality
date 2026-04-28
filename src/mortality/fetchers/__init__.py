"""Data fetchers for mortality tables from various sources."""

from .ssa import SSATableFetcher, fetch_ssa_table

__all__ = ["fetch_ssa_table", "SSATableFetcher"]
