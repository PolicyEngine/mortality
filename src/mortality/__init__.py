"""mortality: Practical mortality calculations for Python.

Simple, clean interface to SSA mortality tables with basic calculations.
Future versions will add more sophisticated mortality modeling.
"""

__version__ = "0.1.0"

from .simple import (
    get_mortality_rate,
    get_life_expectancy,
    simulate_survival,
    MortalityTable
)

__all__ = [
    "get_mortality_rate",
    "get_life_expectancy",
    "simulate_survival",
    "MortalityTable"
]