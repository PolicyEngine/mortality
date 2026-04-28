"""mortality: Practical mortality calculations for Python.

Simple, clean interface to SSA mortality tables with basic calculations.
Future versions will add more sophisticated mortality modeling.
"""

__version__ = "0.1.0"

from .core import Mortality
from .factors import MortalityFactors
from .simple import (
    MortalityTable,
    get_life_expectancy,
    get_mortality_rate,
    simulate_survival,
)

__all__ = [
    "get_mortality_rate",
    "get_life_expectancy",
    "simulate_survival",
    "MortalityTable",
    "Mortality",
    "MortalityFactors",
]
