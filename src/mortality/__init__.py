"""mortality: Practical mortality calculations for Python."""

__version__ = "0.1.0"

from .core import Mortality
from .factors import MortalityFactors

__all__ = ["Mortality", "MortalityFactors"]