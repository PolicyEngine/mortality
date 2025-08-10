"""Mortality data management with automatic fetching and caching."""

from typing import Dict, Literal
from functools import lru_cache
from .fetchers import fetch_ssa_table


@lru_cache(maxsize=2)
def get_base_rates(gender: Literal["male", "female"]) -> Dict[int, float]:
    """Get base mortality rates, fetching from SSA if needed.
    
    Args:
        gender: Biological sex
        
    Returns:
        Dictionary mapping age to annual mortality rate (qx)
    """
    male_rates, female_rates = fetch_ssa_table()
    
    if gender == "male":
        return male_rates
    else:
        return female_rates