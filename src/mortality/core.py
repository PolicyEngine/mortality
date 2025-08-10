"""Core mortality calculations."""

import numpy as np
from typing import Literal, Optional
from functools import lru_cache

from .factors import MortalityFactors
from .data import get_base_rates


class Mortality:
    """Simple mortality calculator with optional personal factors.
    
    Examples:
        >>> mort = Mortality()
        >>> mort.q(65)  # Death probability at 65
        0.01604
        
        >>> mort = Mortality(gender="female")
        >>> mort.e(65)  # Life expectancy at 65
        20.5
    """
    
    def __init__(
        self,
        gender: Literal["male", "female"] = "male",
        factors: Optional[MortalityFactors] = None,
        improvement_rate: float = 0.01,
        base_year: int = 2021
    ):
        """Initialize mortality calculator.
        
        Args:
            gender: Biological sex for base mortality rates
            factors: Personal factors affecting mortality
            improvement_rate: Annual mortality improvement (0.01 = 1% per year)
            base_year: Year of base mortality table
        """
        self.gender = gender
        self.factors = factors or MortalityFactors(gender=gender)
        self.improvement_rate = improvement_rate
        self.base_year = base_year
        self._base_rates = get_base_rates(gender)
    
    @lru_cache(maxsize=1000)
    def q(self, age: int, years_forward: int = 0) -> float:
        """Probability of death within one year (qx).
        
        Args:
            age: Current age
            years_forward: Years in the future (for projections)
            
        Returns:
            Probability of death before next birthday
        """
        # Get base rate via interpolation
        ages = sorted(self._base_rates.keys())
        rates = [self._base_rates[a] for a in ages]
        base_q = float(np.interp(age, ages, rates))
        
        # Apply mortality improvements
        if years_forward > 0 and self.improvement_rate > 0:
            improvement_factor = (1 - self.improvement_rate) ** years_forward
            base_q *= improvement_factor
        
        # Apply personal factors
        multiplier = self.factors.get_multiplier()
        adjusted_q = base_q * multiplier
        
        # Ensure valid probability
        return min(1.0, max(0.0, adjusted_q))
    
    def p(self, age: int, years_forward: int = 0) -> float:
        """Probability of survival for one year (px).
        
        Args:
            age: Current age
            years_forward: Years in the future
            
        Returns:
            Probability of surviving to next birthday
        """
        return 1.0 - self.q(age, years_forward)
    
    def survival(self, from_age: int, to_age: int) -> np.ndarray:
        """Calculate survival curve between two ages.
        
        Args:
            from_age: Starting age
            to_age: Ending age (inclusive)
            
        Returns:
            Array of survival probabilities for each age
        """
        ages = np.arange(from_age, to_age + 1)
        survival = np.ones(len(ages))
        
        for i in range(1, len(ages)):
            # Account for mortality improvements
            years_forward = i - 1
            current_age = ages[i - 1]
            
            # Multiply by survival probability
            survival[i] = survival[i - 1] * self.p(current_age, years_forward)
        
        return survival
    
    def e(self, age: int, max_age: int = 120) -> float:
        """Life expectancy at given age.
        
        Args:
            age: Current age
            max_age: Maximum possible age
            
        Returns:
            Expected remaining years of life
        """
        survival = self.survival(age, max_age)
        
        # Use trapezoidal rule for integration
        # Each year survived contributes to life expectancy
        return np.trapz(survival, dx=1.0)
    
    def simulate(
        self,
        age: int,
        n_sims: int = 1000,
        max_age: int = 120
    ) -> np.ndarray:
        """Simulate death ages using Monte Carlo.
        
        Args:
            age: Starting age
            n_sims: Number of simulations
            max_age: Maximum possible age
            
        Returns:
            Array of simulated death ages
        """
        death_ages = np.full(n_sims, float(max_age))
        
        for sim in range(n_sims):
            for current_age in range(age, max_age):
                years_forward = current_age - age
                
                # Check if death occurs this year
                if np.random.random() < self.q(current_age, years_forward):
                    # Add random fraction for within-year death
                    death_ages[sim] = current_age + np.random.random()
                    break
        
        return death_ages