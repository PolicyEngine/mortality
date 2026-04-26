"""Simple mortality calculations using SSA tables.

This is the v1 implementation focused on providing clean, easy access
to SSA mortality data with basic calculations.
"""

from functools import lru_cache
from typing import Literal, Optional, Tuple

import numpy as np

from .fetchers import fetch_ssa_table


class MortalityTable:
    """Simple interface to SSA mortality tables."""

    def __init__(self, gender: Literal["male", "female"] = "male"):
        """Initialize with SSA mortality data.

        Args:
            gender: Biological sex for mortality rates
        """
        self.gender = gender
        self._load_data()

    def _load_data(self):
        """Load SSA mortality data."""
        male_rates, female_rates = fetch_ssa_table()
        if self.gender == "male":
            self.rates = male_rates
        else:
            self.rates = female_rates

    def get_rate(self, age: int) -> float:
        """Get mortality rate for a specific age.

        Args:
            age: Age in years

        Returns:
            Annual mortality probability (qx)
        """
        if age in self.rates:
            return self.rates[age]

        # Interpolate if needed
        ages = sorted(self.rates.keys())
        if age < ages[0]:
            return self.rates[ages[0]]
        if age > ages[-1]:
            return 1.0  # Certain death past max age

        # Linear interpolation
        rates_array = [self.rates[a] for a in ages]
        return float(np.interp(age, ages, rates_array))

    def life_expectancy(self, age: int) -> float:
        """Calculate remaining life expectancy.

        Args:
            age: Current age

        Returns:
            Expected remaining years of life
        """
        max_age = 120
        survival_prob = 1.0
        total_years = 0.0

        for future_age in range(age, max_age + 1):
            # Add the probability of surviving to this age
            total_years += survival_prob

            # Update survival probability for next year
            if future_age < max_age:
                mortality_rate = self.get_rate(future_age)
                survival_prob *= 1 - mortality_rate

        # Subtract 1 because we include the current year
        return total_years - 1

    def survival_curve(self, start_age: int, end_age: int) -> np.ndarray:
        """Calculate survival probabilities from start_age to end_age.

        Args:
            start_age: Starting age
            end_age: Ending age (inclusive)

        Returns:
            Array of cumulative survival probabilities
        """
        n_years = end_age - start_age + 1
        survival_probs = np.ones(n_years)

        cumulative_survival = 1.0
        for i in range(1, n_years):
            age = start_age + i - 1
            mort_rate = self.get_rate(age)
            cumulative_survival *= 1 - mort_rate
            survival_probs[i] = cumulative_survival

        return survival_probs


@lru_cache(maxsize=4)
def _get_table(gender: Literal["male", "female"]) -> MortalityTable:
    """Get cached mortality table."""
    return MortalityTable(gender)


def get_mortality_rate(age: int, gender: Literal["male", "female"] = "male") -> float:
    """Get mortality rate for a specific age and gender.

    Args:
        age: Age in years
        gender: Biological sex

    Returns:
        Annual mortality probability (qx)

    Example:
        >>> rate = get_mortality_rate(65, "male")
        >>> print(f"65-year-old male mortality rate: {rate:.4f}")
        65-year-old male mortality rate: 0.0160
    """
    table = _get_table(gender)
    return table.get_rate(age)


def get_life_expectancy(age: int, gender: Literal["male", "female"] = "male") -> float:
    """Calculate remaining life expectancy.

    Args:
        age: Current age
        gender: Biological sex

    Returns:
        Expected remaining years of life

    Example:
        >>> le = get_life_expectancy(65, "female")
        >>> print(f"65-year-old female life expectancy: {le:.1f} years")
        65-year-old female life expectancy: 20.2 years
    """
    table = _get_table(gender)
    return table.life_expectancy(age)


def simulate_survival(
    starting_age: int,
    n_simulations: int = 1000,
    gender: Literal["male", "female"] = "male",
    max_age: int = 120,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate survival paths using Monte Carlo.

    Args:
        starting_age: Age at start of simulation
        n_simulations: Number of Monte Carlo paths
        gender: Biological sex
        max_age: Maximum age to simulate to
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (death_ages, alive_matrix)
        - death_ages: Age at death for each simulation
        - alive_matrix: Boolean matrix (n_sims, n_years) of survival

    Example:
        >>> death_ages, alive = simulate_survival(65, 10000, "male")
        >>> life_exp = np.mean(death_ages - 65)
        >>> print(f"Simulated life expectancy: {life_exp:.1f} years")
        Simulated life expectancy: 17.2 years
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    table = _get_table(gender)
    n_years = max_age - starting_age + 1

    # Initialize arrays
    alive = np.ones((n_simulations, n_years), dtype=bool)
    death_ages = np.full(n_simulations, float(max_age))

    # Simulate year by year
    for year in range(n_years):
        age = starting_age + year

        # Get mortality rate for this age
        mort_rate = table.get_rate(age)

        # Only check mortality for those still alive
        still_alive = alive[:, year] if year > 0 else np.ones(n_simulations, dtype=bool)
        n_alive = np.sum(still_alive)

        if n_alive == 0:
            break

        # Simulate deaths
        random_draws = np.random.random(n_simulations)
        deaths_this_year = (random_draws < mort_rate) & still_alive

        # Record death ages (with random fraction for within-year)
        death_fractions = np.random.random(n_simulations)
        death_ages[deaths_this_year] = age + death_fractions[deaths_this_year]

        # Update alive matrix
        if year < n_years - 1:
            alive[deaths_this_year, year + 1 :] = False

    return death_ages, alive


def compare_genders():
    """Compare mortality between males and females."""

    ages = [0, 20, 40, 60, 65, 70, 80, 90, 100]

    print("SSA Mortality Table Comparison")
    print("=" * 60)
    print(
        f"{'Age':<5} {'Male Rate':<12} {'Female Rate':<12} {'Male LE':<10} {'Female LE':<10}"
    )
    print("-" * 60)

    for age in ages:
        male_rate = get_mortality_rate(age, "male")
        female_rate = get_mortality_rate(age, "female")
        male_le = get_life_expectancy(age, "male")
        female_le = get_life_expectancy(age, "female")

        print(
            f"{age:<5} {male_rate:<12.4f} {female_rate:<12.4f} {male_le:<10.1f} {female_le:<10.1f}"
        )


if __name__ == "__main__":
    compare_genders()

    print("\n\nMonte Carlo Simulation (10,000 paths)")
    print("=" * 60)

    for gender in ["male", "female"]:
        death_ages, _ = simulate_survival(65, 10000, gender, random_seed=42)
        life_exp = np.mean(death_ages - 65)
        p10 = np.percentile(death_ages - 65, 10)
        p90 = np.percentile(death_ages - 65, 90)

        print(f"\n{gender.capitalize()} at age 65:")
        print(f"  Life expectancy: {life_exp:.1f} years")
        print(f"  10th percentile: {p10:.1f} years")
        print(f"  90th percentile: {p90:.1f} years")
