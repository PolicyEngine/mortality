"""Vectorized Bayesian mortality calculations for fast Monte Carlo simulation.

Key improvements:
1. Proper Bayesian adjustment (not naive multiplication)
2. Fully vectorized for performance
3. Handles population heterogeneity correctly
"""

from typing import Optional, Tuple, Union

import numpy as np


class VectorizedMortality:
    """Fast, vectorized mortality calculations with Bayesian adjustment.

    Designed for Monte Carlo simulation where we need to evaluate
    mortality for thousands of paths simultaneously.
    """

    def __init__(
        self,
        n_simulations: int,
        gender: Union[str, np.ndarray] = "male",
        base_rates: Optional[dict] = None,
    ):
        """Initialize vectorized mortality calculator.

        Args:
            n_simulations: Number of simulation paths
            gender: Either single gender or array of genders
            base_rates: Pre-loaded base rates (for efficiency)
        """
        self.n_sims = n_simulations

        # Handle gender
        if isinstance(gender, str):
            self.gender = np.array([gender] * n_simulations)
        else:
            self.gender = gender

        # Cache base rates
        if base_rates is None:
            from .data import get_base_rates

            self.male_rates = get_base_rates("male")
            self.female_rates = get_base_rates("female")
        else:
            self.male_rates = base_rates.get("male", {})
            self.female_rates = base_rates.get("female", {})

        # Pre-compute rate arrays for common ages
        self._precompute_base_rates()

    def _precompute_base_rates(self):
        """Pre-compute base rates for vectorized lookup."""
        ages = np.arange(0, 121)

        # Vectorized interpolation for all ages
        male_ages = sorted(self.male_rates.keys())
        male_values = [self.male_rates[a] for a in male_ages]
        self.male_rate_array = np.interp(ages, male_ages, male_values)

        female_ages = sorted(self.female_rates.keys())
        female_values = [self.female_rates[a] for a in female_ages]
        self.female_rate_array = np.interp(ages, female_ages, female_values)

    def get_base_rates_vectorized(self, ages: np.ndarray) -> np.ndarray:
        """Get base rates for array of ages and genders.

        Args:
            ages: Array of ages (n_sims,)

        Returns:
            Array of base mortality rates
        """
        # Clip ages to valid range
        ages = np.clip(ages, 0, 120).astype(int)

        # Initialize result
        rates = np.zeros(self.n_sims)

        # Vectorized lookup for males
        male_mask = self.gender == "male"
        rates[male_mask] = self.male_rate_array[ages[male_mask]]

        # Vectorized lookup for females
        female_mask = self.gender == "female"
        rates[female_mask] = self.female_rate_array[ages[female_mask]]

        return rates

    def adjust_for_characteristics(
        self,
        base_rates: np.ndarray,
        smoker: Optional[np.ndarray] = None,
        income_percentile: Optional[np.ndarray] = None,
        health_score: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply Bayesian adjustment for individual characteristics.

        Uses vectorized operations for efficiency.

        Args:
            base_rates: Base mortality rates (n_sims,)
            smoker: Boolean array of smoking status
            income_percentile: Array of income percentiles (0-100)
            health_score: Array of health scores (-1 to 1, 0 = average)

        Returns:
            Adjusted mortality rates
        """
        # Convert to log-odds for additive model
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        safe_rates = np.clip(base_rates, epsilon, 1 - epsilon)
        log_odds = np.log(safe_rates / (1 - safe_rates))

        # Smoking adjustment (vectorized)
        if smoker is not None:
            # Population parameters
            smoking_prev = 0.15
            smoking_effect = 0.59  # log(1.8)

            # Remove population average
            population_effect = smoking_prev * smoking_effect
            log_odds -= population_effect

            # Add individual effects where smoker=True
            log_odds = np.where(smoker, log_odds + smoking_effect, log_odds)

        # Income adjustment (vectorized)
        if income_percentile is not None:
            # Center around median (50th percentile)
            income_effect = -0.004 * (income_percentile - 50)
            log_odds += income_effect

        # Health adjustment (vectorized)
        if health_score is not None:
            # health_score: -1 (poor) to +1 (excellent), 0 = average
            # This is more flexible than categories
            health_effect = (
                -0.3 * health_score
            )  # Negative because good health reduces mortality

            # No population adjustment needed if centered at 0
            log_odds += health_effect

        # Convert back to probabilities (vectorized)
        adjusted_rates = 1 / (1 + np.exp(-log_odds))

        return np.clip(adjusted_rates, 0, 1)

    def simulate_survival(
        self,
        starting_age: Union[int, np.ndarray],
        max_age: int = 120,
        smoker: Optional[np.ndarray] = None,
        income_percentile: Optional[np.ndarray] = None,
        health_score: Optional[np.ndarray] = None,
        improvement_rate: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate survival paths (fully vectorized).

        Args:
            starting_age: Starting age(s)
            max_age: Maximum age to simulate to
            smoker: Smoking status for each simulation
            income_percentile: Income percentile for each simulation
            health_score: Health score for each simulation
            improvement_rate: Annual mortality improvement

        Returns:
            Tuple of (death_ages, alive_matrix)
            - death_ages: Age at death for each simulation
            - alive_matrix: Boolean matrix (n_sims, n_years)
        """
        # Handle scalar or array starting age
        if isinstance(starting_age, int):
            current_ages = np.full(self.n_sims, starting_age)
        else:
            current_ages = starting_age.copy()

        n_years = max_age - np.min(current_ages) + 1

        # Initialize matrices
        alive = np.ones((self.n_sims, n_years), dtype=bool)
        death_ages = np.full(self.n_sims, max_age, dtype=float)

        # Simulate year by year
        for year in range(n_years):
            # Only process living simulations
            still_alive = (
                alive[:, year] if year > 0 else np.ones(self.n_sims, dtype=bool)
            )

            if not np.any(still_alive):
                break

            # Get current ages for living
            current_ages_alive = current_ages + year

            # Get base rates (vectorized)
            base_rates = self.get_base_rates_vectorized(current_ages_alive)

            # Apply mortality improvements
            if improvement_rate > 0:
                improvement_factor = (1 - improvement_rate) ** year
                base_rates *= improvement_factor

            # Apply individual adjustments
            adjusted_rates = self.adjust_for_characteristics(
                base_rates,
                smoker=smoker,
                income_percentile=income_percentile,
                health_score=health_score,
            )

            # Simulate deaths (vectorized)
            random_draws = np.random.random(self.n_sims)
            deaths_this_year = (random_draws < adjusted_rates) & still_alive

            # Record death ages (with random fraction for within-year)
            death_fractions = np.random.random(self.n_sims)
            death_ages = np.where(
                deaths_this_year, current_ages_alive + death_fractions, death_ages
            )

            # Update alive matrix
            if year < n_years - 1:
                alive[deaths_this_year, year + 1 :] = False

        return death_ages, alive

    def life_expectancy_vectorized(
        self,
        ages: np.ndarray,
        smoker: Optional[np.ndarray] = None,
        income_percentile: Optional[np.ndarray] = None,
        health_score: Optional[np.ndarray] = None,
        n_simulations: int = 10000,
    ) -> np.ndarray:
        """Calculate life expectancy using Monte Carlo (fully vectorized).

        Args:
            ages: Array of current ages
            smoker: Smoking status array
            income_percentile: Income percentile array
            health_score: Health score array
            n_simulations: Number of MC simulations per person

        Returns:
            Array of life expectancies
        """
        # Expand arrays for MC simulation
        n_people = len(ages)

        # Create expanded arrays for MC
        ages_expanded = np.repeat(ages, n_simulations)

        if smoker is not None:
            smoker_expanded = np.repeat(smoker, n_simulations)
        else:
            smoker_expanded = None

        if income_percentile is not None:
            income_expanded = np.repeat(income_percentile, n_simulations)
        else:
            income_expanded = None

        if health_score is not None:
            health_expanded = np.repeat(health_score, n_simulations)
        else:
            health_expanded = None

        # Run simulation
        mort_sim = VectorizedMortality(n_people * n_simulations, self.gender[0])
        death_ages, _ = mort_sim.simulate_survival(
            ages_expanded,
            smoker=smoker_expanded,
            income_percentile=income_expanded,
            health_score=health_expanded,
        )

        # Reshape and calculate mean remaining life
        death_ages_reshaped = death_ages.reshape(n_people, n_simulations)
        remaining_life = death_ages_reshaped - ages[:, np.newaxis]
        life_expectancies = np.mean(remaining_life, axis=1)

        return life_expectancies


def benchmark_vectorized():
    """Benchmark vectorized vs scalar implementation."""
    import time

    n_sims = 10000

    print(f"Benchmarking with {n_sims} simulations...")
    print("=" * 50)

    # Vectorized version
    start = time.time()

    # Create heterogeneous population
    np.random.seed(42)
    smoker = np.random.random(n_sims) < 0.15  # 15% smokers
    income = np.random.normal(50, 20, n_sims)  # Income percentiles
    income = np.clip(income, 1, 99)
    health = np.random.normal(0, 0.5, n_sims)  # Health scores
    health = np.clip(health, -1, 1)

    mort = VectorizedMortality(n_sims)
    death_ages, alive = mort.simulate_survival(
        starting_age=65, smoker=smoker, income_percentile=income, health_score=health
    )

    vectorized_time = time.time() - start

    print(f"Vectorized implementation: {vectorized_time:.2f} seconds")
    print(f"Speed: {n_sims/vectorized_time:.0f} simulations/second")

    # Results
    print("\nResults:")
    print(f"Median death age: {np.median(death_ages):.1f}")
    print(f"10th percentile: {np.percentile(death_ages, 10):.1f}")
    print(f"90th percentile: {np.percentile(death_ages, 90):.1f}")

    # Verify population average matches SSA
    print("\n" + "=" * 50)
    print("Verification: Population average should match SSA")

    # Simple population (no adjustments)
    mort_simple = VectorizedMortality(10000)
    death_ages_simple, _ = mort_simple.simulate_survival(starting_age=65)

    life_exp_simple = np.mean(death_ages_simple - 65)
    print(f"Population life expectancy at 65: {life_exp_simple:.1f} years")
    print("Expected (SSA male): ~18 years")

    # Adjusted population should also average correctly
    life_exp_adjusted = np.mean(death_ages - 65)
    print(f"Adjusted population life expectancy: {life_exp_adjusted:.1f} years")
    print(f"Difference: {abs(life_exp_adjusted - life_exp_simple):.2f} years")


if __name__ == "__main__":
    benchmark_vectorized()
