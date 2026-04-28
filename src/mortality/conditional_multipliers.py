"""Conditional mortality multipliers that account for correlations.

The problem with independent multipliers:
- If smoking is 15% overall but 25% among low-income and 5% among high-income
- And we use centered multipliers assuming independence
- The population average won't be 1.0 anymore due to correlation

The solution: Use conditional probabilities from microdata or research.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConditionalMortalityModel:
    """Mortality model that accounts for correlations between risk factors.

    Based on research showing:
    1. Smoking rates vary by income/education (CDC BRFSS data)
    2. Health status correlates with income (Chetty et al)
    3. Exercise correlates with education and income
    4. These correlations affect the proper centering
    """

    # Smoking rates by income quintile (CDC BRFSS 2020)
    SMOKING_BY_INCOME = {
        1: 0.212,  # Bottom quintile: 21.2%
        2: 0.183,  # 18.3%
        3: 0.156,  # 15.6%
        4: 0.119,  # 11.9%
        5: 0.067,  # Top quintile: 6.7%
    }

    # Health distribution by income quintile (estimated from research)
    HEALTH_BY_INCOME = {
        1: {"excellent": 0.10, "good": 0.20, "average": 0.35, "poor": 0.35},
        2: {"excellent": 0.15, "good": 0.25, "average": 0.35, "poor": 0.25},
        3: {"excellent": 0.20, "good": 0.30, "average": 0.35, "poor": 0.15},
        4: {"excellent": 0.25, "good": 0.35, "average": 0.30, "poor": 0.10},
        5: {"excellent": 0.35, "good": 0.35, "average": 0.25, "poor": 0.05},
    }

    # Base mortality effects (before conditioning)
    BASE_EFFECTS = {
        "smoking": 1.8,
        "health": {"excellent": 0.70, "good": 0.85, "average": 1.00, "poor": 1.30},
        "income_per_decile": 0.94,  # Each decile up reduces mortality by 6%
    }

    @classmethod
    def get_income_quintile(cls, income_percentile: int) -> int:
        """Convert income percentile to quintile."""
        if income_percentile <= 20:
            return 1
        elif income_percentile <= 40:
            return 2
        elif income_percentile <= 60:
            return 3
        elif income_percentile <= 80:
            return 4
        else:
            return 5

    @classmethod
    def calculate_population_average(cls) -> float:
        """Calculate the true population average mortality multiplier.

        This accounts for all the correlations in the population.
        """
        total = 0.0

        # Iterate over income quintiles (each is 20% of population)
        for quintile in range(1, 6):
            quintile_weight = 0.20

            # Get smoking rate for this income group
            smoking_rate = cls.SMOKING_BY_INCOME[quintile]

            # Get health distribution for this income group
            health_dist = cls.HEALTH_BY_INCOME[quintile]

            # Income effect (centered on quintile 3)
            income_mult = cls.BASE_EFFECTS["income_per_decile"] ** (quintile - 3)

            # Calculate average for this quintile
            quintile_total = 0.0

            # Non-smokers in this quintile
            for health, health_prob in health_dist.items():
                mult = income_mult * cls.BASE_EFFECTS["health"][health] * 1.0
                quintile_total += (1 - smoking_rate) * health_prob * mult

            # Smokers in this quintile
            for health, health_prob in health_dist.items():
                mult = (
                    income_mult
                    * cls.BASE_EFFECTS["health"][health]
                    * cls.BASE_EFFECTS["smoking"]
                )
                quintile_total += smoking_rate * health_prob * mult

            total += quintile_weight * quintile_total

        return total

    @classmethod
    def get_calibrated_multiplier(
        cls, income_percentile: int, smoker: bool, health_status: str
    ) -> float:
        """Get mortality multiplier calibrated to population average of 1.0.

        This properly accounts for correlations in the population.
        """
        # First calculate the raw multiplier
        cls.get_income_quintile(income_percentile)

        # Income effect (each decile = 6% reduction)
        income_decile = income_percentile / 10
        income_mult = cls.BASE_EFFECTS["income_per_decile"] ** (income_decile - 5)

        # Health effect
        health_mult = cls.BASE_EFFECTS["health"][health_status]

        # Smoking effect
        smoking_mult = cls.BASE_EFFECTS["smoking"] if smoker else 1.0

        # Combined raw multiplier
        raw_mult = income_mult * health_mult * smoking_mult

        # Calibration factor (calculated once, could be cached)
        pop_avg = cls.calculate_population_average()

        # Return calibrated multiplier
        return raw_mult / pop_avg

    @classmethod
    def simulate_realistic_population(cls, n: int = 10000) -> Dict[str, np.ndarray]:
        """Simulate a population with realistic correlations.

        Returns dictionary with income, smoking, health, and multipliers.
        """
        np.random.seed(42)

        # Income distribution (uniform across percentiles)
        income_percentile = np.random.randint(1, 100, n)

        # Smoking based on income
        smoking = np.zeros(n, dtype=bool)
        for i in range(n):
            quintile = cls.get_income_quintile(income_percentile[i])
            smoking_rate = cls.SMOKING_BY_INCOME[quintile]
            smoking[i] = np.random.random() < smoking_rate

        # Health based on income (and slightly affected by smoking)
        health = []
        for i in range(n):
            quintile = cls.get_income_quintile(income_percentile[i])
            health_dist = cls.HEALTH_BY_INCOME[quintile]

            # If smoker, shift distribution toward worse health
            if smoking[i]:
                # Create adjusted distribution
                adj_dist = {
                    "excellent": health_dist["excellent"] * 0.5,
                    "good": health_dist["good"] * 0.8,
                    "average": health_dist["average"] * 1.1,
                    "poor": health_dist["poor"] * 1.6,
                }
                # Normalize
                total = sum(adj_dist.values())
                adj_dist = {k: v / total for k, v in adj_dist.items()}
            else:
                adj_dist = health_dist

            # Sample health status
            health_status = np.random.choice(
                list(adj_dist.keys()), p=list(adj_dist.values())
            )
            health.append(health_status)

        # Calculate multipliers
        multipliers = np.array(
            [
                cls.get_calibrated_multiplier(
                    income_percentile[i], smoking[i], health[i]
                )
                for i in range(n)
            ]
        )

        return {
            "income_percentile": income_percentile,
            "smoking": smoking,
            "health": np.array(health),
            "multipliers": multipliers,
        }


def demonstrate_conditional_model():
    """Demonstrate the conditional mortality model."""

    print("Conditional Mortality Model (Accounting for Correlations)")
    print("=" * 70)

    # Calculate population average
    model = ConditionalMortalityModel()
    pop_avg = model.calculate_population_average()

    print(f"\nTrue population average (with correlations): {pop_avg:.4f}")
    print("(This is NOT 1.0 because of correlations between factors)")

    # Example profiles
    print("\n\nExample Profiles (calibrated to population = 1.0):")
    print("-" * 50)

    # High risk: low income smoker with poor health
    high_risk = model.get_calibrated_multiplier(10, True, "poor")
    print("Low income (10th percentile) smoker with poor health:")
    print(f"  Mortality multiplier: {high_risk:.2f}x")
    print(f"  Expected lifespan reduction: ~{(high_risk - 1) * 10:.0f} years")

    # Low risk: high income non-smoker with excellent health
    low_risk = model.get_calibrated_multiplier(90, False, "excellent")
    print("\nHigh income (90th percentile) non-smoker with excellent health:")
    print(f"  Mortality multiplier: {low_risk:.2f}x")
    print(f"  Expected lifespan increase: ~{(1 - low_risk) * 10:.0f} years")

    # Average person
    avg_person = model.get_calibrated_multiplier(50, False, "average")
    print("\nMedian income non-smoker with average health:")
    print(f"  Mortality multiplier: {avg_person:.2f}x")

    # Simulate population
    print("\n\nPopulation Simulation (10,000 people):")
    print("-" * 50)

    pop = model.simulate_realistic_population(10000)

    print(f"Average multiplier: {np.mean(pop['multipliers']):.4f}")
    print("  (Should be exactly 1.0000 after calibration)")

    print("\nMultiplier distribution:")
    print(f"  5th percentile: {np.percentile(pop['multipliers'], 5):.3f}x")
    print(f"  25th percentile: {np.percentile(pop['multipliers'], 25):.3f}x")
    print(f"  Median: {np.median(pop['multipliers']):.3f}x")
    print(f"  75th percentile: {np.percentile(pop['multipliers'], 75):.3f}x")
    print(f"  95th percentile: {np.percentile(pop['multipliers'], 95):.3f}x")

    # Show correlations
    print("\n\nObserved Correlations in Simulated Population:")
    print("-" * 50)

    # Smoking by income
    for q in range(1, 6):
        mask = (pop["income_percentile"] > (q - 1) * 20) & (
            pop["income_percentile"] <= q * 20
        )
        smoking_rate = np.mean(pop["smoking"][mask])
        expected = ConditionalMortalityModel.SMOKING_BY_INCOME[q]
        print(f"Quintile {q} smoking: {smoking_rate:.1%} (expected: {expected:.1%})")

    print("\nHealth status by smoking:")
    for health in ["excellent", "good", "average", "poor"]:
        rate_smokers = np.mean(pop["health"][pop["smoking"]] == health)
        rate_nonsmokers = np.mean(pop["health"][~pop["smoking"]] == health)
        print(
            f"  {health}: {rate_smokers:.1%} (smokers) vs {rate_nonsmokers:.1%} (non-smokers)"
        )

    print("\nMortality multiplier by group:")
    # Low income smokers
    mask = (pop["income_percentile"] <= 20) & pop["smoking"]
    if np.any(mask):
        print(f"  Low income smokers: {np.mean(pop['multipliers'][mask]):.2f}x")

    # High income non-smokers
    mask = (pop["income_percentile"] >= 80) & ~pop["smoking"]
    if np.any(mask):
        print(f"  High income non-smokers: {np.mean(pop['multipliers'][mask]):.2f}x")

    # Verify calibration
    print("\n" + "=" * 70)
    print("CALIBRATION CHECK:")
    overall_mean = np.mean(pop["multipliers"])
    if abs(overall_mean - 1.0) < 0.001:
        print(f"✓ Population mean is {overall_mean:.6f} (correctly calibrated to 1.0)")
    else:
        print(f"✗ Population mean is {overall_mean:.6f} (should be 1.0)")


if __name__ == "__main__":
    demonstrate_conditional_model()
