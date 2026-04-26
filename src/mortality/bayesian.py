"""Proper Bayesian mortality adjustment.

The key insight: SSA tables are population averages that ALREADY include
the effects of smoking, income, education, etc. We need to adjust from
the population average to the individual, not compound effects.
"""

import numpy as np
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from .data import get_base_rates


@dataclass
class BayesianMortalityAdjustment:
    """Bayesian adjustment from population average to individual.

    The problem with naive multiplication:
    - SSA rate for 65-year-old male: 0.016
    - This ALREADY includes: 20% smokers, various incomes, etc.
    - If we multiply by 1.8 for a smoker, we get 0.029
    - But that's wrong! We should get the conditional probability
      P(death | age=65, smoker=true) not P(death | age=65) × P(smoker effect)

    The Bayesian approach:
    1. SSA rate is our prior: P(death | age)
    2. We have information about subpopulation rates
    3. We update: P(death | age, characteristics)
    """

    gender: Literal["male", "female"] = "male"

    # Individual characteristics
    smoker: Optional[bool] = None
    income_percentile: Optional[int] = None
    health: Optional[Literal["excellent", "good", "average", "poor"]] = None
    education: Optional[
        Literal["high_school", "some_college", "bachelors", "graduate"]
    ] = None

    def get_adjusted_rate(self, age: int, base_rate: Optional[float] = None) -> float:
        """Get mortality rate adjusted for individual characteristics.

        This uses a log-linear model inspired by Cox proportional hazards:
        log(hazard) = log(baseline) + β₁X₁ + β₂X₂ + ...

        But critically, we calibrate so the population average equals SSA.
        """
        if base_rate is None:
            base_rates = get_base_rates(self.gender)
            ages = sorted(base_rates.keys())
            rates = [base_rates[a] for a in ages]
            base_rate = float(np.interp(age, ages, rates))

        # Start with log hazard
        log_hazard = np.log(base_rate / (1 - base_rate))  # Logit transform

        # Smoking adjustment
        # Population: ~15% smoke, they have 80% higher mortality
        # So we need to SUBTRACT the population average effect first
        if self.smoker is not None:
            smoking_prevalence = 0.15
            smoking_effect = 0.59  # log(1.8)

            # Remove population average effect (already in SSA)
            population_avg_effect = smoking_prevalence * smoking_effect
            log_hazard -= population_avg_effect

            # Add individual effect
            if self.smoker:
                log_hazard += smoking_effect

        # Income adjustment
        # SSA reflects median income (~50th percentile)
        if self.income_percentile is not None:
            # Log-linear effect: top 1% live 15 years longer than bottom 1%
            # This translates to about 0.4 log-hazard units per percentile decile
            income_effect = -0.004 * (self.income_percentile - 50)
            log_hazard += income_effect

        # Health status adjustment
        # SSA reflects average health
        if self.health is not None:
            health_effects = {
                "excellent": -0.35,  # log(0.7)
                "good": -0.16,  # log(0.85)
                "average": 0.0,
                "poor": 0.26,  # log(1.3)
            }

            # SSA is weighted average: assume 20% excellent, 30% good, 30% average, 20% poor
            population_avg = (
                0.2 * health_effects["excellent"]
                + 0.3 * health_effects["good"]
                + 0.3 * health_effects["average"]
                + 0.2 * health_effects["poor"]
            )

            log_hazard -= population_avg
            log_hazard += health_effects[self.health]

        # Education adjustment
        # SSA reflects ~30% bachelor's, ~20% graduate, ~30% some college, ~20% HS
        if self.education is not None:
            education_effects = {
                "high_school": 0.14,  # log(1.15)
                "some_college": 0.05,  # log(1.05)
                "bachelors": -0.08,  # log(0.92)
                "graduate": -0.13,  # log(0.88)
            }

            population_avg = (
                0.2 * education_effects["high_school"]
                + 0.3 * education_effects["some_college"]
                + 0.3 * education_effects["bachelors"]
                + 0.2 * education_effects["graduate"]
            )

            log_hazard -= population_avg
            log_hazard += education_effects[self.education]

        # Convert back to probability
        # Using logistic function: p = exp(log_odds) / (1 + exp(log_odds))
        adjusted_rate = np.exp(log_hazard) / (1 + np.exp(log_hazard))

        # Ensure valid probability
        return np.clip(adjusted_rate, 0.0, 1.0)

    def explain_adjustment(self, age: int) -> str:
        """Explain how the adjustment works."""
        base_rates = get_base_rates(self.gender)
        ages = sorted(base_rates.keys())
        rates = [base_rates[a] for a in ages]
        base_rate = float(np.interp(age, ages, rates))
        adjusted_rate = self.get_adjusted_rate(age, base_rate)

        explanation = f"""
Bayesian Mortality Adjustment for {age}-year-old {self.gender}
========================================================

Population average (SSA): {base_rate:.4f} ({base_rate*100:.2f}%)
Your adjusted rate: {adjusted_rate:.4f} ({adjusted_rate*100:.2f}%)
Ratio: {adjusted_rate/base_rate:.2f}x

Why not simple multiplication?
-------------------------------
The SSA rate of {base_rate:.4f} ALREADY includes:
- ~15% smokers (with 1.8x mortality)
- Full income distribution
- Mix of health statuses
- Various education levels

If we naively multiplied:
- Smoker: {base_rate:.4f} × 1.8 = {base_rate*1.8:.4f} ❌ (too high!)

Instead, we use Bayesian updating:
1. Remove the population average effect
2. Add your individual effect
3. Result: {adjusted_rate:.4f} ✓

This ensures that if we average over the whole population,
we get back to the SSA rate (as we should).
"""
        return explanation


def compare_approaches():
    """Show why Bayesian is better than naive multiplication."""

    # 65-year-old male smoker
    age = 65
    base_rates = get_base_rates("male")
    base_rate = base_rates[age] if age in base_rates else 0.016

    print(f"Base SSA rate for 65-year-old male: {base_rate:.4f}")
    print("=" * 50)

    # Naive approach (what we were doing before)
    naive_smoker_rate = base_rate * 1.8
    print(f"\nNaive approach (multiply by 1.8): {naive_smoker_rate:.4f}")
    print("Problem: This assumes base rate is for non-smokers!")

    # Bayesian approach
    bayes = BayesianMortalityAdjustment(gender="male", smoker=True)
    bayes_smoker_rate = bayes.get_adjusted_rate(age)
    print(f"\nBayesian approach: {bayes_smoker_rate:.4f}")
    print("This accounts for smokers already being in the base rate")

    # Check population average
    print("\n" + "=" * 50)
    print("Verification: Population average should equal SSA")

    # Simulate population
    smoking_prev = 0.15
    non_smoker = BayesianMortalityAdjustment(gender="male", smoker=False)
    smoker = BayesianMortalityAdjustment(gender="male", smoker=True)

    pop_avg = (1 - smoking_prev) * non_smoker.get_adjusted_rate(
        age
    ) + smoking_prev * smoker.get_adjusted_rate(age)

    print(f"Population average: {pop_avg:.4f}")
    print(f"SSA rate: {base_rate:.4f}")
    print(f"Difference: {abs(pop_avg - base_rate):.6f} (should be ~0)")


if __name__ == "__main__":
    compare_approaches()

    print("\n" + "=" * 50)
    print("Example adjustment:")

    adjuster = BayesianMortalityAdjustment(
        gender="male",
        smoker=False,
        income_percentile=75,
        health="good",
        education="graduate",
    )

    print(adjuster.explain_adjustment(65))
