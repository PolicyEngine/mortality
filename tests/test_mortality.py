"""Tests for basic mortality calculations."""

import numpy as np

from mortality import Mortality, MortalityFactors


class TestBasicMortality:
    """Test basic mortality calculations without factors."""

    def test_create_mortality_object(self):
        """Should create a basic mortality object."""
        mort = Mortality()
        assert mort is not None
        assert mort.gender == "male"

    def test_death_probability_at_65(self):
        """Death probability at 65 should be reasonable."""
        mort = Mortality(gender="male")
        q65 = mort.q(65)

        # SSA 2021: Male q65 ≈ 0.01604
        assert 0.015 < q65 < 0.017

    def test_female_lower_mortality(self):
        """Females should have lower mortality than males."""
        mort_male = Mortality(gender="male")
        mort_female = Mortality(gender="female")

        # Test at various ages
        for age in [30, 50, 65, 80]:
            assert mort_female.q(age) < mort_male.q(age)

    def test_mortality_increases_with_age(self):
        """Mortality should increase with age."""
        mort = Mortality()

        q50 = mort.q(50)
        q60 = mort.q(60)
        q70 = mort.q(70)
        q80 = mort.q(80)

        assert q50 < q60 < q70 < q80

    def test_survival_probability(self):
        """Survival probability should be 1 - death probability."""
        mort = Mortality()

        for age in [40, 60, 80]:
            assert abs(mort.p(age) + mort.q(age) - 1.0) < 0.0001

    def test_life_expectancy_reasonable(self):
        """Life expectancy should be in reasonable range."""
        mort = Mortality(gender="male")
        e65 = mort.e(65)

        # Male life expectancy at 65: typically 17-19 years
        assert 16 < e65 < 20

        # Female should be higher
        mort_female = Mortality(gender="female")
        e65_female = mort_female.e(65)
        assert 19 < e65_female < 23
        assert e65_female > e65


class TestMortalityFactors:
    """Test personal mortality factors."""

    def test_default_factors(self):
        """Default factors should give multiplier of 1.0."""
        factors = MortalityFactors()
        assert abs(factors.get_multiplier() - 1.0) < 0.01

    def test_good_health_reduces_mortality(self):
        """Good health should reduce mortality."""
        factors_good = MortalityFactors(health="good")
        factors_avg = MortalityFactors(health="average")

        assert factors_good.get_multiplier() < factors_avg.get_multiplier()
        assert factors_good.get_multiplier() < 1.0

    def test_smoking_increases_mortality(self):
        """Smoking should significantly increase mortality."""
        factors_smoker = MortalityFactors(smoker=True)
        factors_nonsmoker = MortalityFactors(smoker=False)

        mult_smoker = factors_smoker.get_multiplier()
        mult_nonsmoker = factors_nonsmoker.get_multiplier()

        # Smoking roughly doubles mortality
        assert 1.5 < mult_smoker < 2.0
        assert mult_smoker > mult_nonsmoker

    def test_high_income_reduces_mortality(self):
        """High income should reduce mortality."""
        factors_rich = MortalityFactors(income_percentile=95)
        factors_poor = MortalityFactors(income_percentile=10)

        assert factors_rich.get_multiplier() < factors_poor.get_multiplier()
        assert factors_rich.get_multiplier() < 1.0
        assert factors_poor.get_multiplier() > 1.0

    def test_combined_factors(self):
        """Multiple factors should combine multiplicatively."""
        # Healthy wealthy person
        factors_good = MortalityFactors(
            health="excellent",
            income_percentile=90,
            education="graduate",
            exercise="vigorous",
        )

        # Unhealthy poor person
        factors_bad = MortalityFactors(
            health="poor", income_percentile=10, smoker=True, exercise="none"
        )

        mult_good = factors_good.get_multiplier()
        mult_bad = factors_bad.get_multiplier()

        # Should be dramatically different
        assert mult_good < 0.5  # Less than half average mortality
        assert mult_bad > 2.0  # More than double average mortality


class TestMortalityWithFactors:
    """Test mortality calculations with personal factors."""

    def test_factors_affect_death_probability(self):
        """Personal factors should affect death probability."""
        mort_healthy = Mortality(
            gender="male", factors=MortalityFactors(health="excellent")
        )
        mort_average = Mortality(gender="male")

        # Healthy person should have lower mortality
        assert mort_healthy.q(65) < mort_average.q(65)

    def test_factors_affect_life_expectancy(self):
        """Personal factors should affect life expectancy."""
        mort_healthy = Mortality(
            gender="female",
            factors=MortalityFactors(health="excellent", income_percentile=90),
        )
        mort_average = Mortality(gender="female")

        # Healthy wealthy person should live longer
        assert mort_healthy.e(65) > mort_average.e(65)

        # Should be a meaningful difference (several years)
        assert mort_healthy.e(65) - mort_average.e(65) > 2


class TestMortalityProjections:
    """Test mortality improvements over time."""

    def test_mortality_improves_over_time(self):
        """Future mortality should be lower than current."""
        mort = Mortality(improvement_rate=0.01)  # 1% annual improvement

        q_now = mort.q(65, years_forward=0)
        q_10yr = mort.q(65, years_forward=10)
        q_20yr = mort.q(65, years_forward=20)

        assert q_10yr < q_now
        assert q_20yr < q_10yr

        # Should be roughly 1% per year
        assert abs(q_10yr / q_now - 0.9) < 0.02

    def test_no_improvement_option(self):
        """Should be able to turn off improvements."""
        mort = Mortality(improvement_rate=0.0)

        q_now = mort.q(65, years_forward=0)
        q_10yr = mort.q(65, years_forward=10)

        assert abs(q_now - q_10yr) < 0.0001


class TestSurvivalCurve:
    """Test survival curve calculations."""

    def test_survival_starts_at_one(self):
        """Survival curve should start at 100%."""
        mort = Mortality()
        survival = mort.survival(65, 95)

        assert survival[0] == 1.0

    def test_survival_decreases(self):
        """Survival should decrease monotonically."""
        mort = Mortality()
        survival = mort.survival(65, 95)

        for i in range(1, len(survival)):
            assert survival[i] <= survival[i - 1]

    def test_survival_at_old_age(self):
        """Few should survive to very old age."""
        mort = Mortality(gender="male")
        survival = mort.survival(65, 105)

        # Very few survive to 105
        assert survival[-1] < 0.01


class TestMonteCarlo:
    """Test Monte Carlo simulation."""

    def test_simulate_returns_ages(self):
        """Simulation should return death ages."""
        mort = Mortality()
        death_ages = mort.simulate(age=65, n_sims=100)

        assert len(death_ages) == 100
        assert all(age >= 65 for age in death_ages)
        assert all(age <= 120 for age in death_ages)

    def test_simulate_median_reasonable(self):
        """Simulated median should match life expectancy roughly."""
        mort = Mortality(gender="male")
        death_ages = mort.simulate(age=65, n_sims=1000)

        median_death = np.median(death_ages)
        life_exp = mort.e(65) + 65

        # Should be within a few years
        assert abs(median_death - life_exp) < 3

    def test_simulate_variability(self):
        """Simulations should show realistic variability."""
        mort = Mortality()
        death_ages = mort.simulate(age=65, n_sims=1000)

        pct_10 = np.percentile(death_ages, 10)
        pct_90 = np.percentile(death_ages, 90)

        # Should have meaningful spread
        assert pct_90 - pct_10 > 15  # At least 15 year spread
        assert pct_10 > 70  # Most live past 70
        assert pct_90 < 100  # Most die before 100
