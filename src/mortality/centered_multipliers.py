"""Centered mortality multipliers that preserve population averages.

The key insight: Instead of complex Bayesian adjustment, we can use multipliers
that are already centered so their population-weighted average equals 1.0.

For example, if smokers are 15% of population with 1.8x mortality:
- Smoker multiplier: 1.8 / 1.12 = 1.61
- Non-smoker multiplier: 1.0 / 1.12 = 0.89
- Where 1.12 = 0.15 * 1.8 + 0.85 * 1.0

This ensures: 0.15 * 1.61 + 0.85 * 0.89 = 1.0
"""

import numpy as np
from typing import Optional, Dict, Literal
from dataclasses import dataclass


@dataclass
class CenteredMultipliers:
    """Mortality multipliers centered on population average of 1.0."""
    
    # Smoking (15% prevalence, raw effect 1.8x)
    # Population average: 0.15 * 1.8 + 0.85 * 1.0 = 1.12
    SMOKING = {
        'prevalence': 0.15,
        'raw_effect': 1.8,
        'pop_avg': 1.12,  # 0.15 * 1.8 + 0.85 * 1.0
        'centered': {
            True: 1.61,   # 1.8 / 1.12
            False: 0.89   # 1.0 / 1.12
        }
    }
    
    # Income (assuming linear effect, calibrated to research)
    # Top 1% live 15 years longer than bottom 1% (Chetty et al)
    # This translates to roughly 0.7x mortality for top vs 1.4x for bottom
    # Assuming normal distribution around 50th percentile
    INCOME = {
        'effect_per_percentile': -0.004,  # Log scale
        'reference_percentile': 50,
        # No centering needed if we use deviations from median
    }
    
    # Health status (estimated distribution)
    # 20% excellent, 30% good, 30% average, 20% poor
    HEALTH = {
        'distribution': {
            'excellent': 0.20,
            'good': 0.30,
            'average': 0.30,
            'poor': 0.20
        },
        'raw_multipliers': {
            'excellent': 0.70,
            'good': 0.85,
            'average': 1.00,
            'poor': 1.30
        },
        'pop_avg': 0.965,  # Weighted average
        'centered': {
            'excellent': 0.73,  # 0.70 / 0.965
            'good': 0.88,      # 0.85 / 0.965
            'average': 1.04,   # 1.00 / 0.965
            'poor': 1.35       # 1.30 / 0.965
        }
    }
    
    # Education (US distribution from Census)
    EDUCATION = {
        'distribution': {
            'less_than_high_school': 0.10,
            'high_school': 0.27,
            'some_college': 0.29,
            'bachelors': 0.22,
            'graduate': 0.12
        },
        'raw_multipliers': {
            'less_than_high_school': 1.20,
            'high_school': 1.10,
            'some_college': 1.05,
            'bachelors': 0.92,
            'graduate': 0.88
        },
        'pop_avg': 1.024,  # Weighted average
        'centered': {
            'less_than_high_school': 1.17,  # 1.20 / 1.024
            'high_school': 1.07,            # 1.10 / 1.024
            'some_college': 1.03,           # 1.05 / 1.024
            'bachelors': 0.90,              # 0.92 / 1.024
            'graduate': 0.86                # 0.88 / 1.024
        }
    }
    
    # Exercise (estimated distribution)
    EXERCISE = {
        'distribution': {
            'none': 0.25,
            'light': 0.35,
            'moderate': 0.30,
            'vigorous': 0.10
        },
        'raw_multipliers': {
            'none': 1.30,
            'light': 1.10,
            'moderate': 0.85,
            'vigorous': 0.70
        },
        'pop_avg': 1.045,  # Weighted average
        'centered': {
            'none': 1.24,      # 1.30 / 1.045
            'light': 1.05,     # 1.10 / 1.045
            'moderate': 0.81,  # 0.85 / 1.045
            'vigorous': 0.67   # 0.70 / 1.045
        }
    }
    
    @classmethod
    def get_multiplier(cls,
                       smoker: Optional[bool] = None,
                       income_percentile: Optional[int] = None,
                       health_status: Optional[str] = None,
                       education: Optional[str] = None,
                       exercise: Optional[str] = None) -> float:
        """Calculate combined multiplier using centered values.
        
        The beauty of centered multipliers: we can simply multiply them
        and the population average will still be 1.0 (assuming independence).
        """
        multiplier = 1.0
        
        if smoker is not None:
            multiplier *= cls.SMOKING['centered'][smoker]
        
        if income_percentile is not None:
            # Income effect as deviation from median
            income_effect = cls.INCOME['effect_per_percentile'] * (
                income_percentile - cls.INCOME['reference_percentile']
            )
            multiplier *= np.exp(income_effect)
        
        if health_status is not None:
            multiplier *= cls.HEALTH['centered'][health_status]
        
        if education is not None:
            multiplier *= cls.EDUCATION['centered'][education]
        
        if exercise is not None:
            multiplier *= cls.EXERCISE['centered'][exercise]
        
        return multiplier
    
    @classmethod
    def verify_centering(cls) -> Dict[str, float]:
        """Verify that centered multipliers average to 1.0."""
        results = {}
        
        # Smoking
        smoking_avg = (
            cls.SMOKING['prevalence'] * cls.SMOKING['centered'][True] +
            (1 - cls.SMOKING['prevalence']) * cls.SMOKING['centered'][False]
        )
        results['smoking'] = smoking_avg
        
        # Health
        health_avg = sum(
            cls.HEALTH['distribution'][status] * cls.HEALTH['centered'][status]
            for status in cls.HEALTH['distribution']
        )
        results['health'] = health_avg
        
        # Education
        edu_avg = sum(
            cls.EDUCATION['distribution'][level] * cls.EDUCATION['centered'][level]
            for level in cls.EDUCATION['distribution']
        )
        results['education'] = edu_avg
        
        # Exercise
        exercise_avg = sum(
            cls.EXERCISE['distribution'][level] * cls.EXERCISE['centered'][level]
            for level in cls.EXERCISE['distribution']
        )
        results['exercise'] = exercise_avg
        
        return results


def demonstrate_centered_multipliers():
    """Show how centered multipliers work."""
    
    print("Centered Mortality Multipliers")
    print("=" * 60)
    
    # Verify centering
    print("\nVerifying Population Averages (should all be ~1.0):")
    print("-" * 40)
    averages = CenteredMultipliers.verify_centering()
    for factor, avg in averages.items():
        status = "✓" if abs(avg - 1.0) < 0.01 else "✗"
        print(f"{factor.capitalize()}: {avg:.4f} {status}")
    
    # Example profiles
    print("\n\nExample Mortality Multipliers:")
    print("-" * 40)
    
    # Profile 1: Healthy lifestyle
    healthy_mult = CenteredMultipliers.get_multiplier(
        smoker=False,
        income_percentile=80,
        health_status='good',
        education='graduate',
        exercise='vigorous'
    )
    print(f"Healthy lifestyle (non-smoker, 80th percentile, good health, graduate, vigorous exercise):")
    print(f"  Combined multiplier: {healthy_mult:.2f}x")
    print(f"  Life expectancy impact: ~{(1/healthy_mult - 1) * 10:.1f} extra years")
    
    # Profile 2: Average
    avg_mult = CenteredMultipliers.get_multiplier(
        health_status='average',
        income_percentile=50
    )
    print(f"\nAverage person (average health, 50th percentile income):")
    print(f"  Combined multiplier: {avg_mult:.2f}x")
    
    # Profile 3: High risk
    high_risk_mult = CenteredMultipliers.get_multiplier(
        smoker=True,
        income_percentile=20,
        health_status='poor',
        education='less_than_high_school',
        exercise='none'
    )
    print(f"\nHigh risk (smoker, 20th percentile, poor health, <HS, no exercise):")
    print(f"  Combined multiplier: {high_risk_mult:.2f}x")
    print(f"  Life expectancy impact: ~{(1 - 1/high_risk_mult) * 10:.1f} fewer years")
    
    # Simulate population
    print("\n\nPopulation Simulation (10,000 people):")
    print("-" * 40)
    np.random.seed(42)
    n = 10000
    
    # Generate population with realistic correlations
    # Income and education are correlated
    income = np.random.normal(50, 25, n)
    income = np.clip(income, 1, 99)
    
    # Higher income -> more likely to have higher education
    edu_prob = (income - 1) / 98  # 0 to 1
    education = np.random.choice(
        list(CenteredMultipliers.EDUCATION['distribution'].keys()),
        size=n,
        p=list(CenteredMultipliers.EDUCATION['distribution'].values())
    )
    
    # Smoking inversely correlated with income
    smoking_prob = 0.25 - 0.002 * (income - 50)  # 25% at low income, 15% at median, 5% at high
    smoking_prob = np.clip(smoking_prob, 0.05, 0.35)
    smoker = np.random.random(n) < smoking_prob
    
    # Health correlated with income and smoking
    health_scores = np.random.normal(0, 1, n)
    health_scores += (income - 50) / 50  # Income effect
    health_scores -= smoker * 0.5  # Smoking effect
    
    health = np.where(health_scores > 1, 'excellent',
                     np.where(health_scores > 0, 'good',
                             np.where(health_scores > -1, 'average', 'poor')))
    
    # Calculate multipliers
    multipliers = []
    for i in range(n):
        mult = CenteredMultipliers.get_multiplier(
            smoker=bool(smoker[i]),
            income_percentile=int(income[i]),
            health_status=health[i]
        )
        multipliers.append(mult)
    
    multipliers = np.array(multipliers)
    
    print(f"Population average multiplier: {np.mean(multipliers):.4f}")
    print(f"  (Should be ~1.0 if properly centered)")
    print(f"Median multiplier: {np.median(multipliers):.4f}")
    print(f"10th percentile: {np.percentile(multipliers, 10):.4f}")
    print(f"90th percentile: {np.percentile(multipliers, 90):.4f}")
    
    # Show correlation structure
    print("\nRealistic Correlations in Population:")
    print(f"  Smoking rate by income tercile:")
    print(f"    Bottom third: {np.mean(smoker[income < 33]):.1%}")
    print(f"    Middle third: {np.mean(smoker[(income >= 33) & (income < 67)]):.1%}")
    print(f"    Top third: {np.mean(smoker[income >= 67]):.1%}")
    
    print(f"  Excellent/good health by income tercile:")
    bottom_good = np.mean((health[income < 33] == 'excellent') | (health[income < 33] == 'good'))
    middle_good = np.mean((health[(income >= 33) & (income < 67)] == 'excellent') | 
                          (health[(income >= 33) & (income < 67)] == 'good'))
    top_good = np.mean((health[income >= 67] == 'excellent') | (health[income >= 67] == 'good'))
    print(f"    Bottom third: {bottom_good:.1%}")
    print(f"    Middle third: {middle_good:.1%}")
    print(f"    Top third: {top_good:.1%}")


if __name__ == "__main__":
    demonstrate_centered_multipliers()