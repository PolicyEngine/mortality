#!/usr/bin/env python3
"""Basic test to verify the package works."""

import sys
sys.path.insert(0, 'src')

from mortality import Mortality, MortalityFactors
import numpy as np

def test_basic():
    """Test basic functionality."""
    print("Testing basic mortality...")
    
    # Create mortality object
    mort = Mortality(gender="male")
    
    # Test death probability
    q65 = mort.q(65)
    print(f"  Male q(65) = {q65:.4f}")
    assert 0.015 < q65 < 0.017, f"q65 out of range: {q65}"
    
    # Test female lower mortality
    mort_f = Mortality(gender="female")
    q65_f = mort_f.q(65)
    print(f"  Female q(65) = {q65_f:.4f}")
    assert q65_f < q65, "Female should have lower mortality"
    
    # Test life expectancy
    e65 = mort.e(65)
    print(f"  Male e(65) = {e65:.1f} years")
    assert 16 < e65 < 20, f"Life expectancy out of range: {e65}"
    
    print("✓ Basic tests passed")

def test_factors():
    """Test mortality factors."""
    print("\nTesting mortality factors...")
    
    # Test good health
    factors_good = MortalityFactors(health="good")
    mult = factors_good.get_multiplier()
    print(f"  Good health multiplier = {mult:.2f}")
    assert mult < 1.0, "Good health should reduce mortality"
    
    # Test smoking
    factors_smoker = MortalityFactors(smoker=True)
    mult_smoke = factors_smoker.get_multiplier()
    print(f"  Smoker multiplier = {mult_smoke:.2f}")
    assert mult_smoke > 1.5, "Smoking should increase mortality significantly"
    
    # Test with mortality
    mort_healthy = Mortality(
        gender="male",
        factors=MortalityFactors(health="excellent", income_percentile=90)
    )
    q65_healthy = mort_healthy.q(65)
    print(f"  Healthy wealthy q(65) = {q65_healthy:.4f}")
    assert q65_healthy < 0.012, "Healthy wealthy should have low mortality"
    
    print("✓ Factor tests passed")

def test_simulation():
    """Test Monte Carlo simulation."""
    print("\nTesting simulation...")
    
    mort = Mortality(gender="male")
    death_ages = mort.simulate(age=65, n_sims=1000)
    
    median_death = np.median(death_ages)
    print(f"  Median death age = {median_death:.1f}")
    assert 80 < median_death < 85, f"Median death age unexpected: {median_death}"
    
    pct10 = np.percentile(death_ages, 10)
    pct90 = np.percentile(death_ages, 90)
    print(f"  10th percentile = {pct10:.1f}")
    print(f"  90th percentile = {pct90:.1f}")
    assert pct90 - pct10 > 15, "Should have realistic spread"
    
    print("✓ Simulation tests passed")

if __name__ == "__main__":
    test_basic()
    test_factors()
    test_simulation()
    print("\n✅ All tests passed!")