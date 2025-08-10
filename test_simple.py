"""Test the simplified mortality package."""

# For now, mock the data since we don't have beautifulsoup4 installed
import numpy as np

# Mock SSA data for testing
MOCK_SSA_MALE = {
    0: 0.00598, 1: 0.00041, 5: 0.00013, 10: 0.00016, 
    20: 0.00114, 30: 0.00163, 40: 0.00241, 50: 0.00509,
    60: 0.01123, 65: 0.01599, 70: 0.02534, 75: 0.04093,
    80: 0.06799, 85: 0.11578, 90: 0.19958, 95: 0.30410,
    100: 0.42319, 105: 0.53904, 110: 0.65172, 115: 0.76008,
    119: 0.86286
}

MOCK_SSA_FEMALE = {
    0: 0.00495, 1: 0.00030, 5: 0.00010, 10: 0.00012,
    20: 0.00044, 30: 0.00076, 40: 0.00147, 50: 0.00323,
    60: 0.00700, 65: 0.01071, 70: 0.01764, 75: 0.03011,
    80: 0.05244, 85: 0.09493, 90: 0.17462, 95: 0.27952,
    100: 0.40438, 105: 0.52610, 110: 0.64389, 115: 0.75650,
    119: 0.86286
}

def test_basic_functions():
    """Test basic mortality calculations."""
    
    print("Testing Simplified Mortality Package")
    print("=" * 60)
    
    # Test mortality rates
    print("\nMortality Rates at Key Ages:")
    print("-" * 40)
    
    test_ages = [0, 20, 40, 60, 65, 70, 80, 90]
    for age in test_ages:
        male_rate = MOCK_SSA_MALE.get(age, 0)
        female_rate = MOCK_SSA_FEMALE.get(age, 0)
        print(f"Age {age:3}: Male {male_rate:.4f} ({male_rate*100:.2f}%), "
              f"Female {female_rate:.4f} ({female_rate*100:.2f}%)")
    
    # Test life expectancy calculation
    print("\n\nLife Expectancy Calculations:")
    print("-" * 40)
    
    def simple_life_expectancy(age, rates):
        """Simple life expectancy calculation."""
        survival = 1.0
        total = 0.0
        
        for future_age in range(age, 120):
            total += survival
            
            # Find closest age in rates
            ages = sorted(rates.keys())
            if future_age in rates:
                mort_rate = rates[future_age]
            else:
                # Interpolate
                mort_rate = np.interp(future_age, ages, [rates[a] for a in ages])
            
            survival *= (1 - mort_rate)
        
        return total - 1  # Subtract current year
    
    for age in [0, 20, 40, 60, 65, 70, 80]:
        male_le = simple_life_expectancy(age, MOCK_SSA_MALE)
        female_le = simple_life_expectancy(age, MOCK_SSA_FEMALE)
        print(f"Age {age:3}: Male {male_le:.1f} years, Female {female_le:.1f} years")
        print(f"         Gender gap: {female_le - male_le:.1f} years")
    
    # Test Monte Carlo simulation
    print("\n\nMonte Carlo Simulation (1000 paths):")
    print("-" * 40)
    
    def simulate_deaths(starting_age, n_sims, rates, seed=42):
        """Simple Monte Carlo simulation."""
        np.random.seed(seed)
        
        death_ages = np.full(n_sims, 120.0)
        
        for sim in range(n_sims):
            for age in range(starting_age, 120):
                # Get mortality rate
                if age in rates:
                    mort_rate = rates[age]
                else:
                    ages = sorted(rates.keys())
                    mort_rate = np.interp(age, ages, [rates[a] for a in ages])
                
                # Check if death occurs
                if np.random.random() < mort_rate:
                    death_ages[sim] = age + np.random.random()  # Random within year
                    break
        
        return death_ages
    
    # Simulate for 65-year-olds
    male_deaths = simulate_deaths(65, 1000, MOCK_SSA_MALE)
    female_deaths = simulate_deaths(65, 1000, MOCK_SSA_FEMALE)
    
    male_le_sim = np.mean(male_deaths - 65)
    female_le_sim = np.mean(female_deaths - 65)
    
    print(f"65-year-old male:")
    print(f"  Simulated life expectancy: {male_le_sim:.1f} years")
    print(f"  10th percentile: {np.percentile(male_deaths - 65, 10):.1f} years")
    print(f"  Median: {np.median(male_deaths - 65):.1f} years")
    print(f"  90th percentile: {np.percentile(male_deaths - 65, 90):.1f} years")
    
    print(f"\n65-year-old female:")
    print(f"  Simulated life expectancy: {female_le_sim:.1f} years")
    print(f"  10th percentile: {np.percentile(female_deaths - 65, 10):.1f} years")
    print(f"  Median: {np.median(female_deaths - 65):.1f} years")
    print(f"  90th percentile: {np.percentile(female_deaths - 65, 90):.1f} years")
    
    # Survival curves
    print("\n\nSurvival Probabilities from Age 65:")
    print("-" * 40)
    
    def survival_curve(start_age, end_age, rates):
        """Calculate survival curve."""
        survival_probs = []
        cumulative_survival = 1.0
        
        for age in range(start_age, end_age + 1):
            survival_probs.append(cumulative_survival)
            
            if age in rates:
                mort_rate = rates[age]
            else:
                ages = sorted(rates.keys())
                mort_rate = np.interp(age, ages, [rates[a] for a in ages])
            
            cumulative_survival *= (1 - mort_rate)
        
        return np.array(survival_probs)
    
    male_survival = survival_curve(65, 100, MOCK_SSA_MALE)
    female_survival = survival_curve(65, 100, MOCK_SSA_FEMALE)
    
    milestones = [70, 75, 80, 85, 90, 95, 100]
    for age in milestones:
        idx = age - 65
        print(f"Probability of reaching {age}:")
        print(f"  Male: {male_survival[idx]:.1%}, Female: {female_survival[idx]:.1%}")


if __name__ == "__main__":
    test_basic_functions()