# mortality

A practical Python package for mortality calculations and projections.

## Installation

```bash
pip install mortality
```

## Quick Start

```python
from mortality import Mortality

# Simple: just age and gender
mort = Mortality(gender="male")
print(f"Probability of death at 65: {mort.q(65):.3%}")
print(f"Life expectancy at 65: {mort.e(65):.1f} years")

# With personal factors
from mortality import MortalityFactors

mort = Mortality(
    gender="female",
    factors=MortalityFactors(
        health="good",
        income_percentile=75
    )
)
print(f"Personalized life expectancy: {mort.e(65):.1f} years")

# Monte Carlo simulation
death_ages = mort.simulate(age=65, n_sims=1000)
print(f"Median death age: {np.median(death_ages):.1f}")
```

## Features

- 📊 **Simple API** - Start with just age and gender
- 🎯 **Personal Factors** - Health, income, education, lifestyle
- 📈 **Mortality Improvements** - Cohort projections built-in
- 🎲 **Monte Carlo** - Fast simulation for financial planning
- 📚 **Research-Based** - All adjustments cite peer-reviewed sources

## Documentation

See [docs](docs/) for detailed documentation.

## License

MIT