# Future Features for Mortality Package

## Advanced Mortality Adjustment (Removed in d58d918)

The initial development included sophisticated mortality adjustment approaches that were removed for v1 simplicity. These features are preserved in commits b577e4f and earlier.

### 1. Bayesian Mortality Adjustment (`bayesian.py`)
**Commit**: 8d07c5a

Proper Bayesian adjustment that accounts for the fact that SSA tables already include population heterogeneity. Key insight: We need to SUBTRACT population average effects before ADDING individual effects.

```python
# Example: Smoking adjustment
# SSA rates already include 15% smokers with 1.8x mortality
population_effect = smoking_prevalence * smoking_effect
log_hazard -= population_effect  # Remove population average
if self.smoker:
    log_hazard += smoking_effect  # Add individual effect
```

### 2. Centered Multipliers (`centered_multipliers.py`)
**Commit**: b577e4f

Population-weighted multipliers that average to 1.0. However, this approach assumes independence between factors, which isn't realistic.

```python
# Centered so population average = 1.0
SMOKING = {
    'centered': {
        True: 1.61,   # 1.8 / 1.12
        False: 0.89   # 1.0 / 1.12
    }
}
```

### 3. Conditional Mortality Model (`conditional_multipliers.py`)
**Commit**: b577e4f

Accounts for correlations between risk factors using conditional probabilities:
- Smoking rates vary by income (21.2% in bottom quintile vs 6.7% in top)
- Health status correlates with income
- These correlations affect proper calibration

### 4. Vectorized Implementation (`vectorized.py`)
**Commit**: 8d07c5a

Fully vectorized NumPy implementation for fast Monte Carlo simulation (10,000 paths in <1 second).

### 5. Research-Based Factors (`research/factors.yaml`)
**Commit**: 8d07c5a

Data-driven mortality factors from peer-reviewed research with citations.

## Why These Were Removed

1. **Data Requirements**: Proper calibration requires microdata with health variables (smoking, BMI, exercise) that correlate with income/demographics. The CPS doesn't include these.

2. **Complexity**: The conditional probability approach is complex and requires careful calibration to maintain population averages.

3. **Validation**: Without good microdata, we can't properly validate these adjustments.

## Path Forward

### Phase 1: Data Integration
- [ ] Integrate CDC BRFSS data for health behaviors by demographics
- [ ] Link CPS microdata for income/education distributions
- [ ] Validate correlations between factors

### Phase 2: Conditional Model
- [ ] Implement proper conditional probability model
- [ ] Calibrate to maintain SSA population averages
- [ ] Test with actual microdata

### Phase 3: Advanced Features
- [ ] Lee-Carter mortality projection
- [ ] Cohort effects
- [ ] Geographic variation
- [ ] Time trends in mortality improvement

## Resources

- **Chetty et al (2016)**: Income-mortality gradient
- **CDC BRFSS**: Health behavior data by demographics
- **Human Mortality Database**: International mortality data
- **SOA Mortality Tables**: Industry standard tables

## Using the Advanced Features

To experiment with the removed features, checkout commit b577e4f:

```bash
git checkout b577e4f
```

The key insight from this development: **Mortality adjustment is fundamentally about conditional probabilities, not independent multipliers.** SSA tables are population averages that already incorporate all heterogeneity. Any individual adjustment must account for this.