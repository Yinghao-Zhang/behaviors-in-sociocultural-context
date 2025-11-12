# Likelihood Fix Implementation - November 10, 2025

## Changes Made

### 1. Fixed Likelihood Specification

**Before (Incorrect):**
```python
logp += -0.5 * ((e_out[t] - e_mean) ** 2)  # Missing variance!
logp += -0.5 * ((u_out[t] - u_mean) ** 2)  # Missing variance!
```

**After (Correct):**
```python
# Proper Normal log-likelihood
logp += -log_sqrt_2pi - np.log(sigma_e) - 0.5 * ((e_out[t] - e_mean) / sigma_e) ** 2
logp += -log_sqrt_2pi - np.log(sigma_u) - 0.5 * ((u_out[t] - u_mean) / sigma_u) ** 2
```

### 2. Added Variance Parameters

- Changed from single `sigma_out` to separate `sigma_e` and `sigma_u`
- Prior: `HalfNormal(0.3)` matching empirical std of ~0.29-0.30
- These are now estimated from data rather than ad-hoc scaling

### 3. Removed Ad-Hoc Scaling

**Before:**
```python
total_logp += logp_raw / (sigma_val ** 2 + 1e-6)  # Ad-hoc!
```

**After:**
```python
total_logp += logp_raw  # Proper statistical likelihood
```

## Diagnostic Results

### Likelihood Sensitivity (With Fix)

| Parameter | Range | Status | Improvement |
|-----------|-------|--------|-------------|
| w_I       | 0.50  | ⚠️ Low | Same |
| w_E       | 0.02  | ⚠️ Low | Same |
| w_U       | 0.45  | ⚠️ Low | Same |
| aI_pos    | 1.01  | ✓ OK   | Same |
| aI_neg    | 0.82  | ⚠️ Low | Same |
| **a_E**   | **5.99**  | **✓ Much better!** | **17x improvement!** |
| **a_U**   | **2.23**  | **✓ Better** | **9x improvement!** |

### Key Observations

1. **Learning rates (a_E, a_U) now have strong sensitivity** - this is expected since they directly affect how outcomes are predicted

2. **Weights (w_I, w_E, w_U) still have low sensitivity** - this is concerning and suggests:
   - Not enough choice data per person (12-20 observations)
   - Weights may be confounded with each other
   - Soft

max with tau=3 may be too "soft" to distinguish weights

3. **True params sometimes score worse than random** - Average difference: -1.40 log-units
   - Person 0: True = -23.30, Random = -15.61 (random is better!)
   - This suggests model misspecification remains

## Remaining Issues

### 1. Insufficient Data Per Person
- Only 12-20 observations per person
- With 7 parameters to estimate, this is ~2-3 data points per parameter
- Classic underidentification problem

### 2. Weight Confounding
Weights affect the same quantity (choice values):
```python
CV = w_I * inst + w_E * enj + w_U * uti
```
With limited data, many combinations of (w_I, w_E, w_U) can produce similar choice patterns.

### 3. Temporal Learning Signal
Learning rates primarily affect *changes* over time, but with short sequences:
- Limited opportunity to observe learning
- Initial beliefs may dominate
- Noisy outcomes make learning signal weak

## Expected Results

### If Convergence Improves:
- R-hat values closer to 1.0 for a_E and a_U
- Better ESS for learning rate parameters
- Weights may still show poor convergence

### If Parameter Recovery Improves:
- Correlations for a_E and a_U should increase (maybe r > 0.4-0.5)
- Weights may still show low correlations (r < 0.2)
- Overall recovery still limited by data constraints

## What This Tells Us

The likelihood fix was **necessary but not sufficient**:
- ✅ Proper statistical formulation
- ✅ Some parameters now identifiable (a_E, a_U)
- ❌ Fundamental data limitations remain
- ❌ Model complexity exceeds information available

## Next Steps If This Run Fails

### Option 1: Reduce Model Complexity
```python
# Estimate only well-identified parameters
model {
    # Fix weights at reasonable values
    w_I = 0.5
    w_E = 0.5  
    w_U = 0.5
    
    # Estimate only learning rates (better identified)
    a_E ~ Beta(2, 2)
    a_U ~ Beta(2, 2)
}
```

### Option 2: Pooling Across People
Instead of hierarchical (partial pooling), try complete pooling:
```python
# Single set of parameters for everyone
w_I ~ Normal(0, 1)
w_E ~ Normal(0, 1)
w_U ~ Normal(0, 1)
# ... etc
```

### Option 3: Use Summary Statistics
Extract person-level summaries, then fit simpler model:
```python
# For each person:
- proportion_chose_approach
- average_enjoyment_chosen
- learning_slope (regression)

# Then estimate parameters from these summaries
```

### Option 4: Simulation-Based Calibration
Before fitting real data, verify parameter recovery on simulated data:
1. Generate data with known parameters
2. Attempt to recover those parameters
3. Only proceed to real data if recovery works

## Files Modified
- `hierarchical_recovery_v3.py` - Fixed likelihood function
- `diagnose_likelihood.py` - Updated to use fixed likelihood

## Status
🔄 Running 4 chains × 1000 draws (estimated 12-13 minutes)
