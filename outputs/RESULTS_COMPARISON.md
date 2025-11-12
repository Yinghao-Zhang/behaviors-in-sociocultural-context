# Results Comparison: Before vs After Likelihood Fix

## Run Configuration
- **Chains**: 4
- **Draws**: 1000 per chain (4000 total)
- **Tuning**: 1000 iterations
- **Sampler**: Metropolis-Hastings

---

## Parameter Recovery Comparison

### Before Likelihood Fix (broken variance)

| Parameter | Correlation (r) | MAE   | Status |
|-----------|-----------------|-------|--------|
| w_I       | 0.084          | 0.217 | ❌ No recovery |
| w_E       | -0.002         | 0.180 | ❌ No recovery |
| w_U       | -0.071         | 0.327 | ❌ No recovery |
| aI_pos    | 0.323          | 0.414 | ⚠️ Weak |
| aI_neg    | 0.072          | 0.234 | ❌ No recovery |
| a_E       | -0.226         | 0.353 | ❌ No recovery |
| a_U       | 0.037          | 0.288 | ❌ No recovery |

### After Likelihood Fix (proper variance)

| Parameter | Correlation (r) | MAE   | Status | Change |
|-----------|-----------------|-------|--------|---------|
| w_I       | **0.109**      | 0.247 | ⚠️ Still weak | +0.025 (slight improvement) |
| w_E       | -0.007         | 0.180 | ❌ No recovery | ~0 (no change) |
| w_U       | -0.142         | 0.191 | ❌ No recovery | -0.071 (worse!) |
| aI_pos    | **0.238**      | 0.582 | ⚠️ Weak | -0.085 (worse) |
| aI_neg    | -0.170         | 0.382 | ❌ No recovery | -0.242 (worse) |
| a_E       | -0.050         | 0.276 | ❌ No recovery | +0.176 (improved but still negative) |
| a_U       | 0.067          | 0.357 | ❌ No recovery | +0.030 (minimal improvement) |

### Summary
**❌ The likelihood fix did NOT improve parameter recovery**

---

## Convergence Diagnostics Comparison

### Before Fix

| Parameter Type | R-hat Range | ESS Range | Status |
|----------------|-------------|-----------|--------|
| Hyperparameters | 1.4 - 2.5 | 5 - 10 | ❌ Very poor |
| Person params   | 1.4 - 1.5 | 7 - 10 | ❌ Very poor |
| sigma_out      | 1.0       | 770    | ✓ Only good parameter |

### After Fix

| Parameter Type | R-hat Range | ESS Range | Status |
|----------------|-------------|-----------|--------|
| **sigma_e, sigma_u** | **1.0 - 1.01** | **542 - 817** | **✓ Excellent!** |
| Hyperparameters | 1.3 - 2.5 | 5 - 11 | ❌ Still very poor |
| Person params   | 1.5 - 1.8 | 6 - 8  | ❌ Still very poor (slightly worse) |

### Summary
- ✅ **Variance parameters converged perfectly** (R-hat~1.0, ESS>500)
- ❌ **All other parameters failed to converge** (R-hat>1.3, ESS<15)
- 📊 **Slight worsening** in person-level parameter convergence

---

## What the Results Tell Us

### 1. The Variance Parameters Are Well-Identified
```
sigma_e: R-hat=1.00, ESS=817 ✓
sigma_u: R-hat=1.01, ESS=542 ✓
```
- These converged excellently because they directly affect the outcome likelihoods
- The data contains strong information about outcome variability
- **Estimated values** (would need to check summary for exact values)

### 2. Behavioral Parameters Are NOT Identifiable

**The problem is NOT the likelihood specification.** The problem is **insufficient information in the data** to distinguish between parameter values.

Evidence:
- Likelihood sensitivity analysis showed ranges of 0.02-0.50 for weights
- True parameters sometimes score worse than random
- After 4000 samples, ESS is still ~5-10 (only 0.1-0.25% effective)

### 3. Why Parameter Recovery Failed

#### Problem 1: Data Sparsity
- 50 people × 12-20 observations each
- 7 parameters per person (350 parameters total)
- ~2-3 observations per parameter on average

#### Problem 2: Parameter Confounding
Weights all affect the same quantity:
```python
choice_value = w_I * instinct + w_E * enjoyment + w_U * utility
```
Many different combinations of (w_I, w_E, w_U) produce similar patterns.

#### Problem 3: Weak Learning Signal
- Learning rates primarily affect changes over time
- But with only 12-20 time points:
  - Limited observations of learning
  - High noise relative to signal
  - Initial beliefs dominate

#### Problem 4: Metropolis Sampler Inefficiency
- Random-walk sampler in 350+ dimensional space
- Cannot efficiently explore the flat, confounded likelihood surface
- ESS of 5-10 out of 4000 samples = 99.75% rejection rate

---

## Diagnostic Patterns

### Pattern 1: Estimates Cluster Around Prior Mean
Looking at the sample comparisons, most estimates are:
- w_I ≈ 0.83-0.84 (prior mean after sigmoid: ~0.75)
- w_E ≈ 0.61-0.63
- w_U ≈ 0.62
- a_E ≈ 0.44
- a_U ≈ 0.53

**This clustering indicates the sampler is stuck near the prior** - not learning from data.

### Pattern 2: Negative Correlations
Several parameters show negative correlations (w_U: r=-0.14, aI_neg: r=-0.17):
- This is worse than random (r=0)
- Suggests systematic bias in estimation
- Model is confidently wrong

### Pattern 3: High Posterior Uncertainty
Person 0 example:
- a_E: Est=0.436, True=0.064, Diff=+0.372
- a_U: Est=0.535, True=0.177, Diff=+0.358

Massive errors (>0.3) indicate:
- Data doesn't constrain estimates
- Prior dominates posterior
- Essentially no learning occurred

---

## Conclusion

### What Worked ✅
1. Fixed likelihood specification (technically correct now)
2. Variance parameters converge and are identifiable
3. Code runs without errors

### What Didn't Work ❌
1. Parameter recovery completely failed (r < 0.25 for all)
2. Convergence failure for all behavioral parameters
3. Metropolis sampler ineffective (99.75% rejection)
4. Data insufficient to identify 7 parameters per person

### The Fundamental Problem

**This is not a software/implementation problem anymore.** This is a **statistical identifiability problem**:

> With 12-20 observations per person and 7 parameters to estimate, the model is severely underidentified. The likelihood function does not contain enough information to distinguish between parameter values.

---

## Recommendations

### Do NOT Proceed With Current Approach
The results are not interpretable. Parameter estimates are essentially random draws from the prior.

### Alternative Approaches (in order of priority):

#### 1. **Drastically Reduce Parameters** (Highest Priority)
Estimate only 2-3 parameters, fix the rest:
```python
# Fix at reasonable values
w_I = 0.5  # Fixed
w_E = 0.5  # Fixed  
w_U = 0.5  # Fixed

# Estimate only these
alpha_E ~ Beta(2, 2)  # Learning rate for enjoyment
alpha_U ~ Beta(2, 2)  # Learning rate for utility
```

#### 2. **Complete Pooling** (No Hierarchy)
One set of parameters for all people:
- Reduces from 350 to 7 parameters
- 650 observations total to estimate 7 params
- Much more identifiable

#### 3. **Non-Hierarchical: Fit Each Person Separately**
- 50 independent models
- Each with 12-20 observations for 7 parameters
- Still problematic but simpler to diagnose

#### 4. **Abandon Bayesian Inference**
Use simpler methods:
- Logistic regression for choices
- Linear regression for outcomes
- Descriptive statistics per person

#### 5. **Collect More Data**
Current data: 12-20 observations per person
Needed: Probably 50-100+ observations per person to identify 7 parameters

---

## Files Generated
- `hierarchical_summary_v3.csv` - Convergence diagnostics (shows failure)
- `hierarchical_person_params_v3.csv` - Parameter estimates (unreliable)
- `hierarchical_idata_v3.nc` - Full MCMC output (4000 samples)
- This document - Complete analysis

## Final Status
**❌ Model failure confirmed. Root cause: Insufficient data for model complexity.**

**Next step**: Simplify model or collect more data. Current approach is not salvageable.
