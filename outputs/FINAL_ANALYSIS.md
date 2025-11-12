# Hierarchical Bayesian Parameter Recovery - Final Analysis

## Executive Summary

**Status: ❌ MODEL NOT WORKING - ROOT CAUSE IDENTIFIED**

The hierarchical Bayesian model failed to recover parameters due to a **fundamental likelihood problem**: the likelihood function is not sensitive enough to parameter changes.

---

## Key Findings

### 1. **Diagnostic Results: Likelihood Sensitivity**

Testing on Person 0 (12 observations):
- **True parameters log-likelihood**: -6.18
- **Random parameters (all 0.5)**: -7.02  
- **Difference**: Only 0.84 units

**Parameter Sensitivity Analysis:**

| Parameter | Likelihood Range | Status |
|-----------|------------------|---------|
| w_I       | 0.50            | ⚠️ Low |
| w_E       | 0.02            | ❌ **CRITICAL** - virtually no sensitivity |
| w_U       | 0.45            | ⚠️ Low |
| aI_pos    | 1.01            | ✓ Moderate |
| aI_neg    | 0.82            | ⚠️ Low |
| a_E       | 0.35            | ⚠️ Low |
| a_U       | 0.25            | ⚠️ Low |

**Across 10 people:**
- Some people: true params barely better than random (differences < 0.5)
- Person 6: Difference of -0.01 (random is actually better!)
- Person 7: Difference of -0.33 (random is better!)
- Average difference: only 1.52 log-units

### 2. **Convergence Diagnostics**

As expected from poor likelihood:
- **Hyperparameters**: R-hat 1.4-2.5, ESS 5-10 
- **Person parameters**: R-hat 1.4-1.5, ESS 7-10
- Only `sigma_out` converged (R-hat=1.0, ESS=770) - the only well-identified parameter

### 3. **Parameter Recovery**

Correlations between estimated and true values:
- **w_I**: r = 0.08 (essentially random)
- **w_E**: r = -0.00 (completely random)
- **w_U**: r = -0.07 (essentially random)  
- **aI_pos**: r = 0.32 (weak, best of all)
- All others: r < 0.23

---

## Root Cause Analysis

### The Likelihood Problem

The simplified likelihood in `simulate_forward_numpy()` uses:

```python
logp += np.log(probs[ct] + 1e-10)  # Choice likelihood
logp += -0.5 * ((e_out[t] - e_mean) ** 2)  # Enjoyment (no variance!)
logp += -0.5 * ((u_out[t] - u_mean) ** 2)  # Utility (no variance!)
```

**Issues:**

1. **Missing variance parameters**: The outcome likelihoods don't include σ² in the denominator
   - Should be: `-0.5 * ((e_out - e_mean) / sigma_e)^2`
   - Current form makes all outcomes contribute equally regardless of noise

2. **Weak signal from choices**: With only 12-20 observations per person and soft-max choices, there's limited information

3. **Parameter confounding**: 
   - w_I, w_E, w_U all affect choice values similarly
   - Learning rates only update unchosen behavior weakly
   - Model cannot distinguish between different parameter combinations

4. **Numerical issues**: The "scaling by sigma" in the model:
   ```python
   logp_scaled = logp / (sigma_out_enj ** 2 + sigma_out_uti ** 2 + 1e-6)
   ```
   This is ad-hoc and doesn't correspond to proper statistical likelihood

---

## Why the Original Code Failed

The original `hierarchical_recovery.py` had **TWO independent problems**:

1. **Technical**: Unrolled PyTensor graphs that wouldn't compile (solved in v3)
2. **Statistical**: Poor likelihood specification (still present in v3)

Even if the original code had compiled, it would have faced the same parameter recovery issues.

---

## Solutions

### Immediate Fix: Proper Likelihood Specification

The likelihood needs to properly model the data-generating process:

```python
def proper_likelihood(params, data):
    """Proper statistical likelihood with variance parameters."""
    
    # Unpack parameters
    w_I, w_E, w_U, aI_pos, aI_neg, a_E, a_U = params
    sigma_choice = 1.0  # Temperature in softmax
    sigma_e, sigma_u = variance_params  # Need to estimate these
    
    logp = 0.0
    
    for t in range(T):
        # Choice model: Categorical with softmax
        CV = w_I * inst + w_E * enj + w_U * uti + suggestion[t]
        probs = softmax(CV / sigma_choice)
        logp += np.log(probs[choice[t]])
        
        # Outcome models: Normal distributions with proper variance
        e_mean = enj[choice[t]]
        u_mean = uti[choice[t]]
        
        logp += -0.5 * np.log(2 * np.pi * sigma_e**2) - 0.5 * ((e_out[t] - e_mean) / sigma_e)**2
        logp += -0.5 * np.log(2 * np.pi * sigma_u**2) - 0.5 * ((u_out[t] - u_mean) / sigma_u)**2
        
        # Update beliefs...
```

### Alternative Approaches

#### Option 1: Maximum Likelihood Estimation (Recommended)
Instead of full Bayesian inference, use optimization:

```python
from scipy.optimize import minimize

def neg_log_likelihood(params, data):
    return -simulate_forward_numpy(params, data)

# For each person separately
result = minimize(neg_log_likelihood, 
                 x0=initial_guess,
                 bounds=[(0, 1.5), ...],
                 method='L-BFGS-B')
```

**Advantages:**
- Much faster
- No convergence issues
- Can still get standard errors via Hessian
- Easier to diagnose problems

#### Option 2: Simplified Bayesian Model
Focus on fewer, better-identified parameters:

```python
# Just estimate weights, fix learning rates
w_I ~ Normal(0.5, 0.5)
w_E ~ Normal(0.5, 0.5) 
w_U ~ Normal(0.5, 0.5)

# Use population-level learning rates (not person-specific)
alpha_I = 0.15  # Fixed
alpha_E = 0.15  # Fixed
alpha_U = 0.15  # Fixed
```

#### Option 3: Regression-Based Approach
Extract summary statistics and use regression:

```python
# For each person, compute:
# - Average enjoyment of chosen behaviors
# - Learning slopes
# - Choice consistency

# Then regress parameters on these statistics
```

---

## Recommended Next Steps

### 1. Fix the Likelihood (Priority 1)
Implement proper variance scaling in the outcome models.

### 2. Test on Simulated Data (Priority 1)
Before fitting real data:
```python
# Generate data with known parameters
# Verify you can recover those parameters
# If not, the model specification is still wrong
```

### 3. Start Simple (Priority 2)
- Fit one person at a time (no hierarchy)
- Estimate 2-3 parameters only
- Add complexity only after basic version works

### 4. Consider MLE First (Priority 2)
Get point estimates working before attempting full Bayesian inference.

---

## Conclusion

The model failure is **not due to sampling issues** but rather a **fundamental statistical problem**:

**The likelihood function does not contain enough information to distinguish between parameter values.**

Key evidence:
- True parameters barely score better than random parameters
- Some parameters show almost zero sensitivity (w_E range: 0.02)
- Sampler cannot converge because there's nothing to converge to

**Required action**: Redesign the likelihood specification before attempting any inference.

---

## Files Generated

1. `hierarchical_recovery_v3.py` - Working code (but with likelihood issues)
2. `diagnose_likelihood.py` - Diagnostic script revealing the problem
3. `outputs/hierarchical_summary_v3.csv` - MCMC results (unreliable)
4. `outputs/hierarchical_person_params_v3.csv` - Parameter estimates (unreliable)
5. `outputs/analysis_summary.md` - Initial analysis
6. This file - Complete diagnosis

**Status**: Problem identified, solution path clear, implementation needed.
