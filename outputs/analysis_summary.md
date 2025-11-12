# Hierarchical Bayesian Parameter Recovery Analysis

## Summary of Results

**Run Configuration:**
- Chains: 4
- Draws: 1000 per chain
- Tuning: 1000 iterations
- Sampler: Metropolis-Hastings (required for custom likelihood)
- Total samples: 4000 (after tuning)

---

## ⚠️ Major Issues Identified

### 1. **Poor Convergence** 
All parameters show severe convergence problems:

**Hyperparameters:**
- `mu_wI`: R-hat = 1.84, ESS = 6 ⚠️
- `mu_wE`: R-hat = 2.45, ESS = 5 ⚠️  **CRITICAL**
- `mu_wU`: R-hat = 1.82, ESS = 6 ⚠️
- `mu_aIneg`: R-hat = 2.01, ESS = 5 ⚠️
- `mu_aU`: R-hat = 2.22, ESS = 5 ⚠️

**Person-level parameters:**
- All show R-hat > 1.3 (typically 1.4-1.5)
- All show ESS < 10 (most around 8-9)

**Only exception:**
- `sigma_out`: R-hat = 1.00, ESS = 770 ✓

### 2. **Poor Parameter Recovery**

Correlation between estimated and true parameters:

| Parameter | Correlation (r) | MAE   | Assessment |
|-----------|-----------------|-------|------------|
| w_I       | 0.084          | 0.217 | ❌ No recovery |
| w_E       | -0.002         | 0.180 | ❌ No recovery |
| w_U       | -0.071         | 0.327 | ❌ No recovery |
| aI_pos    | 0.323          | 0.414 | ⚠️ Weak |
| aI_neg    | 0.072          | 0.234 | ❌ No recovery |
| a_E       | -0.226         | 0.353 | ❌ No recovery |
| a_U       | 0.037          | 0.288 | ❌ No recovery |

**Interpretation:**
- Correlations near 0 indicate the model is not recovering the true parameters
- The estimates are essentially independent of the true values
- Only `aI_pos` shows weak positive correlation (r=0.32)

---

## Root Causes

### 1. **Metropolis-Hastings Sampler Limitations**
- Metropolis is a random-walk sampler, much less efficient than NUTS
- For complex hierarchical models with 50 people × 7 parameters = 350+ parameters, it struggles
- The ESS values of 5-10 out of 4000 samples means only ~0.1-0.2% effective samples!

### 2. **Model Identification Issues**
The likelihood function may not provide enough information to distinguish between parameters:
- Weights (w_I, w_E, w_U) may be confounded with each other
- Learning rates may be difficult to identify from limited time series
- Hierarchical structure adds complexity

### 3. **Likelihood Specification**
The simplified likelihood (using squared errors without proper variance scaling) may not be informative enough.

---

## Recommendations

### Option A: Simplify the Model (Recommended)
1. **Reduce hierarchical structure**: Try a simpler non-hierarchical model first
2. **Fit individuals separately**: Estimate parameters for each person independently
3. **Fewer parameters**: Start with just 2-3 key parameters instead of 7

### Option B: Improve the Sampling Approach
1. **Better likelihood**: Use proper distributions (Normal, Categorical) with scan
2. **NUTS sampler**: Requires fixing the computational graph issues (scan operations)
3. **Tighter priors**: Use more informative priors based on domain knowledge

### Option C: Alternative Methods
1. **Maximum Likelihood Estimation**: Use scipy.optimize for point estimates
2. **Approximate Bayesian Computation (ABC)**: Simulation-based inference
3. **Variational Inference**: Faster approximation (PyMC supports ADVI)

---

## Next Steps

### Immediate Actions:
1. **Diagnose the likelihood**: Check if it's even changing with different parameters
2. **Fit single person**: Try recovering parameters for just 1 person first
3. **Simulate and recover**: Generate data with known parameters, verify recovery works

### Code to test:
```python
# Test if likelihood responds to parameter changes
from hierarchical_recovery_v3 import simulate_forward_numpy

# Pick one person
pp = per_person[0]

# Test with true parameters
logp_true = simulate_forward_numpy(
    pp['true_w_I'], pp['true_w_E'], pp['true_w_U'],
    pp['true_aI_pos'], pp['true_aI_neg'], pp['true_a_E'], pp['true_a_U'],
    3.0, pp['inst0'], pp['enj0'], pp['uti0'],
    pp['suggestion'], pp['choice'], pp['e_out'], pp['u_out']
)

# Test with random parameters
logp_random = simulate_forward_numpy(
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    3.0, pp['inst0'], pp['enj0'], pp['uti0'],
    pp['suggestion'], pp['choice'], pp['e_out'], pp['u_out']
)

print(f"True params logp: {logp_true}")
print(f"Random params logp: {logp_random}")
# Should see meaningful difference if likelihood is working
```

---

## Conclusion

The current model **is not successfully recovering parameters**. The combination of:
- Poor convergence (R-hat >> 1.1, ESS < 10)
- Near-zero correlations with truth
- Metropolis sampler inefficiency

...means the results are not reliable. A fundamental rethinking of the modeling approach is needed before proceeding with inference.

**Status: ❌ Model not working as intended**
