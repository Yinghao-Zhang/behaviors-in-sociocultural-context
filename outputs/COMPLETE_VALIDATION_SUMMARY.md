# Computational Model Validation: Complete Summary

## Date
November 11, 2025

## Overview

This document summarizes the complete validation of a computational model for predicting behavioral choices in sociocultural contexts, using simulated EMA (Ecological Momentary Assessment) data.

---

## Two Validation Approaches Tested

### 1. Parameter Recovery (Hierarchical Bayesian)
**Goal**: Recover known parameter values used to generate data

**Result**: ❌ **FAILED**
- Could not recover behavioral parameters (weights, learning rates)
- Only variance parameters converged
- R-hat > 1.3 for most parameters
- Correlations with true values near zero (r < 0.25)

**Conclusion**: Data insufficient to identify individual-level parameters

**Files**: 
- `outputs/hierarchical_summary_v3.csv`
- `outputs/hierarchical_person_params_v3.csv`
- `outputs/RESULTS_COMPARISON.md`

---

### 2. Prediction Validation (Train/Test Split)
**Goal**: Predict behavior choices better than baseline models

**Result**: ❌ **FAILED**
- Full model underperformed null baseline
- Learning component provided no benefit
- Simple marginal probability outperformed computational model

**Metrics** (Mean ± SE):
- Null model: 82.0% ± 4.0% accuracy
- Full model: 78.4% ± 4.6% accuracy
- Difference not significant (p=0.47) but trends wrong direction

**Conclusion**: Computational model too complex, overfits training data

**Files**:
- `outputs/prediction_validation_results.csv`
- `outputs/prediction_validation_summary.csv`
- `outputs/PREDICTION_VALIDATION_ANALYSIS.md`
- `outputs/prediction_validation_plot.png`
- `outputs/prediction_validation_boxplot.png`

---

## Model Specification

### Computational Framework
The model simulates decision-making with:
- **3 weights**: w_I (instinct), w_E (enjoyment), w_U (utility)
- **4 learning rates**: α_I+ (instinct positive), α_I- (instinct negative), α_E (enjoyment), α_U (utility)
- **Choice rule**: Softmax with temperature τ
- **Learning rules**: Prediction error updates for enjoyment and utility

### Data Characteristics
- **50 people** in simulated dataset
- **12-20 observations** per person (mean: 13)
- **650 total observations**
- **7 parameters** to estimate per person = 350 total parameters

**Data-to-parameter ratio**: ~2 observations per parameter

---

## Why Both Approaches Failed

### Fundamental Issue: Data Sparsity

**The problem is mathematical, not methodological:**

```
Parameter identifiability requires:
  - n_observations >> n_parameters
  
Current study:
  - n_observations ≈ 2-3 × n_parameters
  
Needed for identification:
  - n_observations ≈ 10-20 × n_parameters
  - Need 70-140 observations per person
  - Have 12-20 observations per person
```

### Secondary Issues

1. **Parameter Confounding**
   - Weights (w_I, w_E, w_U) all affect same outcome
   - Many parameter combinations produce similar behavior
   - Can't distinguish between them with limited data

2. **Weak Learning Signal**
   - Learning manifests over time
   - But only 12-20 time points available
   - Initial beliefs dominate choices

3. **Model Complexity**
   - 7 parameters is too many
   - Non-linear interactions between parameters
   - Optimization easily stuck in local minima

---

## Convergent Evidence

Both validation approaches point to the same conclusion:

| Aspect | Parameter Recovery | Prediction Validation |
|--------|-------------------|---------------------|
| **Result** | Can't recover parameters | Can't predict choices |
| **Comparison** | True vs Estimated | Train vs Test |
| **Outcome** | Correlations near 0 | Null > Full model |
| **Interpretation** | Parameters unidentifiable | Model overfits |
| **Implication** | ❌ Can't estimate | ❌ Can't use for prediction |

**Bottom line**: If you can't recover parameters when you know the truth, you definitely can't estimate them from real data or use them for prediction.

---

## What Actually Works

From the prediction validation results:

### Best Performing Models

1. **Null Model** (82% accuracy)
   - Just predict the most common choice
   - Hard to beat with sparse data

2. **Logistic Regression** (82% accuracy)  
   - Static features (initial beliefs, suggestions)
   - No temporal dynamics
   - Performs identically to null

3. **No-Learning Model** (79% accuracy)
   - Computational structure without learning
   - Slightly worse than null
   - Suggests even the base model structure doesn't help

### Worst Performing Model

4. **Full Model with Learning** (78% accuracy)
   - Adding learning makes things worse
   - Overfits to training data
   - Doesn't generalize

---

## Implications for Research

### For This Specific Project

❌ **Do NOT use this model for**:
- Parameter estimation from EMA data
- Individual differences research
- Predicting future behavior
- Theory testing about learning mechanisms

✅ **Consider instead**:
- Descriptive statistics (choice frequencies, temporal trends)
- Simpler models (2-3 parameters maximum)
- Aggregate-level analysis (one parameter set for all people)
- Qualitative analysis (maybe behavior isn't computational)

### For Future Studies

If you want to use computational modeling with EMA data:

**Option 1: Collect Much More Data**
- Need 70-140 observations per person (vs current 12-20)
- 5-10x more data collection required
- May not be feasible for intensive EMA

**Option 2: Drastically Simplify Model**
- Reduce to 2-3 parameters maximum
- Fix some parameters at reasonable values
- Focus on most theoretically important parameters

**Option 3: Different Study Design**
- Laboratory task with 100-200 trials per person
- More controlled environment
- Clearer learning signal

**Option 4: Population-Level Modeling**
- Estimate one parameter set for entire sample
- Tests model structure, not individual differences
- Much more identifiable

---

## Positive Outcomes

Despite the negative findings, this validation was valuable:

✅ **Discovered problem before using real data**
- Saved months of analysis on inappropriate model
- Prevented drawing incorrect conclusions

✅ **Established ground truth**
- Simulated data provides known parameters
- Clear evidence model doesn't work
- Not confounded by measurement error

✅ **Tested rigorously**
- Two independent validation approaches
- Both reach same conclusion
- High confidence in findings

✅ **Documented limitations**
- Clear understanding of what data CAN'T support
- Informs future study design
- Contributes to methodological literature

---

## Technical Details

### Software
- Python 3.9
- PyMC 5.18.0 (Bayesian inference)
- NumPy, Pandas, SciPy (numerical computing)
- Scikit-learn (machine learning baselines)

### Computational Approach
- **Parameter recovery**: Metropolis-Hastings MCMC (1000 draws × 4 chains)
- **Prediction validation**: Maximum likelihood estimation with L-BFGS-B
- **All code**: Pure NumPy implementation (bypassed PyTensor compilation issues)

### Scripts
1. `experiments/hierarchical_recovery_v3.py` - Parameter recovery
2. `experiments/validate_predictions.py` - Prediction validation
3. `experiments/plot_validation_results.py` - Visualization

---

## Recommendations

### Immediate Next Steps

1. **If using simulated data**:
   - Try simpler model with 2-3 parameters
   - Test if that's recoverable
   - Establish minimum viable model

2. **If planning real data collection**:
   - Pilot test with dense sampling (50-100 observations)
   - Check if model recoverable in pilot
   - Adjust design before main study

3. **If already have real data**:
   - Don't apply this model
   - Use descriptive statistics
   - Consider qualitative analysis

### Long-Term Considerations

**Re-evaluate theoretical assumptions**:
- Is behavior actually driven by trial-by-trial learning?
- Or are choices more habitual/contextual?
- Model complexity should match process complexity

**Consider measurement-level modeling**:
- Maybe EMA is wrong tool for this question
- Laboratory tasks provide better data quality
- Trade-off: ecological validity vs measurement precision

**Embrace negative findings**:
- Null results are scientifically valuable
- Document what doesn't work
- Prevent others from same mistakes

---

## Final Verdict

**The computational model is not viable for this data structure.**

Two independent validation approaches (parameter recovery and prediction validation) both demonstrate that:
- The model is too complex for available data
- Parameters are not identifiable
- Predictions don't generalize
- Simple baselines outperform computational approach

**Recommendation**: Do not proceed with computational modeling for individual-level parameter estimation or prediction using EMA data with 12-20 observations per person. Consider alternative approaches or collect substantially more data.

---

## Contact & Questions

For questions about this analysis, see:
- Full technical details in code comments
- Diagnostic outputs in `outputs/` directory
- Visualization in `.png` files

Last updated: November 11, 2025
