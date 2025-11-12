# Prediction Validation Analysis

## Executive Summary

**❌ The computational model with learning does NOT provide predictive value over simpler alternatives.**

- Full model performs **worse** than null model on all metrics
- Learning component provides **no benefit** over no-learning model
- Simpler models (null, logistic regression) outperform computational approach

---

## Study Design

### Objective
Test whether the computational model can predict behavior choices better than:
1. **Null model**: Marginal probability baseline
2. **Logistic regression**: Static model without temporal dynamics
3. **No-learning model**: Computational model without trial-by-trial learning
4. **Full model**: Complete computational model with learning

### Method
- **Train/Test Split**: 70% training, 30% testing per person
- **Sample**: 31 people with sufficient data (19 excluded due to: too few trials, single-class training sets)
- **Evaluation Metrics**:
  - Accuracy: Percentage of correct predictions
  - Log Loss: Probabilistic calibration (lower is better)
  - AUC: Discrimination ability
  - Brier Score: Calibration quality

---

## Results

### Overall Performance (Mean ± SE)

| Model | Accuracy | Log Loss ↓ | AUC | Brier Score ↓ |
|-------|----------|------------|-----|---------------|
| **Null** | **0.820 ± 0.040** | **0.636 ± 0.027** | 0.500 ± 0.000 | **0.133 ± 0.018** |
| **Logistic Reg** | **0.820 ± 0.040** | **0.636 ± 0.027** | 0.469 ± 0.050 | **0.132 ± 0.018** |
| **No-Learning** | 0.791 ± 0.048 | 0.637 ± 0.026 | **0.547 ± 0.049** | 0.164 ± 0.019 |
| **Full Model** | 0.784 ± 0.046 | 0.680 ± 0.031 | 0.303 ± 0.086 | 0.156 ± 0.021 |

**Bold** = Best performance for that metric

### Key Findings

#### 1. Null Model Outperforms Everything
- **Accuracy**: 82.0% (baseline)
- **Log Loss**: 0.636 (best calibration)
- Simply predicting the most common choice works best

#### 2. Full Model Underperforms
- **Accuracy**: 78.4% (worse than null by 3.6%)
- **Log Loss**: 0.680 (worst calibration)
- **AUC**: 0.303 (worse than random!)
- Adding learning makes predictions **worse**, not better

#### 3. No Statistical Significance
- Full vs Null (accuracy): Δ=-0.036, p=0.465 (ns)
- Full vs Null (log loss): Δ=+0.044, p=0.229 (ns)
- Full vs No-Learning (accuracy): Δ=-0.006, p=0.786 (ns)
- Full vs No-Learning (log loss): Δ=+0.043, p=0.151 (ns)

But all trends point in the **wrong direction** (full model worse).

---

## Interpretation

### Why Did the Model Fail?

#### Problem 1: Overfitting on Sparse Training Data
With only 3-14 training trials per person and 7 parameters to fit:
- Model memorizes noise instead of learning signal
- Overconfident predictions on test set
- High variance, low generalization

#### Problem 2: Test Sets Are Too Easy
Many test sets have:
- Only 3-6 trials
- Often dominated by one choice (82% accuracy just predicting mode)
- Little opportunity to demonstrate learning effects

#### Problem 3: Learning Signal Is Weak or Nonexistent
The AUC results are telling:
- Null model: 0.500 (random, as expected - predicts same probability for all)
- No-Learning: 0.547 (slight discrimination ability)
- Full model: 0.303 (worse than random!)

**This suggests the learning component is actively harmful** - it's fitting noise and making the model more confused.

#### Problem 4: Parameter Confounding
From hierarchical recovery analysis, we know:
- Weights are highly confounded (many combinations produce similar choices)
- MLE optimization may find local optima
- Fitted parameters don't match true parameters

---

## Sample-Level Examples

### Example 1: Person 2 (Full Model Worse)
- Null model: Acc=0.833, LogLoss=0.453
- Full model: Acc=0.833, LogLoss=0.716
- **Full model is less calibrated** (higher log loss despite same accuracy)

### Example 2: Person 11 (Full Model Much Worse)
- Null model: Acc=0.750, LogLoss=0.620
- No-Learning: Acc=0.750, LogLoss=0.610
- Full model: Acc=0.500, LogLoss=0.946
- **Learning component destroyed performance** (25% accuracy drop)

### Example 3: Person 19 (No-Learning Worse Than Null)
- Null model: Acc=0.500
- No-Learning: Acc=0.250 (worse!)
- Full model: Acc=0.500
- **Even the computational structure without learning underperforms**

---

## Data Quality Issues

### People Excluded (N=19)
- 14 people: Only one class in training data (can't fit models)
- 5 people: Too few test trials (<3)

### People Included (N=31)
- 15 people: Both classes in test set (valid for log loss / AUC)
- 16 people: Single class in test set (only accuracy/Brier valid)

**Bottom line**: Only 15 of 50 people (30%) had sufficient data for full evaluation.

---

## Comparison to Hierarchical Recovery

### Hierarchical Recovery Findings
- Could not recover parameters from data
- Data insufficient to distinguish between parameter values
- Only variance parameters were identifiable

### Prediction Validation Findings  
- **Confirms the hierarchical recovery findings**
- If parameters aren't identifiable from data, they can't predict either
- Model complexity exceeds information available in data

**Consistent story**: The data doesn't contain enough signal for this model.

---

## Conclusions

### What We Learned

✅ **The computational framework is too complex for this data**
- 7 parameters per person exceeds what 12-20 observations can support
- Overfits to training data, doesn't generalize

✅ **Learning component provides no value**
- Full model ≤ No-learning model
- Trial-by-trial updates don't improve predictions

✅ **Simple baselines are hard to beat**
- Null model (just predict the mode) gets 82% accuracy
- Logistic regression performs identically
- Hard to do better than "always choose the same option"

✅ **Simulated data confirms theoretical limitations**
- Even with ground truth available, model can't extract signal
- Problem is fundamental, not implementation-specific

### What This Means

❌ **Do NOT use this approach for prediction on real EMA data**
- If it fails on simulated data, real data will be worse
- Model complexity is not justified by predictive performance

❌ **Learning parameters are not estimable from sparse data**
- Need ~10-20 observations per parameter minimum
- Current data: ~2-3 observations per parameter

❌ **The computational model doesn't capture choice behavior**
- Model might be misspecified
- Or behavior is simpler than the model assumes

---

## Recommendations

### For This Project

1. **Simplify the model drastically**
   - Estimate 2-3 parameters maximum
   - Fix weights at reasonable values
   - Focus on learning rates only (if theoretically motivated)

2. **Use descriptive statistics instead**
   - Choice frequencies
   - Temporal trends (first half vs second half)
   - Correlations between outcomes and subsequent choices

3. **Try simpler computational models**
   - Win-stay-lose-shift
   - Exponentially weighted moving average
   - Single learning rate (not separate for I/E/U)

4. **Consider qualitative analysis**
   - If computational approach doesn't work, maybe behavior isn't computational
   - Could be driven by context, mood, habits instead

### For Future Data Collection

If you want to estimate this model in the future:
- **Need 100-200 observations per person minimum**
- Current: 12-20 observations → Target: 100-200 observations
- Or reduce to 2-3 parameters and keep current sample size

### Alternative Approaches

1. **Aggregate-level analysis**
   - Estimate one set of parameters for all people
   - Tests if model structure is valid at population level

2. **Subset analysis**
   - Find people with 30+ observations
   - Fit models only for them

3. **Cross-validation on simulated data**
   - Generate data with different parameter regimes
   - Test which parameters are recoverable
   - Only estimate those in real data

---

## Files Generated

- `prediction_validation_results.csv` - Full results for all 31 people
- `prediction_validation_summary.csv` - Aggregate statistics by model
- This document - Comprehensive analysis

## Technical Details

- **Python script**: `experiments/validate_predictions.py`
- **Optimization**: L-BFGS-B with logit-transformed parameters
- **Maximum iterations**: 1000 per person
- **Random seed**: 123 (reproducible)

---

## Final Verdict

**❌ The computational model fails the prediction validation test.**

The model is:
- Too complex for the available data
- Overfits to training sets
- Does not generalize to test sets
- Outperformed by trivial baselines

**Recommendation**: Do not proceed with this modeling approach for parameter estimation or prediction on real EMA data. Consider simpler alternatives or collect much more data per person.
