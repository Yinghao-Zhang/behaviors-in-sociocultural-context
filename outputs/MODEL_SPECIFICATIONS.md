# Mathematical Specifications of the Four Prediction Models

## Overview

This document provides complete mathematical descriptions of the four models compared in the prediction validation study. All models predict binary choices between two behaviors: $b_0$ (avoid conflict) and $b_1$ (approach conflict with care).

---

## Model 1: Null Model

### Description
The null model predicts that all future choices will be the most common choice observed in training data. This is the simplest possible baseline.

### Mathematical Formulation

**Training Phase:**

Given training data with $T_{train}$ trials and binary choices $c_1, c_2, \ldots, c_{T_{train}}$ where $c_t \in \{0, 1\}$:

$$\hat{p} = \frac{1}{T_{train}} \sum_{t=1}^{T_{train}} c_t$$

This is the empirical marginal probability of choosing behavior $b_1$.

**Prediction Phase:**

For any future trial $t$, the predicted probability distribution is:

$$P(c_t = 1 | \text{train}) = \hat{p}$$
$$P(c_t = 0 | \text{train}) = 1 - \hat{p}$$

**Predicted choice:**

$$\hat{c}_t = \begin{cases}
1 & \text{if } \hat{p} \geq 0.5 \\
0 & \text{otherwise}
\end{cases}$$

### Number of Parameters
- **0 free parameters** (only uses training data statistics)

### Theoretical Justification
- Establishes baseline performance
- Performance floor that any reasonable model should exceed
- Equivalent to predicting the prior distribution ignoring all covariates

### Implementation Note
In cases where $\hat{p} = 0$ or $\hat{p} = 1$ (single class in training), log-likelihood becomes undefined. These cases are excluded from log-loss evaluation.

---

## Model 2: Logistic Regression

### Description
A static generalized linear model that predicts choices from features but includes no temporal dynamics or learning. Uses initial belief states and suggestion terms as covariates.

### Mathematical Formulation

**Feature Vector:**

At trial $t$, the feature vector $\mathbf{x}_t \in \mathbb{R}^{10}$ consists of:

$$\mathbf{x}_t = \begin{bmatrix}
s_{0,t} \\
s_{1,t} \\
I_0^{(0)} \\
I_1^{(0)} \\
E_0^{(0)} \\
E_1^{(0)} \\
U_0^{(0)} \\
U_1^{(0)} \\
\end{bmatrix}$$

Where:
- $s_{j,t}$ = suggestion term for behavior $b_j$ at trial $t$
- $I_j^{(0)}$ = initial instinct value for behavior $b_j$ (constant across trials)
- $E_j^{(0)}$ = initial enjoyment belief for behavior $b_j$ (constant)
- $U_j^{(0)}$ = initial utility belief for behavior $b_j$ (constant)

**Model Equation:**

$$P(c_t = 1 | \mathbf{x}_t, \boldsymbol{\beta}) = \frac{1}{1 + \exp(-\boldsymbol{\beta}^T \mathbf{x}_t)}$$

Or equivalently, the log-odds (logit):

$$\text{logit}(P(c_t = 1)) = \beta_0 + \beta_1 s_{0,t} + \beta_2 s_{1,t} + \beta_3 I_0^{(0)} + \beta_4 I_1^{(0)} + \beta_5 E_0^{(0)} + \beta_6 E_1^{(0)} + \beta_7 U_0^{(0)} + \beta_8 U_1^{(0)}$$

Note: Intercept $\beta_0$ is implicit in scikit-learn implementation.

**Training:**

Parameters $\boldsymbol{\beta} = [\beta_0, \beta_1, \ldots, \beta_8]^T$ are estimated by maximizing the log-likelihood:

$$\mathcal{L}(\boldsymbol{\beta}) = \sum_{t=1}^{T_{train}} \left[ c_t \log p_t + (1-c_t) \log(1-p_t) \right]$$

where $p_t = P(c_t = 1 | \mathbf{x}_t, \boldsymbol{\beta})$.

Optimization uses L-BFGS-B algorithm (quasi-Newton method).

**Prediction:**

$$P(c_{test} = 1) = \frac{1}{1 + \exp(-\boldsymbol{\hat{\beta}}^T \mathbf{x}_{test})}$$

### Number of Parameters
- **9 parameters**: $\beta_0, \beta_1, \ldots, \beta_8$ (intercept + 8 feature weights)

### Theoretical Justification
- Standard approach for binary classification
- Tests whether static features (initial states + suggestions) predict choices
- No temporal dynamics: believes states don't change across trials
- Represents "memoryless" decision-maker

### Key Assumption
**Critically**: This model assumes belief states are constant. It doesn't model trial-by-trial learning. This is the key contrast with the computational models.

---

## Model 3: No-Learning Computational Model

### Description
A computational cognitive model with the full decision architecture (instinct, enjoyment, utility, weighted combination) but with learning rates fixed at zero. Belief states remain at their initial values throughout the task.

### Mathematical Formulation

**Belief States:**

For all trials $t = 1, 2, \ldots, T$:

$$\mathbf{I}_t = \mathbf{I}^{(0)} = \begin{bmatrix} I_0^{(0)} \\ I_1^{(0)} \end{bmatrix}$$

$$\mathbf{E}_t = \mathbf{E}^{(0)} = \begin{bmatrix} E_0^{(0)} \\ E_1^{(0)} \end{bmatrix}$$

$$\mathbf{U}_t = \mathbf{U}^{(0)} = \begin{bmatrix} U_0^{(0)} \\ U_1^{(0)} \end{bmatrix}$$

Where:
- $\mathbf{I}_t \in [-1, 1]^2$ = instinct values (how naturally inclined toward each behavior)
- $\mathbf{E}_t \in [-1, 1]^2$ = enjoyment beliefs (expected hedonic value)
- $\mathbf{U}_t \in [-1, 1]^2$ = utility beliefs (expected instrumental value)

**Choice Values:**

At trial $t$, the choice value for each behavior is:

$$\text{CV}_j(t) = w_I \cdot I_j^{(0)} + w_E \cdot E_j^{(0)} + w_U \cdot U_j^{(0)} + s_{j,t}$$

for $j \in \{0, 1\}$, where:
- $w_I \in [0, 1.5]$ = weight for instinct
- $w_E \in [0, 1.5]$ = weight for enjoyment  
- $w_U \in [0, 1.5]$ = weight for utility
- $s_{j,t}$ = external suggestion term for behavior $j$ at trial $t$

**Choice Probabilities:**

Choices are made via softmax function with temperature parameter $\tau$:

$$P(c_t = j | \mathbf{I}^{(0)}, \mathbf{E}^{(0)}, \mathbf{U}^{(0)}, \mathbf{s}_t, \boldsymbol{\theta}) = \frac{\exp(\tau \cdot \text{CV}_j(t))}{\sum_{k=0}^{1} \exp(\tau \cdot \text{CV}_k(t))}$$

where $\boldsymbol{\theta} = [w_I, w_E, w_U]$ are the free parameters.

**No Learning:**

Crucially, after observing outcome $(e_t, u_t)$ at trial $t$, beliefs do NOT update:

$$\mathbf{I}_{t+1} = \mathbf{I}_t = \mathbf{I}^{(0)}$$
$$\mathbf{E}_{t+1} = \mathbf{E}_t = \mathbf{E}^{(0)}$$
$$\mathbf{U}_{t+1} = \mathbf{U}_t = \mathbf{U}^{(0)}$$

(All learning rates are fixed at 0)

**Training:**

Parameters $\boldsymbol{\theta} = [w_I, w_E, w_U]$ are estimated by maximizing log-likelihood:

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{t=1}^{T_{train}} \log P(c_t = c_t^{obs} | \boldsymbol{\theta})$$

To enforce constraints $w \in [0, 1.5]$, we use logit transformation:

$$w_I = 1.5 \cdot \sigma(\xi_I), \quad w_E = 1.5 \cdot \sigma(\xi_E), \quad w_U = 1.5 \cdot \sigma(\xi_U)$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function, and $\xi_I, \xi_E, \xi_U \in \mathbb{R}$ are unconstrained.

Optimization over $[\xi_I, \xi_E, \xi_U]$ using L-BFGS-B.

**Prediction:**

For test trial:

$$P(c_{test} = 1) = \frac{\exp(\tau \cdot \text{CV}_1)}{\exp(\tau \cdot \text{CV}_0) + \exp(\tau \cdot \text{CV}_1)}$$

using fitted $\hat{w}_I, \hat{w}_E, \hat{w}_U$ and fixed initial states.

### Number of Parameters
- **3 free parameters**: $w_I, w_E, w_U$ (weights)
- **Fixed**: $\alpha_{I,pos} = \alpha_{I,neg} = \alpha_E = \alpha_U = 0$ (no learning)
- **Fixed**: $\tau = 3.0$ (softmax temperature)

### Theoretical Justification
- Tests whether computational architecture (weighted combination of instinct/enjoyment/utility) explains choices
- Isolates contribution of decision structure from learning mechanisms
- If this outperforms logistic regression, the three-component structure is valuable
- If this equals logistic regression, the structure doesn't add explanatory power

### Comparison to Logistic Regression
The key difference:
- **Logistic**: Linear combination of all 8 features, learned weights
- **No-Learning**: Theory-driven structure (separate I/E/U components), only 3 weights, explicitly models choice architecture

---

## Model 4: Full Computational Model (with Learning)

### Description
Complete computational cognitive model with trial-by-trial learning. Belief states update based on prediction errors after each observed outcome. This is the full theoretical model.

### Mathematical Formulation

**Initial Belief States (t=0):**

$$\mathbf{I}_0 = \mathbf{I}^{(0)} = \begin{bmatrix} I_0^{(0)} \\ I_1^{(0)} \end{bmatrix}, \quad
\mathbf{E}_0 = \mathbf{E}^{(0)} = \begin{bmatrix} E_0^{(0)} \\ E_1^{(0)} \end{bmatrix}, \quad
\mathbf{U}_0 = \mathbf{U}^{(0)} = \begin{bmatrix} U_0^{(0)} \\ U_1^{(0)} \end{bmatrix}$$

**Choice Model (Same as Model 3):**

At trial $t$, choice values:

$$\text{CV}_j(t) = w_I \cdot I_{j,t} + w_E \cdot E_{j,t} + w_U \cdot U_{j,t} + s_{j,t}$$

Choice probabilities via softmax:

$$P(c_t = j | \mathbf{I}_t, \mathbf{E}_t, \mathbf{U}_t, \mathbf{s}_t, \boldsymbol{\theta}) = \frac{\exp(\tau \cdot \text{CV}_j(t))}{\sum_{k=0}^{1} \exp(\tau \cdot \text{CV}_k(t))}$$

**Learning Rules:**

After trial $t$ with chosen behavior $c_t$ and observed outcomes $(e_t, u_t)$:

**1. Instinct Update** (approach/avoidance tendency):

$$I_{j,t+1} = \begin{cases}
I_{j,t} + \alpha_{I,pos} \cdot (1 - I_{j,t}) & \text{if } j = c_t \\
I_{j,t} + \alpha_{I,neg} \cdot (-1 - I_{j,t}) & \text{if } j \neq c_t
\end{cases}$$

Interpretation:
- Chosen behavior: instinct moves toward +1 (more attractive)
- Unchosen behavior: instinct moves toward -1 (more aversive)
- Learning rates $\alpha_{I,pos}, \alpha_{I,neg} \in [0,1]$ control speed

**2. Enjoyment Update** (prediction error learning):

$$E_{j,t+1} = \begin{cases}
E_{j,t} + \alpha_E \cdot (e_t - E_{j,t}) & \text{if } j = c_t \\
E_{j,t} & \text{if } j \neq c_t
\end{cases}$$

Where:
- $e_t \in [-1, 1]$ = observed enjoyment outcome
- $\delta_E = e_t - E_{c_t,t}$ = prediction error
- $\alpha_E \in [0,1]$ = enjoyment learning rate
- Only chosen behavior's belief updates

**3. Utility Update** (prediction error learning):

$$U_{j,t+1} = \begin{cases}
U_{j,t} + \alpha_U \cdot (u_t - U_{j,t}) & \text{if } j = c_t \\
U_{j,t} & \text{if } j \neq c_t
\end{cases}$$

Where:
- $u_t \in [-1, 1]$ = observed utility outcome
- $\delta_U = u_t - U_{c_t,t}$ = prediction error
- $\alpha_U \in [0,1]$ = utility learning rate

**Boundary Constraints:**

After each update, beliefs are clipped to valid range:

$$I_{j,t+1} \leftarrow \text{clip}(I_{j,t+1}, -1, 1)$$
$$E_{j,t+1} \leftarrow \text{clip}(E_{j,t+1}, -1, 1)$$
$$U_{j,t+1} \leftarrow \text{clip}(U_{j,t+1}, -1, 1)$$

**Training:**

Parameters $\boldsymbol{\theta} = [w_I, w_E, w_U, \alpha_{I,pos}, \alpha_{I,neg}, \alpha_E, \alpha_U]$ estimated by maximizing log-likelihood:

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{t=1}^{T_{train}} \log P(c_t = c_t^{obs} | \text{history}_{<t}, \boldsymbol{\theta})$$

With transformations to enforce constraints:
- Weights: $w = 1.5 \cdot \sigma(\xi)$ for $w \in [0, 1.5]$
- Learning rates: $\alpha = \sigma(\zeta)$ for $\alpha \in [0, 1]$

Optimization uses L-BFGS-B on unconstrained parameters.

**Prediction:**

For test data, beliefs are propagated forward:
1. Initialize with training-data endpoints (beliefs at end of training)
2. For each test trial:
   - Compute choice probabilities using current beliefs
   - Update beliefs based on observed outcomes
   - Propagate to next trial

Or alternatively (as implemented):
1. Re-initialize with $\mathbf{I}^{(0)}, \mathbf{E}^{(0)}, \mathbf{U}^{(0)}$
2. Simulate forward through test sequence using fitted parameters

### Number of Parameters
- **7 free parameters**: $w_I, w_E, w_U, \alpha_{I,pos}, \alpha_{I,neg}, \alpha_E, \alpha_U$
- **Fixed**: $\tau = 3.0$ (softmax temperature)

### Theoretical Justification
- Full cognitive model of learning and decision-making
- Captures trial-by-trial adaptation
- Distinguishes multiple learning mechanisms:
  - Instinct: Approach/avoidance learning
  - Enjoyment: Hedonic prediction error
  - Utility: Instrumental prediction error
- Tests whether learning improves predictions over static model

### Comparison to No-Learning Model
The only difference is:
- **No-Learning**: $\alpha_{I,pos} = \alpha_{I,neg} = \alpha_E = \alpha_U = 0$ (3 parameters)
- **Full**: All learning rates are free parameters (7 parameters)

This is a **nested model comparison**: Full model contains No-Learning as special case. If Full doesn't outperform No-Learning, the additional 4 parameters (learning rates) don't add value.

---

## Formal Relationships Between Models

### Nesting Structure

```
Null Model (0 params)
    ↓
Logistic Regression (9 params)
    ↓
No-Learning Model (3 params)  ← nested in →
    ↓
Full Model (7 params)
```

**Note**: Logistic and Computational models are NOT nested—they have different functional forms.

### Complexity Hierarchy

| Model | # Parameters | Temporal Dynamics | Learning | Constraints |
|-------|-------------|-------------------|----------|-------------|
| Null | 0 | No | No | None |
| Logistic | 9 | No | No | None |
| No-Learning | 3 | No (static beliefs) | No | Weights ∈ [0,1.5] |
| Full | 7 | Yes | Yes | Weights ∈ [0,1.5], α ∈ [0,1] |

### Identifiability Relationships

**Parameter-to-Data Ratio:**

Assuming average $T_{train} = 9$ trials per person:

| Model | Parameters | Observations | Ratio |
|-------|------------|--------------|-------|
| Null | 0 | 9 | ∞ |
| Logistic | 9 | 9 | 1.0 |
| No-Learning | 3 | 9 | 3.0 |
| Full | 7 | 9 | 1.3 |

**Identifiability Assessment:**
- Null: Always identifiable (no parameters)
- Logistic: Borderline (ratio = 1)
- No-Learning: Marginal (ratio = 3, typically need >5)
- Full: Poor (ratio = 1.3, typically need >10 for identifiability)

---

## Implementation Details

### Optimization Algorithm

All models use **L-BFGS-B** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds):
- Quasi-Newton optimization method
- Uses gradient information (computed via finite differences)
- Handles box constraints efficiently
- Maximum 1000 iterations

### Parameter Transformations

**For Weights** ($w \in [0, 1.5]$):
```
Forward:  w = 1.5 * σ(ξ)
Backward: ξ = logit(w/1.5) = log(w/1.5) - log(1 - w/1.5)
```

**For Learning Rates** ($\alpha \in [0, 1]$):
```
Forward:  α = σ(ζ)
Backward: ζ = logit(α) = log(α) - log(1 - α)
```

Where $\sigma(x) = \frac{1}{1+e^{-x}}$ is the logistic sigmoid.

### Numerical Stability

**Softmax computation** uses log-sum-exp trick:
```python
def softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
```

**Log-probability accumulation** adds small constant:
```python
logp += np.log(prob + 1e-10)  # Avoid log(0)
```

### Convergence Criteria

Optimization terminates when:
1. Gradient norm < $10^{-5}$, or
2. Function change < $10^{-9}$, or  
3. Maximum iterations (1000) reached

---

## Evaluation Metrics

### 1. Accuracy

$$\text{Accuracy} = \frac{1}{T_{test}} \sum_{t=1}^{T_{test}} \mathbb{1}[\hat{c}_t = c_t]$$

where $\mathbb{1}[\cdot]$ is the indicator function and $\hat{c}_t$ is predicted choice:

$$\hat{c}_t = \begin{cases}
1 & \text{if } P(c_t = 1) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}$$

### 2. Log Loss (Cross-Entropy)

$$\text{LogLoss} = -\frac{1}{T_{test}} \sum_{t=1}^{T_{test}} \left[ c_t \log p_t + (1-c_t) \log(1-p_t) \right]$$

where $p_t = P(c_t = 1)$ is predicted probability.

**Lower is better.** Measures probabilistic calibration.

### 3. Area Under ROC Curve (AUC)

AUC measures discrimination ability—how well the model ranks positive examples above negative examples.

$$\text{AUC} = P(\text{score}(c=1) > \text{score}(c=0))$$

Computed via trapezoidal rule over ROC curve.

**Range**: [0, 1], where 0.5 = random, 1.0 = perfect discrimination.

### 4. Brier Score

$$\text{Brier} = \frac{1}{T_{test}} \sum_{t=1}^{T_{test}} (c_t - p_t)^2$$

Measures mean squared error of probabilistic predictions.

**Lower is better.** Range: [0, 1].

---

## Statistical Comparisons

### Paired t-test

For metrics computed per-person, we use paired t-test:

$$t = \frac{\bar{d}}{SE(\bar{d})} = \frac{\bar{d}}{s_d / \sqrt{n}}$$

where:
- $d_i = \text{metric}_{\text{model1},i} - \text{metric}_{\text{model2},i}$ for person $i$
- $\bar{d}$ = mean difference
- $s_d$ = standard deviation of differences
- $n$ = number of people

Null hypothesis: $H_0: \bar{d} = 0$ (models perform equally)

### Wilcoxon Signed-Rank Test

Non-parametric alternative for non-normal distributions:

1. Compute differences $d_i$
2. Rank absolute differences $|d_i|$
3. Sum ranks for positive vs negative differences
4. Test against null of symmetric distribution around 0

Used as robustness check when $n$ is small or normality questionable.

---

## Assumptions and Limitations

### Model Assumptions

**Null Model:**
- ✓ Assumes stationarity (choice probability constant over time)
- ✓ Assumes no effect of covariates
- ✗ Will fail if behavior changes systematically

**Logistic Regression:**
- ✓ Assumes linear relationship between log-odds and features
- ✓ Assumes independence across trials (given features)
- ✗ Ignores temporal dependencies
- ✗ Assumes belief states constant

**No-Learning Model:**
- ✓ Assumes weighted combination of I/E/U determines choices
- ✓ Assumes beliefs don't change (no learning)
- ✗ Will fail if people adapt over time
- ⚠ Requires sufficient observations to estimate 3 weights

**Full Model:**
- ✓ Assumes trial-by-trial learning from outcomes
- ✓ Assumes prediction-error driven updates
- ✓ Assumes softmax choice rule
- ✗ Requires many observations (7 parameters)
- ⚠ Vulnerable to overfitting with sparse data
- ⚠ Assumes learning rates constant across trials

### Data Requirements

**Minimum Sample Sizes** (rule of thumb: 10 observations per parameter):

| Model | Parameters | Minimum Observations | Current Data |
|-------|------------|---------------------|--------------|
| Null | 0 | ~5 | ✓ 9-18 |
| Logistic | 9 | ~90 | ✗ 9-18 |
| No-Learning | 3 | ~30 | ✗ 9-18 |
| Full | 7 | ~70 | ✗ 9-18 |

**Current data is insufficient for all models except Null.**

---

## Connection to Original Implementation

### Hierarchical Bayesian Version

The original `hierarchical_recovery_v3.py` implements a hierarchical Bayesian version of the Full Model:

**Differences:**
1. **Hierarchical structure**: Person-level parameters drawn from population distributions
2. **Bayesian inference**: Full posterior over parameters, not point estimates
3. **Outcome likelihoods**: Models enjoyment and utility outcomes, not just choices
4. **MCMC sampling**: Metropolis-Hastings instead of optimization

**Mathematical additions:**

Population-level hyperpriors:
$$\mu_w \sim \mathcal{N}(0, 1), \quad \sigma_w \sim \text{HalfNormal}(0.5)$$

Person-level parameters:
$$w_{I,i} \sim \mathcal{N}(\mu_{wI}, \sigma_{wI})$$

Outcome likelihoods:
$$e_t | c_t \sim \mathcal{N}(E_{c_t}, \sigma_e)$$
$$u_t | c_t \sim \mathcal{N}(U_{c_t}, \sigma_u)$$

The prediction validation script simplifies to per-person MLE for computational efficiency.

---

## Summary

| Aspect | Null | Logistic | No-Learning | Full |
|--------|------|----------|-------------|------|
| **Paradigm** | Frequentist | Frequentist | Computational | Computational |
| **Complexity** | Minimal | Moderate | Moderate | High |
| **Temporal** | No | No | No | Yes |
| **Learning** | No | No | No | Yes |
| **Theory-driven** | No | No | Yes | Yes |
| **Interpretability** | High | High | High | Moderate |
| **Data needs** | Very low | High | Moderate | Very high |
| **Current fit** | Good | Poor | Poor | Very poor |

**Key Finding**: Null model outperforms all others, suggesting behavior is highly stable and choice-dominant patterns are better predictors than any features or learning dynamics.

---

## References

### Relevant Literature

1. **Softmax choice rule**: Luce (1959) Individual Choice Behavior
2. **Prediction error learning**: Rescorla & Wagner (1972)
3. **Logistic regression**: Cox (1958) regression models for binary data
4. **Model comparison**: Pitt & Myung (2002) When a good fit can be bad
5. **Computational modeling**: Daw (2011) Trial-by-trial data analysis

### Implementation

- Python 3.9
- NumPy for numerical computation
- SciPy for optimization (L-BFGS-B)
- Scikit-learn for logistic regression
- Full code: `experiments/validate_predictions.py`

---

*Document created: November 11, 2025*
*Last updated: November 11, 2025*
