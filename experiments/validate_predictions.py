"""
Predictive validity testing for computational model.

Compares predictive performance of:
1. Null model (marginal probability)
2. Logistic regression (no temporal dynamics)
3. No-learning model (fixed beliefs)
4. Full model (with learning)

Uses train/test split to evaluate out-of-sample prediction accuracy.
"""
import argparse
import pathlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

BEH_KEYS = ["avoid_conflict", "approach_conflict_care"]


def load_data(events_path: str, people_path: str):
    """Load and prepare event and person-level data."""
    df = pd.read_csv(events_path)
    ppl = pd.read_csv(people_path)
    
    df = df.sort_values(["person_id", "t"]).reset_index(drop=True)
    persons = sorted(df.person_id.unique())
    
    per_person = []
    for p in persons:
        ev = df[df.person_id == p].copy()
        T = len(ev)
        
        # Choices: 0 = avoid_conflict, 1 = approach_conflict_care
        choice = (ev["choice_behavior"].values == BEH_KEYS[1]).astype(np.int32)
        
        # Suggestion terms
        sug0 = ev.get(f"suggest_term_{BEH_KEYS[0]}", pd.Series([0.0] * T)).fillna(0.0).values
        sug1 = ev.get(f"suggest_term_{BEH_KEYS[1]}", pd.Series([0.0] * T)).fillna(0.0).values
        suggestion = np.stack([sug0, sug1], axis=1).astype(np.float64)
        
        # Outcomes
        e_out = ev["enjoyment_out"].values.astype(np.float64)
        u_out = ev["utility_out"].values.astype(np.float64)
        
        # Initial states from person data
        base = ppl.loc[ppl.person_id == p].iloc[0]
        inst0 = np.array([base[f"instinct_{k}_0"] for k in BEH_KEYS], dtype=np.float64)
        enj0 = np.array([base[f"enjoyment_{k}_0"] for k in BEH_KEYS], dtype=np.float64)
        uti0 = np.array([base[f"utility_{k}_0"] for k in BEH_KEYS], dtype=np.float64)
        
        # True parameters (for reference)
        true_params = {
            "w_I": float(base.get("w_I", 0.5)),
            "w_E": float(base.get("w_E", 0.5)),
            "w_U": float(base.get("w_U", 0.5)),
            "aI_pos": float(base.get("alpha_I_pos", 0.1)),
            "aI_neg": float(base.get("alpha_I_neg", 0.1)),
            "a_E": float(base.get("alpha_E", 0.1)),
            "a_U": float(base.get("alpha_U", 0.1)),
        }
        
        per_person.append({
            "person_id": int(p),
            "T": T,
            "choice": choice,
            "suggestion": suggestion,
            "e_out": e_out,
            "u_out": u_out,
            "inst0": inst0,
            "enj0": enj0,
            "uti0": uti0,
            "true_params": true_params,
        })
    
    return per_person, ppl


def simulate_forward_predict(w_I, w_E, w_U, aI_pos, aI_neg, a_E, a_U, tau,
                             inst0, enj0, uti0, suggestion, choice, e_out, u_out,
                             return_probs=False):
    """
    Forward simulation that returns choice probabilities at each time step.
    
    Args:
        w_I, w_E, w_U: Weights for instinct, enjoyment, utility
        aI_pos, aI_neg: Learning rates for instinct
        a_E, a_U: Learning rates for enjoyment and utility
        tau: Softmax temperature
        inst0, enj0, uti0: Initial belief states
        suggestion, choice, e_out, u_out: Observed data
        return_probs: If True, return (logp, probs_history), else just logp
        
    Returns:
        logp: Total log probability (negative log-likelihood)
        probs_history: (T, 2) array of choice probabilities at each time step
    """
    inst = inst0.copy()
    enj = enj0.copy()
    uti = uti0.copy()
    
    logp = 0.0
    T = len(choice)
    probs_history = np.zeros((T, 2))
    
    for t in range(T):
        # Compute choice values
        CV = w_I * inst + w_E * enj + w_U * uti + suggestion[t]
        
        # Softmax to get probabilities
        exp_vals = np.exp(tau * CV)
        probs = exp_vals / np.sum(exp_vals)
        probs_history[t] = probs
        
        # Choice log likelihood
        ct = choice[t]
        logp += np.log(probs[ct] + 1e-10)
        
        # Update belief states
        mask = np.array([1.0 if i == ct else 0.0 for i in range(2)])
        
        # Instinct update
        inst = inst + mask * aI_pos * (1.0 - inst) + (1.0 - mask) * aI_neg * (-1.0 - inst)
        
        # Enjoyment and utility updates
        enj = enj + mask * a_E * (e_out[t] - enj)
        uti = uti + mask * a_U * (u_out[t] - uti)
        
        # Clip to [-1, 1]
        inst = np.clip(inst, -1.0, 1.0)
        enj = np.clip(enj, -1.0, 1.0)
        uti = np.clip(uti, -1.0, 1.0)
    
    if return_probs:
        return logp, probs_history
    return logp


def fit_full_model_mle(person_data, tau=3.0, max_iter=1000):
    """
    Fit full computational model using Maximum Likelihood Estimation.
    
    Returns:
        params: Dict with fitted parameter values
        neg_loglik: Negative log-likelihood at optimal parameters
    """
    # Initial parameter guess (transformed to unconstrained space)
    # Use logit transform: param = sigmoid(x) => x = logit(param)
    def logit(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))
    
    # Starting values: moderate weights and learning rates
    x0 = np.array([
        logit(0.5),   # w_I (will be scaled to [0, 1.5])
        logit(0.5),   # w_E
        logit(0.5),   # w_U
        logit(0.2),   # aI_pos
        logit(0.2),   # aI_neg
        logit(0.2),   # a_E
        logit(0.2),   # a_U
    ])
    
    def objective(x):
        """Negative log-likelihood."""
        # Transform parameters
        w_I = 1.5 * expit(x[0])
        w_E = 1.5 * expit(x[1])
        w_U = 1.5 * expit(x[2])
        aI_pos = expit(x[3])
        aI_neg = expit(x[4])
        a_E = expit(x[5])
        a_U = expit(x[6])
        
        # Compute log-likelihood (only on choices, ignore outcomes for speed)
        logp = simulate_forward_predict(
            w_I, w_E, w_U, aI_pos, aI_neg, a_E, a_U, tau,
            person_data["inst0"], person_data["enj0"], person_data["uti0"],
            person_data["suggestion"], person_data["choice"],
            person_data["e_out"], person_data["u_out"]
        )
        
        return -logp  # Return negative for minimization
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', 
                     options={'maxiter': max_iter, 'disp': False})
    
    # Extract fitted parameters
    x_opt = result.x
    params = {
        "w_I": 1.5 * expit(x_opt[0]),
        "w_E": 1.5 * expit(x_opt[1]),
        "w_U": 1.5 * expit(x_opt[2]),
        "aI_pos": expit(x_opt[3]),
        "aI_neg": expit(x_opt[4]),
        "a_E": expit(x_opt[5]),
        "a_U": expit(x_opt[6]),
    }
    
    return params, result.fun


def fit_no_learning_model_mle(person_data, tau=3.0, max_iter=1000):
    """
    Fit model with learning rates fixed at 0.
    Only estimates weights and initial beliefs.
    """
    def logit(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))
    
    x0 = np.array([
        logit(0.5),   # w_I
        logit(0.5),   # w_E
        logit(0.5),   # w_U
    ])
    
    def objective(x):
        w_I = 1.5 * expit(x[0])
        w_E = 1.5 * expit(x[1])
        w_U = 1.5 * expit(x[2])
        
        # Learning rates = 0 (no learning)
        logp = simulate_forward_predict(
            w_I, w_E, w_U, 0.0, 0.0, 0.0, 0.0, tau,
            person_data["inst0"], person_data["enj0"], person_data["uti0"],
            person_data["suggestion"], person_data["choice"],
            person_data["e_out"], person_data["u_out"]
        )
        
        return -logp
    
    result = minimize(objective, x0, method='L-BFGS-B',
                     options={'maxiter': max_iter, 'disp': False})
    
    x_opt = result.x
    params = {
        "w_I": 1.5 * expit(x_opt[0]),
        "w_E": 1.5 * expit(x_opt[1]),
        "w_U": 1.5 * expit(x_opt[2]),
        "aI_pos": 0.0,
        "aI_neg": 0.0,
        "a_E": 0.0,
        "a_U": 0.0,
    }
    
    return params, result.fun


def predict_computational_model(params, person_data, tau=3.0):
    """
    Generate predictions using fitted computational model.
    
    Returns:
        probs: (T, 2) array of choice probabilities
    """
    _, probs = simulate_forward_predict(
        params["w_I"], params["w_E"], params["w_U"],
        params["aI_pos"], params["aI_neg"], 
        params["a_E"], params["a_U"],
        tau,
        person_data["inst0"], person_data["enj0"], person_data["uti0"],
        person_data["suggestion"], person_data["choice"],
        person_data["e_out"], person_data["u_out"],
        return_probs=True
    )
    return probs


def train_test_split(person_data, train_frac=0.7):
    """
    Split person's data into train and test sets.
    
    Returns:
        train_data, test_data: Dicts with same structure as person_data
    """
    T = person_data["T"]
    split_idx = int(T * train_frac)
    
    train_data = {
        "person_id": person_data["person_id"],
        "T": split_idx,
        "choice": person_data["choice"][:split_idx],
        "suggestion": person_data["suggestion"][:split_idx],
        "e_out": person_data["e_out"][:split_idx],
        "u_out": person_data["u_out"][:split_idx],
        "inst0": person_data["inst0"].copy(),
        "enj0": person_data["enj0"].copy(),
        "uti0": person_data["uti0"].copy(),
        "true_params": person_data["true_params"],
    }
    
    test_data = {
        "person_id": person_data["person_id"],
        "T": T - split_idx,
        "choice": person_data["choice"][split_idx:],
        "suggestion": person_data["suggestion"][split_idx:],
        "e_out": person_data["e_out"][split_idx:],
        "u_out": person_data["u_out"][split_idx:],
        "inst0": person_data["inst0"].copy(),
        "enj0": person_data["enj0"].copy(),
        "uti0": person_data["uti0"].copy(),
        "true_params": person_data["true_params"],
    }
    
    return train_data, test_data


def evaluate_model(y_true, y_pred_proba):
    """
    Compute evaluation metrics.
    
    Args:
        y_true: (N,) array of true labels (0 or 1)
        y_pred_proba: (N,) array of predicted probabilities for class 1
        
    Returns:
        metrics: Dict with accuracy, log_loss, auc, brier_score
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Check if we have both classes (needed for AUC and log_loss)
    n_classes = len(np.unique(y_true))
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }
    
    # Log loss and AUC require both classes
    if n_classes > 1:
        try:
            metrics["log_loss"] = log_loss(y_true, y_pred_proba, labels=[0, 1])
            metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics["log_loss"] = np.nan
            metrics["auc"] = np.nan
    else:
        metrics["log_loss"] = np.nan
        metrics["auc"] = np.nan
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Predictive validity testing")
    parser.add_argument("--events", default="outputs/ema_events.csv")
    parser.add_argument("--people", default="outputs/ema_people.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--train_frac", type=float, default=0.7,
                       help="Fraction of data for training (rest for testing)")
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREDICTIVE VALIDITY TESTING")
    print("="*70)
    print(f"Train fraction: {args.train_frac:.1%}")
    print(f"Test fraction: {1-args.train_frac:.1%}")
    print()
    
    # Load data
    print("Loading data...")
    per_person, _ = load_data(args.events, args.people)
    print(f"Loaded {len(per_person)} people")
    print()
    
    # Collect results
    results = []
    
    for i, person_data in enumerate(per_person):
        pid = person_data["person_id"]
        print(f"Processing Person {pid} ({i+1}/{len(per_person)})...")
        
        # Split data
        train_data, test_data = train_test_split(person_data, args.train_frac)
        print(f"  Train: {train_data['T']} trials, Test: {test_data['T']} trials")
        
        if test_data["T"] < 3:
            print(f"  ⚠️  Skipping (too few test trials)")
            continue
        
        # Check if training data has both classes
        if len(np.unique(train_data["choice"])) < 2:
            print(f"  ⚠️  Skipping (training data has only one class)")
            continue
        
        # Ground truth
        y_test = test_data["choice"]
        
        # ==========================
        # 1. NULL MODEL
        # ==========================
        # Predict marginal probability from training set
        p_train = np.mean(train_data["choice"])
        y_pred_null = np.full(test_data["T"], p_train)
        metrics_null = evaluate_model(y_test, y_pred_null)
        print(f"  Null model: Acc={metrics_null['accuracy']:.3f}, LogLoss={metrics_null['log_loss']:.3f}")
        
        # ==========================
        # 2. LOGISTIC REGRESSION
        # ==========================
        # Features: suggestion terms + initial beliefs (no temporal dynamics)
        X_train = np.column_stack([
            train_data["suggestion"],
            np.tile(train_data["inst0"], (train_data["T"], 1)),
            np.tile(train_data["enj0"], (train_data["T"], 1)),
            np.tile(train_data["uti0"], (train_data["T"], 1)),
        ])
        X_test = np.column_stack([
            test_data["suggestion"],
            np.tile(test_data["inst0"], (test_data["T"], 1)),
            np.tile(test_data["enj0"], (test_data["T"], 1)),
            np.tile(test_data["uti0"], (test_data["T"], 1)),
        ])
        
        lr_model = LogisticRegression(max_iter=1000, random_state=args.seed)
        lr_model.fit(X_train, train_data["choice"])
        y_pred_lr = lr_model.predict_proba(X_test)[:, 1]
        metrics_lr = evaluate_model(y_test, y_pred_lr)
        print(f"  Logistic Reg: Acc={metrics_lr['accuracy']:.3f}, LogLoss={metrics_lr['log_loss']:.3f}")
        
        # ==========================
        # 3. NO-LEARNING MODEL
        # ==========================
        print("  Fitting no-learning model...")
        params_no_learn, _ = fit_no_learning_model_mle(train_data, tau=args.tau)
        
        # Predict on test set
        probs_no_learn = predict_computational_model(params_no_learn, test_data, tau=args.tau)
        y_pred_no_learn = probs_no_learn[:, 1]  # Probability of choice=1
        metrics_no_learn = evaluate_model(y_test, y_pred_no_learn)
        print(f"  No-learning:  Acc={metrics_no_learn['accuracy']:.3f}, LogLoss={metrics_no_learn['log_loss']:.3f}")
        
        # ==========================
        # 4. FULL MODEL (with learning)
        # ==========================
        print("  Fitting full model...")
        params_full, _ = fit_full_model_mle(train_data, tau=args.tau)
        
        # Predict on test set
        probs_full = predict_computational_model(params_full, test_data, tau=args.tau)
        y_pred_full = probs_full[:, 1]
        metrics_full = evaluate_model(y_test, y_pred_full)
        print(f"  Full model:   Acc={metrics_full['accuracy']:.3f}, LogLoss={metrics_full['log_loss']:.3f}")
        
        # Store results
        results.append({
            "person_id": pid,
            "n_train": train_data["T"],
            "n_test": test_data["T"],
            # Null model
            "null_accuracy": metrics_null["accuracy"],
            "null_log_loss": metrics_null["log_loss"],
            "null_auc": metrics_null["auc"],
            "null_brier": metrics_null["brier_score"],
            # Logistic regression
            "lr_accuracy": metrics_lr["accuracy"],
            "lr_log_loss": metrics_lr["log_loss"],
            "lr_auc": metrics_lr["auc"],
            "lr_brier": metrics_lr["brier_score"],
            # No-learning model
            "no_learn_accuracy": metrics_no_learn["accuracy"],
            "no_learn_log_loss": metrics_no_learn["log_loss"],
            "no_learn_auc": metrics_no_learn["auc"],
            "no_learn_brier": metrics_no_learn["brier_score"],
            # Full model
            "full_accuracy": metrics_full["accuracy"],
            "full_log_loss": metrics_full["log_loss"],
            "full_auc": metrics_full["auc"],
            "full_brier": metrics_full["brier_score"],
            # Fitted parameters (full model)
            **{f"full_{k}": v for k, v in params_full.items()},
            # Fitted parameters (no-learning)
            **{f"no_learn_{k}": v for k, v in params_no_learn.items()},
            # True parameters
            **{f"true_{k}": v for k, v in person_data["true_params"].items()},
        })
        
        print()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = outdir / "prediction_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to: {results_path}")
    
    # Compute aggregate statistics
    print()
    print("="*70)
    print("AGGREGATE RESULTS (Mean ± SE across all people)")
    print("="*70)
    
    metrics = ["accuracy", "log_loss", "auc", "brier"]
    models = ["null", "lr", "no_learn", "full"]
    model_names = {
        "null": "Null Model",
        "lr": "Logistic Regression",
        "no_learn": "No-Learning Model",
        "full": "Full Model (with learning)",
    }
    
    summary = []
    for model in models:
        print(f"\n{model_names[model]}:")
        row = {"model": model}
        for metric in metrics:
            col = f"{model}_{metric}"
            if col in results_df.columns:
                mean_val = results_df[col].mean()
                se_val = results_df[col].sem()
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_se"] = se_val
                
                # Format based on metric
                if metric == "accuracy":
                    print(f"  Accuracy:     {mean_val:.3f} ± {se_val:.3f}")
                elif metric == "log_loss":
                    print(f"  Log Loss:     {mean_val:.3f} ± {se_val:.3f} (lower is better)")
                elif metric == "auc":
                    print(f"  AUC:          {mean_val:.3f} ± {se_val:.3f}")
                elif metric == "brier":
                    print(f"  Brier Score:  {mean_val:.3f} ± {se_val:.3f} (lower is better)")
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    summary_path = outdir / "prediction_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    
    # Pairwise comparisons
    print()
    print("="*70)
    print("PAIRWISE COMPARISONS (Paired t-tests)")
    print("="*70)
    
    from scipy.stats import ttest_rel
    
    comparisons = [
        ("full", "null", "Full vs Null"),
        ("full", "lr", "Full vs Logistic Regression"),
        ("full", "no_learn", "Full vs No-Learning"),
        ("no_learn", "null", "No-Learning vs Null"),
    ]
    
    for model1, model2, label in comparisons:
        print(f"\n{label}:")
        for metric in ["accuracy", "log_loss"]:
            col1 = f"{model1}_{metric}"
            col2 = f"{model2}_{metric}"
            
            if col1 in results_df.columns and col2 in results_df.columns:
                vals1 = results_df[col1].values
                vals2 = results_df[col2].values
                
                diff = vals1 - vals2
                mean_diff = diff.mean()
                
                t_stat, p_val = ttest_rel(vals1, vals2)
                
                better = "better" if (mean_diff > 0 and metric == "accuracy") or (mean_diff < 0 and metric == "log_loss") else "worse"
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                print(f"  {metric:12s}: Δ={mean_diff:+.3f}, t={t_stat:.2f}, p={p_val:.4f} {sig} ({better})")
    
    print()
    print("="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
