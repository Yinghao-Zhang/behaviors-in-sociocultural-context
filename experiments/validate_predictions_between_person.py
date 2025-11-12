"""
Predictive validity testing with BETWEEN-PERSON train/test split.

Instead of splitting trials within each person (which breaks temporal dependencies),
we split the PEOPLE into training and test sets:
- Training set: Fit model, tune hyperparameters, estimate population parameters
- Test set: Apply learned model to new people, evaluate generalization

This tests: "Can we predict a NEW person's behavior using parameters learned from others?"
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
    """
    def logit(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))
    
    # Starting values
    x0 = np.array([
        logit(0.5),   # w_I
        logit(0.5),   # w_E
        logit(0.5),   # w_U
        logit(0.2),   # aI_pos
        logit(0.2),   # aI_neg
        logit(0.2),   # a_E
        logit(0.2),   # a_U
    ])
    
    def objective(x):
        """Negative log-likelihood."""
        w_I = 1.5 * expit(x[0])
        w_E = 1.5 * expit(x[1])
        w_U = 1.5 * expit(x[2])
        aI_pos = expit(x[3])
        aI_neg = expit(x[4])
        a_E = expit(x[5])
        a_U = expit(x[6])
        
        logp = simulate_forward_predict(
            w_I, w_E, w_U, aI_pos, aI_neg, a_E, a_U, tau,
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
        "aI_pos": expit(x_opt[3]),
        "aI_neg": expit(x_opt[4]),
        "a_E": expit(x_opt[5]),
        "a_U": expit(x_opt[6]),
    }
    
    return params, result.fun


def fit_no_learning_model_mle(person_data, tau=3.0, max_iter=1000):
    """Fit model with learning rates fixed at 0."""
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
    """Generate predictions using computational model."""
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


def evaluate_model(y_true, y_pred_proba):
    """Compute evaluation metrics."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    n_classes = len(np.unique(y_true))
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }
    
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
    parser = argparse.ArgumentParser(description="Predictive validity testing with between-person split")
    parser.add_argument("--events", default="outputs/ema_events.csv")
    parser.add_argument("--people", default="outputs/ema_people.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--train_frac", type=float, default=0.8,
                       help="Fraction of people for training (rest for testing)")
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREDICTIVE VALIDITY TESTING (BETWEEN-PERSON SPLIT)")
    print("="*70)
    print(f"Train fraction: {args.train_frac:.1%} of people")
    print(f"Test fraction: {1-args.train_frac:.1%} of people")
    print()
    
    # Load data
    print("Loading data...")
    per_person, _ = load_data(args.events, args.people)
    print(f"Loaded {len(per_person)} people")
    
    # Split people into train and test
    n_people = len(per_person)
    n_train = int(n_people * args.train_frac)
    n_test = n_people - n_train
    
    # Shuffle people for random split
    person_indices = np.random.permutation(n_people)
    train_indices = person_indices[:n_train]
    test_indices = person_indices[n_train:]
    
    train_people = [per_person[i] for i in train_indices]
    test_people = [per_person[i] for i in test_indices]
    
    print(f"\nTrain set: {n_train} people ({sum(p['T'] for p in train_people)} total trials)")
    print(f"Test set:  {n_test} people ({sum(p['T'] for p in test_people)} total trials)")
    print()
    
    # ================================================================
    # PHASE 1: FIT MODELS ON TRAINING SET
    # ================================================================
    
    print("="*70)
    print("PHASE 1: TRAINING (Fitting models to training people)")
    print("="*70)
    print()
    
    train_results = {
        'full': [],
        'no_learn': [],
    }
    
    for i, person_data in enumerate(train_people):
        pid = person_data["person_id"]
        print(f"Training on Person {pid} ({i+1}/{n_train}, T={person_data['T']})...")
        
        # Skip if too few trials or single class
        if person_data["T"] < 5 or len(np.unique(person_data["choice"])) < 2:
            print(f"  ⚠️  Skipping (insufficient data)")
            continue
        
        # Fit full model
        print("  Fitting full model...")
        params_full, neg_ll_full = fit_full_model_mle(person_data, tau=args.tau)
        train_results['full'].append({
            'person_id': pid,
            **params_full,
            'neg_log_lik': neg_ll_full
        })
        
        # Fit no-learning model
        print("  Fitting no-learning model...")
        params_no_learn, neg_ll_no_learn = fit_no_learning_model_mle(person_data, tau=args.tau)
        train_results['no_learn'].append({
            'person_id': pid,
            **params_no_learn,
            'neg_log_lik': neg_ll_no_learn
        })
        
        print()
    
    # Compute population statistics from training set
    print("="*70)
    print("POPULATION PARAMETERS (from training set)")
    print("="*70)
    
    pop_params_full = {}
    pop_params_no_learn = {}
    
    if len(train_results['full']) > 0:
        df_train_full = pd.DataFrame(train_results['full'])
        print("\nFull Model:")
        for param in ['w_I', 'w_E', 'w_U', 'aI_pos', 'aI_neg', 'a_E', 'a_U']:
            mean_val = df_train_full[param].mean()
            std_val = df_train_full[param].std()
            pop_params_full[param] = mean_val
            print(f"  {param:10s}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    if len(train_results['no_learn']) > 0:
        df_train_no_learn = pd.DataFrame(train_results['no_learn'])
        print("\nNo-Learning Model:")
        for param in ['w_I', 'w_E', 'w_U']:
            mean_val = df_train_no_learn[param].mean()
            std_val = df_train_no_learn[param].std()
            pop_params_no_learn[param] = mean_val
            print(f"  {param:10s}: μ={mean_val:.3f}, σ={std_val:.3f}")
        # Add zero learning rates
        for param in ['aI_pos', 'aI_neg', 'a_E', 'a_U']:
            pop_params_no_learn[param] = 0.0
    
    print()
    
    # ================================================================
    # PHASE 2: EVALUATE ON TEST SET
    # ================================================================
    
    print("="*70)
    print("PHASE 2: TESTING (Evaluating on held-out people)")
    print("="*70)
    print()
    
    test_results = []
    
    for i, person_data in enumerate(test_people):
        pid = person_data["person_id"]
        T = person_data["T"]
        print(f"Testing on Person {pid} ({i+1}/{n_test}, T={T})...")
        
        # Skip if insufficient data
        if T < 3 or len(np.unique(person_data["choice"])) < 2:
            print(f"  ⚠️  Skipping (insufficient data)")
            continue
        
        y_true = person_data["choice"]
        
        # ==========================
        # 1. NULL MODEL (marginal from training set)
        # ==========================
        all_train_choices = np.concatenate([p["choice"] for p in train_people])
        p_train = np.mean(all_train_choices)
        y_pred_null = np.full(T, p_train)
        metrics_null = evaluate_model(y_true, y_pred_null)
        print(f"  Null model:       Acc={metrics_null['accuracy']:.3f}, LogLoss={metrics_null['log_loss']:.3f}")
        
        # ==========================
        # 2. LOGISTIC REGRESSION (trained on all training people)
        # ==========================
        # Aggregate all training data
        X_train_all = []
        y_train_all = []
        for p in train_people:
            if p["T"] >= 5 and len(np.unique(p["choice"])) >= 2:
                X_p = np.column_stack([
                    p["suggestion"],
                    np.tile(p["inst0"], (p["T"], 1)),
                    np.tile(p["enj0"], (p["T"], 1)),
                    np.tile(p["uti0"], (p["T"], 1)),
                ])
                X_train_all.append(X_p)
                y_train_all.append(p["choice"])
        
        if len(X_train_all) > 0:
            X_train_all = np.vstack(X_train_all)
            y_train_all = np.concatenate(y_train_all)
            
            # Fit logistic regression
            lr_model = LogisticRegression(max_iter=1000, random_state=args.seed)
            lr_model.fit(X_train_all, y_train_all)
            
            # Predict on test person
            X_test = np.column_stack([
                person_data["suggestion"],
                np.tile(person_data["inst0"], (T, 1)),
                np.tile(person_data["enj0"], (T, 1)),
                np.tile(person_data["uti0"], (T, 1)),
            ])
            y_pred_lr = lr_model.predict_proba(X_test)[:, 1]
            metrics_lr = evaluate_model(y_true, y_pred_lr)
            print(f"  Logistic Reg:     Acc={metrics_lr['accuracy']:.3f}, LogLoss={metrics_lr['log_loss']:.3f}")
        else:
            metrics_lr = {k: np.nan for k in ['accuracy', 'log_loss', 'auc', 'brier_score']}
            print(f"  Logistic Reg:     SKIPPED (no valid training data)")
        
        # ==========================
        # 3. NO-LEARNING MODEL (population params from training)
        # ==========================
        if len(pop_params_no_learn) > 0:
            probs_no_learn = predict_computational_model(pop_params_no_learn, person_data, tau=args.tau)
            y_pred_no_learn = probs_no_learn[:, 1]
            metrics_no_learn = evaluate_model(y_true, y_pred_no_learn)
            print(f"  No-learning (pop): Acc={metrics_no_learn['accuracy']:.3f}, LogLoss={metrics_no_learn['log_loss']:.3f}")
        else:
            metrics_no_learn = {k: np.nan for k in ['accuracy', 'log_loss', 'auc', 'brier_score']}
            print(f"  No-learning (pop): SKIPPED (no training params)")
        
        # ==========================
        # 4. FULL MODEL (population params from training)
        # ==========================
        if len(pop_params_full) > 0:
            probs_full = predict_computational_model(pop_params_full, person_data, tau=args.tau)
            y_pred_full = probs_full[:, 1]
            metrics_full = evaluate_model(y_true, y_pred_full)
            print(f"  Full model (pop):  Acc={metrics_full['accuracy']:.3f}, LogLoss={metrics_full['log_loss']:.3f}")
        else:
            metrics_full = {k: np.nan for k in ['accuracy', 'log_loss', 'auc', 'brier_score']}
            print(f"  Full model (pop):  SKIPPED (no training params)")
        
        # Store results
        test_results.append({
            "person_id": pid,
            "n_trials": T,
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
            # True parameters (for reference)
            **{f"true_{k}": v for k, v in person_data["true_params"].items()},
        })
        
        print()
    
    # Save results
    if len(test_results) > 0:
        results_df = pd.DataFrame(test_results)
        results_path = outdir / "prediction_validation_between_person.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved test results to: {results_path}")
        
        # Compute aggregate statistics
        print()
        print("="*70)
        print("AGGREGATE TEST SET RESULTS (Mean ± SE)")
        print("="*70)
        
        metrics = ["accuracy", "log_loss", "auc", "brier"]
        models = ["null", "lr", "no_learn", "full"]
        model_names = {
            "null": "Null Model",
            "lr": "Logistic Regression",
            "no_learn": "No-Learning Model (pop params)",
            "full": "Full Model (pop params)",
        }
        
        summary = []
        for model in models:
            print(f"\n{model_names[model]}:")
            row = {"model": model}
            for metric in metrics:
                col = f"{model}_{metric}"
                if col in results_df.columns:
                    vals = results_df[col].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        se_val = vals.sem()
                        row[f"{metric}_mean"] = mean_val
                        row[f"{metric}_se"] = se_val
                        
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
        summary_path = outdir / "prediction_validation_between_person_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
    
    # Save training results
    if len(train_results['full']) > 0:
        train_df_full = pd.DataFrame(train_results['full'])
        train_path_full = outdir / "training_params_full_model.csv"
        train_df_full.to_csv(train_path_full, index=False)
        print(f"\nSaved training params (full model) to: {train_path_full}")
    
    if len(train_results['no_learn']) > 0:
        train_df_no_learn = pd.DataFrame(train_results['no_learn'])
        train_path_no_learn = outdir / "training_params_no_learning.csv"
        train_df_no_learn.to_csv(train_path_no_learn, index=False)
        print(f"Saved training params (no-learning) to: {train_path_no_learn}")
    
    print()
    print("="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
