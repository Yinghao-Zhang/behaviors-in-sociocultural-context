"""
Predictive validity testing with BETWEEN-PERSON train/test split.

Instead of splitting trials within each person (which breaks temporal dependencies),
we split the PEOPLE into training and test sets:
- Training set: Fit model, tune hyperparameters, estimate population parameters
- Test set: Apply learned model to new people, evaluate generalization

This tests: "Can we predict a NEW person's behavior using parameters learned from others?"
"""
import argparse
from functools import lru_cache
import pathlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    LogisticRegression = None
    SKLEARN_AVAILABLE = False
from schema_utils import BEH_KEYS_NEUTRAL, ensure_neutral_behavior_schema

BEH_KEYS = BEH_KEYS_NEUTRAL


def load_data(events_path: str, people_path: str):
    """Load and prepare event and person-level data."""
    df = pd.read_csv(events_path)
    ppl = pd.read_csv(people_path)
    df, ppl = ensure_neutral_behavior_schema(df, ppl)
    if "setup_key" not in df.columns:
        df["setup_key"] = "ema_default"
    if "learning_behavior" not in df.columns:
        df["learning_behavior"] = df["choice_behavior"].where(
            df["choice_behavior"].isin(BEH_KEYS),
            df.get("observed_behavior", pd.Series([np.nan] * len(df))),
        )
    if "learning_role" not in df.columns:
        df["learning_role"] = np.where(df["choice_behavior"].isin(BEH_KEYS), "choice", "observe")

    is_choice = df["choice_behavior"].isin(BEH_KEYS)
    is_learning = df["learning_behavior"].isin(BEH_KEYS)
    df = df[is_choice | is_learning].copy()
    
    df = df.sort_values(["person_id", "t"]).reset_index(drop=True)
    persons = sorted(df.person_id.unique())
    setup_keys = sorted(df["setup_key"].fillna("ema_default").astype(str).unique())
    setup_to_idx = {key: i for i, key in enumerate(setup_keys)}
    
    per_person = []
    for p in persons:
        ev = df[df.person_id == p].copy()
        T = len(ev)
        
        # Choices: 0 = avoid_conflict, 1 = approach_conflict_care
        choice_mask = ev["choice_behavior"].isin(BEH_KEYS).values
        choice = np.full(T, -1, dtype=np.int32)
        choice[choice_mask] = (ev.loc[choice_mask, "choice_behavior"].values == BEH_KEYS[1]).astype(np.int32)

        learning_mask = ev["learning_behavior"].isin(BEH_KEYS).values
        learning_choice = np.full(T, -1, dtype=np.int32)
        learning_choice[learning_mask] = (
            ev.loc[learning_mask, "learning_behavior"].values == BEH_KEYS[1]
        ).astype(np.int32)
        learning_role = ev["learning_role"].fillna("choice").astype(str).values
        setup_idx = ev["setup_key"].fillna("ema_default").astype(str).map(setup_to_idx).values.astype(np.int32)
        
        # Suggestion terms
        sug0 = ev.get(f"suggest_term_{BEH_KEYS[0]}", pd.Series([0.0] * T)).fillna(0.0).values
        sug1 = ev.get(f"suggest_term_{BEH_KEYS[1]}", pd.Series([0.0] * T)).fillna(0.0).values
        suggestion = np.stack([sug0, sug1], axis=1).astype(np.float64)
        
        # Outcomes
        e_out = ev["enjoyment_out"].values.astype(np.float64)
        u_out = ev["utility_out"].values.astype(np.float64)

        reports = np.full((T, len(BEH_KEYS), 3), np.nan, dtype=np.float64)
        for b, behavior_key in enumerate(BEH_KEYS):
            for d, domain in enumerate(["urge", "enjoyment", "utility"]):
                col = f"report_{domain}_{behavior_key}"
                if col in ev.columns:
                    reports[:, b, d] = ev[col].values.astype(np.float64)
        
        # Initial states from person data
        base = ppl.loc[ppl.person_id == p].iloc[0]
        inst0 = np.zeros((len(setup_keys), len(BEH_KEYS)), dtype=np.float64)
        enj0 = np.zeros_like(inst0)
        uti0 = np.zeros_like(inst0)
        for s, setup_key in enumerate(setup_keys):
            for b, behavior_key in enumerate(BEH_KEYS):
                suffix = f"{behavior_key}_{setup_key}_0"
                inst0[s, b] = float(base.get(f"instinct_{suffix}", base.get(f"instinct_{behavior_key}_0", 0.0)))
                enj0[s, b] = float(base.get(f"enjoyment_{suffix}", base.get(f"enjoyment_{behavior_key}_0", 0.0)))
                uti0[s, b] = float(base.get(f"utility_{suffix}", base.get(f"utility_{behavior_key}_0", 0.0)))
        
        tau_val = float(base.get("tau", np.nan))

        # True parameters (for reference)
        true_params = {
            "w_I": float(base.get("w_I", 0.5)),
            "w_E": float(base.get("w_E", 0.5)),
            "w_U": float(base.get("w_U", 0.5)),
            "noise_s": float(base.get("noise_s", 0.0)),
            "social_kappa": float(base.get("social_kappa", 1.0)),
            "aI_pos": float(base.get("alpha_I_pos", 0.1)),
            "aI_neg": float(base.get("alpha_I_neg", 0.1)),
            "a_E": float(base.get("alpha_E", 0.1)),
            "a_U": float(base.get("alpha_U", 0.1)),
        }
        true_params["w_R"] = float(base.get("w_R", true_params["w_E"] + true_params["w_U"]))
        true_params["a_R"] = float(base.get("alpha_R", 0.5 * (true_params["a_E"] + true_params["a_U"])))
        
        per_person.append({
            "person_id": int(p),
            "T": T,
            "n_choice": int(np.sum(choice_mask)),
            "choice": choice,
            "choice_mask": choice_mask,
            "learning_choice": learning_choice,
            "learning_role": learning_role,
            "setup_idx": setup_idx,
            "setup_keys": setup_keys,
            "suggestion": suggestion,
            "e_out": e_out,
            "u_out": u_out,
            "reports": reports,
            "report_mask": np.isfinite(reports),
            "inst0": inst0,
            "enj0": enj0,
            "uti0": uti0,
            "true_params": true_params,
            "tau": tau_val,
        })
    
    return per_person, ppl


def select_tau(person_data, tau_mode: str, default_tau: float) -> float:
    if tau_mode == "person":
        tau_val = person_data.get("tau", np.nan)
        if pd.notna(tau_val):
            return float(tau_val)
    return float(default_tau)


@lru_cache(maxsize=8)
def _cached_hermgauss(gh_n: int):
    return np.polynomial.hermite.hermgauss(gh_n)


def _softmax_binary_prob(delta: float, tau: float, noise_s: float, gh_n: int = 15) -> float:
    if noise_s <= 1e-10:
        return float(expit(tau * delta))
    nodes, weights = _cached_hermgauss(gh_n)
    vals = expit(tau * (delta + 2.0 * noise_s * nodes))
    return float(np.sum(weights * vals) / np.sqrt(np.pi))


def _decision_weights(w_I, w_E, w_U, weight_mode="raw"):
    if weight_mode == "relative":
        total = max(1e-10, w_I + w_E + w_U)
        return w_I / total, w_E / total, w_U / total
    return w_I, w_E, w_U


def _measurement_logp(reports_t, states_t, report_sds, measurement_weight=1.0):
    mask = np.isfinite(reports_t)
    if not np.any(mask):
        return 0.0
    sds = np.array(report_sds, dtype=float)[None, :]
    sigma = np.maximum(1e-3, np.broadcast_to(sds, reports_t.shape))
    resid = reports_t - states_t
    ll = -0.5 * (resid / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))
    return float(measurement_weight * np.sum(ll[mask]))


def simulate_forward_predict(w_I, w_E, w_U, noise_s, aI_pos, aI_neg, a_E, a_U, tau,
                             inst0, enj0, uti0, suggestion, choice, e_out, u_out,
                             choice_mask=None, learning_choice=None, learning_role=None, setup_idx=None,
                             return_probs=False, decision_model="softmax",
                             social_kappa=1.0, weight_mode="raw", reports=None,
                             report_sds=(0.10, 0.10, 0.10), use_measurement_likelihood=False,
                             measurement_weight=1.0, choice_input_mode="latent"):
    """
    Forward simulation that returns choice probabilities at each time step.
    """
    inst = inst0.copy()
    enj = enj0.copy()
    uti = uti0.copy()
    
    logp = 0.0
    T = len(choice)
    probs_history = np.zeros((T, 2))
    if choice_mask is None:
        choice_mask = choice >= 0
    if learning_choice is None:
        learning_choice = choice.copy()
    if learning_role is None:
        learning_role = np.array(["choice"] * T, dtype=object)
    if setup_idx is None:
        setup_idx = np.zeros(T, dtype=np.int32)
    d_w_I, d_w_E, d_w_U = _decision_weights(w_I, w_E, w_U, weight_mode=weight_mode)
    
    for t in range(T):
        sidx = int(setup_idx[t])
        if use_measurement_likelihood and reports is not None:
            states_t = np.stack([inst[sidx], enj[sidx], uti[sidx]], axis=1)
            logp += _measurement_logp(reports[t], states_t, report_sds, measurement_weight)

        choice_inst = inst[sidx]
        choice_enj = enj[sidx]
        choice_uti = uti[sidx]
        if choice_input_mode == "reports" and reports is not None:
            reports_t = reports[t]
            if reports_t.shape == (2, 3):
                choice_inst = np.where(np.isfinite(reports_t[:, 0]), reports_t[:, 0], choice_inst)
                choice_enj = np.where(np.isfinite(reports_t[:, 1]), reports_t[:, 1], choice_enj)
                choice_uti = np.where(np.isfinite(reports_t[:, 2]), reports_t[:, 2], choice_uti)

        if decision_model == "ddm":
            eval_vals = d_w_E * choice_enj + d_w_U * choice_uti
            drift = (eval_vals[1] - eval_vals[0]) + social_kappa * (suggestion[t][1] - suggestion[t][0])
            bias = d_w_I * (choice_inst[1] - choice_inst[0])
            z = 1.0 / (1.0 + np.exp(-bias))
            if abs(drift) < 1e-8:
                p_upper = z
            else:
                scale = 2.0 * drift
                denom = 1.0 - np.exp(-scale)
                p_upper = (1.0 - np.exp(-scale * z)) / denom
            p_upper = np.clip(p_upper, 1e-6, 1.0 - 1e-6)
            probs = np.array([1.0 - p_upper, p_upper], dtype=float)
        else:
            # Compute choice values
            CV = d_w_I * choice_inst + d_w_E * choice_enj + d_w_U * choice_uti + social_kappa * suggestion[t]

            # Integrate Gaussian decision noise deterministically for binary choices.
            p_upper = _softmax_binary_prob(float(CV[1] - CV[0]), tau, noise_s)
            probs = np.array([1.0 - p_upper, p_upper], dtype=float)
        probs_history[t] = probs
        
        # Choice log likelihood
        if choice_mask[t]:
            ct = choice[t]
            logp += np.log(probs[ct] + 1e-10)
        
        # Update belief states
        lt = int(learning_choice[t])
        if lt < 0:
            continue

        gain = 0.5 if learning_role[t] == "observe" else 1.0
        mask = np.array([1.0 if i == lt else 0.0 for i in range(2)])
        
        # Instinct update
        inst[sidx] = inst[sidx] + gain * (
            mask * aI_pos * (1.0 - inst[sidx]) + (1.0 - mask) * aI_neg * (-1.0 - inst[sidx])
        )
        
        # Enjoyment and utility updates
        enj[sidx] = enj[sidx] + gain * mask * a_E * (e_out[t] - enj[sidx])
        uti[sidx] = uti[sidx] + gain * mask * a_U * (u_out[t] - uti[sidx])
        
        # Clip to [-1, 1]
        inst[sidx] = np.clip(inst[sidx], -1.0, 1.0)
        enj[sidx] = np.clip(enj[sidx], -1.0, 1.0)
        uti[sidx] = np.clip(uti[sidx], -1.0, 1.0)
    
    if return_probs:
        return logp, probs_history
    return logp


def simulate_forward_predict_reward(w_I, w_R, social_kappa, noise_s, aI_pos, aI_neg, a_R, tau,
                                    inst0, enj0, uti0, suggestion, choice, e_out, u_out,
                                    choice_mask=None, learning_choice=None, learning_role=None, setup_idx=None,
                                    return_probs=False, decision_model="softmax",
                                    weight_mode="raw", reports=None, report_sds=(0.10, 0.10),
                                    use_measurement_likelihood=False, measurement_weight=1.0,
                                    choice_input_mode="latent"):
    """Forward model for the collapsed instinct + reward model."""
    inst = inst0.copy()
    rew = 0.5 * (enj0.copy() + uti0.copy())

    logp = 0.0
    T = len(choice)
    probs_history = np.zeros((T, 2))
    if choice_mask is None:
        choice_mask = choice >= 0
    if learning_choice is None:
        learning_choice = choice.copy()
    if learning_role is None:
        learning_role = np.array(["choice"] * T, dtype=object)
    if setup_idx is None:
        setup_idx = np.zeros(T, dtype=np.int32)

    if weight_mode == "relative":
        total = max(1e-10, w_I + w_R)
        d_w_I, d_w_R = w_I / total, w_R / total
    else:
        d_w_I, d_w_R = w_I, w_R

    for t in range(T):
        sidx = int(setup_idx[t])
        report_pair = None
        if reports is not None and reports[t].shape == (2, 3):
            report_reward = np.nanmean(reports[t][:, [1, 2]], axis=1)
            report_pair = np.stack([reports[t][:, 0], report_reward], axis=1)

        if use_measurement_likelihood and report_pair is not None:
            states_t = np.stack([inst[sidx], rew[sidx]], axis=1)
            logp += _measurement_logp(report_pair, states_t, report_sds, measurement_weight)

        choice_inst = inst[sidx]
        choice_rew = rew[sidx]
        if choice_input_mode == "reports" and report_pair is not None:
            choice_inst = np.where(np.isfinite(report_pair[:, 0]), report_pair[:, 0], choice_inst)
            choice_rew = np.where(np.isfinite(report_pair[:, 1]), report_pair[:, 1], choice_rew)

        if decision_model == "ddm":
            drift = d_w_R * (choice_rew[1] - choice_rew[0]) + social_kappa * (suggestion[t][1] - suggestion[t][0])
            bias = d_w_I * (choice_inst[1] - choice_inst[0])
            z = 1.0 / (1.0 + np.exp(-bias))
            if abs(drift) < 1e-8:
                p_upper = z
            else:
                scale = 2.0 * drift
                denom = 1.0 - np.exp(-scale)
                p_upper = (1.0 - np.exp(-scale * z)) / denom
            p_upper = np.clip(p_upper, 1e-6, 1.0 - 1e-6)
            probs = np.array([1.0 - p_upper, p_upper], dtype=float)
        else:
            CV = d_w_I * choice_inst + d_w_R * choice_rew + social_kappa * suggestion[t]
            p_upper = _softmax_binary_prob(float(CV[1] - CV[0]), tau, noise_s)
            probs = np.array([1.0 - p_upper, p_upper], dtype=float)
        probs_history[t] = probs

        if choice_mask[t]:
            ct = choice[t]
            logp += np.log(probs[ct] + 1e-10)

        lt = int(learning_choice[t])
        if lt < 0:
            continue

        gain = 0.5 if learning_role[t] == "observe" else 1.0
        mask = np.array([1.0 if i == lt else 0.0 for i in range(2)])
        inst[sidx] = inst[sidx] + gain * (
            mask * aI_pos * (1.0 - inst[sidx]) + (1.0 - mask) * aI_neg * (-1.0 - inst[sidx])
        )
        r_out = 0.5 * (e_out[t] + u_out[t])
        rew[sidx] = rew[sidx] + gain * mask * a_R * (r_out - rew[sidx])
        inst[sidx] = np.clip(inst[sidx], -1.0, 1.0)
        rew[sidx] = np.clip(rew[sidx], -1.0, 1.0)

    if return_probs:
        return logp, probs_history
    return logp


def _relative_weight_transform(x_i, x_e_given_not_i):
    pi_i = float(expit(x_i))
    pi_e_given_not_i = float(expit(x_e_given_not_i))
    remaining = 1.0 - pi_i
    pi_e = remaining * pi_e_given_not_i
    pi_u = remaining * (1.0 - pi_e_given_not_i)
    return pi_i, pi_e, pi_u


def fit_full_model_mle(
    person_data,
    tau=3.0,
    max_iter=1000,
    decision_model="softmax",
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    """
    Fit full computational model using Maximum Likelihood Estimation.
    """
    def logit(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))
    
    def scaled_logit(value, high):
        return logit(value / high)

    if weight_mode == "relative":
        x0 = np.array([
            logit(1.0 / 3.0),  # pi_I
            logit(0.5),        # pi_E within non-instinct share
            logit(0.5),        # social_kappa / 2
            scaled_logit(0.1, 0.5),   # noise_s
            logit(0.2),        # aI_pos
            logit(0.2),        # aI_neg
            logit(0.2),        # a_E
            logit(0.2),        # a_U
        ])
    else:
        x0 = np.array([
            logit(0.5),        # w_I / 1.5
            logit(0.5),        # w_E / 1.5
            logit(0.5),        # w_U / 1.5
            logit(0.5),        # social_kappa / 2
            scaled_logit(0.1, 0.5),   # noise_s
            logit(0.2),        # aI_pos
            logit(0.2),        # aI_neg
            logit(0.2),        # a_E
            logit(0.2),        # a_U
        ])
    
    def objective(x):
        """Negative log-likelihood."""
        if weight_mode == "relative":
            w_I, w_E, w_U = _relative_weight_transform(x[0], x[1])
            social_kappa = 2.0 * expit(x[2])
            noise_s = 0.5 * expit(x[3])
            aI_pos = expit(x[4])
            aI_neg = expit(x[5])
            a_E = expit(x[6])
            a_U = expit(x[7])
        else:
            w_I = 1.5 * expit(x[0])
            w_E = 1.5 * expit(x[1])
            w_U = 1.5 * expit(x[2])
            social_kappa = 2.0 * expit(x[3])
            noise_s = 0.5 * expit(x[4])
            aI_pos = expit(x[5])
            aI_neg = expit(x[6])
            a_E = expit(x[7])
            a_U = expit(x[8])
        
        logp = simulate_forward_predict(
            w_I, w_E, w_U, noise_s, aI_pos, aI_neg, a_E, a_U, tau,
            person_data["inst0"], person_data["enj0"], person_data["uti0"],
            person_data["suggestion"], person_data["choice"],
            person_data["e_out"], person_data["u_out"],
            choice_mask=person_data["choice_mask"],
            learning_choice=person_data["learning_choice"],
            learning_role=person_data["learning_role"],
            setup_idx=person_data["setup_idx"],
            decision_model=decision_model,
            social_kappa=social_kappa,
            weight_mode=weight_mode,
            reports=person_data.get("reports"),
            use_measurement_likelihood=use_measurement_likelihood,
            report_sds=report_sds,
            measurement_weight=measurement_weight,
            choice_input_mode=choice_input_mode,
        )
        
        return -logp
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                     options={'maxiter': max_iter, 'disp': False})
    
    x_opt = result.x
    if weight_mode == "relative":
        w_I, w_E, w_U = _relative_weight_transform(x_opt[0], x_opt[1])
        social_kappa = 2.0 * expit(x_opt[2])
        noise_s = 0.5 * expit(x_opt[3])
        aI_pos = expit(x_opt[4])
        aI_neg = expit(x_opt[5])
        a_E = expit(x_opt[6])
        a_U = expit(x_opt[7])
    else:
        w_I = 1.5 * expit(x_opt[0])
        w_E = 1.5 * expit(x_opt[1])
        w_U = 1.5 * expit(x_opt[2])
        social_kappa = 2.0 * expit(x_opt[3])
        noise_s = 0.5 * expit(x_opt[4])
        aI_pos = expit(x_opt[5])
        aI_neg = expit(x_opt[6])
        a_E = expit(x_opt[7])
        a_U = expit(x_opt[8])
    params = {
        "w_I": w_I,
        "w_E": w_E,
        "w_U": w_U,
        "social_kappa": social_kappa,
        "noise_s": noise_s,
        "aI_pos": aI_pos,
        "aI_neg": aI_neg,
        "a_E": a_E,
        "a_U": a_U,
    }
    
    return params, result.fun


def fit_no_learning_model_mle(
    person_data,
    tau=3.0,
    max_iter=1000,
    decision_model="softmax",
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    """Fit model with learning rates fixed at 0."""
    def logit(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))
    
    if weight_mode == "relative":
        x0 = np.array([
            logit(1.0 / 3.0),  # pi_I
            logit(0.5),        # pi_E within non-instinct share
            logit(0.5),        # social_kappa / 2
            logit(0.2),        # noise_s / 0.5
        ])
    else:
        x0 = np.array([
            logit(0.5),        # w_I / 1.5
            logit(0.5),        # w_E / 1.5
            logit(0.5),        # w_U / 1.5
            logit(0.5),        # social_kappa / 2
            logit(0.2),        # noise_s / 0.5
        ])
    
    def objective(x):
        if weight_mode == "relative":
            w_I, w_E, w_U = _relative_weight_transform(x[0], x[1])
            social_kappa = 2.0 * expit(x[2])
            noise_s = 0.5 * expit(x[3])
        else:
            w_I = 1.5 * expit(x[0])
            w_E = 1.5 * expit(x[1])
            w_U = 1.5 * expit(x[2])
            social_kappa = 2.0 * expit(x[3])
            noise_s = 0.5 * expit(x[4])
        
        logp = simulate_forward_predict(
            w_I, w_E, w_U, noise_s, 0.0, 0.0, 0.0, 0.0, tau,
            person_data["inst0"], person_data["enj0"], person_data["uti0"],
            person_data["suggestion"], person_data["choice"],
            person_data["e_out"], person_data["u_out"],
            choice_mask=person_data["choice_mask"],
            learning_choice=person_data["learning_choice"],
            learning_role=person_data["learning_role"],
            setup_idx=person_data["setup_idx"],
            decision_model=decision_model,
            social_kappa=social_kappa,
            weight_mode=weight_mode,
            reports=person_data.get("reports"),
            use_measurement_likelihood=use_measurement_likelihood,
            report_sds=report_sds,
            measurement_weight=measurement_weight,
            choice_input_mode=choice_input_mode,
        )
        
        return -logp
    
    result = minimize(objective, x0, method='L-BFGS-B',
                     options={'maxiter': max_iter, 'disp': False})
    
    x_opt = result.x
    if weight_mode == "relative":
        w_I, w_E, w_U = _relative_weight_transform(x_opt[0], x_opt[1])
        social_kappa = 2.0 * expit(x_opt[2])
        noise_s = 0.5 * expit(x_opt[3])
    else:
        w_I = 1.5 * expit(x_opt[0])
        w_E = 1.5 * expit(x_opt[1])
        w_U = 1.5 * expit(x_opt[2])
        social_kappa = 2.0 * expit(x_opt[3])
        noise_s = 0.5 * expit(x_opt[4])
    params = {
        "w_I": w_I,
        "w_E": w_E,
        "w_U": w_U,
        "social_kappa": social_kappa,
        "noise_s": noise_s,
        "aI_pos": 0.0,
        "aI_neg": 0.0,
        "a_E": 0.0,
        "a_U": 0.0,
    }
    
    return params, result.fun


def _reward_report_sds(report_sds):
    if len(report_sds) >= 3:
        reward_sd = float(np.sqrt((report_sds[1] ** 2 + report_sds[2] ** 2) / 4.0))
        return (float(report_sds[0]), max(1e-3, reward_sd))
    return tuple(report_sds)


def fit_reward_model_mle(
    person_data,
    tau=3.0,
    max_iter=1000,
    decision_model="softmax",
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    """Fit collapsed instinct + reward model."""
    def logit(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))

    if weight_mode == "relative":
        x0 = np.array([
            logit(0.4),  # pi_I
            logit(0.5),  # social_kappa / 2
            logit(0.2),  # noise_s / 0.5
            logit(0.2),  # aI_pos
            logit(0.2),  # aI_neg
            logit(0.2),  # a_R
        ])
    else:
        x0 = np.array([
            logit(0.5),  # w_I / 1.5
            logit(0.5),  # w_R / 1.5
            logit(0.5),  # social_kappa / 2
            logit(0.2),  # noise_s / 0.5
            logit(0.2),  # aI_pos
            logit(0.2),  # aI_neg
            logit(0.2),  # a_R
        ])

    reward_sds = _reward_report_sds(report_sds)

    def unpack(x):
        if weight_mode == "relative":
            w_I = float(expit(x[0]))
            w_R = 1.0 - w_I
            social_kappa = 2.0 * expit(x[1])
            noise_s = 0.5 * expit(x[2])
            aI_pos = expit(x[3])
            aI_neg = expit(x[4])
            a_R = expit(x[5])
        else:
            w_I = 1.5 * expit(x[0])
            w_R = 1.5 * expit(x[1])
            social_kappa = 2.0 * expit(x[2])
            noise_s = 0.5 * expit(x[3])
            aI_pos = expit(x[4])
            aI_neg = expit(x[5])
            a_R = expit(x[6])
        return w_I, w_R, social_kappa, noise_s, aI_pos, aI_neg, a_R

    def objective(x):
        w_I, w_R, social_kappa, noise_s, aI_pos, aI_neg, a_R = unpack(x)
        logp = simulate_forward_predict_reward(
            w_I, w_R, social_kappa, noise_s, aI_pos, aI_neg, a_R, tau,
            person_data["inst0"], person_data["enj0"], person_data["uti0"],
            person_data["suggestion"], person_data["choice"],
            person_data["e_out"], person_data["u_out"],
            choice_mask=person_data["choice_mask"],
            learning_choice=person_data["learning_choice"],
            learning_role=person_data["learning_role"],
            setup_idx=person_data["setup_idx"],
            decision_model=decision_model,
            weight_mode=weight_mode,
            reports=person_data.get("reports"),
            use_measurement_likelihood=use_measurement_likelihood,
            report_sds=reward_sds,
            measurement_weight=measurement_weight,
            choice_input_mode=choice_input_mode,
        )
        return -logp

    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'disp': False})
    w_I, w_R, social_kappa, noise_s, aI_pos, aI_neg, a_R = unpack(result.x)
    params = {
        "w_I": w_I,
        "w_R": w_R,
        "social_kappa": social_kappa,
        "noise_s": noise_s,
        "aI_pos": aI_pos,
        "aI_neg": aI_neg,
        "a_R": a_R,
    }
    return params, result.fun


def fit_reward_no_learning_model_mle(
    person_data,
    tau=3.0,
    max_iter=1000,
    decision_model="softmax",
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    """Fit collapsed reward model with learning rates fixed at zero."""
    def logit(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))

    if weight_mode == "relative":
        x0 = np.array([logit(0.4), logit(0.5), logit(0.2)])
    else:
        x0 = np.array([logit(0.5), logit(0.5), logit(0.5), logit(0.2)])

    reward_sds = _reward_report_sds(report_sds)

    def unpack(x):
        if weight_mode == "relative":
            w_I = float(expit(x[0]))
            w_R = 1.0 - w_I
            social_kappa = 2.0 * expit(x[1])
            noise_s = 0.5 * expit(x[2])
        else:
            w_I = 1.5 * expit(x[0])
            w_R = 1.5 * expit(x[1])
            social_kappa = 2.0 * expit(x[2])
            noise_s = 0.5 * expit(x[3])
        return w_I, w_R, social_kappa, noise_s

    def objective(x):
        w_I, w_R, social_kappa, noise_s = unpack(x)
        logp = simulate_forward_predict_reward(
            w_I, w_R, social_kappa, noise_s, 0.0, 0.0, 0.0, tau,
            person_data["inst0"], person_data["enj0"], person_data["uti0"],
            person_data["suggestion"], person_data["choice"],
            person_data["e_out"], person_data["u_out"],
            choice_mask=person_data["choice_mask"],
            learning_choice=person_data["learning_choice"],
            learning_role=person_data["learning_role"],
            setup_idx=person_data["setup_idx"],
            decision_model=decision_model,
            weight_mode=weight_mode,
            reports=person_data.get("reports"),
            use_measurement_likelihood=use_measurement_likelihood,
            report_sds=reward_sds,
            measurement_weight=measurement_weight,
            choice_input_mode=choice_input_mode,
        )
        return -logp

    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'disp': False})
    w_I, w_R, social_kappa, noise_s = unpack(result.x)
    params = {
        "w_I": w_I,
        "w_R": w_R,
        "social_kappa": social_kappa,
        "noise_s": noise_s,
        "aI_pos": 0.0,
        "aI_neg": 0.0,
        "a_R": 0.0,
    }
    return params, result.fun


def predict_computational_model(
    params,
    person_data,
    tau=3.0,
    decision_model="softmax",
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    """Generate predictions using computational model."""
    if model_family == "reward":
        _, probs = simulate_forward_predict_reward(
            params["w_I"], params["w_R"],
            params.get("social_kappa", 1.0),
            params.get("noise_s", 0.0),
            params["aI_pos"], params["aI_neg"],
            params["a_R"],
            tau,
            person_data["inst0"], person_data["enj0"], person_data["uti0"],
            person_data["suggestion"], person_data["choice"],
            person_data["e_out"], person_data["u_out"],
            choice_mask=person_data["choice_mask"],
            learning_choice=person_data["learning_choice"],
            learning_role=person_data["learning_role"],
            setup_idx=person_data["setup_idx"],
            return_probs=True,
            decision_model=decision_model,
            weight_mode=weight_mode,
            reports=person_data.get("reports"),
            use_measurement_likelihood=use_measurement_likelihood,
            report_sds=_reward_report_sds(report_sds),
            measurement_weight=measurement_weight,
            choice_input_mode=choice_input_mode,
        )
        return probs

    _, probs = simulate_forward_predict(
        params["w_I"], params["w_E"], params["w_U"],
        params.get("noise_s", 0.0),
        params["aI_pos"], params["aI_neg"], 
        params["a_E"], params["a_U"],
        tau,
        person_data["inst0"], person_data["enj0"], person_data["uti0"],
        person_data["suggestion"], person_data["choice"],
        person_data["e_out"], person_data["u_out"],
        choice_mask=person_data["choice_mask"],
        learning_choice=person_data["learning_choice"],
        learning_role=person_data["learning_role"],
        setup_idx=person_data["setup_idx"],
        return_probs=True,
        decision_model=decision_model,
        social_kappa=params.get("social_kappa", 1.0),
        weight_mode=weight_mode,
        reports=person_data.get("reports"),
        use_measurement_likelihood=use_measurement_likelihood,
        report_sds=report_sds,
        measurement_weight=measurement_weight,
        choice_input_mode=choice_input_mode,
    )
    return probs


def evaluate_model(y_true, y_pred_proba):
    """Compute evaluation metrics."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    n_classes = len(np.unique(y_true))

    def binary_auc(y_true_vals, y_prob):
        y_true_vals = np.asarray(y_true_vals, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        pos = y_true_vals == 1
        n_pos = int(np.sum(pos))
        n_neg = int(len(y_true_vals) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return np.nan
        order = np.argsort(y_prob)
        sorted_scores = y_prob[order]
        ranks = np.empty(len(y_prob), dtype=float)
        i = 0
        while i < len(y_prob):
            j = i + 1
            while j < len(y_prob) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        rank_sum_pos = float(np.sum(ranks[pos]))
        return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def average_precision(y_true_vals, y_prob):
        y_true_vals = np.asarray(y_true_vals, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n_pos = int(np.sum(y_true_vals == 1))
        if n_pos == 0:
            return np.nan
        order = np.argsort(-y_prob)
        sorted_true = y_true_vals[order]
        tp = np.cumsum(sorted_true == 1)
        precision = tp / (np.arange(len(sorted_true)) + 1)
        return float(np.sum(precision[sorted_true == 1]) / n_pos)

    def balanced_acc(y_true_vals, y_pred_vals):
        vals = []
        for cls in [0, 1]:
            mask = y_true_vals == cls
            if np.any(mask):
                vals.append(float(np.mean(y_pred_vals[mask] == cls)))
        return float(np.mean(vals)) if vals else np.nan

    def matthews_corr(y_true_vals, y_pred_vals):
        tp = float(np.sum((y_true_vals == 1) & (y_pred_vals == 1)))
        tn = float(np.sum((y_true_vals == 0) & (y_pred_vals == 0)))
        fp = float(np.sum((y_true_vals == 0) & (y_pred_vals == 1)))
        fn = float(np.sum((y_true_vals == 1) & (y_pred_vals == 0)))
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / denom) if denom > 0 else np.nan

    def expected_calibration_error(y_true_vals, y_prob, n_bins=10):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
            else:
                mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if np.any(mask):
                avg_prob = float(np.mean(y_prob[mask]))
                avg_true = float(np.mean(y_true_vals[mask]))
                ece += abs(avg_prob - avg_true) * (np.sum(mask) / len(y_true_vals))
        return float(ece)
    
    if not SKLEARN_AVAILABLE:
        eps = 1e-10
        metrics = {
            "accuracy": float(np.mean(y_true == y_pred)),
            "brier_score": float(np.mean((y_true - y_pred_proba) ** 2)),
            "log_loss": float(-np.mean(y_true * np.log(y_pred_proba + eps) + (1 - y_true) * np.log(1 - y_pred_proba + eps))) if n_classes > 1 else np.nan,
            "auc": binary_auc(y_true, y_pred_proba) if n_classes > 1 else np.nan,
            "pr_auc": average_precision(y_true, y_pred_proba) if n_classes > 1 else np.nan,
            "balanced_accuracy": balanced_acc(y_true, y_pred) if n_classes > 1 else np.nan,
            "mcc": matthews_corr(y_true, y_pred) if n_classes > 1 else np.nan,
            "ece": expected_calibration_error(y_true, y_pred_proba) if n_classes > 1 else np.nan,
        }
        return metrics

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }
    
    if n_classes > 1:
        try:
            metrics["log_loss"] = log_loss(y_true, y_pred_proba, labels=[0, 1])
            metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
            metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)
            metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
            metrics["ece"] = expected_calibration_error(y_true, y_pred_proba)
        except:
            metrics["log_loss"] = np.nan
            metrics["auc"] = np.nan
            metrics["pr_auc"] = np.nan
            metrics["balanced_accuracy"] = np.nan
            metrics["mcc"] = np.nan
            metrics["ece"] = np.nan
    else:
        metrics["log_loss"] = np.nan
        metrics["auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["balanced_accuracy"] = np.nan
        metrics["mcc"] = np.nan
        metrics["ece"] = np.nan
    
    return metrics


def _empty_metrics():
    return {
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "log_loss": np.nan,
        "auc": np.nan,
        "pr_auc": np.nan,
        "brier_score": np.nan,
        "mcc": np.nan,
        "ece": np.nan,
    }


def _valid_choice_vector(person_data, min_trials=3):
    y = person_data["choice"][person_data["choice_mask"]]
    if len(y) < min_trials or len(np.unique(y)) < 2:
        return None
    return y


def _summarize_prediction_results(results_df, outdir):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "auc",
        "pr_auc",
        "brier",
        "mcc",
        "ece",
    ]
    models = ["null", "lr", "no_learn", "full"]
    rows = []
    for model in models:
        row = {"model": model}
        for metric in metrics:
            col = f"{model}_{metric}"
            if col in results_df.columns:
                vals = results_df[col].dropna()
                row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{metric}_se"] = float(vals.sem()) if len(vals) > 1 else np.nan
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outdir / "prediction_validation_between_person_summary.csv", index=False)
    return summary_df


def run_between_person_prediction(
    per_person,
    tau_mode,
    default_tau,
    outdir,
    train_frac=0.8,
    seed=123,
    decision_model="softmax",
    max_train_people=None,
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    rng = np.random.default_rng(seed)
    person_indices = rng.permutation(len(per_person))
    n_train = int(len(per_person) * train_frac)
    train_people = [per_person[i] for i in person_indices[:n_train]]
    test_people = [per_person[i] for i in person_indices[n_train:]]

    train_results = {"full": [], "no_learn": []}
    for person_data in train_people:
        if max_train_people is not None and len(train_results["full"]) >= max_train_people:
            break
        y_train = _valid_choice_vector(person_data, min_trials=5)
        if y_train is None:
            continue
        tau_val = select_tau(person_data, tau_mode, default_tau)
        if model_family == "reward":
            params_full, neg_ll_full = fit_reward_model_mle(
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )
        else:
            params_full, neg_ll_full = fit_full_model_mle(
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )
        train_results["full"].append({
            "person_id": person_data["person_id"],
            **params_full,
            "neg_log_lik": neg_ll_full,
        })

        if model_family == "reward":
            params_no_learn, neg_ll_no_learn = fit_reward_no_learning_model_mle(
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )
        else:
            params_no_learn, neg_ll_no_learn = fit_no_learning_model_mle(
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )
        train_results["no_learn"].append({
            "person_id": person_data["person_id"],
            **params_no_learn,
            "neg_log_lik": neg_ll_no_learn,
        })

    pop_params_full = {}
    if train_results["full"]:
        df_train_full = pd.DataFrame(train_results["full"])
        full_params = (
            ["w_I", "w_R", "social_kappa", "noise_s", "aI_pos", "aI_neg", "a_R"]
            if model_family == "reward"
            else ["w_I", "w_E", "w_U", "social_kappa", "noise_s", "aI_pos", "aI_neg", "a_E", "a_U"]
        )
        for param in full_params:
            pop_params_full[param] = float(df_train_full[param].mean())
        df_train_full.to_csv(outdir / "training_params_full_model.csv", index=False)

    pop_params_no_learn = {}
    if train_results["no_learn"]:
        df_train_no_learn = pd.DataFrame(train_results["no_learn"])
        no_learn_params = (
            ["w_I", "w_R", "social_kappa", "noise_s"]
            if model_family == "reward"
            else ["w_I", "w_E", "w_U", "social_kappa", "noise_s"]
        )
        for param in no_learn_params:
            pop_params_no_learn[param] = float(df_train_no_learn[param].mean())
        if model_family == "reward":
            for param in ["aI_pos", "aI_neg", "a_R"]:
                pop_params_no_learn[param] = 0.0
        else:
            for param in ["aI_pos", "aI_neg", "a_E", "a_U"]:
                pop_params_no_learn[param] = 0.0
        df_train_no_learn.to_csv(outdir / "training_params_no_learning.csv", index=False)

    train_choice_vectors = [p["choice"][p["choice_mask"]] for p in train_people if np.any(p["choice_mask"])]
    p_train = float(np.mean(np.concatenate(train_choice_vectors))) if train_choice_vectors else 0.5

    logistic_model = None
    if SKLEARN_AVAILABLE:
        X_train_all = []
        y_train_all = []
        for p in train_people:
            y_p = _valid_choice_vector(p, min_trials=5)
            if y_p is None:
                continue
            X_p = np.column_stack([
                p["suggestion"],
                p["inst0"][p["setup_idx"]],
                p["enj0"][p["setup_idx"]],
                p["uti0"][p["setup_idx"]],
                np.nan_to_num(p.get("reports", np.empty((len(p["choice"]), 2, 3))).reshape(len(p["choice"]), -1), nan=0.0),
            ])[p["choice_mask"]]
            X_train_all.append(X_p)
            y_train_all.append(y_p)
        if X_train_all:
            logistic_model = LogisticRegression(max_iter=1000, random_state=seed)
            logistic_model.fit(np.vstack(X_train_all), np.concatenate(y_train_all))

    test_results = []
    for person_data in test_people:
        y_true = _valid_choice_vector(person_data, min_trials=3)
        if y_true is None:
            continue
        choice_mask = person_data["choice_mask"]
        tau_val = select_tau(person_data, tau_mode, default_tau)

        metrics_null = evaluate_model(y_true, np.full(len(y_true), p_train))

        if logistic_model is not None:
            X_test = np.column_stack([
                person_data["suggestion"],
                person_data["inst0"][person_data["setup_idx"]],
                person_data["enj0"][person_data["setup_idx"]],
                person_data["uti0"][person_data["setup_idx"]],
                np.nan_to_num(person_data.get("reports", np.empty((len(person_data["choice"]), 2, 3))).reshape(len(person_data["choice"]), -1), nan=0.0),
            ])[choice_mask]
            metrics_lr = evaluate_model(y_true, logistic_model.predict_proba(X_test)[:, 1])
        else:
            metrics_lr = _empty_metrics()

        if pop_params_no_learn:
            probs_no_learn = predict_computational_model(
                pop_params_no_learn,
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                choice_input_mode=choice_input_mode,
                model_family=model_family,
            )
            metrics_no_learn = evaluate_model(y_true, probs_no_learn[choice_mask, 1])
        else:
            metrics_no_learn = _empty_metrics()

        if pop_params_full:
            probs_full = predict_computational_model(
                pop_params_full,
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                choice_input_mode=choice_input_mode,
                model_family=model_family,
            )
            metrics_full = evaluate_model(y_true, probs_full[choice_mask, 1])
        else:
            metrics_full = _empty_metrics()

        row = {
            "person_id": person_data["person_id"],
            "n_trials": int(person_data["T"]),
            "n_choice": int(np.sum(choice_mask)),
        }
        for prefix, metrics in [
            ("null", metrics_null),
            ("lr", metrics_lr),
            ("no_learn", metrics_no_learn),
            ("full", metrics_full),
        ]:
            row[f"{prefix}_accuracy"] = metrics["accuracy"]
            row[f"{prefix}_balanced_accuracy"] = metrics["balanced_accuracy"]
            row[f"{prefix}_log_loss"] = metrics["log_loss"]
            row[f"{prefix}_auc"] = metrics["auc"]
            row[f"{prefix}_pr_auc"] = metrics["pr_auc"]
            row[f"{prefix}_brier"] = metrics["brier_score"]
            row[f"{prefix}_mcc"] = metrics["mcc"]
            row[f"{prefix}_ece"] = metrics["ece"]
        test_results.append(row)

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(outdir / "prediction_validation_between_person.csv", index=False)
    summary_df = _summarize_prediction_results(results_df, outdir) if not results_df.empty else pd.DataFrame()
    if summary_df.empty:
        summary_df.to_csv(outdir / "prediction_validation_between_person_summary.csv", index=False)
    return results_df, summary_df


def run_parameter_recovery(
    per_person,
    tau_mode,
    default_tau,
    outdir,
    max_people=None,
    decision_model="softmax",
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    rows = []
    for person_data in per_person:
        if max_people is not None and len(rows) >= max_people:
            break
        observed_choices = person_data["choice"][person_data["choice_mask"]]
        if person_data["n_choice"] < 5 or len(np.unique(observed_choices)) < 2:
            continue

        tau_val = select_tau(person_data, tau_mode, default_tau)
        if model_family == "reward":
            params_fit, neg_ll = fit_reward_model_mle(
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )
        else:
            params_fit, neg_ll = fit_full_model_mle(
                person_data,
                tau=tau_val,
                decision_model=decision_model,
                weight_mode=weight_mode,
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )
        true_params = person_data.get("true_params", {})

        row = {
            "person_id": person_data["person_id"],
            "neg_log_lik": neg_ll,
        }
        for key, val in true_params.items():
            row[f"true_{key}"] = val
        for key, val in params_fit.items():
            row[f"fit_{key}"] = val
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = outdir / "parameter_recovery_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_rows = []
    if not results_df.empty:
        params = (
            ["w_I", "w_R", "social_kappa", "noise_s", "aI_pos", "aI_neg", "a_R"]
            if model_family == "reward"
            else ["w_I", "w_E", "w_U", "social_kappa", "noise_s", "aI_pos", "aI_neg", "a_E", "a_U"]
        )
        for param in params:
            if f"true_{param}" not in results_df or f"fit_{param}" not in results_df:
                continue
            true_vals = results_df[f"true_{param}"].values
            fit_vals = results_df[f"fit_{param}"].values
            if len(true_vals) < 2 or np.allclose(np.std(true_vals), 0):
                corr = np.nan
            else:
                corr = float(np.corrcoef(true_vals, fit_vals)[0, 1])
            bias = float(np.mean(fit_vals - true_vals))
            rmse = float(np.sqrt(np.mean((fit_vals - true_vals) ** 2)))
            summary_rows.append({
                "param": param,
                "n": len(true_vals),
                "corr": corr,
                "bias": bias,
                "rmse": rmse,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "parameter_recovery_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved parameter recovery results to: {results_path}")
    print(f"Saved parameter recovery summary to: {summary_path}")

    return results_df


def run_identifiability(
    per_person,
    tau_mode,
    default_tau,
    outdir,
    max_people=None,
    fit_df=None,
    decision_model="softmax",
    weight_mode="raw",
    use_measurement_likelihood=False,
    report_sds=(0.10, 0.10, 0.10),
    measurement_weight=1.0,
    choice_input_mode="latent",
    model_family="tripartite",
):
    bounds = {
        "w_I": (0.0, 1.0 if weight_mode == "relative" else 1.5),
        "w_E": (0.0, 1.0 if weight_mode == "relative" else 1.5),
        "w_U": (0.0, 1.0 if weight_mode == "relative" else 1.5),
        "w_R": (0.0, 1.0 if weight_mode == "relative" else 1.5),
        "social_kappa": (0.0, 2.0),
        "noise_s": (0.0, 0.5),
        "aI_pos": (0.0, 1.0),
        "aI_neg": (0.0, 1.0),
        "a_E": (0.0, 1.0),
        "a_U": (0.0, 1.0),
        "a_R": (0.0, 1.0),
    }
    param_order = (
        ["w_I", "w_R", "social_kappa", "noise_s", "aI_pos", "aI_neg", "a_R"]
        if model_family == "reward"
        else ["w_I", "w_E", "w_U", "social_kappa", "noise_s", "aI_pos", "aI_neg", "a_E", "a_U"]
    )

    results = []
    person_count = 0

    for person_data in per_person:
        if max_people is not None and person_count >= max_people:
            break
        observed_choices = person_data["choice"][person_data["choice_mask"]]
        if person_data["n_choice"] < 5 or len(np.unique(observed_choices)) < 2:
            continue

        tau_val = select_tau(person_data, tau_mode, default_tau)
        person_id = person_data["person_id"]

        if fit_df is not None and not fit_df.empty:
            row = fit_df[fit_df["person_id"] == person_id]
            if row.empty:
                if model_family == "reward":
                    params_fit, _ = fit_reward_model_mle(
                        person_data,
                        tau=tau_val,
                        decision_model=decision_model,
                        weight_mode=weight_mode,
                        use_measurement_likelihood=use_measurement_likelihood,
                        report_sds=report_sds,
                        measurement_weight=measurement_weight,
                        choice_input_mode=choice_input_mode,
                    )
                else:
                    params_fit, _ = fit_full_model_mle(
                        person_data,
                        tau=tau_val,
                        decision_model=decision_model,
                        weight_mode=weight_mode,
                        use_measurement_likelihood=use_measurement_likelihood,
                        report_sds=report_sds,
                        measurement_weight=measurement_weight,
                        choice_input_mode=choice_input_mode,
                    )
            else:
                params_fit = {}
                for param in param_order:
                    col = f"fit_{param}"
                    if col in row:
                        params_fit[param] = float(row[col].iloc[0])
                params_fit.setdefault("social_kappa", 1.0)
                params_fit.setdefault("noise_s", 0.0)
        else:
            if model_family == "reward":
                params_fit, _ = fit_reward_model_mle(
                    person_data,
                    tau=tau_val,
                    decision_model=decision_model,
                    weight_mode=weight_mode,
                    use_measurement_likelihood=use_measurement_likelihood,
                    report_sds=report_sds,
                    measurement_weight=measurement_weight,
                    choice_input_mode=choice_input_mode,
                )
            else:
                params_fit, _ = fit_full_model_mle(
                    person_data,
                    tau=tau_val,
                    decision_model=decision_model,
                    weight_mode=weight_mode,
                    use_measurement_likelihood=use_measurement_likelihood,
                    report_sds=report_sds,
                    measurement_weight=measurement_weight,
                    choice_input_mode=choice_input_mode,
                )

        if model_family == "reward":
            base_logp = simulate_forward_predict_reward(
                params_fit["w_I"], params_fit["w_R"],
                params_fit.get("social_kappa", 1.0),
                params_fit.get("noise_s", 0.0),
                params_fit["aI_pos"], params_fit["aI_neg"], params_fit["a_R"],
                tau_val,
                person_data["inst0"], person_data["enj0"], person_data["uti0"],
                person_data["suggestion"], person_data["choice"],
                person_data["e_out"], person_data["u_out"],
                choice_mask=person_data["choice_mask"],
                learning_choice=person_data["learning_choice"],
                learning_role=person_data["learning_role"],
                setup_idx=person_data["setup_idx"],
                decision_model=decision_model,
                weight_mode=weight_mode,
                reports=person_data.get("reports"),
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=_reward_report_sds(report_sds),
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )
        else:
            base_logp = simulate_forward_predict(
                params_fit["w_I"], params_fit["w_E"], params_fit["w_U"],
                params_fit["noise_s"],
                params_fit["aI_pos"], params_fit["aI_neg"],
                params_fit["a_E"], params_fit["a_U"],
                tau_val,
                person_data["inst0"], person_data["enj0"], person_data["uti0"],
                person_data["suggestion"], person_data["choice"],
                person_data["e_out"], person_data["u_out"],
                choice_mask=person_data["choice_mask"],
                learning_choice=person_data["learning_choice"],
                learning_role=person_data["learning_role"],
                setup_idx=person_data["setup_idx"],
                decision_model=decision_model,
                social_kappa=params_fit.get("social_kappa", 1.0),
                weight_mode=weight_mode,
                reports=person_data.get("reports"),
                use_measurement_likelihood=use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=measurement_weight,
                choice_input_mode=choice_input_mode,
            )

        for param in param_order:
            low, high = bounds[param]
            center = params_fit[param]
            span = 0.3 * (high - low)
            grid_low = max(low, center - span)
            grid_high = min(high, center + span)
            grid = np.linspace(grid_low, grid_high, 7)

            ll_values = []
            for val in grid:
                test_params = params_fit.copy()
                test_params[param] = float(val)
                if model_family == "reward":
                    ll = simulate_forward_predict_reward(
                        test_params["w_I"], test_params["w_R"],
                        test_params.get("social_kappa", 1.0),
                        test_params.get("noise_s", 0.0),
                        test_params["aI_pos"], test_params["aI_neg"], test_params["a_R"],
                        tau_val,
                        person_data["inst0"], person_data["enj0"], person_data["uti0"],
                        person_data["suggestion"], person_data["choice"],
                        person_data["e_out"], person_data["u_out"],
                        choice_mask=person_data["choice_mask"],
                        learning_choice=person_data["learning_choice"],
                        learning_role=person_data["learning_role"],
                        setup_idx=person_data["setup_idx"],
                        decision_model=decision_model,
                        weight_mode=weight_mode,
                        reports=person_data.get("reports"),
                        use_measurement_likelihood=use_measurement_likelihood,
                        report_sds=_reward_report_sds(report_sds),
                        measurement_weight=measurement_weight,
                        choice_input_mode=choice_input_mode,
                    )
                else:
                    ll = simulate_forward_predict(
                        test_params["w_I"], test_params["w_E"], test_params["w_U"],
                        test_params["noise_s"],
                        test_params["aI_pos"], test_params["aI_neg"],
                        test_params["a_E"], test_params["a_U"],
                        tau_val,
                        person_data["inst0"], person_data["enj0"], person_data["uti0"],
                        person_data["suggestion"], person_data["choice"],
                        person_data["e_out"], person_data["u_out"],
                        choice_mask=person_data["choice_mask"],
                        learning_choice=person_data["learning_choice"],
                        learning_role=person_data["learning_role"],
                        setup_idx=person_data["setup_idx"],
                        decision_model=decision_model,
                        social_kappa=test_params.get("social_kappa", 1.0),
                        weight_mode=weight_mode,
                        reports=person_data.get("reports"),
                        use_measurement_likelihood=use_measurement_likelihood,
                        report_sds=report_sds,
                        measurement_weight=measurement_weight,
                        choice_input_mode=choice_input_mode,
                    )
                ll_values.append(ll)

            ll_values = np.array(ll_values)
            best_ll = float(np.max(ll_values))
            ll_drop_max = float(best_ll - np.min(ll_values))
            flat_fraction = float(np.mean(ll_values >= (best_ll - 2.0)))

            results.append({
                "person_id": person_id,
                "param": param,
                "grid_low": float(grid_low),
                "grid_high": float(grid_high),
                "best_ll": best_ll,
                "ll_drop_max": ll_drop_max,
                "flat_fraction": flat_fraction,
                "base_ll": float(base_logp),
            })

        person_count += 1

    results_df = pd.DataFrame(results)
    results_path = outdir / "parameter_identifiability_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_rows = []
    if not results_df.empty:
        for param in results_df["param"].unique():
            subset = results_df[results_df["param"] == param]
            summary_rows.append({
                "param": param,
                "n": len(subset),
                "mean_ll_drop_max": float(subset["ll_drop_max"].mean()),
                "mean_flat_fraction": float(subset["flat_fraction"].mean()),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "parameter_identifiability_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved identifiability results to: {results_path}")
    print(f"Saved identifiability summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Predictive validity testing with between-person split")
    parser.add_argument("--events", default="outputs/ema_events.csv")
    parser.add_argument("--people", default="outputs/ema_people.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--train_frac", type=float, default=0.8,
                       help="Fraction of people for training (rest for testing)")
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--tau_mode", type=str, default="fixed", choices=["fixed", "person"])
    parser.add_argument("--decision_model", type=str, default="softmax", choices=["softmax", "ddm"])
    parser.add_argument("--model_family", type=str, default="tripartite", choices=["tripartite", "reward"])
    parser.add_argument("--weight_mode", type=str, default="raw", choices=["raw", "relative"])
    parser.add_argument("--use_measurement_likelihood", action="store_true")
    parser.add_argument("--choice_input_mode", type=str, default="latent", choices=["latent", "reports"])
    parser.add_argument("--measurement_weight", type=float, default=1.0)
    parser.add_argument("--report_sd_urge", type=float, default=0.10)
    parser.add_argument("--report_sd_enjoyment", type=float, default=0.10)
    parser.add_argument("--report_sd_utility", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip_validation", action="store_true", help="Skip train/test validation and run only recovery/identifiability")
    parser.add_argument("--recovery", action="store_true", help="Run parameter recovery analysis")
    parser.add_argument("--recovery_max_people", type=int, default=None)
    parser.add_argument("--identifiability", action="store_true", help="Run identifiability analysis")
    parser.add_argument("--identifiability_max_people", type=int, default=None)
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
    report_sds = (args.report_sd_urge, args.report_sd_enjoyment, args.report_sd_utility)
    
    if args.skip_validation:
        print("\nSKIPPING predictive validity (requested)")
        recovery_df = None
        if args.recovery:
            print("\n" + "="*70)
            print("PARAMETER RECOVERY")
            print("="*70)
            recovery_df = run_parameter_recovery(
                per_person,
                tau_mode=args.tau_mode,
                default_tau=args.tau,
                outdir=outdir,
                max_people=args.recovery_max_people,
                decision_model=args.decision_model,
                weight_mode=args.weight_mode,
                use_measurement_likelihood=args.use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=args.measurement_weight,
                choice_input_mode=args.choice_input_mode,
                model_family=args.model_family,
            )

        if args.identifiability:
            print("\n" + "="*70)
            print("PARAMETER IDENTIFIABILITY")
            print("="*70)
            run_identifiability(
                per_person,
                tau_mode=args.tau_mode,
                default_tau=args.tau,
                outdir=outdir,
                max_people=args.identifiability_max_people,
                fit_df=recovery_df,
                decision_model=args.decision_model,
                weight_mode=args.weight_mode,
                use_measurement_likelihood=args.use_measurement_likelihood,
                report_sds=report_sds,
                measurement_weight=args.measurement_weight,
                choice_input_mode=args.choice_input_mode,
                model_family=args.model_family,
            )

        print()
        print("="*70)
        print("DONE!")
        print("="*70)
        return

    run_between_person_prediction(
        per_person,
        tau_mode=args.tau_mode,
        default_tau=args.tau,
        outdir=outdir,
        train_frac=args.train_frac,
        seed=args.seed,
        decision_model=args.decision_model,
        weight_mode=args.weight_mode,
        use_measurement_likelihood=args.use_measurement_likelihood,
        report_sds=report_sds,
        measurement_weight=args.measurement_weight,
        choice_input_mode=args.choice_input_mode,
        model_family=args.model_family,
    )

    recovery_df = None
    if args.recovery:
        print("\n" + "="*70)
        print("PARAMETER RECOVERY")
        print("="*70)
        recovery_df = run_parameter_recovery(
            per_person,
            tau_mode=args.tau_mode,
            default_tau=args.tau,
            outdir=outdir,
            max_people=args.recovery_max_people,
            decision_model=args.decision_model,
            weight_mode=args.weight_mode,
            use_measurement_likelihood=args.use_measurement_likelihood,
            report_sds=report_sds,
            measurement_weight=args.measurement_weight,
            choice_input_mode=args.choice_input_mode,
            model_family=args.model_family,
        )

    if args.identifiability:
        print("\n" + "="*70)
        print("PARAMETER IDENTIFIABILITY")
        print("="*70)
        run_identifiability(
            per_person,
            tau_mode=args.tau_mode,
            default_tau=args.tau,
            outdir=outdir,
            max_people=args.identifiability_max_people,
            fit_df=recovery_df,
            decision_model=args.decision_model,
            weight_mode=args.weight_mode,
            use_measurement_likelihood=args.use_measurement_likelihood,
            report_sds=report_sds,
            measurement_weight=args.measurement_weight,
            choice_input_mode=args.choice_input_mode,
            model_family=args.model_family,
        )

    print()
    print("="*70)
    print("DONE!")
    print("="*70)
    return

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
        tau_val = select_tau(person_data, args.tau_mode, args.tau)
        print(f"Training on Person {pid} ({i+1}/{n_train}, T={person_data['T']})...")
        
        # Skip if too few trials or single class
        train_choice = person_data["choice"][person_data["choice_mask"]]
        if person_data["n_choice"] < 5 or len(np.unique(train_choice)) < 2:
            print(f"  ⚠️  Skipping (insufficient data)")
            continue
        
        # Fit full model
        print("  Fitting full model...")
        params_full, neg_ll_full = fit_full_model_mle(
            person_data,
            tau=tau_val,
            decision_model=args.decision_model,
        )
        train_results['full'].append({
            'person_id': pid,
            **params_full,
            'neg_log_lik': neg_ll_full
        })
        
        # Fit no-learning model
        print("  Fitting no-learning model...")
        params_no_learn, neg_ll_no_learn = fit_no_learning_model_mle(
            person_data,
            tau=tau_val,
            decision_model=args.decision_model,
        )
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
        for param in ['w_I', 'w_E', 'w_U', 'noise_s', 'aI_pos', 'aI_neg', 'a_E', 'a_U']:
            mean_val = df_train_full[param].mean()
            std_val = df_train_full[param].std()
            pop_params_full[param] = mean_val
            print(f"  {param:10s}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    if len(train_results['no_learn']) > 0:
        df_train_no_learn = pd.DataFrame(train_results['no_learn'])
        print("\nNo-Learning Model:")
        for param in ['w_I', 'w_E', 'w_U', 'noise_s']:
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
        tau_val = select_tau(person_data, args.tau_mode, args.tau)
        print(f"Testing on Person {pid} ({i+1}/{n_test}, T={T})...")
        
        # Skip if insufficient data
        choice_mask = person_data["choice_mask"]
        y_true = person_data["choice"][choice_mask]
        if len(y_true) < 3 or len(np.unique(y_true)) < 2:
            print(f"  ⚠️  Skipping (insufficient data)")
            continue
        
        # ==========================
        # 1. NULL MODEL (marginal from training set)
        # ==========================
        all_train_choices = np.concatenate([p["choice"][p["choice_mask"]] for p in train_people])
        p_train = np.mean(all_train_choices)
        y_pred_null = np.full(len(y_true), p_train)
        metrics_null = evaluate_model(y_true, y_pred_null)
        print(f"  Null model:       Acc={metrics_null['accuracy']:.3f}, LogLoss={metrics_null['log_loss']:.3f}")
        
        # ==========================
        # 2. LOGISTIC REGRESSION (trained on all training people)
        # ==========================
        # Aggregate all training data
        X_train_all = []
        y_train_all = []
        for p in train_people:
            p_choice = p["choice"][p["choice_mask"]]
            if len(p_choice) >= 5 and len(np.unique(p_choice)) >= 2:
                X_p = np.column_stack([
                    p["suggestion"],
                    p["inst0"][p["setup_idx"]],
                    p["enj0"][p["setup_idx"]],
                    p["uti0"][p["setup_idx"]],
                ])[p["choice_mask"]]
                X_train_all.append(X_p)
                y_train_all.append(p_choice)
        
        if len(X_train_all) > 0 and SKLEARN_AVAILABLE:
            X_train_all = np.vstack(X_train_all)
            y_train_all = np.concatenate(y_train_all)
            
            # Fit logistic regression
            lr_model = LogisticRegression(max_iter=1000, random_state=args.seed)
            lr_model.fit(X_train_all, y_train_all)
            
            # Predict on test person
            X_test = np.column_stack([
                person_data["suggestion"],
                person_data["inst0"][person_data["setup_idx"]],
                person_data["enj0"][person_data["setup_idx"]],
                person_data["uti0"][person_data["setup_idx"]],
            ])[choice_mask]
            y_pred_lr = lr_model.predict_proba(X_test)[:, 1]
            metrics_lr = evaluate_model(y_true, y_pred_lr)
            print(f"  Logistic Reg:     Acc={metrics_lr['accuracy']:.3f}, LogLoss={metrics_lr['log_loss']:.3f}")
        else:
            metrics_lr = {k: np.nan for k in [
                'accuracy', 'log_loss', 'auc', 'brier_score',
                'pr_auc', 'balanced_accuracy', 'mcc', 'ece',
            ]}
            print(f"  Logistic Reg:     SKIPPED (no valid training data)")
        
        # ==========================
        # 3. NO-LEARNING MODEL (population params from training)
        # ==========================
        if len(pop_params_no_learn) > 0:
            probs_no_learn = predict_computational_model(
                pop_params_no_learn,
                person_data,
                tau=tau_val,
                decision_model=args.decision_model,
            )
            y_pred_no_learn = probs_no_learn[choice_mask, 1]
            metrics_no_learn = evaluate_model(y_true, y_pred_no_learn)
            print(f"  No-learning (pop): Acc={metrics_no_learn['accuracy']:.3f}, LogLoss={metrics_no_learn['log_loss']:.3f}")
        else:
            metrics_no_learn = {k: np.nan for k in [
                'accuracy', 'log_loss', 'auc', 'brier_score',
                'pr_auc', 'balanced_accuracy', 'mcc', 'ece',
            ]}
            print(f"  No-learning (pop): SKIPPED (no training params)")
        
        # ==========================
        # 4. FULL MODEL (population params from training)
        # ==========================
        if len(pop_params_full) > 0:
            probs_full = predict_computational_model(
                pop_params_full,
                person_data,
                tau=tau_val,
                decision_model=args.decision_model,
            )
            y_pred_full = probs_full[choice_mask, 1]
            metrics_full = evaluate_model(y_true, y_pred_full)
            print(f"  Full model (pop):  Acc={metrics_full['accuracy']:.3f}, LogLoss={metrics_full['log_loss']:.3f}")
        else:
            metrics_full = {k: np.nan for k in [
                'accuracy', 'log_loss', 'auc', 'brier_score',
                'pr_auc', 'balanced_accuracy', 'mcc', 'ece',
            ]}
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
            "null_pr_auc": metrics_null["pr_auc"],
            "null_balanced_accuracy": metrics_null["balanced_accuracy"],
            "null_mcc": metrics_null["mcc"],
            "null_ece": metrics_null["ece"],
            # Logistic regression
            "lr_accuracy": metrics_lr["accuracy"],
            "lr_log_loss": metrics_lr["log_loss"],
            "lr_auc": metrics_lr["auc"],
            "lr_brier": metrics_lr["brier_score"],
            "lr_pr_auc": metrics_lr["pr_auc"],
            "lr_balanced_accuracy": metrics_lr["balanced_accuracy"],
            "lr_mcc": metrics_lr["mcc"],
            "lr_ece": metrics_lr["ece"],
            # No-learning model
            "no_learn_accuracy": metrics_no_learn["accuracy"],
            "no_learn_log_loss": metrics_no_learn["log_loss"],
            "no_learn_auc": metrics_no_learn["auc"],
            "no_learn_brier": metrics_no_learn["brier_score"],
            "no_learn_pr_auc": metrics_no_learn["pr_auc"],
            "no_learn_balanced_accuracy": metrics_no_learn["balanced_accuracy"],
            "no_learn_mcc": metrics_no_learn["mcc"],
            "no_learn_ece": metrics_no_learn["ece"],
            # Full model
            "full_accuracy": metrics_full["accuracy"],
            "full_log_loss": metrics_full["log_loss"],
            "full_auc": metrics_full["auc"],
            "full_brier": metrics_full["brier_score"],
            "full_pr_auc": metrics_full["pr_auc"],
            "full_balanced_accuracy": metrics_full["balanced_accuracy"],
            "full_mcc": metrics_full["mcc"],
            "full_ece": metrics_full["ece"],
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
        
        metrics = [
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "auc",
            "pr_auc",
            "brier",
            "mcc",
            "ece",
        ]
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
                            print(f"  Accuracy:          {mean_val:.3f} ± {se_val:.3f}")
                        elif metric == "balanced_accuracy":
                            print(f"  Balanced Acc:      {mean_val:.3f} ± {se_val:.3f}")
                        elif metric == "log_loss":
                            print(f"  Log Loss:          {mean_val:.3f} ± {se_val:.3f} (lower is better)")
                        elif metric == "auc":
                            print(f"  AUC:               {mean_val:.3f} ± {se_val:.3f}")
                        elif metric == "pr_auc":
                            print(f"  PR AUC:            {mean_val:.3f} ± {se_val:.3f}")
                        elif metric == "brier":
                            print(f"  Brier Score:       {mean_val:.3f} ± {se_val:.3f} (lower is better)")
                        elif metric == "mcc":
                            print(f"  MCC:               {mean_val:.3f} ± {se_val:.3f}")
                        elif metric == "ece":
                            print(f"  ECE:               {mean_val:.3f} ± {se_val:.3f} (lower is better)")
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
    
    recovery_df = None
    if args.recovery:
        print("\n" + "="*70)
        print("PARAMETER RECOVERY")
        print("="*70)
        recovery_df = run_parameter_recovery(
            per_person,
            tau_mode=args.tau_mode,
            default_tau=args.tau,
            outdir=outdir,
            max_people=args.recovery_max_people,
            decision_model=args.decision_model,
        )

    if args.identifiability:
        print("\n" + "="*70)
        print("PARAMETER IDENTIFIABILITY")
        print("="*70)
        run_identifiability(
            per_person,
            tau_mode=args.tau_mode,
            default_tau=args.tau,
            outdir=outdir,
            max_people=args.identifiability_max_people,
            fit_df=recovery_df,
            decision_model=args.decision_model,
        )

    print()
    print("="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
