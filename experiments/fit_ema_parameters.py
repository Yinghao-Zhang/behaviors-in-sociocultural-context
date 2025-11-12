from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit as sigmoid  # numerically stable sigmoid

# --------- config ---------

BEH_KEYS = ["avoid_conflict", "approach_conflict_care"]

@dataclass
class FitConfig:
    K_mc: int = 20  # Monte-Carlo samples for CV noise integration
    n_restarts: int = 8
    seed: int = 123

# --------- utils ---------

def clip11(x: float) -> float:
    return max(-1.0, min(1.0, x))

def softmax(z: np.ndarray, tau: float) -> np.ndarray:
    z = z * tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

# --------- model roll-forward and log-likelihood ---------

def neg_loglik_person(
    events: pd.DataFrame,
    baseline: Dict[str, float],
    x_uncon: np.ndarray,
    cfg: FitConfig,
) -> float:
    # Unpack unconstrained params and transform to valid ranges
    # x = [w_I, w_E, w_U, tau, noise_s, alpha_I_pos, alpha_I_neg, alpha_E, alpha_U]
    # domains: w_* in [0,1.5], tau in [0.5, 10], noise_s in [0, 0.5], alphas in [0, 1]
    s = 0
    def take(n):  # helper
        nonlocal s
        v = x_uncon[s:s+n]
        s += n
        return v

    w_raw = take(3)   # -> [0,1.5]
    w = 1.5 * sigmoid(w_raw)
    tau = 0.5 + 9.5 * sigmoid(take(1))[0]
    noise_s = 0.5 * sigmoid(take(1))[0]
    alphas = sigmoid(take(4))  # [0,1]

    w_I, w_E, w_U = w
    alpha_I_pos, alpha_I_neg, alpha_E, alpha_U = alphas

    # Latent states initialized from baseline (collected at baseline in your EMA)
    inst = {k: float(baseline[f"instinct_{k}_0"]) for k in BEH_KEYS}
    enj =  {k: float(baseline[f"enjoyment_{k}_0"]) for k in BEH_KEYS}
    uti =  {k: float(baseline[f"utility_{k}_0"]) for k in BEH_KEYS}

    rng = np.random.default_rng(0)  # deterministic for objective smoothness
    nll = 0.0

    # iterate trials
    for _, row in events.sort_values("t").iterrows():
        # suggestion terms (already scaled by receptivity in the simulator)
        sug = {k: float(row.get(f"suggest_term_{k}", 0.0)) for k in BEH_KEYS}

        # build deterministic part of CV for each behavior
        cvs_det = []
        for k in BEH_KEYS:
            H = w_I * inst[k]
            E_eval = w_E * enj[k] + w_U * uti[k]
            cvs_det.append(H + E_eval + sug[k])
        cvs_det = np.array(cvs_det, dtype=float)

        # Monte-Carlo integrate Gaussian noise added to CVs before softmax
        # noise ~ N(0, noise_s); average P(choice) over K samples
        if noise_s > 0:
            probs = []
            for _ in range(cfg.K_mc):
                noise = rng.normal(0.0, noise_s, size=len(BEH_KEYS))
                probs.append(softmax(cvs_det + noise, tau))
            probs = np.stack(probs, axis=0).mean(axis=0)
        else:
            probs = softmax(cvs_det, tau)

        chosen_k = row["choice_behavior"]
        chosen_idx = BEH_KEYS.index(chosen_k)
        p = float(np.clip(probs[chosen_idx], 1e-12, 1.0))
        nll -= np.log(p)

        # Learning updates using observed outcomes
        e_out = float(row["enjoyment_out"])
        u_out = float(row["utility_out"])

        # Instinct: chosen strengthens, unchosen weakens
        for k in BEH_KEYS:
            if k == chosen_k:
                inst[k] = clip11(inst[k] + alpha_I_pos * (1.0 - inst[k]))
            else:
                inst[k] = clip11(inst[k] + alpha_I_neg * (-1.0 - inst[k]))
        # Enjoyment and Utility for chosen only
        enj[chosen_k] = clip11(enj[chosen_k] + alpha_E * (e_out - enj[chosen_k]))
        uti[chosen_k] = clip11(uti[chosen_k] + alpha_U * (u_out - uti[chosen_k]))

    return nll

def fit_person(events: pd.DataFrame, baseline: Dict[str, float], cfg: FitConfig):
    rng = np.random.default_rng(cfg.seed)
    best = None

    def run_start():
        # random init in unconstrained space ~ N(0,1)
        x0 = rng.normal(0.0, 0.75, size=3 + 1 + 1 + 4)
        res = minimize(
            lambda x: neg_loglik_person(events, baseline, x, cfg),
            x0,
            method="L-BFGS-B",
        )
        return res

    for _ in range(cfg.n_restarts):
        res = run_start()
        if (best is None) or (res.fun < best.fun):
            best = res

    # unpack best to readable dict
    x = best.x
    s = 0
    def take(n):
        nonlocal s
        v = x[s:s+n]; s += n; return v
    w = 1.5 * sigmoid(take(3))
    tau = 0.5 + 9.5 * sigmoid(take(1))[0]
    noise_s = 0.5 * sigmoid(take(1))[0]
    alphas = sigmoid(take(4))
    w_I, w_E, w_U = map(float, w)
    alpha_I_pos, alpha_I_neg, alpha_E, alpha_U = map(float, alphas)

    out = dict(
        nll=float(best.fun),
        w_I=w_I, w_E=w_E, w_U=w_U, tau=tau, noise_s=noise_s,
        alpha_I_pos=alpha_I_pos, alpha_I_neg=alpha_I_neg,
        alpha_E=alpha_E, alpha_U=alpha_U,
        success=bool(best.success), message=str(best.message),
        n_iter=int(best.nit),
    )
    return out

# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(description="Parameter recovery for EMA two-behavior simulator")
    ap.add_argument("--events", type=str, default="outputs/ema_events.csv")
    ap.add_argument("--people", type=str, default="outputs/ema_people.csv")
    ap.add_argument("--out", type=str, default="outputs/recovery_results.csv")
    ap.add_argument("--Kmc", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = pd.read_csv(args.events)
    ppl = pd.read_csv(args.people)

    cfg = FitConfig(K_mc=args.Kmc, n_restarts=args.restarts, seed=args.seed)

    results = []
    for pid, ev in df.groupby("person_id"):
        base_row = ppl.loc[ppl["person_id"] == pid].iloc[0].to_dict()
        fit = fit_person(ev, base_row, cfg)
        fit["person_id"] = int(pid)

        # attach ground truth if present for comparison
        for k in ["w_I","w_E","w_U","tau","noise_s","alpha_I_pos","alpha_I_neg","alpha_E","alpha_U"]:
            col = k  # same names in ema_people.csv
            if col in base_row:
                fit[f"true_{k}"] = float(base_row[col])
        results.append(fit)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    # Quick summary
    for k in ["w_I","w_E","w_U","tau","noise_s","alpha_I_pos","alpha_I_neg","alpha_E","alpha_U"]:
        if f"true_{k}" in out_df.columns:
            err = (out_df[k] - out_df[f"true_{k}"]).abs().mean()
            print(f"MAE {k}: {err:.3f}")

if __name__ == "__main__":
    main()