"""
Hierarchical Bayesian parameter recovery for behavioral learning model.
Clean implementation using PyMC with proper scan operations.
"""
import argparse
import pathlib
import numpy as np
import pandas as pd

# Fix for scipy.signal.gaussian removal in scipy >= 1.10
import scipy.signal
if not hasattr(scipy.signal, 'gaussian'):
    from scipy.signal.windows import gaussian
    scipy.signal.gaussian = gaussian

import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az

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
        
        per_person.append({
            "person_id": int(p),
            "T": T,
            "choice": choice,
            "suggestion": suggestion,
            "e_out": e_out,
            "u_out": u_out,
            "inst0": inst0,
            "enj0": enj0,
            "uti0": uti0
        })
    
    return per_person, ppl


def simulate_forward_scan(w_I, w_E, w_U, aI_pos, aI_neg, a_E, a_U, tau,
                          inst0, enj0, uti0, suggestion, choice, e_out, u_out):
    """
    Forward simulation using pt.scan for a single person.
    Returns total log probability.
    """
    
    def step_fn(t, inst_prev, enj_prev, uti_prev, logp_accum):
        """Single time step update."""
        # Current suggestion
        sug_t = suggestion[t]
        
        # Compute choice values
        CV = w_I * inst_prev + w_E * enj_prev + w_U * uti_prev + sug_t
        probs = pm.math.softmax(tau * CV)
        
        # Choice log likelihood
        ct = choice[t]
        logp_choice = pt.log(probs[ct] + 1e-10)  # Add small epsilon for numerical stability
        
        # Outcome log likelihoods (before update)
        # Mean outcome is the pre-update value of chosen behavior
        e_mean = enj_prev[ct]
        u_mean = uti_prev[ct]
        
        logp_e = -0.5 * ((e_out[t] - e_mean) ** 2)  # Simplified, will add sigma later
        logp_u = -0.5 * ((u_out[t] - u_mean) ** 2)
        
        # Update belief states
        # Create one-hot mask for chosen behavior
        mask_0 = pt.cast(pt.eq(ct, 0), "float64")
        mask_1 = pt.cast(pt.eq(ct, 1), "float64")
        mask = pt.stack([mask_0, mask_1])
        
        # Instinct update (different rates for chosen vs unchosen)
        inst_new = inst_prev + mask * aI_pos * (1.0 - inst_prev) + (1.0 - mask) * aI_neg * (-1.0 - inst_prev)
        
        # Enjoyment and utility updates (prediction error on chosen only)
        enj_new = enj_prev + mask * a_E * (e_out[t] - enj_prev)
        uti_new = uti_prev + mask * a_U * (u_out[t] - uti_prev)
        
        # Clip to [-1, 1] for stability
        inst_new = pt.clip(inst_new, -1.0, 1.0)
        enj_new = pt.clip(enj_new, -1.0, 1.0)
        uti_new = pt.clip(uti_new, -1.0, 1.0)
        
        # Accumulate log probability
        logp_new = logp_accum + logp_choice + logp_e + logp_u
        
        return inst_new, enj_new, uti_new, logp_new
    
    # Run scan over time steps
    T = choice.shape[0]
    sequences = pt.arange(T)
    
    (inst_seq, enj_seq, uti_seq, logp_seq), _ = scan(
        fn=step_fn,
        sequences=[sequences],
        outputs_info=[inst0, enj0, uti0, pt.constant(0.0, dtype='float64')],
        n_steps=T
    )
    
    # Return final accumulated log probability
    return logp_seq[-1]


def build_hierarchical_model(per_person, tau_fixed=3.0):
    """
    Build hierarchical Bayesian model for parameter recovery.
    """
    P = len(per_person)
    
    with pm.Model() as model:
        # ===== HYPERPRIORS =====
        # Weights (will be transformed to [0, 1.5])
        mu_wI = pm.Normal("mu_wI", 0.0, 1.0)
        sigma_wI = pm.HalfNormal("sigma_wI", 1.0)
        
        mu_wE = pm.Normal("mu_wE", 0.0, 1.0)
        sigma_wE = pm.HalfNormal("sigma_wE", 1.0)
        
        mu_wU = pm.Normal("mu_wU", 0.0, 1.0)
        sigma_wU = pm.HalfNormal("sigma_wU", 1.0)
        
        # Learning rates (will be transformed to [0, 1])
        mu_aIpos = pm.Normal("mu_aIpos", 0.0, 1.0)
        sigma_aIpos = pm.HalfNormal("sigma_aIpos", 1.0)
        
        mu_aIneg = pm.Normal("mu_aIneg", 0.0, 1.0)
        sigma_aIneg = pm.HalfNormal("sigma_aIneg", 1.0)
        
        mu_aE = pm.Normal("mu_aE", 0.0, 1.0)
        sigma_aE = pm.HalfNormal("sigma_aE", 1.0)
        
        mu_aU = pm.Normal("mu_aU", 0.0, 1.0)
        sigma_aU = pm.HalfNormal("sigma_aU", 1.0)
        
        # ===== PERSON-LEVEL PARAMETERS (raw) =====
        wI_raw = pm.Normal("wI_raw", mu=mu_wI, sigma=sigma_wI, shape=P)
        wE_raw = pm.Normal("wE_raw", mu=mu_wE, sigma=sigma_wE, shape=P)
        wU_raw = pm.Normal("wU_raw", mu=mu_wU, sigma=sigma_wU, shape=P)
        
        aIpos_raw = pm.Normal("aIpos_raw", mu=mu_aIpos, sigma=sigma_aIpos, shape=P)
        aIneg_raw = pm.Normal("aIneg_raw", mu=mu_aIneg, sigma=sigma_aIneg, shape=P)
        aE_raw = pm.Normal("aE_raw", mu=mu_aE, sigma=sigma_aE, shape=P)
        aU_raw = pm.Normal("aU_raw", mu=mu_aU, sigma=sigma_aU, shape=P)
        
        # ===== TRANSFORMATIONS =====
        # Weights: sigmoid to [0, 1], then scale to [0, 1.5]
        w_I = pm.Deterministic("w_I", 1.5 * pm.math.sigmoid(wI_raw))
        w_E = pm.Deterministic("w_E", 1.5 * pm.math.sigmoid(wE_raw))
        w_U = pm.Deterministic("w_U", 1.5 * pm.math.sigmoid(wU_raw))
        
        # Learning rates: sigmoid to [0, 1]
        aI_pos = pm.Deterministic("aI_pos", pm.math.sigmoid(aIpos_raw))
        aI_neg = pm.Deterministic("aI_neg", pm.math.sigmoid(aIneg_raw))
        a_E = pm.Deterministic("a_E", pm.math.sigmoid(aE_raw))
        a_U = pm.Deterministic("a_U", pm.math.sigmoid(aU_raw))
        
        # Outcome noise (for scaling the simplified log probs)
        sigma_out_enj = pm.HalfNormal("sigma_out_enj", 0.5)
        sigma_out_uti = pm.HalfNormal("sigma_out_uti", 0.5)
        
        # ===== LIKELIHOOD FOR EACH PERSON =====
        for i, pp in enumerate(per_person):
            # Get person-specific parameters
            w_I_p = w_I[i]
            w_E_p = w_E[i]
            w_U_p = w_U[i]
            aI_pos_p = aI_pos[i]
            aI_neg_p = aI_neg[i]
            a_E_p = a_E[i]
            a_U_p = a_U[i]
            
            # Convert data to PyTensor tensors
            choice_t = pt.as_tensor_variable(pp["choice"])
            sug_t = pt.as_tensor_variable(pp["suggestion"])
            e_out_t = pt.as_tensor_variable(pp["e_out"])
            u_out_t = pt.as_tensor_variable(pp["u_out"])
            inst0_t = pt.as_tensor_variable(pp["inst0"])
            enj0_t = pt.as_tensor_variable(pp["enj0"])
            uti0_t = pt.as_tensor_variable(pp["uti0"])
            
            # Compute log probability using scan
            logp = simulate_forward_scan(
                w_I_p, w_E_p, w_U_p, aI_pos_p, aI_neg_p, a_E_p, a_U_p, tau_fixed,
                inst0_t, enj0_t, uti0_t, sug_t, choice_t, e_out_t, u_out_t
            )
            
            # Scale by outcome noise
            logp_scaled = logp / (sigma_out_enj ** 2 + sigma_out_uti ** 2 + 1e-6)
            
            # Add to model
            pm.Potential(f"ll_person_{pp['person_id']}", logp_scaled)
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Bayesian parameter recovery (PyMC, improved)"
    )
    parser.add_argument("--events", default="outputs/ema_events.csv")
    parser.add_argument("--people", default="outputs/ema_people.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--tau", type=float, default=3.0, help="Fixed softmax temperature")
    parser.add_argument("--draws", type=int, default=500, help="Number of MCMC draws")
    parser.add_argument("--tune", type=int, default=500, help="Number of tuning steps")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains")
    parser.add_argument("--cores", type=int, default=2, help="Number of cores")
    parser.add_argument("--target_accept", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    per_person, ppl = load_data(args.events, args.people)
    print(f"Loaded {len(per_person)} people with {sum(p['T'] for p in per_person)} total observations")
    
    print("Building model...")
    model = build_hierarchical_model(per_person, tau_fixed=args.tau)
    
    print("Sampling...")
    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            init="jitter+adapt_diag",
            return_inferencedata=True
        )
    
    print("Saving results...")
    # Summary statistics
    var_names = ["w_I", "w_E", "w_U", "aI_pos", "aI_neg", "a_E", "a_U", 
                 "sigma_out_enj", "sigma_out_uti",
                 "mu_wI", "mu_wE", "mu_wU", "mu_aIpos", "mu_aIneg", "mu_aE", "mu_aU"]
    summary = az.summary(idata, var_names=var_names, kind="stats")
    summary_path = outdir / "hierarchical_summary_v2.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary: {summary_path}")
    
    # Per-person comparisons
    persons = [pp["person_id"] for pp in per_person]
    rows = []
    for i, pid in enumerate(persons):
        row = {"person_id": pid}
        
        # Extract posterior means
        for var in ["w_I", "w_E", "w_U", "aI_pos", "aI_neg", "a_E", "a_U"]:
            if var in idata.posterior:
                post_samples = idata.posterior[var].sel({"dim_0": i}).values.flatten()
                row[f"{var}_mean"] = float(np.mean(post_samples))
                row[f"{var}_std"] = float(np.std(post_samples))
        
        # Get true values from person data
        true_row = ppl.loc[ppl.person_id == pid].iloc[0]
        for var, true_var in [
            ("w_I", "w_I"), ("w_E", "w_E"), ("w_U", "w_U"),
            ("aI_pos", "alpha_I_pos"), ("aI_neg", "alpha_I_neg"),
            ("a_E", "alpha_E"), ("a_U", "alpha_U")
        ]:
            if true_var in true_row:
                row[f"true_{var}"] = float(true_row[true_var])
        
        rows.append(row)
    
    person_df = pd.DataFrame(rows)
    person_path = outdir / "hierarchical_person_params_v2.csv"
    person_df.to_csv(person_path, index=False)
    print(f"Saved person-level parameters: {person_path}")
    
    # Save full inference data
    idata_path = outdir / "hierarchical_idata_v2.nc"
    idata.to_netcdf(idata_path)
    print(f"Saved inference data: {idata_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
