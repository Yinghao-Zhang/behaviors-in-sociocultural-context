"""
Hierarchical Bayesian parameter recovery - Simplified approach.
Uses simpler PyMC likelihood approach without scan.
"""
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd

# Fix for scipy.signal.gaussian
import scipy.signal
if not hasattr(scipy.signal, 'gaussian'):
    from scipy.signal.windows import gaussian
    scipy.signal.gaussian = gaussian

import pymc as pm
import arviz as az

# Force pytensor to use numpy mode to avoid C++ compilation issues
import pytensor
pytensor.config.cxx = ""

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


def simulate_forward_numpy(w_I, w_E, w_U, aI_pos, aI_neg, a_E, a_U, tau,
                           inst0, enj0, uti0, suggestion, choice, e_out, u_out,
                           sigma_e=0.3, sigma_u=0.3, return_probs=False):
    """
    Forward simulation using pure numpy for a single person.
    Returns log probability with proper variance scaling.
    
    Args:
        w_I, w_E, w_U: Weights for instinct, enjoyment, utility
        aI_pos, aI_neg: Learning rates for instinct
        a_E, a_U: Learning rates for enjoyment and utility
        tau: Softmax temperature (inverse noise in choice)
        inst0, enj0, uti0: Initial belief states
        suggestion, choice, e_out, u_out: Observed data
        sigma_e, sigma_u: Standard deviations for outcome noise
        return_probs: If True, return (logp, probs_history), else just logp
        
    Returns:
        logp: Total log probability
        probs_history: (T, 2) array of choice probabilities (only if return_probs=True)
    """
    inst = inst0.copy()
    enj = enj0.copy()
    uti = uti0.copy()
    
    logp = 0.0
    T = len(choice)
    
    # Constants for normal log-likelihood
    log_sqrt_2pi = 0.5 * np.log(2 * np.pi)
    
    if return_probs:
        probs_history = np.zeros((T, 2))
    
    for t in range(T):
        # Compute choice values
        CV = w_I * inst + w_E * enj + w_U * uti + suggestion[t]
        
        # Softmax to get probabilities
        exp_vals = np.exp(tau * CV)
        probs = exp_vals / np.sum(exp_vals)
        
        if return_probs:
            probs_history[t] = probs
        
        # Choice log likelihood (categorical)
        ct = choice[t]
        logp += np.log(probs[ct] + 1e-10)
        
        # Outcome log likelihoods (proper Normal distributions)
        e_mean = enj[ct]
        u_mean = uti[ct]
        
        # Log-likelihood for enjoyment outcome: log N(e_out | e_mean, sigma_e^2)
        logp += -log_sqrt_2pi - np.log(sigma_e) - 0.5 * ((e_out[t] - e_mean) / sigma_e) ** 2
        
        # Log-likelihood for utility outcome: log N(u_out | u_mean, sigma_u^2)
        logp += -log_sqrt_2pi - np.log(sigma_u) - 0.5 * ((u_out[t] - u_mean) / sigma_u) ** 2
        
        # Update belief states
        # One-hot mask for chosen behavior
        mask = np.array([1.0 if i == ct else 0.0 for i in range(2)])
        
        # Instinct update
        inst = inst + mask * aI_pos * (1.0 - inst) + (1.0 - mask) * aI_neg * (-1.0 - inst)
        
        # Enjoyment and utility updates (prediction error on chosen only)
        enj = enj + mask * a_E * (e_out[t] - enj)
        uti = uti + mask * a_U * (u_out[t] - uti)
        
        # Clip to [-1, 1] for stability
        inst = np.clip(inst, -1.0, 1.0)
        enj = np.clip(enj, -1.0, 1.0)
        uti = np.clip(uti, -1.0, 1.0)
    
    if return_probs:
        return logp, probs_history
    return logp


def build_hierarchical_model(per_person, tau_fixed=3.0):
    """
    Build hierarchical Bayesian model using custom likelihood.
    """
    P = len(per_person)
    
    with pm.Model() as model:
        # ===== HYPERPRIORS =====
        # Weights
        mu_wI = pm.Normal("mu_wI", 0.0, 1.0)
        sigma_wI = pm.HalfNormal("sigma_wI", 0.5)
        
        mu_wE = pm.Normal("mu_wE", 0.0, 1.0)
        sigma_wE = pm.HalfNormal("sigma_wE", 0.5)
        
        mu_wU = pm.Normal("mu_wU", 0.0, 1.0)
        sigma_wU = pm.HalfNormal("sigma_wU", 0.5)
        
        # Learning rates
        mu_aIpos = pm.Normal("mu_aIpos", 0.0, 1.0)
        sigma_aIpos = pm.HalfNormal("sigma_aIpos", 0.5)
        
        mu_aIneg = pm.Normal("mu_aIneg", 0.0, 1.0)
        sigma_aIneg = pm.HalfNormal("sigma_aIneg", 0.5)
        
        mu_aE = pm.Normal("mu_aE", 0.0, 1.0)
        sigma_aE = pm.HalfNormal("sigma_aE", 0.5)
        
        mu_aU = pm.Normal("mu_aU", 0.0, 1.0)
        sigma_aU = pm.HalfNormal("sigma_aU", 0.5)
        
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
        
        # Outcome noise parameters (separate for enjoyment and utility)
        sigma_e = pm.HalfNormal("sigma_e", 0.3)
        sigma_u = pm.HalfNormal("sigma_u", 0.3)
        
        # ===== CUSTOM LIKELIHOOD =====
        def logp_func(w_I_vals, w_E_vals, w_U_vals, 
                     aI_pos_vals, aI_neg_vals, a_E_vals, a_U_vals, 
                     sigma_e_val, sigma_u_val):
            """Compute total log probability across all persons."""
            total_logp = 0.0
            
            for i, pp in enumerate(per_person):
                logp_raw = simulate_forward_numpy(
                    w_I_vals[i], w_E_vals[i], w_U_vals[i],
                    aI_pos_vals[i], aI_neg_vals[i], a_E_vals[i], a_U_vals[i],
                    tau_fixed,
                    pp["inst0"], pp["enj0"], pp["uti0"],
                    pp["suggestion"], pp["choice"], pp["e_out"], pp["u_out"],
                    sigma_e_val, sigma_u_val
                )
                total_logp += logp_raw
            
            return total_logp
        
        # Register custom likelihood
        pm.Potential("likelihood", 
                    pm.math.sum(pm.logp(
                        pm.CustomDist.dist(
                            w_I, w_E, w_U, aI_pos, aI_neg, a_E, a_U, sigma_e, sigma_u,
                            logp=logp_func,
                            random=lambda *args, **kwargs: np.array(0.0)
                        ),
                        0
                    )))
    
    return model


def build_hierarchical_model_simple(per_person, tau_fixed=3.0):
    """
    Build hierarchical Bayesian model using PyMC Potential directly.
    Simpler approach that should work.
    """
    P = len(per_person)
    
    with pm.Model() as model:
        # ===== HYPERPRIORS =====
        mu_wI = pm.Normal("mu_wI", 0.0, 1.0)
        sigma_wI = pm.HalfNormal("sigma_wI", 0.5)
        
        mu_wE = pm.Normal("mu_wE", 0.0, 1.0)
        sigma_wE = pm.HalfNormal("sigma_wE", 0.5)
        
        mu_wU = pm.Normal("mu_wU", 0.0, 1.0)
        sigma_wU = pm.HalfNormal("sigma_wU", 0.5)
        
        mu_aIpos = pm.Normal("mu_aIpos", 0.0, 1.0)
        sigma_aIpos = pm.HalfNormal("sigma_aIpos", 0.5)
        
        mu_aIneg = pm.Normal("mu_aIneg", 0.0, 1.0)
        sigma_aIneg = pm.HalfNormal("sigma_aIneg", 0.5)
        
        mu_aE = pm.Normal("mu_aE", 0.0, 1.0)
        sigma_aE = pm.HalfNormal("sigma_aE", 0.5)
        
        mu_aU = pm.Normal("mu_aU", 0.0, 1.0)
        sigma_aU = pm.HalfNormal("sigma_aU", 0.5)
        
        # ===== PERSON-LEVEL PARAMETERS (raw) =====
        wI_raw = pm.Normal("wI_raw", mu=mu_wI, sigma=sigma_wI, shape=P)
        wE_raw = pm.Normal("wE_raw", mu=mu_wE, sigma=sigma_wE, shape=P)
        wU_raw = pm.Normal("wU_raw", mu=mu_wU, sigma=sigma_wU, shape=P)
        
        aIpos_raw = pm.Normal("aIpos_raw", mu=mu_aIpos, sigma=sigma_aIpos, shape=P)
        aIneg_raw = pm.Normal("aIneg_raw", mu=mu_aIneg, sigma=sigma_aIneg, shape=P)
        aE_raw = pm.Normal("aE_raw", mu=mu_aE, sigma=sigma_aE, shape=P)
        aU_raw = pm.Normal("aU_raw", mu=mu_aU, sigma=sigma_aU, shape=P)
        
        # ===== TRANSFORMATIONS =====
        w_I = pm.Deterministic("w_I", 1.5 * pm.math.sigmoid(wI_raw))
        w_E = pm.Deterministic("w_E", 1.5 * pm.math.sigmoid(wE_raw))
        w_U = pm.Deterministic("w_U", 1.5 * pm.math.sigmoid(wU_raw))
        
        aI_pos = pm.Deterministic("aI_pos", pm.math.sigmoid(aIpos_raw))
        aI_neg = pm.Deterministic("aI_neg", pm.math.sigmoid(aIneg_raw))
        a_E = pm.Deterministic("a_E", pm.math.sigmoid(aE_raw))
        a_U = pm.Deterministic("a_U", pm.math.sigmoid(aU_raw))
        
        # Outcome noise parameters
        sigma_e = pm.HalfNormal("sigma_e", 0.3)
        sigma_u = pm.HalfNormal("sigma_u", 0.3)
        
        # Store data for likelihood calculation
        model.per_person = per_person
        model.tau_fixed = tau_fixed
    
    return model


def logp_for_sample(point, model):
    """Calculate log probability for a given parameter sample."""
    P = len(model.per_person)
    
    # Extract parameter values
    w_I_vals = 1.5 / (1.0 + np.exp(-point['wI_raw']))
    w_E_vals = 1.5 / (1.0 + np.exp(-point['wE_raw']))
    w_U_vals = 1.5 / (1.0 + np.exp(-point['wU_raw']))
    
    aI_pos_vals = 1.0 / (1.0 + np.exp(-point['aIpos_raw']))
    aI_neg_vals = 1.0 / (1.0 + np.exp(-point['aIneg_raw']))
    a_E_vals = 1.0 / (1.0 + np.exp(-point['aE_raw']))
    a_U_vals = 1.0 / (1.0 + np.exp(-point['aU_raw']))
    
    sigma_e_val = point['sigma_e']
    sigma_u_val = point['sigma_u']
    
    # Compute likelihood
    total_logp = 0.0
    for i, pp in enumerate(model.per_person):
        logp_raw = simulate_forward_numpy(
            w_I_vals[i], w_E_vals[i], w_U_vals[i],
            aI_pos_vals[i], aI_neg_vals[i], a_E_vals[i], a_U_vals[i],
            model.tau_fixed,
            pp["inst0"], pp["enj0"], pp["uti0"],
            pp["suggestion"], pp["choice"], pp["e_out"], pp["u_out"],
            sigma_e_val, sigma_u_val
        )
        total_logp += logp_raw
    
    return total_logp


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Bayesian parameter recovery - simplified"
    )
    parser.add_argument("--events", default="outputs/ema_events.csv")
    parser.add_argument("--people", default="outputs/ema_people.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target_accept", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    per_person, ppl = load_data(args.events, args.people)
    print(f"Loaded {len(per_person)} people with {sum(p['T'] for p in per_person)} total observations")
    
    print("\nBuilding model...")
    model = build_hierarchical_model_simple(per_person, tau_fixed=args.tau)
    
    print("Starting MCMC sampling with Metropolis-Hastings...")
    print("(Using Metropolis because the model has custom likelihood)")
    
    with model:
        # Use Metropolis sampler for custom likelihood
        step = pm.Metropolis()
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            step=step,
            chains=args.chains,
            random_seed=args.seed,
            return_inferencedata=True,
            progressbar=True
        )
        
        # Compute log probability for each sample
        print("\nComputing log probabilities...")
        logps = []
        for chain in range(args.chains):
            chain_logps = []
            for draw in range(args.draws):
                point = {var: idata.posterior[var].sel(chain=chain, draw=draw).values 
                        for var in ['wI_raw', 'wE_raw', 'wU_raw', 
                                   'aIpos_raw', 'aIneg_raw', 'aE_raw', 'aU_raw', 
                                   'sigma_e', 'sigma_u']}
                logp = logp_for_sample(point, model)
                chain_logps.append(logp)
            logps.append(chain_logps)
        
        # Add log probability to inference data
        import xarray as xr
        idata.add_groups({"log_likelihood": xr.Dataset({
            "logp": (["chain", "draw"], logps)
        })})
    
    print("\nSaving results...")
    # Summary statistics
    var_names = ["w_I", "w_E", "w_U", "aI_pos", "aI_neg", "a_E", "a_U", 
                 "sigma_e", "sigma_u",
                 "mu_wI", "mu_wE", "mu_wU", "mu_aIpos", "mu_aIneg", "mu_aE", "mu_aU"]
    summary = az.summary(idata, var_names=var_names)
    summary_path = outdir / "hierarchical_summary_v3.csv"
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
                # Find the correct dimension name (e.g., 'w_I_dim_0', 'w_E_dim_0', etc.)
                dim_name = [d for d in idata.posterior[var].dims if 'dim' in d][0]
                post_samples = idata.posterior[var].sel({dim_name: i}).values.flatten()
                row[f"{var}_mean"] = float(np.mean(post_samples))
                row[f"{var}_std"] = float(np.std(post_samples))
        
        # Get true values
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
    person_path = outdir / "hierarchical_person_params_v3.csv"
    person_df.to_csv(person_path, index=False)
    print(f"Saved person-level parameters: {person_path}")
    
    # Save inference data
    idata_path = outdir / "hierarchical_idata_v3.nc"
    idata.to_netcdf(idata_path)
    print(f"Saved inference data: {idata_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
