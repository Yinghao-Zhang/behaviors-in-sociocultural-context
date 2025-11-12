import argparse
import pathlib
import numpy as np
import pandas as pd

# Fix for scipy.signal.gaussian removal in scipy >= 1.10
# This monkey-patches scipy before arviz imports it
import scipy.signal
if not hasattr(scipy.signal, 'gaussian'):
    from scipy.signal.windows import gaussian
    scipy.signal.gaussian = gaussian

import pymc as pm
import pytensor.tensor as pt
import arviz as az

BEH_KEYS = ["avoid_conflict", "approach_conflict_care"]

def load_data(events_path: str, people_path: str):
    df = pd.read_csv(events_path)
    ppl = pd.read_csv(people_path)

    df = df.sort_values(["person_id", "t"]).reset_index(drop=True)
    persons = sorted(df.person_id.unique())
    pid_map = {p: i for i, p in enumerate(persons)}

    # Build per-person sequences
    per = []
    for p in persons:
        ev = df[df.person_id == p].copy()
        T = len(ev)
        # choices as 0/1
        choice = (ev["choice_behavior"].values == BEH_KEYS[1]).astype("int64")
        # suggestion terms (default 0 if missing)
        sug0 = ev.get(f"suggest_term_{BEH_KEYS[0]}", pd.Series([0.0] * T)).fillna(0.0).values
        sug1 = ev.get(f"suggest_term_{BEH_KEYS[1]}", pd.Series([0.0] * T)).fillna(0.0).values
        suggestion = np.stack([sug0, sug1], axis=1)  # T x 2
        e_out = ev["enjoyment_out"].values.astype("float64")
        u_out = ev["utility_out"].values.astype("float64")

        # baselines
        base = ppl.loc[ppl.person_id == p].iloc[0]
        inst0 = np.array([base[f"instinct_{k}_0"] for k in BEH_KEYS], dtype="float64")
        enj0 = np.array([base[f"enjoyment_{k}_0"] for k in BEH_KEYS], dtype="float64")
        uti0 = np.array([base[f"utility_{k}_0"]   for k in BEH_KEYS], dtype="float64")

        per.append(dict(
            person_id=int(p), T=T, choice=choice, suggestion=suggestion,
            e_out=e_out, u_out=u_out, inst0=inst0, enj0=enj0, uti0=uti0
        ))
    return per, ppl

def person_logp(
    w_I_p, w_E_p, w_U_p, aI_pos_p, aI_neg_p, a_E_p, a_U_p,
    tau_fixed,
    inst0, enj0, uti0, suggestion, choice, e_out, u_out,
    sigma_out_enj, sigma_out_uti
):
    # Convert arrays to tensors
    inst = pt.as_tensor_variable(inst0)
    enj  = pt.as_tensor_variable(enj0)
    uti  = pt.as_tensor_variable(uti0)

    sug = pt.as_tensor_variable(suggestion)  # T x 2
    ch  = pt.as_tensor_variable(choice)      # T
    eobs = pt.as_tensor_variable(e_out)      # T
    uobs = pt.as_tensor_variable(u_out)      # T

    logp = pt.as_tensor_variable(0.0)

    T = suggestion.shape[0]
    # Unroll small T loops in graph
    for t in range(int(T)):
        # Choice values and probabilities
        CV = w_I_p * inst + w_E_p * enj + w_U_p * uti + sug[t]  # shape (2,)
        probs = pm.math.softmax(tau_fixed * CV)

        # Categorical (choice) log-likelihood
        logp = logp + pm.logp(pm.Categorical.dist(p=probs), ch[t])

        # Outcome likelihoods (means are pre-update predictions)
        ct = ch[t]
        ct_f = pt.cast(ct, "float64")
        e_mean = enj[0] * (1.0 - ct_f) + enj[1] * ct_f
        u_mean = uti[0] * (1.0 - ct_f) + uti[1] * ct_f

        logp = logp + pm.logp(pm.Normal.dist(mu=e_mean, sigma=sigma_out_enj), eobs[t])
        logp = logp + pm.logp(pm.Normal.dist(mu=u_mean, sigma=sigma_out_uti), uobs[t])

        # Learning updates
        # mask for chosen behavior: [ct==0, ct==1]
        m0 = pt.cast(pt.eq(ct, 0), "float64")
        m1 = pt.cast(pt.eq(ct, 1), "float64")
        mask = pt.stack([m0, m1])

        # Instinct update
        inst = inst + mask * aI_pos_p * (1.0 - inst) + (1.0 - mask) * aI_neg_p * (-1.0 - inst)
        # Enjoyment/Utility PE updates (chosen only)
        enj = enj + mask * a_E_p * (eobs[t] - enj)
        uti = uti + mask * a_U_p * (uobs[t] - uti)

        # Soft clip to [-1, 1] to stabilize
        inst = pt.clip(inst, -1.0, 1.0)
        enj  = pt.clip(enj,  -1.0, 1.0)
        uti  = pt.clip(uti,  -1.0, 1.0)

    return logp

def build_model(per_person, tau_fixed=3.0):
    P = len(per_person)
    with pm.Model() as m:
        # Hyperpriors (raw scale)
        mu_wI = pm.Normal("mu_wI", 0.0, 1.0);  sigma_wI = pm.HalfNormal("sigma_wI", 1.0)
        mu_wE = pm.Normal("mu_wE", 0.0, 1.0);  sigma_wE = pm.HalfNormal("sigma_wE", 1.0)
        mu_wU = pm.Normal("mu_wU", 0.0, 1.0);  sigma_wU = pm.HalfNormal("sigma_wU", 1.0)

        mu_noise = pm.Normal("mu_noise", 0.0, 1.0); sigma_noise = pm.HalfNormal("sigma_noise", 1.0)

        mu_aIpos = pm.Normal("mu_aIpos", 0.0, 1.0); sigma_aIpos = pm.HalfNormal("sigma_aIpos", 1.0)
        mu_aIneg = pm.Normal("mu_aIneg", 0.0, 1.0); sigma_aIneg = pm.HalfNormal("sigma_aIneg", 1.0)
        mu_aE    = pm.Normal("mu_aE",    0.0, 1.0); sigma_aE    = pm.HalfNormal("sigma_aE", 1.0)
        mu_aU    = pm.Normal("mu_aU",    0.0, 1.0); sigma_aU    = pm.HalfNormal("sigma_aU", 1.0)

        # Person-level raw params
        wI_raw = pm.Normal("wI_raw", mu=mu_wI, sigma=sigma_wI, shape=P)
        wE_raw = pm.Normal("wE_raw", mu=mu_wE, sigma=sigma_wE, shape=P)
        wU_raw = pm.Normal("wU_raw", mu=mu_wU, sigma=sigma_wU, shape=P)

        noise_raw = pm.Normal("noise_raw", mu=mu_noise, sigma=sigma_noise, shape=P)

        aIpos_raw = pm.Normal("aIpos_raw", mu=mu_aIpos, sigma=sigma_aIpos, shape=P)
        aIneg_raw = pm.Normal("aIneg_raw", mu=mu_aIneg, sigma=sigma_aIneg, shape=P)
        aE_raw    = pm.Normal("aE_raw",    mu=mu_aE,    sigma=sigma_aE,    shape=P)
        aU_raw    = pm.Normal("aU_raw",    mu=mu_aU,    sigma=sigma_aU,    shape=P)

        # Transforms to constrained domains
        w_I = pm.Deterministic("w_I", 1.5 * pm.math.sigmoid(wI_raw))
        w_E = pm.Deterministic("w_E", 1.5 * pm.math.sigmoid(wE_raw))
        w_U = pm.Deterministic("w_U", 1.5 * pm.math.sigmoid(wU_raw))

        noise_s = pm.Deterministic("noise_s", 0.5 * pm.math.sigmoid(noise_raw))  # not used in likelihood (sparse-data caveat)

        aI_pos = pm.Deterministic("aI_pos", pm.math.sigmoid(aIpos_raw))
        aI_neg = pm.Deterministic("aI_neg", pm.math.sigmoid(aIneg_raw))
        a_E    = pm.Deterministic("a_E",    pm.math.sigmoid(aE_raw))
        a_U    = pm.Deterministic("a_U",    pm.math.sigmoid(aU_raw))

        # Outcome noise
        sigma_out_enj = pm.HalfNormal("sigma_out_enj", 0.5)
        sigma_out_uti = pm.HalfNormal("sigma_out_uti", 0.5)

        # Sum person log-likelihoods
        total_ll = 0.0
        for i, pp in enumerate(per_person):
            ll_p = person_logp(
                w_I[i], w_E[i], w_U[i], aI_pos[i], aI_neg[i], a_E[i], a_U[i],
                tau_fixed,
                pp["inst0"], pp["enj0"], pp["uti0"],
                pp["suggestion"], pp["choice"], pp["e_out"], pp["u_out"],
                sigma_out_enj, sigma_out_uti
            )
            pm.Potential(f"ll_person_{pp['person_id']}", ll_p)
            total_ll = total_ll + ll_p

        pm.Deterministic("total_ll", total_ll)
    return m

def main():
    ap = argparse.ArgumentParser(description="Hierarchical Bayesian parameter recovery (PyMC)")
    ap.add_argument("--events", default="outputs/ema_events.csv")
    ap.add_argument("--people", default="outputs/ema_people.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per, ppl = load_data(args.events, args.people)
    model = build_model(per, tau_fixed=args.tau)

    with model:
        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
            target_accept=args.target_accept, random_seed=args.seed,
            init="jitter+adapt_diag"
        )

    # Save summaries
    summ = az.summary(idata, var_names=["w_I","w_E","w_U","noise_s","aI_pos","aI_neg","a_E","a_U","sigma_out_enj","sigma_out_uti"], kind="stats")
    summ.to_csv(outdir / "hierarchical_summary_pymc.csv")
    print("Saved:", outdir / "hierarchical_summary_pymc.csv")

    # Per-person posterior means + ground truth for comparison
    persons = [pp["person_id"] for pp in per]
    rows = []
    for i, pid in enumerate(persons, start=1):
        row = {"person_id": pid}
        for v in ["w_I","w_E","w_U","noise_s","aI_pos","aI_neg","a_E","a_U"]:
            key = f"{v}"
            if key in idata.posterior:
                row[f"{v}_mean"] = float(idata.posterior[key].sel(chain=slice(None), draw=slice(None), **{"dim_0": i-1}).mean().values)
        # attach truth from ppl
        tru = ppl.loc[ppl.person_id == pid].iloc[0].to_dict()
        for k in ["w_I","w_E","w_U","noise_s","alpha_I_pos","alpha_I_neg","alpha_E","alpha_U"]:
            if k in tru:
                row[f"true_{k}"] = float(tru[k])
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "hierarchical_person_params_pymc.csv", index=False)
    print("Saved:", outdir / "hierarchical_person_params_pymc.csv")

if __name__ == "__main__":
    main()