
import numpy as np
from cmdstanpy import CmdStanModel
import argparse, pathlib, json
import pandas as pd

def prepare_data(events_path, people_path, tau_fixed=3.0):
    df = pd.read_csv(events_path)
    ppl = pd.read_csv(people_path)

    # Ensure chronological order
    df = df.sort_values(["person_id", "t"]).reset_index(drop=True)

    persons = sorted(df.person_id.unique())
    pid_map = {p:i for i,p in enumerate(persons)}
    P = len(persons)
    B = 2
    # Build per-person counts
    T_counts = df.groupby("person_id")["t"].count().reindex(persons).tolist()
    # Event-level arrays
    E = len(df)
    person_idx = np.array([pid_map[p] + 1 for p in df.person_id])  # 1-based for Stan
    choice = np.array([0 if b=="avoid_conflict" else 1 for b in df.choice_behavior], dtype=int)

    # Baselines (from people file)
    instinct0 = np.zeros((P,B))
    enjoyment0 = np.zeros((P,B))
    utility0 = np.zeros((P,B))
    for p in persons:
        row = ppl.loc[ppl.person_id==p].iloc[0]
        instinct0[pid_map[p],0] = row["instinct_avoid_conflict_0"]
        instinct0[pid_map[p],1] = row["instinct_approach_conflict_care_0"]
        enjoyment0[pid_map[p],0] = row["enjoyment_avoid_conflict_0"]
        enjoyment0[pid_map[p],1] = row["enjoyment_approach_conflict_care_0"]
        utility0[pid_map[p],0]   = row["utility_avoid_conflict_0"]
        utility0[pid_map[p],1]   = row["utility_approach_conflict_care_0"]

    # Suggestion terms
    suggestion_term = np.zeros((E,B))
    if f"suggest_term_avoid_conflict" in df.columns:
        suggestion_term[:,0] = df["suggest_term_avoid_conflict"].fillna(0).values
        suggestion_term[:,1] = df["suggest_term_approach_conflict_care"].fillna(0).values

    data = dict(
        P=P, B=B, E=E,
        person=person_idx,
        choice=choice,
        T_per_person=T_counts,
        instinct0=instinct0,
        enjoyment0=enjoyment0,
        utility0=utility0,
        enjoyment_out=df.enjoyment_out.values,
        utility_out=df.utility_out.values,
        suggestion_term=suggestion_term,
        tau_fixed=tau_fixed
    )
    return data, persons, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="outputs/ema_events.csv")
    ap.add_argument("--people", default="outputs/ema_people.csv")
    ap.add_argument("--stan", default="experiments/hierarchical_recovery.stan")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--iter", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=750)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--thin", type=int, default=1)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)

    data, persons, ppl = prepare_data(args.events, args.people, tau_fixed=args.tau)
    model = CmdStanModel(stan_file=args.stan)
    fit = model.sample(
        data=data,
        chains=args.chains,
        iter_sampling=args.iter,
        iter_warmup=args.warmup,
        thin=args.thin,
        adapt_delta=0.9,
        show_progress=True
    )

    fit.save_csvfiles(outdir)

    # Extract draws
    import numpy as np
    summary = fit.summary()
    summary.to_csv(outdir / "hierarchical_summary.csv")
    print("Saved summary:", outdir / "hierarchical_summary.csv")

    # Person-level posterior means
    draws = fit.draws_pd()
    person_params = []
    for i,p in enumerate(persons):
        row = dict(person_id=p)
        for param in ["w_I_post","w_E_post","w_U_post","noise_s_post",
                      "aI_pos_post","aI_neg_post","a_E_post","a_U_post"]:
            col = f"{param}[{i+1}]"
            if col in draws.columns:
                row[param+"_mean"] = draws[col].mean()
        # attach true values
        true_row = ppl.loc[ppl.person_id==p].iloc[0]
        for k in ["w_I","w_E","w_U","noise_s","alpha_I_pos","alpha_I_neg","alpha_E","alpha_U"]:
            if k in true_row:
                row["true_"+k] = true_row[k]
        person_params.append(row)

    pd.DataFrame(person_params).to_csv(outdir / "hierarchical_person_params.csv", index=False)
    print("Saved per-person posterior means:", outdir / "hierarchical_person_params.csv")

if __name__ == "__main__":
    main()