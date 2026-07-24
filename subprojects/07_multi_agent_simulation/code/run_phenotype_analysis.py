from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import io
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ema_pair_simulation import (  # noqa: E402
    EMAPairSimulator,
    MomentaryReportCfg,
    ParamDistCfg,
    SocialCfg,
    default_behaviors,
    default_contexts,
)
from validate_predictions_between_person import (  # noqa: E402
    load_data,
    run_between_person_prediction,
    run_identifiability,
    run_parameter_recovery,
)


@dataclass(frozen=True)
class PhenotypeConfig:
    name: str
    label: str
    description: str
    ampd_hypothesis: str
    design_implication: str
    weight_mean: Tuple[float, float, float]
    weight_concentration: float = 24.0
    alpha_range: Tuple[float, float] = (0.10, 0.30)
    receptivity_range: Tuple[float, float] = (-0.20, 0.90)
    communion_range: Tuple[float, float] = (-0.50, 0.80)
    social_kappa_range: Tuple[float, float] = (0.00, 2.00)
    noise_range: Tuple[float, float] = (0.05, 0.20)
    tau_mode: str = "lognormal"
    tau_mean: float = 3.0
    tau_sigma: float = 0.20


def phenotype_configs() -> List[PhenotypeConfig]:
    return [
        PhenotypeConfig(
            name="baseline",
            label="Baseline",
            description="Balanced weighting of habit, hedonic value, and instrumental value with moderate learning.",
            ampd_hypothesis="Reference condition rather than a clinical trait hypothesis.",
            design_implication="Useful as a calibration condition for model comparison and recovery checks.",
            weight_mean=(1 / 3, 1 / 3, 1 / 3),
        ),
        PhenotypeConfig(
            name="habit_dominant",
            label="Instinct-Dominant",
            description="Behavior is strongly weighted toward automatic or repeated responses.",
            ampd_hypothesis="Trait rigidity, compulsive repetition, automatic avoidance, or low flexibility across contexts.",
            design_implication="Requires repeated observations of the same behavior-context pairing to separate habit from stable preference.",
            weight_mean=(0.65, 0.175, 0.175),
        ),
        PhenotypeConfig(
            name="affective_dominant",
            label="Enjoyment-Dominant",
            description="Behavior is strongly weighted toward immediate hedonic value or relief.",
            ampd_hypothesis="Disinhibition, impulsivity, urgency, or relief-driven avoidance.",
            design_implication="Requires AA items that separate short-term relief from instrumental consequences.",
            weight_mean=(0.175, 0.65, 0.175),
        ),
        PhenotypeConfig(
            name="goal_dominant",
            label="Utility-Dominant",
            description="Behavior is strongly weighted toward instrumental or long-term value.",
            ampd_hypothesis="More deliberative or goal-directed regulation; useful contrast against relief-driven behavior.",
            design_implication="Requires measurement of perceived goals, relationship consequences, and longer-horizon outcomes.",
            weight_mean=(0.175, 0.175, 0.65),
        ),
        PhenotypeConfig(
            name="fast_learners",
            label="Fast Learners",
            description="High learning rates across habit, enjoyment, and utility.",
            ampd_hypothesis="Affective instability or high sensitivity to recent interpersonal outcomes and feedback.",
            design_implication="Requires dense post-event sampling because parameters are driven by recent events.",
            weight_mean=(1 / 3, 1 / 3, 1 / 3),
            alpha_range=(0.30, 0.50),
        ),
        PhenotypeConfig(
            name="slow_learners",
            label="Slow Learners",
            description="Low learning rates across habit, enjoyment, and utility.",
            ampd_hypothesis="Rigid expectations, low corrective updating, or perseverative interpersonal beliefs.",
            design_implication="May require longer observation windows; dense bursts alone may not reveal change.",
            weight_mean=(1 / 3, 1 / 3, 1 / 3),
            alpha_range=(0.03, 0.12),
        ),
        PhenotypeConfig(
            name="high_social_influence",
            label="High Receptivity",
            description="High receptivity and positive communion amplify partner influence.",
            ampd_hypothesis="Dependency, rejection sensitivity, suggestibility, or strong interpersonal contingency.",
            design_implication="Requires detailed measurement of partner behavior, suggestion, and perceived feedback.",
            weight_mean=(1 / 3, 1 / 3, 1 / 3),
            receptivity_range=(0.30, 0.90),
            communion_range=(0.20, 0.80),
            social_kappa_range=(0.75, 2.00),
        ),
        PhenotypeConfig(
            name="low_social_influence",
            label="Low Receptivity",
            description="Low receptivity and low communion attenuate partner influence.",
            ampd_hypothesis="Detachment, interpersonal mistrust, restricted affiliation, or low responsiveness to social feedback.",
            design_implication="Partner context may have weak predictive value unless the protocol measures threat or avoidance cues directly.",
            weight_mean=(1 / 3, 1 / 3, 1 / 3),
            receptivity_range=(-0.50, 0.20),
            communion_range=(-0.50, 0.00),
            social_kappa_range=(0.00, 0.75),
        ),
        PhenotypeConfig(
            name="heterogeneous",
            label="Heterogeneous",
            description="Broad dispersion of weights and learning rates across people.",
            ampd_hypothesis="Mixed PD presentations, comorbidity, or broad transdiagnostic recruitment.",
            design_implication="Requires hierarchical or partial-pooling models and larger samples.",
            weight_mean=(1 / 3, 1 / 3, 1 / 3),
            weight_concentration=6.0,
            alpha_range=(0.05, 0.40),
            noise_range=(0.05, 0.30),
        ),
        PhenotypeConfig(
            name="homogeneous",
            label="Homogeneous",
            description="Narrow dispersion of weights and learning rates across people.",
            ampd_hypothesis="Targeted mechanism sample or restricted clinical range.",
            design_implication="Simpler population-level models may be adequate, but generalizability is limited.",
            weight_mean=(1 / 3, 1 / 3, 1 / 3),
            weight_concentration=80.0,
            alpha_range=(0.15, 0.25),
            noise_range=(0.05, 0.12),
        ),
    ]


class PhenotypeSimulator(EMAPairSimulator):
    def __init__(self, phenotype: PhenotypeConfig, *args, **kwargs):
        self.phenotype = phenotype
        super().__init__(*args, **kwargs)

    def _sample_weights(self) -> Tuple[float, float, float]:
        mean = np.array(self.phenotype.weight_mean, dtype=float)
        mean = mean / mean.sum()
        if self.dist_cfg.weight_mode == "relative":
            concentration = max(0.3, float(self.phenotype.weight_concentration))
            weights = self.rng.dirichlet(np.maximum(0.05, mean * concentration))
            return float(weights[0]), float(weights[1]), float(weights[2])

        sd = 0.15 / max(1.0, np.sqrt(self.phenotype.weight_concentration / 12.0))
        weights = self.rng.normal(mean, sd, size=3)
        weights = np.clip(weights, 0.05, 1.5)
        return float(weights[0]), float(weights[1]), float(weights[2])

    def _sample_alpha(self, low: float, high: float) -> float:
        del low, high
        lo, hi = self.phenotype.alpha_range
        return float(self.rng.uniform(lo, hi))

    def _sample_noise_s(self) -> float:
        lo, hi = self.phenotype.noise_range
        return float(self.rng.uniform(lo, hi))

    def _sample_social_kappa(self) -> float:
        lo, hi = self.phenotype.social_kappa_range
        return float(self.rng.uniform(lo, hi))

    def _sample_tau(self) -> float:
        if self.phenotype.tau_mode == "fixed":
            tau = self.phenotype.tau_mean
        elif self.phenotype.tau_mode == "bimodal":
            center = self.phenotype.tau_mean * (0.65 if self.rng.random() < 0.5 else 1.65)
            tau = float(self.rng.lognormal(mean=np.log(center), sigma=max(0.05, self.phenotype.tau_sigma)))
        else:
            tau = float(self.rng.lognormal(mean=np.log(self.phenotype.tau_mean), sigma=self.phenotype.tau_sigma))
        return float(np.clip(tau, 0.5, 10.0))


def parse_csv_values(raw: str, cast):
    if raw is None or raw == "":
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def select_profiles(raw: str) -> List[PhenotypeConfig]:
    configs = phenotype_configs()
    if raw == "all":
        return configs
    wanted = set(parse_csv_values(raw, str))
    by_name = {cfg.name: cfg for cfg in configs}
    missing = sorted(wanted.difference(by_name))
    if missing:
        raise ValueError(f"Unknown phenotype profile(s): {', '.join(missing)}")
    return [cfg for cfg in configs if cfg.name in wanted]


def make_social_cfg(profile: PhenotypeConfig, args) -> SocialCfg:
    return SocialCfg(
        p_observe=args.p_observe,
        p_suggest=args.p_suggest,
        p_feedback=args.p_feedback,
        n_partners_min=args.n_partners_min,
        n_partners_max=args.n_partners_max,
        receptivity_range=profile.receptivity_range,
        communion_range=profile.communion_range,
    )


def stable_profile_offset(name: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 1000)


def safe_prop(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return np.nan
    return float(np.mean(series == value))


def eu_report_corr(events: pd.DataFrame) -> float:
    vals = []
    for behavior in ["avoid_conflict", "approach_conflict_care"]:
        e_col = f"report_enjoyment_{behavior}"
        u_col = f"report_utility_{behavior}"
        if e_col not in events or u_col not in events:
            continue
        pair = events[[e_col, u_col]].dropna()
        if len(pair) >= 3 and pair[e_col].std() > 0 and pair[u_col].std() > 0:
            vals.append(float(pair[e_col].corr(pair[u_col])))
    return float(np.nanmean(vals)) if vals else np.nan


def summarize_events(
    events: pd.DataFrame,
    profile: PhenotypeConfig,
    replicate: int,
    sample_size: int,
    density: int,
    missing_rate: float,
) -> Dict[str, float]:
    choice = events[events["choice_behavior"].notna()].copy()
    approach_by_person = (
        choice.groupby("person_id")["choice_behavior"]
        .apply(lambda s: float(np.mean(s == "approach_conflict_care")))
        if not choice.empty
        else pd.Series(dtype=float)
    )
    row: Dict[str, float] = {
        "profile": profile.name,
        "profile_label": profile.label,
        "replicate": replicate,
        "sample_size": sample_size,
        "density": density,
        "missing_rate": missing_rate,
        "n_people": int(events["person_id"].nunique()) if "person_id" in events else 0,
        "n_events": int(len(events)),
        "n_choice": int(len(choice)),
        "n_partners": int(events["partner_id"].dropna().nunique()) if "partner_id" in events else 0,
        "approach_rate": safe_prop(choice["choice_behavior"], "approach_conflict_care") if not choice.empty else np.nan,
        "extreme_choice_fraction": float(np.mean((approach_by_person <= 0.10) | (approach_by_person >= 0.90))) if len(approach_by_person) else np.nan,
        "mean_report_EU_corr": eu_report_corr(events),
    }
    for key in ["observe", "solitary", "suggest", "observe_feedback"]:
        row[f"event_prop_{key}"] = safe_prop(events["situation_type"], key) if "situation_type" in events else np.nan
        row[f"choice_prop_{key}"] = safe_prop(choice["situation_type"], key) if not choice.empty and "situation_type" in choice else np.nan
    for setup_key, prop in events["setup_key"].value_counts(normalize=True).items() if "setup_key" in events else {}:
        row[f"setup_prop_{setup_key}"] = float(prop)
    return row


def summarize_outcomes(
    events: pd.DataFrame,
    profile: PhenotypeConfig,
    replicate: int,
    sample_size: int,
    density: int,
    missing_rate: float,
) -> List[Dict[str, float]]:
    choice = events[events["choice_behavior"].notna()].copy()
    if choice.empty:
        return []
    group_cols = ["situation_type", "setup_key", "choice_behavior"]
    rows = []
    for keys, grp in choice.groupby(group_cols, dropna=False):
        situation_type, setup_key, choice_behavior = keys
        rows.append({
            "profile": profile.name,
            "profile_label": profile.label,
            "replicate": replicate,
            "sample_size": sample_size,
            "density": density,
            "missing_rate": missing_rate,
            "situation_type": situation_type,
            "setup_key": setup_key,
            "choice_behavior": choice_behavior,
            "n": int(len(grp)),
            "enjoyment_mean": float(grp["enjoyment_out"].mean()),
            "enjoyment_sd": float(grp["enjoyment_out"].std()),
            "utility_mean": float(grp["utility_out"].mean()),
            "utility_sd": float(grp["utility_out"].std()),
        })
    return rows


def flatten_prediction_summary(pred_summary: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if pred_summary.empty:
        return out
    for _, row in pred_summary.iterrows():
        model = row["model"]
        for metric in ["accuracy", "balanced_accuracy", "log_loss", "auc", "pr_auc", "brier", "mcc", "ece"]:
            out[f"{model}_{metric}"] = float(row.get(f"{metric}_mean", np.nan))
            out[f"{model}_{metric}_se"] = float(row.get(f"{metric}_se", np.nan))
    return out


def maybe_quiet_call(func, *args, quiet: bool, **kwargs):
    if not quiet:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def summarize_recovery(outdir: Path, model_family: str) -> Dict[str, float]:
    rec_path = outdir / "parameter_recovery_summary.csv"
    ident_path = outdir / "parameter_identifiability_summary.csv"
    out: Dict[str, float] = {}
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        if not rec.empty:
            out["recovery_n_params"] = int(len(rec))
            out["mean_corr"] = float(rec["corr"].mean())
            out["mean_rmse"] = float(rec["rmse"].mean())
            out["mean_abs_bias"] = float(rec["bias"].abs().mean())
            decision_params = ["w_R"] if model_family == "reward" else ["w_E", "w_U"]
            out["decision_weight_corr"] = float(rec[rec["param"].isin(decision_params)]["corr"].mean())
            learning_params = ["a_R"] if model_family == "reward" else ["aI_pos", "aI_neg", "a_E", "a_U"]
            out["learning_rate_corr"] = float(rec[rec["param"].isin(learning_params)]["corr"].mean())
    if ident_path.exists():
        ident = pd.read_csv(ident_path)
        if not ident.empty:
            out["mean_ll_drop_max"] = float(ident["mean_ll_drop_max"].mean())
            out["mean_flat_fraction"] = float(ident["mean_flat_fraction"].mean())
            out["decision_weight_flat_fraction"] = float(ident[ident["param"].isin(["w_R"] if model_family == "reward" else ["w_E", "w_U"])]["mean_flat_fraction"].mean())
            out["learning_rate_flat_fraction"] = float(ident[ident["param"].isin(["a_R"] if model_family == "reward" else ["aI_pos", "aI_neg", "a_E", "a_U"])]["mean_flat_fraction"].mean())
    return out


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        col for col in df.columns
        if col not in set(group_cols + ["replicate", "profile_label"])
        and not col.endswith("_se")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_replicates"] = int(grp["replicate"].nunique()) if "replicate" in grp else int(len(grp))
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_se"] = float(vals.sem()) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_profile_reference(outdir: Path, profiles: Iterable[PhenotypeConfig]) -> None:
    rows = []
    for cfg in profiles:
        row = asdict(cfg)
        row["w_I_mean"], row["w_E_mean"], row["w_U_mean"] = row.pop("weight_mean")
        row["alpha_min"], row["alpha_max"] = row.pop("alpha_range")
        row["receptivity_min"], row["receptivity_max"] = row.pop("receptivity_range")
        row["communion_min"], row["communion_max"] = row.pop("communion_range")
        row["social_kappa_min"], row["social_kappa_max"] = row.pop("social_kappa_range")
        row["noise_min"], row["noise_max"] = row.pop("noise_range")
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "phenotype_definitions.csv", index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def write_markdown_report(outdir: Path, summary: pd.DataFrame, desc_summary: pd.DataFrame) -> None:
    lines = [
        "# Phenotype Analysis Report",
        "",
        "This report summarizes the canonical AMPD-oriented phenotype simulations produced by `run_phenotype_analysis.py`.",
        "",
        "## Outputs",
        "",
        "- `phenotype_definitions.csv`: profile definitions and AMPD-relevant hypotheses.",
        "- `phenotype_run_summary.csv`: one row per profile, replicate, sample size, density, missingness, and model family.",
        "- `phenotype_summary.csv`: replicate-aggregated model and recovery metrics.",
        "- `phenotype_descriptives.csv`: one row per simulated dataset.",
        "- `phenotype_descriptives_summary.csv`: replicate-aggregated descriptive statistics.",
        "- `phenotype_context_outcomes.csv`: outcome means/SDs by situation, context, and behavior.",
        "",
        "## Quick Diagnostics",
        "",
    ]
    if not summary.empty:
        diag_cols = [
            "profile",
            "sample_size",
            "density",
            "missing_rate",
            "model_family",
            "full_auc_mean",
            "full_pr_auc_mean",
            "full_log_loss_mean",
            "no_learn_log_loss_mean",
            "mean_corr_mean",
            "mean_flat_fraction_mean",
        ]
        available = [c for c in diag_cols if c in summary.columns]
        lines.append(markdown_table(summary[available]))
        lines.append("")
    if not desc_summary.empty:
        desc_cols = [
            "profile",
            "sample_size",
            "density",
            "missing_rate",
            "approach_rate_mean",
            "extreme_choice_fraction_mean",
            "mean_report_EU_corr_mean",
        ]
        available = [c for c in desc_cols if c in desc_summary.columns]
        lines.append("## Descriptive Balance")
        lines.append("")
        lines.append(markdown_table(desc_summary[available]))
        lines.append("")
    (outdir / "PHENOTYPE_ANALYSIS_REPORT.md").write_text("\n".join(lines))


def run_condition(
    profile: PhenotypeConfig,
    replicate: int,
    sample_size: int,
    density: int,
    missing_rate: float,
    args,
) -> Tuple[List[Dict[str, float]], Dict[str, float], List[Dict[str, float]]]:
    condition_id = (
        f"{profile.name}_rep{replicate:03d}_N{sample_size}"
        f"_density{density}_miss{int(round(missing_rate * 100)):03d}"
    )
    condition_dir = Path(args.outdir) / "runs" / condition_id
    condition_dir.mkdir(parents=True, exist_ok=True)

    behaviors = default_behaviors()
    sim = PhenotypeSimulator(
        profile,
        behaviors=behaviors,
        N=sample_size,
        t_count=density,
        social_cfg=make_social_cfg(profile, args),
        dist_cfg=ParamDistCfg(weight_mode=args.sim_weight_mode),
        momentary_cfg=MomentaryReportCfg(enabled=not args.no_momentary_reports),
        contexts=default_contexts(behaviors, mode=args.context_mode),
        missing_rate=missing_rate,
        missing_sd=missing_rate * 0.5 if missing_rate > 0 else 0.0,
        seed=(
            args.seed
            + replicate * 1009
            + sample_size * 53
            + density * 37
            + int(missing_rate * 10000)
            + stable_profile_offset(profile.name)
        ),
    )
    events_full, people = sim.run()
    if missing_rate > 0:
        events_obs, missing_summary = sim.apply_missingness(events_full)
        events_full.to_csv(condition_dir / "ema_events_full.csv", index=False)
        missing_summary.to_csv(condition_dir / "missingness_summary.csv", index=False)
        events = events_obs
    else:
        events = events_full
    events.to_csv(condition_dir / "ema_events.csv", index=False)
    people.to_csv(condition_dir / "ema_people.csv", index=False)

    descriptives = summarize_events(events, profile, replicate, sample_size, density, missing_rate)
    outcome_rows = summarize_outcomes(events, profile, replicate, sample_size, density, missing_rate)
    per_person, _ = load_data(str(condition_dir / "ema_events.csv"), str(condition_dir / "ema_people.csv"))

    model_families = ["tripartite"]
    if args.include_reward:
        model_families.append("reward")

    rows = []
    for model_family in model_families:
        family_dir = condition_dir / model_family
        family_dir.mkdir(parents=True, exist_ok=True)
        _, pred_summary = run_between_person_prediction(
            per_person,
            tau_mode=args.tau_mode,
            default_tau=args.default_tau,
            outdir=family_dir,
            train_frac=args.train_frac,
            seed=args.seed + replicate,
            max_train_people=args.prediction_max_train_people,
            weight_mode=args.recovery_weight_mode,
            use_measurement_likelihood=not args.no_measurement_likelihood,
            measurement_weight=args.measurement_weight,
            choice_input_mode=args.choice_input_mode,
            model_family=model_family,
        )
        if args.skip_recovery:
            recovery_df = pd.DataFrame()
        else:
            recovery_df = maybe_quiet_call(
                run_parameter_recovery,
                per_person,
                tau_mode=args.tau_mode,
                default_tau=args.default_tau,
                outdir=family_dir,
                max_people=args.recovery_max_people,
                weight_mode=args.recovery_weight_mode,
                use_measurement_likelihood=not args.no_measurement_likelihood,
                measurement_weight=args.measurement_weight,
                choice_input_mode=args.choice_input_mode,
                model_family=model_family,
                quiet=not args.verbose_validation,
            )
        if not args.skip_identifiability:
            maybe_quiet_call(
                run_identifiability,
                per_person,
                tau_mode=args.tau_mode,
                default_tau=args.default_tau,
                outdir=family_dir,
                max_people=args.identifiability_max_people,
                fit_df=recovery_df,
                weight_mode=args.recovery_weight_mode,
                use_measurement_likelihood=not args.no_measurement_likelihood,
                measurement_weight=args.measurement_weight,
                choice_input_mode=args.choice_input_mode,
                model_family=model_family,
                quiet=not args.verbose_validation,
            )
        row = {
            "profile": profile.name,
            "profile_label": profile.label,
            "replicate": replicate,
            "sample_size": sample_size,
            "density": density,
            "missing_rate": missing_rate,
            "model_family": model_family,
            **flatten_prediction_summary(pred_summary),
            **summarize_recovery(family_dir, model_family),
        }
        rows.append(row)

    return rows, descriptives, outcome_rows


def run_condition_job(job):
    idx, total, profile, replicate, sample_size, density, missing_rate, args_dict = job
    args = argparse.Namespace(**args_dict)
    rows, desc, outcomes = run_condition(profile, replicate, sample_size, density, missing_rate, args)
    return {
        "idx": idx,
        "total": total,
        "profile": profile.name,
        "replicate": replicate,
        "sample_size": sample_size,
        "density": density,
        "missing_rate": missing_rate,
        "rows": rows,
        "desc": desc,
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical AMPD-oriented phenotype simulations for project 07.")
    parser.add_argument("--outdir", default="subprojects/07_multi_agent_simulation/outputs_phenotype_analysis")
    parser.add_argument("--profiles", default="all", help="Comma-separated profile names or 'all'.")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--sample_sizes", default="", help="Optional comma-separated sample sizes. Overrides --N when supplied.")
    parser.add_argument("--tcount", type=int, default=50)
    parser.add_argument("--densities", default="", help="Optional comma-separated densities. Overrides --tcount when supplied.")
    parser.add_argument("--missing_rates", default="0.0", help="Comma-separated missingness rates, e.g. 0,0.2,0.4.")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--context_mode", default="varied", choices=["single", "varied", "orthogonal"])
    parser.add_argument("--sim_weight_mode", default="relative", choices=["relative", "raw"])
    parser.add_argument("--recovery_weight_mode", default="relative", choices=["relative", "raw"])
    parser.add_argument("--include_reward", action="store_true")
    parser.add_argument("--verbose_validation", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel condition workers.")
    parser.add_argument("--skip_recovery", action="store_true")
    parser.add_argument("--skip_identifiability", action="store_true")
    parser.add_argument("--recovery_max_people", type=int, default=30)
    parser.add_argument("--identifiability_max_people", type=int, default=12)
    parser.add_argument("--prediction_max_train_people", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--tau_mode", default="person", choices=["person", "fixed"])
    parser.add_argument("--default_tau", type=float, default=3.0)
    parser.add_argument("--choice_input_mode", default="reports", choices=["latent", "reports"])
    parser.add_argument("--measurement_weight", type=float, default=1.0)
    parser.add_argument("--no_measurement_likelihood", action="store_true")
    parser.add_argument("--no_momentary_reports", action="store_true")
    parser.add_argument("--p_observe", type=float, default=0.20)
    parser.add_argument("--p_suggest", type=float, default=0.30)
    parser.add_argument("--p_feedback", type=float, default=0.30)
    parser.add_argument("--n_partners_min", type=int, default=80)
    parser.add_argument("--n_partners_max", type=int, default=120)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    profiles = select_profiles(args.profiles)
    sample_sizes = parse_csv_values(args.sample_sizes, int) or [args.N]
    densities = parse_csv_values(args.densities, int) or [args.tcount]
    missing_rates = parse_csv_values(args.missing_rates, float) or [0.0]

    write_profile_reference(outdir, profiles)

    all_rows: List[Dict[str, float]] = []
    all_desc: List[Dict[str, float]] = []
    all_outcomes: List[Dict[str, float]] = []

    total = len(profiles) * args.reps * len(sample_sizes) * len(densities) * len(missing_rates)
    jobs = []
    idx = 0
    args_dict = vars(args).copy()
    for profile in profiles:
        for replicate in range(args.reps):
            for sample_size in sample_sizes:
                for density in densities:
                    for missing_rate in missing_rates:
                        idx += 1
                        jobs.append((idx, total, profile, replicate, sample_size, density, missing_rate, args_dict))

    if args.workers <= 1:
        for job in jobs:
            idx, total, profile, replicate, sample_size, density, missing_rate, _ = job
            print(
                f"\n=== [{idx}/{total}] {profile.name} rep={replicate} "
                f"N={sample_size} density={density} missing={missing_rate:.2f} ===",
                flush=True,
            )
            result = run_condition_job(job)
            all_rows.extend(result["rows"])
            all_desc.append(result["desc"])
            all_outcomes.extend(result["outcomes"])
    else:
        print(f"Running {total} conditions with {args.workers} workers.", flush=True)
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(run_condition_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                result = future.result()
                completed += 1
                print(
                    f"=== completed {completed}/{total}: [{result['idx']}/{total}] "
                    f"{result['profile']} rep={result['replicate']} "
                    f"N={result['sample_size']} density={result['density']} "
                    f"missing={result['missing_rate']:.2f} ===",
                    flush=True,
                )
                all_rows.extend(result["rows"])
                all_desc.append(result["desc"])
                all_outcomes.extend(result["outcomes"])

    run_summary = pd.DataFrame(all_rows)
    desc_df = pd.DataFrame(all_desc)
    outcomes_df = pd.DataFrame(all_outcomes)
    run_summary.to_csv(outdir / "phenotype_run_summary.csv", index=False)
    desc_df.to_csv(outdir / "phenotype_descriptives.csv", index=False)
    outcomes_df.to_csv(outdir / "phenotype_context_outcomes.csv", index=False)

    summary = aggregate(run_summary, ["profile", "sample_size", "density", "missing_rate", "model_family"]) if not run_summary.empty else pd.DataFrame()
    desc_summary = aggregate(desc_df, ["profile", "sample_size", "density", "missing_rate"]) if not desc_df.empty else pd.DataFrame()
    outcome_summary = aggregate(outcomes_df, ["profile", "sample_size", "density", "missing_rate", "situation_type", "setup_key", "choice_behavior"]) if not outcomes_df.empty else pd.DataFrame()
    summary.to_csv(outdir / "phenotype_summary.csv", index=False)
    desc_summary.to_csv(outdir / "phenotype_descriptives_summary.csv", index=False)
    outcome_summary.to_csv(outdir / "phenotype_context_outcomes_summary.csv", index=False)
    write_markdown_report(outdir, summary, desc_summary)

    print(f"\nSaved phenotype analysis outputs under: {outdir}")


if __name__ == "__main__":
    main()
