from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
    MomentaryReportCfg,
    ParamDistCfg,
    SocialCfg,
    default_behaviors,
    default_contexts,
)
from run_phenotype_analysis import (  # noqa: E402
    PhenotypeConfig,
    PhenotypeSimulator,
    flatten_prediction_summary,
    markdown_table,
    phenotype_configs,
    summarize_events,
    summarize_recovery,
)
from validate_predictions_between_person import (  # noqa: E402
    load_data,
    run_between_person_prediction,
    run_identifiability,
    run_parameter_recovery,
)


DIST_SCENARIOS = ("nominal", "skewed_clinical", "bimodal_clinical")
TAU_SCENARIOS = ("nominal", "low_tau", "high_tau", "bimodal_tau")
THRESHOLD_GENERATORS = ("softmax", "ddm")
THRESHOLD_FITS = ("softmax", "ddm")
REVIEWER_TARGETED_PROFILES = (
    "baseline",
    "habit_dominant",
    "affective_dominant",
    "high_social_influence",
    "low_social_influence",
    "heterogeneous",
)
TARGETED_PROFILE_REASONS = {
    "baseline": "Calibration condition for interpreting whether a sensitivity changes the basic proof-of-concept pattern.",
    "habit_dominant": "Covers rigid, automatic, or perseverative behavior where habit can dominate context-sensitive learning.",
    "affective_dominant": "Covers relief-driven or impulsive choice, the clearest case for the reviewer's threshold/hedonic-override concern.",
    "high_social_influence": "Covers strong partner-contingent behavior and therefore the context-heavy side of the model.",
    "low_social_influence": "Covers detachment, mistrust, or low interpersonal responsiveness where partner context may carry weak signal.",
    "heterogeneous": "Covers mixed clinical recruitment and broad between-person dispersion, where distributional assumptions matter most.",
}


def parse_csv_values(raw: str, cast=str) -> List:
    if raw is None or raw == "":
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def stable_offset(*parts: str) -> int:
    joined = "::".join(parts)
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(joined)) % 100000)


def sample_scaled_beta(rng: np.random.Generator, low: float, high: float, a: float, b: float) -> float:
    return float(low + (high - low) * rng.beta(a, b))


def sample_bimodal(rng: np.random.Generator, low: float, high: float) -> float:
    span = high - low
    center = low + (0.18 if rng.random() < 0.5 else 0.82) * span
    return float(np.clip(rng.normal(center, 0.06 * span), low, high))


class SupplementalPhenotypeSimulator(PhenotypeSimulator):
    def __init__(
        self,
        phenotype: PhenotypeConfig,
        *args,
        distribution_scenario: str = "nominal",
        tau_scenario: str = "nominal",
        **kwargs,
    ):
        self.distribution_scenario = distribution_scenario
        self.tau_scenario = tau_scenario
        super().__init__(phenotype, *args, **kwargs)

    def _sample_weights(self) -> Tuple[float, float, float]:
        mean = np.array(self.phenotype.weight_mean, dtype=float)
        mean = mean / mean.sum()
        if self.distribution_scenario == "nominal":
            return super()._sample_weights()

        if self.distribution_scenario == "skewed_clinical":
            concentration = max(1.2, float(self.phenotype.weight_concentration) / 5.0)
            weights = self.rng.dirichlet(np.maximum(0.03, mean * concentration))
            return float(weights[0]), float(weights[1]), float(weights[2])

        if self.distribution_scenario == "bimodal_clinical":
            dominant = int(np.argmax(mean))
            if np.isclose(mean.max(), mean.min(), atol=0.04):
                mode_a = np.array([0.72, 0.14, 0.14])
                mode_b = np.array([0.14, 0.43, 0.43])
            else:
                mode_a = mean.copy()
                mode_a[dominant] = min(0.86, max(0.70, mode_a[dominant] + 0.18))
                remainder = 1.0 - mode_a[dominant]
                for idx in range(3):
                    if idx != dominant:
                        mode_a[idx] = remainder / 2.0
                mode_b = mean.copy()
                mode_b[dominant] = max(0.34, mode_b[dominant] - 0.22)
                remainder = 1.0 - mode_b[dominant]
                others = [idx for idx in range(3) if idx != dominant]
                mode_b[others[0]] = remainder * 0.65
                mode_b[others[1]] = remainder * 0.35
            center = mode_a if self.rng.random() < 0.5 else mode_b
            weights = self.rng.dirichlet(np.maximum(0.04, center * 42.0))
            return float(weights[0]), float(weights[1]), float(weights[2])

        raise ValueError(f"Unknown distribution_scenario: {self.distribution_scenario}")

    def _sample_alpha(self, low: float, high: float) -> float:
        lo, hi = self.phenotype.alpha_range
        if self.distribution_scenario == "skewed_clinical":
            return sample_scaled_beta(self.rng, lo, hi, 1.4, 4.5)
        if self.distribution_scenario == "bimodal_clinical":
            return sample_bimodal(self.rng, lo, hi)
        return super()._sample_alpha(low, high)

    def _sample_noise_s(self) -> float:
        lo, hi = self.phenotype.noise_range
        if self.distribution_scenario == "skewed_clinical":
            return sample_scaled_beta(self.rng, lo, hi, 1.3, 3.8)
        if self.distribution_scenario == "bimodal_clinical":
            return sample_bimodal(self.rng, lo, hi)
        return super()._sample_noise_s()

    def _sample_social_kappa(self) -> float:
        lo, hi = self.phenotype.social_kappa_range
        if self.distribution_scenario == "skewed_clinical":
            return sample_scaled_beta(self.rng, lo, hi, 1.5, 3.5)
        if self.distribution_scenario == "bimodal_clinical":
            return sample_bimodal(self.rng, lo, hi)
        return super()._sample_social_kappa()

    def _sample_tau(self) -> float:
        if self.tau_scenario == "low_tau":
            tau = float(self.rng.lognormal(mean=np.log(1.0), sigma=0.25))
        elif self.tau_scenario == "high_tau":
            tau = float(self.rng.lognormal(mean=np.log(6.0), sigma=0.25))
        elif self.tau_scenario == "bimodal_tau":
            center = 1.0 if self.rng.random() < 0.5 else 6.0
            tau = float(self.rng.lognormal(mean=np.log(center), sigma=0.20))
        elif self.distribution_scenario == "bimodal_clinical":
            center = 1.25 if self.rng.random() < 0.5 else 6.0
            tau = float(self.rng.lognormal(mean=np.log(center), sigma=0.18))
        elif self.distribution_scenario == "skewed_clinical":
            tau = float(0.5 + 9.5 * self.rng.beta(1.4, 3.8))
        else:
            return super()._sample_tau()
        return float(np.clip(tau, 0.5, 10.0))


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


def maybe_quiet_call(func, *args, quiet: bool, **kwargs):
    if not quiet:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def run_fit(
    per_person,
    family_dir: Path,
    args,
    *,
    fit_decision_model: str,
    tau_mode: str,
    model_family: str = "tripartite",
    run_recovery: bool = True,
    run_identifiability_check: bool = True,
) -> Dict[str, float]:
    family_dir.mkdir(parents=True, exist_ok=True)
    _, pred_summary = run_between_person_prediction(
        per_person,
        tau_mode=tau_mode,
        default_tau=args.default_tau,
        outdir=family_dir,
        train_frac=args.train_frac,
        seed=args.seed,
        max_train_people=args.prediction_max_train_people,
        decision_model=fit_decision_model,
        weight_mode=args.recovery_weight_mode,
        use_measurement_likelihood=not args.no_measurement_likelihood,
        measurement_weight=args.measurement_weight,
        choice_input_mode=args.choice_input_mode,
        model_family=model_family,
    )
    recovery_df = pd.DataFrame()
    if run_recovery:
        recovery_df = maybe_quiet_call(
            run_parameter_recovery,
            per_person,
            tau_mode=tau_mode,
            default_tau=args.default_tau,
            outdir=family_dir,
            max_people=args.recovery_max_people,
            decision_model=fit_decision_model,
            weight_mode=args.recovery_weight_mode,
            use_measurement_likelihood=not args.no_measurement_likelihood,
            measurement_weight=args.measurement_weight,
            choice_input_mode=args.choice_input_mode,
            model_family=model_family,
            quiet=not args.verbose_validation,
        )
    if run_identifiability_check:
        maybe_quiet_call(
            run_identifiability,
            per_person,
            tau_mode=tau_mode,
            default_tau=args.default_tau,
            outdir=family_dir,
            max_people=args.identifiability_max_people,
            fit_df=recovery_df,
            decision_model=fit_decision_model,
            weight_mode=args.recovery_weight_mode,
            use_measurement_likelihood=not args.no_measurement_likelihood,
            measurement_weight=args.measurement_weight,
            choice_input_mode=args.choice_input_mode,
            model_family=model_family,
            quiet=not args.verbose_validation,
        )
    return {
        **flatten_prediction_summary(pred_summary),
        **summarize_recovery(family_dir, model_family),
    }


def fit_specs_for_condition(analysis: str, args, generator_decision_model: str) -> List[Dict]:
    if analysis == "tau":
        return [
            {"fit_decision_model": "softmax", "tau_mode": "person", "label": "tau_person", "recovery": True},
            {"fit_decision_model": "softmax", "tau_mode": "fixed", "label": "tau_fixed", "recovery": True},
        ]
    if analysis == "threshold":
        return [
            {
                "fit_decision_model": fit_model,
                "tau_mode": "person",
                "label": f"fit_{fit_model}",
                "recovery": fit_model == generator_decision_model,
            }
            for fit_model in THRESHOLD_FITS
        ]
    return [{"fit_decision_model": "softmax", "tau_mode": "person", "label": "primary", "recovery": True}]


def generation_settings_for_condition(analysis: str, scenario: str) -> Tuple[str, str, str]:
    distribution_scenario = "nominal"
    tau_scenario = "nominal"
    generator_decision_model = "softmax"
    if analysis == "distribution":
        distribution_scenario = scenario
    elif analysis == "tau":
        tau_scenario = scenario
    elif analysis == "threshold":
        generator_decision_model = scenario
    return distribution_scenario, tau_scenario, generator_decision_model


def scenario_values(analysis: str, args) -> List[str]:
    if analysis == "distribution":
        return parse_csv_values(args.distribution_scenarios) or list(DIST_SCENARIOS)
    if analysis == "tau":
        return parse_csv_values(args.tau_scenarios) or list(TAU_SCENARIOS)
    if analysis == "threshold":
        return parse_csv_values(args.threshold_generators) or list(THRESHOLD_GENERATORS)
    raise ValueError(f"Unknown analysis: {analysis}")


def condition_identity(analysis: str, profile: PhenotypeConfig, scenario: str, replicate: int) -> str:
    return f"{analysis}_{profile.name}_{scenario}_rep{replicate:03d}"


def condition_inventory(job) -> Dict:
    idx, total, analysis, profile, scenario, replicate, args_dict = job
    args = argparse.Namespace(**args_dict)
    condition_id = condition_identity(analysis, profile, scenario, replicate)
    condition_dir = Path(args.outdir) / "runs" / condition_id
    data_dir = condition_dir / "data"
    events_path = data_dir / "ema_events.csv"
    people_path = data_dir / "ema_people.csv"
    distribution_scenario, tau_scenario, generator_decision_model = generation_settings_for_condition(analysis, scenario)
    fit_specs = fit_specs_for_condition(analysis, args, generator_decision_model)

    required_prediction = 0
    completed_prediction = 0
    required_recovery = 0
    completed_recovery = 0
    required_identifiability = 0
    completed_identifiability = 0
    missing = []
    for fit_spec in fit_specs:
        fit_dir = condition_dir / fit_spec["label"]
        required_prediction += 1
        if (fit_dir / "prediction_validation_between_person_summary.csv").exists():
            completed_prediction += 1
        else:
            missing.append(f"{fit_spec['label']}:prediction")

        run_recovery = bool(fit_spec["recovery"] and not args.skip_recovery)
        run_ident = bool(run_recovery and not args.skip_identifiability)
        if run_recovery:
            required_recovery += 1
            if (fit_dir / "parameter_recovery_summary.csv").exists():
                completed_recovery += 1
            else:
                missing.append(f"{fit_spec['label']}:recovery")
        if run_ident:
            required_identifiability += 1
            if (fit_dir / "parameter_identifiability_summary.csv").exists():
                completed_identifiability += 1
            else:
                missing.append(f"{fit_spec['label']}:identifiability")

    data_complete = events_path.exists() and people_path.exists()
    if not data_complete:
        missing.append("data")
    complete = (
        data_complete
        and completed_prediction == required_prediction
        and completed_recovery == required_recovery
        and completed_identifiability == required_identifiability
    )
    return {
        "idx": idx,
        "total": total,
        "condition_id": condition_id,
        "analysis": analysis,
        "profile": profile.name,
        "profile_label": profile.label,
        "scenario": scenario,
        "replicate": replicate,
        "distribution_scenario": distribution_scenario,
        "tau_scenario": tau_scenario,
        "generator_decision_model": generator_decision_model,
        "data_complete": data_complete,
        "required_prediction_fits": required_prediction,
        "completed_prediction_fits": completed_prediction,
        "required_recovery_fits": required_recovery,
        "completed_recovery_fits": completed_recovery,
        "required_identifiability_fits": required_identifiability,
        "completed_identifiability_fits": completed_identifiability,
        "complete": complete,
        "missing": ";".join(missing),
    }


def completed_condition(job) -> Dict | None:
    idx, total, analysis, profile, scenario, replicate, args_dict = job
    args = argparse.Namespace(**args_dict)
    condition_id = condition_identity(analysis, profile, scenario, replicate)
    condition_dir = Path(args.outdir) / "runs" / condition_id
    data_dir = condition_dir / "data"
    events_path = data_dir / "ema_events.csv"
    people_path = data_dir / "ema_people.csv"
    if not events_path.exists() or not people_path.exists():
        return None

    distribution_scenario, tau_scenario, generator_decision_model = generation_settings_for_condition(analysis, scenario)
    fit_specs = fit_specs_for_condition(analysis, args, generator_decision_model)
    for fit_spec in fit_specs:
        fit_dir = condition_dir / fit_spec["label"]
        pred_path = fit_dir / "prediction_validation_between_person_summary.csv"
        if not pred_path.exists():
            return None
        run_recovery = bool(fit_spec["recovery"] and not args.skip_recovery)
        run_ident = bool(run_recovery and not args.skip_identifiability)
        if run_recovery and not (fit_dir / "parameter_recovery_summary.csv").exists():
            return None
        if run_ident and not (fit_dir / "parameter_identifiability_summary.csv").exists():
            return None

    seed = args.seed + replicate * 1009 + stable_offset(analysis, profile.name, scenario)
    events = pd.read_csv(events_path)
    desc = summarize_events(events, profile, replicate, args.N, args.tcount, args.missing_rate)
    desc.update({
        "analysis": analysis,
        "scenario": scenario,
        "distribution_scenario": distribution_scenario,
        "tau_scenario": tau_scenario,
        "generator_decision_model": generator_decision_model,
        "seed": seed,
        "resumed": True,
    })

    rows = []
    for fit_spec in fit_specs:
        fit_label = fit_spec["label"]
        fit_dir = condition_dir / fit_label
        pred_summary = pd.read_csv(fit_dir / "prediction_validation_between_person_summary.csv")
        run_recovery = bool(fit_spec["recovery"] and not args.skip_recovery)
        fit_summary = {
            **flatten_prediction_summary(pred_summary),
            **summarize_recovery(fit_dir, "tripartite"),
        }
        rows.append({
            "analysis": analysis,
            "profile": profile.name,
            "profile_label": profile.label,
            "scenario": scenario,
            "replicate": replicate,
            "N": args.N,
            "tcount": args.tcount,
            "missing_rate": args.missing_rate,
            "distribution_scenario": distribution_scenario,
            "tau_scenario": tau_scenario,
            "generator_decision_model": generator_decision_model,
            "fit_label": fit_label,
            "fit_decision_model": fit_spec["fit_decision_model"],
            "fit_tau_mode": fit_spec["tau_mode"],
            "recovery_identifiability_run": run_recovery,
            "seed": seed,
            "resumed": True,
            **fit_summary,
        })

    return {
        "idx": idx,
        "total": total,
        "analysis": analysis,
        "profile": profile.name,
        "scenario": scenario,
        "replicate": replicate,
        "rows": rows,
        "desc": desc,
        "resumed": True,
    }


def run_condition(job) -> Dict:
    idx, total, analysis, profile, scenario, replicate, args_dict = job
    args = argparse.Namespace(**args_dict)
    if getattr(args, "resume", False):
        completed = completed_condition(job)
        if completed is not None:
            return completed

    condition_id = condition_identity(analysis, profile, scenario, replicate)
    condition_dir = Path(args.outdir) / "runs" / condition_id
    data_dir = condition_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    distribution_scenario, tau_scenario, generator_decision_model = generation_settings_for_condition(analysis, scenario)

    seed = args.seed + replicate * 1009 + stable_offset(analysis, profile.name, scenario)
    behaviors = default_behaviors()
    sim = SupplementalPhenotypeSimulator(
        profile,
        behaviors=behaviors,
        N=args.N,
        t_count=args.tcount,
        social_cfg=make_social_cfg(profile, args),
        dist_cfg=ParamDistCfg(weight_mode=args.sim_weight_mode),
        momentary_cfg=MomentaryReportCfg(enabled=not args.no_momentary_reports),
        contexts=default_contexts(behaviors, mode=args.context_mode),
        decision_model=generator_decision_model,
        missing_rate=args.missing_rate,
        missing_sd=args.missing_rate * 0.5 if args.missing_rate > 0 else 0.0,
        seed=seed,
        distribution_scenario=distribution_scenario,
        tau_scenario=tau_scenario,
    )
    events_full, people = sim.run()
    if args.missing_rate > 0:
        events, missing_summary = sim.apply_missingness(events_full)
        events_full.to_csv(data_dir / "ema_events_full.csv", index=False)
        missing_summary.to_csv(data_dir / "missingness_summary.csv", index=False)
    else:
        events = events_full
    events.to_csv(data_dir / "ema_events.csv", index=False)
    people.to_csv(data_dir / "ema_people.csv", index=False)
    per_person, _ = load_data(str(data_dir / "ema_events.csv"), str(data_dir / "ema_people.csv"))

    desc = summarize_events(events, profile, replicate, args.N, args.tcount, args.missing_rate)
    desc.update({
        "analysis": analysis,
        "scenario": scenario,
        "distribution_scenario": distribution_scenario,
        "tau_scenario": tau_scenario,
        "generator_decision_model": generator_decision_model,
        "seed": seed,
        "resumed": False,
    })

    rows = []
    for fit_spec in fit_specs_for_condition(analysis, args, generator_decision_model):
        fit_label = fit_spec["label"]
        fit_dir = condition_dir / fit_label
        run_recovery = bool(fit_spec["recovery"] and not args.skip_recovery)
        run_ident = bool(run_recovery and not args.skip_identifiability)
        fit_summary = run_fit(
            per_person,
            fit_dir,
            args,
            fit_decision_model=fit_spec["fit_decision_model"],
            tau_mode=fit_spec["tau_mode"],
            run_recovery=run_recovery,
            run_identifiability_check=run_ident,
        )
        rows.append({
            "analysis": analysis,
            "profile": profile.name,
            "profile_label": profile.label,
            "scenario": scenario,
            "replicate": replicate,
            "N": args.N,
            "tcount": args.tcount,
            "missing_rate": args.missing_rate,
            "distribution_scenario": distribution_scenario,
            "tau_scenario": tau_scenario,
            "generator_decision_model": generator_decision_model,
            "fit_label": fit_label,
            "fit_decision_model": fit_spec["fit_decision_model"],
            "fit_tau_mode": fit_spec["tau_mode"],
            "recovery_identifiability_run": run_recovery,
            "seed": seed,
            "resumed": False,
            **fit_summary,
        })

    return {
        "idx": idx,
        "total": total,
        "analysis": analysis,
        "profile": profile.name,
        "scenario": scenario,
        "replicate": replicate,
        "rows": rows,
        "desc": desc,
    }


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        col for col in df.columns
        if col not in set(group_cols + ["replicate", "seed", "profile_label"])
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


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"no_learn_log_loss", "full_log_loss"}.issubset(out.columns):
        out["full_vs_no_learn_logloss_gain"] = out["no_learn_log_loss"] - out["full_log_loss"]
    if {"lr_log_loss", "full_log_loss"}.issubset(out.columns):
        out["full_vs_lr_logloss_gain"] = out["lr_log_loss"] - out["full_log_loss"]
    if {"null_log_loss", "full_log_loss"}.issubset(out.columns):
        out["full_vs_null_logloss_gain"] = out["null_log_loss"] - out["full_log_loss"]
    return out


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
    pd.DataFrame(rows).to_csv(outdir / "supplemental_profile_reference.csv", index=False)


def write_design_rationale(outdir: Path, analyses: List[str], profiles: List[PhenotypeConfig], args) -> None:
    profile_scope, profile_rationale = profile_selection_statement(profiles)
    rows = [
        {
            "reviewer_concern": "Clinical parameter distributions may be skewed, extreme, or bimodal.",
            "analysis": "distribution",
            "design_choice": f"Cross {profile_scope} with nominal, skewed_clinical, and bimodal_clinical parameter distributions.",
            "rationale": f"{profile_rationale} Skewed and bimodal scenarios perturb weights, learning rates, noise, social influence, and tau while retaining profile-level AMPD meaning.",
        },
        {
            "reviewer_concern": "Fixed temperature ignores erratic choice stochasticity and impulsivity.",
            "analysis": "tau",
            "design_choice": f"Cross {profile_scope} with nominal, low, high, and bimodal tau distributions, and fit both person-specific and fixed-tau variants.",
            "rationale": "Tau changes sensitivity of choices to value differences, whereas noise_s adds stochastic perturbation to value computation. Comparing person-specific versus fixed tau quantifies the cost of ignoring between-person temperature heterogeneity.",
        },
        {
            "reviewer_concern": "The linear additive behavioral selection mechanism may be too simple; threshold-like choice may be plausible.",
            "analysis": "threshold",
            "design_choice": f"Generate {profile_scope} under softmax and threshold/DDM-style choice, then fit both softmax and DDM-style decision rules.",
            "rationale": "This evaluates whether conclusions depend on the additive-softmax choice rule and whether a threshold-style alternative improves prediction or recovery when data are generated with such a rule.",
        },
        {
            "reviewer_concern": "Static social partners ignore co-created relationship dynamics.",
            "analysis": "network",
            "design_choice": "Handled in run_network_sensitivity.py as a structural sensitivity rather than phenotype-crossed analysis.",
            "rationale": "The network analysis isolates partner-learning and hidden social exposure. It is intentionally not crossed with all phenotypes in this runner to avoid confounding social-network structure with AMPD profile contrasts.",
        },
    ]
    pd.DataFrame(rows).to_csv(outdir / "supplemental_design_rationale.csv", index=False)
    profile_rows = [
        {
            "profile": cfg.name,
            "included": cfg.name in {p.name for p in profiles},
            "reason": targeted_profile_reason(cfg.name)
            if cfg.name in {p.name for p in profiles}
            else "Not selected for this supplemental grid; retained in the canonical phenotype and AA design-grid analyses.",
        }
        for cfg in phenotype_configs()
    ]
    pd.DataFrame(profile_rows).to_csv(outdir / "supplemental_profile_inclusion.csv", index=False)


def targeted_profile_reason(name: str) -> str:
    return TARGETED_PROFILE_REASONS.get(
        name,
        "Included to preserve coverage of the full AMPD-oriented phenotype set.",
    )


def profile_selection_statement(profiles: List[PhenotypeConfig]) -> Tuple[str, str]:
    selected = {p.name for p in profiles}
    all_profiles = {p.name for p in phenotype_configs()}
    if selected == all_profiles:
        return (
            "all 10 AMPD-oriented phenotypes",
            "Using all phenotypes avoids selecting only favorable trait presentations.",
        )
    anchor_names = [p.name for p in profiles]
    labels = ", ".join(anchor_names)
    if selected == set(REVIEWER_TARGETED_PROFILES):
        return (
            f"a prespecified {len(anchor_names)}-phenotype anchor subset ({labels})",
            "The anchor subset spans calibration, habit dominance, relief/affective dominance, high and low interpersonal influence, and heterogeneous clinical dispersion; excluded phenotypes remain represented in the canonical phenotype and AA design-grid analyses.",
        )
    return (
        f"a custom {len(anchor_names)}-phenotype subset ({labels})",
        "This custom subset was selected by command line; use the profile-inclusion table to document why each included phenotype was needed and avoid treating it as the reviewer-anchor preset.",
    )


def write_report(
    outdir: Path,
    summary: pd.DataFrame,
    desc_summary: pd.DataFrame,
    *,
    profiles: List[PhenotypeConfig],
    inventory_summary: pd.DataFrame | None = None,
    summarize_only: bool = False,
) -> None:
    _, profile_rationale = profile_selection_statement(profiles)
    lines = [
        "# Supplemental Sensitivity Analysis Report",
        "",
        "This report summarizes reviewer-requested sensitivity checks for project 07. The analyses are designed as supplements to the canonical AMPD phenotype and AA design-grid results rather than as a replacement for them.",
        "",
        "## Reviewer Concerns Addressed",
        "",
        "- Distributional assumptions: skewed and bimodal clinical parameter distributions.",
        "- Choice stochasticity: low, high, and bimodal inverse-temperature conditions, plus fixed-versus-person-specific tau fits.",
        "- Decision rule: softmax versus threshold/DDM-style behavioral selection.",
        "- Dynamic partners: handled by `run_network_sensitivity.py` and summarized separately.",
        "",
        "## Design Rationale",
        "",
        profile_rationale,
        "",
    ]
    if summarize_only:
        lines.extend([
            "This report was generated in summarize-only mode from completed condition folders already present on disk. Missing planned cells are listed in `supplemental_condition_inventory.csv`.",
            "",
        ])
    if inventory_summary is not None and not inventory_summary.empty:
        inv_cols = [
            "analysis",
            "scenario",
            "n_conditions",
            "n_complete",
            "completion_rate",
            "prediction_fit_completion_rate",
            "recovery_fit_completion_rate",
            "identifiability_fit_completion_rate",
        ]
        available = [col for col in inv_cols if col in inventory_summary.columns]
        lines.extend(["## Completion Inventory", "", markdown_table(inventory_summary[available]), ""])
    if not summary.empty:
        key_cols = [
            "analysis",
            "scenario",
            "fit_label",
            "fit_decision_model",
            "fit_tau_mode",
            "n_replicates",
            "full_auc_mean",
            "full_pr_auc_mean",
            "full_log_loss_mean",
            "no_learn_log_loss_mean",
            "lr_log_loss_mean",
            "mean_corr_mean",
            "mean_flat_fraction_mean",
            "full_vs_no_learn_logloss_gain_mean",
            "full_vs_null_logloss_gain_mean",
        ]
        available = [col for col in key_cols if col in summary.columns]
        lines.extend(["## Overall Summary", "", markdown_table(summary[available]), ""])

        profile_cols = [
            "analysis",
            "profile",
            "scenario",
            "fit_label",
            "full_auc_mean",
            "full_pr_auc_mean",
            "full_log_loss_mean",
            "mean_corr_mean",
            "mean_flat_fraction_mean",
        ]
        profile_summary = summary.groupby(
            ["analysis", "profile", "scenario", "fit_label"],
            dropna=False,
        ).first().reset_index() if "profile" in summary.columns else pd.DataFrame()
        if not profile_summary.empty:
            display = profile_summary[[col for col in profile_cols if col in profile_summary.columns]].head(80)
            lines.extend(["## Profile-Level Snapshot", "", markdown_table(display), ""])

    if not desc_summary.empty:
        desc_cols = [
            "analysis",
            "scenario",
            "n_replicates",
            "n_events_mean",
            "n_choice_mean",
            "approach_rate_mean",
            "extreme_choice_fraction_mean",
        ]
        available = [col for col in desc_cols if col in desc_summary.columns]
        lines.extend(["## Data-Generation Diagnostics", "", markdown_table(desc_summary[available]), ""])

    lines.extend([
        "## Manuscript Use",
        "",
        "Use these results to state that the proof-of-concept conclusions were checked against non-normal clinical parameter distributions, tau heterogeneity, a threshold-style decision rule, and dynamic social partners. The supplement should still emphasize that these are synthetic-data robustness checks, not empirical validation.",
        "",
    ])
    (outdir / "SUPPLEMENTAL_SENSITIVITY_REPORT.md").write_text("\n".join(lines))


def summarize_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in inventory.groupby(["analysis", "scenario"], dropna=False):
        analysis, scenario = keys
        req_pred = grp["required_prediction_fits"].sum()
        req_rec = grp["required_recovery_fits"].sum()
        req_ident = grp["required_identifiability_fits"].sum()
        rows.append({
            "analysis": analysis,
            "scenario": scenario,
            "n_conditions": int(len(grp)),
            "n_complete": int(grp["complete"].sum()),
            "completion_rate": float(grp["complete"].mean()) if len(grp) else np.nan,
            "prediction_fit_completion_rate": float(grp["completed_prediction_fits"].sum() / req_pred) if req_pred else np.nan,
            "recovery_fit_completion_rate": float(grp["completed_recovery_fits"].sum() / req_rec) if req_rec else np.nan,
            "identifiability_fit_completion_rate": float(grp["completed_identifiability_fits"].sum() / req_ident) if req_ident else np.nan,
        })
    return pd.DataFrame(rows)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-requested supplemental sensitivity analyses for project 07.")
    parser.add_argument("--outdir", default="subprojects/07_multi_agent_simulation/outputs_supplemental_sensitivity")
    parser.add_argument("--grid_preset", default="custom", choices=["custom", "reviewer_targeted"])
    parser.add_argument("--analyses", default="distribution,tau,threshold")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--N", type=int, default=60)
    parser.add_argument("--tcount", type=int, default=50)
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--context_mode", default="varied", choices=["single", "varied", "orthogonal"])
    parser.add_argument("--sim_weight_mode", default="relative", choices=["relative", "raw"])
    parser.add_argument("--recovery_weight_mode", default="relative", choices=["relative", "raw"])
    parser.add_argument("--distribution_scenarios", default=",".join(DIST_SCENARIOS))
    parser.add_argument("--tau_scenarios", default=",".join(TAU_SCENARIOS))
    parser.add_argument("--threshold_generators", default=",".join(THRESHOLD_GENERATORS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Reuse complete condition outputs under --outdir.")
    parser.add_argument("--summarize_only", action="store_true", help="Aggregate complete condition folders already present under --outdir without running missing cells.")
    parser.add_argument("--resume_log_every", type=int, default=50, help="When resuming serially, log resumed cells only every N conditions.")
    parser.add_argument("--max_new_conditions", type=int, default=None, help="Stop after this many non-resumed conditions; useful for controlled serial batches.")
    parser.add_argument("--verbose_validation", action="store_true")
    parser.add_argument("--skip_recovery", action="store_true")
    parser.add_argument("--skip_identifiability", action="store_true")
    parser.add_argument("--recovery_max_people", type=int, default=30)
    parser.add_argument("--identifiability_max_people", type=int, default=8)
    parser.add_argument("--prediction_max_train_people", type=int, default=40)
    parser.add_argument("--train_frac", type=float, default=0.8)
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
    if args.grid_preset == "reviewer_targeted" and args.profiles == "all":
        args.profiles = ",".join(REVIEWER_TARGETED_PROFILES)
    if args.max_new_conditions is not None and args.workers > 1:
        raise ValueError("--max_new_conditions is only supported with --workers 1.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    analyses = parse_csv_values(args.analyses, str)
    valid_analyses = {"distribution", "tau", "threshold"}
    unknown = sorted(set(analyses).difference(valid_analyses))
    if unknown:
        raise ValueError(f"Unknown analysis/analyses: {', '.join(unknown)}")
    profiles = select_profiles(args.profiles)
    write_profile_reference(outdir, profiles)
    write_design_rationale(outdir, analyses, profiles, args)

    jobs = []
    idx = 0
    args_dict = vars(args).copy()
    for analysis in analyses:
        for profile in profiles:
            for scenario in scenario_values(analysis, args):
                for replicate in range(args.reps):
                    idx += 1
                    jobs.append((idx, 0, analysis, profile, scenario, replicate, args_dict))
    total = len(jobs)
    jobs = [(idx, total, analysis, profile, scenario, replicate, args_dict)
            for idx, _, analysis, profile, scenario, replicate, args_dict in jobs]

    all_rows: List[Dict] = []
    all_desc: List[Dict] = []
    inventory_rows: List[Dict] = []
    if args.summarize_only:
        for job in jobs:
            inventory_rows.append(condition_inventory(job))
            completed = completed_condition(job)
            if completed is not None:
                all_rows.extend(completed["rows"])
                all_desc.append(completed["desc"])
        print(
            f"Summarized {sum(row['complete'] for row in inventory_rows)}/{len(inventory_rows)} complete conditions under: {outdir}",
            flush=True,
        )
    elif args.workers <= 1:
        new_conditions = 0
        for job in jobs:
            idx, total, analysis, profile, scenario, replicate, _ = job
            result = run_condition(job)
            status = "resumed" if result.get("resumed") else "completed"
            if status != "resumed":
                new_conditions += 1
            should_log = (
                status != "resumed"
                or args.resume_log_every <= 1
                or idx == 1
                or idx == total
                or idx % args.resume_log_every == 0
            )
            if should_log:
                print(
                    f"=== {status}: [{idx}/{total}] {analysis} {profile.name} {scenario} "
                    f"rep={replicate} ===",
                    flush=True,
            )
            all_rows.extend(result["rows"])
            all_desc.append(result["desc"])
            if args.max_new_conditions is not None and new_conditions >= args.max_new_conditions:
                print(
                    f"Reached --max_new_conditions={args.max_new_conditions}; stopping this batch early.",
                    flush=True,
                )
                break
    else:
        print(f"Running {total} supplemental conditions with {args.workers} workers.", flush=True)
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(run_condition, job): job for job in jobs}
            for future in as_completed(future_to_job):
                result = future.result()
                completed += 1
                status = "resumed" if result.get("resumed") else "completed"
                should_log = (
                    status != "resumed"
                    or args.resume_log_every <= 1
                    or completed == 1
                    or completed == total
                    or completed % args.resume_log_every == 0
                )
                if should_log:
                    print(
                        f"=== {status} {completed}/{total}: [{result['idx']}/{total}] "
                        f"{result['analysis']} {result['profile']} {result['scenario']} "
                        f"rep={result['replicate']} ===",
                        flush=True,
                    )
                all_rows.extend(result["rows"])
                all_desc.append(result["desc"])

    if not inventory_rows:
        inventory_rows = [condition_inventory(job) for job in jobs]
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(outdir / "supplemental_condition_inventory.csv", index=False)
    inventory_summary = summarize_inventory(inventory)
    inventory_summary.to_csv(outdir / "supplemental_condition_inventory_summary.csv", index=False)

    run_summary = add_derived_metrics(pd.DataFrame(all_rows))
    desc_df = pd.DataFrame(all_desc)
    run_summary.to_csv(outdir / "supplemental_run_summary.csv", index=False)
    desc_df.to_csv(outdir / "supplemental_descriptives.csv", index=False)

    group_cols = [
        "analysis",
        "profile",
        "scenario",
        "fit_label",
        "fit_decision_model",
        "fit_tau_mode",
    ]
    summary = aggregate(run_summary, group_cols) if not run_summary.empty else pd.DataFrame()
    overall = aggregate(
        run_summary,
        ["analysis", "scenario", "fit_label", "fit_decision_model", "fit_tau_mode"],
    ) if not run_summary.empty else pd.DataFrame()
    desc_summary = aggregate(desc_df, ["analysis", "scenario"]) if not desc_df.empty else pd.DataFrame()
    summary.to_csv(outdir / "supplemental_summary_by_profile.csv", index=False)
    overall.to_csv(outdir / "supplemental_summary_overall.csv", index=False)
    desc_summary.to_csv(outdir / "supplemental_descriptives_summary.csv", index=False)
    write_report(
        outdir,
        overall,
        desc_summary,
        profiles=profiles,
        inventory_summary=inventory_summary,
        summarize_only=args.summarize_only,
    )
    print(f"\nSaved supplemental sensitivity outputs under: {outdir}")


if __name__ == "__main__":
    main()
