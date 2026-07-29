from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pdtrt_rerun_core import (
    PROFILE_NAMES,
    atomic_write_csv,
    atomic_write_json,
    build_panel_view,
    load_simulation_result,
    stable_seed,
)
from pdtrt_rerun_fit import (
    EmpiricalBayesConfig,
    effective_choice_consistency,
    fit_conditional_oracle,
    fit_partial_population_block,
    fit_population_model,
    prepare_people,
    unpack_theta,
)


MODES = (
    "full_process",
    "decision_weights",
    "decision_weights_oracle",
)
WEIGHT_NAMES = ("w_i", "w_e", "w_u")


def _csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(item) for item in _csv_strings(value)]


def _event_label(value: int) -> str:
    return f"{int(value):03d}"


def _latent_directory(
    source_root: Path,
    profile: str,
    environment: str,
    mean_events: int,
    replicate: int,
) -> Path:
    return (
        source_root
        / "conditions"
        / "generator=tripartite"
        / f"profile={profile}"
        / f"environment={environment}"
        / f"events={_event_label(mean_events)}"
        / f"replicate={replicate:03d}"
        / "latent"
    )


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _weight_summary(
    view,
    person_parameters: pd.DataFrame,
    population_theta: np.ndarray,
) -> Dict[str, float]:
    truth = view.people_truth[
        ["focal_id", *(f"true_{name}" for name in WEIGHT_NAMES)]
    ]
    fitted = person_parameters[
        ["focal_id", *(f"fit_{name}" for name in WEIGHT_NAMES)]
    ]
    merged = fitted.merge(truth, on="focal_id", how="inner")
    true_matrix = merged[
        [f"true_{name}" for name in WEIGHT_NAMES]
    ].to_numpy(dtype=float)
    fitted_matrix = merged[
        [f"fit_{name}" for name in WEIGHT_NAMES]
    ].to_numpy(dtype=float)
    population = unpack_theta("tripartite", population_theta)
    shared_matrix = np.tile(
        np.asarray([population[name] for name in WEIGHT_NAMES], dtype=float),
        (len(merged), 1),
    )

    individual_rmse = float(
        np.sqrt(np.mean((fitted_matrix - true_matrix) ** 2))
    )
    shared_rmse = float(
        np.sqrt(np.mean((shared_matrix - true_matrix) ** 2))
    )
    correlations = [
        _safe_correlation(true_matrix[:, index], fitted_matrix[:, index])
        for index in range(len(WEIGHT_NAMES))
    ]
    true_sd = np.std(true_matrix, axis=0, ddof=1)
    fitted_sd = np.std(fitted_matrix, axis=0, ddof=1)
    valid_sd = true_sd > 1e-10
    return {
        "participants_fitted": float(len(merged)),
        "weight_correlation_mean": float(np.nanmean(correlations)),
        **{
            f"{name}_correlation": correlations[index]
            for index, name in enumerate(WEIGHT_NAMES)
        },
        "weight_rmse": individual_rmse,
        "shared_weight_rmse": shared_rmse,
        "weight_rmse_improvement_over_shared": (
            float(1.0 - individual_rmse / shared_rmse)
            if shared_rmse > 1e-12
            else np.nan
        ),
        "weight_mean_absolute_error": float(
            np.mean(np.abs(fitted_matrix - true_matrix))
        ),
        "weight_dispersion_ratio": (
            float(np.mean(fitted_sd[valid_sd] / true_sd[valid_sd]))
            if np.any(valid_sd)
            else np.nan
        ),
        "dominant_weight_accuracy": float(
            np.mean(
                np.argmax(fitted_matrix, axis=1)
                == np.argmax(true_matrix, axis=1)
            )
        ),
    }


def _choice_consistency_summary(
    view,
    person_parameters: pd.DataFrame,
    population_theta: np.ndarray,
) -> Dict[str, float]:
    if "fit_tau" not in person_parameters:
        return {}
    population = unpack_theta("tripartite", population_theta)
    fitted = person_parameters[["focal_id", "fit_tau"]].merge(
        view.people_truth[
            ["focal_id", "true_tau", "true_noise_s"]
        ],
        on="focal_id",
        how="inner",
    )
    fitted_consistency = np.asarray(
        [
            effective_choice_consistency(tau, population["noise_s"])
            for tau in fitted["fit_tau"]
        ],
        dtype=float,
    )
    true_consistency = np.asarray(
        [
            effective_choice_consistency(tau, noise)
            for tau, noise in zip(
                fitted["true_tau"],
                fitted["true_noise_s"],
            )
        ],
        dtype=float,
    )
    return {
        "choice_consistency_correlation": _safe_correlation(
            true_consistency,
            fitted_consistency,
        ),
        "choice_consistency_rmse": float(
            np.sqrt(np.mean((fitted_consistency - true_consistency) ** 2))
        ),
    }


def _mode_directory(parent: Path, mode: str) -> Path:
    return parent / f"mode={mode}"


def _run_mode(
    mode: str,
    view,
    cfg: EmpiricalBayesConfig,
    population_theta: np.ndarray,
):
    if mode == "decision_weights_oracle":
        person_parameters, recovery, diagnostics = fit_conditional_oracle(
            view,
            "tripartite",
            "decision_weights",
            cfg,
            initial_population_theta=population_theta,
        )
        diagnostics["nuisance_parameters_fixed_to_generating_values"] = 1.0
        return person_parameters, recovery, diagnostics

    block = {
        "full_process": "full_process",
        "decision_weights": "decision_weights",
    }[mode]
    person_parameters, recovery, diagnostics, _ = (
        fit_partial_population_block(
            view,
            "tripartite",
            block,
            cfg,
            population_theta=population_theta,
        )
    )
    return person_parameters, recovery, diagnostics


def _aggregate_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"replicate", "sample_size", "mean_events"}
    ]
    groups = ["mode", "profile", "environment", "mean_events"]
    rows: List[Dict[str, object]] = []
    for keys, subset in frame.groupby(groups, dropna=False):
        row: Dict[str, object] = dict(zip(groups, keys))
        row["datasets"] = int(len(subset))
        for column in numeric:
            row[f"{column}_mean"] = float(subset[column].mean())
            row[f"{column}_min"] = float(subset[column].min())
            row[f"{column}_max"] = float(subset[column].max())
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    source_root = Path(args.source_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    profiles = _csv_strings(args.profiles)
    environments = _csv_strings(args.environments)
    mean_events_values = _csv_ints(args.mean_events)
    modes = _csv_strings(args.modes)
    unknown_profiles = sorted(set(profiles) - set(PROFILE_NAMES))
    unknown_modes = sorted(set(modes) - set(MODES))
    if unknown_profiles:
        raise ValueError(f"Unknown profiles: {unknown_profiles}")
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}")

    manifest = {
        "source_root": str(source_root),
        "profiles": profiles,
        "environments": environments,
        "mean_events": mean_events_values,
        "replicates": args.replicates,
        "sample_size": args.sample_size,
        "missing_rate": args.missing_rate,
        "modes": modes,
        "estimation": {
            "eb_iterations": args.eb_iterations,
            "fit_max_iter": args.fit_max_iter,
            "multistarts": args.multistarts,
            "variance_update": args.variance_update,
            "baseline_reliability": args.baseline_reliability,
        },
    }
    atomic_write_json(outdir / "manifest.json", manifest)

    summary_rows: List[Dict[str, object]] = []
    for profile in profiles:
        for environment in environments:
            for mean_events in mean_events_values:
                for replicate in range(1, args.replicates + 1):
                    latent_dir = _latent_directory(
                        source_root,
                        profile,
                        environment,
                        mean_events,
                        replicate,
                    )
                    if not (latent_dir / "config.json").exists():
                        raise FileNotFoundError(
                            f"Missing production condition: {latent_dir}"
                        )
                    simulation = load_simulation_result(latent_dir)
                    view = build_panel_view(
                        simulation,
                        args.sample_size,
                        args.missing_rate,
                    )
                    cfg = EmpiricalBayesConfig(
                        eb_iterations=args.eb_iterations,
                        max_iter=args.fit_max_iter,
                        multistarts=args.multistarts,
                        variance_update=args.variance_update,
                        baseline_reliability=args.baseline_reliability,
                        observer_penalty=simulation.config.observer_penalty,
                        seed=stable_seed(
                            args.seed,
                            profile,
                            environment,
                            mean_events,
                            replicate,
                        ),
                    )
                    population_theta, population_diagnostics = (
                        fit_population_model(
                            prepare_people(view),
                            "tripartite",
                            cfg,
                        )
                    )
                    condition_dir = (
                        outdir
                        / f"profile={profile}"
                        / f"environment={environment}"
                        / f"events={_event_label(mean_events)}"
                        / f"replicate={replicate:03d}"
                    )
                    for mode in modes:
                        mode_dir = _mode_directory(condition_dir, mode)
                        summary_path = mode_dir / "summary.json"
                        if args.resume and summary_path.exists():
                            summary_rows.append(
                                json.loads(summary_path.read_text())
                            )
                            continue
                        mode_started = time.perf_counter()
                        person_parameters, recovery, diagnostics = _run_mode(
                            mode,
                            view,
                            cfg,
                            population_theta,
                        )
                        summary: Dict[str, object] = {
                            "mode": mode,
                            "profile": profile,
                            "environment": environment,
                            "mean_events": mean_events,
                            "replicate": replicate,
                            "sample_size": args.sample_size,
                            "missing_rate": args.missing_rate,
                            "runtime_seconds": (
                                time.perf_counter() - mode_started
                            ),
                            **_weight_summary(
                                view,
                                person_parameters,
                                population_theta,
                            ),
                            **(
                                _choice_consistency_summary(
                                    view,
                                    person_parameters,
                                    population_theta,
                                )
                                if mode == "full_process"
                                else {}
                            ),
                            **{
                                f"diagnostic_{name}": value
                                for name, value in diagnostics.items()
                            },
                            **{
                                f"population_{name}": value
                                for name, value in population_diagnostics.items()
                            },
                        }
                        mode_dir.mkdir(parents=True, exist_ok=True)
                        atomic_write_csv(
                            person_parameters,
                            mode_dir / "person_parameters.csv.gz",
                        )
                        atomic_write_csv(
                            recovery,
                            mode_dir / "recovery.csv",
                        )
                        atomic_write_json(summary_path, summary)
                        summary_rows.append(summary)
                        atomic_write_csv(
                            pd.DataFrame(summary_rows),
                            outdir / "dataset_summary.csv",
                        )
                        atomic_write_csv(
                            _aggregate_summaries(
                                pd.DataFrame(summary_rows)
                            ),
                            outdir / "aggregate_summary.csv",
                        )
                        print(
                            (
                                f"{profile}/{environment}/{mean_events}/"
                                f"{replicate}: {mode} complete"
                            ),
                            flush=True,
                        )

    atomic_write_json(
        outdir / "run_status.json",
        {
            "status": "complete",
            "dataset_mode_count": len(summary_rows),
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate participant-specific and parameter-block recovery "
            "using the setup-free PDTRT production simulations."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--profiles", default=",".join(PROFILE_NAMES))
    parser.add_argument("--environments", default="mixed")
    parser.add_argument("--mean-events", default="10,25,50")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--eb-iterations", type=int, default=3)
    parser.add_argument("--fit-max-iter", type=int, default=120)
    parser.add_argument("--multistarts", type=int, default=2)
    parser.add_argument(
        "--variance-update",
        choices=("laplace", "modal"),
        default="laplace",
    )
    parser.add_argument("--baseline-reliability", type=float, default=0.70)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
