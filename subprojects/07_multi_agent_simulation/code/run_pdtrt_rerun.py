from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import product
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn

from pdtrt_rerun_core import (
    ENJOYMENT_MEANS,
    ENJOYMENT_SDS,
    ENVIRONMENT_SPECS,
    ENVIRONMENT_NAMES,
    GENERATOR_MODELS,
    PROFILE_SPECS,
    PROFILE_NAMES,
    PDTRTRerunSimulator,
    RerunConfig,
    SimulationResult,
    atomic_write_csv,
    atomic_write_json,
    build_panel_view,
    config_fingerprint,
    load_simulation_result,
    simulator_seed_manifest,
    stable_seed,
    UTILITY_MEANS,
    UTILITY_SDS,
    validate_nested_views,
    write_panel_view,
    write_simulation_result,
)
from pdtrt_rerun_fit import (
    CANDIDATE_MODELS,
    EmpiricalBayesConfig,
    ModelFitResult,
    assert_prediction_targets_match,
    evaluate_predictions,
    fit_and_evaluate_model,
    flatten_result,
    null_predictions,
    prepare_people,
)


def _csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> List[float]:
    return [float(item) for item in _csv_strings(value)]


def _event_label(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):03d}"
    return str(value).replace(".", "p")


def _missing_label(value: float) -> str:
    return f"{int(round(100.0 * value)):02d}"


def _validate_levels(
    profiles: Sequence[str],
    environments: Sequence[str],
    generator_model: str,
    sample_sizes: Sequence[int],
    event_means: Sequence[float],
    missing_rates: Sequence[float],
    models: Sequence[str],
) -> None:
    unknown_profiles = sorted(set(profiles) - set(PROFILE_NAMES))
    unknown_environments = sorted(set(environments) - set(ENVIRONMENT_NAMES))
    unknown_models = sorted(set(models) - set(CANDIDATE_MODELS))
    if unknown_profiles:
        raise ValueError(f"Unknown personality-process configurations: {unknown_profiles}")
    if unknown_environments:
        raise ValueError(f"Unknown social-environment profiles: {unknown_environments}")
    if generator_model not in GENERATOR_MODELS:
        raise ValueError(f"Unknown generator model: {generator_model}")
    if unknown_models:
        raise ValueError(f"Unknown candidate models: {unknown_models}")
    if not sample_sizes or min(sample_sizes) < 2:
        raise ValueError("Sample sizes must contain integers of at least 2")
    if sample_sizes != sorted(set(sample_sizes)):
        raise ValueError("Sample sizes must be unique and sorted from smallest to largest")
    if not event_means or min(event_means) <= 0:
        raise ValueError("Mean event yields must be positive")
    if sorted(set(missing_rates)) != list(missing_rates):
        raise ValueError("Missingness rates must be unique and sorted")
    invalid_missing = [rate for rate in missing_rates if rate not in (0.0, 0.1, 0.2)]
    if invalid_missing:
        raise ValueError(f"Unsupported missingness rates: {invalid_missing}")


def _parent_directory(
    outdir: Path,
    generator_model: str,
    profile: str,
    environment: str,
    event_mean: float,
    replicate: int,
) -> Path:
    return (
        outdir
        / "conditions"
        / f"generator={generator_model}"
        / f"profile={profile}"
        / f"environment={environment}"
        / f"events={_event_label(event_mean)}"
        / f"replicate={replicate:03d}"
    )


def _view_directory(parent: Path, sample_size: int, missing_rate: float) -> Path:
    return (
        parent
        / "views"
        / f"N={sample_size:03d}"
        / f"missing={_missing_label(missing_rate)}"
    )


def _fit_directory(view_dir: Path, estimator: str, model: str) -> Path:
    return (
        view_dir
        / "fits"
        / f"estimator={estimator}"
        / f"model={model}"
    )


def _load_or_generate(
    config: RerunConfig,
    latent_dir: Path,
    resume: bool,
) -> tuple[SimulationResult, float, bool]:
    config_path = latent_dir / "config.json"
    if resume and config_path.exists():
        prior = RerunConfig(**json.loads(config_path.read_text()))
        expected_fingerprint = config_fingerprint(config)
        diagnostics_path = latent_dir / "generation_diagnostics.json"
        stored_fingerprint = None
        if diagnostics_path.exists():
            stored_fingerprint = json.loads(
                diagnostics_path.read_text()
            ).get("config_fingerprint")
        if (
            config_fingerprint(prior) != expected_fingerprint
            or stored_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                f"Existing latent condition has a different configuration: {latent_dir}"
            )
        _verify_checksum_manifest(latent_dir)
        runtime_path = latent_dir / "generation_runtime.json"
        recorded_runtime = (
            float(json.loads(runtime_path.read_text())["runtime_seconds"])
            if runtime_path.exists()
            else np.nan
        )
        return load_simulation_result(latent_dir), recorded_runtime, False
    started = time.perf_counter()
    result = PDTRTRerunSimulator(config).run()
    write_simulation_result(result, latent_dir)
    elapsed = time.perf_counter() - started
    atomic_write_json(
        latent_dir / "generation_runtime.json",
        {"runtime_seconds": elapsed},
    )
    _write_checksum_manifest(latent_dir)
    return result, elapsed, True


def _file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum_manifest(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "checksums.json"
    )
    atomic_write_json(
        directory / "checksums.json",
        {
            path.name: {
                "sha256": _file_checksum(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    )


def _verify_checksum_manifest(directory: Path) -> None:
    manifest_path = directory / "checksums.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Checksummed output is missing: {path}")
        observed = _file_checksum(path)
        if observed != expected["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")


def _fit_summary_payload(
    result: ModelFitResult,
    condition: Mapping[str, object],
) -> Dict[str, object]:
    return {
        **condition,
        "model": result.model,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "population_parameters": result.population_parameters,
    }


def _write_fit_result(
    result: ModelFitResult,
    fit_dir: Path,
    condition: Mapping[str, object],
) -> None:
    fit_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(result.predictions, fit_dir / "predictions.csv.gz")
    if not result.person_parameters.empty:
        atomic_write_csv(result.person_parameters, fit_dir / "person_parameters.csv.gz")
    if not result.recovery.empty:
        atomic_write_csv(result.recovery, fit_dir / "recovery.csv")
    atomic_write_json(
        fit_dir / "fit_summary.json",
        _fit_summary_payload(result, condition),
    )
    _write_checksum_manifest(fit_dir)


def _load_fit_result(fit_dir: Path) -> ModelFitResult:
    _verify_checksum_manifest(fit_dir)
    payload = json.loads((fit_dir / "fit_summary.json").read_text())
    person_path = fit_dir / "person_parameters.csv.gz"
    recovery_path = fit_dir / "recovery.csv"
    return ModelFitResult(
        model=str(payload["model"]),
        metrics={key: float(value) for key, value in payload["metrics"].items()},
        diagnostics={
            key: float(value) for key, value in payload["diagnostics"].items()
        },
        predictions=pd.read_csv(fit_dir / "predictions.csv.gz"),
        person_parameters=(
            pd.read_csv(person_path) if person_path.exists() else pd.DataFrame()
        ),
        recovery=pd.read_csv(recovery_path) if recovery_path.exists() else pd.DataFrame(),
        population_parameters={
            key: float(value)
            for key, value in payload["population_parameters"].items()
        },
    )


def _null_result(view) -> ModelFitResult:
    predictions = null_predictions(prepare_people(view))
    if not predictions.empty:
        predictions["model"] = "prevalence_null"
    prevalence = (
        float(predictions["probability"].iloc[0]) if not predictions.empty else np.nan
    )
    return ModelFitResult(
        model="prevalence_null",
        metrics=evaluate_predictions(predictions),
        diagnostics={"optimizer_success_rate": 1.0},
        predictions=predictions,
        person_parameters=pd.DataFrame(),
        recovery=pd.DataFrame(),
        population_parameters={"training_approach_rate": prevalence},
    )


def _condition_metadata(
    config: RerunConfig,
    sample_size: int,
    missing_rate: float,
    replicate: int,
    estimator: str,
) -> Dict[str, object]:
    return {
        "generator_model": config.generator_model,
        "profile": config.profile,
        "environment": config.environment,
        "mean_events": config.mean_events,
        "sample_size": sample_size,
        "missing_rate": missing_rate,
        "replicate": replicate,
        "estimator": estimator,
        "seed": config.seed,
        "config_fingerprint": config_fingerprint(config),
    }


def _summary_row(
    result: ModelFitResult,
    condition: Mapping[str, object],
) -> Dict[str, object]:
    return {
        **condition,
        **flatten_result(
            result,
            sample_size=int(condition["sample_size"]),
            missing_rate=float(condition["missing_rate"]),
        ),
    }


def _write_error(
    errors: List[Dict[str, object]],
    outdir: Path,
    condition: Mapping[str, object],
    exc: BaseException,
) -> None:
    row = {
        **condition,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    errors.append(row)
    atomic_write_json(
        outdir / "errors" / f"error_{len(errors):05d}.json",
        row,
    )


def _generate_parent_task(task: Mapping[str, object]) -> Dict[str, object]:
    config = RerunConfig(**dict(task["config"]))
    parent = Path(str(task["parent"]))
    latent_dir = parent / "latent"
    replicate = int(task["replicate"])
    estimator = str(task["estimator"])
    sample_sizes = [int(value) for value in task["sample_sizes"]]
    missing_rates = [float(value) for value in task["missing_rates"]]

    simulation, generation_runtime, generated_now = _load_or_generate(
        config,
        latent_dir,
        resume=bool(task["resume"]),
    )
    atomic_write_json(
        parent / "seed_manifest.json",
        {
            "replicate_seed": config.seed,
            "streams": simulator_seed_manifest(config.seed),
        },
    )
    views = [
        build_panel_view(simulation, sample_size, missing_rate)
        for sample_size in sample_sizes
        for missing_rate in missing_rates
    ]
    nested_checks = validate_nested_views(views)
    if nested_checks and not all(nested_checks.values()):
        failed = [name for name, passed in nested_checks.items() if not passed]
        raise AssertionError(f"Nested-panel checks failed: {failed}")
    atomic_write_json(parent / "nested_panel_checks.json", nested_checks)

    inventory_rows: List[Dict[str, object]] = []
    for view in views:
        view_dir = _view_directory(
            parent,
            view.sample_size,
            view.missing_rate,
        )
        if bool(task["write_panel_data"]):
            write_panel_view(view, view_dir / "data")
        condition = _condition_metadata(
            config,
            view.sample_size,
            view.missing_rate,
            replicate,
            estimator,
        )
        inventory_rows.append(
            {
                **condition,
                **{
                    f"panel_{key}": value
                    for key, value in view.diagnostics.items()
                },
                **{
                    f"generation_{key}": value
                    for key, value in simulation.generation_diagnostics.items()
                },
                **{
                    f"network_{key}": value
                    for key, value in simulation.network_diagnostics.items()
                },
                "generation_runtime_seconds": generation_runtime,
                "generated_in_current_run": int(generated_now),
            }
        )
    return {
        "parent_meta": dict(task["parent_meta"]),
        "inventory_rows": inventory_rows,
        "generation_runtime_seconds": generation_runtime,
        "generated_in_current_run": int(generated_now),
    }


def _run_generation_stage(
    args: argparse.Namespace,
    outdir: Path,
    profiles: Sequence[str],
    environments: Sequence[str],
    event_means: Sequence[float],
    sample_sizes: Sequence[int],
    missing_rates: Sequence[float],
    max_focal: int,
    run_started: float,
) -> int:
    tasks: List[Dict[str, object]] = []
    combinations = product(
        profiles,
        environments,
        event_means,
        range(1, args.reps + 1),
    )
    for profile, environment, event_mean, replicate in combinations:
        if (
            args.max_parent_conditions is not None
            and len(tasks) >= args.max_parent_conditions
        ):
            break
        seed = stable_seed(args.seed, "replicate", replicate)
        config = RerunConfig(
            profile=profile,
            environment=environment,
            generator_model=args.generator_model,
            seed=seed,
            days=args.days,
            network_size=args.network_size,
            max_focal=max_focal,
            mean_events=event_mean,
            event_dispersion=args.event_dispersion,
            mean_degree=args.mean_degree,
            homophily_scale=args.homophily_scale,
            n_social_foci=args.social_foci,
            hidden_events_per_person_day=args.hidden_events_per_person_day,
            baseline_report_sd=args.baseline_report_sd,
            outcome_report_sd=args.outcome_report_sd,
            relationship_report_sd=args.relationship_report_sd,
            suggestion_report_sd=args.suggestion_report_sd,
            feedback_report_sd=args.feedback_report_sd,
            pilot_version=args.pilot_version,
        )
        parent = _parent_directory(
            outdir,
            args.generator_model,
            profile,
            environment,
            event_mean,
            replicate,
        )
        tasks.append(
            {
                "config": dict(config.__dict__),
                "parent": str(parent),
                "replicate": replicate,
                "estimator": args.estimator,
                "sample_sizes": list(sample_sizes),
                "missing_rates": list(missing_rates),
                "resume": args.resume,
                "write_panel_data": args.write_panel_data,
                "parent_meta": {
                    "generator_model": args.generator_model,
                    "profile": profile,
                    "environment": environment,
                    "mean_events": event_mean,
                    "replicate": replicate,
                    "seed": seed,
                },
            }
        )

    inventory_rows: List[Dict[str, object]] = []
    generation_runtimes: List[float] = []
    errors: List[Dict[str, object]] = []
    completed = 0

    def record(result: Mapping[str, object]) -> None:
        nonlocal completed
        completed += 1
        inventory_rows.extend(result["inventory_rows"])
        generation_runtimes.append(
            float(result["generation_runtime_seconds"])
        )
        if completed % 10 == 0 or completed == len(tasks):
            ordered = pd.DataFrame(inventory_rows).sort_values(
                [
                    "profile",
                    "environment",
                    "mean_events",
                    "replicate",
                    "sample_size",
                    "missing_rate",
                ]
            )
            atomic_write_csv(
                ordered,
                outdir / "condition_inventory.csv",
            )
            print(
                f"simulation parents complete: {completed}/{len(tasks)}",
                flush=True,
            )

    if args.workers == 1:
        for task in tasks:
            try:
                record(_generate_parent_task(task))
            except Exception as exc:
                _write_error(errors, outdir, task["parent_meta"], exc)
                if not args.continue_on_error:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_tasks = {
                executor.submit(_generate_parent_task, task): task
                for task in tasks
            }
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    record(future.result())
                except Exception as exc:
                    _write_error(errors, outdir, task["parent_meta"], exc)
                    if not args.continue_on_error:
                        for pending in future_tasks:
                            pending.cancel()
                        raise

    status = {
        "status": (
            "simulation_complete"
            if not errors and completed == len(tasks)
            else "simulation_complete_with_errors"
        ),
        "stage": "generate",
        "workers": args.workers,
        "parent_conditions_expected": len(tasks),
        "parent_conditions_processed": completed,
        "panel_count": len(inventory_rows),
        "analysis_count_including_null": 0,
        "error_count": len(errors),
        "runtime_seconds_current_invocation": time.perf_counter() - run_started,
        "recorded_generation_runtime_seconds": float(
            np.nansum(generation_runtimes)
        ),
        "recorded_model_fit_runtime_seconds": 0.0,
    }
    atomic_write_json(outdir / "simulation_status.json", status)
    atomic_write_json(outdir / "run_status.json", status)
    _write_checksum_manifest(outdir)
    return 0 if not errors and completed == len(tasks) else 1


def _fit_parent_task(task: Mapping[str, object]) -> Dict[str, object]:
    config = RerunConfig(**dict(task["config"]))
    parent = Path(str(task["parent"]))
    latent_dir = parent / "latent"
    replicate = int(task["replicate"])
    estimator = str(task["estimator"])
    models = [str(value) for value in task["models"]]
    sample_sizes = [int(value) for value in task["sample_sizes"]]
    missing_rates = [float(value) for value in task["missing_rates"]]
    base_seed = int(task["base_seed"])
    continue_on_error = bool(task["continue_on_error"])

    if not (latent_dir / "config.json").exists():
        raise FileNotFoundError(f"Simulation stage is incomplete: {latent_dir}")
    simulation, _, _ = _load_or_generate(
        config,
        latent_dir,
        resume=True,
    )
    views = [
        build_panel_view(simulation, sample_size, missing_rate)
        for sample_size in sample_sizes
        for missing_rate in missing_rates
    ]
    nested_checks = validate_nested_views(views)
    if nested_checks and not all(nested_checks.values()):
        failed = [name for name, passed in nested_checks.items() if not passed]
        raise AssertionError(f"Nested-panel checks failed: {failed}")

    summary_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    for view in views:
        view_dir = _view_directory(
            parent,
            view.sample_size,
            view.missing_rate,
        )
        condition = _condition_metadata(
            config,
            view.sample_size,
            view.missing_rate,
            replicate,
            estimator,
        )
        fit_cfg = EmpiricalBayesConfig(
            eb_iterations=int(task["eb_iterations"]),
            max_iter=int(task["fit_max_iter"]),
            multistarts=int(task["multistarts"]),
            variance_update=str(task["eb_variance_update"]),
            baseline_reliability=float(task["baseline_reliability"]),
            seed=stable_seed(
                base_seed,
                "fit",
                config.profile,
                config.environment,
                config.mean_events,
                replicate,
                view.sample_size,
                view.missing_rate,
            ),
        )
        panel_results: List[ModelFitResult] = []
        for model in models:
            fit_dir = _fit_directory(view_dir, estimator, model)
            try:
                if bool(task["resume"]) and (
                    fit_dir / "fit_summary.json"
                ).exists():
                    result = _load_fit_result(fit_dir)
                else:
                    fit_started = time.perf_counter()
                    result = fit_and_evaluate_model(
                        view,
                        model,
                        fit_cfg,
                        estimator=estimator,
                    )
                    result.diagnostics["runtime_seconds"] = (
                        time.perf_counter() - fit_started
                    )
                    _write_fit_result(result, fit_dir, condition)
                panel_results.append(result)
                summary_rows.append(_summary_row(result, condition))
            except Exception as exc:
                error_rows.append(
                    {
                        **condition,
                        "model": model,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if not continue_on_error:
                    raise

        null_dir = _fit_directory(
            view_dir,
            estimator,
            "prevalence_null",
        )
        try:
            if bool(task["resume"]) and (
                null_dir / "fit_summary.json"
            ).exists():
                null_result = _load_fit_result(null_dir)
            else:
                null_started = time.perf_counter()
                null_result = _null_result(view)
                null_result.diagnostics["runtime_seconds"] = (
                    time.perf_counter() - null_started
                )
                _write_fit_result(null_result, null_dir, condition)
            summary_rows.append(_summary_row(null_result, condition))
            if panel_results:
                assert_prediction_targets_match(
                    [*panel_results, null_result]
                )
        except Exception as exc:
            error_rows.append(
                {
                    **condition,
                    "model": "prevalence_null_or_target_alignment",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if not continue_on_error:
                raise

    return {
        "parent_meta": dict(task["parent_meta"]),
        "summary_rows": summary_rows,
        "error_rows": error_rows,
    }


def _run_fit_stage(
    args: argparse.Namespace,
    outdir: Path,
    profiles: Sequence[str],
    environments: Sequence[str],
    event_means: Sequence[float],
    sample_sizes: Sequence[int],
    missing_rates: Sequence[float],
    max_focal: int,
    run_started: float,
) -> int:
    simulation_status_path = outdir / "simulation_status.json"
    if not simulation_status_path.exists():
        raise FileNotFoundError(
            "The official simulation stage has not completed"
        )
    simulation_status = json.loads(simulation_status_path.read_text())
    if simulation_status.get("status") != "simulation_complete":
        raise RuntimeError(
            "Fitting requires a complete, error-free simulation stage"
        )

    tasks: List[Dict[str, object]] = []
    combinations = product(
        profiles,
        environments,
        event_means,
        range(1, args.reps + 1),
    )
    for profile, environment, event_mean, replicate in combinations:
        if (
            args.max_parent_conditions is not None
            and len(tasks) >= args.max_parent_conditions
        ):
            break
        seed = stable_seed(args.seed, "replicate", replicate)
        config = RerunConfig(
            profile=profile,
            environment=environment,
            generator_model=args.generator_model,
            seed=seed,
            days=args.days,
            network_size=args.network_size,
            max_focal=max_focal,
            mean_events=event_mean,
            event_dispersion=args.event_dispersion,
            mean_degree=args.mean_degree,
            homophily_scale=args.homophily_scale,
            n_social_foci=args.social_foci,
            hidden_events_per_person_day=args.hidden_events_per_person_day,
            baseline_report_sd=args.baseline_report_sd,
            outcome_report_sd=args.outcome_report_sd,
            relationship_report_sd=args.relationship_report_sd,
            suggestion_report_sd=args.suggestion_report_sd,
            feedback_report_sd=args.feedback_report_sd,
            pilot_version=args.pilot_version,
        )
        parent = _parent_directory(
            outdir,
            args.generator_model,
            profile,
            environment,
            event_mean,
            replicate,
        )
        tasks.append(
            {
                "config": dict(config.__dict__),
                "parent": str(parent),
                "replicate": replicate,
                "estimator": args.estimator,
                "models": list(_csv_strings(args.models)),
                "sample_sizes": list(sample_sizes),
                "missing_rates": list(missing_rates),
                "base_seed": args.seed,
                "eb_iterations": args.eb_iterations,
                "fit_max_iter": args.fit_max_iter,
                "multistarts": args.multistarts,
                "eb_variance_update": args.eb_variance_update,
                "baseline_reliability": args.baseline_reliability,
                "resume": args.resume,
                "continue_on_error": args.continue_on_error,
                "parent_meta": {
                    "generator_model": args.generator_model,
                    "profile": profile,
                    "environment": environment,
                    "mean_events": event_mean,
                    "replicate": replicate,
                    "seed": seed,
                },
            }
        )

    expected_parents = int(
        simulation_status.get("parent_conditions_expected", -1)
    )
    if (
        args.max_parent_conditions is None
        and expected_parents != len(tasks)
    ):
        raise RuntimeError(
            "The simulation inventory does not match the requested fit design"
        )

    summary_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    completed = 0

    def record(result: Mapping[str, object]) -> None:
        nonlocal completed
        completed += 1
        summary_rows.extend(result["summary_rows"])
        for row in result["error_rows"]:
            errors.append(dict(row))
            atomic_write_json(
                outdir / "errors" / f"error_{len(errors):05d}.json",
                row,
            )
        if completed % 10 == 0 or completed == len(tasks):
            ordered = pd.DataFrame(summary_rows).sort_values(
                [
                    "profile",
                    "environment",
                    "mean_events",
                    "replicate",
                    "sample_size",
                    "missing_rate",
                    "model",
                ]
            )
            atomic_write_csv(ordered, outdir / "run_summary.csv")
            print(
                f"fit parents complete: {completed}/{len(tasks)}",
                flush=True,
            )

    if args.workers == 1:
        for task in tasks:
            try:
                record(_fit_parent_task(task))
            except Exception as exc:
                _write_error(errors, outdir, task["parent_meta"], exc)
                if not args.continue_on_error:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_tasks = {
                executor.submit(_fit_parent_task, task): task
                for task in tasks
            }
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    record(future.result())
                except Exception as exc:
                    _write_error(errors, outdir, task["parent_meta"], exc)
                    if not args.continue_on_error:
                        for pending in future_tasks:
                            pending.cancel()
                        raise

    status = {
        "status": (
            "complete"
            if not errors and completed == len(tasks)
            else "complete_with_errors"
        ),
        "stage": "fit",
        "workers": args.workers,
        "parent_conditions_expected": len(tasks),
        "parent_conditions_processed": completed,
        "panel_count": completed * len(sample_sizes) * len(missing_rates),
        "analysis_count_including_null": len(summary_rows),
        "error_count": len(errors),
        "runtime_seconds_current_invocation": time.perf_counter() - run_started,
        "recorded_generation_runtime_seconds": float(
            simulation_status["recorded_generation_runtime_seconds"]
        ),
        "recorded_model_fit_runtime_seconds": float(
            np.nansum(
                [
                    row.get("diagnostic_runtime_seconds", np.nan)
                    for row in summary_rows
                ]
            )
        ),
    }
    atomic_write_json(outdir / "fit_status.json", status)
    atomic_write_json(outdir / "run_status.json", status)
    _write_checksum_manifest(outdir)
    return 0 if not errors and completed == len(tasks) else 1


def run(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    profiles = _csv_strings(args.profiles)
    environments = _csv_strings(args.environments)
    event_means = _csv_floats(args.event_means)
    sample_sizes = _csv_ints(args.sample_sizes)
    missing_rates = _csv_floats(args.missing_rates)
    models = _csv_strings(args.models)
    _validate_levels(
        profiles,
        environments,
        args.generator_model,
        sample_sizes,
        event_means,
        missing_rates,
        models,
    )

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    max_focal = max(sample_sizes)
    if args.max_focal is not None:
        max_focal = max(max_focal, args.max_focal)
    if args.network_size < max_focal:
        raise ValueError("network-size must be at least the largest focal panel")

    manifest = {
        "profiles": profiles,
        "environments": environments,
        "generator_model": args.generator_model,
        "event_means": event_means,
        "sample_sizes": sample_sizes,
        "missing_rates": missing_rates,
        "models": models,
        "estimator": args.estimator,
        "replicates": args.reps,
        "base_seed": args.seed,
        "network_size": args.network_size,
        "max_focal": max_focal,
        "days": args.days,
        "stage": args.stage,
        "workers": args.workers,
        "resume": args.resume,
        "write_panel_data": args.write_panel_data,
        "measurement": {
            "baseline_report_sd": args.baseline_report_sd,
            "outcome_report_sd": args.outcome_report_sd,
            "relationship_report_sd": args.relationship_report_sd,
            "suggestion_report_sd": args.suggestion_report_sd,
            "feedback_report_sd": args.feedback_report_sd,
            "baseline_reliability": args.baseline_reliability,
        },
        "empirical_bayes": {
            "iterations": args.eb_iterations,
            "max_iter": args.fit_max_iter,
            "multistarts": args.multistarts,
            "variance_update": args.eb_variance_update,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    atomic_write_json(outdir / "run_manifest.json", manifest)
    atomic_write_json(outdir / f"{args.stage}_run_manifest.json", manifest)
    atomic_write_json(
        outdir / "pilot_design_constants.json",
        {
            "status": (
                "frozen"
                if str(args.pilot_version).startswith("pdtrt_production")
                else "pilot_not_frozen"
            ),
            "personality_process_configurations": PROFILE_SPECS,
            "social_environment_profiles": ENVIRONMENT_SPECS,
            "enjoyment_means": ENJOYMENT_MEANS,
            "enjoyment_sds": ENJOYMENT_SDS,
            "utility_means": UTILITY_MEANS,
            "utility_sds": UTILITY_SDS,
        },
    )

    if args.stage == "generate":
        return _run_generation_stage(
            args,
            outdir,
            profiles,
            environments,
            event_means,
            sample_sizes,
            missing_rates,
            max_focal,
            run_started,
        )
    if args.stage == "fit":
        return _run_fit_stage(
            args,
            outdir,
            profiles,
            environments,
            event_means,
            sample_sizes,
            missing_rates,
            max_focal,
            run_started,
        )
    if args.workers != 1:
        raise ValueError("The interleaved all stage supports one worker")

    summary_rows: List[Dict[str, object]] = []
    inventory_rows: List[Dict[str, object]] = []
    generation_runtimes: List[float] = []
    errors: List[Dict[str, object]] = []
    parent_count = 0

    for profile in profiles:
        for environment in environments:
            for event_mean in event_means:
                for replicate in range(1, args.reps + 1):
                    if args.max_parent_conditions and parent_count >= args.max_parent_conditions:
                        break
                    parent_count += 1
                    parent = _parent_directory(
                        outdir,
                        args.generator_model,
                        profile,
                        environment,
                        event_mean,
                        replicate,
                    )
                    latent_dir = parent / "latent"
                    # A replicate uses one paired social world across all crossed
                    # factors. Separate simulator streams keep each manipulation
                    # from shifting unrelated draws.
                    seed = stable_seed(args.seed, "replicate", replicate)
                    config = RerunConfig(
                        profile=profile,
                        environment=environment,
                        generator_model=args.generator_model,
                        seed=seed,
                        days=args.days,
                        network_size=args.network_size,
                        max_focal=max_focal,
                        mean_events=event_mean,
                        event_dispersion=args.event_dispersion,
                        mean_degree=args.mean_degree,
                        homophily_scale=args.homophily_scale,
                        n_social_foci=args.social_foci,
                        hidden_events_per_person_day=args.hidden_events_per_person_day,
                        baseline_report_sd=args.baseline_report_sd,
                        outcome_report_sd=args.outcome_report_sd,
                        relationship_report_sd=args.relationship_report_sd,
                        suggestion_report_sd=args.suggestion_report_sd,
                        feedback_report_sd=args.feedback_report_sd,
                        pilot_version=args.pilot_version,
                    )
                    parent_meta = {
                        "generator_model": args.generator_model,
                        "profile": profile,
                        "environment": environment,
                        "mean_events": event_mean,
                        "replicate": replicate,
                        "seed": seed,
                    }
                    try:
                        if (
                            args.stage == "fit"
                            and not (latent_dir / "config.json").exists()
                        ):
                            raise FileNotFoundError(
                                f"Simulation stage is incomplete: {latent_dir}"
                            )
                        simulation, generation_runtime, generated_now = _load_or_generate(
                            config,
                            latent_dir,
                            resume=args.resume,
                        )
                        generation_runtimes.append(generation_runtime)
                        atomic_write_json(
                            parent / "seed_manifest.json",
                            {
                                "replicate_seed": seed,
                                "streams": simulator_seed_manifest(seed),
                            },
                        )
                        views = [
                            build_panel_view(simulation, sample_size, missing_rate)
                            for sample_size in sample_sizes
                            for missing_rate in missing_rates
                        ]
                        nested_checks = validate_nested_views(views)
                        if nested_checks and not all(nested_checks.values()):
                            failed = [
                                name for name, passed in nested_checks.items() if not passed
                            ]
                            raise AssertionError(f"Nested-panel checks failed: {failed}")
                        atomic_write_json(parent / "nested_panel_checks.json", nested_checks)
                    except Exception as exc:
                        _write_error(errors, outdir, parent_meta, exc)
                        if not args.continue_on_error:
                            raise
                        continue

                    for view in views:
                        view_dir = _view_directory(
                            parent,
                            view.sample_size,
                            view.missing_rate,
                        )
                        if args.write_panel_data:
                            write_panel_view(view, view_dir / "data")
                        condition = _condition_metadata(
                            config,
                            view.sample_size,
                            view.missing_rate,
                            replicate,
                            args.estimator,
                        )
                        inventory_rows.append(
                            {
                                **condition,
                                **{
                                    f"panel_{key}": value
                                    for key, value in view.diagnostics.items()
                                },
                                **{
                                    f"generation_{key}": value
                                    for key, value in simulation.generation_diagnostics.items()
                                },
                                **{
                                    f"network_{key}": value
                                    for key, value in simulation.network_diagnostics.items()
                                },
                                "generation_runtime_seconds": generation_runtime,
                                "generated_in_current_run": int(generated_now),
                            }
                        )
                        panel_results: List[ModelFitResult] = []
                        fit_cfg = EmpiricalBayesConfig(
                            eb_iterations=args.eb_iterations,
                            max_iter=args.fit_max_iter,
                            multistarts=args.multistarts,
                            variance_update=args.eb_variance_update,
                            baseline_reliability=args.baseline_reliability,
                            seed=stable_seed(
                                args.seed,
                                "fit",
                                profile,
                                environment,
                                event_mean,
                                replicate,
                                view.sample_size,
                                view.missing_rate,
                            ),
                        )
                        for model in models:
                            fit_dir = _fit_directory(
                                view_dir,
                                args.estimator,
                                model,
                            )
                            try:
                                if args.resume and (fit_dir / "fit_summary.json").exists():
                                    result = _load_fit_result(fit_dir)
                                else:
                                    fit_started = time.perf_counter()
                                    result = fit_and_evaluate_model(
                                        view,
                                        model,
                                        fit_cfg,
                                        estimator=args.estimator,
                                    )
                                    result.diagnostics["runtime_seconds"] = (
                                        time.perf_counter() - fit_started
                                    )
                                    _write_fit_result(result, fit_dir, condition)
                                panel_results.append(result)
                                summary_rows.append(_summary_row(result, condition))
                            except Exception as exc:
                                _write_error(
                                    errors,
                                    outdir,
                                    {**condition, "model": model},
                                    exc,
                                )
                                if not args.continue_on_error:
                                    raise

                        null_dir = _fit_directory(
                            view_dir,
                            args.estimator,
                            "prevalence_null",
                        )
                        null_started = time.perf_counter()
                        null_result = _null_result(view)
                        null_result.diagnostics["runtime_seconds"] = (
                            time.perf_counter() - null_started
                        )
                        _write_fit_result(null_result, null_dir, condition)
                        summary_rows.append(_summary_row(null_result, condition))
                        if panel_results:
                            assert_prediction_targets_match(
                                [*panel_results, null_result]
                            )

                    if summary_rows:
                        atomic_write_csv(
                            pd.DataFrame(summary_rows),
                            outdir / "run_summary.csv",
                        )
                    if inventory_rows:
                        atomic_write_csv(
                            pd.DataFrame(inventory_rows),
                            outdir / "condition_inventory.csv",
                        )
                if args.max_parent_conditions and parent_count >= args.max_parent_conditions:
                    break
            if args.max_parent_conditions and parent_count >= args.max_parent_conditions:
                break
        if args.max_parent_conditions and parent_count >= args.max_parent_conditions:
            break

    atomic_write_json(
        outdir / "run_status.json",
        {
            "status": "complete" if not errors else "complete_with_errors",
            "parent_conditions_processed": parent_count,
            "panel_count": len(inventory_rows),
            "analysis_count_including_null": len(summary_rows),
            "error_count": len(errors),
            "runtime_seconds_current_invocation": time.perf_counter()
            - run_started,
            "recorded_generation_runtime_seconds": float(
                np.nansum(generation_runtimes)
            ),
            "recorded_model_fit_runtime_seconds": float(
                np.nansum(
                    [
                        row.get("diagnostic_runtime_seconds", np.nan)
                        for row in summary_rows
                    ]
                )
            ),
        },
    )
    _write_checksum_manifest(outdir)
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paired PDTRT AA simulation and model-comparison design."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "fit"),
        default="all",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--profiles", default=",".join(PROFILE_NAMES))
    parser.add_argument("--environments", default=",".join(ENVIRONMENT_NAMES))
    parser.add_argument("--generator-model", default="tripartite")
    parser.add_argument("--event-means", default="10,25,50")
    parser.add_argument("--sample-sizes", default="30,60,100")
    parser.add_argument("--missing-rates", default="0,0.1,0.2")
    parser.add_argument("--models", default=",".join(CANDIDATE_MODELS))
    parser.add_argument(
        "--estimator",
        choices=("empirical_bayes", "population"),
        default="empirical_bayes",
    )
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--network-size", type=int, default=1000)
    parser.add_argument("--max-focal", type=int)
    parser.add_argument("--mean-degree", type=float, default=12.0)
    parser.add_argument("--homophily-scale", type=float, default=0.55)
    parser.add_argument("--social-foci", type=int, default=12)
    parser.add_argument("--hidden-events-per-person-day", type=float, default=0.10)
    parser.add_argument("--event-dispersion", type=float, default=3.0)
    parser.add_argument("--baseline-report-sd", type=float, default=0.20)
    parser.add_argument("--outcome-report-sd", type=float, default=0.12)
    parser.add_argument("--relationship-report-sd", type=float, default=0.10)
    parser.add_argument("--suggestion-report-sd", type=float, default=0.08)
    parser.add_argument("--feedback-report-sd", type=float, default=0.08)
    parser.add_argument("--baseline-reliability", type=float, default=0.70)
    parser.add_argument("--eb-iterations", type=int, default=3)
    parser.add_argument(
        "--eb-variance-update",
        choices=("laplace", "modal"),
        default="laplace",
    )
    parser.add_argument("--fit-max-iter", type=int, default=120)
    parser.add_argument("--multistarts", type=int, default=2)
    parser.add_argument("--pilot-version", default="pilot_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-panel-data", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-parent-conditions", type=int)
    return parser


if __name__ == "__main__":
    try:
        sys.exit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        sys.exit(130)
