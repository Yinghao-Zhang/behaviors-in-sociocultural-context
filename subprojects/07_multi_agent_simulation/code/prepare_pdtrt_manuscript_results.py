from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pdtrt_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdtrt_rerun_core import atomic_write_csv, atomic_write_json


MODELS = (
    "tripartite",
    "collapsed_reward",
    "no_learning",
    "lagged",
)
MODEL_LABELS = {
    "tripartite": "Tripartite",
    "collapsed_reward": "Collapsed reward",
    "no_learning": "No learning",
    "lagged": "Lagged discriminative",
}
INFORMATION_DESIGNS = (
    ("low", 30, 10, 0.20),
    ("intermediate", 60, 25, 0.10),
    ("high", 100, 50, 0.00),
)
PROFILE_LABELS = {
    "balanced": "baseline",
    "rigid_habitual": "habit-oriented",
    "relief_reactive": "affect-oriented",
    "consequence_sensitive": "outcome-oriented",
    "socially_contingent": "social-oriented",
}
ENVIRONMENT_LABELS = {
    "repair_supportive": "Approach-supportive",
    "escalation_prone": "Avoidance-supportive",
    "inconsistent_ambiguous": "Mixed/ambiguous",
}
REPLICATE_PASS_FRACTION_MINIMUM = 0.80
RELATIVE_LOG_LOSS_SKILL_MINIMUM = 0.05
MEAN_ECE_MAXIMUM = 0.10
REPLICATE_ECE_MAXIMUM = 0.15
UNIT_INTERVAL_CLASS_ERROR_MAXIMUM = 0.10
SOCIAL_INFLUENCE_RANGE = 2.0
CHOICE_CONSISTENCY_LOG_ERROR_MAXIMUM = math.log(1.25)

CELL_KEYS = (
    "profile",
    "environment",
    "mean_events",
    "sample_size",
    "missing_rate",
)
REPLICATE_KEYS = (*CELL_KEYS, "replicate")
RECOVERY_FILE_PATTERN = re.compile(
    r"profile=([^/]+)/environment=([^/]+)/events=(\d+)/"
    r"replicate=(\d+)/views/N=(\d+)/missing=(\d+)/fits/"
    r"estimator=population/model=tripartite/recovery\.csv$"
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _prediction_diagnostics(run_summary: pd.DataFrame) -> pd.DataFrame:
    tripartite = run_summary.loc[
        run_summary["model"] == "tripartite"
    ].copy()
    base_rate = run_summary.loc[
        run_summary["model"] == "prevalence_null",
        [*REPLICATE_KEYS, "metric_log_loss"],
    ].rename(columns={"metric_log_loss": "base_rate_log_loss"})
    no_learning = run_summary.loc[
        run_summary["model"] == "no_learning",
        [*REPLICATE_KEYS, "metric_log_loss"],
    ].rename(columns={"metric_log_loss": "no_learning_log_loss"})
    tripartite = tripartite.merge(
        base_rate,
        on=list(REPLICATE_KEYS),
        how="left",
        validate="one_to_one",
    )
    tripartite = tripartite.merge(
        no_learning,
        on=list(REPLICATE_KEYS),
        how="left",
        validate="one_to_one",
    )
    tripartite["relative_log_loss_skill"] = (
        tripartite["base_rate_log_loss"] - tripartite["metric_log_loss"]
    ) / tripartite["base_rate_log_loss"]

    rows = []
    for keys, group in tripartite.groupby(list(CELL_KEYS), dropna=False):
        finite = (
            np.isfinite(group["metric_log_loss"])
            & np.isfinite(group["relative_log_loss_skill"])
            & np.isfinite(group["metric_ece"])
        )
        optimizer = (
            group["diagnostic_optimizer_success_rate"].to_numpy(dtype=float)
            >= 0.80
        )
        skill = group["relative_log_loss_skill"].to_numpy(dtype=float)
        ece = group["metric_ece"].to_numpy(dtype=float)
        row = dict(zip(CELL_KEYS, keys))
        row.update(
            {
                "replicate_count": len(group),
                "complete_replicates": len(group) == 20,
                "finite_prediction_fraction": float(np.mean(finite)),
                "optimizer_pass_fraction": float(np.mean(optimizer)),
                "mean_log_loss": float(group["metric_log_loss"].mean()),
                "mean_base_rate_log_loss": float(
                    group["base_rate_log_loss"].mean()
                ),
                "mean_relative_log_loss_skill": float(np.mean(skill)),
                "relative_skill_pass_fraction": float(
                    np.mean(skill >= RELATIVE_LOG_LOSS_SKILL_MINIMUM)
                ),
                "mean_ece": float(np.mean(ece)),
                "ece_stability_pass_fraction": float(
                    np.mean(ece <= REPLICATE_ECE_MAXIMUM)
                ),
                "mean_tripartite_improvement_over_no_learning": float(
                    np.mean(
                        group["no_learning_log_loss"]
                        - group["metric_log_loss"]
                    )
                ),
            }
        )
        row["minimum_predictive_signal"] = bool(
            row["complete_replicates"]
            and row["finite_prediction_fraction"]
            >= REPLICATE_PASS_FRACTION_MINIMUM
            and row["optimizer_pass_fraction"]
            >= REPLICATE_PASS_FRACTION_MINIMUM
            and row["mean_relative_log_loss_skill"]
            >= RELATIVE_LOG_LOSS_SKILL_MINIMUM
            and row["relative_skill_pass_fraction"]
            >= REPLICATE_PASS_FRACTION_MINIMUM
            and row["mean_ece"] <= MEAN_ECE_MAXIMUM
            and row["ece_stability_pass_fraction"]
            >= REPLICATE_PASS_FRACTION_MINIMUM
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_population_recovery(main: Path) -> pd.DataFrame:
    rows = []
    for path in main.glob(
        "conditions/**/fits/estimator=population/model=tripartite/recovery.csv"
    ):
        match = RECOVERY_FILE_PATTERN.search(path.as_posix())
        if match is None:
            continue
        profile, environment, events, replicate, sample_size, missing = (
            match.groups()
        )
        parameters: Dict[str, Dict[str, float]] = {}
        with path.open(newline="") as stream:
            for record in csv.DictReader(stream):
                if record["level"] != "population":
                    continue
                parameters[record["parameter"]] = {
                    "absolute_error": abs(float(record["bias"])),
                    "true_mean": float(record["true_mean"]),
                    "estimate": float(record["estimate"]),
                }

        weight_error = float(
            np.mean(
                [
                    parameters[name]["absolute_error"]
                    for name in ("w_i", "w_e", "w_u")
                ]
            )
        )
        learning_error = float(
            np.mean(
                [
                    parameters[name]["absolute_error"]
                    for name in (
                        "alpha_i_pos",
                        "alpha_i_neg",
                        "alpha_e",
                        "alpha_u",
                    )
                ]
            )
        )
        social_error = (
            parameters["social_kappa"]["absolute_error"]
            / SOCIAL_INFLUENCE_RANGE
        )
        choice_true = parameters["choice_consistency"]["true_mean"]
        choice_estimate = parameters["choice_consistency"]["estimate"]
        choice_error = abs(math.log(choice_estimate / choice_true))
        rows.append(
            {
                "profile": profile,
                "environment": environment,
                "mean_events": float(events),
                "sample_size": float(sample_size),
                "missing_rate": float(missing) / 100.0,
                "replicate": int(replicate),
                "decision_weight_scaled_error": weight_error,
                "learning_rate_scaled_error": learning_error,
                "social_influence_scaled_error": social_error,
                "choice_consistency_scaled_error": choice_error,
            }
        )
    recovery = pd.DataFrame(rows)
    expected = 5 * 3 * 3 * 3 * 3 * 20
    if len(recovery) != expected:
        raise ValueError(
            f"Expected {expected} tripartite recovery files, found "
            f"{len(recovery)}."
        )
    return recovery


def _population_recovery_diagnostics(recovery: pd.DataFrame) -> pd.DataFrame:
    thresholds = {
        "decision_weight": UNIT_INTERVAL_CLASS_ERROR_MAXIMUM,
        "learning_rate": UNIT_INTERVAL_CLASS_ERROR_MAXIMUM,
        "social_influence": UNIT_INTERVAL_CLASS_ERROR_MAXIMUM,
        "choice_consistency": CHOICE_CONSISTENCY_LOG_ERROR_MAXIMUM,
    }
    rows = []
    for keys, group in recovery.groupby(list(CELL_KEYS), dropna=False):
        row = dict(zip(CELL_KEYS, keys))
        row["recovery_replicate_count"] = len(group)
        class_flags = []
        for class_name, threshold in thresholds.items():
            values = group[f"{class_name}_scaled_error"].to_numpy(
                dtype=float
            )
            mean_error = float(np.mean(values))
            pass_fraction = float(np.mean(values <= threshold))
            class_pass = bool(
                np.isfinite(mean_error)
                and mean_error <= threshold
                and pass_fraction >= REPLICATE_PASS_FRACTION_MINIMUM
            )
            row[f"mean_{class_name}_scaled_error"] = mean_error
            row[f"{class_name}_pass_fraction"] = pass_fraction
            row[f"{class_name}_recovery"] = class_pass
            class_flags.append(class_pass)
        row["full_vector_recovery"] = bool(all(class_flags))
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_diagnostics(
    main: Path,
    run_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction = _prediction_diagnostics(run_summary)
    recovery_replicates = _read_population_recovery(main)
    recovery = _population_recovery_diagnostics(recovery_replicates)
    cells = prediction.merge(
        recovery,
        on=list(CELL_KEYS),
        how="inner",
        validate="one_to_one",
    )
    return cells, recovery_replicates


def _prediction_threshold_sensitivity(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    tripartite = run_summary.loc[
        run_summary["model"] == "tripartite"
    ].copy()
    base_rate = run_summary.loc[
        run_summary["model"] == "prevalence_null",
        [*REPLICATE_KEYS, "metric_log_loss"],
    ].rename(columns={"metric_log_loss": "base_rate_log_loss"})
    tripartite = tripartite.merge(
        base_rate,
        on=list(REPLICATE_KEYS),
        validate="one_to_one",
    )
    tripartite["relative_log_loss_skill"] = (
        tripartite["base_rate_log_loss"] - tripartite["metric_log_loss"]
    ) / tripartite["base_rate_log_loss"]

    rows = []
    for threshold in (0.00, 0.025, 0.05, 0.075, 0.10):
        for require_calibration in (False, True):
            pass_count = 0
            for _, group in tripartite.groupby(
                list(CELL_KEYS),
                dropna=False,
            ):
                skill = group["relative_log_loss_skill"]
                passes = bool(
                    skill.mean() >= threshold
                    and (skill >= threshold).mean()
                    >= REPLICATE_PASS_FRACTION_MINIMUM
                )
                if require_calibration:
                    passes = bool(
                        passes
                        and group["metric_ece"].mean() <= MEAN_ECE_MAXIMUM
                        and (group["metric_ece"] <= REPLICATE_ECE_MAXIMUM).mean()
                        >= REPLICATE_PASS_FRACTION_MINIMUM
                    )
                pass_count += int(passes)
            rows.append(
                {
                    "minimum_relative_log_loss_skill": threshold,
                    "calibration_screen_applied": require_calibration,
                    "simulation_condition_count": 405,
                    "conditions_meeting_criterion": pass_count,
                    "condition_pass_rate": pass_count / 405.0,
                }
            )
    return pd.DataFrame(rows)


def _recovery_threshold_sensitivity(
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    base_thresholds = {
        "decision_weight": UNIT_INTERVAL_CLASS_ERROR_MAXIMUM,
        "learning_rate": UNIT_INTERVAL_CLASS_ERROR_MAXIMUM,
        "social_influence": UNIT_INTERVAL_CLASS_ERROR_MAXIMUM,
        "choice_consistency": CHOICE_CONSISTENCY_LOG_ERROR_MAXIMUM,
    }
    rows = []
    for multiplier in (0.75, 1.00, 1.25, 1.50):
        class_counts = {name: 0 for name in base_thresholds}
        full_count = 0
        for _, group in recovery.groupby(list(CELL_KEYS), dropna=False):
            flags = []
            for class_name, base_threshold in base_thresholds.items():
                threshold = multiplier * base_threshold
                values = group[f"{class_name}_scaled_error"]
                class_pass = bool(
                    values.mean() <= threshold
                    and (values <= threshold).mean()
                    >= REPLICATE_PASS_FRACTION_MINIMUM
                )
                class_counts[class_name] += int(class_pass)
                flags.append(class_pass)
            full_count += int(all(flags))
        row = {
            "threshold_multiplier": multiplier,
            "simulation_condition_count": 405,
            "full_vector_recovery_count": full_count,
            "full_vector_recovery_rate": full_count / 405.0,
        }
        for class_name, count in class_counts.items():
            row[f"{class_name}_recovery_count"] = count
            row[f"{class_name}_recovery_rate"] = count / 405.0
        rows.append(row)
    return pd.DataFrame(rows)


def _design_grid(cells: pd.DataFrame) -> pd.DataFrame:
    grouped = cells.groupby(
        ["missing_rate", "sample_size", "mean_events"],
        dropna=False,
    )
    grid = grouped.agg(
        design_cell_count=("profile", "size"),
        predictive_signal_count=("minimum_predictive_signal", "sum"),
        predictive_signal_rate=("minimum_predictive_signal", "mean"),
        full_vector_recovery_count=("full_vector_recovery", "sum"),
        full_vector_recovery_rate=("full_vector_recovery", "mean"),
        decision_weight_recovery_rate=("decision_weight_recovery", "mean"),
        learning_rate_recovery_rate=("learning_rate_recovery", "mean"),
        social_influence_recovery_rate=("social_influence_recovery", "mean"),
        choice_consistency_recovery_rate=(
            "choice_consistency_recovery",
            "mean",
        ),
        mean_log_loss=("mean_log_loss", "mean"),
        mean_relative_log_loss_skill=(
            "mean_relative_log_loss_skill",
            "mean",
        ),
        mean_ece=("mean_ece", "mean"),
        mean_tripartite_improvement_over_no_learning=(
            "mean_tripartite_improvement_over_no_learning",
            "mean",
        ),
        mean_decision_weight_scaled_error=(
            "mean_decision_weight_scaled_error",
            "mean",
        ),
        mean_learning_rate_scaled_error=(
            "mean_learning_rate_scaled_error",
            "mean",
        ),
        mean_social_influence_scaled_error=(
            "mean_social_influence_scaled_error",
            "mean",
        ),
        mean_choice_consistency_scaled_error=(
            "mean_choice_consistency_scaled_error",
            "mean",
        ),
    ).reset_index()
    return grid


def _factor_summary(
    cells: pd.DataFrame,
    comparisons: pd.DataFrame,
    factor: str,
) -> pd.DataFrame:
    delta_columns = [
        "tripartite_improvement_over_prevalence_mean",
        "no_learning_improvement_over_prevalence_mean",
        "collapsed_reward_improvement_over_prevalence_mean",
        "lagged_improvement_over_prevalence_mean",
        "tripartite_improvement_over_no_learning_mean",
        "collapsed_minus_tripartite_mean",
    ]
    deltas = (
        comparisons.groupby(factor)[delta_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    deltas.columns = [
        column
        if isinstance(column, str)
        else "_".join(part for part in column if part)
        for column in deltas.columns
    ]
    rates = (
        cells.groupby(factor)
        .agg(
            predictive_signal_rate=("minimum_predictive_signal", "mean"),
            full_vector_recovery_rate=("full_vector_recovery", "mean"),
            decision_weight_recovery_rate=(
                "decision_weight_recovery",
                "mean",
            ),
            learning_rate_recovery_rate=(
                "learning_rate_recovery",
                "mean",
            ),
            social_influence_recovery_rate=(
                "social_influence_recovery",
                "mean",
            ),
            choice_consistency_recovery_rate=(
                "choice_consistency_recovery",
                "mean",
            ),
            mean_log_loss=("mean_log_loss", "mean"),
            mean_relative_log_loss_skill=(
                "mean_relative_log_loss_skill",
                "mean",
            ),
            mean_ece=("mean_ece", "mean"),
            mean_decision_weight_scaled_error=(
                "mean_decision_weight_scaled_error",
                "mean",
            ),
            mean_learning_rate_scaled_error=(
                "mean_learning_rate_scaled_error",
                "mean",
            ),
            mean_social_influence_scaled_error=(
                "mean_social_influence_scaled_error",
                "mean",
            ),
            mean_choice_consistency_scaled_error=(
                "mean_choice_consistency_scaled_error",
                "mean",
            ),
        )
        .reset_index()
    )
    return rates.merge(deltas, on=factor, how="left", validate="one_to_one")


def _information_design_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "metric_log_loss",
        "metric_auc",
        "metric_pr_auc",
        "metric_ece",
    )
    rows = []
    for information, sample_size, mean_events, missing_rate in (
        INFORMATION_DESIGNS
    ):
        frame = run_summary.loc[
            run_summary["model"].isin(MODELS)
            & (run_summary["sample_size"] == sample_size)
            & (run_summary["mean_events"] == mean_events)
            & np.isclose(run_summary["missing_rate"], missing_rate)
        ]
        cell_means = (
            frame.groupby(["model", "profile", "environment"])[
                list(metrics)
            ]
            .mean()
            .reset_index()
        )
        for model, group in cell_means.groupby("model"):
            model_replicates = frame.loc[frame["model"] == model]
            row = {
                "information": information,
                "sample_size": sample_size,
                "mean_events": mean_events,
                "missing_rate": missing_rate,
                "model": model,
                "condition_cell_count": len(group),
                "replicate_count": len(model_replicates),
            }
            for metric in metrics:
                label = metric.removeprefix("metric_")
                row[f"mean_{label}"] = float(group[metric].mean())
                row[f"se_{label}"] = float(
                    group[metric].std(ddof=1) / math.sqrt(len(group))
                )
            rows.append(row)
    summary = pd.DataFrame(rows)
    expected = len(INFORMATION_DESIGNS) * len(MODELS)
    if len(summary) != expected:
        raise ValueError(
            f"Expected {expected} information-by-model rows, found "
            f"{len(summary)}."
        )
    return summary


def _missingness_summary(
    contrasts: pd.DataFrame,
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        model_rows = contrasts.loc[contrasts["model"] == model]
        for missing_percent in (10, 20):
            log_loss_column = (
                f"metric_log_loss_change_00_to_{missing_percent:02d}"
            )
            bias_column = (
                "population_mean_abs_bias_change_00_to_"
                f"{missing_percent:02d}"
            )
            log_loss = model_rows[log_loss_column].dropna()
            bias = model_rows[bias_column].dropna()
            rows.append(
                {
                    "model": model,
                    "missingness_percent": missing_percent,
                    "n_paired_datasets": len(log_loss),
                    "mean_log_loss_change": log_loss.mean(),
                    "sd_log_loss_change": log_loss.std(ddof=1),
                    "log_loss_worsened_fraction": (log_loss > 0).mean(),
                    "mean_population_bias_change": (
                        bias.mean() if len(bias) else np.nan
                    ),
                    "sd_population_bias_change": (
                        bias.std(ddof=1) if len(bias) else np.nan
                    ),
                    "population_bias_worsened_fraction": (
                        (bias > 0).mean() if len(bias) else np.nan
                    ),
                }
            )
    output = pd.DataFrame(rows)
    recovery_keys = [
        "profile",
        "environment",
        "mean_events",
        "sample_size",
        "replicate",
    ]
    error_columns = [
        "decision_weight_scaled_error",
        "learning_rate_scaled_error",
        "social_influence_scaled_error",
        "choice_consistency_scaled_error",
    ]
    wide = recovery.pivot_table(
        index=recovery_keys,
        columns="missing_rate",
        values=error_columns,
        aggfunc="first",
    )
    for missing_percent in (10, 20):
        missing_rate = missing_percent / 100.0
        mask = (
            (output["model"] == "tripartite")
            & (output["missingness_percent"] == missing_percent)
        )
        for error_column in error_columns:
            changes = (
                wide[(error_column, missing_rate)]
                - wide[(error_column, 0.0)]
            )
            prefix = error_column.removesuffix("_scaled_error")
            output.loc[mask, f"mean_{prefix}_error_change"] = changes.mean()
            output.loc[mask, f"sd_{prefix}_error_change"] = changes.std(ddof=1)
            output.loc[mask, f"{prefix}_error_worsened_fraction"] = (
                changes > 0
            ).mean()
    return output


def _phenotype_environment_event_summary(
    cells: pd.DataFrame,
) -> pd.DataFrame:
    criteria = {
        "prediction_criterion_rate": "minimum_predictive_signal",
        "decision_weight_recovery_rate": "decision_weight_recovery",
        "learning_rate_recovery_rate": "learning_rate_recovery",
        "social_influence_recovery_rate": "social_influence_recovery",
        "choice_consistency_recovery_rate": "choice_consistency_recovery",
    }
    rows = []
    group_keys = ["mean_events", "profile", "environment"]
    for keys, group in cells.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys))
        row["condition_count"] = len(group)
        for output_name, source_name in criteria.items():
            row[output_name] = float(group[source_name].astype(float).mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary) != 45:
        raise ValueError(
            "Expected 45 event-by-phenotype-by-environment rows, found "
            f"{len(summary)}."
        )
    if not (summary["condition_count"] == 9).all():
        raise ValueError(
            "Each phenotype-environment-event cell must summarize nine "
            "sample-size-by-missingness conditions."
        )
    return summary


def _benchmark_tables(
    benchmark_paths: Sequence[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rate_frames = []
    confusion_frames = []
    margin_rows = []
    for path in benchmark_paths:
        manifest = json.loads((path / "benchmark_manifest.json").read_text())
        profile = str(manifest["profile"])
        rates = pd.read_csv(path / "false_selection_rates.csv")
        rates.insert(0, "profile", profile)
        rate_frames.append(rates)
        confusion = pd.read_csv(path / "model_selection_confusion.csv")
        confusion.insert(0, "profile", profile)
        confusion_frames.append(confusion)

        summary = pd.read_csv(path / "benchmark_run_summary.csv")
        candidates = summary.loc[summary["model"].isin(MODELS)].copy()
        index = ["scenario", "generating_model", "replicate"]
        wide = candidates.pivot_table(
            index=index,
            columns="model",
            values="metric_log_loss",
            aggfunc="first",
        )
        for keys, row in wide.iterrows():
            scenario, generating_model, replicate = keys
            correct = float(row[generating_model])
            best_alternative = float(row.drop(generating_model).min())
            tripartite_minus_collapsed = float(
                row["tripartite"] - row["collapsed_reward"]
            )
            margin_rows.append(
                {
                    "profile": profile,
                    "scenario": scenario,
                    "generating_model": generating_model,
                    "replicate": replicate,
                    "correct_minus_best_alternative": (
                        correct - best_alternative
                    ),
                    "tripartite_minus_collapsed_reward": (
                        tripartite_minus_collapsed
                    ),
                }
            )
    rates = pd.concat(rate_frames, ignore_index=True)
    confusion = pd.concat(confusion_frames, ignore_index=True)
    margins = pd.DataFrame(margin_rows)
    grouped = margins.groupby(
        ["profile", "scenario", "generating_model"],
        dropna=False,
    )
    margin_summary = grouped.agg(
        replicate_count=("replicate", "size"),
        mean_correct_minus_best_alternative=(
            "correct_minus_best_alternative",
            "mean",
        ),
        median_correct_minus_best_alternative=(
            "correct_minus_best_alternative",
            "median",
        ),
        correct_model_selected_fraction=(
            "correct_minus_best_alternative",
            lambda values: np.mean(values < 0),
        ),
        correct_margin_within_005_fraction=(
            "correct_minus_best_alternative",
            lambda values: np.mean(np.abs(values) < 0.005),
        ),
        correct_margin_within_010_fraction=(
            "correct_minus_best_alternative",
            lambda values: np.mean(np.abs(values) < 0.010),
        ),
        mean_tripartite_minus_collapsed_reward=(
            "tripartite_minus_collapsed_reward",
            "mean",
        ),
        tripartite_collapsed_within_005_fraction=(
            "tripartite_minus_collapsed_reward",
            lambda values: np.mean(np.abs(values) < 0.005),
        ),
    ).reset_index()
    return rates, confusion, margin_summary


def _runtime_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    computational = run_summary.loc[
        run_summary["model"].isin(MODELS)
    ].copy()
    return (
        computational.groupby(
            ["model", "sample_size", "mean_events"],
            dropna=False,
        )["diagnostic_runtime_seconds"]
        .agg(["count", "mean", "median", "min", "max", "sum"])
        .reset_index()
        .rename(
            columns={
                "count": "fit_count",
                "mean": "mean_runtime_seconds",
                "median": "median_runtime_seconds",
                "min": "minimum_runtime_seconds",
                "max": "maximum_runtime_seconds",
                "sum": "total_runtime_seconds",
            }
        )
    )


def _heatmap(
    axis: plt.Axes,
    grid: pd.DataFrame,
    value: str,
    title: str,
    cmap: str,
) -> matplotlib.image.AxesImage:
    pivot = grid.pivot(
        index="sample_size",
        columns="mean_events",
        values=value,
    ).sort_index()
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        aspect="auto",
    )
    axis.set_xticks(range(len(pivot.columns)), [int(v) for v in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [int(v) for v in pivot.index])
    axis.set_xlabel("Mean events per participant")
    axis.set_ylabel("Sample size")
    axis.set_title(title)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value_at_cell = pivot.iloc[row, column]
            color = "white" if value_at_cell < 0.35 else "black"
            axis.text(
                column,
                row,
                f"{100 * value_at_cell:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    return image


def _plot_design_grid(grid: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(
        5,
        3,
        figsize=(10.5, 13.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    missing_rates = (0.0, 0.1, 0.2)
    row_specs = (
        ("predictive_signal_rate", "Prediction benchmark"),
        ("decision_weight_recovery_rate", "Decision weights"),
        ("learning_rate_recovery_rate", "Learning rates"),
        ("social_influence_recovery_rate", "Social influence"),
        ("choice_consistency_recovery_rate", "Choice consistency"),
    )
    image = None
    for row, (value, row_label) in enumerate(row_specs):
        for column, missing_rate in enumerate(missing_rates):
            axis = axes[row, column]
            frame = grid.loc[grid["missing_rate"] == missing_rate]
            image = _heatmap(axis, frame, value, "", "viridis")
            if row == 0:
                axis.set_title(f"{int(100 * missing_rate)}% missingness")
            if column == 0:
                axis.set_ylabel(f"{row_label}\nSample size")
            else:
                axis.set_ylabel("")
            if row < len(row_specs) - 1:
                axis.set_xlabel("")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            shrink=0.84,
            label="Conditions meeting criterion",
        )
    figure.suptitle(
        "Percentage of Simulation Conditions Meeting Prediction and "
        "Parameter-Recovery Criteria\nby Sample Size, Mean Number of Events "
        "per Participant, and Missingness",
        fontsize=12,
    )
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_phenotype_environment_by_events(
    summary: pd.DataFrame,
    output: Path,
) -> None:
    profile_order = [
        "balanced",
        "rigid_habitual",
        "relief_reactive",
        "consequence_sensitive",
        "socially_contingent",
    ]
    environment_order = [
        "repair_supportive",
        "inconsistent_ambiguous",
        "escalation_prone",
    ]
    event_yields = [10, 25, 50]
    row_specs = (
        ("prediction_criterion_rate", "Prediction benchmark"),
        ("decision_weight_recovery_rate", "Decision weights"),
        ("learning_rate_recovery_rate", "Learning rates"),
        ("social_influence_recovery_rate", "Social influence"),
        ("choice_consistency_recovery_rate", "Choice consistency"),
    )
    figure, axes = plt.subplots(
        5,
        3,
        figsize=(11.5, 14.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    image = None
    for row, (value_column, row_label) in enumerate(row_specs):
        for column, mean_events in enumerate(event_yields):
            axis = axes[row, column]
            frame = summary.loc[
                summary["mean_events"].astype(float) == float(mean_events)
            ]
            pivot = (
                frame.pivot(
                    index="profile",
                    columns="environment",
                    values=value_column,
                )
                .reindex(index=profile_order, columns=environment_order)
            )
            if pivot.isna().any().any():
                raise ValueError(
                    "Incomplete phenotype-by-environment heatmap for "
                    f"{mean_events} mean events and {value_column}."
                )
            image = axis.imshow(
                pivot.to_numpy(dtype=float),
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                aspect="auto",
            )
            axis.set_xticks(
                range(len(environment_order)),
                [ENVIRONMENT_LABELS[value] for value in environment_order],
                rotation=28,
                ha="right",
            )
            axis.set_yticks(
                range(len(profile_order)),
                [PROFILE_LABELS[value] for value in profile_order],
            )
            for tick in axis.get_yticklabels():
                tick.set_fontstyle("italic")
            for profile_index in range(len(profile_order)):
                for environment_index in range(len(environment_order)):
                    cell_value = pivot.iloc[
                        profile_index,
                        environment_index,
                    ]
                    text_color = "white" if cell_value < 0.35 else "black"
                    axis.text(
                        environment_index,
                        profile_index,
                        f"{100 * cell_value:.0f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8.5,
                    )
            if row == 0:
                axis.set_title(
                    f"{mean_events} mean events per participant",
                    fontsize=10,
                )
            if column == 0:
                axis.set_ylabel(row_label)
            else:
                axis.set_ylabel("")
            if row < len(row_specs) - 1:
                axis.tick_params(labelbottom=False)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            shrink=0.84,
            label="Simulation conditions meeting criterion",
        )
    figure.suptitle(
        "Percentage of Simulation Conditions Meeting Prediction and "
        "Parameter-Recovery Criteria\nby Mean Event Yield, Person-Behavior "
        "Phenotype, and Social-Environment Consequence Profile",
        fontsize=12,
    )
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_profile_comparison(profile: pd.DataFrame, output: Path) -> None:
    ordered = [
        "balanced",
        "rigid_habitual",
        "relief_reactive",
        "consequence_sensitive",
        "socially_contingent",
    ]
    frame = profile.set_index("profile").loc[ordered].reset_index()
    mean_column = "collapsed_minus_tripartite_mean_mean"
    sd_column = "collapsed_minus_tripartite_mean_std"
    count_per_profile = 81
    error = 1.96 * frame[sd_column] / np.sqrt(count_per_profile)

    figure, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    colors = [
        "#4C78A8" if value > 0 else "#E45756"
        for value in frame[mean_column]
    ]
    axis.bar(
        range(len(frame)),
        frame[mean_column],
        yerr=error,
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        range(len(frame)),
        [PROFILE_LABELS[value] for value in frame["profile"]],
        rotation=25,
        ha="right",
    )
    for label in axis.get_xticklabels():
        label.set_fontstyle("italic")
    axis.set_ylabel("Collapsed reward minus tripartite log loss")
    axis.set_title(
        r"Separation of $\mathit{enjoyment}$ and $\mathit{utility}$ Depends on Person-Behavior Phenotype"
    )
    axis.text(
        0.01,
        0.98,
        "Positive values favor the tripartite model",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_information_model_comparison(
    summary: pd.DataFrame,
    output: Path,
) -> None:
    information_order = [value[0] for value in INFORMATION_DESIGNS]
    tick_labels = [
        f"{label.title()}\n{sample_size} / {mean_events} / "
        f"{int(100 * missing_rate)}%"
        for label, sample_size, mean_events, missing_rate in INFORMATION_DESIGNS
    ]
    colors = {
        "tripartite": "#0072B2",
        "collapsed_reward": "#E69F00",
        "no_learning": "#D55E00",
        "lagged": "#009E73",
    }
    markers = {
        "tripartite": "o",
        "collapsed_reward": "s",
        "no_learning": "^",
        "lagged": "D",
    }
    panels = (
        ("mean_log_loss", "se_log_loss", "Held-out log loss", "Lower is better"),
        ("mean_auc", "se_auc", "ROC area under the curve", "Higher is better"),
        ("mean_pr_auc", "se_pr_auc", "Precision-recall area under the curve", "Higher is better"),
        ("mean_ece", "se_ece", "Expected calibration error", "Lower is better"),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 6.4),
        sharex=True,
    )
    x = np.arange(len(information_order))
    for axis, (mean_column, se_column, ylabel, direction) in zip(
        axes.flat,
        panels,
    ):
        for model in MODELS:
            frame = (
                summary.loc[summary["model"] == model]
                .set_index("information")
                .loc[information_order]
            )
            axis.errorbar(
                x,
                frame[mean_column],
                yerr=1.96 * frame[se_column],
                color=colors[model],
                marker=markers[model],
                markersize=5,
                linewidth=1.8,
                capsize=3,
                label=MODEL_LABELS[model],
            )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.set_title(direction, loc="right", fontsize=8.5, color="#444444")
        axis.set_xticks(x, tick_labels)
    figure.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
    )
    figure.supxlabel(
        "Information condition (participants / mean events / missingness)",
        y=0.01,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.91), h_pad=2.0)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_missingness(missingness: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.0, 6.2),
        constrained_layout=True,
    )
    colors = {
        "tripartite": "#4C78A8",
        "collapsed_reward": "#F58518",
        "no_learning": "#54A24B",
        "lagged": "#B279A2",
    }
    for model in MODELS:
        frame = missingness.loc[missingness["model"] == model].sort_values(
            "missingness_percent"
        )
        axes[0, 0].plot(
            frame["missingness_percent"],
            frame["mean_log_loss_change"],
            marker="o",
            label=model.replace("_", " ").title(),
            color=colors[model],
        )
    tripartite = missingness.loc[
        missingness["model"] == "tripartite"
    ].sort_values("missingness_percent")
    class_specs = (
        ("decision_weight", "Decision weights"),
        ("learning_rate", "Learning rates"),
        ("social_influence", "Social influence"),
        ("choice_consistency", "Choice consistency"),
    )
    for axis, (column, title) in zip(axes.flat[1:], class_specs):
        axis.plot(
            tripartite["missingness_percent"],
            tripartite[f"mean_{column}_error_change"],
            marker="o",
            color="#4C78A8",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(
            xlabel="Event-level missingness (%)",
            ylabel="Change in scale-adjusted error",
            title=title,
            xticks=[10, 20],
        )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(
        xlabel="Event-level missingness (%)",
        ylabel="Change in held-out log loss",
        title="Prediction",
        xticks=[10, 20],
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 2].axis("off")
    figure.suptitle("Consequences of an Incomplete Observed Learning History")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_model_recovery(rates: pd.DataFrame, output: Path) -> None:
    profiles = ["balanced", "relief_reactive", "consequence_sensitive"]
    scenarios = ["low", "middle", "high"]
    generators = ["tripartite", "collapsed_reward", "no_learning", "lagged"]
    figure, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(9.2, 7.0),
        constrained_layout=True,
        sharex=True,
    )
    for axis, profile in zip(axes, profiles):
        subset = rates.loc[rates["profile"] == profile]
        available_generators = [
            generator
            for generator in generators
            if generator in set(subset["generating_model"])
        ]
        width = 0.8 / max(1, len(available_generators))
        positions = np.arange(len(scenarios))
        for index, generator in enumerate(available_generators):
            generator_rows = subset.loc[
                subset["generating_model"] == generator
            ].set_index("scenario")
            values = [
                generator_rows.loc[scenario, "correct_selection_rate"]
                if scenario in generator_rows.index
                else np.nan
                for scenario in scenarios
            ]
            axis.bar(
                positions
                - 0.4
                + width / 2
                + index * width,
                values,
                width=width,
                label=generator.replace("_", " ").title(),
            )
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Correct selection")
        axis.set_title(PROFILE_LABELS[profile], fontstyle="italic")
        axis.axhline(0.5, color="black", linewidth=0.6, linestyle="--")
    scenario_labels = {
        "low": "Sparse",
        "middle": "Intermediate",
        "high": "Dense",
    }
    axes[-1].set_xticks(
        positions,
        [scenario_labels[value] for value in scenarios],
    )
    axes[-1].set_xlabel("Benchmark design")
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
    figure.suptitle("Exact Held-Out Model Selection by Person-Behavior Phenotype")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _legacy_interpretation_report(
    audit: Dict[str, object],
    fit_status: Dict[str, object],
    grid: pd.DataFrame,
    profile: pd.DataFrame,
    environment: pd.DataFrame,
    missingness: pd.DataFrame,
    recovery_rates: pd.DataFrame,
) -> str:
    def grid_row(
        sample_size: int,
        mean_events: int,
        missing_rate: float,
    ) -> pd.Series:
        return grid.loc[
            (grid["sample_size"] == sample_size)
            & (grid["mean_events"] == mean_events)
            & (grid["missing_rate"] == missing_rate)
        ].iloc[0]

    def pooled_row(sample_size: int, mean_events: int) -> pd.Series:
        rows = grid.loc[
            (grid["sample_size"] == sample_size)
            & (grid["mean_events"] == mean_events)
        ]
        return rows.mean(numeric_only=True)

    sparse = pooled_row(30, 10)
    dense_events = pooled_row(30, 50)
    middle_dense = pooled_row(60, 50)
    generous = pooled_row(100, 50)
    generous_complete = grid_row(100, 50, 0.0)
    generous_missing_10 = grid_row(100, 50, 0.1)
    generous_missing_20 = grid_row(100, 50, 0.2)
    profile_index = profile.set_index("profile")
    environment_index = environment.set_index("environment")
    missing_trip = missingness.loc[
        missingness["model"] == "tripartite"
    ].set_index("missingness_percent")

    def selection(profile_name: str, scenario: str, generator: str) -> float:
        rows = recovery_rates.loc[
            (recovery_rates["profile"] == profile_name)
            & (recovery_rates["scenario"] == scenario)
            & (recovery_rates["generating_model"] == generator)
        ]
        return float(rows.iloc[0]["correct_selection_rate"])

    return f"""# PDTRT Production Results: Interpretation and Writing Guide

## Integrity

- Main factorial audit: `{audit["status"]}`.
- Observed datasets: {audit["observed_dataset_count"]:,} of {audit["expected_dataset_count"]:,}.
- Analysis rows: {audit["observed_analysis_rows"]:,} of {audit["expected_analysis_rows"]:,}.
- Failed or missing model rows: {audit["failed_or_missing_model_rows"]}.
- Recorded model-fit CPU time: {fit_status["recorded_model_fit_runtime_seconds"] / 3600:.1f} hours.

## Finding 1: Population-Average Parameter Recovery Is an Interpretive Prerequisite

Population-average recovery was evaluated independently of prediction. None
of the phenotype x environment x missingness cells met the recovery criterion
at N = 30, including the 50-event condition. At N = 60 with 50 mean events,
{100 * middle_dense["population_recovery_rate"]:.1f}% met the criterion. At
N = 100 with 50 mean events, recovery was adequate in
{100 * generous_complete["population_recovery_rate"]:.1f}% of complete-history
conditions, {100 * generous_missing_10["population_recovery_rate"]:.1f}% with
10% event-level missingness, and
{100 * generous_missing_20["population_recovery_rate"]:.1f}% with 20%
missingness.

Mean population-center absolute bias declined from
{sparse["mean_population_center_abs_bias"]:.3f} at N = 30 with 10 mean events
to {generous["mean_population_center_abs_bias"]:.3f} at N = 100 with 50 mean
events. Recovery is necessary before fitted parameters are interpreted as
population learning processes, but it is not itself a substantive study goal.

## Finding 2: Prediction Adequacy Depends Primarily on Sampling Information

Across the 45 phenotype x environment x missingness cells represented at each
sample-size-by-event-yield combination, N = 30 with 10 mean events met the
held-out prediction criterion in {100 * sparse["prediction_adequate_rate"]:.1f}%
of cells. Increasing event yield to 50 at the same sample size raised this to
{100 * dense_events["prediction_adequate_rate"]:.1f}%. N = 60 with 50 events
and N = 100 with 50 events were adequate in
{100 * middle_dense["prediction_adequate_rate"]:.1f}% and
{100 * generous["prediction_adequate_rate"]:.1f}% of cells, respectively.

Designs that predict held-out behavior adequately can still recover the
population parameters poorly. Prediction and recovery should therefore be
reported separately and combined only when discussing whether a fitted
population parameter is potentially interpretable.

## Finding 3: Learning Versus No Learning Is a Candidate-Model Comparison

The tripartite model's mean held-out log-loss advantage over the no-learning
model increased from {sparse["mean_tripartite_improvement_over_no_learning"]:.3f}
at N = 30 with 10 mean events to
{dense_events["mean_tripartite_improvement_over_no_learning"]:.3f} at the same
sample size with 50 mean events. Increasing sample size at 10 events produced
a smaller gain than increasing event yield. More participants improve
population estimation, but repeated observations within participants supplied
more evidence for distinguishing updating from a static decision process.
This contrast belongs with the collapsed-reward and lagged comparisons rather
than serving as a general study-design adequacy criterion.

## Finding 4: Missing Events Have a Small Average Cost and a Meaningful Reliability Cost

For the tripartite model, masking 10% of eligible events increased held-out log
loss by {missing_trip.loc[10, "mean_log_loss_change"]:.3f} on average and
increased population-center absolute bias by
{missing_trip.loc[10, "mean_population_bias_change"]:.3f}. Masking 20% increased
them by {missing_trip.loc[20, "mean_log_loss_change"]:.3f} and
{missing_trip.loc[20, "mean_population_bias_change"]:.3f}. Log loss worsened in
{100 * missing_trip.loc[20, "log_loss_worsened_fraction"]:.1f}% of paired
20%-missing datasets, and parameter bias worsened in
{100 * missing_trip.loc[20, "population_bias_worsened_fraction"]:.1f}%.

The average changes are modest because many events remain observed, but an
missing event has a larger conceptual cost in a learning model than an ordinary
missing row. It changes the person's later latent state while leaving the
fitted model unaware of the update.

## Finding 5: The Value of Separating Enjoyment and Utility Is Theory Dependent

Positive collapsed-minus-tripartite log-loss differences favor the
tripartite model. The mean differences were
{profile_index.loc["relief_reactive", "collapsed_minus_tripartite_mean_mean"]:.3f}
for the affect-oriented phenotype and
{profile_index.loc["consequence_sensitive", "collapsed_minus_tripartite_mean_mean"]:.3f}
for the outcome-oriented phenotype. They were close to zero and slightly
favored the simpler collapsed model for the baseline phenotype
({profile_index.loc["balanced", "collapsed_minus_tripartite_mean_mean"]:.3f}),
the habit-oriented phenotype
({profile_index.loc["rigid_habitual", "collapsed_minus_tripartite_mean_mean"]:.3f}),
and the social-oriented phenotype
({profile_index.loc["socially_contingent", "collapsed_minus_tripartite_mean_mean"]:.3f}).

Separate enjoyment and utility states are therefore empirically useful when
the process hypothesis makes them behave differently. The simulation does not
support treating additional computational detail as automatically beneficial.

## Finding 6: Social Environment Changes Informativeness

The mean tripartite log loss was
{environment_index.loc["escalation_prone", "mean_log_loss"]:.3f} in
Escalation-Prone environments,
{environment_index.loc["repair_supportive", "mean_log_loss"]:.3f} in
Repair-Supportive environments, and
{environment_index.loc["inconsistent_ambiguous", "mean_log_loss"]:.3f} in
Inconsistent/Ambiguous environments. The ambiguous environment also produced
the smallest average tripartite advantage over no learning
({environment_index.loc["inconsistent_ambiguous", "tripartite_improvement_over_no_learning_mean_mean"]:.3f}).
The same AA design can therefore be more or less informative depending on the
consequence ecology in which behavior occurs.

## Finding 7: Model Recovery Places a Strong Boundary on Mechanistic Claims

Under the baseline phenotype, exact selection of the tripartite generator
was {100 * selection("balanced", "low", "tripartite"):.0f}% in the sparse,
{100 * selection("balanced", "middle", "tripartite"):.0f}% in the intermediate,
and {100 * selection("balanced", "high", "tripartite"):.0f}% in the dense
design. In the dense design, tripartite and collapsed-reward log loss differed
by less than .005 in every replicate. The two models are functionally difficult
to distinguish when enjoyment and utility have similar roles.

In diagnostic phenotypes, bidirectional recovery improved. For the
affect-oriented phenotype, the tripartite generator was selected in
{100 * selection("relief_reactive", "middle", "tripartite"):.0f}% of intermediate-design
and {100 * selection("relief_reactive", "high", "tripartite"):.0f}% of dense-design
replicates. The corresponding rates were
{100 * selection("consequence_sensitive", "middle", "tripartite"):.0f}% and
{100 * selection("consequence_sensitive", "high", "tripartite"):.0f}% for the
outcome-oriented phenotype. Recovery of the collapsed generator also
improved with information.

The lagged generator was readily distinguishable, but the no-learning generator
was often selected interchangeably with flexible learning models whose fitted
learning rates approached zero. Raw minimum log loss does not reward the
parsimony of the nested no-learning model. These findings argue for
bidirectional recovery, theoretically diagnostic contexts, and cautious
interpretation when candidate models make nearly equivalent predictions.

## Manuscript Boundaries

1. These are model- and generator-specific design results, not universal AA
   sample-size recommendations.
2. Population-average parameter recovery is an interpretive prerequisite,
   prediction is a substantive performance criterion, and candidate-model
   comparisons test particular process assumptions. They must be reported
   separately.
3. Person-specific learning rates are not production estimands because the
   preproduction recovery ladder did not support their interpretation.
4. Synthetic recovery supports feasibility under stated assumptions, not
   empirical validation of a clinical mechanism.
5. Exact cell thresholds can be nonmonotonic across nested designs because of
   finite Monte Carlo variation and heterogeneous sampled participants.
   Emphasize broad gradients, pass fractions, and Pareto sets.
6. Exact model-winner rates should be accompanied by the magnitude of
   predictive differences; many tripartite-versus-collapsed comparisons are
   practically equivalent.

## Output Map

- `manuscript_tripartite_design_grid.csv`: primary N x event-yield x
  missingness results.
- `manuscript_model_comparison_by_profile.csv`: trait-process heterogeneity.
- `manuscript_model_performance_by_information.csv`: prediction across the
  low-, intermediate-, and high-information designs.
- `manuscript_model_comparison_by_environment.csv`: contextual heterogeneity.
- `manuscript_missingness_effects.csv`: paired hidden-history effects.
- `supplement_model_recovery_rates.csv`: correct-selection rates.
- `supplement_model_recovery_confusion.csv`: full confusion matrices.
- `supplement_model_recovery_margins.csv`: practical-equivalence diagnostics.
- `supplement_runtime_summary.csv`: computational feasibility.
- `figure_design_adequacy_heatmaps.*`: design adequacy figure.
- `figure_model_comparison_by_information.*`: four-model prediction across
  the low-, intermediate-, and high-information designs.
- `figure_missingness_effects.*`: hidden-history figure.
- `figure_model_recovery.*`: model-selection figure.
    """


def _interpretation_report(
    audit: Dict[str, object],
    fit_status: Dict[str, object],
    grid: pd.DataFrame,
    profile: pd.DataFrame,
    environment: pd.DataFrame,
    missingness: pd.DataFrame,
    recovery_rates: pd.DataFrame,
) -> str:
    def pooled(sample_size: int, mean_events: int) -> pd.Series:
        rows = grid.loc[
            (grid["sample_size"] == sample_size)
            & (grid["mean_events"] == mean_events)
        ]
        return rows.mean(numeric_only=True)

    sparse = pooled(30, 10)
    dense_within_person = pooled(30, 50)
    intermediate = pooled(60, 25)
    generous = pooled(100, 50)
    missing_tripartite = missingness.loc[
        missingness["model"] == "tripartite"
    ].set_index("missingness_percent")

    return f"""# PDTRT Production Results: Interpretation and Writing Guide

## Integrity

- Main factorial audit: `{audit["status"]}`.
- Observed datasets: {audit["observed_dataset_count"]:,} of {audit["expected_dataset_count"]:,}.
- Analysis rows: {audit["observed_analysis_rows"]:,} of {audit["expected_analysis_rows"]:,}.
- Failed or missing model rows: {audit["failed_or_missing_model_rows"]}.
- Recorded model-fit CPU time: {fit_status["recorded_model_fit_runtime_seconds"] / 3600:.1f} hours.

## Prediction

The manuscript-facing prediction benchmark requires a mean relative log-loss
improvement of at least 5% over the base-rate model, the same minimum in at
least 80% of replicates, mean expected calibration error no larger than .10,
expected calibration error no larger than .15 in at least 80% of replicates,
and finite converged results. The benchmark was met in
{100 * sparse["predictive_signal_rate"]:.1f}% of conditions at N = 30 with 10
mean events, {100 * dense_within_person["predictive_signal_rate"]:.1f}% at N =
30 with 50 events, and {100 * intermediate["predictive_signal_rate"]:.1f}% at
N = 60 with 25 events. Continuous skill and calibration should accompany all
thresholded summaries.

## Population-Average Parameter Recovery

Recovery is evaluated separately for decision weights, learning rates, social
influence, and effective choice consistency. Errors for parameters bounded
from 0 to 1 are expressed on that unit scale, social-influence error is divided
by its 0-to-2 range, and choice-consistency error is an absolute log ratio.
A class must meet its scale-specific threshold on average and in at least 80%
of replicates. The full vector passes only if all four classes pass, so strong
recovery of one class cannot offset poor recovery of another.

Across all 405 factorial conditions, decision weights met their criterion in
{100 * grid["decision_weight_recovery_rate"].mean():.1f}%, learning rates in
{100 * grid["learning_rate_recovery_rate"].mean():.1f}%, social influence in
{100 * grid["social_influence_recovery_rate"].mean():.1f}%, and choice
consistency in {100 * grid["choice_consistency_recovery_rate"].mean():.1f}%.
No condition recovered all four parameter classes. Even at N = 100 with 50
mean events, the pooled class pass rates were
{100 * generous["decision_weight_recovery_rate"]:.1f}% for decision weights,
{100 * generous["learning_rate_recovery_rate"]:.1f}% for learning rates,
{100 * generous["social_influence_recovery_rate"]:.1f}% for social influence,
and {100 * generous["choice_consistency_recovery_rate"]:.1f}% for choice
consistency. Population parameters should therefore not be treated as jointly
recovered in this demonstration.

## Missingness

For the tripartite model, masking 10% of eligible events increased held-out log
loss by {missing_tripartite.loc[10, "mean_log_loss_change"]:.3f} and the former
aggregate population absolute-error summary by
{missing_tripartite.loc[10, "mean_population_bias_change"]:.3f} on average.
Masking 20% increased them by
{missing_tripartite.loc[20, "mean_log_loss_change"]:.3f} and
{missing_tripartite.loc[20, "mean_population_bias_change"]:.3f}. Parameter-class
errors are the primary recovery summaries; this aggregate is retained only for
the paired missingness comparison with the frozen production output.

## Reporting Boundaries

1. The prediction benchmark is a minimum signal-and-calibration screen, not a
   universal definition of adequate prediction.
2. Parameter recovery is class specific. No population parameter should be
   interpreted solely because the former across-parameter mean error was small.
3. Threshold sensitivity and continuous metrics accompany binary pass rates.
4. Person-specific learning rates are not production estimands.
5. Candidate-model comparisons test process assumptions separately from
   prediction and parameter recovery.

## Output Map

- `manuscript_tripartite_design_grid.csv`: prediction and parameter-class
  recovery by sample size, event yield, and missingness.
- `supplement_prediction_recovery_by_condition.csv`: all 405 factorial cells.
- `supplement_prediction_threshold_sensitivity.csv`: prediction benchmark
  sensitivity.
- `supplement_recovery_threshold_sensitivity.csv`: parameter-class recovery
  sensitivity.
- `manuscript_model_comparison_by_profile.csv`: phenotype heterogeneity.
- `manuscript_phenotype_environment_by_events.csv`: prediction and
  parameter-recovery criteria by event yield, phenotype, and
  social-environment consequence profile.
- `manuscript_model_performance_by_information.csv`: prediction across the
  low-, intermediate-, and high-information designs.
- `manuscript_model_comparison_by_environment.csv`: contextual heterogeneity.
- `manuscript_missingness_effects.csv`: paired hidden-history effects.
- `supplement_model_recovery_rates.csv`: bidirectional model recovery.
"""


def run(args: argparse.Namespace) -> int:
    main = Path(args.main_outdir).expanduser().resolve()
    output = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else main / "manuscript_results"
    )
    output.mkdir(parents=True, exist_ok=True)

    audit = json.loads((main / "production_run_audit.json").read_text())
    fit_status = json.loads((main / "fit_status.json").read_text())
    comparisons = pd.read_csv(main / "paired_model_comparison_summary.csv")
    missingness_contrasts = pd.read_csv(
        main / "paired_missingness_contrasts.csv"
    )
    run_summary = pd.read_csv(main / "run_summary.csv")

    benchmark_paths = [
        Path(value).expanduser().resolve()
        for value in args.benchmark_outdirs
    ]
    cells, recovery_replicates = _cell_diagnostics(main, run_summary)
    grid = _design_grid(cells)
    profile = _factor_summary(cells, comparisons, "profile")
    environment = _factor_summary(cells, comparisons, "environment")
    phenotype_environment_events = _phenotype_environment_event_summary(cells)
    information_designs = _information_design_summary(run_summary)
    missingness = _missingness_summary(
        missingness_contrasts,
        recovery_replicates,
    )
    prediction_sensitivity = _prediction_threshold_sensitivity(run_summary)
    recovery_sensitivity = _recovery_threshold_sensitivity(
        recovery_replicates
    )
    recovery_rates, recovery_confusion, recovery_margins = _benchmark_tables(
        benchmark_paths
    )
    runtime = _runtime_summary(run_summary)

    atomic_write_csv(
        grid,
        output / "manuscript_tripartite_design_grid.csv",
    )
    atomic_write_csv(
        profile,
        output / "manuscript_model_comparison_by_profile.csv",
    )
    atomic_write_csv(
        environment,
        output / "manuscript_model_comparison_by_environment.csv",
    )
    atomic_write_csv(
        phenotype_environment_events,
        output / "manuscript_phenotype_environment_by_events.csv",
    )
    atomic_write_csv(
        information_designs,
        output / "manuscript_model_performance_by_information.csv",
    )
    atomic_write_csv(
        missingness,
        output / "manuscript_missingness_effects.csv",
    )
    atomic_write_csv(
        cells,
        output / "supplement_prediction_recovery_by_condition.csv",
    )
    atomic_write_csv(
        prediction_sensitivity,
        output / "supplement_prediction_threshold_sensitivity.csv",
    )
    atomic_write_csv(
        recovery_sensitivity,
        output / "supplement_recovery_threshold_sensitivity.csv",
    )
    atomic_write_csv(
        recovery_rates,
        output / "supplement_model_recovery_rates.csv",
    )
    atomic_write_csv(
        recovery_confusion,
        output / "supplement_model_recovery_confusion.csv",
    )
    atomic_write_csv(
        recovery_margins,
        output / "supplement_model_recovery_margins.csv",
    )
    atomic_write_csv(
        runtime,
        output / "supplement_runtime_summary.csv",
    )

    _plot_design_grid(
        grid,
        output / "figure_design_adequacy_heatmaps",
    )
    _plot_phenotype_environment_by_events(
        phenotype_environment_events,
        output / "figure_phenotype_environment_adequacy_by_events",
    )
    _plot_information_model_comparison(
        information_designs,
        output / "figure_model_comparison_by_information",
    )
    _plot_missingness(
        missingness,
        output / "figure_missingness_effects",
    )
    _plot_model_recovery(
        recovery_rates,
        output / "figure_model_recovery",
    )
    report = _interpretation_report(
        audit,
        fit_status,
        grid,
        profile,
        environment,
        missingness,
        recovery_rates,
    )
    _write_text(output / "PDTRT_PRODUCTION_RESULTS_INTERPRETATION.md", report)
    atomic_write_json(
        output / "manuscript_results_manifest.json",
        {
            "status": "complete",
            "main_outdir": str(main),
            "benchmark_outdirs": [str(path) for path in benchmark_paths],
            "main_audit_status": audit["status"],
            "main_analysis_rows": audit["observed_analysis_rows"],
            "main_dataset_count": audit["observed_dataset_count"],
            "benchmark_dataset_count": int(
                recovery_rates["replicate_count"].sum()
            ),
            "figures": 5,
            "tables": 12,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare PDTRT production results for manuscript writing."
    )
    parser.add_argument("--main-outdir", required=True)
    parser.add_argument(
        "--benchmark-outdirs",
        nargs="+",
        required=True,
    )
    parser.add_argument("--outdir")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
