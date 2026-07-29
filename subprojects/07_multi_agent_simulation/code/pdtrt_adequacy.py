"""Shared manuscript-facing prediction and shared-parameter accuracy criteria."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
import re
from typing import Dict, Mapping

import numpy as np
import pandas as pd


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


@dataclass(frozen=True)
class AdequacyThresholds:
    replicate_pass_fraction_minimum: float = 0.80
    relative_log_loss_skill_minimum: float = 0.05
    mean_ece_maximum: float = 0.10
    replicate_ece_maximum: float = 0.15
    optimizer_success_rate_minimum: float = 0.80
    unit_interval_class_error_maximum: float = 0.10
    choice_consistency_log_error_maximum: float = math.log(1.25)

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object] | None,
    ) -> "AdequacyThresholds":
        if not manifest:
            return cls()
        values = dict(manifest.get("adequacy_thresholds", {}))
        defaults = cls()
        return cls(
            replicate_pass_fraction_minimum=float(
                values.get(
                    "replicate_pass_fraction_minimum",
                    defaults.replicate_pass_fraction_minimum,
                )
            ),
            relative_log_loss_skill_minimum=float(
                values.get(
                    "relative_log_loss_skill_minimum",
                    defaults.relative_log_loss_skill_minimum,
                )
            ),
            mean_ece_maximum=float(
                values.get("mean_ece_maximum", defaults.mean_ece_maximum)
            ),
            replicate_ece_maximum=float(
                values.get(
                    "replicate_ece_maximum",
                    defaults.replicate_ece_maximum,
                )
            ),
            optimizer_success_rate_minimum=float(
                values.get(
                    "optimizer_success_rate_minimum",
                    defaults.optimizer_success_rate_minimum,
                )
            ),
            unit_interval_class_error_maximum=float(
                values.get(
                    "unit_interval_class_error_maximum",
                    defaults.unit_interval_class_error_maximum,
                )
            ),
            choice_consistency_log_error_maximum=float(
                values.get(
                    "choice_consistency_log_error_maximum",
                    defaults.choice_consistency_log_error_maximum,
                )
            ),
        )


DEFAULT_THRESHOLDS = AdequacyThresholds()
REPLICATE_PASS_FRACTION_MINIMUM = (
    DEFAULT_THRESHOLDS.replicate_pass_fraction_minimum
)
RELATIVE_LOG_LOSS_SKILL_MINIMUM = (
    DEFAULT_THRESHOLDS.relative_log_loss_skill_minimum
)
MEAN_ECE_MAXIMUM = DEFAULT_THRESHOLDS.mean_ece_maximum
REPLICATE_ECE_MAXIMUM = DEFAULT_THRESHOLDS.replicate_ece_maximum
UNIT_INTERVAL_CLASS_ERROR_MAXIMUM = (
    DEFAULT_THRESHOLDS.unit_interval_class_error_maximum
)
CHOICE_CONSISTENCY_LOG_ERROR_MAXIMUM = (
    DEFAULT_THRESHOLDS.choice_consistency_log_error_maximum
)


def expected_recovery_count(design: Mapping[str, object]) -> int:
    return int(
        len(design["profiles"])
        * len(design["environments"])
        * len(design["mean_events"])
        * len(design["sample_sizes"])
        * len(design["missing_rates"])
        * int(design["replicates"])
    )


def prediction_diagnostics(
    run_summary: pd.DataFrame,
    *,
    expected_replicates: int,
    thresholds: AdequacyThresholds = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
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
            >= thresholds.optimizer_success_rate_minimum
        )
        skill = group["relative_log_loss_skill"].to_numpy(dtype=float)
        ece = group["metric_ece"].to_numpy(dtype=float)
        row = dict(zip(CELL_KEYS, keys))
        row.update(
            {
                "replicate_count": len(group),
                "complete_replicates": len(group) == expected_replicates,
                "finite_prediction_fraction": float(np.mean(finite)),
                "optimizer_pass_fraction": float(np.mean(optimizer)),
                "mean_log_loss": float(group["metric_log_loss"].mean()),
                "mean_base_rate_log_loss": float(
                    group["base_rate_log_loss"].mean()
                ),
                "mean_relative_log_loss_skill": float(np.mean(skill)),
                "relative_skill_pass_fraction": float(
                    np.mean(
                        skill
                        >= thresholds.relative_log_loss_skill_minimum
                    )
                ),
                "mean_ece": float(np.mean(ece)),
                "ece_stability_pass_fraction": float(
                    np.mean(ece <= thresholds.replicate_ece_maximum)
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
            >= thresholds.replicate_pass_fraction_minimum
            and row["optimizer_pass_fraction"]
            >= thresholds.replicate_pass_fraction_minimum
            and row["mean_relative_log_loss_skill"]
            >= thresholds.relative_log_loss_skill_minimum
            and row["relative_skill_pass_fraction"]
            >= thresholds.replicate_pass_fraction_minimum
            and row["mean_ece"] <= thresholds.mean_ece_maximum
            and row["ece_stability_pass_fraction"]
            >= thresholds.replicate_pass_fraction_minimum
        )
        rows.append(row)
    return pd.DataFrame(rows)


def read_population_recovery(
    main: Path,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
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

        required = {
            "w_i",
            "w_e",
            "w_u",
            "alpha_i_pos",
            "alpha_i_neg",
            "alpha_e",
            "alpha_u",
            "kappa_suggestion",
            "kappa_feedback",
            "choice_consistency",
        }
        missing_parameters = sorted(required - set(parameters))
        if missing_parameters:
            raise ValueError(
                f"{path} is missing recovery parameters: "
                f"{missing_parameters}"
            )
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
        social_error = float(
            np.mean(
                [
                    parameters[name]["absolute_error"]
                    for name in (
                        "kappa_suggestion",
                        "kappa_feedback",
                    )
                ]
            )
        )
        choice_true = parameters["choice_consistency"]["true_mean"]
        choice_estimate = parameters["choice_consistency"]["estimate"]
        if choice_true <= 0 or choice_estimate <= 0:
            choice_error = np.inf
        else:
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
                "social_integration_scaled_error": social_error,
                "choice_consistency_scaled_error": choice_error,
            }
        )
    recovery = pd.DataFrame(rows)
    if expected_count is not None and len(recovery) != expected_count:
        raise ValueError(
            f"Expected {expected_count} tripartite recovery files, found "
            f"{len(recovery)}."
        )
    return recovery


def population_recovery_diagnostics(
    recovery: pd.DataFrame,
    *,
    thresholds: AdequacyThresholds = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    class_thresholds = {
        "decision_weight": thresholds.unit_interval_class_error_maximum,
        "learning_rate": thresholds.unit_interval_class_error_maximum,
        "social_integration": thresholds.unit_interval_class_error_maximum,
        "choice_consistency": (
            thresholds.choice_consistency_log_error_maximum
        ),
    }
    rows = []
    for keys, group in recovery.groupby(list(CELL_KEYS), dropna=False):
        row = dict(zip(CELL_KEYS, keys))
        row["recovery_replicate_count"] = len(group)
        class_flags = []
        for class_name, threshold in class_thresholds.items():
            values = group[f"{class_name}_scaled_error"].to_numpy(
                dtype=float
            )
            mean_error = float(np.mean(values))
            pass_fraction = float(np.mean(values <= threshold))
            class_pass = bool(
                np.isfinite(mean_error)
                and mean_error <= threshold
                and pass_fraction
                >= thresholds.replicate_pass_fraction_minimum
            )
            row[f"mean_{class_name}_scaled_error"] = mean_error
            row[f"{class_name}_pass_fraction"] = pass_fraction
            row[f"{class_name}_recovery"] = class_pass
            class_flags.append(class_pass)
        row["full_vector_recovery"] = bool(all(class_flags))
        rows.append(row)
    return pd.DataFrame(rows)


def cell_diagnostics(
    main: Path,
    run_summary: pd.DataFrame,
    *,
    expected_replicates: int,
    expected_recovery_files: int | None = None,
    thresholds: AdequacyThresholds = DEFAULT_THRESHOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction = prediction_diagnostics(
        run_summary,
        expected_replicates=expected_replicates,
        thresholds=thresholds,
    )
    recovery_replicates = read_population_recovery(
        main,
        expected_count=expected_recovery_files,
    )
    recovery = population_recovery_diagnostics(
        recovery_replicates,
        thresholds=thresholds,
    )
    cells = prediction.merge(
        recovery,
        on=list(CELL_KEYS),
        how="inner",
        validate="one_to_one",
    )
    return cells, recovery_replicates


def manifest_adequacy_thresholds(
    thresholds: AdequacyThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, float]:
    return {
        "replicate_pass_fraction_minimum": (
            thresholds.replicate_pass_fraction_minimum
        ),
        "relative_log_loss_skill_minimum": (
            thresholds.relative_log_loss_skill_minimum
        ),
        "mean_ece_maximum": thresholds.mean_ece_maximum,
        "replicate_ece_maximum": thresholds.replicate_ece_maximum,
        "optimizer_success_rate_minimum": (
            thresholds.optimizer_success_rate_minimum
        ),
        "unit_interval_class_error_maximum": (
            thresholds.unit_interval_class_error_maximum
        ),
        "choice_consistency_log_error_maximum": (
            thresholds.choice_consistency_log_error_maximum
        ),
    }
