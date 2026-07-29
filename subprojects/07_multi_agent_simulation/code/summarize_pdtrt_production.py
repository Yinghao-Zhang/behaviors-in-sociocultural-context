from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from pdtrt_rerun_core import atomic_write_csv, atomic_write_json
from pdtrt_adequacy import (
    AdequacyThresholds,
    cell_diagnostics,
    expected_recovery_count,
)


DESIGN_KEYS = [
    "profile",
    "environment",
    "mean_events",
    "sample_size",
    "missing_rate",
]
REPLICATE_KEYS = [*DESIGN_KEYS, "replicate"]


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = []
    for column in frame.columns:
        if isinstance(column, tuple):
            flattened.append(
                "_".join(str(part) for part in column if str(part))
            )
        else:
            flattened.append(str(column))
    frame.columns = flattened
    return frame


def _cell_summaries(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in summary.columns
        if column.startswith("metric_")
        or column
        in {
            "population_mean_abs_bias",
            "diagnostic_optimizer_success_rate",
            "diagnostic_runtime_seconds",
        }
    ]
    grouped = summary.groupby([*DESIGN_KEYS, "model"], dropna=False)
    cells = grouped[metrics].agg(
        ["mean", "std", "median", "min", "max"]
    ).reset_index()
    cells = _flatten_columns(cells)
    cells["replicate_count"] = grouped.size().to_numpy()
    for metric in metrics:
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"
        if mean_column not in cells or std_column not in cells:
            continue
        standard_error = cells[std_column] / np.sqrt(
            cells["replicate_count"]
        )
        cells[f"{metric}_ci95_low"] = (
            cells[mean_column] - 1.96 * standard_error
        )
        cells[f"{metric}_ci95_high"] = (
            cells[mean_column] + 1.96 * standard_error
        )
    return cells


def _paired_model_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index=REPLICATE_KEYS,
        columns="model",
        values="metric_log_loss",
        aggfunc="first",
    ).reset_index()
    for model in (
        "tripartite",
        "no_learning",
        "collapsed_reward",
        "lagged",
    ):
        wide[f"{model}_improvement_over_prevalence"] = (
            wide["prevalence_null"] - wide[model]
        )
    wide["tripartite_improvement_over_no_learning"] = (
        wide["no_learning"] - wide["tripartite"]
    )
    wide["collapsed_minus_tripartite"] = (
        wide["collapsed_reward"] - wide["tripartite"]
    )
    return wide


def _comparison_summaries(
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    delta_columns = [
        column
        for column in comparisons
        if "improvement_" in column or column == "collapsed_minus_tripartite"
    ]
    rows = []
    for keys, group in comparisons.groupby(DESIGN_KEYS, dropna=False):
        row = dict(zip(DESIGN_KEYS, keys))
        row["replicate_count"] = len(group)
        for column in delta_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sd"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else np.nan
            )
            row[f"{column}_positive_fraction"] = float(
                np.mean(values > 0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _design_adequacy(
    outdir: Path,
    summary: pd.DataFrame,
    manifest: Dict[str, object],
    *,
    require_complete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = AdequacyThresholds.from_manifest(manifest)
    expected_replicates = int(manifest["design"]["replicates"])
    cells, recovery_replicates = cell_diagnostics(
        outdir,
        summary,
        expected_replicates=expected_replicates,
        expected_recovery_files=(
            expected_recovery_count(manifest["design"])
            if require_complete
            else None
        ),
        thresholds=thresholds,
    )
    cells["model"] = "tripartite"
    cells["adequate_prediction"] = cells[
        "minimum_predictive_signal"
    ].astype(bool)
    cells["population_recovery_target_matches_generator"] = (
        manifest["design"]["generator_model"] == "tripartite"
    )
    cells["adequate_population_interpretation"] = (
        cells["adequate_prediction"]
        & cells["population_recovery_target_matches_generator"]
        & cells["full_vector_recovery"]
    )
    return cells, recovery_replicates


def _design_frontiers(
    adequacy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier_rows = []
    recruitment_first_rows = []
    priority_rows = []
    for goal, flag in (
        ("prediction", "adequate_prediction"),
        (
            "population_interpretation",
            "adequate_population_interpretation",
        ),
    ):
        eligible = adequacy.loc[adequacy[flag]].copy()
        grouping = ["profile", "environment", "missing_rate", "model"]
        for keys, group in eligible.groupby(grouping, dropna=False):
            group = group.sort_values(
                ["sample_size", "mean_events"]
            ).copy()
            is_frontier = []
            for _, row in group.iterrows():
                dominated = (
                    (group["sample_size"] <= row["sample_size"])
                    & (group["mean_events"] <= row["mean_events"])
                    & (
                        (group["sample_size"] < row["sample_size"])
                        | (group["mean_events"] < row["mean_events"])
                    )
                ).any()
                is_frontier.append(not bool(dominated))
            frontier = group.loc[is_frontier].copy()
            frontier["inferential_goal"] = goal
            frontier_rows.append(frontier)
            if not frontier.empty:
                recruitment_first = frontier.sort_values(
                    ["sample_size", "mean_events"]
                ).iloc[[0]].copy()
                recruitment_first["inferential_goal"] = goal
                recruitment_first["selection_rule"] = (
                    "minimize_participants_then_events"
                )
                recruitment_first_rows.append(recruitment_first)

                priority_specs = (
                    (
                        "minimize_participants_then_events",
                        ["sample_size", "mean_events"],
                    ),
                    (
                        "minimize_events_then_participants",
                        ["mean_events", "sample_size"],
                    ),
                    (
                        "minimize_total_expected_observations",
                        [
                            "total_expected_observations",
                            "sample_size",
                            "mean_events",
                        ],
                    ),
                )
                candidates = frontier.copy()
                candidates["total_expected_observations"] = (
                    candidates["sample_size"]
                    * candidates["mean_events"]
                )
                for rule, sort_columns in priority_specs:
                    selected = candidates.sort_values(
                        sort_columns
                    ).iloc[[0]].copy()
                    selected["inferential_goal"] = goal
                    selected["selection_rule"] = rule
                    priority_rows.append(selected)
    frontier = (
        pd.concat(frontier_rows, ignore_index=True)
        if frontier_rows
        else adequacy.head(0).assign(
            inferential_goal=pd.Series(dtype=str)
        )
    )
    recruitment_first = (
        pd.concat(recruitment_first_rows, ignore_index=True)
        if recruitment_first_rows
        else adequacy.head(0).assign(
            inferential_goal=pd.Series(dtype=str),
            selection_rule=pd.Series(dtype=str),
        )
    )
    priority_recommendations = (
        pd.concat(priority_rows, ignore_index=True)
        if priority_rows
        else adequacy.head(0).assign(
            inferential_goal=pd.Series(dtype=str),
            selection_rule=pd.Series(dtype=str),
            total_expected_observations=pd.Series(dtype=float),
        )
    )
    return frontier, recruitment_first, priority_recommendations


def _missingness_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    index = [
        "profile",
        "environment",
        "mean_events",
        "sample_size",
        "replicate",
        "model",
    ]
    wide = summary.pivot_table(
        index=index,
        columns="missing_rate",
        values=["metric_log_loss", "population_mean_abs_bias"],
        aggfunc="first",
    )
    wide.columns = [
        f"{metric}_missing_{int(round(100 * missing)):02d}"
        for metric, missing in wide.columns
    ]
    wide = wide.reset_index()
    for metric in ("metric_log_loss", "population_mean_abs_bias"):
        base = f"{metric}_missing_00"
        for missing in (10, 20):
            target = f"{metric}_missing_{missing:02d}"
            if base in wide and target in wide:
                wide[f"{metric}_change_00_to_{missing:02d}"] = (
                    wide[target] - wide[base]
                )
    return wide


def _write_report(
    outdir: Path,
    audit: Dict[str, object],
    adequacy: pd.DataFrame,
    priority_recommendations: pd.DataFrame,
) -> None:
    prediction_count = int(adequacy["adequate_prediction"].sum())
    interpretation_count = int(
        adequacy["adequate_population_interpretation"].sum()
    )
    report = f"""# PDTRT Production Analysis Report

## Run Audit

- Status: {audit["status"]}
- Observed analysis rows: {audit["observed_analysis_rows"]}
- Expected analysis rows: {audit["expected_analysis_rows"]}
- Observed datasets: {audit["observed_dataset_count"]}
- Expected datasets: {audit["expected_dataset_count"]}
- Failed or missing model rows: {audit["failed_or_missing_model_rows"]}

## Adequacy

- Design-model cells adequate for held-out prediction: {prediction_count}
- Design-model cells meeting the shared-parameter accuracy criterion: {interpretation_count}
- Design-priority recommendations produced: {len(priority_recommendations)}

Person-specific learning parameters are not treated as production estimands.
The separate preproduction recovery ladder documents why.

No single design is labeled universally least burdensome. Recommendations are
reported separately when prioritizing enrolled sample size, events per person,
or total expected observations.

## Output Files

- `production_cell_summary.csv`
- `paired_model_comparisons.csv`
- `paired_model_comparison_summary.csv`
- `design_adequacy.csv`
- `population_recovery_replicates.csv`
- `design_pareto_frontier.csv`
- `least_burdensome_adequate_designs.csv`
- `design_priority_recommendations.csv`
- `paired_missingness_contrasts.csv`
- `production_run_audit.json`
"""
    (outdir / "PRODUCTION_ANALYSIS_REPORT.md").write_text(report)


def run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    summary = pd.read_csv(outdir / "run_summary.csv")
    inventory = pd.read_csv(outdir / "condition_inventory.csv")

    design = manifest["design"]
    expected_datasets = (
        len(design["profiles"])
        * len(design["environments"])
        * len(design["mean_events"])
        * len(design["sample_sizes"])
        * len(design["missing_rates"])
        * int(design["replicates"])
    )
    expected_models = len(design["candidate_models"]) + 1
    expected_rows = expected_datasets * expected_models
    observed_rows = len(summary)
    expected_model_names = {
        *design["candidate_models"],
        "prevalence_null",
    }
    expected_keys = pd.MultiIndex.from_tuples(
        product(
            design["profiles"],
            design["environments"],
            [float(value) for value in design["mean_events"]],
            [int(value) for value in design["sample_sizes"]],
            [float(value) for value in design["missing_rates"]],
            range(1, int(design["replicates"]) + 1),
            sorted(expected_model_names),
        ),
        names=[*REPLICATE_KEYS, "model"],
    )
    observed_key_frame = summary[[*REPLICATE_KEYS, "model"]].copy()
    observed_key_frame["mean_events"] = observed_key_frame[
        "mean_events"
    ].astype(float)
    observed_key_frame["sample_size"] = observed_key_frame[
        "sample_size"
    ].astype(int)
    observed_key_frame["missing_rate"] = observed_key_frame[
        "missing_rate"
    ].astype(float)
    observed_key_frame["replicate"] = observed_key_frame[
        "replicate"
    ].astype(int)
    duplicate_key_count = int(observed_key_frame.duplicated().sum())
    observed_keys = pd.MultiIndex.from_frame(
        observed_key_frame.drop_duplicates()
    )
    missing_key_count = len(expected_keys.difference(observed_keys))
    unexpected_key_count = len(observed_keys.difference(expected_keys))
    observed_datasets = len(summary[REPLICATE_KEYS].drop_duplicates())
    complete = (
        observed_rows == expected_rows
        and observed_datasets == expected_datasets
        and duplicate_key_count == 0
        and missing_key_count == 0
        and unexpected_key_count == 0
    )
    if not complete and not args.allow_incomplete:
        raise ValueError(
            f"Production output is incomplete: {observed_rows}/{expected_rows} "
            f"analysis rows and {observed_datasets}/{expected_datasets} datasets"
        )

    cells = _cell_summaries(summary)
    comparisons = _paired_model_comparisons(summary)
    comparison_summary = _comparison_summaries(comparisons)
    adequacy, recovery_replicates = _design_adequacy(
        outdir,
        summary,
        manifest,
        require_complete=complete,
    )
    (
        frontier,
        recruitment_first,
        priority_recommendations,
    ) = _design_frontiers(adequacy)
    missingness = _missingness_contrasts(summary)

    atomic_write_csv(cells, outdir / "production_cell_summary.csv")
    atomic_write_csv(
        comparisons,
        outdir / "paired_model_comparisons.csv",
    )
    atomic_write_csv(
        comparison_summary,
        outdir / "paired_model_comparison_summary.csv",
    )
    atomic_write_csv(adequacy, outdir / "design_adequacy.csv")
    atomic_write_csv(
        recovery_replicates,
        outdir / "population_recovery_replicates.csv",
    )
    atomic_write_csv(frontier, outdir / "design_pareto_frontier.csv")
    atomic_write_csv(
        recruitment_first,
        outdir / "least_burdensome_adequate_designs.csv",
    )
    atomic_write_csv(
        priority_recommendations,
        outdir / "design_priority_recommendations.csv",
    )
    atomic_write_csv(
        missingness,
        outdir / "paired_missingness_contrasts.csv",
    )

    thresholds = AdequacyThresholds.from_manifest(manifest)
    failed_rows = int(
        (
            ~np.isfinite(summary["metric_log_loss"])
            | (
                summary["diagnostic_optimizer_success_rate"]
                < thresholds.optimizer_success_rate_minimum
            )
        ).sum()
    )
    audit = {
        "status": "complete" if complete and failed_rows == 0 else "incomplete",
        "allow_incomplete": bool(args.allow_incomplete),
        "observed_analysis_rows": observed_rows,
        "expected_analysis_rows": expected_rows,
        "observed_dataset_count": observed_datasets,
        "expected_dataset_count": expected_datasets,
        "observed_inventory_rows": len(inventory),
        "duplicate_analysis_key_count": duplicate_key_count,
        "missing_analysis_key_count": missing_key_count,
        "unexpected_analysis_key_count": unexpected_key_count,
        "failed_or_missing_model_rows": failed_rows,
        "models": sorted(summary["model"].unique().tolist()),
        "replicates": sorted(
            summary["replicate"].astype(int).unique().tolist()
        ),
    }
    atomic_write_json(outdir / "production_run_audit.json", audit)
    _write_report(
        outdir,
        audit,
        adequacy,
        priority_recommendations,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and summarize the frozen PDTRT production run."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[1]
            / "config"
            / "pdtrt_production_v2.json"
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
