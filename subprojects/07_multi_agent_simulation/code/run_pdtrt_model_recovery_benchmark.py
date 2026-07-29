from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pdtrt_rerun_core import GENERATOR_MODELS, atomic_write_csv, atomic_write_json
from pdtrt_rerun_fit import CANDIDATE_MODELS


DEFAULT_SCENARIOS = "low:30:10:0.2,middle:60:25:0.1,high:100:50:0"


def _parse_scenarios(value: str) -> List[Dict[str, object]]:
    scenarios = []
    for item in value.split(","):
        name, sample_size, mean_events, missing_rate = item.strip().split(":")
        scenarios.append(
            {
                "scenario": name,
                "sample_size": int(sample_size),
                "mean_events": float(mean_events),
                "missing_rate": float(missing_rate),
            }
        )
    return scenarios


def _run_condition(
    runner: Path,
    outdir: Path,
    generator_model: str,
    scenario: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        "-B",
        str(runner),
        "--outdir",
        str(outdir),
        "--profiles",
        args.profile,
        "--environments",
        args.environment,
        "--generator-model",
        generator_model,
        "--event-means",
        str(scenario["mean_events"]),
        "--sample-sizes",
        str(scenario["sample_size"]),
        "--missing-rates",
        str(scenario["missing_rate"]),
        "--models",
        ",".join(CANDIDATE_MODELS),
        "--estimator",
        args.estimator,
        "--reps",
        str(args.reps),
        "--seed",
        str(args.seed),
        "--days",
        str(args.days),
        "--network-size",
        str(args.network_size),
        "--mean-degree",
        str(args.mean_degree),
        "--homophily-scale",
        str(args.homophily_scale),
        "--social-foci",
        str(args.social_foci),
        "--hidden-events-per-person-day",
        str(args.hidden_events_per_person_day),
        "--eb-iterations",
        str(args.eb_iterations),
        "--eb-variance-update",
        args.eb_variance_update,
        "--fit-max-iter",
        str(args.fit_max_iter),
        "--multistarts",
        str(args.multistarts),
        "--pilot-version",
        args.pilot_version,
        "--resume",
    ]
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> int:
    scenarios = _parse_scenarios(args.scenarios)
    generator_models = [
        model.strip() for model in args.generator_models.split(",") if model.strip()
    ]
    unknown = sorted(set(generator_models) - set(GENERATOR_MODELS))
    if unknown:
        raise ValueError(f"Unknown generator models: {unknown}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_pdtrt_rerun.py")
    summary_frames = []
    conditions = [
        (scenario, generator_model)
        for scenario in scenarios
        for generator_model in generator_models
    ]

    def collect(
        scenario: Dict[str, object],
        generator_model: str,
    ) -> None:
        condition_dir = (
            outdir
            / "runs"
            / f"scenario={scenario['scenario']}"
            / f"generator={generator_model}"
        )
        frame = pd.read_csv(
            condition_dir / "run_summary.csv",
            keep_default_na=False,
            na_values=[""],
        )
        frame["scenario"] = scenario["scenario"]
        frame["generating_model"] = generator_model
        summary_frames.append(frame)

    if args.workers == 1:
        for scenario, generator_model in conditions:
            condition_dir = (
                outdir
                / "runs"
                / f"scenario={scenario['scenario']}"
                / f"generator={generator_model}"
            )
            _run_condition(
                runner,
                condition_dir,
                generator_model,
                scenario,
                args,
            )
            collect(scenario, generator_model)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_condition,
                    runner,
                    outdir
                    / "runs"
                    / f"scenario={scenario['scenario']}"
                    / f"generator={generator_model}",
                    generator_model,
                    scenario,
                    args,
                ): (scenario, generator_model)
                for scenario, generator_model in conditions
            }
            for future in as_completed(futures):
                scenario, generator_model = futures[future]
                future.result()
                collect(scenario, generator_model)

    summary = pd.concat(summary_frames, ignore_index=True)
    atomic_write_csv(summary, outdir / "benchmark_run_summary.csv")

    candidates = summary.loc[summary["model"].isin(CANDIDATE_MODELS)].copy()
    grouping = [
        "scenario",
        "generating_model",
        "profile",
        "environment",
        "replicate",
        "sample_size",
        "missing_rate",
    ]
    winner_indices = candidates.groupby(grouping)["metric_log_loss"].idxmin()
    selections = candidates.loc[
        winner_indices,
        grouping + ["model", "metric_log_loss"],
    ].rename(columns={"model": "selected_model"})
    selections["correct_selection"] = (
        selections["selected_model"] == selections["generating_model"]
    )
    atomic_write_csv(selections, outdir / "model_selections.csv")

    confusion = (
        selections.groupby(
            ["scenario", "generating_model", "selected_model"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    totals = confusion.groupby(
        ["scenario", "generating_model"]
    )["count"].transform("sum")
    confusion["selection_rate"] = confusion["count"] / totals
    atomic_write_csv(confusion, outdir / "model_selection_confusion.csv")

    false_selection = (
        selections.groupby(["scenario", "generating_model"])
        .agg(
            replicate_count=("correct_selection", "size"),
            correct_selection_rate=("correct_selection", "mean"),
        )
        .reset_index()
    )
    false_selection["false_selection_rate"] = (
        1.0 - false_selection["correct_selection_rate"]
    )
    atomic_write_csv(false_selection, outdir / "false_selection_rates.csv")
    atomic_write_json(
        outdir / "benchmark_manifest.json",
        {
            "status": "complete",
            "scenarios": scenarios,
            "generator_models": generator_models,
            "candidate_models": list(CANDIDATE_MODELS),
            "estimator": args.estimator,
            "profile": args.profile,
            "environment": args.environment,
            "replicates": args.reps,
            "selection_rule": "minimum held-out sequential log loss",
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the targeted bidirectional PDTRT model-recovery benchmark."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--generator-models", default=",".join(GENERATOR_MODELS))
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--environment", default="mixed")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--network-size", type=int, default=1000)
    parser.add_argument("--mean-degree", type=float, default=12.0)
    parser.add_argument("--homophily-scale", type=float, default=0.55)
    parser.add_argument("--social-foci", type=int, default=12)
    parser.add_argument("--hidden-events-per-person-day", type=float, default=0.10)
    parser.add_argument("--eb-iterations", type=int, default=3)
    parser.add_argument(
        "--eb-variance-update",
        choices=("laplace", "modal"),
        default="laplace",
    )
    parser.add_argument(
        "--estimator",
        choices=("empirical_bayes", "population"),
        default="population",
    )
    parser.add_argument("--fit-max-iter", type=int, default=120)
    parser.add_argument("--multistarts", type=int, default=2)
    parser.add_argument("--workers", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--pilot-version", default="pilot_v1")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
