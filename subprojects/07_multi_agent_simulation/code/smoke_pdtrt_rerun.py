from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


MODELS = (
    "tripartite",
    "no_learning",
    "collapsed_reward",
    "lagged",
    "prevalence_null",
)
SAMPLE_SIZES = (6, 12)
MISSING_RATES = (0, 10, 20)


def _run_smoke(outdir: Path) -> None:
    runner = Path(__file__).with_name("run_pdtrt_rerun.py")
    command = [
        sys.executable,
        "-B",
        str(runner),
        "--outdir",
        str(outdir),
        "--profiles",
        "relief_reactive",
        "--environments",
        "mixed",
        "--event-means",
        "12",
        "--sample-sizes",
        "6,12",
        "--missing-rates",
        "0,0.1,0.2",
        "--models",
        "tripartite,no_learning,collapsed_reward,lagged",
        "--reps",
        "1",
        "--days",
        "7",
        "--network-size",
        "60",
        "--mean-degree",
        "7",
        "--social-foci",
        "5",
        "--hidden-events-per-person-day",
        "0.05",
        "--eb-iterations",
        "2",
        "--fit-max-iter",
        "50",
        "--multistarts",
        "1",
        "--pilot-version",
        "smoke_v1",
        "--write-panel-data",
        "--resume",
    ]
    subprocess.run(command, check=True)


def _condition_root(outdir: Path) -> Path:
    return (
        outdir
        / "conditions"
        / "generator=tripartite"
        / "profile=relief_reactive"
        / "environment=mixed"
        / "events=012"
        / "replicate=001"
    )


def _target_set(predictions: pd.DataFrame) -> Set[Tuple[int, int]]:
    return set(
        zip(
            predictions["focal_id"].astype(int),
            predictions["event_id"].astype(int),
        )
    )


def _validate(outdir: Path) -> Dict[str, object]:
    root = _condition_root(outdir)
    status = json.loads((outdir / "run_status.json").read_text())
    assert status["status"] == "complete", status
    assert status["error_count"] == 0, status
    assert status["panel_count"] == 6, status
    assert status["analysis_count_including_null"] == 30, status

    summary = pd.read_csv(outdir / "run_summary.csv")
    assert len(summary) == 30
    assert set(summary["model"]) == set(MODELS)
    assert set(summary["sample_size"].astype(int)) == set(SAMPLE_SIZES)
    assert set((100 * summary["missing_rate"]).round().astype(int)) == set(MISSING_RATES)

    checks = json.loads((root / "nested_panel_checks.json").read_text())
    assert checks and all(checks.values()), checks

    event_sets: Dict[Tuple[int, int], Set[int]] = {}
    participant_sets: Dict[Tuple[int, int], Set[int]] = {}
    prediction_counts: List[int] = []
    for sample_size in SAMPLE_SIZES:
        model_targets: Dict[int, Set[Tuple[int, int]]] = {}
        for missing in MISSING_RATES:
            view = (
                root
                / "views"
                / f"N={sample_size:03d}"
                / f"missing={missing:02d}"
            )
            events = pd.read_csv(view / "data" / "aa_events.csv")
            people = pd.read_csv(view / "data" / "aa_people.csv")
            forbidden_event_columns = [
                column
                for column in events.columns
                if column.startswith(("true_", "focal_pre_", "focal_post_", "miss_"))
            ]
            forbidden_person_columns = [
                column
                for column in people.columns
                if column.startswith(("true_", "initial_", "missing_"))
            ]
            assert not forbidden_event_columns, forbidden_event_columns
            assert not forbidden_person_columns, forbidden_person_columns
            assert not any("expected_" in column for column in events.columns)
            assert "context_idx" not in events.columns
            assert "context" not in events.columns
            assert {
                "suggestion_active",
                "suggestion_avoid",
                "suggestion_approach",
                "feedback_active",
                "feedback",
                "utility_out",
                "relationship_receptivity",
            }.issubset(events.columns)
            assert (
                events.loc[
                    events["suggestion_active"] == 0,
                    ["suggestion_avoid", "suggestion_approach"],
                ]
                .abs()
                .to_numpy()
                .max(initial=0.0)
                == 0.0
            )
            assert (
                events.loc[events["feedback_active"] == 0, "feedback"]
                .abs()
                .to_numpy()
                .max(initial=0.0)
                == 0.0
            )
            baseline_columns = {
                column
                for column in people.columns
                if column.startswith("baseline_")
            }
            assert baseline_columns == {
                f"baseline_{state}_{behavior}"
                for state in ("instinct", "enjoyment", "utility")
                for behavior in ("avoid", "approach")
            }

            key = (sample_size, missing)
            event_sets[key] = set(events["event_id"].astype(int))
            participant_sets[key] = set(people["focal_id"].astype(int))
            assert len(participant_sets[key]) == sample_size

            target_sets = {}
            for model in MODELS:
                predictions = pd.read_csv(
                    view
                    / "fits"
                    / "estimator=empirical_bayes"
                    / f"model={model}"
                    / "predictions.csv.gz"
                )
                target_sets[model] = _target_set(predictions)
            first_targets = target_sets[MODELS[0]]
            assert first_targets
            assert all(targets == first_targets for targets in target_sets.values())
            model_targets[missing] = first_targets
            prediction_counts.append(len(first_targets))

            tripartite_parameters = pd.read_csv(
                view
                / "fits"
                / "estimator=empirical_bayes"
                / "model=tripartite"
                / "person_parameters.csv.gz"
            )
            assert {
                "fit_tau",
                "fit_noise_s",
                "fit_kappa_suggestion",
                "fit_kappa_feedback",
            }.issubset(tripartite_parameters.columns)

        assert event_sets[(sample_size, 20)].issubset(event_sets[(sample_size, 10)])
        assert event_sets[(sample_size, 10)].issubset(event_sets[(sample_size, 0)])
        assert model_targets[0] == model_targets[10] == model_targets[20]

    for missing in MISSING_RATES:
        assert participant_sets[(6, missing)].issubset(
            participant_sets[(12, missing)]
        )

    latent = root / "latent"
    assert (latent / "truth_events.csv.gz").exists()
    assert (latent / "complete_reports_private.csv.gz").exists()
    diagnostics = json.loads((latent / "generation_diagnostics.json").read_text())
    generation_diagnostics = diagnostics.get(
        "generation_diagnostics",
        diagnostics,
    )
    assert generation_diagnostics["report_event_count"] > 0
    assert generation_diagnostics["hidden_event_count"] > 0

    return {
        "panels": int(status["panel_count"]),
        "analyses_including_null": int(status["analysis_count_including_null"]),
        "all_nested_checks": True,
        "all_prediction_targets_aligned": True,
        "public_data_truth_leakage": False,
        "prediction_count_range": [
            int(min(prediction_counts)),
            int(max(prediction_counts)),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir).expanduser().resolve()

    _run_smoke(outdir)
    fit_summary = (
        _condition_root(outdir)
        / "views"
        / "N=012"
        / "missing=10"
        / "fits"
        / "estimator=empirical_bayes"
        / "model=tripartite"
        / "fit_summary.json"
    )
    first_mtime = fit_summary.stat().st_mtime_ns
    _run_smoke(outdir)
    second_mtime = fit_summary.stat().st_mtime_ns
    assert first_mtime == second_mtime, "Resume unexpectedly refit a completed model"

    result = _validate(outdir)
    result["resume_preserved_completed_fit"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
