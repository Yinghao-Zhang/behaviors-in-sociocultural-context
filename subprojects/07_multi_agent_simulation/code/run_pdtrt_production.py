from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from pdtrt_rerun_core import (
    atomic_write_json,
    generator_constants_fingerprint,
    generator_constants_payload,
)


def _csv(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("status") != "frozen":
        raise ValueError("Production manifest is not frozen")
    expected_constants = manifest["generator_constants"]
    current_constants = generator_constants_payload()
    if _canonical(expected_constants) != _canonical(current_constants):
        raise ValueError(
            "Current generator constants differ from the frozen production manifest"
        )
    expected_fingerprint = str(
        manifest["generator_constants_fingerprint"]
    )
    current_fingerprint = generator_constants_fingerprint()
    if expected_fingerprint != current_fingerprint:
        raise ValueError(
            "Current generator-constant fingerprint differs from the frozen manifest"
        )


def _build_command(
    manifest: Mapping[str, object],
    outdir: Path,
    max_parent_conditions: int | None,
    stage: str,
    workers: int,
) -> list[str]:
    design = manifest["design"]
    estimation = manifest["estimation"]
    constants = manifest["generator_constants"]["rerun_config_defaults"]
    runner = Path(__file__).with_name("run_pdtrt_rerun.py")
    command = [
        sys.executable,
        "-B",
        str(runner),
        "--outdir",
        str(outdir),
        "--stage",
        stage,
        "--workers",
        str(workers),
        "--profiles",
        _csv(design["profiles"]),
        "--environments",
        _csv(design["environments"]),
        "--generator-model",
        str(design["generator_model"]),
        "--event-means",
        _csv(design["mean_events"]),
        "--sample-sizes",
        _csv(design["sample_sizes"]),
        "--missing-rates",
        _csv(design["missing_rates"]),
        "--models",
        _csv(design["candidate_models"]),
        "--estimator",
        str(estimation["primary_estimator"]),
        "--reps",
        str(design["replicates"]),
        "--seed",
        str(design["base_seed"]),
        "--days",
        str(constants["days"]),
        "--network-size",
        str(constants["network_size"]),
        "--mean-degree",
        str(constants["mean_degree"]),
        "--homophily-scale",
        str(constants["homophily_scale"]),
        "--social-foci",
        str(constants["n_social_foci"]),
        "--hidden-events-per-person-day",
        str(constants["hidden_events_per_person_day"]),
        "--event-dispersion",
        str(constants["event_dispersion"]),
        "--baseline-report-sd",
        str(constants["baseline_report_sd"]),
        "--outcome-report-sd",
        str(constants["outcome_report_sd"]),
        "--relationship-report-sd",
        str(constants["relationship_report_sd"]),
        "--suggestion-report-sd",
        str(constants["suggestion_report_sd"]),
        "--feedback-report-sd",
        str(constants["feedback_report_sd"]),
        "--baseline-reliability",
        str(estimation["baseline_reliability"]),
        "--fit-max-iter",
        str(estimation["fit_max_iter"]),
        "--multistarts",
        str(estimation["production_multistarts"]),
        "--pilot-version",
        str(manifest["version"]),
        "--resume",
    ]
    if max_parent_conditions is not None:
        command.extend(
            ["--max-parent-conditions", str(max_parent_conditions)]
        )
    if stage == "fit":
        command.append("--continue-on-error")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the frozen PDTRT production simulation."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[1]
            / "config"
            / "pdtrt_production_v1.json"
        ),
    )
    parser.add_argument("--max-parent-conditions", type=int)
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "fit"),
        default="all",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest)
    outdir = Path(args.outdir).expanduser().resolve()
    command = _build_command(
        manifest,
        outdir,
        args.max_parent_conditions,
        args.stage,
        args.workers,
    )

    if args.dry_run:
        print(shlex.join(command))
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(outdir / "frozen_production_manifest.json", manifest)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
