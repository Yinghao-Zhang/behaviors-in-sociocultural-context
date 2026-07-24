from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pdtrt_rerun_core import (
    atomic_write_csv,
    atomic_write_json,
    build_panel_view,
    config_fingerprint,
    load_simulation_result,
    stable_seed,
)
from pdtrt_rerun_fit import (
    CONDITIONAL_ORACLE_BLOCKS,
    EmpiricalBayesConfig,
    fit_conditional_oracle,
)


def _find_latent_directory(source_run: Path) -> Path:
    matches = sorted(source_run.glob("conditions/**/latent/config.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one latent condition under {source_run}, found {len(matches)}"
        )
    return matches[0].parent


def _parse_blocks(value: str) -> List[str]:
    blocks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(blocks) - set(CONDITIONAL_ORACLE_BLOCKS))
    if unknown:
        raise ValueError(f"Unknown conditional blocks: {unknown}")
    return blocks


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    source_run = Path(args.source_run).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    blocks = _parse_blocks(args.blocks)

    latent_dir = _find_latent_directory(source_run)
    simulation = load_simulation_result(latent_dir)
    sample_size = args.sample_size or simulation.config.max_focal
    view = build_panel_view(simulation, sample_size, 0.0)
    cfg = EmpiricalBayesConfig(
        eb_iterations=args.eb_iterations,
        max_iter=args.fit_max_iter,
        multistarts=args.multistarts,
        variance_update=args.eb_variance_update,
        baseline_reliability=args.baseline_reliability,
        seed=args.seed,
    )

    atomic_write_json(
        outdir / "conditional_recovery_manifest.json",
        {
            "source_run": str(source_run),
            "source_latent_directory": str(latent_dir),
            "source_config_fingerprint": config_fingerprint(simulation.config),
            "sample_size": sample_size,
            "blocks": blocks,
            "empirical_bayes": {
                "iterations": cfg.eb_iterations,
                "max_iter": cfg.max_iter,
                "multistarts": cfg.multistarts,
                "variance_update": cfg.variance_update,
                "baseline_reliability": cfg.baseline_reliability,
            },
            "oracle_rule": (
                "Only the named parameter block is estimated; all other "
                "person parameters are fixed to their generating values."
            ),
        },
    )

    summary_rows: List[Dict[str, object]] = []
    for block in blocks:
        block_dir = outdir / f"block={block}"
        summary_path = block_dir / "summary.json"
        if args.resume and summary_path.exists():
            summary_rows.append(json.loads(summary_path.read_text()))
            continue

        block_started = time.perf_counter()
        block_cfg = EmpiricalBayesConfig(
            **{
                **cfg.__dict__,
                "seed": stable_seed(cfg.seed, block),
            }
        )
        person_parameters, recovery, diagnostics = fit_conditional_oracle(
            view,
            "tripartite",
            block,
            block_cfg,
        )
        runtime = time.perf_counter() - block_started
        summary: Dict[str, object] = {
            "block": block,
            "sample_size": sample_size,
            "mean_events": simulation.config.mean_events,
            "source_hidden_events_per_person_day": (
                simulation.config.hidden_events_per_person_day
            ),
            "source_baseline_report_sd": simulation.config.baseline_report_sd,
            "source_outcome_report_sd": simulation.config.outcome_report_sd,
            "runtime_seconds": runtime,
            **diagnostics,
        }
        block_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(
            person_parameters,
            block_dir / "person_parameters.csv.gz",
        )
        atomic_write_csv(recovery, block_dir / "recovery.csv")
        atomic_write_json(summary_path, summary)
        summary_rows.append(summary)
        atomic_write_csv(
            pd.DataFrame(summary_rows),
            outdir / "conditional_recovery_summary.csv",
        )

    atomic_write_json(
        outdir / "run_status.json",
        {
            "status": "complete",
            "block_count": len(summary_rows),
            "runtime_seconds_current_invocation": time.perf_counter() - started,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run conditional parameter-block recovery against a generated "
            "PDTRT simulation condition."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--blocks",
        default=",".join(CONDITIONAL_ORACLE_BLOCKS),
    )
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--eb-iterations", type=int, default=3)
    parser.add_argument("--fit-max-iter", type=int, default=120)
    parser.add_argument("--multistarts", type=int, default=2)
    parser.add_argument(
        "--eb-variance-update",
        choices=("laplace", "modal"),
        default="laplace",
    )
    parser.add_argument("--baseline-reliability", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
