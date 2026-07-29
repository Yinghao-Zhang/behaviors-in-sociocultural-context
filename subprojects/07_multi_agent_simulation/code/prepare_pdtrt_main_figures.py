from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pdtrt_rerun_core import atomic_write_csv, atomic_write_json
from prepare_pdtrt_manuscript_results import (
    _cell_diagnostics,
    _design_grid,
    _information_design_summary,
    _phenotype_environment_event_summary,
    _plot_design_grid,
    _plot_information_model_comparison,
    _plot_phenotype_environment_by_events,
)


def run(args: argparse.Namespace) -> int:
    main = Path(args.main_outdir).expanduser().resolve()
    output = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else main / "manuscript_results"
    )
    audit = json.loads((main / "production_run_audit.json").read_text())
    if audit["status"] != "complete":
        raise RuntimeError("Production audit is not complete")
    if int(audit["failed_or_missing_model_rows"]) != 0:
        raise RuntimeError("Production audit contains failed model rows")

    run_summary = pd.read_csv(main / "run_summary.csv")
    cells, _ = _cell_diagnostics(main, run_summary)
    design_grid = _design_grid(cells)
    phenotype_environment = _phenotype_environment_event_summary(cells)
    information_models = _information_design_summary(run_summary)

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        design_grid,
        output / "figure_2_design_adequacy_data.csv",
    )
    atomic_write_csv(
        phenotype_environment,
        output / "figure_3_phenotype_environment_data.csv",
    )
    atomic_write_csv(
        information_models,
        output / "figure_4_model_comparison_data.csv",
    )

    _plot_design_grid(
        design_grid,
        output / "figure_2_design_adequacy",
    )
    _plot_phenotype_environment_by_events(
        phenotype_environment,
        output / "figure_3_phenotype_environment",
    )
    _plot_information_model_comparison(
        information_models,
        output / "figure_4_model_comparison",
    )
    atomic_write_json(
        output / "main_figure_manifest.json",
        {
            "status": "complete",
            "source_outdir": str(main),
            "source_analysis_rows": int(audit["observed_analysis_rows"]),
            "source_dataset_count": int(audit["observed_dataset_count"]),
            "figures": {
                "2": {
                    "stem": "figure_2_design_adequacy",
                    "data_rows": int(len(design_grid)),
                },
                "3": {
                    "stem": "figure_3_phenotype_environment",
                    "data_rows": int(len(phenotype_environment)),
                },
                "4": {
                    "stem": "figure_4_model_comparison",
                    "data_rows": int(len(information_models)),
                },
            },
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create main-manuscript figures from a completed PDTRT run."
    )
    parser.add_argument("--main-outdir", required=True)
    parser.add_argument("--outdir")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
