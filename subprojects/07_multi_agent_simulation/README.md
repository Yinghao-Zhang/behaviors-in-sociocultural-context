# Project 07: Computational Modeling in Ambulatory Assessment

Project 07 contains the code used to simulate and evaluate an
ambulatory-assessment (AA) study of approach and avoidance during interpersonal
conflict. The workflow evaluates whether different study designs provide
enough information for held-out behavioral prediction, shared-parameter
accuracy, and comparison of theoretically distinct learning models.

This directory contains code and configuration only. Generated datasets,
fitted models, result summaries, figures, and manuscript files are intentionally
excluded from the public repository.

## Directory

### Configuration

- `config/pdtrt_production_v2.json`: frozen revised production design,
  generator constants, estimation settings, adequacy criteria, and runtime
  projection.
- `config/pdtrt_production_v1.json`: superseded contingency-based production
  design retained for provenance. Its status prevents accidental reuse with
  the revised generator.
- `requirements.txt`: Python packages and versions used for the production
  analysis.

### Main factorial workflow

- `code/pdtrt_rerun_core.py`: dynamic social-network generator, simulated AA
  protocol, person-behavior phenotypes, social-environment profiles, nested
  participant samples, and nested missing-event masks. It imports the social
  integration and observer-penalty equations from the
  repository-level `social_influence.py` module rather than redefining them.
- `code/pdtrt_rerun_fit.py`: population estimation, candidate models,
  sequential held-out prediction, and parameter-recovery metrics. Suggestion
  and feedback integration weights are estimated separately.
- `code/run_pdtrt_rerun.py`: resumable condition-level runner for the factorial
  design.
- `code/run_pdtrt_production.py`: production launcher that validates the frozen
  configuration before running or resuming the analysis.
- `code/summarize_pdtrt_production.py`: completeness checks and aggregate
  design, prediction, recovery, and paired-comparison tables.
- `code/pdtrt_adequacy.py`: shared manuscript and production criteria for
  prediction and shared-parameter accuracy.
- `code/prepare_pdtrt_manuscript_results.py`: figure- and table-generation
  layer for the main and supplemental displays.
- `code/prepare_pdtrt_main_figures.py`: creates the three numbered results
  figures from a completed production run.
- `code/freeze_pdtrt_production_v2.py`: records the audited constants in the
  version-2 production manifest.

### Model-recovery workflows

- `code/run_pdtrt_model_recovery_benchmark.py`: bidirectional benchmark in
  which each candidate model can generate and fit data.
- `code/run_pdtrt_conditional_recovery.py`: parameter-block recovery with the
  remaining parameter blocks fixed at their generating values.
- `code/run_pdtrt_person_recovery.py`: participant-specific recovery using the
  version-2 generator. It compares a full individual process model,
  decision-weight-only estimation with shared nuisance parameters, and an
  oracle-nuisance upper bound.

### Revision status

The targeted-sensitivity scripts used the superseded context-specific,
single-social-weight generator. They are retained for provenance and support
the earlier design-development checks identified as such in the supplemental
materials. They must not be interpreted as extensions of the final version-2
factorial generator. Future sensitivity checks should be built on
`pdtrt_rerun_core.py` and `pdtrt_rerun_fit.py`.

### Checks

- `code/smoke_pdtrt_rerun.py`: end-to-end checks for sample nesting,
  missing-event nesting, leakage, fitting, and resume behavior.
- `code/smoke_pdtrt_factor_calibration.py`: checks phenotype ordering,
  environmental initial-state and social-signal ordering, and behavioral
  class balance.

## Environment

The production run recorded the following versions:

- Python 3.9.6
- NumPy 1.26.4
- pandas 2.3.3
- SciPy 1.13.1
- scikit-learn 1.6.1
- Matplotlib 3.9.4

From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r subprojects/07_multi_agent_simulation/requirements.txt
```

The commands below assume they are run from the repository root. Output paths
under `generated/` are ignored by Git.

## Smoke Tests

Run the end-to-end workflow check:

```bash
python -B subprojects/07_multi_agent_simulation/code/smoke_pdtrt_rerun.py \
  --outdir subprojects/07_multi_agent_simulation/generated/smoke
```

Check the simulated phenotype and environment factors:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/smoke_pdtrt_factor_calibration.py
```

## Main Factorial Analysis

Run or resume the frozen version-2 design with:

```bash
python -B subprojects/07_multi_agent_simulation/code/run_pdtrt_production.py \
  --outdir subprojects/07_multi_agent_simulation/generated/production \
  --workers 1
```

The superseded version-1 manifest is intentionally rejected by the production
launcher. The full run is computationally intensive and resumable.

Audit and summarize a completed run:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/summarize_pdtrt_production.py \
  --outdir subprojects/07_multi_agent_simulation/generated/production
```

## Bidirectional Model-Recovery Benchmark

The default scenarios reproduce the low-, intermediate-, and high-information
benchmark designs. Run the benchmark for the balanced, affect-oriented, and
outcome-oriented person-behavior phenotypes:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/run_pdtrt_model_recovery_benchmark.py \
  --outdir subprojects/07_multi_agent_simulation/generated/model_recovery_balanced \
  --profile balanced \
  --reps 20 \
  --workers 1

python -B \
  subprojects/07_multi_agent_simulation/code/run_pdtrt_model_recovery_benchmark.py \
  --outdir subprojects/07_multi_agent_simulation/generated/model_recovery_affect \
  --profile relief_reactive \
  --reps 20 \
  --workers 1

python -B \
  subprojects/07_multi_agent_simulation/code/run_pdtrt_model_recovery_benchmark.py \
  --outdir subprojects/07_multi_agent_simulation/generated/model_recovery_outcome \
  --profile consequence_sensitive \
  --reps 20 \
  --workers 1
```

## Participant-Specific Recovery

After generating the version-2 production datasets, the exploratory
participant-specific recovery grid can be run without repeating data
generation:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/run_pdtrt_person_recovery.py \
  --source-root subprojects/07_multi_agent_simulation/generated/production \
  --outdir subprojects/07_multi_agent_simulation/generated/person_recovery \
  --profiles balanced,rigid_habitual,relief_reactive,consequence_sensitive,socially_contingent \
  --environments mixed \
  --mean-events 10,25,50 \
  --replicates 3 \
  --sample-size 100 \
  --modes full_process,decision_weights,decision_weights_oracle
```

## Preparing Tables and Figures

After completing the main run and the three model-recovery benchmarks, generate
the display package:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/prepare_pdtrt_manuscript_results.py \
  --main-outdir subprojects/07_multi_agent_simulation/generated/production \
  --benchmark-outdirs \
    subprojects/07_multi_agent_simulation/generated/model_recovery_balanced \
    subprojects/07_multi_agent_simulation/generated/model_recovery_affect \
    subprojects/07_multi_agent_simulation/generated/model_recovery_outcome
```

Use `--help` on any entry point for the complete set of options. Each production
runner records its configuration, seeds, software versions, status, and
checksums in the selected output directory.

## Citation

If you use the Project 07 code, cite the software using the repository-level
[`CITATION.cff`](../../CITATION.cff). A preferred citation for the associated
article can be added to that file after publication.
