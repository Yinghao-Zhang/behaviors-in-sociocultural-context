# Project 07: Computational Modeling in Ambulatory Assessment

Project 07 contains the code used to simulate and evaluate an
ambulatory-assessment (AA) study of approach and avoidance during interpersonal
conflict. The workflow evaluates whether different study designs provide
enough information for held-out behavioral prediction, population-parameter
recovery, and comparison of theoretically distinct learning models.

This directory contains code and configuration only. Generated datasets,
fitted models, result summaries, figures, and manuscript files are intentionally
excluded from the public repository.

## Directory

### Configuration

- `config/pdtrt_production_v1.json`: frozen production design, generator
  constants, estimator settings, thresholds, and staging decisions.
- `requirements.txt`: Python packages and versions used for the production
  analysis.

### Main factorial workflow

- `code/pdtrt_rerun_core.py`: dynamic social-network generator, simulated AA
  protocol, person-behavior phenotypes, social-environment profiles, nested
  participant samples, and nested missing-event masks.
- `code/pdtrt_rerun_fit.py`: population estimation, candidate models,
  sequential held-out prediction, and parameter-recovery metrics.
- `code/run_pdtrt_rerun.py`: resumable condition-level runner for the factorial
  design.
- `code/run_pdtrt_production.py`: production launcher that validates the frozen
  configuration before running or resuming the analysis.
- `code/summarize_pdtrt_production.py`: completeness checks and aggregate
  design, prediction, recovery, and paired-comparison tables.
- `code/prepare_pdtrt_manuscript_results.py`: figure- and table-generation
  layer for the main and supplemental displays.

### Model-recovery and sensitivity workflows

- `code/run_pdtrt_model_recovery_benchmark.py`: bidirectional benchmark in
  which each candidate model can generate and fit data.
- `code/run_pdtrt_conditional_recovery.py`: parameter-block recovery with the
  remaining parameter blocks fixed at their generating values.
- `code/run_supplemental_sensitivity.py`: parameter-distribution,
  choice-consistency, and decision-rule sensitivity analyses.
- `code/run_network_sensitivity.py`: social-network, hidden-event, and
  social-partner-learning sensitivity analysis.

### Supporting simulation and estimation modules

- `code/ema_pair_simulation.py`: simulator used by the targeted sensitivity
  analyses.
- `code/run_phenotype_analysis.py`: person-behavior profile definitions and
  simulation helpers used by the sensitivity runners.
- `code/validate_predictions_between_person.py`: between-person prediction,
  recovery, and identifiability utilities used by those runners.
- `code/schema_utils.py`: neutral behavior-schema conversion used by the
  validation utilities.

### Checks

- `code/smoke_pdtrt_rerun.py`: end-to-end checks for sample nesting,
  missing-event nesting, leakage, fitting, and resume behavior.
- `code/smoke_pdtrt_factor_calibration.py`: checks phenotype ordering,
  environmental exposure, and behavioral class balance.

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

Run or resume the frozen production design:

```bash
python -B subprojects/07_multi_agent_simulation/code/run_pdtrt_production.py \
  --outdir subprojects/07_multi_agent_simulation/generated/production \
  --workers 1
```

The full run is computationally intensive. The runner is resumable, and the
worker count can be increased when hardware permits.

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

## Distributional Sensitivity

Run the all-profile distributional check:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/run_supplemental_sensitivity.py \
  --outdir subprojects/07_multi_agent_simulation/generated/distribution_sensitivity \
  --profiles all \
  --reps 20 \
  --N 60 \
  --tcount 50 \
  --analyses distribution \
  --distribution_scenarios nominal,skewed_clinical,bimodal_clinical \
  --prediction_max_train_people 40 \
  --skip_recovery \
  --skip_identifiability \
  --workers 1
```

## Choice-Consistency and Decision-Rule Sensitivity

Run the targeted choice-consistency and softmax-versus-threshold checks:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/run_supplemental_sensitivity.py \
  --outdir subprojects/07_multi_agent_simulation/generated/targeted_sensitivity \
  --grid_preset reviewer_targeted \
  --reps 5 \
  --N 60 \
  --tcount 50 \
  --analyses tau,threshold \
  --tau_scenarios nominal,low_tau,high_tau,bimodal_tau \
  --threshold_generators softmax,ddm \
  --prediction_max_train_people 15 \
  --skip_recovery \
  --skip_identifiability \
  --workers 1
```

## Social-Network Sensitivity

Run the social-partner-learning comparison:

```bash
python -B \
  subprojects/07_multi_agent_simulation/code/run_network_sensitivity.py \
  --outdir subprojects/07_multi_agent_simulation/generated/network_sensitivity \
  --reps 5 \
  --network_size 120 \
  --sample_size 30 \
  --tcount 20 \
  --mean_degree 8 \
  --homophily_scale 0.45 \
  --burnin_events 240 \
  --hidden_events_per_wave 120 \
  --variants static_pair_pool,full_network,static_nonrecruited,static_all \
  --prediction_max_train_people 20 \
  --recovery_max_people 12 \
  --identifiability_max_people 4
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
