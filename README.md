# Behaviors in Sociocultural Context

This repository contains a computational framework for modeling how behavior
is selected and learned within situational and social contexts. The current
public release is intentionally limited to the reusable model code and the
simulation workflow developed in project 07.

## Repository Map

### Core model

- `agent.py`: individual, group, and cultural agents; behavioral states;
  learning parameters; and social relationships.
- `behavior.py`: behavior definitions, priors, and relations among behaviors.
- `setup.py`: situational-context definitions and context management.
- `situation.py`: behavioral selection, direct and observational learning,
  suggestions, feedback, and social interaction modes.
- `behavior_taxonomy.py`, `behavior_features.py`, and `behavior_priors.py`:
  optional behavior metadata, feature encoding, and prior specification.
- `hyperparameter_tuning.py`: reusable hyperparameter-search utilities.
- `configs/`: configuration examples and schema guides for the core model.

### Project 07

`subprojects/07_multi_agent_simulation/` contains the simulation and analysis
code for evaluating an ambulatory-assessment design built around a
learning-based generative computational model. Its
[project README](subprojects/07_multi_agent_simulation/README.md) provides the
code directory, environment specification, smoke tests, and reproduction
commands.

## Installation

The production analysis used Python 3.9. Create an isolated environment and
install the project-07 dependencies:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r subprojects/07_multi_agent_simulation/requirements.txt
```

Run the lightweight end-to-end check from the repository root:

```bash
python -B subprojects/07_multi_agent_simulation/code/smoke_pdtrt_rerun.py \
  --outdir subprojects/07_multi_agent_simulation/generated/smoke
```

## Release Boundary

Generated result files, manuscript materials, local environments, local
archives, and unreleased subprojects are not part of the public code release.
The analyses documented in the project-07 README can be regenerated from the
included code, configuration, and recorded random seeds.

## License

The code is released under the [BSD 3-Clause License](LICENSE).

## Citation

Citation metadata for this software is provided in
[`CITATION.cff`](CITATION.cff).
