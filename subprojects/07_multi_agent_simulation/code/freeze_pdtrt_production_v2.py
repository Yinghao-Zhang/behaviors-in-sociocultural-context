from __future__ import annotations

from datetime import date
from pathlib import Path

from pdtrt_adequacy import manifest_adequacy_thresholds
from pdtrt_rerun_core import (
    ENVIRONMENT_NAMES,
    PROFILE_NAMES,
    atomic_write_json,
    generator_constants_fingerprint,
    generator_constants_payload,
)


def main() -> int:
    output = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "pdtrt_production_v2.json"
    )
    runtime_by_event = {
        "10": 49.021443,
        "25": 100.389588,
        "50": 236.870454,
    }
    generation_by_event = {
        "10": 1.240285,
        "25": 1.538882,
        "50": 1.937527,
    }
    manifest = {
        "status": "frozen",
        "version": "pdtrt_production_v2",
        "frozen_date": str(date(2026, 7, 26)),
        "generator_constants_fingerprint": (
            generator_constants_fingerprint()
        ),
        "design": {
            "profiles": list(PROFILE_NAMES),
            "environments": list(ENVIRONMENT_NAMES),
            "generator_model": "tripartite",
            "sample_sizes": [30, 60, 100],
            "mean_events": [10, 25, 50],
            "missing_rates": [0.0, 0.1, 0.2],
            "replicates": 20,
            "base_seed": 20260718,
            "candidate_models": [
                "tripartite",
                "no_learning",
                "collapsed_reward",
                "lagged",
            ],
        },
        "estimation": {
            "primary_estimator": "population",
            "choice_parameter": "effective_choice_consistency",
            "latent_value_noise_in_fitted_model": "fixed_near_zero",
            "fit_max_iter": 120,
            "production_multistarts": 1,
            "diagnostic_multistarts": 2,
            "baseline_reliability": 0.70,
            "observer_penalty": 0.50,
            "held_out_participant_fraction": 0.20,
            "primary_prediction_metric": (
                "held_out_sequential_log_loss"
            ),
            "person_level_parameter_interpretation": False,
        },
        "diagnostic_staging": {
            "multistart_validation": (
                "Two starts in three balanced generous-information "
                "replicates and six phenotype-targeted replicates."
            ),
            "conditional_recovery": (
                "Parameter blocks fitted with all remaining person "
                "parameters fixed to their generating values."
            ),
            "bidirectional_model_recovery": (
                "Separate supplemental benchmark using each candidate "
                "model as a data generator."
            ),
            "failed_fits": (
                "Retain, count, and report; never silently drop."
            ),
        },
        "generator_constants": generator_constants_payload(),
        "adequacy_thresholds": manifest_adequacy_thresholds(),
        "runtime_projection": {
            "workers": 1,
            "measured_parent_fit_seconds_by_mean_events": runtime_by_event,
            "measured_parent_generation_seconds_by_mean_events": (
                generation_by_event
            ),
            "projected_recorded_cpu_hours": 32.58,
            "projected_elapsed_hours_with_overhead": "35-40",
            "projected_disk_gigabytes": "2.2-2.8",
        },
        "preproduction_findings": {
            "generator_and_analysis_gates": "passed",
            "generous_prediction_criterion": "passed",
            "generous_population_recovery": {
                "decision_weights": "partial",
                "learning_rates": "passed",
                "social_integration": "not_recovered",
                "choice_consistency": "passed",
                "full_vector": "not_recovered",
            },
            "interpretation": (
                "Shared-parameter accuracy remains an empirical outcome "
                "of the factorial design. Failure to recover the full vector "
                "is retained as an interpretability boundary rather than "
                "removed by changing thresholds."
            ),
        },
    }
    atomic_write_json(output, manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
