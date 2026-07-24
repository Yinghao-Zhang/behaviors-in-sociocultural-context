from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pdtrt_rerun_core import (
    ENVIRONMENT_NAMES,
    PROFILE_NAMES,
    PDTRTRerunSimulator,
    RerunConfig,
    stable_seed,
)


def main() -> int:
    seed = stable_seed(20260718, "factor_calibration_smoke")
    results = {}
    reference_edges = None
    reference_focal_ids = None
    reference_initial_states = None
    reference_nuisance_parameters = None

    for profile in PROFILE_NAMES:
        for environment in ENVIRONMENT_NAMES:
            config = RerunConfig(
                profile=profile,
                environment=environment,
                seed=seed,
                days=7,
                network_size=120,
                max_focal=30,
                mean_events=20,
                mean_degree=8,
                n_social_foci=6,
                hidden_events_per_person_day=0.05,
                pilot_version="factor_calibration_smoke_v1",
            )
            result = PDTRTRerunSimulator(config).run()
            key = (profile, environment)
            results[key] = result

            edges = result.network_edges.reset_index(drop=True)
            focal_ids = result.people_truth[["focal_id", "network_id"]].reset_index(
                drop=True
            )
            initial_columns = [
                column
                for column in result.people_truth.columns
                if column.startswith("initial_")
            ]
            initial_states = result.people_truth[initial_columns].reset_index(drop=True)
            nuisance_parameters = result.people_truth[
                ["true_tau", "true_noise_s"]
            ].reset_index(drop=True)
            if reference_edges is None:
                reference_edges = edges
                reference_focal_ids = focal_ids
                reference_initial_states = initial_states
                reference_nuisance_parameters = nuisance_parameters
            else:
                pd.testing.assert_frame_equal(edges, reference_edges)
                pd.testing.assert_frame_equal(focal_ids, reference_focal_ids)
                pd.testing.assert_frame_equal(initial_states, reference_initial_states)
                pd.testing.assert_frame_equal(
                    nuisance_parameters,
                    reference_nuisance_parameters,
                )

    profile_means = {}
    for profile in PROFILE_NAMES:
        truth = results[(profile, "repair_supportive")].people_truth
        profile_means[profile] = {
            "w_i": float(truth["true_w_i"].mean()),
            "w_e": float(truth["true_w_e"].mean()),
            "w_u": float(truth["true_w_u"].mean()),
            "alpha_i_pos": float(truth["true_alpha_i_pos"].mean()),
            "alpha_i_neg": float(truth["true_alpha_i_neg"].mean()),
            "alpha_e": float(truth["true_alpha_e"].mean()),
            "alpha_u": float(truth["true_alpha_u"].mean()),
            "social_kappa": float(truth["true_social_kappa"].mean()),
        }

    assert profile_means["rigid_habitual"]["w_i"] > 0.55
    assert (
        profile_means["rigid_habitual"]["alpha_i_pos"]
        > profile_means["rigid_habitual"]["alpha_e"]
    )
    assert profile_means["relief_reactive"]["w_e"] > 0.55
    assert (
        profile_means["relief_reactive"]["alpha_e"]
        > profile_means["relief_reactive"]["alpha_u"]
    )
    assert profile_means["consequence_sensitive"]["w_u"] > 0.55
    assert (
        profile_means["consequence_sensitive"]["alpha_u"]
        > profile_means["consequence_sensitive"]["alpha_e"]
    )
    assert (
        profile_means["socially_contingent"]["social_kappa"]
        > profile_means["balanced"]["social_kappa"]
    )

    environment_means = {}
    for environment in ENVIRONMENT_NAMES:
        rows = [
            results[(profile, environment)].generation_diagnostics
            for profile in PROFILE_NAMES
        ]
        environment_means[environment] = {
            name: float(np.mean([row[f"context_prop_{name}"] for row in rows]))
            for name in (
                "repair_opportunity",
                "escalation_risk",
                "ambiguous_conflict",
            )
        }

    assert (
        environment_means["repair_supportive"]["repair_opportunity"]
        > environment_means["repair_supportive"]["escalation_risk"]
    )
    assert (
        environment_means["escalation_prone"]["escalation_risk"]
        > environment_means["escalation_prone"]["repair_opportunity"]
    )
    assert (
        environment_means["inconsistent_ambiguous"]["ambiguous_conflict"]
        > max(
            environment_means["inconsistent_ambiguous"]["repair_opportunity"],
            environment_means["inconsistent_ambiguous"]["escalation_risk"],
        )
    )

    approach_rates = [
        result.generation_diagnostics["approach_rate"] for result in results.values()
    ]
    assert min(approach_rates) > 0.05
    assert max(approach_rates) < 0.95

    print(
        json.dumps(
            {
                "paired_network_and_focal_sample": True,
                "paired_initial_states": True,
                "paired_nuisance_parameters": True,
                "profile_parameter_ordering": profile_means,
                "environment_context_ordering": environment_means,
                "approach_rate_range": [
                    float(min(approach_rates)),
                    float(max(approach_rates)),
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
