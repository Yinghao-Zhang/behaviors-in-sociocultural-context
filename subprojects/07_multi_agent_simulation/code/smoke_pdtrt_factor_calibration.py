from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdtrt_rerun_core import (
    ENVIRONMENT_NAMES,
    PROFILE_NAMES,
    PDTRTRerunSimulator,
    RerunConfig,
    atomic_write_json,
    stable_seed,
)


def _initial_contrasts(truth: pd.DataFrame) -> dict[str, float]:
    return {
        state_name: float(
            (
                truth[f"initial_{state_name}_approach"]
                - truth[f"initial_{state_name}_avoid"]
            ).mean()
        )
        for state_name in ("instinct", "enjoyment", "utility")
    }


def _social_signal_summary(result) -> dict[str, float]:
    truth = result.truth_events
    suggestions = truth.loc[truth["suggestion_active"] == 1]
    feedback = truth.loc[truth["feedback_active"] == 1].copy()
    if feedback.empty:
        raise AssertionError("Calibration generated no feedback events")
    feedback["approach_direction"] = np.where(
        feedback["behavior_idx"].to_numpy(dtype=int) == 1,
        1.0,
        -1.0,
    )
    return {
        "suggestion_approach_minus_avoid": float(
            (
                suggestions["suggestion_approach_true"]
                - suggestions["suggestion_avoid_true"]
            ).mean()
        ),
        "feedback_approach_preference": float(
            (
                feedback["feedback_signal"]
                * feedback["approach_direction"]
            ).mean()
        ),
        "received_feedback_approach_preference": float(
            (
                feedback["qualified_feedback"]
                * feedback["approach_direction"]
            ).mean()
        ),
        "feedback_abs_p95": float(
            np.quantile(np.abs(feedback["feedback_signal"]), 0.95)
        ),
        "feedback_event_count": int(len(feedback)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--network-size", type=int, default=180)
    parser.add_argument("--max-focal", type=int, default=40)
    parser.add_argument("--mean-events", type=float, default=20)
    parser.add_argument("--mean-degree", type=float, default=8)
    parser.add_argument("--social-foci", type=int, default=6)
    parser.add_argument(
        "--hidden-events-per-person-day",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()
    seed = stable_seed(20260718, "factor_calibration_smoke_v2")
    results = {}
    reference_edges = None
    reference_focal_ids = None
    reference_nuisance_parameters = None
    reference_background_parameters = None

    for profile in PROFILE_NAMES:
        for environment in ENVIRONMENT_NAMES:
            config = RerunConfig(
                profile=profile,
                environment=environment,
                seed=seed,
                days=args.days,
                network_size=args.network_size,
                max_focal=args.max_focal,
                mean_events=args.mean_events,
                mean_degree=args.mean_degree,
                n_social_foci=args.social_foci,
                hidden_events_per_person_day=(
                    args.hidden_events_per_person_day
                ),
                pilot_version="factor_calibration_smoke_v2",
            )
            simulator = PDTRTRerunSimulator(config)
            result = simulator.run()
            results[(profile, environment)] = result

            edges = result.network_edges.reset_index(drop=True)
            focal_ids = result.people_truth[
                ["focal_id", "network_id"]
            ].reset_index(drop=True)
            nuisance_parameters = result.people_truth[
                ["true_tau", "true_noise_s"]
            ].reset_index(drop=True)
            background_mask = np.ones(config.network_size, dtype=bool)
            background_mask[simulator.focal_network_ids] = False
            background_parameters = {
                name: values[background_mask].copy()
                for name, values in simulator.parameters.items()
            }
            if reference_edges is None:
                reference_edges = edges
                reference_focal_ids = focal_ids
                reference_nuisance_parameters = nuisance_parameters
                reference_background_parameters = background_parameters
            else:
                pd.testing.assert_frame_equal(edges, reference_edges)
                pd.testing.assert_frame_equal(focal_ids, reference_focal_ids)
                pd.testing.assert_frame_equal(
                    nuisance_parameters,
                    reference_nuisance_parameters,
                )
                for name, values in background_parameters.items():
                    np.testing.assert_allclose(
                        values,
                        reference_background_parameters[name],
                        rtol=0.0,
                        atol=0.0,
                    )

    profile_means = {}
    for profile in PROFILE_NAMES:
        truth = results[(profile, "mixed")].people_truth
        profile_means[profile] = {
            "w_i": float(truth["true_w_i"].mean()),
            "w_e": float(truth["true_w_e"].mean()),
            "w_u": float(truth["true_w_u"].mean()),
            "alpha_i_pos": float(truth["true_alpha_i_pos"].mean()),
            "alpha_i_neg": float(truth["true_alpha_i_neg"].mean()),
            "alpha_e": float(truth["true_alpha_e"].mean()),
            "alpha_u": float(truth["true_alpha_u"].mean()),
            "kappa_suggestion": float(
                truth["true_kappa_suggestion"].mean()
            ),
            "kappa_feedback": float(
                truth["true_kappa_feedback"].mean()
            ),
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
        profile_means["socially_contingent"]["kappa_suggestion"]
        > profile_means["balanced"]["kappa_suggestion"]
    )
    assert (
        profile_means["socially_contingent"]["kappa_feedback"]
        > profile_means["balanced"]["kappa_feedback"]
    )

    environment_summaries = {}
    for environment in ENVIRONMENT_NAMES:
        initial_rows = [
            _initial_contrasts(results[(profile, environment)].people_truth)
            for profile in PROFILE_NAMES
        ]
        signal_rows = [
            _social_signal_summary(results[(profile, environment)])
            for profile in PROFILE_NAMES
        ]
        environment_summaries[environment] = {
            "initial_contrasts": {
                state_name: float(
                    np.mean([row[state_name] for row in initial_rows])
                )
                for state_name in ("instinct", "enjoyment", "utility")
            },
            "social_signals": {
                key: float(np.mean([row[key] for row in signal_rows]))
                for key in signal_rows[0]
            },
            "approach_rate": float(
                np.mean(
                    [
                        results[(profile, environment)]
                        .generation_diagnostics["approach_rate"]
                        for profile in PROFILE_NAMES
                    ]
                )
            ),
        }

    ordered_environments = (
        "avoidance_oriented",
        "mixed",
        "approach_oriented",
    )
    for state_name in ("instinct", "enjoyment", "utility"):
        values = [
            environment_summaries[environment]["initial_contrasts"][
                state_name
            ]
            for environment in ordered_environments
        ]
        assert values[0] < values[1] < values[2], (state_name, values)

    for signal_name in (
        "suggestion_approach_minus_avoid",
        "feedback_approach_preference",
        "received_feedback_approach_preference",
    ):
        values = [
            environment_summaries[environment]["social_signals"][
                signal_name
            ]
            for environment in ordered_environments
        ]
        assert values[0] < values[1] < values[2], (signal_name, values)

    approach_rates = [
        result.generation_diagnostics["approach_rate"]
        for result in results.values()
    ]
    assert min(approach_rates) > 0.05
    assert max(approach_rates) < 0.95
    assert max(
        summary["social_signals"]["feedback_abs_p95"]
        for summary in environment_summaries.values()
    ) < 0.75
    generation_rows = [
        result.generation_diagnostics for result in results.values()
    ]
    network_rows = [
        result.network_diagnostics for result in results.values()
    ]
    event_means = [
        row["eligible_event_mean"] for row in generation_rows
    ]
    event_sds = [row["eligible_event_sd"] for row in generation_rows]
    assert max(abs(value - args.mean_events) for value in event_means) <= max(
        1.0,
        0.15 * args.mean_events,
    )
    assert min(event_sds) > np.sqrt(max(args.mean_events, 1.0))
    assert max(
        row["boundary_enjoyment_fraction"] for row in generation_rows
    ) <= 0.05
    assert max(
        row["boundary_utility_fraction"] for row in generation_rows
    ) <= 0.05
    assert max(
        row["boundary_perceived_utility_fraction"]
        for row in generation_rows
    ) <= 0.05
    assert min(
        row["largest_component_fraction"] for row in network_rows
    ) >= 0.95
    assert min(row["mean_clustering"] for row in network_rows) >= 0.05
    assert max(
        abs(row["mean_degree"] - args.mean_degree)
        for row in network_rows
    ) <= max(2.0, 0.25 * args.mean_degree)

    outcome_contrasts = {}
    for environment in ENVIRONMENT_NAMES:
        frames = [
            results[(profile, environment)].truth_events
            for profile in PROFILE_NAMES
        ]
        events = pd.concat(frames, ignore_index=True)
        means = events.groupby("behavior_idx")[
            ["raw_enjoyment", "raw_utility"]
        ].mean()
        enjoyment_avoid_minus_approach = float(
            means.loc[0, "raw_enjoyment"]
            - means.loc[1, "raw_enjoyment"]
        )
        utility_approach_minus_avoid = float(
            means.loc[1, "raw_utility"]
            - means.loc[0, "raw_utility"]
        )
        assert enjoyment_avoid_minus_approach > 0.25
        assert utility_approach_minus_avoid > 0.15
        outcome_contrasts[environment] = {
            "enjoyment_avoid_minus_approach": (
                enjoyment_avoid_minus_approach
            ),
            "utility_approach_minus_avoid": utility_approach_minus_avoid,
        }

    payload = {
        "calibration_design": {
            "days": args.days,
            "network_size": args.network_size,
            "recruited_participants": args.max_focal,
            "mean_events": args.mean_events,
            "mean_degree": args.mean_degree,
            "profiles": list(PROFILE_NAMES),
            "environments": list(ENVIRONMENT_NAMES),
        },
        "paired_network_and_focal_sample": True,
        "paired_nuisance_parameters": True,
        "paired_nonrecruited_background_parameters": True,
        "profile_parameter_ordering": profile_means,
        "environment_calibration": environment_summaries,
        "outcome_tradeoff": outcome_contrasts,
        "network_diagnostics": network_rows[0],
        "event_yield_range": [
            float(min(event_means)),
            float(max(event_means)),
        ],
        "event_sd_range": [
            float(min(event_sds)),
            float(max(event_sds)),
        ],
        "maximum_boundary_fraction": float(
            max(
                max(
                    row["boundary_enjoyment_fraction"],
                    row["boundary_utility_fraction"],
                    row["boundary_perceived_utility_fraction"],
                )
                for row in generation_rows
            )
        ),
        "approach_rate_range": [
            float(min(approach_rates)),
            float(max(approach_rates)),
        ],
        "checks": {
            "profile_parameter_ordering": "passed",
            "environment_initial_state_ordering": "passed",
            "suggestion_direction_ordering": "passed",
            "feedback_direction_ordering": "passed",
            "behavioral_class_balance": "passed",
            "feedback_saturation": "passed",
            "event_yield_and_overdispersion": "passed",
            "network_structure": "passed",
            "outcome_tradeoff": "passed",
            "outcome_clipping": "passed",
            "nonrecruited_background_invariance": "passed",
        },
        "interpretation": (
            "The revised environments produced ordered but overlapping "
            "differences in initial states, suggestions, feedback, and "
            "behavior. Social signals remained modest, and all phenotype-by-"
            "environment cells retained both behavioral classes."
        ),
    }
    if args.out:
        atomic_write_json(Path(args.out).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
