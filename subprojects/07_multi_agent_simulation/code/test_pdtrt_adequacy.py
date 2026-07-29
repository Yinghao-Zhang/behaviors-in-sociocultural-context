from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from pdtrt_adequacy import cell_diagnostics


class AdequacyTests(unittest.TestCase):
    def test_prediction_and_recovery_share_manuscript_criteria(self) -> None:
        keys = {
            "profile": "balanced",
            "environment": "mixed",
            "mean_events": 50.0,
            "sample_size": 100,
            "missing_rate": 0.0,
            "replicate": 1,
        }
        rows = []
        for model, log_loss in (
            ("tripartite", 0.55),
            ("no_learning", 0.60),
            ("collapsed_reward", 0.57),
            ("lagged", 0.61),
            ("prevalence_null", 0.65),
        ):
            rows.append(
                {
                    **keys,
                    "model": model,
                    "metric_log_loss": log_loss,
                    "metric_ece": 0.05,
                    "diagnostic_optimizer_success_rate": 1.0,
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery_dir = (
                root
                / "conditions"
                / "generator=tripartite"
                / "profile=balanced"
                / "environment=mixed"
                / "events=050"
                / "replicate=001"
                / "views"
                / "N=100"
                / "missing=00"
                / "fits"
                / "estimator=population"
                / "model=tripartite"
            )
            recovery_dir.mkdir(parents=True)
            parameters = [
                "w_i",
                "w_e",
                "w_u",
                "alpha_i_pos",
                "alpha_i_neg",
                "alpha_e",
                "alpha_u",
                "kappa_suggestion",
                "kappa_feedback",
                "choice_consistency",
            ]
            pd.DataFrame(
                [
                    {
                        "level": "population",
                        "parameter": parameter,
                        "true_mean": 1.0,
                        "estimate": 1.05,
                        "bias": 0.05,
                    }
                    for parameter in parameters
                ]
            ).to_csv(recovery_dir / "recovery.csv", index=False)

            cells, replicate_recovery = cell_diagnostics(
                root,
                pd.DataFrame(rows),
                expected_replicates=1,
                expected_recovery_files=1,
            )

        self.assertEqual(len(replicate_recovery), 1)
        self.assertTrue(bool(cells.loc[0, "minimum_predictive_signal"]))
        self.assertTrue(bool(cells.loc[0, "full_vector_recovery"]))


if __name__ == "__main__":
    unittest.main()
