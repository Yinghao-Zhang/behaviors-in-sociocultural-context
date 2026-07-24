from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class GlobalBehaviorPriors:
    """
    Stores global mean priors for behaviors (instinct, enjoyment, utility).
    Values are optional; absence indicates no prior for that component.
    """
    priors: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    def set_prior(self, behavior_id: str, instinct: Optional[float], enjoyment: Optional[float], utility: Optional[float]):
        self.priors[behavior_id] = {
            "instinct": instinct,
            "enjoyment": enjoyment,
            "utility": utility,
        }

    def get_prior(self, behavior_id: str) -> Optional[Dict[str, Optional[float]]]:
        return self.priors.get(behavior_id)


def compute_global_priors(rows: Iterable[dict], behavior_key: str, instinct_key: str,
                          enjoyment_key: str, utility_key: str) -> GlobalBehaviorPriors:
    """
    Compute global behavior mean priors from row-wise data.
    Rows should include behavior ID and baseline instinct/enjoyment/utility values.
    """
    buckets: Dict[str, Dict[str, list]] = {}
    for row in rows:
        behavior_id = row.get(behavior_key)
        if behavior_id is None:
            continue
        buckets.setdefault(behavior_id, {"instinct": [], "enjoyment": [], "utility": []})
        for key, dest in [(instinct_key, "instinct"), (enjoyment_key, "enjoyment"), (utility_key, "utility")]:
            val = row.get(key)
            if val is not None and not np.isnan(val):
                buckets[behavior_id][dest].append(float(val))

    priors = GlobalBehaviorPriors()
    for behavior_id, vals in buckets.items():
        prior = {
            "instinct": float(np.mean(vals["instinct"])) if vals["instinct"] else None,
            "enjoyment": float(np.mean(vals["enjoyment"])) if vals["enjoyment"] else None,
            "utility": float(np.mean(vals["utility"])) if vals["utility"] else None,
        }
        priors.set_prior(behavior_id, prior["instinct"], prior["enjoyment"], prior["utility"])

    return priors
