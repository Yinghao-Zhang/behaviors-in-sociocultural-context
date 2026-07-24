from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from behavior import Behavior
from behavior_taxonomy import BehaviorTaxonomy


@dataclass
class BehaviorFeatureSpec:
    include_domain: bool = True
    include_motivational_system: bool = True
    include_regulatory_direction: bool = True
    include_relation_similarity: bool = True
    include_missing_indicators: bool = False


def _one_hot(value: Optional[str], allowed: Iterable[str], prefix: str, include_missing: bool) -> Dict[str, float]:
    values = list(allowed)
    out: Dict[str, float] = {}
    for v in values:
        out[f"{prefix}_{v}"] = 1.0 if value == v else 0.0
    if include_missing:
        out[f"{prefix}_missing"] = 1.0 if value is None or value not in values else 0.0
    return out


def encode_behavior_features(
    behavior: Behavior,
    taxonomy: Optional[BehaviorTaxonomy],
    spec: BehaviorFeatureSpec,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if taxonomy is None:
        return features

    if spec.include_domain:
        features.update(_one_hot(behavior.primary_domain, taxonomy.list_domains(), "domain", spec.include_missing_indicators))
    if spec.include_motivational_system:
        features.update(_one_hot(behavior.motivational_system, taxonomy.list_motivational_systems(), "motivation", spec.include_missing_indicators))
    if spec.include_regulatory_direction:
        features.update(_one_hot(behavior.regulatory_direction, taxonomy.list_regulatory_directions(), "direction", spec.include_missing_indicators))

    return features


def encode_relation_features(
    behavior_a: Behavior,
    behavior_b: Behavior,
    spec: BehaviorFeatureSpec,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if not spec.include_relation_similarity:
        return features

    rel = Behavior.get_relation(behavior_a, behavior_b)
    if rel is None:
        features["relation_similarity"] = 0.0
        return features

    features["relation_similarity"] = float(rel.get("similarity", 0.0))
    return features
