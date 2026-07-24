from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set
import json
import pathlib


@dataclass
class BehaviorTaxonomy:
    domains: Dict[str, Dict[str, str]]
    motivational_systems: Set[str]
    regulatory_directions: Set[str]

    @classmethod
    def from_dict(cls, data: dict) -> "BehaviorTaxonomy":
        domains = data.get("domains", {}) or {}
        motivational_systems = set(data.get("motivational_systems", []) or [])
        regulatory_directions = set(data.get("regulatory_directions", []) or [])
        return cls(domains=domains, motivational_systems=motivational_systems, regulatory_directions=regulatory_directions)

    @classmethod
    def load_json(cls, path: str) -> "BehaviorTaxonomy":
        path_obj = pathlib.Path(path)
        data = json.loads(path_obj.read_text())
        return cls.from_dict(data)

    def validate_domain(self, domain_id: str) -> bool:
        return domain_id in self.domains

    def validate_motivational_system(self, system_id: str) -> bool:
        return system_id in self.motivational_systems

    def validate_regulatory_direction(self, direction_id: str) -> bool:
        return direction_id in self.regulatory_directions

    def list_domains(self) -> Iterable[str]:
        return sorted(self.domains.keys())

    def list_motivational_systems(self) -> Iterable[str]:
        return sorted(self.motivational_systems)

    def list_regulatory_directions(self) -> Iterable[str]:
        return sorted(self.regulatory_directions)
