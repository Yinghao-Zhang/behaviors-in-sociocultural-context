from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax


CONTEXTS: Tuple[str, ...] = (
    "repair_opportunity",
    "escalation_risk",
    "ambiguous_conflict",
)
BEHAVIORS: Tuple[str, ...] = ("avoid", "approach")
STATE_NAMES: Tuple[str, ...] = ("instinct", "enjoyment", "utility")
PROFILE_NAMES: Tuple[str, ...] = (
    "balanced",
    "rigid_habitual",
    "relief_reactive",
    "consequence_sensitive",
    "socially_contingent",
)
ENVIRONMENT_NAMES: Tuple[str, ...] = (
    "repair_supportive",
    "escalation_prone",
    "inconsistent_ambiguous",
)
GENERATOR_MODELS: Tuple[str, ...] = (
    "tripartite",
    "no_learning",
    "collapsed_reward",
    "lagged",
)


PROFILE_SPECS: Dict[str, Dict[str, object]] = {
    "balanced": {
        "weight_mean": (1 / 3, 1 / 3, 1 / 3),
        "weight_concentration": 24.0,
        "alpha_i_pos": (0.10, 0.30),
        "alpha_i_neg": (0.10, 0.30),
        "alpha_e": (0.10, 0.30),
        "alpha_u": (0.10, 0.30),
        "social_kappa": (0.25, 1.25),
        "receptivity_bias": (-0.15, 0.30),
    },
    "rigid_habitual": {
        "weight_mean": (0.65, 0.175, 0.175),
        "weight_concentration": 28.0,
        "alpha_i_pos": (0.30, 0.50),
        "alpha_i_neg": (0.03, 0.12),
        "alpha_e": (0.03, 0.12),
        "alpha_u": (0.03, 0.12),
        "social_kappa": (0.20, 1.10),
        "receptivity_bias": (-0.15, 0.25),
    },
    "relief_reactive": {
        "weight_mean": (0.175, 0.65, 0.175),
        "weight_concentration": 28.0,
        "alpha_i_pos": (0.10, 0.30),
        "alpha_i_neg": (0.10, 0.30),
        "alpha_e": (0.30, 0.50),
        "alpha_u": (0.05, 0.18),
        "social_kappa": (0.20, 1.20),
        "receptivity_bias": (-0.10, 0.30),
    },
    "consequence_sensitive": {
        "weight_mean": (0.175, 0.175, 0.65),
        "weight_concentration": 28.0,
        "alpha_i_pos": (0.10, 0.30),
        "alpha_i_neg": (0.10, 0.30),
        "alpha_e": (0.05, 0.18),
        "alpha_u": (0.30, 0.50),
        "social_kappa": (0.20, 1.20),
        "receptivity_bias": (-0.10, 0.30),
    },
    "socially_contingent": {
        "weight_mean": (1 / 3, 1 / 3, 1 / 3),
        "weight_concentration": 24.0,
        "alpha_i_pos": (0.10, 0.30),
        "alpha_i_neg": (0.10, 0.30),
        "alpha_e": (0.10, 0.30),
        "alpha_u": (0.10, 0.30),
        "social_kappa": (0.90, 1.80),
        "receptivity_bias": (0.20, 0.65),
    },
}


ENVIRONMENT_SPECS: Dict[str, Dict[str, object]] = {
    "repair_supportive": {
        "context_mean": (0.60, 0.15, 0.25),
        "context_concentration": 14.0,
    },
    "escalation_prone": {
        "context_mean": (0.15, 0.60, 0.25),
        "context_concentration": 14.0,
    },
    "inconsistent_ambiguous": {
        "context_mean": (0.20, 0.20, 0.60),
        "context_concentration": 10.0,
    },
}


# Rows are contexts and columns are avoid/approach. Immediate experience and
# instrumental outcome are deliberately distinct.
ENJOYMENT_MEANS = np.array(
    [
        [0.45, -0.15],
        [0.50, -0.45],
        [0.25, -0.05],
    ],
    dtype=float,
)
UTILITY_MEANS = np.array(
    [
        [-0.25, 0.65],
        [0.35, -0.55],
        [0.05, 0.10],
    ],
    dtype=float,
)
ENJOYMENT_SDS = np.array(
    [
        [0.20, 0.24],
        [0.22, 0.28],
        [0.38, 0.40],
    ],
    dtype=float,
)
UTILITY_SDS = np.array(
    [
        [0.28, 0.21],
        [0.28, 0.34],
        [0.42, 0.44],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class RerunConfig:
    profile: str = "balanced"
    environment: str = "repair_supportive"
    generator_model: str = "tripartite"
    seed: int = 20260718
    days: int = 28
    network_size: int = 1000
    max_focal: int = 100
    mean_events: float = 25.0
    event_dispersion: float = 3.0
    mean_degree: float = 12.0
    homophily_scale: float = 0.55
    n_social_foci: int = 12
    hidden_events_per_person_day: float = 0.10
    baseline_report_sd: float = 0.20
    outcome_report_sd: float = 0.12
    relationship_report_sd: float = 0.10
    suggestion_report_sd: float = 0.08
    feedback_report_sd: float = 0.08
    initial_prior_alignment: float = 0.25
    state_sd: float = 0.35
    outcome_relationship_scale: float = 0.12
    observation_attenuation: float = 0.50
    missing_person_sd: float = 0.40
    min_tau: float = 0.5
    max_tau: float = 10.0
    pilot_version: str = "pilot_v1"

    def __post_init__(self) -> None:
        if self.profile not in PROFILE_NAMES:
            raise ValueError(f"Unknown profile: {self.profile}")
        if self.environment not in ENVIRONMENT_NAMES:
            raise ValueError(f"Unknown environment: {self.environment}")
        if self.generator_model not in GENERATOR_MODELS:
            raise ValueError(f"Unknown generator model: {self.generator_model}")
        if self.network_size < self.max_focal:
            raise ValueError("network_size must be at least max_focal")
        if self.max_focal < 2:
            raise ValueError("max_focal must be at least 2")
        if self.mean_events < 0:
            raise ValueError("mean_events must be nonnegative")
        if self.event_dispersion <= 0:
            raise ValueError("event_dispersion must be positive")


@dataclass
class SimulationResult:
    config: RerunConfig
    truth_events: pd.DataFrame
    complete_reports: pd.DataFrame
    people_truth: pd.DataFrame
    people_observed: pd.DataFrame
    network_edges: pd.DataFrame
    network_relationships: pd.DataFrame
    network_diagnostics: Dict[str, float]
    generation_diagnostics: Dict[str, float]


@dataclass
class PanelView:
    sample_size: int
    missing_rate: float
    events: pd.DataFrame
    people: pd.DataFrame
    people_truth: pd.DataFrame
    diagnostics: Dict[str, float]


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def simulator_seed_manifest(base_seed: int) -> Dict[str, int]:
    return {
        name: stable_seed(base_seed, label)
        for name, label in (
            ("network", "network"),
            ("environment", "environment"),
            ("focal_sample", "focal"),
            ("background_parameters", "background_parameters"),
            ("focal_profile_parameters", "focal_profile_parameters"),
            ("initial_states", "initial_states"),
            ("protocol", "protocol"),
            ("events", "events"),
            ("outcomes", "outcomes"),
            ("reports", "reports"),
            ("missingness", "missingness"),
        )
    }


def config_fingerprint(config: RerunConfig) -> str:
    payload = {
        "config": asdict(config),
        "generator_constants": generator_constants_payload(),
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]


def generator_constants_payload() -> Dict[str, object]:
    defaults = asdict(RerunConfig())
    for key in (
        "profile",
        "environment",
        "generator_model",
        "seed",
        "mean_events",
        "pilot_version",
    ):
        defaults.pop(key)
    return {
        "rerun_config_defaults": defaults,
        "profile_specs": PROFILE_SPECS,
        "environment_specs": ENVIRONMENT_SPECS,
        "enjoyment_means": ENJOYMENT_MEANS.tolist(),
        "enjoyment_sds": ENJOYMENT_SDS.tolist(),
        "utility_means": UTILITY_MEANS.tolist(),
        "utility_sds": UTILITY_SDS.tolist(),
    }


def generator_constants_fingerprint() -> str:
    payload = json.dumps(
        generator_constants_payload(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _sample_uniform(rng: np.random.Generator, bounds: Sequence[float], n: int) -> np.ndarray:
    return rng.uniform(float(bounds[0]), float(bounds[1]), size=n)


def _sample_profile_parameters(
    rng: np.random.Generator,
    profile: str,
    n: int,
) -> Dict[str, np.ndarray]:
    spec = PROFILE_SPECS[profile]
    mean = np.asarray(spec["weight_mean"], dtype=float)
    concentration = float(spec["weight_concentration"])
    weights = rng.dirichlet(np.maximum(0.05, mean * concentration), size=n)
    tau = np.clip(rng.lognormal(np.log(3.0), 0.25, size=n), 0.5, 10.0)
    noise = rng.uniform(0.05, 0.20, size=n)
    return {
        "w_i": weights[:, 0],
        "w_e": weights[:, 1],
        "w_u": weights[:, 2],
        "alpha_i_pos": _sample_uniform(rng, spec["alpha_i_pos"], n),
        "alpha_i_neg": _sample_uniform(rng, spec["alpha_i_neg"], n),
        "alpha_e": _sample_uniform(rng, spec["alpha_e"], n),
        "alpha_u": _sample_uniform(rng, spec["alpha_u"], n),
        "social_kappa": _sample_uniform(rng, spec["social_kappa"], n),
        "receptivity_bias": _sample_uniform(rng, spec["receptivity_bias"], n),
        "tau": tau,
        "noise_s": noise,
    }


def _sample_paired_focal_parameters(
    base_seed: int,
    profile: str,
    n: int,
) -> Dict[str, np.ndarray]:
    spec = PROFILE_SPECS[profile]
    weight_rng = np.random.default_rng(
        stable_seed(base_seed, "focal_profile_parameters", "weights")
    )
    mean = np.asarray(spec["weight_mean"], dtype=float)
    concentration = float(spec["weight_concentration"])
    weights = weight_rng.dirichlet(
        np.maximum(0.05, mean * concentration),
        size=n,
    )
    output = {
        "w_i": weights[:, 0],
        "w_e": weights[:, 1],
        "w_u": weights[:, 2],
    }
    for field in (
        "alpha_i_pos",
        "alpha_i_neg",
        "alpha_e",
        "alpha_u",
        "social_kappa",
        "receptivity_bias",
    ):
        field_rng = np.random.default_rng(
            stable_seed(base_seed, "focal_profile_parameters", field)
        )
        output[field] = _sample_uniform(field_rng, spec[field], n)
    tau_rng = np.random.default_rng(
        stable_seed(base_seed, "focal_profile_parameters", "tau")
    )
    noise_rng = np.random.default_rng(
        stable_seed(base_seed, "focal_profile_parameters", "noise_s")
    )
    output["tau"] = np.clip(
        tau_rng.lognormal(np.log(3.0), 0.25, size=n),
        0.5,
        10.0,
    )
    output["noise_s"] = noise_rng.uniform(0.05, 0.20, size=n)
    return output


def _sample_background_parameters(
    rng: np.random.Generator,
    n: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    mixture_names = np.asarray(PROFILE_NAMES, dtype=object)
    mixture_probs = np.array([0.60, 0.10, 0.10, 0.10, 0.10], dtype=float)
    assignments = rng.choice(mixture_names, size=n, p=mixture_probs)
    fields = (
        "w_i",
        "w_e",
        "w_u",
        "alpha_i_pos",
        "alpha_i_neg",
        "alpha_e",
        "alpha_u",
        "social_kappa",
        "receptivity_bias",
        "tau",
        "noise_s",
    )
    output = {field: np.empty(n, dtype=float) for field in fields}
    for name in PROFILE_NAMES:
        idx = np.flatnonzero(assignments == name)
        if len(idx) == 0:
            continue
        block = _sample_profile_parameters(rng, name, len(idx))
        for field in fields:
            output[field][idx] = block[field]
    return output, assignments


def _negative_binomial_counts(
    rng: np.random.Generator,
    mean: float,
    dispersion: float,
    size: int,
) -> np.ndarray:
    if mean <= 0:
        return np.zeros(size, dtype=int)
    p = dispersion / (dispersion + mean)
    return rng.negative_binomial(dispersion, p, size=size).astype(int)


def _safe_clip_state(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -1.0, 1.0)


class PDTRTRerunSimulator:
    def __init__(self, config: RerunConfig):
        self.config = config
        self.rng_network = np.random.default_rng(stable_seed(config.seed, "network"))
        self.rng_environment = np.random.default_rng(
            stable_seed(config.seed, "environment")
        )
        self.rng_focal = np.random.default_rng(stable_seed(config.seed, "focal"))
        self.rng_background = np.random.default_rng(
            stable_seed(config.seed, "background_parameters")
        )
        self.rng_states = np.random.default_rng(
            stable_seed(config.seed, "initial_states")
        )
        self.rng_protocol = np.random.default_rng(
            stable_seed(config.seed, "protocol")
        )
        self.rng_events = np.random.default_rng(stable_seed(config.seed, "events"))
        self.rng_outcomes = np.random.default_rng(stable_seed(config.seed, "outcomes"))
        self.rng_reports = np.random.default_rng(stable_seed(config.seed, "reports"))
        self.rng_missing = np.random.default_rng(stable_seed(config.seed, "missingness"))

        self.coords: np.ndarray
        self.focus: np.ndarray
        self.activity: np.ndarray
        self.popularity: np.ndarray
        self.neighbors: List[np.ndarray]
        self.neighbor_probs: List[np.ndarray]
        self.relationships: Dict[Tuple[int, int], Dict[str, object]]
        self.focal_network_ids: np.ndarray
        self.focal_lookup: Dict[int, int]
        self.parameters: Dict[str, np.ndarray]
        self.background_assignments: np.ndarray
        self.states: np.ndarray
        self.last_choice: np.ndarray
        self.last_enjoyment: np.ndarray
        self.last_utility: np.ndarray
        self.self_event_probability: np.ndarray
        self.suggestion_probability: np.ndarray
        self.feedback_probability: np.ndarray
        self.missing_p10: np.ndarray
        self.missing_p20: np.ndarray

    def _build_network(self) -> pd.DataFrame:
        cfg = self.config
        n = cfg.network_size
        rng = self.rng_network
        self.coords = rng.normal(0.0, 1.0, size=(n, 2))
        self.focus = rng.integers(0, cfg.n_social_foci, size=n)
        self.activity = rng.lognormal(0.0, 0.55, size=n)
        self.popularity = rng.lognormal(0.0, 0.65, size=n)

        adjacency: List[set[int]] = [set() for _ in range(n)]
        outgoing_mean = max(1.0, cfg.mean_degree / 2.0)
        for i in range(n):
            target = int(np.clip(rng.poisson(outgoing_mean), 1, max(1, n - 1)))
            diff = self.coords - self.coords[i]
            distance = np.sqrt(np.sum(diff * diff, axis=1))
            scale = max(1e-6, cfg.homophily_scale)
            weights = np.exp(-(distance**2) / (2.0 * scale**2))
            weights *= self.popularity
            weights *= np.where(self.focus == self.focus[i], 2.0, 1.0)
            weights[i] = 0.0
            if np.sum(weights) <= 0:
                weights = np.ones(n, dtype=float)
                weights[i] = 0.0
            chosen = rng.choice(n, size=min(target, n - 1), replace=False, p=weights / weights.sum())
            for j_raw in chosen:
                j = int(j_raw)
                adjacency[i].add(j)
                adjacency[j].add(i)

        for i in range(n):
            if adjacency[i]:
                continue
            distance = np.sqrt(np.sum((self.coords - self.coords[i]) ** 2, axis=1))
            distance[i] = np.inf
            j = int(np.argmin(distance))
            adjacency[i].add(j)
            adjacency[j].add(i)

        edge_rows: List[Dict[str, float]] = []
        self.relationships = {}
        all_distances = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    all_distances.append(float(np.linalg.norm(self.coords[i] - self.coords[j])))
        distance_scale = float(np.quantile(all_distances, 0.95)) if all_distances else 1.0
        distance_scale = max(distance_scale, 1e-6)

        for i in range(n):
            for j in sorted(adjacency[i]):
                raw_distance = float(np.linalg.norm(self.coords[i] - self.coords[j]))
                distance = float(np.clip(raw_distance / distance_scale, 0.0, 1.0))
                if (i, j) not in self.relationships:
                    warmth = float(np.clip(0.65 - 1.05 * distance + rng.normal(0.0, 0.20), -1.0, 1.0))
                    receptivity = float(
                        np.clip(0.05 + 0.55 * (1.0 - distance) + 0.20 * warmth + rng.normal(0.0, 0.18), -1.0, 1.0)
                    )
                    strength = float(
                        max(
                            0.01,
                            np.exp(-1.8 * distance)
                            * (1.0 + 0.45 * max(0.0, warmth))
                            * self.popularity[j],
                        )
                    )
                    env = ENVIRONMENT_SPECS[cfg.environment]
                    mean = np.asarray(env["context_mean"], dtype=float)
                    concentration = float(env["context_concentration"])
                    context_probs = self.rng_environment.dirichlet(
                        np.maximum(0.05, mean * concentration)
                    )
                    self.relationships[(i, j)] = {
                        "distance": distance,
                        "warmth": warmth,
                        "receptivity": receptivity,
                        "strength": strength,
                        "context_probs": context_probs,
                        "enjoyment_offset": float(rng.normal(0.0, 0.08)),
                        "utility_offset": float(rng.normal(0.0, 0.10)),
                    }
                if i < j:
                    edge_rows.append(
                        {
                            "source": i,
                            "target": j,
                            "distance": distance,
                            "same_focus": int(self.focus[i] == self.focus[j]),
                        }
                    )

        self.neighbors = []
        self.neighbor_probs = []
        for i in range(n):
            nbr = np.asarray(sorted(adjacency[i]), dtype=int)
            weights = np.asarray([float(self.relationships[(i, int(j))]["strength"]) for j in nbr], dtype=float)
            self.neighbors.append(nbr)
            self.neighbor_probs.append(weights / weights.sum())

        return pd.DataFrame(edge_rows)

    def _network_diagnostics(self, edges: pd.DataFrame) -> Dict[str, float]:
        n = self.config.network_size
        degrees = np.asarray([len(nbr) for nbr in self.neighbors], dtype=float)
        local_clustering = []
        adjacency_sets = [set(int(x) for x in nbr) for nbr in self.neighbors]
        for i, neighbors_i in enumerate(adjacency_sets):
            degree = len(neighbors_i)
            if degree < 2:
                local_clustering.append(0.0)
                continue
            links = 0
            neighbors_list = list(neighbors_i)
            for a_idx in range(degree):
                a = neighbors_list[a_idx]
                for b in neighbors_list[a_idx + 1 :]:
                    if b in adjacency_sets[a]:
                        links += 1
            local_clustering.append(2.0 * links / (degree * (degree - 1)))

        visited = np.zeros(n, dtype=bool)
        component_sizes = []
        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for nxt in self.neighbors[node]:
                    nxt_i = int(nxt)
                    if not visited[nxt_i]:
                        visited[nxt_i] = True
                        stack.append(nxt_i)
            component_sizes.append(size)

        directed_rels = list(self.relationships.values())
        strengths = np.asarray(
            [float(relationship["strength"]) for relationship in directed_rels]
        )
        warmth_forward = []
        warmth_reverse = []
        receptivity_forward = []
        receptivity_reverse = []
        strength_asymmetry = []
        for _, edge in edges.iterrows():
            source = int(edge["source"])
            target = int(edge["target"])
            forward = self.relationships[(source, target)]
            reverse = self.relationships[(target, source)]
            warmth_forward.append(float(forward["warmth"]))
            warmth_reverse.append(float(reverse["warmth"]))
            receptivity_forward.append(float(forward["receptivity"]))
            receptivity_reverse.append(float(reverse["receptivity"]))
            strength_asymmetry.append(
                abs(float(forward["strength"]) - float(reverse["strength"]))
            )

        sampled_nodes = np.linspace(
            0,
            n - 1,
            num=min(n, 100),
            dtype=int,
        )
        sampled_distances = []
        for start in sampled_nodes:
            distances = {int(start): 0}
            queue = [int(start)]
            cursor = 0
            while cursor < len(queue):
                node = queue[cursor]
                cursor += 1
                for neighbor in self.neighbors[node]:
                    neighbor_id = int(neighbor)
                    if neighbor_id in distances:
                        continue
                    distances[neighbor_id] = distances[node] + 1
                    queue.append(neighbor_id)
            sampled_distances.extend(
                distance
                for node, distance in distances.items()
                if node != int(start)
            )

        def paired_correlation(first: Sequence[float], second: Sequence[float]) -> float:
            first_array = np.asarray(first, dtype=float)
            second_array = np.asarray(second, dtype=float)
            if len(first_array) < 3 or np.std(first_array) <= 1e-12 or np.std(second_array) <= 1e-12:
                return np.nan
            return float(np.corrcoef(first_array, second_array)[0, 1])

        return {
            "network_size": float(n),
            "edge_count": float(len(edges)),
            "mean_degree": float(np.mean(degrees)),
            "degree_sd": float(np.std(degrees, ddof=1)) if n > 1 else 0.0,
            "degree_min": float(np.min(degrees)),
            "degree_max": float(np.max(degrees)),
            "mean_clustering": float(np.mean(local_clustering)),
            "component_count": float(len(component_sizes)),
            "largest_component_fraction": float(max(component_sizes) / n),
            "same_focus_edge_fraction": float(edges["same_focus"].mean()) if not edges.empty else np.nan,
            "mean_tie_distance": float(np.mean([float(x["distance"]) for x in directed_rels])),
            "mean_tie_warmth": float(np.mean([float(x["warmth"]) for x in directed_rels])),
            "mean_tie_receptivity": float(np.mean([float(x["receptivity"]) for x in directed_rels])),
            "mean_tie_strength": float(np.mean(strengths)),
            "tie_strength_cv": float(np.std(strengths, ddof=1) / np.mean(strengths)),
            "mean_abs_strength_asymmetry": float(np.mean(strength_asymmetry)),
            "dyad_warmth_direction_correlation": paired_correlation(
                warmth_forward,
                warmth_reverse,
            ),
            "dyad_receptivity_direction_correlation": paired_correlation(
                receptivity_forward,
                receptivity_reverse,
            ),
            "structural_reciprocity": 1.0,
            "mean_shortest_path_sampled": float(np.mean(sampled_distances)),
        }

    def _relationship_table(self) -> pd.DataFrame:
        rows = []
        for (source, target), relationship in sorted(self.relationships.items()):
            context_probs = np.asarray(
                relationship["context_probs"],
                dtype=float,
            )
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "distance": float(relationship["distance"]),
                    "warmth": float(relationship["warmth"]),
                    "receptivity": float(relationship["receptivity"]),
                    "strength": float(relationship["strength"]),
                    "context_prob_repair_opportunity": float(context_probs[0]),
                    "context_prob_escalation_risk": float(context_probs[1]),
                    "context_prob_ambiguous_conflict": float(context_probs[2]),
                    "enjoyment_offset": float(relationship["enjoyment_offset"]),
                    "utility_offset": float(relationship["utility_offset"]),
                }
            )
        return pd.DataFrame(rows)

    def _choose_focal_people(self) -> None:
        eligible = np.asarray([i for i, nbr in enumerate(self.neighbors) if len(nbr) > 0], dtype=int)
        chosen = self.rng_focal.choice(
            eligible,
            size=self.config.max_focal,
            replace=False,
        )
        self.focal_network_ids = np.asarray(chosen, dtype=int)
        self.focal_lookup = {int(network_id): rank for rank, network_id in enumerate(self.focal_network_ids)}

    def _build_population(self) -> None:
        cfg = self.config
        n = cfg.network_size
        params, assignments = _sample_background_parameters(self.rng_background, n)
        focal_params = _sample_paired_focal_parameters(
            cfg.seed,
            cfg.profile,
            cfg.max_focal,
        )
        for field, values in focal_params.items():
            params[field][self.focal_network_ids] = values
        assignments = assignments.astype(object)
        assignments[self.focal_network_ids] = cfg.profile
        self.parameters = params
        self.background_assignments = assignments

        initial_target = np.stack([ENJOYMENT_MEANS, UTILITY_MEANS], axis=-1)
        state = np.empty((n, len(CONTEXTS), len(BEHAVIORS), len(STATE_NAMES)), dtype=float)
        state[..., 0] = self.rng_states.normal(
            0.0,
            0.20,
            size=(n, len(CONTEXTS), len(BEHAVIORS)),
        )
        state[..., 1] = (
            cfg.initial_prior_alignment * initial_target[None, :, :, 0]
            + self.rng_states.normal(
                0.0,
                cfg.state_sd,
                size=(n, len(CONTEXTS), len(BEHAVIORS)),
            )
        )
        state[..., 2] = (
            cfg.initial_prior_alignment * initial_target[None, :, :, 1]
            + self.rng_states.normal(
                0.0,
                cfg.state_sd,
                size=(n, len(CONTEXTS), len(BEHAVIORS)),
            )
        )
        self.states = _safe_clip_state(state)

        self.last_choice = np.full(n, 0.5, dtype=float)
        self.last_enjoyment = np.zeros(n, dtype=float)
        self.last_utility = np.zeros(n, dtype=float)
        self.self_event_probability = self.rng_protocol.beta(
            6.0,
            2.0,
            size=cfg.max_focal,
        )
        self.suggestion_probability = self.rng_protocol.beta(3.0, 4.0, size=n)
        self.feedback_probability = self.rng_protocol.beta(3.0, 4.0, size=n)

        person_shift = self.rng_missing.normal(0.0, cfg.missing_person_sd, size=cfg.max_focal)
        p10 = expit(logit(0.10) + person_shift)
        p20 = expit(logit(0.20) + person_shift)
        self.missing_p10 = np.minimum(p10, p20)
        self.missing_p20 = np.maximum(p10, p20)

    def _relationship(self, source: int, target: int) -> Dict[str, object]:
        rel = self.relationships[(source, target)]
        adjusted = dict(rel)
        adjusted["receptivity"] = float(
            np.clip(
                float(rel["receptivity"]) + self.parameters["receptivity_bias"][source],
                -1.0,
                1.0,
            )
        )
        return adjusted

    def _choose_partner(self, actor: int) -> int:
        return int(self.rng_events.choice(self.neighbors[actor], p=self.neighbor_probs[actor]))

    def _choice_values(
        self,
        actor: int,
        partner: int,
        context_idx: int,
        suggestion_active: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        state = self.states[actor, context_idx]
        rel = self._relationship(actor, partner)
        partner_state = self.states[partner, context_idx]
        partner_weights = np.array(
            [
                self.parameters["w_i"][partner],
                self.parameters["w_e"][partner],
                self.parameters["w_u"][partner],
            ]
        )
        partner_values = partner_state @ partner_weights
        suggestion = np.zeros(len(BEHAVIORS), dtype=float)
        if suggestion_active:
            suggestion = float(rel["receptivity"]) * (partner_values - np.mean(partner_values))

        if self.config.generator_model == "collapsed_reward":
            w_i = self.parameters["w_i"][actor]
            reward = 0.5 * (state[:, 1] + state[:, 2])
            values = w_i * state[:, 0] + (1.0 - w_i) * reward
        elif self.config.generator_model == "lagged":
            context_effect = np.array([-0.75, 0.75, 0.0], dtype=float)[context_idx]
            delta = (
                context_effect
                + 1.00 * (self.last_choice[actor] - 0.5)
                + 0.45 * self.last_utility[actor]
                - 0.20 * self.last_enjoyment[actor]
                + self.parameters["social_kappa"][actor] * (suggestion[1] - suggestion[0])
            )
            values = np.array([-0.5 * delta, 0.5 * delta], dtype=float)
            suggestion = np.zeros(len(BEHAVIORS), dtype=float)
        else:
            weights = np.array(
                [
                    self.parameters["w_i"][actor],
                    self.parameters["w_e"][actor],
                    self.parameters["w_u"][actor],
                ]
            )
            values = state @ weights

        values = values + self.parameters["social_kappa"][actor] * suggestion
        values = values + self.rng_events.normal(0.0, self.parameters["noise_s"][actor], size=2)
        return values, suggestion

    def _select_behavior(self, actor: int, values: np.ndarray) -> Tuple[int, float]:
        tau = self.parameters["tau"][actor]
        probs = softmax(tau * values)
        behavior = int(self.rng_events.choice(2, p=probs))
        return behavior, float(probs[1])

    def _generate_outcome(
        self,
        actor: int,
        partner: int,
        context_idx: int,
        behavior_idx: int,
        feedback_active: bool,
    ) -> Dict[str, float]:
        rel = self._relationship(actor, partner)
        raw_e = (
            ENJOYMENT_MEANS[context_idx, behavior_idx]
            + float(rel["enjoyment_offset"])
            + self.config.outcome_relationship_scale * float(rel["warmth"])
            + self.rng_outcomes.normal(0.0, ENJOYMENT_SDS[context_idx, behavior_idx])
        )
        raw_u = (
            UTILITY_MEANS[context_idx, behavior_idx]
            + float(rel["utility_offset"])
            + 0.5 * self.config.outcome_relationship_scale * float(rel["warmth"])
            + self.rng_outcomes.normal(0.0, UTILITY_SDS[context_idx, behavior_idx])
        )
        actor_expected_u = self.states[actor, context_idx, behavior_idx, 2]
        partner_expected_u = self.states[partner, context_idx, behavior_idx, 2]
        feedback = 0.0
        if feedback_active:
            feedback = float(rel["receptivity"]) * 0.20 * (partner_expected_u - actor_expected_u)
        perceived_e = float(np.clip(raw_e, -1.0, 1.0))
        perceived_u = float(np.clip(raw_u + feedback, -1.0, 1.0))
        return {
            "raw_enjoyment": float(np.clip(raw_e, -1.0, 1.0)),
            "raw_utility": float(np.clip(raw_u, -1.0, 1.0)),
            "perceived_enjoyment": perceived_e,
            "perceived_utility": perceived_u,
            "feedback": feedback,
        }

    def _update_tripartite(
        self,
        learner: int,
        context_idx: int,
        behavior_idx: int,
        enjoyment: float,
        utility: float,
        gain: float,
    ) -> None:
        state = self.states[learner, context_idx]
        chosen = behavior_idx
        other = 1 - behavior_idx
        state[chosen, 0] += (
            gain
            * self.parameters["alpha_i_pos"][learner]
            * (1.0 - state[chosen, 0])
        )
        state[other, 0] += (
            gain
            * self.parameters["alpha_i_neg"][learner]
            * (-1.0 - state[other, 0])
        )
        state[chosen, 1] += (
            gain
            * self.parameters["alpha_e"][learner]
            * (enjoyment - state[chosen, 1])
        )
        state[chosen, 2] += (
            gain
            * self.parameters["alpha_u"][learner]
            * (utility - state[chosen, 2])
        )
        self.states[learner, context_idx] = _safe_clip_state(state)

    def _update_collapsed(
        self,
        learner: int,
        context_idx: int,
        behavior_idx: int,
        enjoyment: float,
        utility: float,
        gain: float,
    ) -> None:
        reward = 0.5 * (enjoyment + utility)
        state = self.states[learner, context_idx]
        chosen = behavior_idx
        other = 1 - behavior_idx
        state[chosen, 0] += (
            gain
            * self.parameters["alpha_i_pos"][learner]
            * (1.0 - state[chosen, 0])
        )
        state[other, 0] += (
            gain
            * self.parameters["alpha_i_neg"][learner]
            * (-1.0 - state[other, 0])
        )
        alpha_r = 0.5 * (
            self.parameters["alpha_e"][learner] + self.parameters["alpha_u"][learner]
        )
        for state_idx in (1, 2):
            state[chosen, state_idx] += gain * alpha_r * (reward - state[chosen, state_idx])
        self.states[learner, context_idx] = _safe_clip_state(state)

    def _apply_learning(
        self,
        actor: int,
        observer: int,
        context_idx: int,
        behavior_idx: int,
        outcome: Mapping[str, float],
    ) -> None:
        model = self.config.generator_model
        if model == "no_learning":
            return
        if model == "lagged":
            self.last_choice[actor] = float(behavior_idx)
            self.last_enjoyment[actor] = float(outcome["perceived_enjoyment"])
            self.last_utility[actor] = float(outcome["perceived_utility"])
            self.last_choice[observer] = float(behavior_idx)
            self.last_enjoyment[observer] = float(outcome["raw_enjoyment"])
            self.last_utility[observer] = float(outcome["raw_utility"])
            return

        observer_rel = self._relationship(observer, actor)
        observer_gain = self.config.observation_attenuation * (
            0.5 + 0.5 * float(observer_rel["receptivity"])
        )
        observer_gain = float(np.clip(observer_gain, 0.0, self.config.observation_attenuation))
        if model == "collapsed_reward":
            update = self._update_collapsed
        else:
            update = self._update_tripartite
        update(
            actor,
            context_idx,
            behavior_idx,
            float(outcome["perceived_enjoyment"]),
            float(outcome["perceived_utility"]),
            1.0,
        )
        update(
            observer,
            context_idx,
            behavior_idx,
            float(outcome["raw_enjoyment"]),
            float(outcome["raw_utility"]),
            observer_gain,
        )

    def _simulate_event(
        self,
        timestamp: float,
        event_id: int,
        scheduled_focal_rank: int | None,
    ) -> Tuple[Dict[str, object], Dict[str, object] | None]:
        if scheduled_focal_rank is None:
            actor = int(self.rng_events.choice(self.config.network_size, p=self.activity / self.activity.sum()))
            partner = self._choose_partner(actor)
            observer = partner
            report_role = None
            focal_network_id = None
        else:
            focal_network_id = int(self.focal_network_ids[scheduled_focal_rank])
            if self.rng_events.random() < self.self_event_probability[scheduled_focal_rank]:
                actor = focal_network_id
                partner = self._choose_partner(actor)
                observer = partner
                report_role = "self"
            else:
                observer = focal_network_id
                actor = self._choose_partner(observer)
                partner = observer
                report_role = "observe"

        rel = self._relationship(actor, partner)
        context_idx = int(self.rng_events.choice(len(CONTEXTS), p=np.asarray(rel["context_probs"], dtype=float)))
        suggestion_active = bool(self.rng_events.random() < self.suggestion_probability[actor])
        feedback_active = bool(self.rng_events.random() < self.feedback_probability[partner])

        focal_pre = None
        if focal_network_id is not None:
            focal_pre = self.states[focal_network_id, context_idx].copy()
        values, suggestion = self._choice_values(actor, partner, context_idx, suggestion_active)
        behavior_idx, approach_probability = self._select_behavior(actor, values)
        outcome = self._generate_outcome(
            actor,
            partner,
            context_idx,
            behavior_idx,
            feedback_active,
        )
        self._apply_learning(actor, observer, context_idx, behavior_idx, outcome)
        focal_post = None
        if focal_network_id is not None:
            focal_post = self.states[focal_network_id, context_idx].copy()

        truth_row: Dict[str, object] = {
            "event_id": event_id,
            "timestamp_day": timestamp,
            "scheduled_focal_rank": scheduled_focal_rank,
            "report_role": report_role,
            "actor_network_id": actor,
            "observer_network_id": observer,
            "partner_network_id": partner,
            "context_idx": context_idx,
            "context": CONTEXTS[context_idx],
            "behavior_idx": behavior_idx,
            "behavior": BEHAVIORS[behavior_idx],
            "approach_probability_true": approach_probability,
            "suggestion_active": int(suggestion_active),
            "feedback_active": int(feedback_active),
            "suggestion_avoid_true": float(suggestion[0]),
            "suggestion_approach_true": float(suggestion[1]),
            **outcome,
        }
        if focal_pre is not None and focal_post is not None:
            for b_idx, behavior in enumerate(BEHAVIORS):
                for s_idx, state_name in enumerate(STATE_NAMES):
                    truth_row[f"focal_pre_{state_name}_{behavior}"] = float(focal_pre[b_idx, s_idx])
                    truth_row[f"focal_post_{state_name}_{behavior}"] = float(focal_post[b_idx, s_idx])

        if scheduled_focal_rank is None or focal_network_id is None:
            return truth_row, None

        focal_rel = self._relationship(focal_network_id, actor if report_role == "observe" else partner)
        if report_role == "self":
            focal_e = float(outcome["perceived_enjoyment"])
            focal_u = float(outcome["perceived_utility"])
            suggestion_report = suggestion.copy()
            feedback_value = float(outcome["feedback"])
            choice_behavior = behavior_idx
        else:
            focal_e = float(outcome["raw_enjoyment"])
            focal_u = float(outcome["raw_utility"])
            suggestion_report = np.zeros(2, dtype=float)
            feedback_value = 0.0
            choice_behavior = np.nan

        miss_u = float(self.rng_missing.random())
        report_row: Dict[str, object] = {
            "event_id": event_id,
            "focal_id": scheduled_focal_rank,
            "focal_network_id": focal_network_id,
            "timestamp_day": timestamp,
            "role": report_role,
            "context_idx": context_idx,
            "context": CONTEXTS[context_idx],
            "partner_id": actor if report_role == "observe" else partner,
            "behavior_idx": behavior_idx,
            "behavior": BEHAVIORS[behavior_idx],
            "choice_behavior": choice_behavior,
            "suggestion_avoid": float(
                suggestion_report[0]
                + self.rng_reports.normal(0.0, self.config.suggestion_report_sd)
            ),
            "suggestion_approach": float(
                suggestion_report[1]
                + self.rng_reports.normal(0.0, self.config.suggestion_report_sd)
            ),
            "feedback": float(
                feedback_value + self.rng_reports.normal(0.0, self.config.feedback_report_sd)
            ),
            "enjoyment_out": float(
                np.clip(
                    focal_e + self.rng_reports.normal(0.0, self.config.outcome_report_sd),
                    -1.0,
                    1.0,
                )
            ),
            "utility_out": float(
                np.clip(
                    focal_u + self.rng_reports.normal(0.0, self.config.outcome_report_sd),
                    -1.0,
                    1.0,
                )
            ),
            "relationship_distance": float(
                np.clip(
                    float(focal_rel["distance"])
                    + self.rng_reports.normal(0.0, self.config.relationship_report_sd),
                    0.0,
                    1.0,
                )
            ),
            "relationship_warmth": float(
                np.clip(
                    float(focal_rel["warmth"])
                    + self.rng_reports.normal(0.0, self.config.relationship_report_sd),
                    -1.0,
                    1.0,
                )
            ),
            "relationship_receptivity": float(
                np.clip(
                    float(focal_rel["receptivity"])
                    + self.rng_reports.normal(0.0, self.config.relationship_report_sd),
                    -1.0,
                    1.0,
                )
            ),
            "miss_u": miss_u,
            "miss_p10": float(self.missing_p10[scheduled_focal_rank]),
            "miss_p20": float(self.missing_p20[scheduled_focal_rank]),
            "eval_common": bool(report_role == "self" and miss_u >= self.missing_p20[scheduled_focal_rank]),
        }
        return truth_row, report_row

    def _people_tables(self, baseline_truth: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        truth_rows = []
        observed_rows = []
        for focal_id, network_id_raw in enumerate(self.focal_network_ids):
            network_id = int(network_id_raw)
            split = "test" if focal_id % 5 == 0 else "train"
            true_row: Dict[str, object] = {
                "focal_id": focal_id,
                "network_id": network_id,
                "panel_rank": focal_id,
                "split": split,
                "profile": self.config.profile,
                "background_assignment": self.background_assignments[network_id],
                "self_event_probability": float(self.self_event_probability[focal_id]),
                "missing_p10": float(self.missing_p10[focal_id]),
                "missing_p20": float(self.missing_p20[focal_id]),
            }
            for field, values in self.parameters.items():
                true_row[f"true_{field}"] = float(values[network_id])
            observed_row = {
                "focal_id": focal_id,
                "panel_rank": focal_id,
                "split": split,
                "profile": self.config.profile,
            }
            for context_idx, context in enumerate(CONTEXTS):
                for behavior_idx, behavior in enumerate(BEHAVIORS):
                    for state_idx, state_name in enumerate(STATE_NAMES):
                        true_value = float(baseline_truth[focal_id, context_idx, behavior_idx, state_idx])
                        reported_value = float(
                            np.clip(
                                true_value
                                + self.rng_reports.normal(0.0, self.config.baseline_report_sd),
                                -1.0,
                                1.0,
                            )
                        )
                        true_row[f"initial_{state_name}_{context}_{behavior}"] = true_value
                        observed_row[f"baseline_{state_name}_{context}_{behavior}"] = reported_value
            truth_rows.append(true_row)
            observed_rows.append(observed_row)
        return pd.DataFrame(truth_rows), pd.DataFrame(observed_rows)

    def run(self) -> SimulationResult:
        edges = self._build_network()
        network_diagnostics = self._network_diagnostics(edges)
        relationships = self._relationship_table()
        self._choose_focal_people()
        self._build_population()
        baseline_truth = self.states[self.focal_network_ids].copy()

        counts = _negative_binomial_counts(
            self.rng_events,
            self.config.mean_events,
            self.config.event_dispersion,
            self.config.max_focal,
        )
        schedule: List[Tuple[float, int | None]] = []
        for focal_rank, count in enumerate(counts):
            times = self.rng_events.uniform(0.0, float(self.config.days), size=int(count))
            schedule.extend((float(time), focal_rank) for time in times)
        hidden_count = int(
            round(
                self.config.network_size
                * self.config.days
                * self.config.hidden_events_per_person_day
            )
        )
        hidden_times = self.rng_events.uniform(0.0, float(self.config.days), size=hidden_count)
        schedule.extend((float(time), None) for time in hidden_times)
        schedule.sort(key=lambda item: item[0])

        truth_rows: List[Dict[str, object]] = []
        report_rows: List[Dict[str, object]] = []
        for event_id, (timestamp, focal_rank) in enumerate(schedule):
            truth_row, report_row = self._simulate_event(timestamp, event_id, focal_rank)
            truth_rows.append(truth_row)
            if report_row is not None:
                report_rows.append(report_row)

        truth_events = pd.DataFrame(truth_rows)
        complete_reports = pd.DataFrame(report_rows)
        if not complete_reports.empty:
            complete_reports = complete_reports.sort_values(["focal_id", "timestamp_day", "event_id"]).reset_index(drop=True)
            complete_reports["event_order"] = complete_reports.groupby("focal_id").cumcount()
            complete_reports["elapsed_since_prior_report"] = (
                complete_reports.groupby("focal_id")["timestamp_day"].diff()
            )
        people_truth, people_observed = self._people_tables(baseline_truth)

        choice_reports = complete_reports.loc[complete_reports["role"] == "self"] if not complete_reports.empty else complete_reports
        context_props = (
            complete_reports["context"].value_counts(normalize=True).to_dict()
            if not complete_reports.empty
            else {}
        )
        generation_diagnostics: Dict[str, float] = {
            "eligible_event_mean": float(np.mean(counts)),
            "eligible_event_sd": float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0,
            "eligible_event_min": float(np.min(counts)) if len(counts) else 0.0,
            "eligible_event_max": float(np.max(counts)) if len(counts) else 0.0,
            "zero_event_fraction": float(np.mean(counts == 0)) if len(counts) else 0.0,
            "hidden_event_count": float(hidden_count),
            "report_event_count": float(len(complete_reports)),
            "self_event_fraction": float(np.mean(complete_reports["role"] == "self")) if len(complete_reports) else np.nan,
            "approach_rate": float(np.mean(choice_reports["behavior_idx"] == 1)) if len(choice_reports) else np.nan,
            "common_eval_count": float(complete_reports["eval_common"].sum()) if len(complete_reports) else 0.0,
            "boundary_enjoyment_fraction": float(np.mean(np.abs(truth_events["raw_enjoyment"]) >= 0.999)),
            "boundary_utility_fraction": float(np.mean(np.abs(truth_events["raw_utility"]) >= 0.999)),
        }
        if not complete_reports.empty:
            report_counts = complete_reports.groupby("focal_id").size()
            partner_counts = complete_reports.groupby(
                ["focal_id", "partner_id"]
            ).size()
            partner_shares = partner_counts / partner_counts.groupby(level=0).sum()
            max_partner_share = partner_shares.groupby(level=0).max()
            partner_coverage = complete_reports.groupby("focal_id")[
                "partner_id"
            ].nunique()
            context_coverage = complete_reports.groupby("focal_id")[
                "context"
            ].nunique()
            elapsed = complete_reports["elapsed_since_prior_report"].dropna()
            for quantile in (0.10, 0.25, 0.50, 0.75, 0.90):
                label = int(round(100 * quantile))
                generation_diagnostics[f"eligible_event_count_q{label:02d}"] = float(
                    np.quantile(counts, quantile)
                )
                generation_diagnostics[f"elapsed_days_q{label:02d}"] = (
                    float(np.quantile(elapsed, quantile))
                    if len(elapsed)
                    else np.nan
                )
            generation_diagnostics.update(
                {
                    "observed_role_event_count": float(
                        np.sum(complete_reports["role"] == "observe")
                    ),
                    "self_role_event_count": float(
                        np.sum(complete_reports["role"] == "self")
                    ),
                    "suggestion_nonzero_fraction": float(
                        np.mean(
                            np.abs(
                                complete_reports["suggestion_approach"]
                                - complete_reports["suggestion_avoid"]
                            )
                            > 0.05
                        )
                    ),
                    "feedback_nonzero_fraction": float(
                        np.mean(np.abs(complete_reports["feedback"]) > 0.05)
                    ),
                    "mean_partner_coverage": float(partner_coverage.mean()),
                    "mean_context_coverage": float(context_coverage.mean()),
                    "mean_max_partner_event_share": float(max_partner_share.mean()),
                    "report_count_sd": float(report_counts.std(ddof=1)),
                }
            )
        for context in CONTEXTS:
            generation_diagnostics[f"context_prop_{context}"] = float(context_props.get(context, 0.0))

        return SimulationResult(
            config=self.config,
            truth_events=truth_events,
            complete_reports=complete_reports,
            people_truth=people_truth,
            people_observed=people_observed,
            network_edges=edges,
            network_relationships=relationships,
            network_diagnostics=network_diagnostics,
            generation_diagnostics=generation_diagnostics,
        )


def build_panel_view(
    result: SimulationResult,
    sample_size: int,
    missing_rate: float,
) -> PanelView:
    if sample_size > result.config.max_focal:
        raise ValueError("sample_size exceeds the generated focal panel")
    if missing_rate not in (0.0, 0.1, 0.2):
        raise ValueError("missing_rate must be one of 0.0, 0.1, or 0.2")

    selected = set(range(sample_size))
    complete = result.complete_reports.loc[result.complete_reports["focal_id"].isin(selected)].copy()
    if missing_rate == 0.0:
        keep = np.ones(len(complete), dtype=bool)
    elif missing_rate == 0.1:
        keep = complete["miss_u"].to_numpy() >= complete["miss_p10"].to_numpy()
    else:
        keep = complete["miss_u"].to_numpy() >= complete["miss_p20"].to_numpy()
    events = complete.loc[keep].copy()
    events["missing_rate"] = missing_rate
    events = events.sort_values(["focal_id", "timestamp_day", "event_id"]).reset_index(drop=True)
    events["observed_event_order"] = events.groupby("focal_id").cumcount()
    events["elapsed_since_prior_observed"] = events.groupby("focal_id")["timestamp_day"].diff()

    private_columns = {
        "miss_u",
        "miss_p10",
        "miss_p20",
    }
    events = events.drop(columns=[column for column in private_columns if column in events])
    people = result.people_observed.loc[result.people_observed["focal_id"].isin(selected)].copy()
    people_truth = result.people_truth.loc[result.people_truth["focal_id"].isin(selected)].copy()

    complete_counts = complete.groupby("focal_id").size().reindex(range(sample_size), fill_value=0)
    observed_counts = events.groupby("focal_id").size().reindex(range(sample_size), fill_value=0)
    choice = events.loc[events["role"] == "self"]
    diagnostics = {
        "sample_size_enrolled": float(sample_size),
        "sample_size_observed": float(np.sum(observed_counts > 0)),
        "sample_size_with_choice": float(choice["focal_id"].nunique()),
        "complete_event_mean": float(complete_counts.mean()),
        "observed_event_mean": float(observed_counts.mean()),
        "realized_missing_fraction": float(1.0 - len(events) / len(complete)) if len(complete) else 0.0,
        "choice_event_count": float(len(choice)),
        "common_eval_count": float(events["eval_common"].sum()) if len(events) else 0.0,
        "approach_rate": float(np.mean(choice["behavior_idx"] == 1)) if len(choice) else np.nan,
        "one_class_person_fraction": _one_class_fraction(choice, sample_size),
    }
    return PanelView(
        sample_size=sample_size,
        missing_rate=missing_rate,
        events=events,
        people=people,
        people_truth=people_truth,
        diagnostics=diagnostics,
    )


def _one_class_fraction(choice_events: pd.DataFrame, sample_size: int) -> float:
    if sample_size == 0:
        return np.nan
    n_classes = choice_events.groupby("focal_id")["behavior_idx"].nunique()
    n_classes = n_classes.reindex(range(sample_size), fill_value=0)
    return float(np.mean(n_classes <= 1))


def validate_nested_views(views: Sequence[PanelView]) -> Dict[str, bool]:
    by_key = {(view.sample_size, view.missing_rate): view for view in views}
    sample_sizes = sorted({view.sample_size for view in views})
    missing_rates = sorted({view.missing_rate for view in views})
    checks: Dict[str, bool] = {}

    for sample_size in sample_sizes:
        event_sets = {
            missing_rate: set(by_key[(sample_size, missing_rate)].events["event_id"].astype(int))
            for missing_rate in missing_rates
        }
        if {0.0, 0.1, 0.2}.issubset(event_sets):
            checks[f"missing_nested_N{sample_size}"] = (
                event_sets[0.2].issubset(event_sets[0.1])
                and event_sets[0.1].issubset(event_sets[0.0])
            )
        common_sets = {
            missing_rate: set(
                by_key[(sample_size, missing_rate)]
                .events.loc[lambda frame: frame["eval_common"], "event_id"]
                .astype(int)
            )
            for missing_rate in missing_rates
        }
        if common_sets:
            first = next(iter(common_sets.values()))
            checks[f"common_eval_equal_N{sample_size}"] = all(values == first for values in common_sets.values())

    for missing_rate in missing_rates:
        for smaller, larger in zip(sample_sizes, sample_sizes[1:]):
            small_people = set(by_key[(smaller, missing_rate)].people["focal_id"].astype(int))
            large_people = set(by_key[(larger, missing_rate)].people["focal_id"].astype(int))
            checks[f"sample_nested_M{int(100*missing_rate)}_N{smaller}_{larger}"] = small_people.issubset(large_people)
    return checks


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    tmp.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(tmp, index=index, compression=compression)
    tmp.replace(path)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_simulation_result(result: SimulationResult, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(outdir / "config.json", asdict(result.config))
    atomic_write_json(
        outdir / "generation_diagnostics.json",
        {
            "config_fingerprint": config_fingerprint(result.config),
            "network_diagnostics": result.network_diagnostics,
            "generation_diagnostics": result.generation_diagnostics,
        },
    )
    atomic_write_csv(result.truth_events, outdir / "truth_events.csv.gz")
    atomic_write_csv(result.complete_reports, outdir / "complete_reports_private.csv.gz")
    atomic_write_csv(result.people_truth, outdir / "people_truth.csv.gz")
    atomic_write_csv(result.people_observed, outdir / "people_observed.csv")
    atomic_write_csv(result.network_edges, outdir / "network_edges.csv.gz")
    atomic_write_csv(
        result.network_relationships,
        outdir / "network_relationships_private.csv.gz",
    )


def load_simulation_result(outdir: Path) -> SimulationResult:
    config = RerunConfig(**json.loads((outdir / "config.json").read_text()))
    diagnostics = json.loads((outdir / "generation_diagnostics.json").read_text())
    if "network_diagnostics" in diagnostics:
        network_diagnostics = dict(diagnostics["network_diagnostics"])
        generation_diagnostics = dict(diagnostics["generation_diagnostics"])
    else:
        # Backward compatibility for pilot outputs written before diagnostics
        # were stored in separate namespaces.
        network_keys = {
            "network_size",
            "edge_count",
            "mean_degree",
            "degree_sd",
            "degree_min",
            "degree_max",
            "mean_clustering",
            "component_count",
            "largest_component_fraction",
            "same_focus_edge_fraction",
            "mean_tie_distance",
            "mean_tie_warmth",
            "mean_tie_receptivity",
            "mean_tie_strength",
            "tie_strength_cv",
            "mean_abs_strength_asymmetry",
            "dyad_warmth_direction_correlation",
            "dyad_receptivity_direction_correlation",
            "structural_reciprocity",
            "mean_shortest_path_sampled",
        }
        network_diagnostics = {
            key: value for key, value in diagnostics.items() if key in network_keys
        }
        generation_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key not in network_keys and key != "config_fingerprint"
        }
    relationships_path = outdir / "network_relationships_private.csv.gz"
    return SimulationResult(
        config=config,
        truth_events=pd.read_csv(outdir / "truth_events.csv.gz"),
        complete_reports=pd.read_csv(outdir / "complete_reports_private.csv.gz"),
        people_truth=pd.read_csv(outdir / "people_truth.csv.gz"),
        people_observed=pd.read_csv(outdir / "people_observed.csv"),
        network_edges=pd.read_csv(outdir / "network_edges.csv.gz"),
        network_relationships=(
            pd.read_csv(relationships_path)
            if relationships_path.exists()
            else pd.DataFrame()
        ),
        network_diagnostics=network_diagnostics,
        generation_diagnostics=generation_diagnostics,
    )


def write_panel_view(view: PanelView, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(view.events, outdir / "aa_events.csv")
    atomic_write_csv(view.people, outdir / "aa_people.csv")
    atomic_write_csv(view.people_truth, outdir / "people_truth_private.csv.gz")
    atomic_write_json(outdir / "panel_diagnostics.json", view.diagnostics)
