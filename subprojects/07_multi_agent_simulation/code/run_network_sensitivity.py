from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent, Individual  # noqa: E402
from behavior import Behavior  # noqa: E402
from setup import Setup  # noqa: E402
from situation import Situation  # noqa: E402
from ema_pair_simulation import (  # noqa: E402
    EMAPairSimulator,
    MomentaryReportCfg,
    ParamDistCfg,
    SocialCfg,
    default_behaviors,
    default_contexts,
)
from run_phenotype_analysis import (  # noqa: E402
    flatten_prediction_summary,
    markdown_table,
    summarize_recovery,
)
from validate_predictions_between_person import (  # noqa: E402
    load_data,
    run_between_person_prediction,
    run_identifiability,
    run_parameter_recovery,
)


NETWORK_VARIANTS = ("full_network", "static_nonrecruited", "static_all")


def reset_registries() -> None:
    Agent._registry = {}
    Agent._debug_log = []
    Behavior._registry = {}
    Behavior._relations = {}
    Behavior._global_priors = {}
    Behavior._taxonomy = None
    Setup._registry = {}


def snapshot_agents(agents: Iterable[Individual]) -> Dict[str, Dict]:
    snapshot = {}
    for agent in agents:
        snapshot[agent.id] = {
            behavior: {setup: params.copy() for setup, params in setup_dict.items()}
            for behavior, setup_dict in agent.behaviors.items()
        }
    return snapshot


def restore_agents(agents: Iterable[Individual], snapshot: Dict[str, Dict]) -> None:
    for agent in agents:
        agent_snapshot = snapshot.get(agent.id)
        if agent_snapshot is None:
            continue
        for behavior, setup_dict in agent.behaviors.items():
            behavior_snapshot = agent_snapshot.get(behavior, {})
            for setup, params in setup_dict.items():
                setup_snapshot = behavior_snapshot.get(setup)
                if setup_snapshot is not None:
                    params.clear()
                    params.update(setup_snapshot)


def stable_variant_offset(name: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 10000)


class DynamicNetworkSimulator(EMAPairSimulator):
    """Project 07 simulator with an evolving population network and recruited focal panel."""

    def __init__(
        self,
        *args,
        network_size: int,
        sample_size: int,
        mean_degree: float,
        homophily_scale: float,
        burnin_events: int,
        hidden_events_per_wave: int,
        p_focal_active: float,
        learning_variant: str,
        **kwargs,
    ):
        self.network_size = int(network_size)
        self.sample_size = int(sample_size)
        self.mean_degree = float(mean_degree)
        self.homophily_scale = float(homophily_scale)
        self.burnin_events = int(burnin_events)
        self.hidden_events_per_wave = int(hidden_events_per_wave)
        self.p_focal_active = float(p_focal_active)
        self.learning_variant = learning_variant
        self.population: List[Individual] = []
        self.population_params: List[Dict[str, float]] = []
        self.recruited_indices: np.ndarray | None = None
        self.recruited_set: set[int] = set()
        self.latent_coords: np.ndarray | None = None
        self.activity: np.ndarray | None = None
        self.popularity: np.ndarray | None = None
        self.neighbors: Dict[int, List[int]] = {}
        self.hidden_events = 0
        self.hidden_events_touching_recruited = 0
        super().__init__(*args, N=sample_size, **kwargs)
        self._build_population()
        self._build_network()
        self.recruited_indices = self.rng.choice(self.network_size, size=self.sample_size, replace=False)
        self.recruited_set = set(int(i) for i in self.recruited_indices)

    def _create_partner_pool(self) -> List[Dict]:
        return []

    def _build_population(self) -> None:
        self.population = []
        self.population_params = []
        for idx in range(self.network_size):
            agent, params = self._sample_person(idx)
            agent.name = f"network_agent_{idx}"
            self.population.append(agent)
            self.population_params.append(params)
        self.latent_coords = self.rng.normal(0.0, 1.0, size=(self.network_size, 2))
        self.activity = self.rng.lognormal(mean=0.0, sigma=0.55, size=self.network_size)
        self.popularity = self.rng.lognormal(mean=0.0, sigma=0.75, size=self.network_size)

    def _relationship_params(self, i: int, j: int, distance: float) -> Dict[str, float]:
        similarity = 1.0 - distance
        communion = float(np.clip(0.65 - 1.20 * distance + self.rng.normal(0.0, 0.22), -1.0, 1.0))
        receptivity = float(
            np.clip(-0.10 + 0.80 * similarity + 0.20 * communion + self.rng.normal(0.0, 0.18), -1.0, 1.0)
        )
        power = float(np.clip(self.rng.normal(0.0, 0.35), -1.0, 1.0))
        return {
            "distance": float(distance),
            "receptivity": receptivity,
            "power": power,
            "connection": communion,
        }

    def _build_network(self) -> None:
        assert self.latent_coords is not None
        assert self.popularity is not None
        n = self.network_size
        coords = self.latent_coords
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        norm_dist = np.clip(dist / np.quantile(dist[dist > 0], 0.90), 0.0, 1.0)
        affinity = np.exp(-(norm_dist ** 2) / max(1e-6, 2.0 * self.homophily_scale ** 2))
        self.neighbors = {i: [] for i in range(n)}
        edges = set()

        for i in range(n):
            target_degree = int(np.clip(self.rng.poisson(max(0.5, self.mean_degree - 1.0)) + 1, 1, n - 1))
            weights = affinity[i].copy() * self.popularity
            weights[i] = 0.0
            if weights.sum() <= 0:
                weights = np.ones(n, dtype=float)
                weights[i] = 0.0
            probs = weights / weights.sum()
            chosen = self.rng.choice(n, size=target_degree, replace=False, p=probs)
            for j_raw in chosen:
                j = int(j_raw)
                edge = tuple(sorted((i, j)))
                if edge in edges:
                    continue
                edges.add(edge)
                self.neighbors[i].append(j)
                self.neighbors[j].append(i)
                distance = float(norm_dist[i, j])
                self.population[i].relationships[self.population[j].id] = self._relationship_params(i, j, distance)
                self.population[j].relationships[self.population[i].id] = self._relationship_params(j, i, distance)

    def _choose_setup(self) -> Setup:
        return super()._choose_setup()

    def _choose_network_actor(self) -> int:
        assert self.activity is not None
        probs = self.activity / self.activity.sum()
        return int(self.rng.choice(self.network_size, p=probs))

    def _choose_neighbor(self, idx: int) -> int:
        neighbors = self.neighbors.get(idx, [])
        if not neighbors:
            choices = [j for j in range(self.network_size) if j != idx]
            return int(self.rng.choice(choices))
        weights = []
        source = self.population[idx]
        for neighbor in neighbors:
            rel = source.relationships.get(self.population[neighbor].id, {})
            tie_strength = (1.0 - rel.get("distance", 0.5)) * (1.0 + max(0.0, rel.get("connection", 0.0)))
            weights.append(max(0.01, tie_strength))
        probs = np.array(weights, dtype=float)
        probs = probs / probs.sum()
        return int(self.rng.choice(neighbors, p=probs))

    def _agents_to_restore(self, participants: Tuple[int, int]) -> List[Individual]:
        if self.learning_variant == "full_network":
            return []
        if self.learning_variant == "static_all":
            return [self.population[i] for i in participants]
        if self.learning_variant == "static_nonrecruited":
            return [self.population[i] for i in participants if i not in self.recruited_set]
        raise ValueError(f"Unknown learning variant: {self.learning_variant}")

    def _simulate_network_event(
        self,
        active_idx: int,
        partner_idx: int,
        mode: str,
        setup: Setup,
        count_hidden: bool,
    ) -> Situation:
        active = self.population[active_idx]
        partner = self.population[partner_idx]
        participants = (active_idx, partner_idx)
        agents_to_restore = self._agents_to_restore(participants)
        snapshot = snapshot_agents(agents_to_restore) if agents_to_restore else {}

        if mode == "solitary":
            environment = active
        else:
            environment = partner

        situation = Situation(
            setup.id,
            active.id,
            environment.id,
            mode,
            behaviors=self.behaviors,
        )
        situation._simulate_situation()

        if agents_to_restore:
            restore_agents(agents_to_restore, snapshot)

        if count_hidden:
            self.hidden_events += 1
            if active_idx in self.recruited_set or partner_idx in self.recruited_set:
                self.hidden_events_touching_recruited += 1
        return situation

    def _simulate_hidden_events(self, n_events: int) -> None:
        modes = np.array(["suggest", "observe_feedback", "co-participate", "solitary"], dtype=object)
        probs = np.array([0.35, 0.35, 0.15, 0.15], dtype=float)
        for _ in range(n_events):
            active_idx = self._choose_network_actor()
            partner_idx = self._choose_neighbor(active_idx)
            mode = str(self.rng.choice(modes, p=probs))
            self._simulate_network_event(active_idx, partner_idx, mode, self._choose_setup(), count_hidden=True)

    def _people_table_at_study_start(self) -> pd.DataFrame:
        rows = []
        assert self.recruited_indices is not None
        for panel_id, pop_idx_raw in enumerate(self.recruited_indices):
            pop_idx = int(pop_idx_raw)
            person = self.population[pop_idx]
            base_row = {
                "person_id": panel_id,
                "network_agent_id": pop_idx,
                **self.population_params[pop_idx],
            }
            for setup in self.setups:
                for behavior in self.behaviors:
                    b_params = person.behaviors[behavior][setup]
                    base_row[f"instinct_{behavior.name}_{setup.name}_0"] = b_params["instinct"]
                    base_row[f"enjoyment_{behavior.name}_{setup.name}_0"] = b_params["enjoyment"]
                    base_row[f"utility_{behavior.name}_{setup.name}_0"] = b_params["utility"]
                    if setup == self.setup:
                        base_row[f"instinct_{behavior.name}_0"] = b_params["instinct"]
                        base_row[f"enjoyment_{behavior.name}_0"] = b_params["enjoyment"]
                        base_row[f"utility_{behavior.name}_0"] = b_params["utility"]
            rows.append(base_row)
        return pd.DataFrame(rows)

    def _observed_event_for_focal(self, panel_id: int, pop_idx: int, wave: int) -> Dict:
        setup = self._choose_setup()
        partner_idx = self._choose_neighbor(pop_idx)
        focal_is_active = self.rng.random() < self.p_focal_active

        if focal_is_active:
            mode = str(self.rng.choice(["suggest", "observe_feedback", "solitary"], p=[0.40, 0.40, 0.20]))
            active_idx = pop_idx
            situation_type = mode
            event_partner_idx = partner_idx if mode != "solitary" else None
            sim_partner_idx = partner_idx
        else:
            mode = str(self.rng.choice(["observe_s", "observe_feedback_s"], p=[0.50, 0.50]))
            active_idx = partner_idx
            sim_partner_idx = pop_idx
            situation_type = "observe"
            event_partner_idx = partner_idx

        situation = self._simulate_network_event(
            active_idx,
            sim_partner_idx,
            mode,
            setup,
            count_hidden=False,
        )
        event_data = self._event_from_situation(situation, situation_type, setup, event_partner_idx)
        partner_rel = {}
        if event_partner_idx is not None:
            partner_rel = self.population[pop_idx].relationships.get(self.population[event_partner_idx].id, {})
        row = {
            "person_id": panel_id,
            "network_agent_id": pop_idx,
            "t": wave,
            "partner_is_recruited": bool(event_partner_idx in self.recruited_set) if event_partner_idx is not None else False,
            "tie_distance": partner_rel.get("distance", np.nan),
            "tie_receptivity": partner_rel.get("receptivity", np.nan),
            "tie_communion": partner_rel.get("connection", np.nan),
            **event_data,
        }
        for behavior in self.behaviors:
            b_params = self.population[pop_idx].behaviors[behavior][setup]
            row[f"instinct_{behavior.name}"] = b_params["instinct"]
            row[f"enjoyment_{behavior.name}"] = b_params["enjoyment"]
            row[f"utility_{behavior.name}"] = b_params["utility"]
        return row

    def network_descriptives(self) -> Dict[str, float]:
        degrees = np.array([len(self.neighbors[i]) for i in range(self.network_size)], dtype=float)
        rels = []
        for i, agent in enumerate(self.population):
            for neighbor_id, rel in agent.relationships.items():
                rels.append(rel)
        rel_df = pd.DataFrame(rels)
        recruited = self.recruited_indices if self.recruited_indices is not None else np.array([])
        return {
            "network_size": self.network_size,
            "sample_size": self.sample_size,
            "mean_degree": float(degrees.mean()),
            "degree_sd": float(degrees.std(ddof=1)) if len(degrees) > 1 else np.nan,
            "degree_max": float(degrees.max()) if len(degrees) else np.nan,
            "mean_tie_distance": float(rel_df["distance"].mean()) if not rel_df.empty else np.nan,
            "mean_tie_receptivity": float(rel_df["receptivity"].mean()) if not rel_df.empty else np.nan,
            "mean_tie_communion": float(rel_df["connection"].mean()) if not rel_df.empty else np.nan,
            "recruited_mean_degree": float(degrees[recruited].mean()) if len(recruited) else np.nan,
            "hidden_events": int(self.hidden_events),
            "hidden_events_touching_recruited": int(self.hidden_events_touching_recruited),
            "hidden_recruited_touch_rate": (
                float(self.hidden_events_touching_recruited / self.hidden_events)
                if self.hidden_events else np.nan
            ),
        }

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.t_count is None:
            raise ValueError("Dynamic network sensitivity requires a fixed t_count.")
        self._simulate_hidden_events(self.burnin_events)
        people = self._people_table_at_study_start()
        rows = []
        assert self.recruited_indices is not None
        for wave in range(int(self.t_count)):
            self._simulate_hidden_events(self.hidden_events_per_wave)
            for panel_id, pop_idx_raw in enumerate(self.recruited_indices):
                rows.append(self._observed_event_for_focal(panel_id, int(pop_idx_raw), wave))
        return pd.DataFrame(rows), people


def maybe_quiet_call(func, *args, quiet: bool, **kwargs):
    if not quiet:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def summarize_events(events: pd.DataFrame) -> Dict[str, float]:
    choice = events[events["choice_behavior"].notna()].copy()
    rows = {
        "n_events": int(len(events)),
        "n_people": int(events["person_id"].nunique()) if "person_id" in events else 0,
        "n_choice": int(len(choice)),
        "approach_rate": (
            float(np.mean(choice["choice_behavior"] == "approach_conflict_care")) if not choice.empty else np.nan
        ),
        "observed_event_rate": (
            float(np.mean(events["situation_type"] == "observe")) if "situation_type" in events else np.nan
        ),
        "mean_tie_distance_observed": float(events["tie_distance"].mean()) if "tie_distance" in events else np.nan,
        "mean_tie_receptivity_observed": float(events["tie_receptivity"].mean()) if "tie_receptivity" in events else np.nan,
        "mean_tie_communion_observed": float(events["tie_communion"].mean()) if "tie_communion" in events else np.nan,
        "partner_recruited_rate": (
            float(events["partner_is_recruited"].mean()) if "partner_is_recruited" in events else np.nan
        ),
    }
    return rows


def write_dataset(events: pd.DataFrame, people: pd.DataFrame, outdir: Path) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    events_path = outdir / "ema_events.csv"
    people_path = outdir / "ema_people.csv"
    events.to_csv(events_path, index=False)
    people.to_csv(people_path, index=False)
    return events_path, people_path


def run_validation(events_path: Path, people_path: Path, outdir: Path, args) -> Tuple[Dict[str, float], float]:
    per_person, _ = load_data(str(events_path), str(people_path))
    family_dir = outdir / "tripartite"
    family_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    _, pred_summary = run_between_person_prediction(
        per_person,
        tau_mode=args.tau_mode,
        default_tau=args.default_tau,
        outdir=family_dir,
        train_frac=args.train_frac,
        seed=args.seed,
        max_train_people=args.prediction_max_train_people,
        weight_mode=args.recovery_weight_mode,
        use_measurement_likelihood=not args.no_measurement_likelihood,
        measurement_weight=args.measurement_weight,
        choice_input_mode=args.choice_input_mode,
        model_family="tripartite",
    )
    recovery_df = pd.DataFrame()
    if not args.skip_recovery:
        recovery_df = maybe_quiet_call(
            run_parameter_recovery,
            per_person,
            tau_mode=args.tau_mode,
            default_tau=args.default_tau,
            outdir=family_dir,
            max_people=args.recovery_max_people,
            weight_mode=args.recovery_weight_mode,
            use_measurement_likelihood=not args.no_measurement_likelihood,
            measurement_weight=args.measurement_weight,
            choice_input_mode=args.choice_input_mode,
            model_family="tripartite",
            quiet=not args.verbose_validation,
        )
    if not args.skip_identifiability:
        maybe_quiet_call(
            run_identifiability,
            per_person,
            tau_mode=args.tau_mode,
            default_tau=args.default_tau,
            outdir=family_dir,
            max_people=args.identifiability_max_people,
            fit_df=recovery_df,
            weight_mode=args.recovery_weight_mode,
            use_measurement_likelihood=not args.no_measurement_likelihood,
            measurement_weight=args.measurement_weight,
            choice_input_mode=args.choice_input_mode,
            model_family="tripartite",
            quiet=not args.verbose_validation,
        )
    elapsed = time.perf_counter() - start
    summary = {
        **flatten_prediction_summary(pred_summary),
        **summarize_recovery(family_dir, "tripartite"),
    }
    return summary, elapsed


def generate_static_baseline(args, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], float]:
    reset_registries()
    start = time.perf_counter()
    behaviors = default_behaviors()
    sim = EMAPairSimulator(
        behaviors=behaviors,
        N=args.sample_size,
        t_count=args.tcount,
        social_cfg=SocialCfg(n_partners_min=args.network_size, n_partners_max=args.network_size),
        dist_cfg=ParamDistCfg(weight_mode=args.sim_weight_mode),
        momentary_cfg=MomentaryReportCfg(enabled=not args.no_momentary_reports),
        contexts=default_contexts(behaviors, mode=args.context_mode),
        seed=seed,
    )
    events, people = sim.run()
    elapsed = time.perf_counter() - start
    descriptives = {
        "network_size": np.nan,
        "sample_size": args.sample_size,
        "mean_degree": np.nan,
        "degree_sd": np.nan,
        "degree_max": np.nan,
        "mean_tie_distance": np.nan,
        "mean_tie_receptivity": np.nan,
        "mean_tie_communion": np.nan,
        "recruited_mean_degree": np.nan,
        "hidden_events": 0,
        "hidden_events_touching_recruited": 0,
        "hidden_recruited_touch_rate": np.nan,
    }
    return events, people, descriptives, elapsed


def generate_network_variant(args, variant: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], float]:
    reset_registries()
    start = time.perf_counter()
    behaviors = default_behaviors()
    sim = DynamicNetworkSimulator(
        behaviors=behaviors,
        network_size=args.network_size,
        sample_size=args.sample_size,
        t_count=args.tcount,
        mean_degree=args.mean_degree,
        homophily_scale=args.homophily_scale,
        burnin_events=args.burnin_events,
        hidden_events_per_wave=args.hidden_events_per_wave,
        p_focal_active=args.p_focal_active,
        learning_variant=variant,
        social_cfg=SocialCfg(n_partners_min=0, n_partners_max=0),
        dist_cfg=ParamDistCfg(weight_mode=args.sim_weight_mode),
        momentary_cfg=MomentaryReportCfg(enabled=not args.no_momentary_reports),
        contexts=default_contexts(behaviors, mode=args.context_mode),
        seed=seed,
    )
    events, people = sim.run()
    elapsed = time.perf_counter() - start
    return events, people, sim.network_descriptives(), elapsed


def run_condition(args, variant: str, replicate: int) -> Dict[str, float]:
    seed = args.seed + replicate * 1009 + stable_variant_offset(variant)
    run_id = f"{variant}_rep{replicate:03d}"
    run_dir = Path(args.outdir) / "runs" / run_id
    if variant == "static_pair_pool":
        events, people, network_desc, sim_time = generate_static_baseline(args, seed)
    else:
        events, people, network_desc, sim_time = generate_network_variant(args, variant, seed)
    events_path, people_path = write_dataset(events, people, run_dir)
    validation_summary, fit_time = run_validation(events_path, people_path, run_dir, args)
    row = {
        "variant": variant,
        "replicate": replicate,
        "seed": seed,
        "sim_seconds": sim_time,
        "fit_seconds": fit_time,
        "total_seconds": sim_time + fit_time,
        **network_desc,
        **summarize_events(events),
        **validation_summary,
    }
    return row


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        col for col in df.columns
        if col not in set(group_cols + ["replicate", "seed"])
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_replicates"] = int(grp["replicate"].nunique()) if "replicate" in grp else int(len(grp))
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_se"] = float(vals.sem()) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(outdir: Path, summary: pd.DataFrame, run_summary: pd.DataFrame) -> None:
    lines = [
        "# Dynamic Network Sensitivity Report",
        "",
        "This report benchmarks a small project 07 extension in which a broader agent network evolves before and during an ambulatory-assessment panel. Recruited participants contribute observed EMA rows, while other network events can remain hidden from the study dataset.",
        "",
        "## Variants",
        "",
        "- `static_pair_pool`: current-style project 07 cohort with a shared partner pool and no latent population network.",
        "- `full_network`: recruited and non-recruited agents all learn during hidden and observed network events.",
        "- `static_nonrecruited`: recruited agents learn, but non-recruited social partners are reset after each event.",
        "- `static_all`: all agents are reset after each event, producing a no-dynamic-learning data-generation check.",
        "",
        "## Runtime And Diagnostics",
        "",
    ]
    diag_cols = [
        "variant",
        "n_replicates",
        "sim_seconds_mean",
        "fit_seconds_mean",
        "total_seconds_mean",
        "hidden_events_mean",
        "hidden_recruited_touch_rate_mean",
        "full_auc_mean",
        "full_pr_auc_mean",
        "full_log_loss_mean",
        "no_learn_log_loss_mean",
        "mean_corr_mean",
        "mean_flat_fraction_mean",
    ]
    available = [col for col in diag_cols if col in summary.columns]
    if available:
        lines.append(markdown_table(summary[available]))
        lines.append("")
    if not run_summary.empty and "variant" in run_summary:
        base = summary.loc[summary["variant"] == "static_pair_pool"]
        if not base.empty and "total_seconds_mean" in summary:
            base_time = float(base["total_seconds_mean"].iloc[0])
            if base_time > 0:
                ratio_rows = []
                for _, row in summary.iterrows():
                    ratio_rows.append({
                        "variant": row["variant"],
                        "total_time_ratio_vs_static": float(row["total_seconds_mean"] / base_time),
                        "sim_time_ratio_vs_static": float(row["sim_seconds_mean"] / base["sim_seconds_mean"].iloc[0])
                        if float(base["sim_seconds_mean"].iloc[0]) > 0 else np.nan,
                    })
                lines.extend(["## Complexity Ratios", "", markdown_table(pd.DataFrame(ratio_rows)), ""])
    (outdir / "NETWORK_SENSITIVITY_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Small dynamic social-network sensitivity benchmark for project 07.")
    parser.add_argument("--outdir", default="subprojects/07_multi_agent_simulation/outputs_network_sensitivity")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--network_size", type=int, default=120)
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--tcount", type=int, default=20)
    parser.add_argument("--mean_degree", type=float, default=8.0)
    parser.add_argument("--homophily_scale", type=float, default=0.45)
    parser.add_argument("--burnin_events", type=int, default=240)
    parser.add_argument("--hidden_events_per_wave", type=int, default=120)
    parser.add_argument("--p_focal_active", type=float, default=0.70)
    parser.add_argument("--variants", default="static_pair_pool,full_network,static_nonrecruited,static_all")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--context_mode", default="varied", choices=["single", "varied", "orthogonal"])
    parser.add_argument("--sim_weight_mode", default="relative", choices=["relative", "raw"])
    parser.add_argument("--recovery_weight_mode", default="relative", choices=["relative", "raw"])
    parser.add_argument("--verbose_validation", action="store_true")
    parser.add_argument("--skip_recovery", action="store_true")
    parser.add_argument("--skip_identifiability", action="store_true")
    parser.add_argument("--recovery_max_people", type=int, default=12)
    parser.add_argument("--identifiability_max_people", type=int, default=4)
    parser.add_argument("--prediction_max_train_people", type=int, default=20)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--tau_mode", default="person", choices=["person", "fixed"])
    parser.add_argument("--default_tau", type=float, default=3.0)
    parser.add_argument("--choice_input_mode", default="reports", choices=["latent", "reports"])
    parser.add_argument("--measurement_weight", type=float, default=1.0)
    parser.add_argument("--no_measurement_likelihood", action="store_true")
    parser.add_argument("--no_momentary_reports", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    valid = {"static_pair_pool", *NETWORK_VARIANTS}
    unknown = sorted(set(variants).difference(valid))
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")

    rows = []
    total = len(variants) * args.reps
    idx = 0
    for replicate in range(args.reps):
        for variant in variants:
            idx += 1
            print(f"=== [{idx}/{total}] {variant} rep={replicate} ===", flush=True)
            rows.append(run_condition(args, variant, replicate))

    run_summary = pd.DataFrame(rows)
    run_summary.to_csv(outdir / "network_sensitivity_run_summary.csv", index=False)
    summary = aggregate(run_summary, ["variant"]) if not run_summary.empty else pd.DataFrame()
    summary.to_csv(outdir / "network_sensitivity_summary.csv", index=False)
    write_report(outdir, summary, run_summary)
    print(f"\nSaved network sensitivity outputs under: {outdir}")


if __name__ == "__main__":
    main()
