from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent, Individual
from behavior import Behavior
from setup import Setup
from situation import Situation


@dataclass
class BehaviorCfg:
    key: str
    label: str
    difficulty: float          # 0..1
    base_outcome: float        # mean baseline in [-1, 1]
    outcome_volatility: float  # st.dev of outcomes (0..1)


@dataclass
class ContextBehaviorCfg:
    utility_outcome: float
    difficulty: float
    outcome_volatility: float
    enjoyment_outcome: float | None = None
    enjoyment_volatility: float | None = None


@dataclass
class ContextCfg:
    key: str
    label: str
    probability: float
    behavior_params: Dict[str, ContextBehaviorCfg]


@dataclass
class SocialCfg:
    # Event type probabilities (should sum to <= 1.0, remainder is solitary)
    p_observe: float = 0.20
    p_suggest: float = 0.30
    p_feedback: float = 0.30
    # Number of social partners in the pool
    n_partners_min: int = 80
    n_partners_max: int = 120
    # Relationship variable ranges
    receptivity_range: Tuple[float, float] = (-0.2, 0.9)
    communion_range: Tuple[float, float] = (-0.5, 0.8)
    power_range: Tuple[float, float] = (0.0, 1.0)
    distance_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class ParamDistCfg:
    weight_dist: str = "uniform"
    alpha_dist: str = "uniform"
    tau_dist: str = "lognormal"
    fixed_noise_s: float | None = None
    weight_mode: str = "relative"


@dataclass
class MomentaryReportCfg:
    enabled: bool = True
    context_sd_urge: float = 0.20
    context_sd_enjoyment: float = 0.20
    context_sd_utility: float = 0.20
    report_sd_urge: float = 0.10
    report_sd_enjoyment: float = 0.10
    report_sd_utility: float = 0.10


def clip11(x: float) -> float:
    return max(-1.0, min(1.0, x))


class EMAPairSimulator:
    def __init__(
        self,
        behaviors: List[BehaviorCfg],
        N: int = 50,
        T_range: Tuple[int, int] = (5, 20),
        t_count: int | None = None,
        social_cfg: SocialCfg | None = None,
        dist_cfg: ParamDistCfg | None = None,
        momentary_cfg: MomentaryReportCfg | None = None,
        contexts: List[ContextCfg] | None = None,
        decision_model: str = "softmax",
        missing_rate: float = 0.0,
        missing_sd: float | None = None,
        seed: int = 42,
    ):
        self.behavior_cfgs = behaviors
        self.N = N
        self.T_range = T_range
        self.t_count = t_count
        self.social_cfg = social_cfg if social_cfg is not None else SocialCfg()
        self.dist_cfg = dist_cfg if dist_cfg is not None else ParamDistCfg()
        self.momentary_cfg = momentary_cfg if momentary_cfg is not None else MomentaryReportCfg()
        self.context_cfgs = contexts if contexts is not None else default_contexts(behaviors)
        self.decision_model = decision_model
        self.missing_rate = missing_rate
        self.missing_sd = missing_sd
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        self.setups = [Setup(name=cfg.key, description=cfg.label) for cfg in self.context_cfgs]
        self.setup = self.setups[0]
        self.setup_by_key = {setup.name: setup for setup in self.setups}
        self.context_by_key = {cfg.key: cfg for cfg in self.context_cfgs}
        self.behaviors = self._build_behaviors(behaviors)
        self.behavior_by_name = {b.name: b for b in self.behaviors}
        self.social_partners = self._create_partner_pool()

    def _build_behaviors(self, cfgs: List[BehaviorCfg]) -> List[Behavior]:
        behaviors = []
        for cfg in cfgs:
            setup_modifiers = {}
            for context_cfg, setup in zip(self.context_cfgs, self.setups):
                context_behavior = context_cfg.behavior_params.get(cfg.key)
                if context_behavior is None:
                    continue
                setup_modifiers[setup] = {
                    "base_outcome_mod": context_behavior.utility_outcome - cfg.base_outcome,
                    "difficulty_mod": context_behavior.difficulty - cfg.difficulty,
                    "outcome_volatility": context_behavior.outcome_volatility,
                    "enjoyment_outcome": (
                        context_behavior.enjoyment_outcome
                        if context_behavior.enjoyment_outcome is not None
                        else context_behavior.utility_outcome
                    ),
                    "enjoyment_volatility": (
                        context_behavior.enjoyment_volatility
                        if context_behavior.enjoyment_volatility is not None
                        else 0.10
                    ),
                }
            behaviors.append(
                Behavior(
                    name=cfg.key,
                    difficulty=cfg.difficulty,
                    base_outcome=cfg.base_outcome,
                    outcome_volatility=cfg.outcome_volatility,
                    setup_modifiers=setup_modifiers,
                )
            )
        return behaviors

    def _sample_bounded(self, dist: str, low: float, high: float) -> float:
        if dist == "uniform":
            value = self.rng.uniform(low, high)
        elif dist == "skewed":
            value = low + (high - low) * self.rng.beta(2.0, 5.0)
        elif dist == "bimodal":
            mid_low = low + 0.25 * (high - low)
            mid_high = low + 0.75 * (high - low)
            center = mid_low if self.rng.random() < 0.5 else mid_high
            value = self.rng.normal(center, 0.1 * (high - low))
        else:
            raise ValueError(f"Unknown distribution: {dist}")
        return float(np.clip(value, low, high))

    def _sample_tau(self) -> float:
        dist = self.dist_cfg.tau_dist
        if dist == "fixed":
            tau = 3.0
        elif dist == "lognormal":
            tau = float(self.rng.lognormal(mean=np.log(3.0), sigma=0.2))
        elif dist == "bimodal":
            mean = np.log(2.0) if self.rng.random() < 0.5 else np.log(5.0)
            tau = float(self.rng.lognormal(mean=mean, sigma=0.15))
        else:
            raise ValueError(f"Unknown tau distribution: {dist}")
        return float(np.clip(tau, 0.5, 10.0))

    def _sample_weight(self) -> float:
        return self._sample_bounded(self.dist_cfg.weight_dist, 0.2, 1.0)

    def _sample_weights(self) -> Tuple[float, float, float]:
        if self.dist_cfg.weight_mode == "relative":
            weights = self.rng.dirichlet([2.0, 2.0, 2.0])
            return float(weights[0]), float(weights[1]), float(weights[2])
        if self.dist_cfg.weight_mode != "raw":
            raise ValueError(f"Unknown weight_mode: {self.dist_cfg.weight_mode}")
        return self._sample_weight(), self._sample_weight(), self._sample_weight()

    def _sample_social_kappa(self) -> float:
        return self._sample_bounded("uniform", 0.0, 2.0)

    def _sample_alpha(self, low: float, high: float) -> float:
        return self._sample_bounded(self.dist_cfg.alpha_dist, low, high)

    def _sample_noise_s(self) -> float:
        if self.dist_cfg.fixed_noise_s is not None:
            return float(max(0.0, self.dist_cfg.fixed_noise_s))
        return float(self.rng.uniform(0.05, 0.20))

    def _configure_decision_agent(
        self,
        agent: Individual,
        w_I: float,
        tau: float,
        noise_s: float,
        social_kappa: float,
    ):
        agent.w_instinct = w_I
        agent.tau = tau
        agent.noise_scale = noise_s
        agent.social_kappa = social_kappa
        agent.decision_weight_mode = self.dist_cfg.weight_mode
        agent.decision_model = self.decision_model
        agent.use_momentary_appraisals = self.momentary_cfg.enabled
        agent.emit_momentary_reports = self.momentary_cfg.enabled
        agent.momentary_context_sd_urge = self.momentary_cfg.context_sd_urge
        agent.momentary_context_sd_enjoyment = self.momentary_cfg.context_sd_enjoyment
        agent.momentary_context_sd_utility = self.momentary_cfg.context_sd_utility
        agent.momentary_report_sd_urge = self.momentary_cfg.report_sd_urge
        agent.momentary_report_sd_enjoyment = self.momentary_cfg.report_sd_enjoyment
        agent.momentary_report_sd_utility = self.momentary_cfg.report_sd_utility

    def _context_behavior_mean(self, behavior: Behavior, setup: Setup) -> float:
        mods = behavior.setup_modifiers.get(setup, {})
        return clip11(behavior.base_outcome + mods.get("base_outcome_mod", 0.0))

    def _context_behavior_enjoyment_mean(self, behavior: Behavior, setup: Setup) -> float:
        mods = behavior.setup_modifiers.get(setup, {})
        return clip11(mods.get("enjoyment_outcome", self._context_behavior_mean(behavior, setup)))

    def _resolve_missing_sd(self) -> float:
        if self.missing_rate <= 0:
            return 0.0
        if self.missing_sd is None or self.missing_sd < 0:
            return max(0.01, self.missing_rate * 0.5)
        return float(self.missing_sd)

    def _sample_missing_rate(self) -> float:
        if self.missing_rate <= 0:
            return 0.0
        sd = self._resolve_missing_sd()
        rate = self.rng.normal(self.missing_rate, sd)
        return float(np.clip(rate, 0.0, 0.95))

    def _create_partner_pool(self) -> List[Dict]:
        n_partners = self.rng.integers(
            self.social_cfg.n_partners_min,
            self.social_cfg.n_partners_max,
            endpoint=True,
        )
        partners = []
        for pid in range(n_partners):
            w_I, w_E, w_U = self._sample_weights()
            tau = self._sample_tau()
            noise_s = self._sample_noise_s()
            social_kappa = self._sample_social_kappa()

            alpha_I_pos = self._sample_alpha(0.05, 0.25)
            alpha_I_neg = self._sample_alpha(0.05, 0.25)
            alpha_E = self._sample_alpha(0.05, 0.30)
            alpha_U = self._sample_alpha(0.05, 0.30)

            receptivity = self.rng.uniform(*self.social_cfg.receptivity_range)
            communion = self.rng.uniform(*self.social_cfg.communion_range)
            power = self.rng.uniform(*self.social_cfg.power_range)
            distance = self.rng.uniform(*self.social_cfg.distance_range)

            partner = Individual(name=f"partner_{pid}")
            self._configure_decision_agent(partner, w_I, tau, noise_s, social_kappa)

            for setup in self.setups:
                for behavior in self.behaviors:
                    mean_utility = self._context_behavior_mean(behavior, setup)
                    mean_enjoyment = self._context_behavior_enjoyment_mean(behavior, setup)
                    instinct = self.rng.uniform(-0.25, 0.25)
                    enjoyment = clip11(self.rng.normal(mean_enjoyment, 0.15))
                    utility = clip11(self.rng.normal(mean_utility, 0.15))
                    partner.add_behavior(
                        behavior.id,
                        setup.id,
                        instinct=instinct,
                        utility=utility,
                        enjoyment=enjoyment,
                        alpha_instinct_plus=alpha_I_pos,
                        alpha_instinct_minus=alpha_I_neg,
                        alpha_utility=alpha_U,
                        alpha_enjoyment=alpha_E,
                        w_enjoyment=w_E,
                        w_utility=w_U,
                        bias_scaling_factor=1.0,
                        exposure_count=0,
                    )

            partners.append({
                "partner_id": pid,
                "agent": partner,
                "receptivity": receptivity,
                "communion": communion,
                "power": power,
                "distance": distance,
            })
        return partners

    def _sample_person(self, pid: int) -> Tuple[Individual, Dict[str, float]]:
        w_I, w_E, w_U = self._sample_weights()
        tau = self._sample_tau()
        noise_s = self._sample_noise_s()
        social_kappa = self._sample_social_kappa()

        alpha_I_pos = self._sample_alpha(0.05, 0.25)
        alpha_I_neg = self._sample_alpha(0.05, 0.25)
        alpha_E = self._sample_alpha(0.05, 0.30)
        alpha_U = self._sample_alpha(0.05, 0.30)

        person = Individual(name=f"person_{pid}")
        self._configure_decision_agent(person, w_I, tau, noise_s, social_kappa)

        for setup in self.setups:
            for behavior in self.behaviors:
                instinct = self.rng.uniform(-0.25, 0.25)
                mean_utility = self._context_behavior_mean(behavior, setup)
                mean_enjoyment = self._context_behavior_enjoyment_mean(behavior, setup)
                enjoyment = clip11(self.rng.normal(mean_enjoyment, 0.15))
                utility = clip11(self.rng.normal(mean_utility, 0.15))
                person.add_behavior(
                    behavior.id,
                    setup.id,
                    instinct=instinct,
                    utility=utility,
                    enjoyment=enjoyment,
                    alpha_instinct_plus=alpha_I_pos,
                    alpha_instinct_minus=alpha_I_neg,
                    alpha_utility=alpha_U,
                    alpha_enjoyment=alpha_E,
                    w_enjoyment=w_E,
                    w_utility=w_U,
                    bias_scaling_factor=1.0,
                    exposure_count=0,
                )

        params = dict(
            w_I=w_I,
            w_E=w_E,
            w_U=w_U,
            tau=tau,
            noise_s=noise_s,
            social_kappa=social_kappa,
            alpha_I_pos=alpha_I_pos,
            alpha_I_neg=alpha_I_neg,
            alpha_E=alpha_E,
            alpha_U=alpha_U,
        )
        return person, params

    def _link_relationships(self, person: Individual):
        for partner in self.social_partners:
            Agent.add_relationship(
                person.id,
                partner["agent"].id,
                partner["distance"],
                partner["receptivity"],
                partner["power"],
                partner["communion"],
            )

    def _draw_event_type(self) -> str:
        rand = self.rng.random()
        cumulative = 0.0
        if rand < (cumulative := cumulative + self.social_cfg.p_observe):
            return "observe"
        if rand < (cumulative := cumulative + self.social_cfg.p_suggest):
            return "suggest"
        if rand < (cumulative := cumulative + self.social_cfg.p_feedback):
            return "observe_feedback"
        return "solitary"

    def _choose_partner(self) -> Dict | None:
        if not self.social_partners:
            return None
        idx = self.rng.integers(0, len(self.social_partners))
        return self.social_partners[idx]

    def _choose_setup(self) -> Setup:
        probs = np.array([cfg.probability for cfg in self.context_cfgs], dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(len(self.setups), dtype=float) / len(self.setups)
        else:
            probs = probs / probs.sum()
        idx = self.rng.choice(len(self.setups), p=probs)
        return self.setups[int(idx)]

    def _event_from_situation(
        self,
        situation: Situation,
        situation_type: str,
        setup: Setup,
        partner_id: int | None,
    ) -> Dict:
        if situation_type == "observe":
            choice_behavior = None
            observed_behavior = situation.selected_behavior.name if situation.selected_behavior else None
            choice_prob = None
        else:
            choice_behavior = situation.selected_behavior.name if situation.selected_behavior else None
            observed_behavior = None
            choice_prob = situation.choice_prob

        suggestion_terms = situation.suggestion_terms or {}
        suggest_cols = {
            f"suggest_term_{behavior.name}": float(suggestion_terms.get(behavior, 0.0))
            for behavior in self.behaviors
        }
        focal_momentary = situation.focal_momentary_appraisals or {}
        focal_reported = situation.focal_reported_appraisals or {}
        appraisal_cols = {}
        for behavior in self.behaviors:
            momentary = focal_momentary.get(behavior, {})
            reported = focal_reported.get(behavior, {})
            for domain in ["urge", "enjoyment", "utility"]:
                appraisal_cols[f"momentary_{domain}_{behavior.name}"] = momentary.get(domain, np.nan)
                appraisal_cols[f"report_{domain}_{behavior.name}"] = reported.get(domain, np.nan)

        return {
            "situation_type": situation_type,
            "setup_key": setup.name,
            "choice_behavior": choice_behavior,
            "choice_prob": choice_prob,
            "observed_behavior": observed_behavior,
            "learning_behavior": situation.learning_behavior.name if situation.learning_behavior else None,
            "learning_role": situation.learning_role,
            "partner_id": partner_id,
            "enjoyment_out": situation.focal_perceived_enjoyment,
            "utility_out": situation.focal_perceived_utility,
            "raw_enjoyment_out": situation.raw_enjoyment_out,
            "raw_utility_out": situation.raw_utility_out,
            **suggest_cols,
            **appraisal_cols,
        }

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows: List[Dict] = []
        people_tbl: List[Dict] = []

        for pid in range(self.N):
            person, params = self._sample_person(pid)
            self._link_relationships(person)
            if self.t_count is not None:
                T = int(self.t_count)
            else:
                T = self.rng.integers(self.T_range[0], self.T_range[1], endpoint=True)

            base_row = {
                "person_id": pid,
                **params,
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
            people_tbl.append(base_row)

            for t in range(T):
                situation_type = self._draw_event_type()
                setup = self._choose_setup()
                partner = self._choose_partner()

                if partner is None:
                    situation_type = "solitary"

                if situation_type == "observe":
                    interaction_mode = "observe_s"
                elif situation_type == "suggest":
                    interaction_mode = "suggest"
                elif situation_type == "observe_feedback":
                    interaction_mode = "observe_feedback"
                else:
                    interaction_mode = "solitary"

                environment = partner["agent"] if partner is not None else person
                partner_id = partner["partner_id"] if partner is not None else None

                situation = Situation(
                    setup.id,
                    person.id,
                    environment.id,
                    interaction_mode,
                    behaviors=self.behaviors,
                )
                situation._simulate_situation()

                event_data = self._event_from_situation(
                    situation,
                    situation_type,
                    setup,
                    partner_id,
                )

                row = {
                    "person_id": pid,
                    "t": t,
                    **event_data,
                }
                for behavior in self.behaviors:
                    b_params = person.behaviors[behavior][setup]
                    row[f"instinct_{behavior.name}"] = b_params["instinct"]
                    row[f"enjoyment_{behavior.name}"] = b_params["enjoyment"]
                    row[f"utility_{behavior.name}"] = b_params["utility"]
                rows.append(row)

        df = pd.DataFrame(rows)
        people = pd.DataFrame(people_tbl)
        return df, people

    def apply_missingness(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.missing_rate <= 0:
            return df.copy(), pd.DataFrame()

        keep_indices = []
        summary_rows = []
        for person_id, grp in df.groupby("person_id"):
            n_total = len(grp)
            if n_total == 0:
                continue
            rate = self._sample_missing_rate()
            n_drop = int(np.round(rate * n_total))
            n_drop = min(n_drop, n_total)
            if n_drop > 0:
                drop_idx = self.rng.choice(grp.index, size=n_drop, replace=False)
                keep_idx = grp.index.difference(drop_idx)
            else:
                keep_idx = grp.index
            keep_indices.extend(keep_idx)
            summary_rows.append({
                "person_id": int(person_id),
                "n_total": int(n_total),
                "n_drop": int(n_drop),
                "n_keep": int(len(keep_idx)),
                "missing_rate_target": float(self.missing_rate),
                "missing_rate_person": float(rate),
                "missing_sd": float(self._resolve_missing_sd()),
            })

        observed = df.loc[sorted(keep_indices)].copy()
        observed = observed.sort_values(["person_id", "t"]).reset_index(drop=True)
        summary = pd.DataFrame(summary_rows)
        return observed, summary


def default_behaviors() -> List[BehaviorCfg]:
    return [
        BehaviorCfg(
            key="avoid_conflict",
            label="Avoid conflict",
            difficulty=0.25,
            base_outcome=0.25,
            outcome_volatility=0.25,
        ),
        BehaviorCfg(
            key="approach_conflict_care",
            label="Approach conflict w/ care",
            difficulty=0.65,
            base_outcome=0.35,
            outcome_volatility=0.35,
        ),
    ]


def _context_params(
    base_outcome: float,
    difficulty: float,
    volatility: float,
    enjoyment_outcome: float | None = None,
    enjoyment_volatility: float | None = None,
) -> ContextBehaviorCfg:
    return ContextBehaviorCfg(
        utility_outcome=base_outcome,
        difficulty=difficulty,
        outcome_volatility=volatility,
        enjoyment_outcome=enjoyment_outcome,
        enjoyment_volatility=enjoyment_volatility,
    )


def default_contexts(behaviors: List[BehaviorCfg] | None = None, mode: str = "varied") -> List[ContextCfg]:
    if mode == "single":
        return [
            ContextCfg(
                key="ema_default",
                label="Default EMA context",
                probability=1.0,
                behavior_params={
                    "avoid_conflict": _context_params(0.25, 0.25, 0.25),
                    "approach_conflict_care": _context_params(0.35, 0.65, 0.35),
                },
            )
        ]

    if mode == "orthogonal":
        return [
            ContextCfg(
                key="approach_enjoyable_costly",
                label="Approach feels good but costs later",
                probability=0.25,
                behavior_params={
                    "avoid_conflict": _context_params(0.70, 0.15, 0.20, enjoyment_outcome=-0.55, enjoyment_volatility=0.12),
                    "approach_conflict_care": _context_params(-0.55, 0.70, 0.25, enjoyment_outcome=0.75, enjoyment_volatility=0.12),
                },
            ),
            ContextCfg(
                key="approach_useful_unpleasant",
                label="Approach helps but feels bad",
                probability=0.25,
                behavior_params={
                    "avoid_conflict": _context_params(-0.55, 0.25, 0.25, enjoyment_outcome=0.70, enjoyment_volatility=0.12),
                    "approach_conflict_care": _context_params(0.75, 0.65, 0.20, enjoyment_outcome=-0.55, enjoyment_volatility=0.12),
                },
            ),
            ContextCfg(
                key="approach_aligned_good",
                label="Approach feels good and helps",
                probability=0.25,
                behavior_params={
                    "avoid_conflict": _context_params(-0.50, 0.20, 0.20, enjoyment_outcome=-0.45, enjoyment_volatility=0.12),
                    "approach_conflict_care": _context_params(0.75, 0.55, 0.20, enjoyment_outcome=0.75, enjoyment_volatility=0.12),
                },
            ),
            ContextCfg(
                key="avoid_aligned_good",
                label="Avoiding feels good and helps",
                probability=0.25,
                behavior_params={
                    "avoid_conflict": _context_params(0.75, 0.15, 0.20, enjoyment_outcome=0.75, enjoyment_volatility=0.12),
                    "approach_conflict_care": _context_params(-0.50, 0.80, 0.20, enjoyment_outcome=-0.45, enjoyment_volatility=0.12),
                },
            ),
        ]

    return [
        ContextCfg(
            key="approach_favorable",
            label="Repair opportunity",
            probability=0.34,
            behavior_params={
                "avoid_conflict": _context_params(0.00, 0.25, 0.25),
                "approach_conflict_care": _context_params(0.70, 0.55, 0.30),
            },
        ),
        ContextCfg(
            key="cooldown_favorable",
            label="Escalation risk",
            probability=0.33,
            behavior_params={
                "avoid_conflict": _context_params(0.60, 0.15, 0.25),
                "approach_conflict_care": _context_params(-0.10, 0.85, 0.40),
            },
        ),
        ContextCfg(
            key="ambiguous_mixed",
            label="Ambiguous conflict",
            probability=0.33,
            behavior_params={
                "avoid_conflict": _context_params(0.25, 0.25, 0.35),
                "approach_conflict_care": _context_params(0.35, 0.65, 0.45),
            },
        ),
    ]


def main():
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="EMA-style cohort simulation for two behaviors.")
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--tmin", type=int, default=5)
    parser.add_argument("--tmax", type=int, default=20)
    parser.add_argument("--tcount", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision_model", type=str, default="softmax", choices=["softmax", "ddm"])
    parser.add_argument("--weight_dist", type=str, default="uniform", choices=["uniform", "skewed", "bimodal"])
    parser.add_argument("--weight_mode", type=str, default="relative", choices=["raw", "relative"])
    parser.add_argument("--alpha_dist", type=str, default="uniform", choices=["uniform", "skewed", "bimodal"])
    parser.add_argument("--tau_dist", type=str, default="lognormal", choices=["fixed", "lognormal", "bimodal"])
    parser.add_argument("--context_mode", type=str, default="varied", choices=["single", "varied", "orthogonal"])
    parser.add_argument("--fixed_noise_s", type=float, default=None)
    parser.add_argument("--no_momentary_reports", action="store_true")
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--missing_sd", type=float, default=-1.0)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--plot", action="store_true", help="Save a quick outcome plot to the output directory")
    args = parser.parse_args()

    dist_cfg = ParamDistCfg(
        weight_dist=args.weight_dist,
        alpha_dist=args.alpha_dist,
        tau_dist=args.tau_dist,
        fixed_noise_s=args.fixed_noise_s,
        weight_mode=args.weight_mode,
    )
    behaviors = default_behaviors()
    momentary_cfg = MomentaryReportCfg(enabled=not args.no_momentary_reports)

    sim = EMAPairSimulator(
        behaviors=behaviors,
        N=args.N,
        T_range=(args.tmin, args.tmax),
        t_count=args.tcount,
        social_cfg=SocialCfg(),
        dist_cfg=dist_cfg,
        momentary_cfg=momentary_cfg,
        contexts=default_contexts(behaviors, mode=args.context_mode),
        decision_model=args.decision_model,
        missing_rate=args.missing_rate,
        missing_sd=args.missing_sd,
        seed=args.seed,
    )
    df, ppl = sim.run()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.missing_rate > 0:
        df_full = df
        df_obs, missing_summary = sim.apply_missingness(df_full)
        df_full.to_csv(outdir / "ema_events_full.csv", index=False)
        df_obs.to_csv(outdir / "ema_events.csv", index=False)
        missing_summary.to_csv(outdir / "missingness_summary.csv", index=False)
    else:
        df.to_csv(outdir / "ema_events.csv", index=False)
    ppl.to_csv(outdir / "ema_people.csv", index=False)

    print("Saved:")
    print(f" - {outdir / 'ema_people.csv'}  (N={len(ppl)})")
    if args.missing_rate > 0:
        print(f" - {outdir / 'ema_events_full.csv'}  (rows={len(df)})")
        print(f" - {outdir / 'ema_events.csv'}  (rows={len(df_obs)})")
    else:
        print(f" - {outdir / 'ema_events.csv'}  (rows={len(df)})")

    if not args.plot:
        return

    try:
        import importlib
        if importlib.util.find_spec("matplotlib") is None:
            raise ImportError("matplotlib is not available")
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns  # type: ignore
        except Exception:
            sns = None  # type: ignore

        plt.figure(figsize=(9, 4))
        if sns is not None:
            sns.lineplot(
                data=df.melt(
                    id_vars=["person_id", "t"],
                    value_vars=["enjoyment_out", "utility_out"],
                    var_name="signal",
                    value_name="value",
                ),
                x="t",
                y="value",
                hue="signal",
                estimator="mean",
                errorbar=("pi", 50),
            )
        else:
            melted = df.melt(
                id_vars=["person_id", "t"],
                value_vars=["enjoyment_out", "utility_out"],
                var_name="signal",
                value_name="value",
            )
            mean_df = melted.groupby(["t", "signal"], as_index=False)["value"].mean()
            for sig, grp in mean_df.groupby("signal"):
                plt.plot(grp["t"], grp["value"], label=sig)
            plt.legend()

        plt.ylim(-1, 1)
        plt.title("Average outcomes across people (EMA events)")
        plt.tight_layout()
        plot_path = outdir / "ema_outcomes.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f" - {plot_path}")
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
