from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# Config dataclasses
# -----------------------------

@dataclass
class BehaviorCfg:
    key: str
    label: str
    difficulty: float          # 0..1 (used to adjust perceived enjoyment/utility)
    base_outcome: float        # mean enjoyment/utility baseline in [-1, 1]
    outcome_volatility: float  # st.dev of outcomes (0..1)

@dataclass
class SocialCfg:
    # Event type probabilities (should sum to ≤ 1.0, remainder is solitary)
    p_observe: float = 0.20              # Pr(observational learning event)
    p_suggest: float = 0.30              # Pr(suggestion before choice)
    p_feedback: float = 0.30             # Pr(feedback after action)
    # Number of social partners in the pool
    n_partners_min: int = 80
    n_partners_max: int = 120
    # Distributions for relationship variables
    receptivity_range: Tuple[float, float] = (-0.2, 0.9)  # R_{A<-O}
    communion_range: Tuple[float, float] = (-0.5, 0.8)    # C_{A,O}
    power_range: Tuple[float, float] = (0.0, 1.0)         # not directly used, kept for extension
    presence_salience_range: Tuple[float, float] = (0.2, 1.0)  # P_O
    # Observational learning parameter
    observer_penalty: float = 0.5        # Penalty for observational learning (from situation.py)

@dataclass
class PersonCfg:
    # Weights and noise
    w_I: float
    w_E: float
    w_U: float
    tau: float
    noise_s: float
    # Learning rates
    alpha_I_pos: float
    alpha_I_neg: float
    alpha_E: float
    alpha_U: float
    # Baseline states per behavior key
    instinct: Dict[str, float]     # [-1,1]
    enjoyment: Dict[str, float]    # [-1,1]
    utility: Dict[str, float]      # [-1,1]

@dataclass
class SocialPartner:
    """Social partner agent with fixed parameters (no learning, α=0)."""
    partner_id: int              # Unique ID for this partner
    receptivity_to_partner: float  # R_{A<-O} in [-1,1], how receptive main agent is
    communion: float             # C_{A,O} in [-1,1], affective quality
    presence_salience: float     # P_O in [0,1], how salient partner's presence is
    # Partner's value system (FIXED, no learning)
    w_I: float                   # Partner's instinct weight
    w_E: float                   # Partner's enjoyment weight
    w_U: float                   # Partner's utility weight
    tau: float                   # Partner's inverse temperature
    noise_s: float               # Partner's noise std dev
    # Partner's beliefs (FIXED, no learning - this is key difference)
    instinct: Dict[str, float]   # Partner's instinct per behavior
    enjoyment: Dict[str, float]  # Partner's enjoyment predictions per behavior
    utility: Dict[str, float]    # Partner's utility predictions per behavior

# -----------------------------
# Helpers
# -----------------------------

def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x * tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def clip11(x: float) -> float:
    return max(-1.0, min(1.0, x))

# -----------------------------
# Simulator
# -----------------------------

class EMAPairSimulator:
    def __init__(
        self,
        behaviors: List[BehaviorCfg],
        N: int = 50,
        T_range: Tuple[int, int] = (5, 20),
        social_cfg: SocialCfg | None = None,
        seed: int = 42,
    ):
        self.behaviors = behaviors
        self.N = N
        self.T_range = T_range
        self.social_cfg = social_cfg if social_cfg is not None else SocialCfg()
        self.rng = np.random.default_rng(seed)

        self.beh_keys = [b.key for b in behaviors]
        self.beh_index = {b.key: i for i, b in enumerate(behaviors)}
        
        # Create social partner pool (100 agents with fixed parameters)
        self.social_partners = self._create_partner_pool()

    # ----- Social Partner Pool -----
    
    def _create_partner_pool(self) -> List[SocialPartner]:
        """Create a pool of 80-120 social partners with fixed parameters (no learning)."""
        n_partners = self.rng.integers(
            self.social_cfg.n_partners_min, 
            self.social_cfg.n_partners_max, 
            endpoint=True
        )
        partners = []
        for pid in range(n_partners):
            # Sample partner's value system
            w_I = self.rng.uniform(0.2, 1.0)
            w_E = self.rng.uniform(0.2, 1.0)
            w_U = self.rng.uniform(0.2, 1.0)
            tau = 3.0
            noise_s = self.rng.uniform(0.05, 0.20)
            
            # Sample relationship parameters (from main agent's perspective)
            R = self.rng.uniform(*self.social_cfg.receptivity_range)
            C = self.rng.uniform(*self.social_cfg.communion_range)
            P = self.rng.uniform(*self.social_cfg.presence_salience_range)
            
            # Partner's beliefs (FIXED - no learning)
            instinct = {b.key: self.rng.uniform(-0.25, 0.25) for b in self.behaviors}
            enjoyment = {b.key: clip11(self.rng.normal(b.base_outcome, 0.15)) for b in self.behaviors}
            utility = {b.key: clip11(self.rng.normal(b.base_outcome, 0.15)) for b in self.behaviors}
            
            partners.append(SocialPartner(
                partner_id=pid,
                receptivity_to_partner=R,
                communion=C,
                presence_salience=P,
                w_I=w_I, w_E=w_E, w_U=w_U,
                tau=tau, noise_s=noise_s,
                instinct=instinct,
                enjoyment=enjoyment,
                utility=utility,
            ))
        return partners

    # ----- cohort generation -----

    def _sample_person(self, pid: int) -> PersonCfg:
        """Sample a main agent (learns via α>0)."""
        # Sample weights/temperatures
        w_I = self.rng.uniform(0.2, 1.0)
        w_E = self.rng.uniform(0.2, 1.0)
        w_U = self.rng.uniform(0.2, 1.0)
        tau = 3.0 # FIXED inverse temperature for recovery study
        noise_s = self.rng.uniform(0.05, 0.20)

        # Learning rates (α>0 for main agents)
        alpha_I_pos = self.rng.uniform(0.05, 0.25)
        alpha_I_neg = self.rng.uniform(0.05, 0.25)
        alpha_E = self.rng.uniform(0.05, 0.30)
        alpha_U = self.rng.uniform(0.05, 0.30)

        # Baseline states per behavior
        instinct = {b.key: self.rng.uniform(-0.25, 0.25) for b in self.behaviors}
        enjoyment = {b.key: self.rng.uniform(-0.25, 0.25) for b in self.behaviors}
        utility = {b.key: self.rng.uniform(-0.25, 0.25) for b in self.behaviors}

        return PersonCfg(
            w_I=w_I, w_E=w_E, w_U=w_U, tau=tau, noise_s=noise_s,
            alpha_I_pos=alpha_I_pos, alpha_I_neg=alpha_I_neg,
            alpha_E=alpha_E, alpha_U=alpha_U,
            instinct=instinct, enjoyment=enjoyment, utility=utility,
        )

    # ----- single EMA event (4 situation types) -----
    
    def _partner_choice_values(self, partner: SocialPartner) -> np.ndarray:
        """Compute choice values for a social partner (used in observation/suggestion)."""
        values = []
        for b in self.behaviors:
            H = partner.w_I * partner.instinct[b.key]
            E_eval = partner.w_E * partner.enjoyment[b.key] + partner.w_U * partner.utility[b.key]
            noise = np.random.normal(0.0, partner.noise_s)
            CV = H + E_eval + noise
            values.append(CV)
        return np.array(values, dtype=float)

    def _choice_values(self, p: PersonCfg, suggestions: Dict[str, float]) -> np.ndarray:
        """Compute choice values for main agent (with optional suggestions)."""
        values = []
        for b in self.behaviors:
            H = p.w_I * p.instinct[b.key]
            E_eval = p.w_E * p.enjoyment[b.key] + p.w_U * p.utility[b.key]
            sug_term = suggestions.get(b.key, 0.0)
            # Gaussian noise
            noise = np.random.normal(0.0, p.noise_s)
            CV = H + E_eval + sug_term + noise
            values.append(CV)
        return np.array(values, dtype=float)
    
    def _simulate_observe(self, p: PersonCfg, partner: SocialPartner) -> Dict:
        """
        Situation type: observe
        Main agent observes partner's choice → updates instinct via observational learning.
        Based on situation.py lines 300-420.
        """
        # Partner makes choice (using their fixed parameters)
        partner_cvs = self._partner_choice_values(partner)
        partner_probs = softmax(partner_cvs, partner.tau)
        partner_choice_idx = np.random.choice(len(self.behaviors), p=partner_probs)
        partner_choice = self.behaviors[partner_choice_idx]
        
        # Simulate outcome (partner experiences it)
        base = partner_choice.base_outcome
        vol = max(1e-6, partner_choice.outcome_volatility)
        e_out = clip11(np.random.normal(loc=base - 0.25 * partner_choice.difficulty, scale=vol))
        u_out = clip11(np.random.normal(loc=base - 0.15 * partner_choice.difficulty, scale=vol))
        
        # Main agent learns vicariously (observational learning with penalty)
        observer_penalty = self.social_cfg.observer_penalty
        
        # Update instinct: strengthen observed behavior, weaken alternatives (with penalty)
        for bkey in self.beh_keys:
            if bkey == partner_choice.key:
                delta_I = (p.alpha_I_pos * observer_penalty) * (1.0 - p.instinct[bkey])
            else:
                delta_I = (p.alpha_I_neg * observer_penalty) * (-1.0 - p.instinct[bkey])
            p.instinct[bkey] = clip11(p.instinct[bkey] + delta_I)
        
        # Update enjoyment/utility predictions (with penalty)
        p.enjoyment[partner_choice.key] = clip11(
            p.enjoyment[partner_choice.key] + 
            (p.alpha_E * observer_penalty) * (e_out - p.enjoyment[partner_choice.key])
        )
        p.utility[partner_choice.key] = clip11(
            p.utility[partner_choice.key] + 
            (p.alpha_U * observer_penalty) * (u_out - p.utility[partner_choice.key])
        )
        
        return {
            "situation_type": "observe",
            "choice_behavior": None,  # Main agent didn't choose
            "observed_behavior": partner_choice.key,
            "partner_id": partner.partner_id,
            "enjoyment_out": e_out,
            "utility_out": u_out,
        }
    
    def _simulate_solitary(self, p: PersonCfg) -> Dict:
        """
        Situation type: solitary
        Main agent chooses alone, no social influence.
        """
        # Choice without suggestions
        cvs = self._choice_values(p, {})
        probs = softmax(cvs, p.tau)
        choice_idx = np.random.choice(len(self.behaviors), p=probs)
        chosen = self.behaviors[choice_idx]
        
        # Simulate outcome
        base = chosen.base_outcome
        vol = max(1e-6, chosen.outcome_volatility)
        e_out = clip11(np.random.normal(loc=base - 0.25 * chosen.difficulty, scale=vol))
        u_out = clip11(np.random.normal(loc=base - 0.15 * chosen.difficulty, scale=vol))
        
        # Learning update
        self._update_learning(p, chosen.key, e_out, u_out)
        
        return {
            "situation_type": "solitary",
            "choice_behavior": chosen.key,
            "choice_prob": probs[choice_idx],
            "partner_id": None,
            "enjoyment_out": e_out,
            "utility_out": u_out,
            # Suggestion terms (all zero for solitary)
            **{f"suggest_term_{b.key}": 0.0 for b in self.behaviors},
        }
    
    def _simulate_suggest(self, p: PersonCfg, partner: SocialPartner) -> Dict:
        """
        Situation type: suggest
        Partner provides suggestion → main agent incorporates it → chooses.
        Based on situation.py lines 314-330.
        """
        # Partner's suggestion based on their value system
        partner_cvs = self._partner_choice_values(partner)
        suggestions = {}
        for i, b in enumerate(self.behaviors):
            # Suggestion weighted by receptivity
            suggestions[b.key] = partner_cvs[i] * partner.receptivity_to_partner
        
        # Main agent's choice with suggestion influence
        cvs = self._choice_values(p, suggestions)
        probs = softmax(cvs, p.tau)
        choice_idx = np.random.choice(len(self.behaviors), p=probs)
        chosen = self.behaviors[choice_idx]
        
        # Simulate outcome
        base = chosen.base_outcome
        vol = max(1e-6, chosen.outcome_volatility)
        e_out = clip11(np.random.normal(loc=base - 0.25 * chosen.difficulty, scale=vol))
        u_out = clip11(np.random.normal(loc=base - 0.15 * chosen.difficulty, scale=vol))
        
        # Learning update (no social modulation in suggest-only)
        self._update_learning(p, chosen.key, e_out, u_out)
        
        return {
            "situation_type": "suggest",
            "choice_behavior": chosen.key,
            "choice_prob": probs[choice_idx],
            "partner_id": partner.partner_id,
            "enjoyment_out": e_out,
            "utility_out": u_out,
            # Suggestion terms for all behaviors
            **{f"suggest_term_{b.key}": suggestions[b.key] for b in self.behaviors},
        }
    
    def _simulate_observe_feedback(self, p: PersonCfg, partner: SocialPartner) -> Dict:
        """
        Situation type: observe_feedback
        Main agent chooses → partner provides feedback based on their utility prediction.
        Based on situation.py lines 354-375.
        """
        # Main agent's choice (no suggestion)
        cvs = self._choice_values(p, {})
        probs = softmax(cvs, p.tau)
        choice_idx = np.random.choice(len(self.behaviors), p=probs)
        chosen = self.behaviors[choice_idx]
        
        # Simulate base outcome
        base = chosen.base_outcome
        vol = max(1e-6, chosen.outcome_volatility)
        
        # Base enjoyment with social presence effect: E_A = E_{A(alone)} + C_{A,O} * P_O
        base_enjoyment = np.random.normal(loc=base - 0.25 * chosen.difficulty, scale=vol)
        social_enjoyment = clip11(base_enjoyment + partner.communion * partner.presence_salience)
        
        # Base utility with feedback effect
        base_utility = np.random.normal(loc=base - 0.15 * chosen.difficulty, scale=vol)
        
        # Partner's utility prediction for this behavior
        partner_utility_pred = partner.w_E * partner.enjoyment[chosen.key] + partner.w_U * partner.utility[chosen.key]
        
        # Feedback: (U_A - utility_{O,B,S}) + M_O (mood ~ N(0, 0.1))
        mood = np.random.normal(0, 0.1)
        feedback = (base_utility - partner_utility_pred) + mood
        social_utility = clip11(base_utility + feedback * partner.receptivity_to_partner)
        
        # Learning update with social modulation
        self._update_learning(p, chosen.key, social_enjoyment, social_utility)
        
        return {
            "situation_type": "observe_feedback",
            "choice_behavior": chosen.key,
            "choice_prob": probs[choice_idx],
            "partner_id": partner.partner_id,
            "enjoyment_out": social_enjoyment,
            "utility_out": social_utility,
            # Suggestion terms (all zero for feedback-only)
            **{f"suggest_term_{b.key}": 0.0 for b in self.behaviors},
        }

    def _draw_outcome(self, b: BehaviorCfg, partner: SocialPartner | None) -> Tuple[float, float]:
        """Legacy method - kept for compatibility but deprecated."""
        # Base enjoyment/utility from behavior config
        base = b.base_outcome
        vol = max(1e-6, b.outcome_volatility)
        # Difficulty reduces enjoyment slightly and utility a bit less (tunable assumptions)
        e = np.random.normal(loc=base - 0.25 * b.difficulty, scale=vol)
        u_alone = np.random.normal(loc=base - 0.15 * b.difficulty, scale=vol)

        # Social modulation of enjoyment (communion*presence)
        if partner is not None:
            e = e + partner.communion * partner.presence_salience
        e = clip11(e)

        # Social feedback on utility after action:
        # feedback = (utility_A - utility_O) + M_O; we fold mood M_O ~ N(0, 0.1)
        if partner is not None:
            u_O_pred = clip11(partner.w_E * partner.enjoyment[b.key] + partner.w_U * partner.utility[b.key])
            mood = np.random.normal(0, 0.1)
            fb = (u_alone - u_O_pred) + mood
            u = clip11(u_alone + fb * partner.receptivity_to_partner)
        else:
            u = clip11(u_alone)
        return e, u

    def _update_learning(self, p: PersonCfg, chosen_key: str, e_out: float, u_out: float):
        # Instinct update: strengthen chosen, weaken unchosen
        for bkey in self.beh_keys:
            if bkey == chosen_key:
                delta_I = p.alpha_I_pos * (1.0 - p.instinct[bkey])
            else:
                delta_I = p.alpha_I_neg * (-1.0 - p.instinct[bkey])
            p.instinct[bkey] = clip11(p.instinct[bkey] + delta_I)
        # Enjoyment and Utility prediction error updates
        p.enjoyment[chosen_key] = clip11(p.enjoyment[chosen_key] + p.alpha_E * (e_out - p.enjoyment[chosen_key]))
        p.utility[chosen_key] = clip11(p.utility[chosen_key] + p.alpha_U * (u_out - p.utility[chosen_key]))

    # ----- run -----

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run simulation with 4 situation types:
        - observe: Main agent observes partner's choice (observational learning)
        - solitary: Main agent chooses alone
        - suggest: Partner provides suggestion, main agent chooses
        - observe_feedback: Main agent chooses, partner provides feedback
        """
        rows: List[Dict] = []
        people_tbl: List[Dict] = []

        for pid in range(self.N):
            p = self._sample_person(pid)
            T = self.rng.integers(self.T_range[0], self.T_range[1], endpoint=True)

            people_tbl.append({
                "person_id": pid,
                "w_I": p.w_I, "w_E": p.w_E, "w_U": p.w_U, "tau": p.tau, "noise_s": p.noise_s,
                "alpha_I_pos": p.alpha_I_pos, "alpha_I_neg": p.alpha_I_neg,
                "alpha_E": p.alpha_E, "alpha_U": p.alpha_U,
                **{f"instinct_{k}_0": v for k, v in p.instinct.items()},
                **{f"enjoyment_{k}_0": v for k, v in p.enjoyment.items()},
                **{f"utility_{k}_0": v for k, v in p.utility.items()},
            })

            for t in range(T):
                # Determine situation type probabilistically
                rand = self.rng.random()
                cumulative = 0.0
                
                # Choose a random partner from the pool (if social event)
                partner = None
                if len(self.social_partners) > 0:
                    partner = self.social_partners[self.rng.integers(0, len(self.social_partners))]
                
                # Decide situation type
                if rand < (cumulative := cumulative + self.social_cfg.p_observe):
                    # (a) Observational learning
                    if partner is None:
                        event_data = self._simulate_solitary(p)
                    else:
                        event_data = self._simulate_observe(p, partner)
                        
                elif rand < (cumulative := cumulative + self.social_cfg.p_suggest):
                    # (c) Suggestion
                    if partner is None:
                        event_data = self._simulate_solitary(p)
                    else:
                        event_data = self._simulate_suggest(p, partner)
                        
                elif rand < (cumulative := cumulative + self.social_cfg.p_feedback):
                    # (d) Feedback
                    if partner is None:
                        event_data = self._simulate_solitary(p)
                    else:
                        event_data = self._simulate_observe_feedback(p, partner)
                else:
                    # (b) Solitary
                    event_data = self._simulate_solitary(p)

                # Log event
                row = {
                    "person_id": pid,
                    "t": t,
                    **event_data,
                    # Current states after update
                    **{f"instinct_{k}": p.instinct[k] for k in self.beh_keys},
                    **{f"enjoyment_{k}": p.enjoyment[k] for k in self.beh_keys},
                    **{f"utility_{k}": p.utility[k] for k in self.beh_keys},
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        people = pd.DataFrame(people_tbl)
        return df, people


# -----------------------------
# Default config for your two behaviors
# -----------------------------

def default_behaviors() -> List[BehaviorCfg]:
    return [
        BehaviorCfg(
            key="avoid_conflict",
            label="Avoid conflict",
            difficulty=0.15,           # low difficulty
            base_outcome=0.15,         # low base outcome
            outcome_volatility=0.10,   # low volatility
        ),
        BehaviorCfg(
            key="approach_conflict_care",
            label="Approach conflict w/ care",
            difficulty=0.75,           # high difficulty
            base_outcome=0.65,         # high base outcome
            outcome_volatility=0.25,   # medium volatility
        ),
    ]


# -----------------------------
# CLI entry
# -----------------------------

def main():
    import argparse, pathlib

    parser = argparse.ArgumentParser(description="EMA-style cohort simulation for two behaviors.")
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--tmin", type=int, default=5)
    parser.add_argument("--tmax", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    sim = EMAPairSimulator(
        behaviors=default_behaviors(),
        N=args.N,
        T_range=(args.tmin, args.tmax),
        social_cfg=SocialCfg(),
        seed=args.seed,
    )
    df, ppl = sim.run()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "ema_events.csv", index=False)
    ppl.to_csv(outdir / "ema_people.csv", index=False)

    # Quick sanity prints
    print("Saved:")
    print(f" - {outdir / 'ema_people.csv'}  (N={len(ppl)})")
    print(f" - {outdir / 'ema_events.csv'}  (rows={len(df)})")

    # Optional quick plots (y-axis fixed to [-1, 1])
    try:
        # Import matplotlib (required for plotting); seaborn is optional.
        import importlib
        if importlib.util.find_spec("matplotlib") is None:
            raise ImportError("matplotlib is not available")
        import matplotlib.pyplot as plt

        # Try to import seaborn for nicer plots; fall back to matplotlib if unavailable.
        try:
            import seaborn as sns  # type: ignore
        except Exception:
            sns = None  # type: ignore

        plt.figure(figsize=(9,4))
        if sns is not None:
            sns.lineplot(
                data=df.melt(id_vars=["person_id", "t"], value_vars=["enjoyment_out", "utility_out"],
                            var_name="signal", value_name="value"),
                x="t", y="value", hue="signal", estimator="mean", errorbar=("pi", 50)
            )
        else:
            # Basic matplotlib fallback: plot mean value per t for each signal
            melted = df.melt(id_vars=["person_id", "t"], value_vars=["enjoyment_out", "utility_out"],
                             var_name="signal", value_name="value")
            mean_df = melted.groupby(["t", "signal"], as_index=False)["value"].mean()
            for sig, grp in mean_df.groupby("signal"):
                plt.plot(grp["t"], grp["value"], label=sig)
            plt.legend()

        plt.ylim(-1, 1)
        plt.title("Average outcomes across people (EMA events)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)

if __name__ == "__main__":
    main()