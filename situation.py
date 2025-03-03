import numpy as np
import pandas as pd
from agent import Agent, Individual, Group, Culture
from behavior import Behavior
from setup import Setup, SetupManager

class Situation:
    def __init__(self, setup, individual, environment, interaction_mode="observational",
                 behaviors=None):
        """
        Initialize a Situation instance.

        Parameters:
        - setup (str): The type of setup the situation represents.
        - individual (Individual): The agent involved in the situation.
        - environment (Environment): The environment involved in the situation.
        - interaction_mode (str): The type of interaction. Valid inputs include:
          "co-participation", "observational", "solitary", "benefit_exchange",
          "skill_exchange". The default is "observational".
        - behaviors (list of Behavior): A set of behaviors to consider in the situation.
          If no input, behaviors from both the individual and environment matching the setup will be considered.
        """
        # Define allowed interaction modes
        ALLOWED_INTERACTION_MODES = {
            "co-participation", "solitary", "observational", "benefit_exchange", "skill_exchange"
        }

        # Validate `setup`
        if not isinstance(setup, str) or not setup.strip():
            raise ValueError(f"Invalid setup: {setup}. Setup must be a non-empty string.")

        # Validate `individual`
        if not isinstance(agent, Agent):
            raise TypeError(f"Invalid individual: {individual}. Must be an instance of the Individual class.")

        # Validate `environment`
        if not isinstance(environment, Agent):
            raise TypeError(f"Invalid environment: {environment}. Must be an instance of the Environment class.")

        # Validate that the agent has a relationship with the environment
        if environment not in agent.relationships:
            raise ValueError(f"Individual has no relationship with environment {environment}.")

        # Validate `interaction_mode`
        if not isinstance(interaction_mode, str) or interaction_mode not in ALLOWED_INTERACTION_MODES:
            raise ValueError(f"Invalid interaction_mode: {interaction_mode}. "
                             f"Allowed modes are {ALLOWED_INTERACTION_MODES}")

        # Validate `behaviors`
        if behaviors is not None:
            if not isinstance(behaviors, list):
                raise TypeError(f"Behaviors must be a list or None, got {type(behaviors)}.")
            for behavior in behaviors:
                if not isinstance(behavior, Behavior):
                    raise TypeError(f"Each behavior must be an instance of the Behavior class, got {type(behavior)}.")

        # Store input parameters
        self.setup = setup
        self.individual = individual
        self.environment = environment
        self.interaction_mode = interaction_mode

        # Initialize behaviors: combine individual and environment behaviors matching the setup if none provided
        if behaviors is None:
            # Extract individual behaviors matching the setup
            individual_behaviors = [
                b for b, setups in individual.behaviors.items() if setup in setups
            ]

            # Extract environment behaviors matching the setup
            environment_behaviors = [
                b for b, setups in environment.behaviors.items() if setup in setups
            ]

            # Combine unique behaviors from both individual and environment
            self.behaviors = list(set(individual_behaviors + environment_behaviors))
        else:
            self.behaviors = behaviors

    def __repr__(self):
        return (f"Situation(setup={self.setup}, interaction_mode={self.interaction_mode}, "
                f"num_behaviors={len(self.behaviors)}, visibility={self.visibility})")

    # Validation methods
    def impose_range(self, benefit, min_benefit, max_benefit):
        """Ensure that a variable stays within its defined range."""
        return min(max(min_benefit, benefit), max_benefit)

    def impose_binary(self, benefit, benefitA, benefitB):
        """Ensure that a variable takes on a valid benefit."""
        return min(max(min_benefit, benefit), max_benefit)

    def _simulate_situation(self, seed=None):
        """
        Simulate the situation based on the mode of interaction.

        Parameters:
        - seed (optional int): Seed for randomization to ensure reproducibility.
        """
        # Validate `seed`
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer or None, got {type(seed)}.")

        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        if self.interaction_mode == "co-participation":
            self._simulate_co_participation()
        elif self.interaction_mode == "solitary":
            self._simulate_solitary()
        elif self.interaction_mode == "observational":
            self._simulate_observational()
        elif self.interaction_mode == "benefit_exchange":
            self._simulate_benefit_exchange()
        elif self.interaction_mode == "skill_exchange":
            self._simulate_skill_exchange()
        else:
            raise ValueError(f"Undefined interaction_mode: {self.interaction_mode}")

    def _simulate_co_participation(self):
        """
        Simulate a co-participation situation between two generic agents.
        Handles parameter storage locally and implements skill performance calculation.
        """
        # ===== Phase 1: Initialize Agents =====
        agent_a = self.individual
        agent_b = self.environment

        # ===== Phase 2: Get Relationship Parameters =====
        a_to_b = agent_a.relationships.get(agent_b.id, {})

        power_b_over_a = a_to_b.get('receptivity', 0)  # How much B influences A
        power_a_over_b = a_to_b.get('power', 0)  # How much A influences B

        # ===== Phase 3: Intention Blending & Select Behavior =====
        base_a = agent_a.form_intention(self.setup, self.behaviors)
        base_b = agent_b.form_intention(self.setup, self.behaviors)

        combined_intentions = {}
        for behavior in set(base_a)|set(base_b):
            intention_a = base_a.get(behavior, 0)
            intention_b = base_b.get(behavior, 0)

            # Weighted combination of agents A and B's intentions
            combined_intention = power_a_over_b * intention_a + power_b_over_a * intention_b

            combined_intentions[behavior] = max(0, min(1, combined_intention))  # Clamp to [0, 1]

        self.selected_behavior = self._select_behavior(combined_intentions)

        # ===== Phase 4: Effort Calculation =====
        # Get behavior parameters
        params_a = agent_a.behaviors[self.selected_behavior][self.setup]
        params_b = agent_b.behaviors[self.selected_behavior][self.setup]

        # Calculate effective efforts
        effort_a = self._calculate_effort(
            intention=intentions_a.get(self.selected_behavior, 0),
            skill=params_a['skill'],
            difficulty=self.selected_behavior.difficulty
        )

        effort_b = self._calculate_effort(
            intention=intentions_b.get(self.selected_behavior, 0),
            skill=params_b['skill'],
            difficulty=self.selected_behavior.difficulty
        )

        # ===== Phase 5: Outcome Determination =====
        # Simulate outcome with noise and get raw performance metrics
        final_outcome, raw_output = self._simulate_outcome(
            efforts=(effort_a, effort_b),
            skills=(params_a['skill'], params_b['skill']),
            behavior=self.selected_behavior
        )

        # ===== Phase 6: Skill Performance Calculation =====
        # Calculate objective skill demonstration
        total_required = self.selected_behavior.difficulty
        contribution_a = effort_a * params_a['skill']
        contribution_b = effort_b * params_b['skill']
        total_contribution = contribution_a + contribution_b

        success_ratio = min(total_contribution / total_required, 1.0)

        self.skill_performance_a = np.clip(contribution_a * success_ratio, 0, 1)
        self.skill_performance_b = np.clip(contribution_b * success_ratio, 0, 1)

        # ===== Phase 7: Evaluation and Learning =====
        # Calculate evaluations with local variables
        eval_a = agent_a.form_evaluation(
            intention=intentions_a.get(self.selected_behavior, 0),
            outcome=final_outcome,
            effort=effort_a,
            pleasure=self._simulate_pleasure(self.selected_behavior, agent_a)
        )

        eval_b = agent_b.form_evaluation(
            intention=intentions_b.get(self.selected_behavior, 0),
            outcome=final_outcome,
            effort=effort_b,
            pleasure=self._simulate_pleasure(self.selected_behavior, agent_b)
        )

        # Apply mutual influence to evaluations
        blended_eval_a = eval_a + receptivity_a * power_b * eval_b
        blended_eval_b = eval_b + receptivity_b * power_a * eval_a

        # Update both agents
        agent_a.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            evaluation=blended_eval_a,
            skill_level=self.skill_performance_a
        )

        agent_b.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            evaluation=blended_eval_b,
            skill_level=self.skill_performance_b
        )

    def _execute_solitary(self):
        pass

    def _simulate_observational(
        self,
        observer_suggestion=True,
        observer_feedback=True,
        observer_penalty=0.5,
        role_switching=False
    ):
        # Determine roles
        active_agent, observer = (self.environment, self.individual) if role_switching \
                              else (self.individual, self.environment)

        # Get relationship parameters from ACTIVE AGENT'S perspective
        relationship = active_agent.relationships.get(observer.id, {})
        receptivity = relationship.get("receptivity", 0)  # How open active is to suggestions
        power = relationship.get("power", 0)  # Observer's influence power

        # === Phase 1: Decision-Making ===
        active_intentions = active_agent.form_intention(self.setup, self.behaviors)

        if observer_suggestion:
            observer_intentions = observer.form_intention(self.setup, self.behaviors)
            # Blend using only receptivity for suggestion uptake
            combined_intentions = {
                b: (1 - receptivity) * active_intentions.get(b, 0) +
                  receptivity * observer_intentions.get(b, 0)
                for b in set(active_intentions) | set(observer_intentions)
            }
        else:
            combined_intentions = active_intentions

        self.selected_behavior = self._select_behavior(combined_intentions)

        # === Phase 2: Action Execution ===
        active_params = active_agent.behaviors[self.selected_behavior][self.setup]
        active_effort = self._calculate_effort(
            intention=combined_intentions[self.selected_behavior],
            skill=active_params["skill"],
            difficulty=self.selected_behavior.difficulty
        )

        self.outcome = self._simulate_outcome(
            efforts=(active_effort,),
            skills=(active_params["skill"],),
            behavior=self.selected_behavior
        )

        # === Phase 3: Feedback and Learning ===
        # Active agent learning (full self-evaluation)
        active_evaluation = active_agent.form_evaluation(
            intention=combined_intentions[self.selected_behavior],
            outcome=self.outcome,
            effort=active_effort,
            pleasure=self._simulate_pleasure(self.selected_behavior, active_agent)
        )
        active_adjusted_evaluation = active_evaluation + \
         self.individual.environments[self.environment]['receptivity']*self.environment_evaluation
        active_agent.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            evaluation=active_evaluation,
            performed=True
        )

        # Observer learning (if providing feedback)
        if observer_feedback:
            # Calculate observer's hypothetical evaluation
            obs_intention = observer.form_intention(self.setup, [self.selected_behavior]).get(self.selected_behavior, 0)
            obs_effort = self._calculate_effort(
                intention=obs_intention,
                skill=observer.behaviors[self.selected_behavior][self.setup]["skill"],
                difficulty=self.selected_behavior.difficulty
            )
            obs_evaluation = observer.form_evaluation(
                intention=obs_intention,
                outcome=self.outcome,
                effort=obs_effort,
                pleasure=self._simulate_pleasure(self.selected_behavior, observer)
            )
            obs_adjusted_evaluation = obs_evaluation + (1 - power) * observer_penalty
            # Apply power-modulated learning
            observer.update_behavior(
                behavior=self.selected_behavior,
                setup=self.setup,
                evaluation=obs_evaluation * (1 - power) * observer_penalty,
                performed=False,
                observer_penalty=observer_penalty
            )

    def _execute_benefit_exchange(self):
        pass

    def _execute_skill_exchange(self):
        pass

    def _select_behavior(self, intentions, base_threshold=3.0, noise_factor=0.5, max_steps=1000):
        """
        Select a behavior using a multi-alternative drift diffusion model.

        Parameters:
        - intentions (dict of float): A dictionary mapping behaviors to their intention benefits (0 to 1).
        - base_threshold (float): Baseline threshold for decision-making, scaled by average combined intention.
        - noise_factor (float): Factor controlling the magnitude of noise relative to the drift rates.
        - max_steps (int): Maximum number of steps allowed to prevent infinite looping.

        Returns:
        - The selected behavior.
        """
        # Initialize accumulators for each behavior
        accumulators = {behavior: 0 for behavior in intentions}
        drift_rates = intentions.copy()  # Start with the combined intentions as drift rates

        # Calculate the dynamic threshold based on maximum intention
        max_intention = max(drift_rates.benefits())
        threshold = base_threshold * max_intention

        # Calculate dynamic noise scaling based on total intention
        total_intention = sum(drift_rates.benefits())
        noise_multiplier = noise_factor / (total_intention + 1e-6)  # Avoid division by zero

        # Simulate the drift diffusion process
        for step in range(max_steps):
            for behavior in accumulators.keys():
                # Add evidence (drift + noise)
                noise = np.random.normal(0, noise_multiplier)
                accumulators[behavior] += drift_rates[behavior] + noise

                # Check if the evidence for any behavior crosses the threshold
                if accumulators[behavior] >= threshold:
                    return behavior

        # If no behavior reaches the threshold within max_steps, return the behavior with the highest evidence
        return max(accumulators, key=accumulators.get)

    def _calculate_effort(self, intention, skill, difficulty, noise_std=0.1,
                         scaling_factor=5, stability_constant=0.1):
        """
        Calculate the amount of effort an agent puts into a behavior.

        Parameters:
        - intention (float): Agent's intention to perform the behavior (0 to 1).
        - skill (float): Agent's skill level for the behavior (0 to 1).
        - difficulty (float): Difficulty of the behavior (0 to 1).
        - noise_std (float): Standard deviation of random noise (default 0.05).
        - scaling_factor (float): Scaling factor for intention (default 5).
        - stability_constant (float): Stabilizes skill adjustment (default 0.1).

        Returns:
        - float: Effort benefit between 0 and 1.
        """
        # Validate inputs
        assert 0 <= skill <= 1, f"Skill must be in [0, 1], got {skill}."
        assert 0 <= difficulty <= 1, f"Difficulty must be in [0, 1], got {difficulty}."

        # Calculate base effort from intention
        base_effort = 1 / (1 + np.exp(-scaling_factor * (intention - 0.5)))

        # Adjust for skill and difficulty with stability constant
        skill_adjustment = (difficulty + stability_constant) / (skill + stability_constant)
        effort = base_effort * skill_adjustment

        # Add random noise
        noise = np.random.normal(0, noise_std)
        effort = effort + noise

        # Clip effort to be within 0 and 1
        return max(0, min(1, effort))

    def _simulate_pleasure(self, behavior, agent, noise_std=0.1):
        """
        Simulate the pleasure an agent experiences while performing a behavior.

        Parameters:
        - behavior (Behavior): the behavior object containing its parameters.
        - agent (Individual or Group): an agent performing the behavior.
        - noise_std (float): standard deviation of the noise. Default is 0.1

        Returns:
        - float: The estimated pleasure an agent experiences [-1, 1].
        """

        # Validate inputs
        if not isinstance(agent, Agent) or setup not in agent.behaviors.get(behavior, {}):
            raise ValueError("Agent has no data for this behavior in the current setup.")

        # Retrieve enjoyment from the agent's behavior parameters
        enjoyment = agent.behaviors[behavior][self.setup]["enjoyment"]
        noise = np.random.normal(0, noise_std)
        pleasure = enjoyment + noise
        return np.clip(pleasure, -1, 1)

    def _simulate_outcome(self, efforts, skills, behavior):
        """
        Simulate outcome using the situation's setup context and behavior modifiers.

        Parameters:
        - efforts: Tuple[float] - Effort values from contributing agents (0-1)
        - skills: Tuple[float] - Corresponding skill values (0-1)
        - behavior: Behavior - The behavior being performed

        Returns:
        - outcome: float in [-1, 1]
        - raw_output: float
        """
        # Validate inputs
        assert len(efforts) == len(skills), "Efforts and skills must have same length"
        assert len(efforts) >= 1, "At least one agent must contribute"

        # Get setup-adjusted parameters
        setup_mods = behavior.setup_modifiers.get(self.setup, {})
        base_outcome = behavior.base_outcome + setup_mods.get("base_outcome_mod", 0)
        difficulty = behavior.difficulty + setup_mods.get("difficulty_mod", 0)

        # Clip to valid ranges
        base_outcome = np.clip(base_outcome, -1, 1)
        difficulty = np.clip(difficulty, 0, 1)

        # Calculate combined output from all contributors
        output = sum(e * s for e, s in zip(efforts, skills))

        # Determine success threshold
        if output >= difficulty:
            raw_outcome = base_outcome
        else:
            failure_ratio = (difficulty - output) / difficulty
            raw_outcome = base_outcome * (1 - failure_ratio)

        # Add volatility and clamp
        outcome = np.clip(raw_outcome + np.random.normal(0, behavior.outcome_volatility), -1, 1)
        return outcome, raw_outcome