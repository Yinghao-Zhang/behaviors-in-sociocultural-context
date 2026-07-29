import numpy as np
import pandas as pd
from agent import Agent, Individual, Group, Culture
from behavior import Behavior
from setup import Setup, SetupManager
from social_influence import (
    integrate_feedback,
    integrate_suggestion,
    qualify_social_signal,
)

class Situation:
    def __init__(self, setup_id, individual_id, environment_id, interaction_mode, behaviors=None):
        """
        Initialize a Situation instance.

        Parameters:
        - setup_id (Setup): The type of setup the situation represents.
        - individual (Agent; Individual or Group): The agent involved in the situation.
        - environment (Agent; Individual, Group, or Culture): The environment involved in the situation.
        - interaction_mode (str): The type of interaction. Valid inputs include:
                    "solitary", "observe", "co-participate", "talk".
        - behaviors (list of Behavior): A set of behaviors to consider in the situation.
          If no input, behaviors from both the individual and environment matching the setup will be considered.
        """

        self._simulation_methods = {
            "solitary": self._simulate_solitary,
            "co-participate": self._simulate_co_participation,
            "observe": lambda: self._simulate_observational(observer_suggestion=False, observer_feedback=False),
            "observe_s": lambda: self._simulate_observational(observer_suggestion=False, observer_feedback=False, role_switching=True),
            "suggest": lambda: self._simulate_observational(observer_suggestion=True, observer_feedback=False),
            "suggest_s": lambda: self._simulate_observational(observer_suggestion=True, observer_feedback=False, role_switching=True),
            "observe_feedback": lambda: self._simulate_observational(observer_suggestion=False, observer_feedback=True),
            "observe_feedback_s": lambda: self._simulate_observational(observer_suggestion=False, observer_feedback=True, role_switching=True),
            "suggest_feedback": lambda: self._simulate_observational(observer_suggestion=True, observer_feedback=True),
            "suggest_feedback_s": lambda: self._simulate_observational(observer_suggestion=True, observer_feedback=True, role_switching=True),
            "just_talk": self._simulate_talking
        }

        # Store input parameters
        self.setup_id = setup_id
        self.setup = Setup.get(setup_id)  # Get actual object when needed

        self.individual_id = individual_id
        self.individual = Agent.get_agent(individual_id)

        self.environment_id = environment_id
        self.environment = Agent.get_agent(environment_id)

        self.interaction_mode = interaction_mode

        # Initialize behaviors: combine individual and environment behaviors matching the setup if none provided
        if behaviors is None:
            # Extract individual behaviors matching the setup
            individual_behaviors = [
                b for b, setups in self.individual.behaviors.items() if self.setup in setups
            ]

            # Extract environment behaviors matching the setup
            environment_behaviors = [
                b for b, setups in self.environment.behaviors.items() if self.setup in setups
            ]

            # Combine unique behaviors from both individual and environment
            self.behaviors = list(set(individual_behaviors + environment_behaviors))
        else:
            self.behaviors = behaviors

        self.visibility = None
        self.selected_behavior = None
        self.outcome = None
        self.choice_probs = None
        self.choice_prob = None
        self.suggestion_terms = None
        self.feedback_signal = None
        self.perceived_utility = None
        self.perceived_enjoyment = None
        self.learning_behavior = None
        self.learning_role = None
        self.focal_perceived_utility = None
        self.focal_perceived_enjoyment = None
        self.raw_utility_out = None
        self.raw_enjoyment_out = None
        self.focal_momentary_appraisals = None
        self.focal_reported_appraisals = None

    def __repr__(self):
        return (f"Situation(setup={self.setup}, interaction_mode={self.interaction_mode}, "
                f"num_behaviors={len(self.behaviors)}, visibility={self.visibility})")

    def _sample_momentary_appraisals(self, agent):
        """Sample proximal I/E/U appraisals and noisy self-reports for this setup."""
        use_momentary = getattr(agent, 'use_momentary_appraisals', False)
        emit_reports = getattr(agent, 'emit_momentary_reports', use_momentary)
        domain_map = {
            "urge": "instinct",
            "enjoyment": "enjoyment",
            "utility": "utility",
        }
        momentary = {}
        reported = {}
        for behavior in self.behaviors:
            params = agent.behaviors[behavior][self.setup]
            momentary[behavior] = {}
            reported[behavior] = {}
            for out_key, state_key in domain_map.items():
                stable_value = params[state_key]
                context_sd = getattr(
                    agent,
                    f"momentary_context_sd_{out_key}",
                    getattr(agent, "momentary_context_sd", 0.0),
                )
                report_sd = getattr(
                    agent,
                    f"momentary_report_sd_{out_key}",
                    getattr(agent, "momentary_report_sd", 0.0),
                )
                context_shift = np.random.normal(0, context_sd) if use_momentary else 0.0
                true_value = np.clip(stable_value + context_shift, -1, 1)
                report_noise = np.random.normal(0, report_sd) if emit_reports else 0.0
                report_value = np.clip(true_value + report_noise, -1, 1) if emit_reports else np.nan
                momentary[behavior][out_key] = float(true_value)
                reported[behavior][out_key] = float(report_value) if np.isfinite(report_value) else np.nan
        return momentary, reported

    def _store_focal_appraisals(self, momentary, reported):
        self.focal_momentary_appraisals = momentary
        self.focal_reported_appraisals = reported

    def _choice_values_from_appraisals(self, agent, appraisals):
        values = {}
        for behavior, components in appraisals.items():
            params = agent.behaviors[behavior][self.setup]
            values[behavior] = agent.choice_value_from_components(
                components["urge"],
                components["enjoyment"],
                components["utility"],
                params=params,
            )
        return values

    @staticmethod
    def _center_values(values):
        """Center behavior values so social signals express relative preference."""
        if not values:
            return {}
        center = float(np.mean(list(values.values())))
        return {
            behavior: float(value - center)
            for behavior, value in values.items()
        }

    def _relative_utility_feedback(self, evaluator, selected_behavior):
        """Evaluate a selected behavior relative to the available alternatives."""
        utility_values = {
            behavior: evaluator.behaviors[behavior][self.setup]["utility"]
            for behavior in self.behaviors
            if behavior in evaluator.behaviors
            and self.setup in evaluator.behaviors[behavior]
        }
        if selected_behavior not in utility_values:
            return 0.0
        alternatives = [
            value
            for behavior, value in utility_values.items()
            if behavior != selected_behavior
        ]
        reference = float(np.mean(alternatives)) if alternatives else 0.0
        contrast = float(utility_values[selected_behavior] - reference)
        strength = float(getattr(evaluator, "feedback_strength", 1.0))
        noise_scale = max(
            0.0,
            float(getattr(evaluator, "feedback_noise_scale", 0.0)),
        )
        noise = np.random.normal(0.0, noise_scale) if noise_scale else 0.0
        return strength * contrast + noise

    def _simulate_situation(self, seed=None):

        # Validate `seed`
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer or None, got {type(seed)}.")
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        try:
            if self.interaction_mode in self._simulation_methods:
                self._simulation_methods[self.interaction_mode]()
            else:
                raise ValueError(f"Undefined interaction_mode: {self.interaction_mode}")
        except KeyError as e:
            # Add more context to KeyErrors
            if isinstance(e.args[0], str) and e.args[0] in Agent._registry:
                raise KeyError(f"Agent {e.args[0]} exists in registry but relationship lookup failed") from e
            else:
                raise KeyError(f"Agent {e.args[0]} not found in registry") from e
        except Exception as e:
            print(f"ERROR in _simulate_situation for {self.selected_behavior.name if hasattr(self, 'selected_behavior') else 'unknown'}: {str(e)}")
            raise type(e)(f"{str(e)} - in {self.__class__.__name__}._simulate_situation") from e

    def _simulate_solitary(self):
        """
        Simulate a solitary scenario where the individual agent performs a behavior independently.
        """
        focal_momentary, focal_reported = self._sample_momentary_appraisals(self.individual)
        self._store_focal_appraisals(focal_momentary, focal_reported)
        choice_values = self._choice_values_from_appraisals(self.individual, focal_momentary)
        self.selected_behavior, self.choice_probs = self.individual.select_behavior(
            setup=self.setup,
            behaviors=self.behaviors,
            choice_probs=choice_values,
            return_probs=True
        )
        self.choice_prob = self.choice_probs.get(self.selected_behavior)
        self.suggestion_terms = {behavior: 0.0 for behavior in self.behaviors}

        # Simulate the outcome (utility)
        self.outcome = self._simulate_outcome(
            efforts=(1.0,),  # Fixed effort in the new model
            skills=(1.0,),   # No skill in the new model
            behavior=self.selected_behavior
        )

        # Simulate the experienced pleasure (enjoyment)
        experienced_pleasure = self._simulate_pleasure(self.selected_behavior, self.individual)
        self.perceived_utility = self.outcome[0]
        self.perceived_enjoyment = experienced_pleasure
        self.learning_behavior = self.selected_behavior
        self.learning_role = "choice"
        self.focal_perceived_utility = self.outcome[0]
        self.focal_perceived_enjoyment = experienced_pleasure
        self.raw_utility_out = self.outcome[0]
        self.raw_enjoyment_out = experienced_pleasure

        # Update the individual's behavior parameters
        self.individual.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            performed=True,
            perceived_utility=self.outcome[0],
            perceived_enjoyment=experienced_pleasure
        )

        # For behaviors not selected, update instinct as alternative behaviors
        for behavior in self.behaviors:
            if behavior != self.selected_behavior:
                self.individual.update_behavior(
                    behavior=behavior,
                    setup=self.setup,
                    performed=True,
                    alternative_behavior=True
                )

    def _simulate_co_participation(self):
        """
        Simulate a co-participation scenario in which both the individual agent and the environment agent
        actively engage and contribute simultaneously. This approach contrasts with _simulate_observational(),
        where one agent takes the primary role while the other mainly observes and provides occasional feedback.

        In the tripartite model with multi-agent extension, co-participation involves:
            - Both agents forming choice values (via form_intention) based on their instinct, enjoyment, and utility predictions
            - Agent B's preferences serving as suggestions to agent A, qualified by agent A's receptivity
            - A behavior is selected using the softmax probabilistic model
            - Both agents experience outcomes and update their parameters with social influence

        Parameters:
            None

        Returns:
            None

        Side Effects:
            - Sets self.selected_behavior to the behavior chosen based on blended drift rates
            - Invokes updates on both agents by calling their respective update_behavior methods
        """
        # ===== Phase 1: Initialize Agents =====
        agent_a = self.individual
        agent_b = self.environment

        # ===== Phase 2: Get Relationship Parameters =====
        a_to_b = agent_a.relationships.get(agent_b.id, {})
        b_to_a = agent_b.relationships.get(agent_a.id, {})

        # Get receptivity parameters (how much each agent is influenced by the other)
        receptivity_a = a_to_b.get('receptivity', 0)  # How much A is influenced by B
        receptivity_b = b_to_a.get('receptivity', 0)  # How much B is influenced by A

        # Get communion parameters (affective quality of relationship)
        communion_a = a_to_b.get('connection', 0)  # How A feels about B
        communion_b = b_to_a.get('connection', 0)  # How B feels about A

        # ===== Phase 3: Choice Value Blending & Behavior Selection =====
        momentary_a, reported_a = self._sample_momentary_appraisals(agent_a)
        momentary_b, _ = self._sample_momentary_appraisals(agent_b)
        self._store_focal_appraisals(momentary_a, reported_a)
        choice_values_a = self._choice_values_from_appraisals(agent_a, momentary_a)
        choice_values_b = self._choice_values_from_appraisals(agent_b, momentary_b)

        # Agent B's preference serves as a suggestion to agent A.
        centered_b = self._center_values(choice_values_b)
        blended_values = {}
        for behavior in set(choice_values_a) | set(choice_values_b):
            val_a = choice_values_a.get(behavior, 0)
            raw_suggestion = centered_b.get(behavior, 0)
            blended_values[behavior] = integrate_suggestion(
                val_a,
                raw_suggestion,
                receptivity_a,
                agent_a.kappa_suggestion,
            )

        # Select behavior using the softmax probabilistic model
        # Use agent_a for parameter access, but pass blended values
        self.selected_behavior, self.choice_probs = agent_a.select_behavior(
            setup=self.setup,
            behaviors=list(blended_values.keys()),
            choice_probs=blended_values,
            return_probs=True
        )
        self.choice_prob = self.choice_probs.get(self.selected_behavior)
        self.suggestion_terms = {}
        for behavior in self.behaviors:
            self.suggestion_terms[behavior] = qualify_social_signal(
                centered_b.get(behavior, 0),
                receptivity_a,
            )

        # ===== Phase 4: Utility Outcome Determination =====
        # Simulate outcome (utility)
        self.outcome = self._simulate_outcome(
            efforts=(1.0,),  # Fixed effort in the new model
            skills=(1.0,),   # No skill in the new model
            behavior=self.selected_behavior
        )
        base_utility = self.outcome[0]

        # ===== Phase 5: Social Influence on Learning =====
        # Simulate base enjoyment for each agent
        base_enjoyment_a = self._simulate_pleasure(self.selected_behavior, agent_a)
        base_enjoyment_b = self._simulate_pleasure(self.selected_behavior, agent_b)

        # Apply social presence effect on enjoyment for each agent
        # E_A = E_{A(alone)} + C_{A,O} * P_O
        social_enjoyment_a = base_enjoyment_a + communion_a * 1.0  # P_O = 1.0 for simplicity
        social_enjoyment_b = base_enjoyment_b + communion_b * 1.0

        # Ensure values stay within valid range
        social_enjoyment_a = np.clip(social_enjoyment_a, -1, 1)
        social_enjoyment_b = np.clip(social_enjoyment_b, -1, 1)

        # Each agent evaluates the selected behavior relative to alternatives.
        feedback_a_to_b = self._relative_utility_feedback(
            agent_a, self.selected_behavior
        )
        feedback_b_to_a = self._relative_utility_feedback(
            agent_b, self.selected_behavior
        )
        self.feedback_signal = {
            "agent_a_to_b": feedback_a_to_b,
            "agent_b_to_a": feedback_b_to_a,
        }

        social_utility_a = integrate_feedback(
            base_utility,
            feedback_b_to_a,
            receptivity_a,
            agent_a.kappa_feedback,
        )
        social_utility_b = integrate_feedback(
            base_utility,
            feedback_a_to_b,
            receptivity_b,
            agent_b.kappa_feedback,
        )

        # Ensure values stay within valid range
        social_utility_a = np.clip(social_utility_a, -1, 1)
        social_utility_b = np.clip(social_utility_b, -1, 1)
        self.perceived_utility = {
            "agent_a": social_utility_a,
            "agent_b": social_utility_b
        }
        self.perceived_enjoyment = {
            "agent_a": social_enjoyment_a,
            "agent_b": social_enjoyment_b
        }
        self.learning_behavior = self.selected_behavior
        self.learning_role = "choice"
        self.focal_perceived_utility = social_utility_a
        self.focal_perceived_enjoyment = social_enjoyment_a
        self.raw_utility_out = base_utility
        self.raw_enjoyment_out = base_enjoyment_a

        # ===== Phase 6: Update Both Agents =====
        # Update agent A
        agent_a.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            performed=True,
            perceived_utility=social_utility_a,
            perceived_enjoyment=social_enjoyment_a
        )

        # Update agent B
        agent_b.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            performed=True,
            perceived_utility=social_utility_b,
            perceived_enjoyment=social_enjoyment_b
        )

        # For behaviors not selected, update instinct as alternative behaviors for both agents
        for behavior in self.behaviors:
            if behavior != self.selected_behavior:
                # Update agent A
                agent_a.update_behavior(
                    behavior=behavior,
                    setup=self.setup,
                    performed=True,
                    alternative_behavior=True
                )

                # Update agent B
                agent_b.update_behavior(
                    behavior=behavior,
                    setup=self.setup,
                    performed=True,
                    alternative_behavior=True
                )

    def _simulate_observational(
        self,
        observer_suggestion=True,
        observer_feedback=True,
        observer_penalty=0.5,
        role_switching=False
    ):
        """
        Simulates an observational learning scenario where an active agent performs a behavior
        while an observer may provide suggestions and feedback.

        Parameters:
        - observer_suggestion (bool): If True, the observer provides suggestions that influence the active agent's drift rates.
        - observer_feedback (bool): If True, the observer provides feedback that influences the active agent's learning.
        - observer_penalty (float): The penalty applied to observational learning (0 to 1).
        - role_switching (bool): If True, the roles of the active agent and observer are switched.

        The simulation implements the multi-agent extension of the tripartite model with three phases:
        1. Decision-Making: The active agent forms drift rates, potentially influenced by the observer's suggestions.
        2. Action Execution: The active agent executes the selected behavior and outcomes are simulated.
        3. Feedback and Learning: Both agents update their parameters based on the outcome and social feedback.

        Returns:
        None
        """

        # Determine roles
        active_agent, observer = (self.environment, self.individual) if role_switching \
                              else (self.individual, self.environment)

        # Get relationship parameters from ACTIVE AGENT'S perspective
        relationship = active_agent.relationships.get(observer.id, {})
        receptivity = relationship.get("receptivity", 0)  # How open active is to suggestions
        communion = relationship.get("connection", 0)  # Affective quality of relationship

        # === Phase 1: Decision-Making with Social Influence ===
        focal_momentary, focal_reported = self._sample_momentary_appraisals(self.individual)
        self._store_focal_appraisals(focal_momentary, focal_reported)
        if self.individual.id == active_agent.id:
            active_momentary = focal_momentary
        else:
            active_momentary, _ = self._sample_momentary_appraisals(active_agent)
        active_values = self._choice_values_from_appraisals(active_agent, active_momentary)

        # Apply suggestion influence if enabled
        self.suggestion_terms = {behavior: 0.0 for behavior in self.behaviors}

        if observer_suggestion:
            # Get observer's choice values
            if self.individual.id == observer.id:
                observer_momentary = focal_momentary
            else:
                observer_momentary, _ = self._sample_momentary_appraisals(observer)
            observer_values = self._choice_values_from_appraisals(observer, observer_momentary)
            centered_observer_values = self._center_values(observer_values)

            # Calculate and integrate the relationship-qualified suggestion.
            blended_values = {}
            for behavior in set(active_values) | set(observer_values):
                base_val = active_values.get(behavior, 0)
                raw_suggestion = centered_observer_values.get(behavior, 0)
                blended_values[behavior] = integrate_suggestion(
                    base_val,
                    raw_suggestion,
                    receptivity,
                    active_agent.kappa_suggestion,
                )
            for behavior in self.behaviors:
                self.suggestion_terms[behavior] = qualify_social_signal(
                    centered_observer_values.get(behavior, 0),
                    receptivity,
                )
        else:
            blended_values = active_values

        # Select behavior using softmax probabilistic model
        self.selected_behavior, self.choice_probs = active_agent.select_behavior(
            setup=self.setup,
            behaviors=list(blended_values.keys()),
            choice_probs=blended_values,
            return_probs=True
        )
        self.choice_prob = self.choice_probs.get(self.selected_behavior)

        # === Phase 2: Action Execution ===
        # Simulate outcome (utility)
        self.outcome = self._simulate_outcome(
            efforts=(1.0,),  # Fixed effort in the new model
            skills=(1.0,),   # No skill in the new model
            behavior=self.selected_behavior
        )

        # === Phase 3: Social Influence on Learning ===
        # Simulate base enjoyment for active agent
        base_enjoyment = self._simulate_pleasure(self.selected_behavior, active_agent)

        # Apply social presence effect on enjoyment: E_A = E_{A(alone)} + C_{A,O} * P_O
        # Where P_O is the observer's participation/salience (set to 1.0 for simplicity)
        social_enjoyment = base_enjoyment + communion * 1.0
        social_enjoyment = np.clip(social_enjoyment, -1, 1)

        # Get base utility outcome
        base_utility = self.outcome[0]

        # Apply social feedback effect on utility if enabled
        if observer_feedback:
            feedback = self._relative_utility_feedback(
                observer, self.selected_behavior
            )
            self.feedback_signal = feedback
            social_utility = integrate_feedback(
                base_utility,
                feedback,
                receptivity,
                active_agent.kappa_feedback,
            )
            social_utility = np.clip(social_utility, -1, 1)
        else:
            self.feedback_signal = 0.0
            social_utility = base_utility

        self.perceived_enjoyment = social_enjoyment
        self.perceived_utility = social_utility
        self.learning_behavior = self.selected_behavior
        self.raw_utility_out = base_utility
        self.raw_enjoyment_out = base_enjoyment

        if self.individual.id == active_agent.id:
            self.learning_role = "choice"
            self.focal_perceived_utility = social_utility
            self.focal_perceived_enjoyment = social_enjoyment
        else:
            observer_relationship = observer.relationships.get(active_agent.id, {})
            observer_receptivity = observer_relationship.get("receptivity", 0)
            self.learning_role = "observe"
            self.focal_perceived_utility = base_utility * observer_receptivity
            self.focal_perceived_enjoyment = base_enjoyment * observer_receptivity

        # Update active agent's behavior parameters
        active_agent.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            performed=True,
            perceived_utility=social_utility,
            perceived_enjoyment=social_enjoyment
        )

        # Update observer through observational learning
        observer.update_behavior(
            behavior=self.selected_behavior,
            setup=self.setup,
            performed=False,
            perceived_utility=base_utility,
            perceived_enjoyment=base_enjoyment,
            observer_penalty=observer_penalty,
            observed_agent=active_agent
        )

        # For behaviors not selected, update instinct as alternative behaviors for both agents
        for behavior in self.behaviors:
            if behavior != self.selected_behavior:
                # Active agent learns this was not selected
                active_agent.update_behavior(
                    behavior=behavior,
                    setup=self.setup,
                    performed=True,
                    alternative_behavior=True
                )

                # Observer learns vicariously
                observer.update_behavior(
                    behavior=behavior,
                    setup=self.setup,
                    performed=False,
                    alternative_behavior=True,
                    observer_penalty=observer_penalty
                )

    def _simulate_talking(self, benefits_exchange=True, skill_exchange=True):
        """
        Simulate a talking scenario where two agents discuss the setup and a set of behaviors.
        First, the agents form intentions based on the setup and behaviors.
        Then, they express their intentions behavior based on the combined intentions.
        Finally, they update their evaluations based on the outcome and feedback.

        Parameters:
        - benefits_exchange (bool): If True, agents exchange benefits.
        - skill_exchange (bool): If True, agents exchange skills.
        """
        return


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
        - float: Effort value between 0 and 1.
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

    def _simulate_pleasure(self, behavior, agent, co_participant=None):
        """
        Simulate pleasure experienced from performing a behavior.

        Parameters:
        - behavior: The behavior being performed
        - agent: The agent experiencing pleasure
        - co_participant: Optional co-participant that may affect pleasure through connection
        """
        # Get agent's parameters for this behavior in this setup
        params = agent.behaviors[behavior][self.setup]
        setup_mods = behavior.setup_modifiers.get(self.setup, {})
        if "enjoyment_outcome" in setup_mods:
            base_enjoyment = setup_mods["enjoyment_outcome"]
        else:
            base_enjoyment = params['enjoyment'] + setup_mods.get("enjoyment_outcome_mod", 0.0)
        base_enjoyment = np.clip(base_enjoyment, -1, 1)

        # Apply habituation effect to raw pleasure
        exposure = params.get('exposure_count', 0)
        habituation_factor = 1.0 / (1.0 + 0.05 * exposure)

        # For initially enjoyable behaviors: pleasure decreases with exposure
        # For initially unpleasant behaviors: displeasure decreases with exposure
        modulated_enjoyment = base_enjoyment * habituation_factor

        # Apply connection effect if co-participating
        connection_effect = 0
        if co_participant:
            try:
                # Get connection value from relationship
                connection = agent.relationships[co_participant.id].get('connection', 0)

                # Calculate connection effect: stronger effect for extreme enjoyment values
                # For positive connection: enhances pleasure or reduces displeasure
                # For negative connection: reduces pleasure or enhances displeasure
                connection_effect = connection * abs(base_enjoyment) * 0.3
            except KeyError:
                # Handle case where relationship doesn't exist
                pass

        # Apply randomization for variability in experiences
        enjoyment_volatility = setup_mods.get("enjoyment_volatility", 0.1)
        noise = np.random.normal(0, enjoyment_volatility)

        # Calculate final pleasure value with connection effect
        pleasure = modulated_enjoyment + connection_effect + noise

        # Ensure pleasure stays within valid range
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
        outcome_volatility = setup_mods.get("outcome_volatility", behavior.outcome_volatility)

        # Clip to valid ranges
        base_outcome = np.clip(base_outcome, -1, 1)
        difficulty = np.clip(difficulty, 0, 1)
        outcome_volatility = np.clip(outcome_volatility, 0, 1)

        # Calculate combined output from all contributors
        output = sum(e * s for e, s in zip(efforts, skills))

        # Determine success threshold
        if output >= difficulty:
            raw_outcome = base_outcome
        else:
            failure_ratio = (difficulty - output) / difficulty
            raw_outcome = base_outcome * (1 - failure_ratio)

        # Add volatility and clamp
        outcome = np.clip(raw_outcome + np.random.normal(0, outcome_volatility), -1, 1)
        return outcome, raw_outcome
