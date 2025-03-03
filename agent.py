import numpy as np
import uuid
import weakref
from behavior import Behavior

class Agent:
    # Global registry: {agent_id: Agent}
    _registry = weakref.WeakValueDictionary()  # Automatically removes unreferenced agents

    @classmethod
    def get_agent(cls, agent_id):
        """Retrieve an agent from the registry by ID."""
        return cls._registry.get(agent_id)

    @classmethod
    def add_relationship(cls, agent_a, agent_b, distance, receptivity, power=0):
        """
        Modified to handle cultural relationships properly:
        - Cultures never have reciprocal relationships
        - Cultural relationships have power=0 by default
        """
        # Validate parameters
        for param in [distance, receptivity, power]:
            if not (0 <= param <= 1):
                raise ValueError(f"Parameter {param} must be in [0,1]")

        # Cultural relationship handling
        if isinstance(agent_b, Culture):
            # Unidirectional: Only agent_a gets relationship
            agent_a.relationships[agent_b.id] = {
                'distance': distance,
                'receptivity': receptivity,
                'power': 0.0  # Force power=0 for cultural relationships
            }
        elif isinstance(agent_a, Culture):
            raise ValueError("Cultures cannot initiate relationships")
        else:
            # Standard bidirectional relationship
            agent_a.relationships[agent_b.id] = {
                'distance': distance,
                'receptivity': receptivity,
                'power': power
            }
            agent_b.relationships[agent_a.id] = {
                'distance': distance,
                'receptivity': receptivity,
                'power': power
            }

    @classmethod
    def get_relationship(cls, agent_a, agent_b):
        """Retrieve a relationship between two agents."""
        return agent_a.relationships.get(agent_b.id)

    def __init__(self, name="", setups=None, behaviors=None, process_matrix=None, relationships=None):
        """
        Base class for all agents (individuals or groups).

        Parameters:
        - name (str): Optional name for the agent.
        - setups (dict): Setup configurations (e.g., affinity benefits).
        - behaviors (dict): Hierarchical dictionary of behaviors, setups and parameters.
        - process_matrix (dict): Weights for decision-making, evaluation, and learning.
        - relationships (dict): Hierarchical dictionary of relationships with other agents, structured as:
            {agent_id: {"distance": float, "receptivity": float, "power": float}}
        """

        self.id = str(uuid.uuid4())  # Unique identifier in the global registry
        self.name = name
        # Validate setups
        self.setups = {}
        if setups:
            if not isinstance(setups, dict):
                raise TypeError("Setups must be a dictionary with setup names as keys and parameter dictionaries as benefits.")
            for setup, params in setups.items():
                if not isinstance(params, dict) or 'affinity' not in params:
                    raise ValueError(f"Each setup must be a dictionary with an 'affinity' parameter. Problematic setup: {setup}")
                if not (0 <= params['affinity'] <= 1):
                    raise ValueError(f"Affinity for setup {setup} must be in range [0, 1].")
                self.setups[setup] = params

        # Validate behaviors
        self.behaviors = {}
        if behaviors:
            if not isinstance(behaviors, dict):
                raise TypeError("Behaviors must be a hierarchical dictionary with Behavior instances as keys.")
            for behavior, setups_data in behaviors.items():
                if not isinstance(behavior, Behavior):
                    raise TypeError(f"Key {behavior} is not an instance of the Behavior class.")
                if not isinstance(setups_data, dict):
                    raise TypeError(f"Value for behavior {behavior} must be a dictionary of setups and parameters.")
                self.behaviors[behavior] = {}
                for setup, params in setups_data.items():
                    required_params = {'instinct', 'benefit', 'skill', 'alpha_instinct', 'alpha_benefit', 'alpha_skill'}
                    if not isinstance(params, dict) or not required_params.issubset(params):
                        raise ValueError(f"Setup {setup} for behavior {behavior} must contain parameters: {required_params}")
                    # Validate parameter ranges
                    if not (-1 <= params['instinct'] <= 1):
                        raise ValueError(f"instinct for behavior {behavior}, setup {setup} must be in range [-1, 1].")
                    if not (-1 <= params['benefit'] <= 1):
                        raise ValueError(f"Value for behavior {behavior}, setup {setup} must be in range [-1, 1].")
                    if not (0 <= params['skill'] <= 1):
                        raise ValueError(f"Skill for behavior {behavior}, setup {setup} must be in range [0, 1].")
                    if not (-1 <= params['enjoyment'] <= 1):
                        raise ValueError(f"Enjoyment for behavior {behavior}, setup {setup} must be in range [-1, 1].")
                    if not (0 <= params['alpha_instinct'] <= 1):
                        raise ValueError(f"Alpha_instinct for behavior {behavior}, setup {setup} must be in range [0, 1].")
                    if not (0 <= params['alpha_benefit'] <= 1):
                        raise ValueError(f"Alpha_benefit for behavior {behavior}, setup {setup} must be in range [0, 1].")
                    if not (0 <= params['alpha_skill'] <= 1):
                        raise ValueError(f"Alpha_skill for behavior {behavior}, setup {setup} must be in range [0, 1].")
                    self.behaviors[behavior][setup] = params

        # Validate relationships
        self.relationships = {}
        if relationships:
            if not isinstance(relationships, dict):
                raise TypeError("Relationships must be a dictionary.")
            for agent_id, params in relationships.items():
                other_agent = get_agent(agent_id)
                required_params = {'distance', 'receptivity', 'power'}
                if not isinstance(params, dict) or not required_params.issubset(params):
                    raise ValueError(f"Agent {agent} must contain parameters: {required_params}")
                # Validate parameter ranges
                if not (0 <= params['distance'] <= 1):
                    raise ValueError(f"Distance to agent {agent} must be in range [0, 1].")
                if not (-1 <= params['receptivity'] <= 1):
                    raise ValueError(f"Receptivity to agent {agent} must be in range [-1, 1].")
                if not (-1 <= params['power'] <= 1):
                    raise ValueError(f"Power to agent {agent} must be in range [-1, 1].")
                classmethod.add_relationship(self, other_agent, params['distance'], params['power'], params['receptivity'])

        # Validate process_matrix
        allowed_process_keys = ['instinct', 'benefit', 'skill', 'intention', 'outcome', 'effort', 'pleasure']
        self.process_matrix = {}
        if process_matrix:
            if not isinstance(process_matrix, dict):
                raise TypeError("Process_matrix must be a dictionary.")
            for key, benefit in process_matrix.items():
                if key not in allowed_process_keys:
                    raise ValueError(f"Invalid process_matrix key: {key}. Allowed keys are {allowed_process_keys}.")
                self.process_matrix[key] = benefit
        else:
            self.process_matrix = {key: 1 for key in allowed_process_keys}


        self.setups = setups if setups is not None else {}
        self.behaviors = behaviors if behaviors is not None else {}
        self.process_matrix = process_matrix if process_matrix is not None else {}
        self.relationships = relationships if relationships is not None else {}
        Agent._registry[self.id] = self  # Register the agent

    def add_setup(self, setup, affinity=0):
        """
        Add a new setup to the setups memory or update an existing setup's affinity.

        Parameters:
        - setup (str): The name of the new setup.
        - affinity (float): The affinity of the new setup [0, 1]. Default is 0.

        Raises:
        - ValueError: If `affinity` is not in the range [0, 1].
        """
        # Validate affinity range
        if not (0 <= affinity <= 1):
            raise ValueError("Affinity must be in the range [0, 1].")

        # Add or update the setup in the setup matrix
        self.setups[setup] = affinity  # Update existing setup's affinity

    def add_behavior(self, behavior, setup, instinct, benefit, skill,\
                     enjoyment, alpha_instinct, alpha_benefit, alpha_skill, affinity=0):
        """
        Add/update a behavior for a specific setup with all parameters.

        Parameters:
        - behavior (Behavior): an object of the Behavior class.
        - setup (str): The name of the setup.
        - instinct ():
        """
        # Validate input ranges
        params = {
            "instinct": instinct,
            "benefit": benefit,
            "skill": skill,
            "enjoyment": enjoyment,
            "alpha_instinct": alpha_instinct,
            "alpha_benefit": alpha_benefit,
            "alpha_skill": alpha_skill
        }
        for key, value in params.items():
            if key in ["instinct", "benefit", "enjoyment"] and not (-1 <= value <= 1):
                raise ValueError(f"{key} must be in [-1, 1], got {value}.")
            elif key in ["skill", "alpha_instinct", "alpha_benefit", "alpha_skill"] and not (0 <= value <= 1):
                raise ValueError(f"{key} must be in [0, 1], got {value}.")

        # Add setup if missing
        if setup not in self.setups:
            self.add_setup(setup, affinity)

        # Add/update behavior
        if behavior not in self.behaviors:
            self.behaviors[behavior] = {}
        self.behaviors[behavior][setup] = params

    def add_relationship(self, other_agent, distance=0.5, receptivity=0.5, power=0.5):
        """Link to another agent using their ID instead of the object."""
        if not isinstance(other_agent, Agent):
            raise TypeError("Relationships can only be formed with Agent instances.")
        classmethod.add_relationship(self, other_agent.id, distance, receptivity, power)

    def form_intention(self, setup, behaviors=None):
        """Calculate intention scores for behaviors, incorporating Group dynamics.
        This method cannot be called by a Culture agent.
        """
        # Filter behaviors for the setup
        filtered = self._filter_behaviors(setup, behaviors)

        intentions = {}
        for behavior in filtered:
            params = self.behaviors[behavior][setup]
            w_instinct = self.process_matrix.get('instinct', 1)
            w_benefit = self.process_matrix.get('benefit', 1)
            w_skill = self.process_matrix.get('skill', 1)

            # Base intention calculation
            base_intention = (
                w_instinct * params['instinct'] +
                w_benefit * params['benefit'] +
                w_skill * params['skill']
            )

            # Group-specific adjustments
            if isinstance(self, Group):
                # Add noise inversely proportional to homogeneity
                noise = np.random.normal(0, scale=(1 - self.homogeneity))
                base_intention += noise

                # Scale by group size (logarithmic damping)
                base_intention *= np.log(self.size + 1) / (self.size + 1)

            intentions[behavior] = np.clip(base_intention, 0, 1)

        # Add cultural influence
        for agent, params in self.relationships.items():
            if not isinstance(agent, Culture):
                continue # Iterate through all Cultures in the Agent's relationships

            if np.random.rand() > params['distance']:  # Cultural exposure chance
                cultural_norms = agent.form_intention(setup, behaviors)
                rec = params['receptivity']

                for behavior in behaviors:
                    base_intentions[behavior] += rec * cultural_norms.get(behavior, 0)

        return {b: np.clip(v, 0, 1) for b, v in base_intentions.items()}


    def form_evaluation(self, intention, outcome, effort, pleasure):
        """Calculate evaluation with Group-specific sensitivity."""
        weights = [
            self.process_matrix.get('intention', 1),
            self.process_matrix.get('outcome', 1),
            self.process_matrix.get('effort', 1),
            self.process_matrix.get('pleasure', 1)
        ]
        total_weight = sum(weights)

        # Base evaluation
        evaluation = (
            weights[0] * intention +
            weights[1] * outcome +
            weights[2] * effort +
            weights[3] * pleasure
        ) / total_weight

        # Group-specific adjustments
        if isinstance(self, Group):
            # Dampen evaluation variance through homogeneity
            evaluation *= self.homogeneity

            # Scale by group size (larger groups less sensitive)
            evaluation /= np.log(self.size + 1)

        return np.clip(evaluation, -1, 1)

    def update_behavior(self, behavior, setup, evaluation, performed, observer_penalty):
        """
        Update parameters with the agent's learning rates.

        Parameters:
        - behavior (Behavior): the behavior performed, observed, or considered

        """
        params = self.behaviors[behavior][setup]

        # Base learning rates
        alpha_instinct = params['alpha_instinct']
        alpha_benefit = params['alpha_benefit']
        alpha_skill = params['alpha_skill']

        # Group-specific learning modulation
        if isinstance(self, Group):
            # Learning inhibited by size and enhanced by homogeneity
            size_factor = 1 / np.sqrt(self.size)
            alpha_instinct *= self.homogeneity * size_factor
            alpha_benefit *= self.homogeneity * size_factor
            alpha_skill *= self.homogeneity * size_factor

        # Parameter updates
        if performed:
            params['instinct'] += alpha_instinct * (1 - params['instinct'] * np.sign(evaluation))
            params['benefit'] += alpha_benefit * (evaluation - params['benefit'])
        else:  # Observational learning
            params['instinct'] += observer_penalty * alpha_instinct * (evaluation - params['instinct'])

        params['skill'] += alpha_skill * (evaluation - params['skill'])

        # Enforce boundaries
        params['instinct'] = np.clip(params['instinct'], -1, 1)
        params['benefit'] = np.clip(params['benefit'], -1, 1)

        # Cultural resource boost
        for culture, params in self.relationships.items():
            if not isinstance(culture, Culture):
                continue

            res = culture.get_cultural_params(behavior, setup)['resource']
            params['skill'] += params['receptivity'] * res * (1 - params['distance'])

        params['skill'] = np.clip(new_skill, 0, 1)


    def _filter_behaviors(self, setup, behaviors):
        # Filter behaviors for the given setup
        if behaviors is None:
            # Use all behaviors associated with the setup if not provided
            filtered_behaviors = {
                behavior: setups
                for behavior, setups in self.behaviors.items()
                if setup in setups
            }
        else:
            # Validate and filter the specified behaviors
            filtered_behaviors = {
                behavior: self.behaviors[behavior]
                for behavior in behaviors
                if behavior in self.behaviors and setup in self.behaviors[behavior]
            }
            if len(filtered_behaviors) < len(behaviors):
                raise ValueError("One or more specified behaviors are not associated with the specified setup.")

        if not filtered_behaviors:
            raise ValueError(f"No behaviors are associated with the setup '{setup}'.")

        return filtered_behaviors

class Individual(Agent):
    def __init__(self, name="", setups=None, behaviors=None, process_matrix=None, relationships=None):
        super().__init__(setups, behaviors, process_matrix, relationships)
        # Individual-specific attributes can be added here in the future (e.g., personality traits)

    def form_intention(self, setup, behaviors=None):
        super().form_intention(setup, behaviors)

    def form_evaluation(self, intention, outcome, effort, pleasure):
        super().form_evaluation(intention, outcome, effort, pleasure)


    def update_behavior(self, behavior, setup, evaluation=None, skill_gain=0):
        """
        Update the individual's parameters associated with relevant behaviors after a situation.

        Parameters:
        - behavior (Behavior): The behavior performed, observed, or discussed.
        - setup (str): The setup of the behavior.
        - evaluation (float): The final evaluation of the behavior [-1, 1]. Default is None.
        - skill_gain (float): The amount of skill increase [0, 1]. Default is 0.

        """
        # Validate the setup and behavior
        if setup not in self.setups['setup'].benefits:
            raise ValueError(f"The setup '{setup}' is not among the setups the individual knows.")
        if behavior not in self.behaviors:
            raise ValueError(f"The behavior '{behavior}' is not among the behaviors the individual knows.")

        # Learning rate (could be made more sophisticated)
        alpha_instinct = 0.1  # instinct learning rate
        alpha_benefit = 0.1  # Value learning rate

        # Extract behaviors relevant to the setup
        relevant_behaviors = {
            b: d for b, d in self.behaviors.items() if d['setup'] == setup
        }

        # 1. instinct Update
        for b, details in relevant_behaviors.items():
            if b == behavior:
                # Increase instinct for the selected behavior
                details['instinct'] += alpha_instinct * (1 - details['instinct'])
            else:
                # Decrease instinct for unselected behaviors
                details['instinct'] -= alpha_instinct * details['instinct']
            # Ensure instinct remains in [-1, 1]
            details['instinct'] = max(-1, min(1, details['instinct']))

        # 2. Value Update (only for the selected behavior)
        if evaluation is not None:
            current_benefit = self.behaviors[behavior]['benefit']
            self.behaviors[behavior]['benefit'] += alpha_benefit * (evaluation - current_benefit)
            # Ensure benefit remains in [-1, 1]
            self.behaviors[behavior]['benefit'] = max(-1, min(1, self.behaviors[behavior]['benefit']))

        # 3. Skill Update (only for the selected behavior)
        if skill_gain > 0:
            current_skill = self.behaviors[behavior]['skill']
            self.behaviors[behavior]['skill'] += skill_gain
            # Ensure skill remains in [0, 1]
            self.behaviors[behavior]['skill'] = max(0, min(1, self.behaviors[behavior]['skill']))

class Group(Agent):
    def __init__(self, name="", setups=None, behaviors=None, process_matrix=None, relationships=None, size=1, homogeneity=1.0):
        super().__init__(setups, behaviors, process_matrix, relationships)
        self.size = size
        self.homogeneity = homogeneity

class Culture(Agent):
    def __init__(self, cultural_norms, relationships=None, name=""):
        """
        Static cultural framework that influences agents through norms/merit/resources.
        Inherits from Agent but with frozen parameters.
        """
        super().__init__(
            setups={},  # Cultures don't have setups
            behaviors=cultural_norms,  # Reuse behaviors dict for norms: {Behavior: {Setup: {normativity, merit, resource}}
            process_matrix={},  # No decision-making process
            relationships=relationships if relationships else {},
        )
        self.name = name
        self.frozen = True  # Immutable flag

    def update_behavior(self, *args, **kwargs):
        """Override to disable updates for static cultures"""
        if self.frozen:
            raise NotImplementedError("Culture parameters cannot be updated")
        super().update_behavior(*args, **kwargs)

    def form_intention(self, setup, behaviors=None):
        """Cultural intentions represent ideal norms, not actual behaviors"""
        return {b: self.behaviors[b][setup]['normativity']
                for b in (behaviors or self.behaviors)}

    def get_cultural_params(self, behavior, setup):
        """Helper to access normativity, merit, resource"""
        return self.behaviors.get(behavior, {}).get(setup, {
            'normativity': 0.5,
            'merit': 0,
            'resource': 0.5
        })