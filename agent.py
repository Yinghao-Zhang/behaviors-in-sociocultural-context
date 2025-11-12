import numpy as np
import uuid
from setup import Setup
from behavior import Behavior
import json


class HyperparameterManager:
    """Centralized manager for all tunable hyperparameters."""
    def __init__(self, config=None):
        # Default values (can be loaded/updated)
        self.params = {
            'alpha_instinct_plus': 0.1,
            'alpha_instinct_minus': 0.1,
            'alpha_utility': 0.1,
            'alpha_enjoyment': 0.1,
            'w_enjoyment': 0.5,
            'w_utility': 0.5,
            'bias_scaling_factor': 1.0
        }
        if config:
            self.params.update(config)

    def get(self, key):
        return self.params.get(key)

    def set(self, key, value):
        self.params[key] = value

    def load_from_dict(self, config):
        self.params.update(config)

# Singleton instance
hyperparam_manager = HyperparameterManager()

class Agent:
    # Global registry for all agents
    _registry = {}  # Use regular dictionary
    _debug_log = []  # Track registry operations

    @classmethod
    def get_agent(cls, agent_id):
        """Get agent by ID with better error handling."""
        if agent_id not in cls._registry:
            # Log the missing agent info
            cls._debug_log.append(f"Missing agent: {agent_id}")
            raise KeyError(f"Agent '{agent_id}' not found in registry")
        return cls._registry[agent_id]

    @classmethod
    def remove_agent(cls, agent_id):
        """Safely remove an agent from the registry and clean up references."""
        if agent_id in cls._registry:
            # First remove this agent from all other agents' relationship dictionaries
            for other_agent in cls._registry.values():
                if agent_id in other_agent.relationships:
                    del other_agent.relationships[agent_id]

            # Then remove the agent from registry
            del cls._registry[agent_id]
            cls._debug_log.append(f"Removed agent: {agent_id}")

    @classmethod
    def add_relationship(cls, agent1_id, agent2_id, distance, receptivity, power, connection=0):
        """Use IDs instead of objects for relationships"""
        agent1 = cls.get_agent(agent1_id)
        agent2 = cls.get_agent(agent2_id)

        # Store relationships by ID
        agent1.relationships[agent2_id] = {
            'distance': distance,
            'receptivity': receptivity,
            'power': power,
            'connection': connection
        }

        # Store reciprocal relationship
        agent2.relationships[agent1_id] = {
            'distance': distance,
            'receptivity': power,
            'power': receptivity,
            'connection': connection
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
        - setups (list): List of Setup objects the agent is aware of.
        - behaviors (dict): Hierarchical dictionary of behaviors and their parameters.
        - process_matrix (dict): Weights for decision-making, evaluation, and learning.
        - relationships (dict): Hierarchical dictionary of relationships with other agents, structured as:
            {agent_id: {"distance": float, "receptivity": float, "power": float}}
        """
        self.id = str(uuid.uuid4())
        self.name = name

        # Validate and store setups
        self.setups = []
        if setups:
            if not isinstance(setups, (list, tuple, Setup)):
                raise TypeError("Setups must be a list or tuple of Setup objects")
            for setup in setups:
                if not isinstance(setup, Setup):
                    raise TypeError(f"Expected Setup object, got {type(setup)}")
            self.setups = list(setups)

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
                    if not isinstance(setup, Setup):
                        raise TypeError(f"Setup key must be a Setup object, got {type(setup)}")
                    required_params = {'instinct', 'utility', 'enjoyment', 'exposure_count',
                                    'alpha_instinct_plus', 'alpha_instinct_minus', 'alpha_utility', 'alpha_enjoyment'}
                    if not isinstance(params, dict) or not required_params.issubset(params):
                        raise ValueError(f"Setup {setup} for behavior {behavior} must contain parameters: {required_params}")
                    # Validate parameter ranges
                    self._validate_behavior_params(behavior, setup, params)
                    self.behaviors[behavior][setup] = params

        # Validate relationships
        self.relationships = {}
        if relationships:
            if not isinstance(relationships, dict):
                raise TypeError("Relationships must be a dictionary.")
            for agent_id, params in relationships.items():
                other_agent = Agent.get_agent(agent_id)
                required_params = {'distance', 'receptivity', 'power', 'connection'}
                if not isinstance(params, dict) or not required_params.issubset(params):
                    raise ValueError(f"Agent {other_agent} must contain parameters: {required_params}")
                # Validate parameter ranges
                if not (0 <= params['distance'] <= 1):
                    raise ValueError(f"Distance to agent {other_agent} must be in range [0, 1].")
                if not (-1 <= params['receptivity'] <= 1):
                    raise ValueError(f"Receptivity to agent {other_agent} must be in range [-1, 1].")
                if not (-1 <= params['power'] <= 1):
                    raise ValueError(f"Power to agent {other_agent} must be in range [-1, 1].")
                if not (-1 <= params['connection'] <= 1):
                    raise ValueError(f"Connection to agent {other_agent} must be in range [-1, 1].")
                Agent.add_relationship(self, other_agent, params['distance'], params['power'], params['receptivity'], params['connection'])

        # Validate process_matrix
        allowed_process_keys = ['instinct', 'utility', 'skill', 'intention', 'outcome', 'effort', 'pleasure']
        self.process_matrix = {}
        if process_matrix:
            if not isinstance(process_matrix, dict):
                raise TypeError("Process_matrix must be a dictionary.")
            for key, value in process_matrix.items():
                if key not in allowed_process_keys:
                    raise ValueError(f"Invalid process_matrix key: {key}. Allowed keys are {allowed_process_keys}.")
                self.process_matrix[key] = value
        else:
            self.process_matrix = {key: 1 for key in allowed_process_keys}

        Agent._debug_log.append(f"Registered: {self.id} ({name})")
        Agent._registry[self.id] = self  # Register the agent

    def _validate_behavior_params(self, behavior, setup, params):
        """Validate behavior parameters ranges."""
        ranges = {
            'instinct': (-1, 1),
            'utility': (-1, 1),
            'enjoyment': (-1, 1),
            'alpha_instinct_plus': (0, 1),  # Habit strengthening rate
            'alpha_instinct_minus': (0, 1),  # Habit decay rate
            'alpha_utility': (0, 1),
            'alpha_enjoyment': (0, 1),
            'w_enjoyment': (0, 1),  # Weight for enjoyment in drift rate
            'w_utility': (0, 1),    # Weight for utility in drift rate
            'bias_scaling_factor': (0, 10),  # Scaling factor for instinct's influence on starting bias
            'exposure_count': (0, float('inf'))  # Non-negative value
        }
        for param, (min_val, max_val) in ranges.items():
            if param not in params:
                raise ValueError(f"Missing parameter '{param}' for behavior {behavior}, setup {setup}")
            if not (min_val <= params[param] <= max_val):
                raise ValueError(
                    f"{param} for behavior {behavior}, setup {setup} "
                    f"must be in range [{min_val}, {max_val}], got {params[param]}."
                )

    def add_setup(self, setup):
        """
        Add a new setup to the setups memory or update an existing setup's affinity.

        Parameters:
        - setup (Setup): a setup object.
        """
        # Validate affinity range
        if not isinstance (setup, Setup):
            raise TypeError("Setup must be an instance of Setup.")

        # Add or update the setup in the setup matrix
        self.setups.append(setup)  # Update existing setup

    def add_behavior(self, behavior_id, setup_id, instinct, utility, enjoyment,
                     alpha_instinct_plus=None, alpha_instinct_minus=None, alpha_utility=None, alpha_enjoyment=None,
                     w_enjoyment=None, w_utility=None, bias_scaling_factor=None, exposure_count=0, affinity=0):
        """
        Add/update a behavior for a specific setup with all parameters.
        Hyperparameters can be set per call or defaulted from manager.

        Parameters:
        - behavior_id (str): The ID of the behavior.
        - setup_id (str): The ID of the setup.
        - instinct (float): Habitual tendency (-1 to 1)
        - utility (float): Predicted outcome value (-1 to 1)
        - enjoyment (float): Predicted enjoyment (-1 to 1)
        - alpha_instinct_plus (float): Learning rate for strengthening instinct (0 to 1)
        - alpha_instinct_minus (float): Learning rate for weakening instinct (0 to 1)
        - alpha_utility (float): Learning rate for utility (0 to 1)
        - alpha_enjoyment (float): Learning rate for enjoyment (0 to 1)
        - w_enjoyment (float): Weight for enjoyment in drift rate (0 to 1)
        - w_utility (float): Weight for utility in drift rate (0 to 1)
        - bias_scaling_factor (float): Scaling factor for instinct's influence on starting bias (0 to 10)
        - exposure_count (int): Number of prior exposures to this behavior (default 0)
        - affinity (float): Setup affinity (0 to 1)
        """
        behavior_obj = Behavior.get(behavior_id)
        setup_obj = Setup.get(setup_id)

        # Use manager defaults if not provided
        alpha_instinct_plus = alpha_instinct_plus if alpha_instinct_plus is not None else hyperparam_manager.get('alpha_instinct_plus')
        alpha_instinct_minus = alpha_instinct_minus if alpha_instinct_minus is not None else hyperparam_manager.get('alpha_instinct_minus')
        alpha_utility = alpha_utility if alpha_utility is not None else hyperparam_manager.get('alpha_utility')
        alpha_enjoyment = alpha_enjoyment if alpha_enjoyment is not None else hyperparam_manager.get('alpha_enjoyment')
        w_enjoyment = w_enjoyment if w_enjoyment is not None else hyperparam_manager.get('w_enjoyment')
        w_utility = w_utility if w_utility is not None else hyperparam_manager.get('w_utility')
        bias_scaling_factor = bias_scaling_factor if bias_scaling_factor is not None else hyperparam_manager.get('bias_scaling_factor')

        # Validate input ranges
        params = {
            "instinct": instinct,
            "utility": utility,
            "enjoyment": enjoyment,
            "alpha_instinct_plus": alpha_instinct_plus,
            "alpha_instinct_minus": alpha_instinct_minus,
            "alpha_utility": alpha_utility,
            "alpha_enjoyment": alpha_enjoyment,
            "w_enjoyment": w_enjoyment,
            "w_utility": w_utility,
            "bias_scaling_factor": bias_scaling_factor,
            "exposure_count": exposure_count
        }

        for key, value in params.items():
            if key in ["instinct", "utility", "enjoyment"] and not (-1 <= value <= 1):
                raise ValueError(f"{key} must be in [-1, 1], got {value}.")
            elif key in ["alpha_instinct_plus", "alpha_instinct_minus", "alpha_utility", "alpha_enjoyment", "w_enjoyment", "w_utility"] and not (0 <= value <= 1):
                raise ValueError(f"{key} must be in [0, 1], got {value}.")
            elif key == "bias_scaling_factor" and not (0 <= value <= 10):
                raise ValueError(f"{key} must be in [0, 10], got {value}.")
            elif key == "exposure_count" and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{key} must be a non-negative number, got {value}.")

        # Add setup if missing
        if setup_obj not in self.setups:
            self.add_setup(setup_obj, affinity)

        # Add/update behavior
        if behavior_obj not in self.behaviors:
            self.behaviors[behavior_obj] = {}
        self.behaviors[behavior_obj][setup_obj] = params

    def form_intention(self, setup, behaviors=None):
        """
        Compute choice probabilities for each behavior in a setup using the tripartite model:
        - Habitual propensity (instinct)
        - Evaluative strength (enjoyment, utility)
        - Softmax over choice values (with tau and noise)
        Returns: dict mapping behavior to probability
        """
        filtered = self._filter_behaviors(setup, behaviors)
        tau = getattr(self, 'tau', 2.0)  # inverse temperature
        noise_scale = getattr(self, 'noise_scale', 0.2)  # stochasticity
        w_I = getattr(self, 'w_instinct', 1.0)
        choice_values = {}
        for behavior in filtered:
            params = self.behaviors[behavior][setup]
            instinct = params['instinct']
            enjoyment = params['enjoyment']
            utility = params['utility']
            w_E = params.get('w_enjoyment', 0.5)
            w_U = params.get('w_utility', 0.5)
            # Normalize weights
            total_weight = w_E + w_U
            if total_weight > 0:
                w_E = w_E / total_weight
                w_U = w_U / total_weight
            # Habitual propensity
            H = instinct * w_I
            # Evaluative strength
            E = enjoyment * w_E + utility * w_U
            # Noise
            noise = np.random.normal(0, noise_scale)
            CV = H + E + noise
            choice_values[behavior] = CV
        # Softmax
        exp_vals = np.array([np.exp(tau * choice_values[b]) for b in choice_values])
        probs = exp_vals / np.sum(exp_vals)
        return {b: p for b, p in zip(choice_values.keys(), probs)}


    def form_evaluation(self, intention, outcome, effort, pleasure):
        """
        Calculate an evaluation score based on weighted behavioral and outcome factors.

        This method computes an evaluation score by weighting four key factors (intention,
        outcome, effort, pleasure) according to the agent's process_matrix values. When
        the agent is a Group, the evaluation is further adjusted by the group's homogeneity
        and size characteristics.

        Parameters
        ----------
        intention : float
            The intentionality factor, typically ranging from -1 (negative) to 1 (positive).
        outcome : float
            The outcome quality factor, typically ranging from -1 (negative) to 1 (positive).
        effort : float
            The effort investment factor, typically ranging from -1 (negative) to 1 (positive).
        pleasure : float
            The pleasure/satisfaction factor, typically ranging from -1 (negative) to 1 (positive).

        Returns
        -------
        float
            The calculated evaluation score, clipped to range from -1 to 1.

        Notes
        -----
        - For Group instances, the evaluation is dampened by the group's homogeneity factor
        - For Group instances, the evaluation is also scaled down logarithmically based on group size,
          reflecting that larger groups tend to be less sensitive to individual factors
        """
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

    def update_instinct(self, behavior, setup, performed=True, observer_penalty=0.5, alternative_behavior=False):
        """Update the instinct parameter for a behavior in a given setup.

        Parameters:
        ----------
        behavior : Behavior
            The behavior to update.
        setup : Setup
            The context in which the behavior occurs.
        performed : bool, default=True
            Whether the agent performed the behavior or observed it.
        observer_penalty : float, default=0.5
            Learning penalty when observing rather than performing (0 to 1).
        alternative_behavior : bool, default=False
            If True, this indicates an alternative behavior was performed instead,
            which should weaken the instinct for this behavior.
        """
        params = self.behaviors[behavior][setup]

        # Get appropriate learning rates
        if alternative_behavior:
            alpha = params.get('alpha_instinct_minus', 0.1)  # Default if not specified
            target = -1  # Target value for weakening instinct
        else:
            alpha = params.get('alpha_instinct_plus', 0.1)   # Default if not specified
            target = 1   # Target value for strengthening instinct

        # Group-specific learning modulation
        if isinstance(self, Group):
            size_factor = 1 / np.sqrt(self.size)
            alpha *= self.homogeneity * size_factor

        if performed:
            # Direct experience learning
            # Δinstinct_{B,S} = α_{+I} · (1 - instinct_{B,S}) for performed behavior
            # Δinstinct_{B,S} = α_{-I} · (-1 - instinct_{B,S}) for alternative behavior
            delta_instinct = alpha * (target - params['instinct'])
            params['instinct'] += delta_instinct
        else:
            # Observational learning with penalty
            # Reduce learning rate by observer penalty factor
            delta_instinct = (1 - observer_penalty) * alpha * (target - params['instinct'])
            params['instinct'] += delta_instinct

        # Ensure instinct stays within valid range
        params['instinct'] = np.clip(params['instinct'], -1, 1)

    def update_utility(self, behavior, setup, perceived_utility, performed=True, observer_penalty=0.5):
        """Update the utility parameter for a behavior based on outcome prediction error.

        Parameters:
        ----------
        behavior : Behavior
            The behavior to update.
        setup : Setup
            The context in which the behavior occurs.
        perceived_utility : float
            The perceived utility value of the behavior (-1 to 1).
        performed : bool, default=True
            Whether the agent performed the behavior or observed it.
        observer_penalty : float, default=0.5
            Learning penalty when observing rather than performing (0 to 1).
        """
        params = self.behaviors[behavior][setup]
        alpha_utility = params['alpha_utility']

        # Group-specific learning modulation
        if isinstance(self, Group):
            size_factor = 1 / np.sqrt(self.size)
            alpha_utility *= self.homogeneity * size_factor

        # Prediction error based learning
        # Δutility_{B,S} = α_U · (U - utility_{B,S})
        prediction_error = perceived_utility - params['utility']

        if performed:
            # Direct experience learning
            params['utility'] += alpha_utility * prediction_error
        else:
            # Observational learning with penalty
            params['utility'] += (1 - observer_penalty) * alpha_utility * prediction_error

        params['utility'] = np.clip(params['utility'], -1, 1)

    def update_enjoyment(self, behavior, setup, perceived_enjoyment, performed=True, observer_penalty=0.5):
        """Update the enjoyment parameter for a behavior based on experienced pleasure.

        Parameters:
        ----------
        behavior : Behavior
            The behavior to update.
        setup : Setup
            The context in which the behavior occurs.
        perceived_enjoyment : float
            The perceived enjoyment of the behavior (-1 to 1).
        performed : bool, default=True
            Whether the agent performed the behavior or observed it.
        observer_penalty : float, default=0.5
            Learning penalty when observing rather than performing (0 to 1).
        """
        if perceived_enjoyment is None:
            return  # Skip update if no enjoyment data

        params = self.behaviors[behavior][setup]
        alpha_enjoyment = params['alpha_enjoyment']

        # Group-specific learning modulation
        if isinstance(self, Group):
            size_factor = 1 / np.sqrt(self.size)
            alpha_enjoyment *= self.homogeneity * size_factor

        # Initialize exposure count if needed
        if 'exposure_count' not in params:
            params['exposure_count'] = 0

        # Increment exposure count if performed
        if performed:
            params['exposure_count'] += 1

        # Calculate prediction error
        # Δenjoyment_{B,S} = α_E · (E - enjoyment_{B,S})
        prediction_error = perceived_enjoyment - params['enjoyment']

        if performed:
            # Direct experience learning
            params['enjoyment'] += alpha_enjoyment * prediction_error
        else:
            # Observational learning with penalty
            params['enjoyment'] += (1 - observer_penalty) * alpha_enjoyment * prediction_error

        params['enjoyment'] = np.clip(params['enjoyment'], -1, 1)

    def update_skill(self, behavior, setup, skill_level, performed=True, observer_penalty=0.5):
        """Update the skill parameter for a behavior based on practice and observation.

        Parameters:
        ----------
        behavior : Behavior
            The behavior to update.
        setup : Setup
            The context in which the behavior occurs.
        skill_level : float
            The demonstrated skill level for the behavior (0 to 1).
        performed : bool, default=True
            Whether the agent performed the behavior or observed it.
        observer_penalty : float, default=0.5
            Learning penalty when observing rather than performing.
        """
        params = self.behaviors[behavior][setup]
        alpha_skill = params['alpha_skill']
        current_skill = params['skill']

        # Apply power law of practice - diminishing returns at higher skill levels
        skill_modifier = (1 - current_skill)**1.5

        # Calculate challenge factor
        difficulty = behavior.difficulty
        challenge_factor = max(0, min(1.5, (difficulty / (current_skill + 0.1))))

        # Group-specific learning modulation
        if isinstance(self, Group):
            size_factor = 1 / np.sqrt(self.size)
            alpha_skill *= self.homogeneity * size_factor

        if performed:
            # Direct skill learning through performance
            params['skill'] += alpha_skill * skill_modifier * challenge_factor * (skill_level - params['skill'])

            # Apply "use it or lose it" decay to other behaviors
            for other_behavior, setups_dict in self.behaviors.items():
                if other_behavior == behavior:
                    continue

                if setup in setups_dict:
                    other_params = setups_dict[setup]
                    other_exposure = other_params.get('exposure_count', 0)
                    decay_factor = 0.01 / (1 + 0.001 * other_exposure)

                    if other_params['skill'] > 0:
                        other_params['skill'] -= decay_factor * other_params['skill']
                    other_params['skill'] = np.clip(other_params['skill'], 0, 1)
        else:
            # Calculate expertise gap between observer and model
            expertise_gap = max(0, skill_level - current_skill)
            # Learning is enhanced when observing someone slightly better than you
            observation_factor = min(1.5, 1 + expertise_gap)

            # Update skill with observation factors
            params['skill'] += observer_penalty * alpha_skill * skill_modifier * observation_factor * (skill_level - params['skill'])

        # Cultural resource boost - USE DEFENSIVE LOOKUP
        for culture_id, rel_params in self.relationships.items():
            try:
                culture = Agent.get_agent(culture_id)
                if not isinstance(culture, Culture):
                    continue

                res = culture.get_cultural_params(behavior, setup)['resource']
                cultural_boost = rel_params['receptivity'] * res * (1 - rel_params['distance'])
                params['skill'] += cultural_boost * (1 - current_skill)  # Diminishing returns
            except KeyError:
                # Agent not found in registry - log and continue
                print(f"Warning: Agent {culture_id} not found during update_skill")
                continue

        params['skill'] = np.clip(params['skill'], 0, 1)

    def update_behavior(self, behavior, setup, performed=True, perceived_utility=None, perceived_enjoyment=None,
                       observer_penalty=0.5, alternative_behavior=False, observed_agent=None):
        """Update all behavior parameters after performing or observing a behavior.

        Parameters:
        ----------
        behavior : Behavior
            The behavior to update.
        setup : Setup
            The context in which the behavior occurs.
        performed : bool, default=True
            Whether the agent performed the behavior or observed it.
        perceived_utility : float, optional
            The perceived utility of the behavior (-1 to 1).
        perceived_enjoyment : float, optional
            The perceived enjoyment of the behavior (-1 to 1).
        observer_penalty : float, default=0.5
            Learning penalty when observing rather than performing (0 to 1).
        alternative_behavior : bool, default=False
            If True, this indicates an alternative behavior was performed instead,
            which should weaken the instinct for this behavior.
        observed_agent : Agent, optional
            The agent that was observed (for observational learning).
        """
        # Update instinct based on whether behavior was performed or an alternative was chosen
        self.update_instinct(behavior, setup, performed, observer_penalty, alternative_behavior)

        # Update utility if perceived utility is provided
        if perceived_utility is not None:
            # Apply social influence if this is observational learning and we have an observed agent
            if not performed and observed_agent is not None:
                # Get relationship parameters
                rel_params = self.relationships.get(observed_agent.id, {})
                receptivity = rel_params.get('receptivity', 0)

                # Adjust perceived utility based on relationship
                # Incorporate the observer's receptivity to the actor
                perceived_utility = perceived_utility * receptivity

            self.update_utility(behavior, setup, perceived_utility, performed, observer_penalty)

        # Update enjoyment if perceived enjoyment is provided
        if perceived_enjoyment is not None:
            # Apply social influence if this is observational learning and we have an observed agent
            if not performed and observed_agent is not None:
                # Get relationship parameters
                rel_params = self.relationships.get(observed_agent.id, {})
                receptivity = rel_params.get('receptivity', 0)

                # Adjust perceived enjoyment based on relationship
                # Incorporate the observer's receptivity to the actor
                perceived_enjoyment = perceived_enjoyment * receptivity

            self.update_enjoyment(behavior, setup, perceived_enjoyment, performed, observer_penalty)

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

    def to_dict(self):
        """Convert agent to serializable dictionary."""
        data = {
            'id': self.id,
            'name': self.name,
            'type': self.__class__.__name__,
            'setups': [setup.name for setup in self.setups],
            'behaviors': {},
            'relationships': {},
            'process_matrix': self.process_matrix,
        }

        # Serialize behaviors
        for behavior, setup_dict in self.behaviors.items():
            behavior_name = behavior.name
            data['behaviors'][behavior_name] = {}
            for setup, params in setup_dict.items():
                setup_name = setup.name
                data['behaviors'][behavior_name][setup_name] = params.copy()

        # Serialize relationships (just IDs and parameters)
        for agent_id, params in self.relationships.items():
            data['relationships'][agent_id] = params.copy()

        # Add subclass-specific data
        if isinstance(self, Group):
            data['size'] = self.size
            data['homogeneity'] = self.homogeneity

        return data

    @classmethod
    def from_dict(cls, data, setup_registry, behavior_registry):
        """Recreate agent from dictionary."""
        # Create appropriate agent subclass
        agent_type = data['type']
        agent_class = globals()[agent_type]

        # Add mandatory parameters
        kwargs = {
            'name': data['name'],
            'process_matrix': data['process_matrix'],
        }

        # Add setups
        setups = [setup_registry[name] for name in data['setups']]
        kwargs['setups'] = setups

        # Add behaviors
        behaviors = {}
        for behavior_name, setup_dict in data['behaviors'].items():
            behavior = behavior_registry[behavior_name]
            behaviors[behavior] = {}
            for setup_name, params in setup_dict.items():
                setup = setup_registry[setup_name]
                behaviors[behavior][setup] = params.copy()
        kwargs['behaviors'] = behaviors

        # Add subclass-specific parameters
        if agent_type == 'Group':
            kwargs['size'] = data['size']
            kwargs['homogeneity'] = data['homogeneity']

        # Create agent but don't add relationships yet
        agent = agent_class(**kwargs)
        agent.id = data['id']  # Preserve original ID

        # Store relationships for later processing
        agent._pending_relationships = data['relationships']

        return agent

    def tune_hyperparameters(self, train_data, eval_fn, method='grid', param_grid=None, k=5, n_iter=20):
        """
        Tune hyperparameters using grid or random search with k-fold cross-validation.
        eval_fn(agent, train_folds, val_fold) -> score
        """
        # Import here to avoid circular import
        from hyperparameter_tuning import HyperparameterTuner
        if param_grid is None:
            param_grid = {
                'alpha_instinct_plus': [0.05, 0.1, 0.2],
                'alpha_instinct_minus': [0.05, 0.1, 0.2],
                'alpha_utility': [0.05, 0.1, 0.2],
                'alpha_enjoyment': [0.05, 0.1, 0.2],
                'w_enjoyment': [0.3, 0.5, 0.7],
                'w_utility': [0.3, 0.5, 0.7],
                'bias_scaling_factor': [0.5, 1.0, 2.0]
            }
        tuner = HyperparameterTuner(self.__class__, param_grid, k, hyperparam_manager=hyperparam_manager)
        if method == 'grid':
            best_params, best_score = tuner.grid_search(train_data, eval_fn)
        else:
            best_params, best_score = tuner.random_search(train_data, eval_fn, n_iter)
        hyperparam_manager.load_from_dict(best_params)
        return best_params, best_score

class Individual(Agent):
    def __init__(self, name="", setups=None, behaviors=None, process_matrix=None, relationships=None):
        # Change to explicitly use keyword arguments
        super().__init__(
            name=name,
            setups=setups,
            behaviors=behaviors,
            process_matrix=process_matrix,
            relationships=relationships
        )
        # Any Individual-specific initialization here

class Group(Agent):
    def __init__(self, name="", setups=None, behaviors=None, process_matrix=None,
                 relationships=None, size=1, homogeneity=1.0):
        # Change to explicitly use keyword arguments
        super().__init__(
            name=name,
            setups=setups,
            behaviors=behaviors,
            process_matrix=process_matrix,
            relationships=relationships
        )
        self.size = size
        self.homogeneity = homogeneity

class Culture(Agent):
    def __init__(self, cultural_norms, relationships=None, name=""):
        """
        Static cultural framework that influences agents through norms/merit/resources.
        Inherits from Agent but with frozen parameters.
        """
        super().__init__(
            setups=[],  # Cultures don't have setups
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
