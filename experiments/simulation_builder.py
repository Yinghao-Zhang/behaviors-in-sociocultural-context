import numpy as np
import matplotlib.pyplot as plt
from uuid import uuid4
from situation import Situation
from agent import Agent, Individual, Group, Culture
from behavior import Behavior
from setup import Setup, SetupManager
import json

class SimulationBuilder:
    def __init__(self):
        # Change structure to include agent dimension
        self.initial_parameters = {}  # {agent_id: {behavior_name: {setup_name: {param: value}}}}

        # Registry dictionaries with name->id mapping
        self.setup_ids = {}      # {name: setup_id}
        self.behavior_ids = {}   # {name: behavior_id}
        self.agent_ids = {}      # {name: agent_id}
        self.culture_ids = {}    # {name: culture_id}

        # Central agent tracking
        self.central_agent_id = None
        self.central_agent_name = None

        # Setup manager for spatial organization
        self.setup_manager = SetupManager()

        # Store agent affinities by ID
        self.agent_setup_affinities = {}  # {agent_id: {setup_id: affinity}}

        self.simulation_params = {}
        self.metrics = {
            'behavior_params': [],
            'situation_metadata': []
        }

    # Phase 1: Setup Configuration
    def configure_setups(self):
        print("\n=== SETUP CONFIGURATION ===")
        num_setups = self._get_input("Number of setups to model", int, min_val=1)

        for _ in range(num_setups):
            while True:
                name = input("Setup name: ").strip()
                if name in self.setup_ids:
                    print(f"Setup '{name}' already exists!")
                    continue
                break

            setup = Setup(name)
            # Store ID mapping
            self.setup_ids[name] = setup.id

            # Register with setup manager
            if len(self.setup_ids) == 1:
                self.setup_manager.add_setup(setup)
            else:
                print("Define distance to an existing setup:")
                reference_options = {str(i+1): s for i, s in enumerate(self.setup_ids.keys()) if s != name}
                reference_name = self._get_choice("Reference setup:", reference_options)
                reference_id = self.setup_ids[reference_options[reference_name]]
                reference = Setup.get(reference_id)
                distance = self._get_input("Distance (0-1)", float, 0, 1)
                self.setup_manager.add_setup(setup, reference_distance=distance, reference_setup=reference)

            print(f"Created setup: {name}")

    # Phase 2: Behavior Configuration
    def configure_behaviors(self):
        print("\n=== BEHAVIOR CONFIGURATION ===")
        num_behaviors = self._get_input("Number of behaviors to model", int, min_val=2)

        for _ in range(num_behaviors):
            while True:
                name = input("Behavior name: ").strip()
                if name in self.behavior_ids:
                    print(f"Behavior '{name}' already exists!")
                    continue
                break

            difficulty = self._get_input("Difficulty (0-1)", float, 0, 1)
            base_outcome = self._get_input("Base outcome (-1-1)", float, -1, 1)
            volatility = self._get_input("Outcome volatility (0-1)", float, 0, 1)

            # Configure setup modifiers using IDs
            setup_modifiers = {}
            for setup_name, setup_id in self.setup_ids.items():
                if self._get_choice(
                    f"Add modifiers for {setup_name}?",
                    {'y': 'Yes', 'n': 'No'}
                ) == 'y':
                    base_mod = self._get_input("Base outcome modifier (-1-1)", float, -1, 1)
                    difficulty_mod = self._get_input("Difficulty modifier (-1-1)", float, -1, 1)
                    # Store by setup_id
                    setup_modifiers[setup_id] = {
                        'base_outcome_mod': base_mod,
                        'difficulty_mod': difficulty_mod
                    }

            behavior = Behavior(
                name=name,
                difficulty=difficulty,
                base_outcome=base_outcome,
                outcome_volatility=volatility,
                setup_modifiers=setup_modifiers
            )
            # Store ID mapping
            self.behavior_ids[name] = behavior.id
            print(f"Created behavior: {name}")

    # Phase 3: Agent Configuration
    def configure_agents(self):
        print("\n=== ADDITIONAL AGENT CONFIGURATION ===")
        agent_types = {
            '1': ('Individual', Individual),
            '2': ('Group', Group),
            '3': ('Culture', Culture)
        }

        num_agents = self._get_input("Number of additional agents to create", int, 1)

        for agent_num in range(1, num_agents + 1):
            print(f"\n=== Configuring Agent {agent_num}/{num_agents} ===")
            agent_type = self._get_choice("Agent type:", agent_types)
            cls_name, cls = agent_types[agent_type]

            while True:
                name = input(f"{cls_name} name: ").strip()
                if name in self.agent_ids or name in self.culture_ids:
                    print(f"Name '{name}' already exists!")
                    continue
                break

            # Configure setup affinities by name (user-friendly) but store by ID
            print(f"\n=== SETUP AFFINITIES FOR {name} ===")
            self._configure_setup_affinities(name)

            # Create the agent with list of setup IDs
            setup_ids = list(self.setup_ids.values())
            agent = self._create_agent(cls, name, setup_ids)

            # Store ID mapping
            if isinstance(agent, Culture):
                self.culture_ids[name] = agent.id
            else:
                self.agent_ids[name] = agent.id

            # Transfer affinities from name to ID
            if name in self.agent_setup_affinities:
                self.agent_setup_affinities[agent.id] = self.agent_setup_affinities[name]

            print(f"Created {cls_name}: {name}")

        # Configure relationships after ALL agents exist
        print("\n=== CONFIGURING RELATIONSHIPS ===")

        # Start with central agent
        central_agent = Agent.get_agent(self.central_agent_id)
        print(f"\nConfiguring relationships for central agent {central_agent.name}")
        self._configure_relationships(central_agent)

        # Configure relationships for other agents
        for agent_name, agent_id in self.agent_ids.items():
            if agent_id != self.central_agent_id:  # Skip central agent as we already configured it
                agent = Agent.get_agent(agent_id)
                print(f"\nConfiguring relationships for {agent_name}")
                self._configure_relationships(agent)

    def _create_agent(self, agent_class, name, setup_ids):
        """Create agent using IDs for all references"""

        setups = [Setup.get(setup_id) for setup_id in setup_ids]

        # Create a temporary ID to use for configuring
        temp_id = str(uuid4())

        # Prepare behaviors dict with object references
        behaviors = {}
        behavior_params = self._configure_behavior_params(agent_id=temp_id)
        for behavior_id, setup_data in behavior_params.items():
            behavior = Behavior.get(behavior_id)
            behaviors[behavior] = {}
            for setup_id, params in setup_data.items():
                setup = Setup.get(setup_id)
                behaviors[behavior][setup] = params

        process_matrix = {} # Placeholder for process matrix; this may be modified later

        if agent_class == Culture:
            agent = Culture(
                name=name,
                cultural_norms=behaviors,
                relationships={}
            )
        elif agent_class == Group:
            size = self._get_input("Group size", int, 1)
            homogeneity = self._get_input("Group homogeneity (0-1)", float, 0, 1)
            agent = Group(
                name=name,
                setups=setups,
                behaviors=behaviors,
                process_matrix=process_matrix,
                relationships={},
                size=size,
                homogeneity=homogeneity
            )
        else:
            # For Individual
            agent = Individual(
                name=name,
                setups=setups,
                behaviors=behaviors,
                process_matrix=process_matrix,
                relationships={}
            )

        # After agent creation, update the ID reference
        if temp_id in self.initial_parameters:
            self.initial_parameters[agent.id] = self.initial_parameters.pop(temp_id)

        return agent

    def _configure_setup_affinities(self, agent_name):
        """Store affinities by agent name initially"""
        # Create entry for this agent's name (not ID yet)
        self.agent_setup_affinities[agent_name] = {}

        # Get affinity for each setup
        for setup_name, setup_id in self.setup_ids.items():
            affinity = self._get_input(
                f"Affinity for {setup_name} (0-1)",
                float, 0, 1
            )
            self.agent_setup_affinities[agent_name][setup_id] = affinity

    def _configure_behavior_params(self, agent_id=None):
        """Configure behavior parameters using IDs"""
        params = {}
        print("\n=== BEHAVIOR PARAMETERS ===")

        # For each behavior
        for behavior_name, behavior_id in self.behavior_ids.items():
            params[behavior_id] = {}

            # Initialize in our initial_parameters structure
            if agent_id not in self.initial_parameters:
                self.initial_parameters[agent_id] = {}

            if behavior_name not in self.initial_parameters[agent_id]:
                self.initial_parameters[agent_id][behavior_name] = {}

            # For each setup
            for setup_name, setup_id in self.setup_ids.items():
                print(f"\nParameters for {behavior_name} in {setup_name}:")

                # Get user inputs for tripartite model
                instinct = self._get_input("Instinct (-1-1)", float, -1, 1)
                utility = self._get_input("Utility (-1-1)", float, -1, 1)
                enjoyment = self._get_input("Enjoyment (-1-1)", float, -1, 1)
                from agent import hyperparam_manager
                alpha_instinct_plus = self._get_input("Instinct strengthening α (0-1)", float, 0, 1, default=hyperparam_manager.get('alpha_instinct_plus'))
                alpha_instinct_minus = self._get_input("Instinct weakening α (0-1)", float, 0, 1, default=hyperparam_manager.get('alpha_instinct_minus'))
                alpha_utility = self._get_input("Utility α (0-1)", float, 0, 1, default=hyperparam_manager.get('alpha_utility'))
                alpha_enjoyment = self._get_input("Enjoyment α (0-1)", float, 0, 1, default=hyperparam_manager.get('alpha_enjoyment'))
                w_enjoyment = self._get_input("Enjoyment weight (0-1)", float, 0, 1, default=hyperparam_manager.get('w_enjoyment'))
                w_utility = self._get_input("Utility weight (0-1)", float, 0, 1, default=hyperparam_manager.get('w_utility'))
                bias_scaling_factor = self._get_input("Bias scaling factor (0-10)", float, 0, 10, default=hyperparam_manager.get('bias_scaling_factor'))
                exposure_count = self._get_input("Exposure count", int, 0, default=0)

                # Store in params for agent creation
                params[behavior_id][setup_id] = {
                    'instinct': instinct,
                    'utility': utility,
                    'enjoyment': enjoyment,
                    'alpha_instinct_plus': alpha_instinct_plus,
                    'alpha_instinct_minus': alpha_instinct_minus,
                    'alpha_utility': alpha_utility,
                    'alpha_enjoyment': alpha_enjoyment,
                    'w_enjoyment': w_enjoyment,
                    'w_utility': w_utility,
                    'bias_scaling_factor': bias_scaling_factor,
                    'exposure_count': exposure_count
                }

                # Also store in initial_parameters for later reference
                if setup_name not in self.initial_parameters[agent_id][behavior_name]:
                    self.initial_parameters[agent_id][behavior_name][setup_name] = {}

                self.initial_parameters[agent_id][behavior_name][setup_name] = {
                    'instinct': instinct,
                    'utility': utility,
                    'enjoyment': enjoyment,
                    'exposure_count': exposure_count
                }

        return params

    # Phase 4: Simulation Parameters
    def configure_simulation(self):
        print("\n=== SIMULATION PARAMETERS ===")

        # Set up interaction modes
        interaction_modes = [
            "solitary", "co-participate", "observe", "observe_s", "suggest", "suggest_s",
            "observe_feedback", "observe_feedback_s", "suggest_feedback", "suggest_feedback_s"
        ]

        # Ask if user wants to customize interaction mode proportions
        customize_proportions = self._get_choice(
            "Customize interaction mode proportions?",
            {'1': 'Yes', '2': 'No (equal probability for all modes)'}
        ) == '1'

        # Initialize mode proportions dict
        mode_proportions = {}

        if customize_proportions:
            print("\n=== INTERACTION MODE PROPORTIONS ===")
            print("Enter the proportion (0-1) for each interaction mode you want to customize.")
            print("The remaining proportion will be distributed randomly among unspecified modes.")

            remaining_proportion = 1.0
            available_modes = interaction_modes.copy()

            while remaining_proportion > 0 and available_modes:
                # Display available modes
                print("\nAvailable interaction modes:")
                for i, mode in enumerate(available_modes, 1):
                    print(f"{i}. {mode}")
                print(f"Remaining proportion to allocate: {remaining_proportion:.2f}")
                print("0. Finish (distribute remaining proportion randomly)")

                choice = self._get_input("Select mode to customize", int, 0, len(available_modes))
                if choice == 0:
                    break

                selected_mode = available_modes[choice-1]
                max_prop = min(1.0, remaining_proportion)
                proportion = self._get_input(
                    f"Proportion for '{selected_mode}' (0-{max_prop:.2f})",
                    float, 0, max_prop
                )

                # Store the proportion
                mode_proportions[selected_mode] = proportion
                remaining_proportion -= proportion

                # Remove the mode from the available list
                available_modes.remove(selected_mode)

                if not available_modes:
                    print("All modes have been assigned proportions.")
                    break

            # If there's any remaining proportion, store it for random distribution
            if remaining_proportion > 0:
                print(f"\nRemaining proportion ({remaining_proportion:.2f}) will be distributed randomly among unspecified modes.")
                mode_proportions['_remaining'] = remaining_proportion
                mode_proportions['_unspecified'] = [mode for mode in interaction_modes if mode not in mode_proportions]

        self.simulation_params = {
            'num_situations': self._get_input("Number of situations", int, 1),
            'interaction_modes': interaction_modes,
            'mode_proportions': mode_proportions,
            'customize_proportions': customize_proportions,
            'log_level': self._get_choice(
                "Logging detail:",
                {'1': 'Minimal', '2': 'Detailed', '3': 'Debug'}
            ),
            'visualize': self._get_choice(
                "Generate visualizations?",
                {'1': 'Yes', '2': 'No'}
            ) == '1'
        }

    # Phase 5: Simulation Execution
    def run_simulation(self):
        print("\n=== RUNNING SIMULATION ===")

        if not self.central_agent_id:
            print("Error: Central agent not configured!")
            return

        # Get lists of IDs instead of objects
        non_central_agent_ids = [id for id in self.agent_ids.values() if id != self.central_agent_id]

        # Track parameter changes for visualization
        self.parameter_history = {
            'instinct': {},  # {behavior_name: {setup_name: [values]}}
            'utility': {},
            'enjoyment': {}
        }

        # Store initial parameters
        central_agent = Agent.get_agent(self.central_agent_id)
        for behavior, setups in central_agent.behaviors.items():
            for setup, params in setups.items():
                # Initialize parameter tracking for this behavior-setup pair
                for param in ['instinct', 'utility', 'enjoyment']:
                    if behavior.name not in self.parameter_history[param]:
                        self.parameter_history[param][behavior.name] = {}
                    if setup.name not in self.parameter_history[param][behavior.name]:
                        self.parameter_history[param][behavior.name][setup.name] = []

                    # Store initial value
                    self.parameter_history[param][behavior.name][setup.name].append(params[param])

        first_situation = True
        for i in range(self.simulation_params['num_situations']):
            try:
                # Central agent is always the individual
                individual_id = self.central_agent_id

                # Environment is randomly chosen from non-central, non-culture agents
                environment_candidates = non_central_agent_ids
                if not environment_candidates:
                    raise ValueError("No valid environment agents available.")
                environment_id = np.random.choice(environment_candidates)

                # Get agent objects
                individual = Agent.get_agent(individual_id)
                environment = Agent.get_agent(environment_id)

                # Get setup keys and determine probabilities based on agent affinities
                setup_ids = list(self.setup_ids.values())
                individual_affinities = self.agent_setup_affinities.get(individual.id, {})
                if not individual_affinities and individual.name in self.agent_setup_affinities:
                    individual_affinities = self.agent_setup_affinities[individual.name]
                    self.agent_setup_affinities[individual.id] = individual_affinities

                env_affinities = self.agent_setup_affinities.get(environment.id, {})
                if not env_affinities and environment.name in self.agent_setup_affinities:
                    env_affinities = self.agent_setup_affinities[environment.name]
                    self.agent_setup_affinities[environment.id] = env_affinities

                # Build probability distribution for setup selection
                if isinstance(environment, Culture):
                    # For cultures, just use individual's affinities
                    probabilities = [individual_affinities.get(setup_id, 0.1) for setup_id in setup_ids]
                else:
                    # Environment is a regular agent
                    probabilities = []
                    for setup_id in setup_ids:
                        ind_aff = individual_affinities.get(setup_id, 0.1)
                        env_aff = env_affinities.get(setup_id, 0.1)
                        combined_aff = (ind_aff + env_aff) / 2
                        probabilities.append(combined_aff)

                # Normalize probabilities
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p/total_prob for p in probabilities]
                else:
                    probabilities = [1.0/len(setup_ids) for _ in setup_ids]

                # Choose setup based on weighted probabilities
                setup_id = np.random.choice(setup_ids, p=probabilities)
                setup = Setup.get(setup_id)

                # Create and simulate situation
                situation = Situation(
                    setup_id=setup.id,
                    individual_id=individual_id,
                    environment_id=environment_id,
                    interaction_mode=self._select_interaction_mode()
                )
                self.current_situation = situation

                # Simulation step
                try:
                    situation._simulate_situation(seed=42+i)
                    # Store parameters AFTER successful simulation
                    if first_situation:
                        # Initialize history with current parameters
                        self._initialize_parameter_history(central_agent)
                        first_situation = False
                    else:
                        # Append current parameters
                        self._update_parameter_history(central_agent)
                except Exception as e:
                    print(f"Situation {i+1} simulation failed: {e.__class__.__name__}: {str(e)}")
                    continue

                # Logging step
                try:
                    self._log_situation(i, situation)
                except Exception as e:
                    print(f"Situation {i+1} logging failed: {e.__class__.__name__}: {str(e)}")
                    continue

                # Metrics step
                try:
                    self._collect_metrics()
                except Exception as e:
                    print(f"Situation {i+1} metrics failed: {e.__class__.__name__}: {str(e)}")

            except Exception as e:
                print(f"Situation {i+1} general failure: {e.__class__.__name__}: {str(e)}")

        if self.simulation_params['log_level'] == '3':
            print("\n=== AGENT REGISTRY DEBUG LOG ===")
            for entry in Agent._debug_log:
                print(entry)

        if self.simulation_params['visualize']:
            self._generate_visualizations()

    def _select_interaction_mode(self):
        if self.simulation_params.get('customize_proportions', False):
            # Get the mode proportions
            mode_proportions = self.simulation_params['mode_proportions']

            # Check if we need to randomly select from unspecified modes
            if '_remaining' in mode_proportions and mode_proportions['_remaining'] > 0:
                # Determine if we're using specified modes or random unspecified
                use_specified = np.random.random() < (1 - mode_proportions['_remaining'])

                if use_specified:
                    # Select from specified modes based on their proportions
                    specified_modes = [m for m in mode_proportions if not m.startswith('_')]
                    if specified_modes:
                        specified_probs = [mode_proportions[m] for m in specified_modes]
                        # Normalize probabilities
                        total_prob = sum(specified_probs)
                        specified_probs = [p/total_prob for p in specified_probs]
                        interaction_mode = np.random.choice(specified_modes, p=specified_probs)
                    else:
                        # Fallback to random if no modes were specified
                        interaction_mode = np.random.choice(self.simulation_params['interaction_modes'])
                else:
                    # Select randomly from unspecified modes
                    unspecified_modes = mode_proportions['_unspecified']
                    if unspecified_modes:
                        interaction_mode = np.random.choice(unspecified_modes)
                    else:
                        # If all modes were specified but we still have remaining probability
                        interaction_mode = np.random.choice(self.simulation_params['interaction_modes'])
            else:
                # Use only the specified modes with their exact proportions
                modes = list(mode_proportions.keys())
                probabilities = list(mode_proportions.values())
                # Normalize probabilities
                total_prob = sum(probabilities)
                probabilities = [p/total_prob for p in probabilities]
                interaction_mode = np.random.choice(modes, p=probabilities)
        else:
            # Default: equal probability for all modes
            interaction_mode = np.random.choice(self.simulation_params['interaction_modes'])
        return interaction_mode

    def _configure_relationships(self, agent):
        """Configure relationships using IDs"""
        print("\n=== RELATIONSHIPS ===")

        # Get all entities except the current agent
        available_ids = []
        already_related_ids = []

        # Collect available agents by ID
        for name, agent_id in list(self.agent_ids.items()) + list(self.culture_ids.items()):
            if agent_id != agent.id:
                if agent_id in agent.relationships:
                    already_related_ids.append(agent_id)
                else:
                    available_ids.append((name, agent_id))

        # First, display entities with existing relationships
        if already_related_ids:
            print("\nEntities with existing relationships:")
            for related_id in already_related_ids:
                related_agent = Agent.get_agent(related_id)
                rel = agent.relationships[related_id]
                print(f"- {related_agent.name}: Power={rel['power']:.2f}, Distance={rel['distance']:.2f}, "
                     f"Receptivity={rel['receptivity']:.2f}, Connection={rel.get('connection', 0):.2f}")

        # Then allow configuring new relationships
        while available_ids:
            print("\nAvailable entities:")
            for i, (name, _) in enumerate(available_ids, 1):
                entity = Agent.get_agent(available_ids[i-1][1])
                print(f"{i}. {name} ({type(entity).__name__})")
            print("0. Done configuring relationships")

            choice = self._get_input("Select entity to relate to", int, 0, len(available_ids))
            if choice == 0:
                break

            target_name, target_id = available_ids[choice-1]
            target = Agent.get_agent(target_id)
            print(f"\nConfiguring relationship with {target_name}:")

            # Cultural relationships have fixed power=0
            is_culture = isinstance(target, Culture)
            power = 0 if is_culture else self._get_input("Power (0-1)", float, 0, 1)
            distance = self._get_input("Distance (0-1)", float, 0, 1)
            receptivity = self._get_input("Receptivity (0-1)", float, 0, 1)
            connection = self._get_input("Connection (-1 to 1)", float, -1, 1)

            # Use Agent class method to create bidirectional relationship
            Agent.add_relationship(agent.id, target_id, distance, receptivity, power, connection)

            # Remove this entity from available entities
            available_ids.remove((target_name, target_id))

    # Visualization
    def _generate_visualizations(self):
        print("\nGenerating visualizations...")
        central_agent = Agent.get_agent(self.central_agent_id)

        # Let user select which parameters to visualize
        selected_params = self._select_visualization_parameters()

        # Create one plot per behavior-setup combination
        for behavior in central_agent.behaviors:
            for setup in central_agent.behaviors[behavior]:
                plt.figure(figsize=(14, 8))

                # Get data including initial parameters
                num_situations = len(self.metrics['situation_metadata'])

                # Create labels - add "Initial" for the first point
                x_labels = ["Initial\nParameters"] + [self._make_label(i) for i in range(num_situations)]

                # Get the parameter history and separate the true initial parameters
                params = {}
                for p in selected_params:  # Only process selected parameters
                    # Get true initial value from our stored initial parameters
                    initial_value = self.initial_parameters[central_agent.id][behavior.name][setup.name][p]

                    # Get the situation values (all values in history)
                    situation_values = self.parameter_history[p][behavior.name][setup.name][:num_situations]

                    # Combine into full parameter history with correct initial values
                    params[p] = [initial_value] + situation_values

                # Get selected behaviors for color coding - add None for initial point
                selected_behaviors = [None] + [meta['selected_behavior'] for meta in self.metrics['situation_metadata']]

                # Create a color map for unique behaviors
                unique_behaviors = list(set(filter(None, selected_behaviors)))
                color_map = plt.cm.get_cmap('tab10', max(10, len(unique_behaviors)))
                behavior_colors = {b: color_map(i) for i, b in enumerate(unique_behaviors)}

                # Plot parameters with color-coded dots
                for param, values in params.items():
                    # Plot line
                    plt.plot(x_labels, values, linestyle='-', label=param.capitalize())

                    # Plot individual points with color coding
                    for i, (x, y) in enumerate(zip(x_labels, values)):
                        selected_behavior = selected_behaviors[i]

                        # Special formatting for initial point
                        if i == 0:
                            plt.plot(x, y, 'o', markersize=10, color='black', markeredgecolor='white')
                        elif selected_behavior:
                            # Use behavior-specific color for the dot
                            color = behavior_colors.get(selected_behavior, 'gray')
                            plt.plot(x, y, 'o', markersize=8, color=color)
                        else:
                            # Use gray for situations with no behavior
                            plt.plot(x, y, 'o', markersize=8, color='gray')

                # Add behavior color legend
                handles, labels = plt.gca().get_legend_handles_labels()

                # Add initial parameters to legend
                handles.append(plt.Line2D([], [], marker='o', markersize=10, color='black',
                                         markeredgecolor='white', linestyle='None'))
                labels.append('Initial Parameters')

                # Add selected behaviors to legend
                for behavior_name, color in behavior_colors.items():
                    handles.append(plt.Line2D([], [], marker='o', markersize=8, color=color, linestyle='None'))
                    labels.append(f'Selected: {behavior_name}')

                plt.title(f"Changes in {central_agent.name}'s {behavior.name} Parameters in {setup.name}")
                plt.xlabel("Situation Context")
                plt.ylabel("Parameter Value")
                plt.xticks(rotation=45, ha='right')
                plt.legend(handles, labels)
                plt.grid(True)
                plt.ylim(-1, 1)
                plt.tight_layout()
                plt.show()

    def _select_visualization_parameters(self):
        """Let users select which parameters to visualize"""
        print("\n=== VISUALIZATION PARAMETERS ===")
        print("Select which parameters to include in the visualization:")

        all_params = {
            '1': 'instinct',
            '2': 'utility',
            '3': 'enjoyment',
            '4': 'All'
        }

        for key, param in all_params.items():
            print(f"{key}. {param.capitalize()}")

        selected_params = []
        while len(selected_params) == 0:
            choices = input("Enter numbers (comma separated): ").strip()
            if '4' in choices.split(','):
                selected_params = ['instinct', 'utility', 'enjoyment']
                break

            for choice in choices.split(','):
                choice = choice.strip()
                if choice in all_params and choice != '4':
                    selected_params.append(all_params[choice])

            if len(selected_params) == 0:
                print("Please select at least one parameter.")

        return selected_params

    # Input Validation Utilities
    def _get_input(self, prompt, dtype, min_val=None, max_val=None, default=None):
        while True:
            try:
                raw = input(f"{prompt}{f' [default: {default}]' if default is not None else ''}: ").strip()
                if raw == '' and default is not None:
                    return default
                val = dtype(raw)
                if min_val is not None and val < min_val:
                    raise ValueError(f"Must be ≥ {min_val}")
                if max_val is not None and val > max_val:
                    raise ValueError(f"Must be ≤ {max_val}")
                return val
            except ValueError as e:
                print(f"Invalid input: {e}")

    def _get_choice(self, prompt, options):
        print(prompt)
        for k, v in options.items():
            print(f"{k}. {v[0]}" if isinstance(v, tuple) else f"{k}. {v}")
        while True:
            choice = input("Select option: ").strip()
            if choice in options:
                return choice
            print("Invalid choice")

    def _get_list(self, dtype, expected_length=None):
        while True:
            try:
                raw = input(">> ").strip()
                values = [dtype(x) for x in raw.split(",")]
                if expected_length and len(values) != expected_length:
                    raise ValueError(f"Need exactly {expected_length} values")
                return values
            except ValueError as e:
                print(f"Invalid input: {e}. Try again.")

    # Logging and Metrics
    def _log_situation(self, idx, situation):
        # Add metadata collection here instead of _collect_metrics
        meta = {
            'interaction_mode': situation.interaction_mode,
            'environment': Agent.get_agent(situation.environment_id).name,
            'selected_behavior': situation.selected_behavior.name if situation.selected_behavior else None
        }
        self.metrics['situation_metadata'].append(meta)

        if self.simulation_params['log_level'] in ['2', '3']:
            print(f"\nSituation {idx+1} Results:")
            print(f" - Setup: {situation.setup.name}")
            print(f" - Mode: {situation.interaction_mode}")
            print(f" - Individual: {Agent.get_agent(situation.individual_id).name}")
            print(f" - Environment: {Agent.get_agent(situation.environment_id).name}")
            if hasattr(situation, 'selected_behavior') and situation.selected_behavior:
                print(f" - Selected Behavior: {situation.selected_behavior.name}")
            else:
                print(" - Selected Behavior: None")
            if hasattr(situation, 'outcome'):
                if isinstance(situation.outcome[0], (float, int)):
                    print(f" - Outcome: {situation.outcome[0]:.2f}")
                else:
                    print(f" - Outcome: {situation.outcome[0]:.2f}")
            else:
                print(" - Outcome: Not calculated")

    def _collect_metrics(self):
        # Collect behavior parameters for all agents
        for agent_name, agent_id in self.agent_ids.items():
            agent = Agent.get_agent(agent_id)
            for behavior, setups in agent.behaviors.items():
                for setup, params in setups.items():
                    try:
                        self.metrics['behavior_params'].append({
                            'agent': agent_name,
                            'behavior': behavior.name,
                            'setup': setup.name,
                            'instinct': params['instinct'],
                            'utility': params['utility'],
                            'enjoyment': params['enjoyment'],
                            'exposure_count': params['exposure_count']
                        })
                    except (KeyError, AttributeError) as e:
                        print(f"Warning: Could not collect metrics for {agent_name}/{behavior.name} - {str(e)}")

    # Helper methods for entity access
    def get_setup(self, name_or_id):
        """Get setup by name or ID"""
        if name_or_id in self.setup_ids:
            setup_id = self.setup_ids[name_or_id]
        else:
            setup_id = name_or_id
        return Setup.get(setup_id)

    def get_behavior(self, name_or_id):
        """Get behavior by name or ID"""
        if name_or_id in self.behavior_ids:
            behavior_id = self.behavior_ids[name_or_id]
        else:
            behavior_id = name_or_id
        return Behavior.get(behavior_id)

    def get_agent(self, name_or_id):
        """Get agent by name or ID"""
        if name_or_id in self.agent_ids:
            agent_id = self.agent_ids[name_or_id]
        elif name_or_id in self.culture_ids:
            agent_id = self.culture_ids[name_or_id]
        else:
            agent_id = name_or_id
        return Agent.get_agent(agent_id)

    def save_state(self, filename):
        """Save entire simulation state to file using IDs"""
        # Collect all objects by ID
        setups = {name: Setup.get(id).to_dict() for name, id in self.setup_ids.items()}
        behaviors = {name: Behavior.get(id).to_dict() for name, id in self.behavior_ids.items()}
        agents = {name: Agent.get_agent(id).to_dict() for name, id in self.agent_ids.items()}
        cultures = {name: Agent.get_agent(id).to_dict() for name, id in self.culture_ids.items()}

        data = {
            'setup_ids': self.setup_ids,
            'behavior_ids': self.behavior_ids,
            'agent_ids': self.agent_ids,
            'culture_ids': self.culture_ids,
            'setups': setups,
            'behaviors': behaviors,
            'agents': agents,
            'cultures': cultures,
            'agent_setup_affinities': self.agent_setup_affinities,
            'simulation_params': self.simulation_params,
            'metrics': self.metrics,
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Simulation state saved to {filename}")

    def configure_central_agent(self):
        """Configure the central agent who will be the individual in all situations"""
        print("\n=== CENTRAL AGENT CONFIGURATION ===")
        print("First, let's configure the central agent (an Individual).")
        print("This agent will be the individual in all simulated situations.")

        while True:
            name = input("Central agent name: ").strip()
            if name in self.agent_ids or name in self.culture_ids:
                print(f"Name '{name}' already exists!")
                continue
            break

        # Save the name for reference
        self.central_agent_name = name

        # Configure setup affinities
        print(f"\n=== SETUP AFFINITIES FOR {name} ===")
        self._configure_setup_affinities(name)

        # Create a temporary ID to use for configuring
        temp_id = str(uuid4())

        # Get behavior params using the temporary ID
        behavior_params = self._configure_behavior_params(agent_id=temp_id)

        # Prepare setups list from IDs
        setup_ids = list(self.setup_ids.values())
        setups = [Setup.get(setup_id) for setup_id in setup_ids]

        # Prepare behaviors dict with object references
        behaviors = {}
        for behavior_id, setup_data in behavior_params.items():
            behavior = Behavior.get(behavior_id)
            behaviors[behavior] = {}
            for setup_id, params in setup_data.items():
                setup = Setup.get(setup_id)
                behaviors[behavior][setup] = params

        process_matrix = {}  # Placeholder for process matrix

        # Create the individual
        individual = Individual(
            name=name,
            setups=setups,
            behaviors=behaviors,
            process_matrix=process_matrix,
            relationships={}
        )

        # Store ID mapping
        self.agent_ids[name] = individual.id
        self.central_agent_id = individual.id

        # Update the initial parameters with the real agent ID
        if temp_id in self.initial_parameters:
            self.initial_parameters[individual.id] = self.initial_parameters.pop(temp_id)

        # Transfer affinities from name to ID
        if name in self.agent_setup_affinities:
            self.agent_setup_affinities[individual.id] = self.agent_setup_affinities[name]

        print(f"Created central agent: {name}")

    def _initialize_parameter_history(self, agent):
        """Initialize history with current parameters"""
        self.parameter_history = {'instinct': {}, 'utility': {}, 'enjoyment': {}}
        for behavior in agent.behaviors:
            for setup in agent.behaviors[behavior]:
                for param in ['instinct', 'utility', 'enjoyment']:
                    self.parameter_history[param].setdefault(behavior.name, {})[setup.name] = [
                        agent.behaviors[behavior][setup][param]
                    ]

    def _update_parameter_history(self, agent):
        """Append current parameters to history"""
        for behavior in agent.behaviors:
            for setup in agent.behaviors[behavior]:
                for param in ['instinct', 'utility', 'enjoyment']:
                    print(f"Recording {param}={agent.behaviors[behavior][setup][param]} for {behavior.name} in {setup.name}")
                    self.parameter_history[param][behavior.name][setup.name].append(
                        agent.behaviors[behavior][setup][param]
                    )

    def _make_label(self, index):
        meta = self.metrics['situation_metadata'][index]
        return (
            f"Sit {index+1}\n"
            f"Mode: {meta['interaction_mode']}\n"
            f"Env: {meta['environment']}\n"
            f"Beh: {meta['selected_behavior']}"
        )

# Usage Example
if __name__ == "__main__":
    builder = SimulationBuilder()
    builder.configure_setups()
    builder.configure_behaviors()
    builder.configure_central_agent()  # Configure central agent first
    builder.configure_agents()         # Configure additional agents
    builder.configure_simulation()
    builder.run_simulation()
