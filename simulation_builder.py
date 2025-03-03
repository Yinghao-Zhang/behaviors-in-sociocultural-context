import numpy as np
import matplotlib.pyplot as plt
from uuid import uuid4

class SimulationBuilder:
    def __init__(self):
        self.setups = {}
        self.behaviors = {}
        self.agents = {}
        self.cultures = {}
        self.simulation_params = {}

    # Phase 1: Setup Configuration
    def configure_setups(self):
        print("\n=== SETUP CONFIGURATION ===")
        num_setups = self._get_input("Number of setups to model", int, min_val=1)

        for _ in range(num_setups):
            name = input("Setup name: ").strip()
            print(f"Define latent space coordinates for {name} (comma-separated floats)")
            coords = self._get_list(float)
            self.setups[name] = Setup(name, np.array(coords))

    # Phase 2: Behavior Configuration
    def configure_behaviors(self):
        print("\n=== BEHAVIOR CONFIGURATION ===")
        num_behaviors = self._get_input("Number of behaviors to model", int, min_val=2)

        for _ in range(num_behaviors):
            name = input("Behavior name: ").strip()
            difficulty = self._get_input(f"Difficulty for {name}", float, 0, 1)
            base_outcome = self._get_input(f"Base outcome for {name}", float, -1, 1)
            volatility = self._get_input(f"Outcome volatility for {name}", float, 0, 1)
            self.behaviors[name] = Behavior(name, difficulty, base_outcome, volatility)

    # Phase 3: Agent Configuration
    def configure_agents(self):
        print("\n=== AGENT CONFIGURATION ===")
        agent_types = {
            '1': ('Individual', Individual),
            '2': ('Group', Group)
        }

        num_agents = self._get_input("Number of agents to create", int, 1)
        for i in range(1, num_agents+1):
            print(f"\nAgent {i} Configuration")
            type_choice = self._get_choice("Agent type:", agent_types)
            agent_class = agent_types[type_choice][1]

            agent_name = input("Agent name: ").strip()
            agent = self._create_agent(agent_class, agent_name)
            self.agents[agent_name] = agent

    def _create_agent(self, agent_class, name):
        setups = self._configure_setup_affinities()
        behaviors = self._configure_behavior_params()
        relationships = self._configure_relationships()

        if agent_class == Group:
            size = self._get_input("Group size", int, 1)
            homogeneity = self._get_input("Group homogeneity", float, 0, 1)
            return Group(name=name, size=size, homogeneity=homogeneity,
                        setups=setups, behaviors=behaviors, relationships=relationships)
        return Individual(name=name, setups=setups, behaviors=behaviors,
                         relationships=relationships)

    # Helper methods with validation

    def _get_list(self, dtype, expected_length=None):
        """Helper to get validated list input"""
        while True:
            try:
                raw = input(">> ").strip()
                values = [dtype(x) for x in raw.split(",")]

                if expected_length and len(values) != expected_length:
                    raise ValueError(f"Need exactly {expected_length} values")
                return values
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")

    def _get_input(self, prompt, dtype, min_val=None, max_val=None):
        while True:
            try:
                val = dtype(input(f"{prompt}: "))
                if min_val is not None and val < min_val:
                    raise ValueError(f"Must be ≥ {min_val}")
                if max_val is not None and val > max_val:
                    raise ValueError(f"Must be ≤ {max_val}")
                return val
            except ValueError as e:
                print(f"Invalid input: {e}")

    def _get_choice(self, prompt, options):
        print(prompt)
        for k, (name, _) in options.items():
            print(f"{k}. {name}")
        while True:
            choice = input("Select option: ").strip()
            if choice in options:
                return choice
            print("Invalid choice")

    # ... (similar methods for other configuration steps)

    # Phase 4: Simulation Parameters
    def configure_simulation(self):
        print("\n=== SIMULATION PARAMETERS ===")
        self.simulation_params = {
            'num_situations': self._get_input("Number of situations", int, 1),
            'log_level': self._get_choice(
                "Logging detail:",
                {'1': 'Minimal', '2': 'Detailed', '3': 'Debug'}
            ),
            'visualize': self._get_choice(
                "Generate visualizations?",
                {'1': 'Yes', '2': 'No'}
            ) == '1'
        }

    def _configure_setup_affinities(self):
        """Configure agent's affinity for each setup"""
        affinities = {}
        print("\n=== SETUP AFFINITIES ===")
        for setup_name, setup in self.setups.items():  # setup_name is now the key
            affinity = self._get_input(
                f"Affinity for {setup_name} (0-1)",
                float, 0, 1
            )
            affinities[setup_name] = {'affinity': affinity}  # Use setup_name as key
        return affinities

    def _configure_behavior_params(self):
        """Configure parameters for each behavior"""
        params = {}
        print("\n=== BEHAVIOR PARAMETERS ===")
        for behavior_name, behavior in self.behaviors.items():
            print(f"\nParameters for {behavior_name}:")
            params[behavior] = {} # initialize behavior dict
            for setup_name, setup in self.setups.items(): # loop with setup_name
                print(f"\nParameters for {behavior_name} in setup {setup_name}:")
                params[behavior][setup] = { # keep using setup object as key
                    'instinct': self._get_input("Instinct (-1 to 1)", float, -1, 1),
                    'benefit': self._get_input("Benefit (-1 to 1)", float, -1, 1),
                    'skill': self._get_input("Skill (0-1)", float, 0, 1),
                    'enjoyment': self._get_input("Enjoyment (-1 to 1)", float, -1, 1),
                    'alpha_instinct': self._get_input("Instinct learning rate (0-1)", float, 0, 1),
                    'alpha_benefit': self._get_input("Benefit learning rate (0-1)", float, 0, 1),
                    'alpha_skill': self._get_input("Skill learning rate (0-1)", float, 0, 1),
                }
        return params

    def _configure_relationships(self):
        """Configure relationships with other entities"""
        relationships = {}
        print("\n=== RELATIONSHIPS ===")
        while True:
            print("\nAvailable entities:")
            entities = list(self.agents.values()) + list(self.cultures.values())
            for i, entity in enumerate(entities, 1):
                print(f"{i}. {entity.name} ({type(entity).__name__})")
            print("0. Done configuring relationships")

            choice = self._get_input("Select entity to relate to", int, 0, len(entities))
            if choice == 0:
                break

            target = entities[choice-1]
            print(f"\nConfiguring relationship with {target.name}:")
            relationships[target.id] = {
                'distance': self._get_input("Distance (0-1)", float, 0, 1),
                'receptivity': self._get_input("Receptivity (0-1)", float, 0, 1),
                'power': 0 if isinstance(target, Culture) else self._get_input("Power (0-1)", float, 0, 1)
            }
        return relationships

    # Phase 5: Run Simulation
    def run_simulation(self):
        print("\n=== RUNNING SIMULATION ===")
        # Initialize agents and situations
        # Implement simulation loop
        # Log results based on settings

        if self.simulation_params['visualize']:
            self._generate_visualizations()

    def _generate_visualizations(self):
        # Implement visualization logic
        pass

# Usage:
if __name__ == "__main__":
    builder = SimulationBuilder()
    builder.configure_setups()
    builder.configure_behaviors()
    builder.configure_agents()
    builder.configure_simulation()
    builder.run_simulation()