import argparse
import pandas as pd
import numpy as np
import os
import logging
import io
from experiments.simulation_builder import SimulationBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FileBasedSimulationBuilder(SimulationBuilder):
    """Extends SimulationBuilder with CSV/Excel-based configuration"""
    
    def __init__(self, auto_mode=False, debug=False):
        super().__init__()
        self.auto_mode = auto_mode  # Determines if we use config file inputs automatically
        self.debug = debug
        self.config_data = {}  # Holds loaded config data
        
        # Maps from prompts to config keys
        self.prompt_map = {
            # Setup configuration
            "Number of setups to model": "setup.count",
            "Setup name": "setup.name",
            
            # Behavior configuration
            "Number of behaviors to model": "behavior.count",
            "Behavior name": "behavior.name",
            "Difficulty (0-1)": "behavior.difficulty",
            "Base outcome (-1-1)": "behavior.base_outcome",
            "Outcome volatility (0-1)": "behavior.volatility",
            "Add modifiers for": "behavior.setup_modifier",
            
            # Agent configuration
            "Number of agents to create": "agent.count",
            "Agent type": "agent.type",
            "Individual name": "agent.name",
            "Group name": "agent.name",
            "Culture name": "agent.name", 
            "Group size": "agent.size",
            "Group homogeneity (0-1)": "agent.homogeneity",
            
            # Setup affinities
            "Affinity for": "agent.setup_affinity",
            
            # Behavior parameters
            "Instinct (-1-1)": "behavior_param.instinct",
            "Benefit (-1-1)": "behavior_param.benefit",
            "Skill (0-1)": "behavior_param.skill",
            "Enjoyment (-1-1)": "behavior_param.enjoyment",
            "Instinct α (0-1)": "behavior_param.alpha_instinct",
            "Benefit α (0-1)": "behavior_param.alpha_benefit",
            "Skill α (0-1)": "behavior_param.alpha_skill",
            
            # Relationships
            "Power (0-1)": "relationship.power",
            "Distance (0-1)": "relationship.distance",
            "Receptivity (0-1)": "relationship.receptivity",
            
            # Simulation parameters
            "Number of situations": "simulation.num_situations",
            "Logging detail": "simulation.log_level",
            "Generate visualizations?": "simulation.visualize"
        }
        
        # Keep track of context for sequential parameters
        self.current_section = None
        self.current_setup = None
        self.current_behavior = None
        self.current_agent = None
        self.current_target = None
        self.section_counters = {}
        
        # Text buffer for capturing inputs/outputs
        self.buffer = io.StringIO()
    
    def capture_input(self, prompt=""):
        """Override input to either use config values or capture prompts for CSV generation"""
        # Always log the prompt
        self.buffer.write(f"PROMPT: {prompt}\n")
        
        # Process the prompt to determine what parameter we're looking for
        param_key, param_index = self._map_prompt_to_config_key(prompt)
        
        if self.auto_mode and param_key is not None:
            # Try to get the value from config data
            value = self._get_config_value(param_key, param_index)
            if value is not None:
                self.buffer.write(f"AUTO-ANSWER: {value}\n")
                return str(value)
        
        # Fall back to regular input if no configured value
        value = input(prompt)
        self.buffer.write(f"USER-ANSWER: {value}\n")
        return value
    
    def _map_prompt_to_config_key(self, prompt):
        """Map a prompt to the corresponding config key"""
        prompt_lower = prompt.lower().strip()
        
        # Direct mapping first
        for key, value in self.prompt_map.items():
            if key.lower() in prompt_lower:
                # Update context based on the section
                section = value.split('.')[0]
                if section != self.current_section:
                    self.current_section = section
                    if section not in self.section_counters:
                        self.section_counters[section] = 0
                    else:
                        self.section_counters[section] += 1
                
                # Handle special case for setup names
                if "setup name" in prompt_lower:
                    setup_idx = self.section_counters.get('setup', 0)
                    self.current_setup = f"setup_{setup_idx}"
                
                # Handle special case for behavior names
                elif "behavior name" in prompt_lower:
                    behavior_idx = self.section_counters.get('behavior', 0)
                    self.current_behavior = f"behavior_{behavior_idx}"
                
                # Handle special case for agent names
                elif "name" in prompt_lower and any(x in prompt_lower for x in ["individual", "group", "culture"]):
                    agent_idx = self.section_counters.get('agent', 0)
                    self.current_agent = f"agent_{agent_idx}"
                
                # Return the mapped key and current index
                return value, self.section_counters.get(section, 0)
                
        # Handle special cases for behavior parameters
        if self.current_behavior and self.current_setup:
            if "instinct (-1-1)" in prompt_lower:
                return f"behavior_param.{self.current_behavior}.{self.current_setup}.instinct", 0
            if "benefit (-1-1)" in prompt_lower:
                return f"behavior_param.{self.current_behavior}.{self.current_setup}.benefit", 0
            if "skill (0-1)" in prompt_lower:
                return f"behavior_param.{self.current_behavior}.{self.current_setup}.skill", 0
            if "enjoyment (-1-1)" in prompt_lower:
                return f"behavior_param.{self.current_behavior}.{self.current_setup}.enjoyment", 0
            if "α" in prompt_lower and "instinct" in prompt_lower:
                return f"behavior_param.{self.current_behavior}.{self.current_setup}.alpha_instinct", 0
            if "α" in prompt_lower and "benefit" in prompt_lower:
                return f"behavior_param.{self.current_behavior}.{self.current_setup}.alpha_benefit", 0
            if "α" in prompt_lower and "skill" in prompt_lower:
                return f"behavior_param.{self.current_behavior}.{self.current_setup}.alpha_skill", 0
        
        # Handle special case for setup affinities
        match = None
        if "affinity for" in prompt_lower:
            import re
            match = re.search(r"affinity for (\w+)", prompt_lower)
            if match and self.current_agent:
                setup_name = match.group(1)
                return f"agent.{self.current_agent}.affinity.{setup_name}", 0
        
        # If no mapping found
        if self.debug:
            logger.debug(f"No config mapping found for prompt: {prompt}")
        return None, 0
    
    def _get_config_value(self, key, index=0):
        """Get value from config data based on key and index"""
        if key in self.config_data:
            values = self.config_data[key]
            if isinstance(values, list):
                if index < len(values):
                    return values[index]
                elif len(values) > 0:
                    # Fall back to first value if index is out of range
                    return values[0]
            else:
                # Convert non-list values to list for consistent handling
                self.config_data[key] = [values]
                return values
        
        if self.debug:
            logger.debug(f"No config value found for key '{key}' with index {index}")
        return None
    
    def generate_example_csv(self, file_path, simple=True):
        """Generate an example CSV configuration file by capturing the interactive process"""
        import builtins
        original_input = builtins.input
        
        try:
            # Override input for capture
            builtins.input = self.capture_input
            
            # Run through the simulation setup process
            # Don't actually run the simulation
            self.configure_setups()
            self.configure_behaviors()
            self.configure_agents()
            self.configure_simulation()
            
            # Parse the captured buffer into config data
            config_data = self._parse_buffer_to_config(self.buffer.getvalue())
            
            # Convert to DataFrame and save
            df = self._config_to_dataframe(config_data)
            
            # Save based on file extension
            if file_path.endswith('.xlsx'):
                try:
                    df.to_excel(file_path, index=False)
                except ImportError:
                    logger.warning("Excel writing support not installed. Try: pip install openpyxl")
                    file_path = file_path.replace('.xlsx', '.csv')
                    df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
                
            logger.info(f"Example configuration saved to {file_path}")
            
        finally:
            # Restore original input function
            builtins.input = original_input
    
    def _parse_buffer_to_config(self, buffer_content):
        """Parse the captured input/output buffer to config data"""
        config = {}
        lines = buffer_content.split('\n')
        
        current_key = None
        current_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('PROMPT:'):
                prompt = line[len('PROMPT:'):].strip()
                
                # Get the corresponding config key if any
                key, idx = self._map_prompt_to_config_key(prompt)
                if key:
                    current_key = key
                    current_idx = idx
            
            elif line.startswith('USER-ANSWER:') or line.startswith('AUTO-ANSWER:'):
                if current_key:
                    value = line.split(':', 1)[1].strip()
                    
                    # Convert to appropriate type
                    if value.replace('.', '', 1).isdigit():
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    
                    # Store in config
                    if current_key not in config:
                        config[current_key] = []
                    
                    # Ensure list is long enough
                    while len(config[current_key]) <= current_idx:
                        config[current_key].append(None)
                    
                    # Set the value
                    config[current_key][current_idx] = value
        
        return config
    
    def _config_to_dataframe(self, config_data):
        """Convert config data to a pandas DataFrame"""
        rows = []
        
        for key, values in config_data.items():
            if isinstance(values, list):
                for i, value in enumerate(values):
                    if value is not None:
                        rows.append({
                            'key': f"{key}_{i+1}" if len(values) > 1 else key,
                            'value': value
                        })
            else:
                rows.append({
                    'key': key,
                    'value': values
                })
        
        return pd.DataFrame(rows)
    
    def run_from_csv(self, file_path):
        """Run a simulation using parameters from a CSV/Excel file"""
        # Load the configuration
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Use .csv, .xlsx, or .xls")
        
        if self.debug:
            logger.debug(f"Loaded data with {len(df)} rows and columns: {df.columns}")
        
        # Convert DataFrame to config dict
        self.config_data = {}
        for _, row in df.iterrows():
            key = row['key']
            value = row['value']
            
            if self.debug:
                logger.debug(f"Processing config row: key={key}, value={value}")
            
            # Handle indexed keys (e.g., setup.name_1, setup.name_2)
            import re
            match = re.match(r'(.+)_(\d+)$', key)
            if match:
                base_key = match.group(1)
                index = int(match.group(2)) - 1
                
                # Initialize as list if not exists
                if base_key not in self.config_data:
                    self.config_data[base_key] = []
                
                # Ensure list is long enough
                while len(self.config_data[base_key]) <= index:
                    self.config_data[base_key].append(None)
                
                self.config_data[base_key][index] = value
            else:
                # IMPORTANT CHANGE: Always store as list for consistent handling
                if key not in self.config_data:
                    self.config_data[key] = [value]
                else:
                    # If it already exists as a list, append
                    if isinstance(self.config_data[key], list):
                        self.config_data[key].append(value)
                    else:
                        # Convert to list if not already
                        self.config_data[key] = [self.config_data[key], value]
        
        if self.debug:
            logger.debug("Configuration loaded:")
            for key, value in self.config_data.items():
                logger.debug(f"  {key}: {value}")
        
        # Enable auto mode to use config values
        self.auto_mode = True
        
        # Run the simulation
        import builtins
        original_input = builtins.input
        
        try:
            # Override input function
            builtins.input = self.capture_input
            
            # Configure and run simulation
            self.configure_setups()
            self.configure_behaviors()
            self.configure_agents()
            self.configure_simulation()
            self.run_simulation()
            
            logger.info("Simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            raise
            
        finally:
            # Restore original input
            builtins.input = original_input
    
    def create_simple_csv_template(self, file_path):
        """Create a simple CSV template with the most common parameters"""
        data = {
            'key': [
                'setup.count', 'setup.name_1',
                'behavior.count', 'behavior.name_1', 'behavior.difficulty_1', 'behavior.base_outcome_1', 'behavior.volatility_1',
                'behavior.name_2', 'behavior.difficulty_2', 'behavior.base_outcome_2', 'behavior.volatility_2',
                'agent.count', 'agent.type_1', 'agent.name_1',
                'agent.setup_affinity.home_1', 
                'behavior_param.behavior_0.setup_0.instinct', 'behavior_param.behavior_0.setup_0.benefit',
                'behavior_param.behavior_0.setup_0.skill', 'behavior_param.behavior_0.setup_0.enjoyment',
                'behavior_param.behavior_0.setup_0.alpha_instinct', 'behavior_param.behavior_0.setup_0.alpha_benefit',
                'behavior_param.behavior_0.setup_0.alpha_skill',
                'behavior_param.behavior_1.setup_0.instinct', 'behavior_param.behavior_1.setup_0.benefit',
                'behavior_param.behavior_1.setup_0.skill', 'behavior_param.behavior_1.setup_0.enjoyment',
                'behavior_param.behavior_1.setup_0.alpha_instinct', 'behavior_param.behavior_1.setup_0.alpha_benefit',
                'behavior_param.behavior_1.setup_0.alpha_skill',
                'agent.type_2', 'agent.name_2', 'agent.size_2', 'agent.homogeneity_2',
                'agent.setup_affinity.home_2',
                'behavior_param.behavior_0.setup_0.instinct_2', 'behavior_param.behavior_0.setup_0.benefit_2',
                'simulation.num_situations', 'simulation.log_level', 'simulation.visualize'
            ],
            'value': [
                1, 'home',
                2, 'drinking', 0.1, -0.3, 0.3,
                'idle', 0.0, 0.0, 0.1,
                2, '1', 'nick',
                1.0,
                0.8, -0.5, 1.0, 0.4, 0.5, 0.5, 0.5,
                -0.2, 0.1, 1.0, -0.1, 0.5, 0.5, 0.5,
                '2', 'parents', 2, 0.9,
                1.0,
                -0.1, -0.3,
                10, 3, 1
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Save based on file extension
        if file_path.endswith('.xlsx'):
            try:
                df.to_excel(file_path, index=False)
            except ImportError:
                logger.warning("Excel writing support not installed. Try: pip install openpyxl")
                file_path = file_path.replace('.xlsx', '.csv')
                df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)
            
        logger.info(f"Template configuration saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation from file or generate example configuration')
    parser.add_argument('--generate', type=str, help='Generate example configuration file (path to save)')
    parser.add_argument('--template', type=str, help='Generate a simple template configuration file')
    parser.add_argument('--run', type=str, help='Run simulation from configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.generate:
        builder = FileBasedSimulationBuilder(debug=args.debug)
        try:
            builder.generate_example_csv(args.generate)
        except Exception as e:
            logger.error(f"Error generating example: {e}")
    
    elif args.template:
        builder = FileBasedSimulationBuilder(debug=args.debug)
        try:
            builder.create_simple_csv_template(args.template)
        except Exception as e:
            logger.error(f"Error creating template: {e}")
    
    elif args.run:
        builder = FileBasedSimulationBuilder(debug=args.debug)
        try:
            builder.run_from_csv(args.run)
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
    
    else:
        logger.info("No action specified. Use --generate, --template, or --run")
        logger.info("Examples:")
        logger.info("  python run_simulation.py --template config_template.csv")
        logger.info("  python run_simulation.py --generate config_example.csv")
        logger.info("  python run_simulation.py --run my_config.csv")