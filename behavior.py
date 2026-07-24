import numpy as np
import pandas as pd
from setup import Setup, SetupManager
from behavior_taxonomy import BehaviorTaxonomy
from uuid import uuid4
from typing import Dict, Optional

class Behavior:
    _registry = {}  # Class-level registry
    _relations = {}  # (behavior_id_a, behavior_id_b) -> relation params
    _taxonomy = None  # Optional BehaviorTaxonomy
    _global_priors = {}  # behavior_id -> {instinct, enjoyment, utility}

    @classmethod
    def set_taxonomy(cls, taxonomy: Optional[BehaviorTaxonomy]):
        cls._taxonomy = taxonomy

    @classmethod
    def clear_taxonomy(cls):
        cls._taxonomy = None

    @classmethod
    def set_global_prior(cls, behavior_or_id, instinct: Optional[float] = None,
                         enjoyment: Optional[float] = None, utility: Optional[float] = None):
        behavior_id = cls._resolve_behavior_id(behavior_or_id)
        if cls.get(behavior_id) is None:
            raise KeyError(f"Behavior '{behavior_id}' not found in registry.")

        if instinct is not None and not (-1.0 <= instinct <= 1.0):
            raise ValueError(f"Instinct prior must be in [-1, 1], got {instinct}.")
        if enjoyment is not None and not (-1.0 <= enjoyment <= 1.0):
            raise ValueError(f"Enjoyment prior must be in [-1, 1], got {enjoyment}.")
        if utility is not None and not (-1.0 <= utility <= 1.0):
            raise ValueError(f"Utility prior must be in [-1, 1], got {utility}.")

        cls._global_priors[behavior_id] = {
            "instinct": instinct,
            "enjoyment": enjoyment,
            "utility": utility,
        }

    @classmethod
    def get_global_prior(cls, behavior_or_id) -> Optional[Dict[str, Optional[float]]]:
        behavior_id = cls._resolve_behavior_id(behavior_or_id)
        return cls._global_priors.get(behavior_id)

    @classmethod
    def clear_global_prior(cls, behavior_or_id=None):
        if behavior_or_id is None:
            cls._global_priors = {}
            return
        behavior_id = cls._resolve_behavior_id(behavior_or_id)
        cls._global_priors.pop(behavior_id, None)

    @classmethod
    def _resolve_behavior_id(cls, behavior_or_id):
        if isinstance(behavior_or_id, Behavior):
            return behavior_or_id.id
        if isinstance(behavior_or_id, str):
            return behavior_or_id
        raise TypeError(f"Behavior identifier must be Behavior or str, got {type(behavior_or_id)}")

    @classmethod
    def add_relation(cls, behavior_a, behavior_b, similarity: float, metadata: Optional[Dict] = None):
        behavior_a_id = cls._resolve_behavior_id(behavior_a)
        behavior_b_id = cls._resolve_behavior_id(behavior_b)

        if cls.get(behavior_a_id) is None or cls.get(behavior_b_id) is None:
            raise KeyError("Both behaviors must exist in the registry before adding a relation.")
        if not (-1.0 <= similarity <= 1.0):
            raise ValueError(f"Similarity must be in [-1, 1], got {similarity}.")

        payload = {
            "similarity": float(similarity),
            "metadata": metadata or {},
        }
        cls._relations[(behavior_a_id, behavior_b_id)] = payload
        cls._relations[(behavior_b_id, behavior_a_id)] = payload

    @classmethod
    def get_relation(cls, behavior_a, behavior_b):
        behavior_a_id = cls._resolve_behavior_id(behavior_a)
        behavior_b_id = cls._resolve_behavior_id(behavior_b)
        return cls._relations.get((behavior_a_id, behavior_b_id))

    def to_dict(self):
        """Convert behavior to serializable dictionary."""
        return {
            'name': self.name,
            'difficulty': self.difficulty,
            'base_outcome': self.base_outcome,
            'outcome_volatility': self.outcome_volatility,
            'setup_modifiers': self.setup_modifiers,
            'primary_domain': self.primary_domain,
            'motivational_system': self.motivational_system,
            'regulatory_direction': self.regulatory_direction
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create behavior from dictionary."""
        return cls(
            name=data['name'],
            difficulty=data['difficulty'],
            base_outcome=data['base_outcome'],
            outcome_volatility=data['outcome_volatility'],
            setup_modifiers=data.get('setup_modifiers'),
            primary_domain=data.get('primary_domain'),
            motivational_system=data.get('motivational_system'),
            regulatory_direction=data.get('regulatory_direction')
        )
    
    def __init__(
        self,
        name,
        difficulty=None,
        base_outcome=None,
        outcome_volatility=None,
        setup_modifiers=None,
        primary_domain=None,
        motivational_system=None,
        regulatory_direction=None
    ):
        """
        Represents a behavior with parameters that influence its execution and outcomes.

        Parameters:
        - name (str): Unique identifier for the behavior.
        - difficulty (float): Threshold for success (0 = trivial, 1 = extremely hard).
        - base_outcome (float): Baseline outcome if successful (-1 = harmful, 1 = beneficial).
        - outcome_volatility (float): Magnitude of randomness in outcomes (0 = deterministic, 1 = chaotic).
        - setup_modifiers (dict): Contextual adjustments for specific setups. Format:
            {
                setup_or_setup_name: {
                    "base_outcome_mod": float  # Added to base_outcome
                    "difficulty_mod": float  # Added to difficulty
                    "outcome_volatility": float  # Optional context-specific volatility
                },
                ...
            }
        - primary_domain (str): Optional domain label for this behavior.
        - motivational_system (str): Optional motivational system label.
        - regulatory_direction (str): Optional regulatory direction label.
        """
        self.id = str(uuid4())  # Add unique ID
        self.name = name

        # Set defaults using research-driven distributions
        self.difficulty = self._default_difficulty() if difficulty is None else difficulty
        self.base_outcome = self._default_base_outcome() if base_outcome is None else base_outcome
        self.outcome_volatility = self._default_outcome_volatility() if outcome_volatility is None else outcome_volatility

        self._validate()
        # Setup-specific modifiers (default: empty dict)
        self.setup_modifiers = setup_modifiers if setup_modifiers is not None else {}
        self._validate_setup_modifiers()

        self.primary_domain = primary_domain
        self.motivational_system = motivational_system
        self.regulatory_direction = regulatory_direction
        self._validate_taxonomy_fields()

        # Register instance
        Behavior._registry[self.id] = self

    def _default_difficulty(self):
        """Right-skewed Beta distribution: Most behaviors are moderately easy."""
        return np.random.beta(2, 5)

    def _default_base_outcome(self):
        """Bimodal distribution: 50% effective (positive), 50% ineffective (negative)."""
        mode = np.random.choice(["negative", "positive"], p=[0.5, 0.5])
        if mode == "negative":
            return -np.random.beta(2, 8)  # Mildly harmful (range: -1 to 0)
        else:
            return np.random.beta(8, 2)    # Strongly beneficial (range: 0 to 1)

    def _default_outcome_volatility(self):
        """Right-skewed Beta distribution: Most behaviors have low luck dependency."""
        return np.random.beta(2, 8)

    def _validate(self):
        """Ensure parameters are within valid ranges."""
        if not (0 <= self.difficulty <= 1):
            raise ValueError(f"Difficulty must be in [0, 1], got {self.difficulty}.")
        if not (-1 <= self.base_outcome <= 1):
            raise ValueError(f"Base outcome must be in [-1, 1], got {self.base_outcome}.")
        if not (0 <= self.outcome_volatility <= 1):
            raise ValueError(f"Luck factor must be in [0, 1], got {self.outcome_volatility}.")

    def _validate_taxonomy_fields(self):
        taxonomy = Behavior._taxonomy
        if taxonomy is None:
            return

        if self.primary_domain is not None and not taxonomy.validate_domain(self.primary_domain):
            raise ValueError(f"Unknown primary_domain: {self.primary_domain}.")
        if self.motivational_system is not None and not taxonomy.validate_motivational_system(self.motivational_system):
            raise ValueError(f"Unknown motivational_system: {self.motivational_system}.")
        if self.regulatory_direction is not None and not taxonomy.validate_regulatory_direction(self.regulatory_direction):
            raise ValueError(f"Unknown regulatory_direction: {self.regulatory_direction}.")

    def __eq__(self, other):
        return isinstance(other, Behavior) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return (
            f"Behavior(name='{self.name}', difficulty={self.difficulty:.2f}, "
            f"base_outcome={self.base_outcome:.2f}"
            f"outcome_volatility={self.outcome_volatility:.2f})"
        )

    def _validate_setup_modifiers(self):
        """Validate structure of setup_modifiers."""
        for setup_name, modifiers in self.setup_modifiers.items():
            if not isinstance(setup_name, (str, Setup)):
                raise ValueError(f"Setup key must be a string or Setup, got {type(setup_name)}.")
            if not isinstance(modifiers, dict):
                raise ValueError(f"Modifiers for {setup_name} must be a dict.")
            for key in modifiers:
                if key not in [
                    "base_outcome_mod",
                    "difficulty_mod",
                    "outcome_volatility",
                    "enjoyment_outcome",
                    "enjoyment_outcome_mod",
                    "enjoyment_volatility",
                ]:
                    raise ValueError(f"Invalid modifier key '{key}' for {setup_name}.")

    def get_contextual_outcome(self, setup_name):
        """Get outcome adjusted for a specific setup."""
        mod = self.setup_modifiers.get(setup_name, {})
        adjusted = self.base_outcome + mod.get("base_outcome_mod", 0)
        return np.clip(adjusted, -1, 1)

    @classmethod
    def get(cls, behavior_id):
        return cls._registry.get(behavior_id)
