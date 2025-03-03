import numpy as np
import pandas as pd

class Behavior:
    def __init__(
        self,
        name,
        difficulty=None,
        base_outcome=None,
        pleasantness=None,
        outcome_volatility=None,
        setup_modifiers=None
    ):
        """
        Represents a behavior with parameters that influence its execution and outcomes.

        Parameters:
        - name (str): Unique identifier for the behavior.
        - difficulty (float): Threshold for success (0 = trivial, 1 = extremely hard).
        - base_outcome (float): Baseline outcome if successful (-1 = harmful, 1 = beneficial).
        - pleasantness (float): Inherent pleasure/pain of the behavior (-1 = painful, 1 = pleasurable).
        - outcome_volatility (float): Magnitude of randomness in outcomes (0 = deterministic, 1 = chaotic).
        - setup_modifiers (dict): Contextual adjustments for specific setups. Format:
            {
                "setup_name": {
                    "base_outcome_mod": float,  # Added to base_outcome
                    "pleasantness_mod": float    # Added to pleasantness
                },
                ...
            }
        """
        self.name = name

        # Set defaults using research-driven distributions
        self.difficulty = self._default_difficulty() if difficulty is None else difficulty
        self.base_outcome = self._default_base_outcome() if base_outcome is None else base_outcome
        self.pleasantness = self._default_pleasantness() if pleasantness is None else pleasantness
        self.outcome_volatility = self._default_outcome_volatility() if outcome_volatility is None else outcome_volatility

        self._validate()
        # Setup-specific modifiers (default: empty dict)
        self.setup_modifiers = setup_modifiers if setup_modifiers is not None else {}
        self._validate_setup_modifiers()

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

    def _default_pleasantness(self):
        """Trimodal distribution: Neutral (60%), Pleasant (20%), Unpleasant (20%)."""
        mode = np.random.choice(["neutral", "pleasant", "unpleasant"], p=[0.6, 0.2, 0.2])
        if mode == "neutral":
            return np.clip(np.random.normal(0, 0.15), -1, 1)
        elif mode == "pleasant":
            return np.random.beta(8, 2) * 0.5 + 0.5  # Range: 0.5 to 1
        else:
            return - (np.random.beta(8, 2) * 0.5 + 0.5)  # Range: -1 to -0.5

    def _default_outcome_volatility(self):
        """Right-skewed Beta distribution: Most behaviors have low luck dependency."""
        return np.random.beta(2, 8)

    def _validate(self):
        """Ensure parameters are within valid ranges."""
        if not (0 <= self.difficulty <= 1):
            raise ValueError(f"Difficulty must be in [0, 1], got {self.difficulty}.")
        if not (-1 <= self.base_outcome <= 1):
            raise ValueError(f"Base outcome must be in [-1, 1], got {self.base_outcome}.")
        if not (-1 <= self.pleasantness <= 1):
            raise ValueError(f"Pleasantness must be in [-1, 1], got {self.pleasantness}.")
        if not (0 <= self.outcome_volatility <= 1):
            raise ValueError(f"Luck factor must be in [0, 1], got {self.outcome_volatility}.")

    def __eq__(self, other):
        return isinstance(other, Behavior) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return (
            f"Behavior(name='{self.name}', difficulty={self.difficulty:.2f}, "
            f"base_outcome={self.base_outcome:.2f}, pleasantness={self.pleasantness:.2f}, "
            f"outcome_volatility={self.outcome_volatility:.2f})"
        )

    def _validate_setup_modifiers(self):
        """Validate structure of setup_modifiers."""
        for setup_name, modifiers in self.setup_modifiers.items():
            if not isinstance(setup_name, str):
                raise ValueError(f"Setup name must be a string, got {type(setup_name)}.")
            if not isinstance(modifiers, dict):
                raise ValueError(f"Modifiers for {setup_name} must be a dict.")
            for key in modifiers:
                if key not in ["base_outcome_mod", "pleasantness_mod"]:
                    raise ValueError(f"Invalid modifier key '{key}' for {setup_name}.")

    def get_contextual_outcome(self, setup_name):
        """Get outcome adjusted for a specific setup."""
        mod = self.setup_modifiers.get(setup_name, {})
        adjusted = self.base_outcome + mod.get("base_outcome_mod", 0)
        return np.clip(adjusted, -1, 1)

    def get_contextual_pleasantness(self, setup_name):
        """Get pleasantness adjusted for a specific setup."""
        mod = self.setup_modifiers.get(setup_name, {})
        adjusted = self.pleasantness + mod.get("pleasantness_mod", 0)
        return np.clip(adjusted, -1, 1)
