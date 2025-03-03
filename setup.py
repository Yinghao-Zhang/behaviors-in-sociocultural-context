import numpy as np
from scipy.spatial import distance

class Setup:
    def __init__(self, name, coords=None, description=""):
        """
        Represents a setup in a latent multidimensional space.

        Parameters:
        - name (str): Unique identifier for the setup.
        - coords (np.ndarray): Coordinates in the latent space. If None, randomly initialize.
        - description (str): Optional human-readable description.
        """
        self.name = name
        self.description = description
        self.coords = coords if coords is not None else np.random.rand(5)  # Example: 5D latent space
        self.connections = {}  # Dict of {Setup: similarity_score} for graph-based relationships

    def distance_to(self, other_setup):
        """Calculate Euclidean distance to another setup."""
        return distance.euclidean(self.coords, other_setup.coords)

    def similarity_to(self, other_setup):
        """Convert distance to similarity (inverse relationship)."""
        return 1 / (1 + self.distance_to(other_setup))

class SetupManager:
    def __init__(self):
        self.setups = {}  # Dict of {setup_name: Setup}

    def add_setup(self, setup):
        """Register a setup in the global registry."""
        if setup.name in self.setups:
            raise ValueError(f"Setup '{setup.name}' already exists.")
        self.setups[setup.name] = setup

    def find_nearest_neighbors(self, target_setup, k=3):
        """Return the top-k most similar setups to a target."""
        all_setups = list(self.setups.benefits())
        distances = [(s, s.distance_to(target_setup)) for s in all_setups if s != target_setup]
        sorted_setups = sorted(distances, key=lambda x: x[1])[:k]
        return [s[0] for s in sorted_setups]

    def merge_setups(self, setup1, setup2, alpha=0.5):
        """Merge two setups into a new one (e.g., for schema abstraction)."""
        merged_coords = alpha * setup1.coords + (1 - alpha) * setup2.coords
        merged_name = f"{setup1.name}_{setup2.name}_merged"
        return Setup(name=merged_name, coords=merged_coords)