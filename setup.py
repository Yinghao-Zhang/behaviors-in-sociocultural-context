import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize
from uuid import uuid4

class Setup:
    _registry = {}  # {id: setup_object}
    
    def to_dict(self):
        """Convert setup to serializable dictionary."""
        return {
            'name': self.name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create setup from dictionary."""
        return cls(name=data['name'], description=data['description'])
    
    def __init__(self, name, description=""):
        """
        Represents a setup in a latent multidimensional space.

        Parameters:
        - name (str): Unique identifier for the setup.
        - description (str): Optional human-readable description.
        """
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.connections = {}  # Dict of {Setup: similarity_score} for graph-based relationships
        Setup._registry[self.id] = self
    
    @classmethod
    def get(cls, setup_id):
        return cls._registry.get(setup_id)

class SetupManager:
    def __init__(self):
        self.setups = {}  # Dict of {setup_name: Setup}
        self.coords = {}  # Dict of {setup_name: np.ndarray}
        self.distances = {}  # Dict of {(setup1_name, setup2_name): distance}
        
    def add_setup(self, setup, reference_distance=None, reference_setup=None):
        """
        Register a setup in the global registry.
        
        Parameters:
        - setup: Setup object to add
        - reference_distance: Optional float indicating distance to reference_setup
        - reference_setup: Optional Setup object to measure distance from
        """
        if setup.name in self.setups:
            raise ValueError(f"Setup '{setup.name}' already exists.")
            
        self.setups[setup.name] = setup
        
        # Handle the first setup differently
        if len(self.setups) == 1:
            self.coords[setup.name] = np.zeros(1)
            return
            
        # For subsequent setups
        n_dims = len(self.setups) - 1
        
        if reference_distance is not None and reference_setup is not None:
            # Store the specified distance
            self.distances[(setup.name, reference_setup.name)] = reference_distance
            self.distances[(reference_setup.name, setup.name)] = reference_distance
            
        # Recompute all coordinates to satisfy distances
        self._update_coordinates()
    
    def _update_coordinates(self):
        """Update coordinates of all setups to satisfy known distances."""
        n = len(self.setups)
        if n <= 1:
            return
            
        # Initialize coordinates randomly in (n-1)-dimensional space
        setup_names = list(self.setups.keys())
        coords = np.random.rand(n, n-1)
        
        def objective(x):
            # Reshape flattened coordinates back to matrix
            curr_coords = x.reshape(n, n-1)
            
            # Calculate error between desired and actual distances
            error = 0
            for i in range(n):
                for j in range(i+1, n):
                    name1, name2 = setup_names[i], setup_names[j]
                    if (name1, name2) in self.distances:
                        desired_dist = self.distances[(name1, name2)]
                        actual_dist = np.linalg.norm(curr_coords[i] - curr_coords[j])
                        error += (desired_dist - actual_dist) ** 2
            return error
        
        # Optimize coordinates to minimize distance errors
        result = minimize(objective, coords.flatten(), method='L-BFGS-B')
        optimized_coords = result.x.reshape(n, n-1)
        
        # Update stored coordinates
        for i, name in enumerate(setup_names):
            self.coords[name] = optimized_coords[i]
    
    def distance_between(self, setup1, setup2):
        """Calculate Euclidean distance between two setups."""
        if setup1.name not in self.coords or setup2.name not in self.coords:
            raise ValueError("Both setups must be registered in the manager")
        return np.linalg.norm(self.coords[setup1.name] - self.coords[setup2.name])
    
    def similarity_between(self, setup1, setup2):
        """Convert distance to similarity (inverse relationship)."""
        return 1 / (1 + self.distance_between(setup1, setup2))

    def find_nearest_neighbors(self, target_setup, k=3):
        """Return the top-k most similar setups to a target."""
        all_setups = [s for s in self.setups.values() if s != target_setup]
        distances = [(s, self.distance_between(s, target_setup)) for s in all_setups]
        sorted_setups = sorted(distances, key=lambda x: x[1])[:k]
        return [s[0] for s in sorted_setups]
