import numpy as np
"""
Hyperparameter tuning utilities for Agent hyperparameters.
Do NOT import Agent or hyperparam_manager here to avoid circular imports.
"""
from copy import deepcopy


class HyperparameterTuner:
    """
    Implements k-fold cross-validation and grid/random search for hyperparameter tuning.
    Agent class and hyperparam_manager must be passed in at runtime to avoid circular import.
    """
    def __init__(self, agent_class, param_grid, k=5, hyperparam_manager=None):
        self.agent_class = agent_class
        self.param_grid = param_grid  # Dict of param: list of values
        self.k = k
        self.hyperparam_manager = hyperparam_manager

    def k_fold_split(self, data):
        np.random.shuffle(data)
        fold_size = len(data) // self.k
        return [data[i*fold_size:(i+1)*fold_size] for i in range(self.k)]

    def grid_search(self, train_data, eval_fn):
        from itertools import product
        keys, values = zip(*self.param_grid.items())
        best_score = -np.inf
        best_params = None
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            if self.hyperparam_manager:
                self.hyperparam_manager.load_from_dict(param_dict)
            scores = []
            folds = self.k_fold_split(train_data)
            for i in range(self.k):
                val_fold = folds[i]
                train_folds = [item for j, fold in enumerate(folds) if j != i for item in fold]
                agent = self.agent_class()  # Re-init agent for each fold
                score = eval_fn(agent, train_folds, val_fold)
                scores.append(score)
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = param_dict.copy()
        return best_params, best_score

    def random_search(self, train_data, eval_fn, n_iter=20):
        keys = list(self.param_grid.keys())
        best_score = -np.inf
        best_params = None
        for _ in range(n_iter):
            param_dict = {k: np.random.choice(v) for k, v in self.param_grid.items()}
            if self.hyperparam_manager:
                self.hyperparam_manager.load_from_dict(param_dict)
            scores = []
            folds = self.k_fold_split(train_data)
            for i in range(self.k):
                val_fold = folds[i]
                train_folds = [item for j, fold in enumerate(folds) if j != i for item in fold]
                agent = self.agent_class()
                score = eval_fn(agent, train_folds, val_fold)
                scores.append(score)
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = param_dict.copy()
        return best_params, best_score

# Example usage:
# param_grid = {
#     'alpha_instinct_plus': [0.05, 0.1, 0.2],
#     'alpha_instinct_minus': [0.05, 0.1, 0.2],
#     'w_enjoyment': [0.3, 0.5, 0.7],
#     'w_utility': [0.3, 0.5, 0.7],
#     'bias_scaling_factor': [0.5, 1.0, 2.0]
# }
# tuner = HyperparameterTuner(Agent, param_grid, k=5)
# best_params, best_score = tuner.grid_search(train_data, eval_fn)
