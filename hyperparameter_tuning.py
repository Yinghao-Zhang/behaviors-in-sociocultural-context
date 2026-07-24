import numpy as np
"""
Hyperparameter tuning utilities for Agent hyperparameters.
Do NOT import Agent or hyperparam_manager here to avoid circular imports.
"""
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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


def split_by_group(rows: Sequence[dict], group_key: str, k: int, seed: int = 123) -> List[List[dict]]:
    if k <= 1:
        return [list(rows)]
    rng = np.random.default_rng(seed)
    groups = {}
    for row in rows:
        if group_key not in row:
            raise KeyError(f"Missing group_key '{group_key}' in row.")
        groups.setdefault(row[group_key], []).append(row)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    fold_size = max(1, len(group_ids) // k)
    folds: List[List[dict]] = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < (k - 1) else len(group_ids)
        fold_groups = group_ids[start:end]
        fold_rows = [row for gid in fold_groups for row in groups[gid]]
        folds.append(fold_rows)
    return folds


class EmbeddingPooler:
    """
    Gaussian-kernel pooling over embeddings with shrinkage to global mean.
    This is fold-safe when fit() is called on training data only.
    """
    def __init__(self, shrinkage: float = 6.0, exclude_keys: Optional[Iterable[str]] = None):
        self.shrinkage = float(shrinkage)
        self.exclude_keys = set(exclude_keys or {"neg_log_lik", "fit_ok"})
        self._train_embeddings: Optional[np.ndarray] = None
        self._train_rows: Optional[List[Dict[str, float]]] = None
        self._global_mean: Optional[Dict[str, float]] = None

    def fit(self, embeddings: np.ndarray, param_rows: List[Dict[str, float]]):
        if len(embeddings) != len(param_rows):
            raise ValueError("Embeddings and param_rows must have the same length.")
        self._train_embeddings = np.asarray(embeddings, dtype=float)
        self._train_rows = param_rows
        self._global_mean = self._compute_mean(param_rows)

    def _compute_mean(self, rows: List[Dict[str, float]]) -> Dict[str, float]:
        keys = rows[0].keys() if rows else []
        out: Dict[str, float] = {}
        for k in keys:
            if k in self.exclude_keys:
                continue
            vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
            if vals:
                out[k] = float(np.mean(vals))
        return out

    def _weighted_mean(self, rows: List[Dict[str, float]], weights: np.ndarray) -> Dict[str, float]:
        w = np.asarray(weights, dtype=float)
        if np.sum(w) <= 0:
            w = np.ones(len(rows), dtype=float)
        w = w / np.sum(w)
        out: Dict[str, float] = {}
        keys = rows[0].keys() if rows else []
        for k in keys:
            if k in self.exclude_keys:
                continue
            vals = np.array([float(r[k]) for r in rows if k in r and r[k] is not None], dtype=float)
            if len(vals) == len(rows):
                out[k] = float(np.dot(w, vals))
        return out

    def predict(self, target_embedding: np.ndarray) -> Dict[str, float]:
        if self._train_embeddings is None or self._train_rows is None or self._global_mean is None:
            raise RuntimeError("Pooler must be fit() before predict().")
        tgt = np.asarray(target_embedding, dtype=float).reshape(1, -1)
        d = np.linalg.norm(self._train_embeddings - tgt, axis=1)
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        sigma = float(max(med, 1e-6))
        sim_w = np.exp(-0.5 * (d / sigma) ** 2)

        local = self._weighted_mean(self._train_rows, sim_w)
        n_eff = float((np.sum(sim_w) ** 2) / max(np.sum(sim_w ** 2), 1e-8))
        lam = n_eff / (n_eff + self.shrinkage)
        pooled: Dict[str, float] = {}
        for k, v in local.items():
            if k in self._global_mean:
                pooled[k] = float(lam * v + (1.0 - lam) * float(self._global_mean[k]))
            else:
                pooled[k] = float(v)
        return pooled


class BetweenPersonTuner:
    """
    Between-person cross-validation tuner with fold-safe feature preparation.
    The eval_fn receives (param_dict, train_rows, val_rows, fold_context).
    """
    def __init__(
        self,
        param_grid: Dict[str, Sequence],
        k: int = 5,
        group_key: str = "person_id",
        seed: int = 123,
        fold_builder: Optional[Callable[[List[dict]], Dict]] = None,
    ):
        self.param_grid = param_grid
        self.k = int(k)
        self.group_key = group_key
        self.seed = int(seed)
        self.fold_builder = fold_builder

    def grid_search(self, rows: Sequence[dict], eval_fn: Callable):
        from itertools import product
        keys, values = zip(*self.param_grid.items()) if self.param_grid else ([], [])
        best_score = -np.inf
        best_params = None

        folds = split_by_group(rows, self.group_key, self.k, self.seed)
        for combo in product(*values) if values else [()]:
            param_dict = dict(zip(keys, combo)) if keys else {}
            scores = []
            for i in range(len(folds)):
                val_fold = folds[i]
                train_folds = [item for j, fold in enumerate(folds) if j != i for item in fold]
                fold_context = self.fold_builder(train_folds) if self.fold_builder else {}
                score = eval_fn(param_dict, train_folds, val_fold, fold_context)
                scores.append(score)
            avg_score = float(np.mean(scores)) if scores else -np.inf
            if avg_score > best_score:
                best_score = avg_score
                best_params = param_dict.copy()
        return best_params, best_score

    def random_search(self, rows: Sequence[dict], eval_fn: Callable, n_iter: int = 20):
        keys = list(self.param_grid.keys())
        best_score = -np.inf
        best_params = None
        folds = split_by_group(rows, self.group_key, self.k, self.seed)

        for _ in range(n_iter):
            param_dict = {k: np.random.choice(v) for k, v in self.param_grid.items()}
            scores = []
            for i in range(len(folds)):
                val_fold = folds[i]
                train_folds = [item for j, fold in enumerate(folds) if j != i for item in fold]
                fold_context = self.fold_builder(train_folds) if self.fold_builder else {}
                score = eval_fn(param_dict, train_folds, val_fold, fold_context)
                scores.append(score)
            avg_score = float(np.mean(scores)) if scores else -np.inf
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
