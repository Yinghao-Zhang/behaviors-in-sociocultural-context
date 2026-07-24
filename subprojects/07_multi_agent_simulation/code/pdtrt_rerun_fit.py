from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from pdtrt_rerun_core import (
    BEHAVIORS,
    CONTEXTS,
    STATE_NAMES,
    PanelView,
    stable_seed,
)


COMPUTATIONAL_MODELS: Tuple[str, ...] = (
    "tripartite",
    "no_learning",
    "collapsed_reward",
)
CANDIDATE_MODELS: Tuple[str, ...] = (*COMPUTATIONAL_MODELS, "lagged")


@dataclass(frozen=True)
class EmpiricalBayesConfig:
    eb_iterations: int = 3
    max_iter: int = 120
    multistarts: int = 2
    variance_update: str = "laplace"
    prior_sd_initial: float = 0.90
    prior_sd_floor: float = 0.30
    prior_sd_ceiling: float = 2.50
    baseline_reliability: float = 0.70
    observation_attenuation: float = 0.50
    lagged_c: float = 1.0
    seed: int = 20260718


@dataclass
class ModelFitResult:
    model: str
    metrics: Dict[str, float]
    diagnostics: Dict[str, float]
    predictions: pd.DataFrame
    person_parameters: pd.DataFrame
    recovery: pd.DataFrame
    population_parameters: Dict[str, float]


@dataclass(frozen=True)
class ParameterSchema:
    model: str
    theta_names: Tuple[str, ...]
    bounds: Tuple[Tuple[float, float], ...]


SCHEMAS: Dict[str, ParameterSchema] = {
    "tripartite": ParameterSchema(
        model="tripartite",
        theta_names=(
            "weight_i_logratio",
            "weight_e_logratio",
            "alpha_i_pos_logit",
            "alpha_i_neg_logit",
            "alpha_e_logit",
            "alpha_u_logit",
            "social_kappa_logit",
            "tau_logit",
            "noise_logit",
        ),
        bounds=(
            (-5.0, 5.0),
            (-5.0, 5.0),
            (-6.0, 4.0),
            (-6.0, 4.0),
            (-6.0, 4.0),
            (-6.0, 4.0),
            (-6.0, 6.0),
            (-6.0, 6.0),
            (-6.0, 4.0),
        ),
    ),
    "no_learning": ParameterSchema(
        model="no_learning",
        theta_names=(
            "weight_i_logratio",
            "weight_e_logratio",
            "social_kappa_logit",
            "tau_logit",
            "noise_logit",
        ),
        bounds=(
            (-5.0, 5.0),
            (-5.0, 5.0),
            (-6.0, 6.0),
            (-6.0, 6.0),
            (-6.0, 4.0),
        ),
    ),
    "collapsed_reward": ParameterSchema(
        model="collapsed_reward",
        theta_names=(
            "weight_i_logit",
            "alpha_i_pos_logit",
            "alpha_i_neg_logit",
            "alpha_reward_logit",
            "social_kappa_logit",
            "tau_logit",
            "noise_logit",
        ),
        bounds=(
            (-5.0, 5.0),
            (-6.0, 4.0),
            (-6.0, 4.0),
            (-6.0, 4.0),
            (-6.0, 6.0),
            (-6.0, 6.0),
            (-6.0, 4.0),
        ),
    ),
}

CONDITIONAL_ORACLE_BLOCKS: Dict[str, Tuple[str, ...]] = {
    "decision_weights": (
        "weight_i_logratio",
        "weight_e_logratio",
    ),
    "instinct_learning": (
        "alpha_i_pos_logit",
        "alpha_i_neg_logit",
    ),
    "outcome_learning": (
        "alpha_e_logit",
        "alpha_u_logit",
    ),
    "social_influence": ("social_kappa_logit",),
    "choice_both": (
        "tau_logit",
        "noise_logit",
    ),
    "tau_only": ("tau_logit",),
    "noise_only": ("noise_logit",),
}

CONDITIONAL_ORACLE_NATURAL_PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "decision_weights": ("w_i", "w_e", "w_u"),
    "instinct_learning": ("alpha_i_pos", "alpha_i_neg"),
    "outcome_learning": ("alpha_e", "alpha_u"),
    "social_influence": ("social_kappa",),
    "choice_both": ("tau", "noise_s"),
    "tau_only": ("tau",),
    "noise_only": ("noise_s",),
}


def _default_theta(model: str) -> np.ndarray:
    if model == "tripartite":
        return np.array(
            [
                0.0,
                0.0,
                logit(0.20),
                logit(0.20),
                logit(0.20),
                logit(0.20),
                logit(0.50),
                logit((3.0 - 0.5) / 9.5),
                logit(0.10 / 0.5),
            ],
            dtype=float,
        )
    if model == "no_learning":
        return np.array(
            [
                0.0,
                0.0,
                logit(0.50),
                logit((3.0 - 0.5) / 9.5),
                logit(0.10 / 0.5),
            ],
            dtype=float,
        )
    if model == "collapsed_reward":
        return np.array(
            [
                logit(0.40),
                logit(0.20),
                logit(0.20),
                logit(0.20),
                logit(0.50),
                logit((3.0 - 0.5) / 9.5),
                logit(0.10 / 0.5),
            ],
            dtype=float,
        )
    raise ValueError(f"Unknown computational model: {model}")


def unpack_theta(model: str, theta: np.ndarray) -> Dict[str, float]:
    if model in ("tripartite", "no_learning"):
        logits = np.array([theta[0], theta[1], 0.0], dtype=float)
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()
        if model == "tripartite":
            alpha_i_pos = float(expit(theta[2]))
            alpha_i_neg = float(expit(theta[3]))
            alpha_e = float(expit(theta[4]))
            alpha_u = float(expit(theta[5]))
            offset = 6
        else:
            alpha_i_pos = 0.0
            alpha_i_neg = 0.0
            alpha_e = 0.0
            alpha_u = 0.0
            offset = 2
        return {
            "w_i": float(weights[0]),
            "w_e": float(weights[1]),
            "w_u": float(weights[2]),
            "alpha_i_pos": alpha_i_pos,
            "alpha_i_neg": alpha_i_neg,
            "alpha_e": alpha_e,
            "alpha_u": alpha_u,
            "social_kappa": float(2.0 * expit(theta[offset])),
            "tau": float(0.5 + 9.5 * expit(theta[offset + 1])),
            "noise_s": float(0.5 * expit(theta[offset + 2])),
        }
    if model == "collapsed_reward":
        w_i = float(expit(theta[0]))
        return {
            "w_i": w_i,
            "w_reward": 1.0 - w_i,
            "alpha_i_pos": float(expit(theta[1])),
            "alpha_i_neg": float(expit(theta[2])),
            "alpha_reward": float(expit(theta[3])),
            "social_kappa": float(2.0 * expit(theta[4])),
            "tau": float(0.5 + 9.5 * expit(theta[5])),
            "noise_s": float(0.5 * expit(theta[6])),
        }
    raise ValueError(f"Unknown computational model: {model}")


def natural_to_theta(model: str, params: Mapping[str, float]) -> np.ndarray:
    def bounded_logit(value: float, low: float, high: float) -> float:
        proportion = (float(value) - low) / (high - low)
        return float(logit(np.clip(proportion, 1e-5, 1.0 - 1e-5)))

    if model in ("tripartite", "no_learning"):
        w_u = max(1e-8, float(params["w_u"]))
        values = [
            np.log(max(1e-8, float(params["w_i"])) / w_u),
            np.log(max(1e-8, float(params["w_e"])) / w_u),
        ]
        if model == "tripartite":
            values.extend(
                [
                    bounded_logit(params["alpha_i_pos"], 0.0, 1.0),
                    bounded_logit(params["alpha_i_neg"], 0.0, 1.0),
                    bounded_logit(params["alpha_e"], 0.0, 1.0),
                    bounded_logit(params["alpha_u"], 0.0, 1.0),
                ]
            )
        values.extend(
            [
                bounded_logit(params["social_kappa"], 0.0, 2.0),
                bounded_logit(params["tau"], 0.5, 10.0),
                bounded_logit(params["noise_s"], 0.0, 0.5),
            ]
        )
        return np.asarray(values, dtype=float)
    if model == "collapsed_reward":
        alpha_reward = 0.5 * (float(params["alpha_e"]) + float(params["alpha_u"]))
        return np.array(
            [
                bounded_logit(params["w_i"], 0.0, 1.0),
                bounded_logit(params["alpha_i_pos"], 0.0, 1.0),
                bounded_logit(params["alpha_i_neg"], 0.0, 1.0),
                bounded_logit(alpha_reward, 0.0, 1.0),
                bounded_logit(params["social_kappa"], 0.0, 2.0),
                bounded_logit(params["tau"], 0.5, 10.0),
                bounded_logit(params["noise_s"], 0.0, 0.5),
            ],
            dtype=float,
        )
    raise ValueError(f"Unknown computational model: {model}")


def _baseline_state(person_row: pd.Series, reliability: float) -> np.ndarray:
    state = np.zeros((len(CONTEXTS), len(BEHAVIORS), len(STATE_NAMES)), dtype=float)
    for context_idx, context in enumerate(CONTEXTS):
        for behavior_idx, behavior in enumerate(BEHAVIORS):
            for state_idx, state_name in enumerate(STATE_NAMES):
                column = f"baseline_{state_name}_{context}_{behavior}"
                state[context_idx, behavior_idx, state_idx] = reliability * float(person_row[column])
    return np.clip(state, -1.0, 1.0)


def prepare_people(view: PanelView) -> List[Dict[str, object]]:
    people: List[Dict[str, object]] = []
    for _, row in view.people.sort_values("focal_id").iterrows():
        focal_id = int(row["focal_id"])
        events = view.events.loc[view.events["focal_id"] == focal_id].copy()
        events = events.sort_values(["timestamp_day", "event_id"]).reset_index(drop=True)
        model_arrays = {
            "context_idx": events["context_idx"].to_numpy(dtype=np.int8),
            "behavior_idx": events["behavior_idx"].to_numpy(dtype=np.int8),
            "role_self": (events["role"] == "self").to_numpy(dtype=bool),
            "suggestion_avoid": events["suggestion_avoid"].to_numpy(dtype=float),
            "suggestion_approach": events["suggestion_approach"].to_numpy(dtype=float),
            "relationship_receptivity": events[
                "relationship_receptivity"
            ].to_numpy(dtype=float),
            "enjoyment_out": events["enjoyment_out"].to_numpy(dtype=float),
            "utility_out": events["utility_out"].to_numpy(dtype=float),
            "eval_common": events["eval_common"].to_numpy(dtype=bool),
            "event_id": events["event_id"].to_numpy(dtype=np.int64),
        }
        people.append(
            {
                "focal_id": focal_id,
                "split": str(row["split"]),
                "person_row": row,
                "events": events,
                "model_arrays": model_arrays,
                "n_choice": int(np.sum(events["role"] == "self")) if not events.empty else 0,
            }
        )
    return people


def _marginal_choice_probability(delta: float, tau: float, noise_s: float) -> float:
    # Logistic-normal approximation for independent Gaussian option noise.
    variance = 2.0 * noise_s**2
    denominator = np.sqrt(1.0 + np.pi * tau**2 * variance / 8.0)
    return float(np.clip(expit(tau * delta / denominator), 1e-7, 1.0 - 1e-7))


def effective_choice_consistency(tau: float, noise_s: float) -> float:
    variance = 2.0 * float(noise_s) ** 2
    denominator = np.sqrt(
        1.0 + np.pi * float(tau) ** 2 * variance / 8.0
    )
    return float(float(tau) / denominator)


def _forward_person(
    theta: np.ndarray,
    model: str,
    person: Mapping[str, object],
    cfg: EmpiricalBayesConfig,
    collect_predictions: bool = False,
) -> Tuple[float, List[Dict[str, float]]]:
    params = unpack_theta(model, theta)
    state = _baseline_state(person["person_row"], cfg.baseline_reliability)
    arrays = person["model_arrays"]
    predictions: List[Dict[str, float]] = []
    nll = 0.0

    if model == "collapsed_reward":
        instinct = state[..., 0].copy()
        reward = 0.5 * (state[..., 1] + state[..., 2])

    for event_index in range(len(arrays["context_idx"])):
        context_idx = int(arrays["context_idx"][event_index])
        behavior_idx = int(arrays["behavior_idx"][event_index])
        suggestion = np.array(
            [
                float(arrays["suggestion_avoid"][event_index]),
                float(arrays["suggestion_approach"][event_index]),
            ],
            dtype=float,
        )

        if model == "collapsed_reward":
            choice_values = (
                params["w_i"] * instinct[context_idx]
                + params["w_reward"] * reward[context_idx]
                + params["social_kappa"] * suggestion
            )
        else:
            weights = np.array([params["w_i"], params["w_e"], params["w_u"]], dtype=float)
            choice_values = state[context_idx] @ weights + params["social_kappa"] * suggestion

        probability = _marginal_choice_probability(
            float(choice_values[1] - choice_values[0]),
            params["tau"],
            params["noise_s"],
        )
        if arrays["role_self"][event_index]:
            observed = int(behavior_idx)
            nll -= observed * np.log(probability) + (1 - observed) * np.log(1.0 - probability)
            if collect_predictions and bool(arrays["eval_common"][event_index]):
                predictions.append(
                    {
                        "focal_id": int(person["focal_id"]),
                        "event_id": int(arrays["event_id"][event_index]),
                        "y_true": observed,
                        "probability": probability,
                    }
                )

        if model == "no_learning":
            continue
        gain = 1.0
        if not arrays["role_self"][event_index]:
            receptivity = float(
                arrays["relationship_receptivity"][event_index]
            )
            gain = cfg.observation_attenuation * (0.5 + 0.5 * receptivity)
            gain = float(np.clip(gain, 0.0, cfg.observation_attenuation))
        other = 1 - behavior_idx
        enjoyment = float(arrays["enjoyment_out"][event_index])
        utility = float(arrays["utility_out"][event_index])

        if model == "collapsed_reward":
            instinct[context_idx, behavior_idx] += (
                gain
                * params["alpha_i_pos"]
                * (1.0 - instinct[context_idx, behavior_idx])
            )
            instinct[context_idx, other] += (
                gain
                * params["alpha_i_neg"]
                * (-1.0 - instinct[context_idx, other])
            )
            outcome = 0.5 * (enjoyment + utility)
            reward[context_idx, behavior_idx] += (
                gain
                * params["alpha_reward"]
                * (outcome - reward[context_idx, behavior_idx])
            )
            instinct[context_idx] = np.clip(instinct[context_idx], -1.0, 1.0)
            reward[context_idx] = np.clip(reward[context_idx], -1.0, 1.0)
        else:
            state[context_idx, behavior_idx, 0] += (
                gain
                * params["alpha_i_pos"]
                * (1.0 - state[context_idx, behavior_idx, 0])
            )
            state[context_idx, other, 0] += (
                gain
                * params["alpha_i_neg"]
                * (-1.0 - state[context_idx, other, 0])
            )
            state[context_idx, behavior_idx, 1] += (
                gain
                * params["alpha_e"]
                * (enjoyment - state[context_idx, behavior_idx, 1])
            )
            state[context_idx, behavior_idx, 2] += (
                gain
                * params["alpha_u"]
                * (utility - state[context_idx, behavior_idx, 2])
            )
            state[context_idx] = np.clip(state[context_idx], -1.0, 1.0)
    return float(nll), predictions


def _inverse_hessian_condition(result) -> float:
    matrix = _inverse_hessian_matrix(result)
    if matrix is None:
        return np.nan
    condition = float(np.linalg.cond(matrix))
    return condition if np.isfinite(condition) else np.nan


def _inverse_hessian_matrix(result) -> np.ndarray | None:
    try:
        matrix = np.asarray(result.hess_inv.todense(), dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
        ):
            return None
        matrix = 0.5 * (matrix + matrix.T)
        return matrix
    except Exception:
        return None


def _update_population_distribution(
    theta_matrix: np.ndarray,
    posterior_variance_matrix: np.ndarray,
    cfg: EmpiricalBayesConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(theta_matrix, axis=0)
    if cfg.variance_update == "laplace":
        centered_second_moment = np.mean(
            (theta_matrix - mu) ** 2 + posterior_variance_matrix,
            axis=0,
        )
        sigma = np.sqrt(np.maximum(centered_second_moment, 0.0))
    elif cfg.variance_update == "modal":
        empirical_sd = (
            np.std(theta_matrix, axis=0, ddof=1)
            if len(theta_matrix) > 1
            else np.zeros(len(mu))
        )
        sigma = np.sqrt(empirical_sd**2 + cfg.prior_sd_floor**2)
    else:
        raise ValueError(
            "variance_update must be either 'laplace' or 'modal'"
        )
    sigma = np.clip(
        sigma,
        cfg.prior_sd_floor,
        cfg.prior_sd_ceiling,
    )
    return mu, sigma


def fit_empirical_bayes(
    people: Sequence[Mapping[str, object]],
    model: str,
    cfg: EmpiricalBayesConfig,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, float]]:
    if model not in COMPUTATIONAL_MODELS:
        raise ValueError(f"Empirical-Bayes fitting is not defined for {model}")
    if cfg.variance_update not in {"laplace", "modal"}:
        raise ValueError(
            "variance_update must be either 'laplace' or 'modal'"
        )
    schema = SCHEMAS[model]
    train_people = [
        person
        for person in people
        if person["split"] == "train" and int(person["n_choice"]) >= 1
    ]
    if not train_people:
        raise ValueError("No training participants have an observed choice")

    mu = _default_theta(model)
    sigma = np.full(len(mu), cfg.prior_sd_initial, dtype=float)
    rng = np.random.default_rng(stable_seed(cfg.seed, model, "empirical_bayes"))
    final_rows: List[Dict[str, float]] = []

    for iteration in range(cfg.eb_iterations):
        theta_rows = []
        posterior_variance_rows = []
        iteration_rows: List[Dict[str, float]] = []
        for person in train_people:
            starts = [mu.copy()]
            for _ in range(max(0, cfg.multistarts - 1)):
                starts.append(
                    np.clip(
                        mu + rng.normal(0.0, 0.30 * sigma),
                        [bound[0] for bound in schema.bounds],
                        [bound[1] for bound in schema.bounds],
                    )
                )
            solutions = []
            for start_idx, start in enumerate(starts):
                def objective(theta: np.ndarray) -> float:
                    data_nll, _ = _forward_person(theta, model, person, cfg, collect_predictions=False)
                    prior = 0.5 * np.sum(((theta - mu) / sigma) ** 2)
                    return float(data_nll + prior)

                result = minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=schema.bounds,
                    options={"maxiter": cfg.max_iter, "ftol": 1e-9},
                )
                solutions.append((float(result.fun), start_idx, result))
            solutions.sort(key=lambda item: item[0])
            best_value, best_start, best = solutions[0]
            converged_solutions = [
                result for _, _, result in solutions if bool(result.success)
            ]
            comparison_solutions = (
                converged_solutions
                if len(converged_solutions) >= 2
                else [result for _, _, result in solutions]
            )
            solution_matrix = np.vstack(
                [np.asarray(result.x, dtype=float) for result in comparison_solutions]
            )
            theta_max_range = (
                float(np.max(np.ptp(solution_matrix, axis=0)))
                if len(solution_matrix) >= 2
                else 0.0
            )
            best_second_distance = (
                float(
                    np.linalg.norm(
                        np.asarray(solutions[0][2].x, dtype=float)
                        - np.asarray(solutions[1][2].x, dtype=float)
                    )
                )
                if len(solutions) >= 2
                else 0.0
            )
            theta_rows.append(np.asarray(best.x, dtype=float))
            inverse_hessian = _inverse_hessian_matrix(best)
            if inverse_hessian is None:
                posterior_variance = np.zeros(len(mu), dtype=float)
            else:
                posterior_variance = np.clip(
                    np.diag(inverse_hessian),
                    0.0,
                    cfg.prior_sd_ceiling**2,
                )
            posterior_variance_rows.append(posterior_variance)
            natural = unpack_theta(model, np.asarray(best.x, dtype=float))
            row: Dict[str, float] = {
                "focal_id": int(person["focal_id"]),
                "iteration": iteration,
                "n_choice": int(person["n_choice"]),
                "objective": best_value,
                "optimizer_success": float(bool(best.success)),
                "optimizer_status": float(best.status),
                "best_start": float(best_start),
                "multistart_objective_range": float(
                    max(value for value, _, _ in solutions)
                    - min(value for value, _, _ in solutions)
                ),
                "multistart_theta_max_range": theta_max_range,
                "best_second_theta_distance": best_second_distance,
                "converged_start_fraction": float(
                    np.mean([bool(result.success) for _, _, result in solutions])
                ),
                "inverse_hessian_condition": _inverse_hessian_condition(best),
            }
            for name, value in zip(schema.theta_names, best.x):
                row[f"theta_{name}"] = float(value)
            for name, value in zip(schema.theta_names, posterior_variance):
                row[f"posterior_variance_{name}"] = float(value)
            for name, value in natural.items():
                row[f"fit_{name}"] = float(value)
            iteration_rows.append(row)

        theta_matrix = np.vstack(theta_rows)
        posterior_variance_matrix = np.vstack(posterior_variance_rows)
        mu, sigma = _update_population_distribution(
            theta_matrix,
            posterior_variance_matrix,
            cfg,
        )
        final_rows = iteration_rows

    person_frame = pd.DataFrame(final_rows)
    diagnostics = {
        "training_people_total": float(sum(person["split"] == "train" for person in people)),
        "training_people_with_choice": float(len(train_people)),
        "optimizer_success_rate": float(person_frame["optimizer_success"].mean()),
        "mean_multistart_objective_range": float(person_frame["multistart_objective_range"].mean()),
        "median_multistart_theta_max_range": float(
            person_frame["multistart_theta_max_range"].median()
        ),
        "p90_multistart_theta_max_range": float(
            person_frame["multistart_theta_max_range"].quantile(0.90)
        ),
        "median_best_second_theta_distance": float(
            person_frame["best_second_theta_distance"].median()
        ),
        "mean_converged_start_fraction": float(
            person_frame["converged_start_fraction"].mean()
        ),
        "median_inverse_hessian_condition": float(person_frame["inverse_hessian_condition"].median()),
        "mean_posterior_variance": float(
            person_frame[
                [
                    column
                    for column in person_frame
                    if column.startswith("posterior_variance_")
                ]
            ]
            .to_numpy(dtype=float)
            .mean()
        ),
        "eb_iterations": float(cfg.eb_iterations),
        "multistarts": float(cfg.multistarts),
        "variance_update_laplace": float(cfg.variance_update == "laplace"),
    }
    return mu, sigma, person_frame, diagnostics


def fit_conditional_oracle(
    view: PanelView,
    model: str,
    block: str,
    cfg: EmpiricalBayesConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if model != "tripartite":
        raise ValueError("Conditional oracle blocks are defined for tripartite fits")
    if block not in CONDITIONAL_ORACLE_BLOCKS:
        raise ValueError(f"Unknown conditional oracle block: {block}")
    if cfg.variance_update not in {"laplace", "modal"}:
        raise ValueError("variance_update must be either 'laplace' or 'modal'")

    schema = SCHEMAS[model]
    free_names = CONDITIONAL_ORACLE_BLOCKS[block]
    free_indices = np.asarray(
        [schema.theta_names.index(name) for name in free_names],
        dtype=int,
    )
    free_bounds = tuple(schema.bounds[index] for index in free_indices)
    people = prepare_people(view)
    train_people = [
        person
        for person in people
        if person["split"] == "train" and int(person["n_choice"]) >= 1
    ]
    if not train_people:
        raise ValueError("No training participants have an observed choice")

    truth_natural = _true_parameter_table(view, model).set_index("focal_id")
    true_theta_by_id: Dict[int, np.ndarray] = {}
    for focal_id, row in truth_natural.iterrows():
        natural = {
            column.removeprefix("true_"): float(value)
            for column, value in row.items()
            if column.startswith("true_")
        }
        true_theta_by_id[int(focal_id)] = natural_to_theta(model, natural)

    population_theta = _default_theta(model)
    sigma_free = np.full(
        len(free_indices),
        cfg.prior_sd_initial,
        dtype=float,
    )
    rng = np.random.default_rng(
        stable_seed(cfg.seed, model, block, "conditional_oracle")
    )
    final_rows: List[Dict[str, float]] = []

    for iteration in range(cfg.eb_iterations):
        active_rows = []
        posterior_variance_rows = []
        iteration_rows: List[Dict[str, float]] = []
        for person in train_people:
            focal_id = int(person["focal_id"])
            fixed_theta = true_theta_by_id[focal_id].copy()
            starts = [population_theta[free_indices].copy()]
            for _ in range(max(0, cfg.multistarts - 1)):
                starts.append(
                    np.clip(
                        population_theta[free_indices]
                        + rng.normal(0.0, 0.30 * sigma_free),
                        [bound[0] for bound in free_bounds],
                        [bound[1] for bound in free_bounds],
                    )
                )

            solutions = []
            for start_idx, start in enumerate(starts):

                def objective(active_theta: np.ndarray) -> float:
                    theta = fixed_theta.copy()
                    theta[free_indices] = active_theta
                    data_nll, _ = _forward_person(
                        theta,
                        model,
                        person,
                        cfg,
                        collect_predictions=False,
                    )
                    prior = 0.5 * np.sum(
                        (
                            (
                                active_theta
                                - population_theta[free_indices]
                            )
                            / sigma_free
                        )
                        ** 2
                    )
                    return float(data_nll + prior)

                result = minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=free_bounds,
                    options={"maxiter": cfg.max_iter, "ftol": 1e-9},
                )
                solutions.append((float(result.fun), start_idx, result))

            solutions.sort(key=lambda item: item[0])
            best_value, best_start, best = solutions[0]
            active_theta = np.asarray(best.x, dtype=float)
            fitted_theta = fixed_theta.copy()
            fitted_theta[free_indices] = active_theta
            active_rows.append(active_theta)

            inverse_hessian = _inverse_hessian_matrix(best)
            if inverse_hessian is None:
                posterior_variance = np.zeros(
                    len(free_indices),
                    dtype=float,
                )
            else:
                posterior_variance = np.clip(
                    np.diag(inverse_hessian),
                    0.0,
                    cfg.prior_sd_ceiling**2,
                )
            posterior_variance_rows.append(posterior_variance)

            solution_matrix = np.vstack(
                [
                    np.asarray(result.x, dtype=float)
                    for _, _, result in solutions
                ]
            )
            natural = unpack_theta(model, fitted_theta)
            row: Dict[str, float] = {
                "focal_id": focal_id,
                "iteration": float(iteration),
                "n_choice": float(person["n_choice"]),
                "objective": best_value,
                "optimizer_success": float(bool(best.success)),
                "optimizer_status": float(best.status),
                "best_start": float(best_start),
                "multistart_objective_range": float(
                    max(value for value, _, _ in solutions)
                    - min(value for value, _, _ in solutions)
                ),
                "multistart_theta_max_range": (
                    float(np.max(np.ptp(solution_matrix, axis=0)))
                    if len(solution_matrix) >= 2
                    else 0.0
                ),
                "inverse_hessian_condition": _inverse_hessian_condition(best),
            }
            for name, value in zip(schema.theta_names, fitted_theta):
                row[f"theta_{name}"] = float(value)
            for name, value in natural.items():
                row[f"fit_{name}"] = float(value)
            iteration_rows.append(row)

        active_matrix = np.vstack(active_rows)
        posterior_variance_matrix = np.vstack(posterior_variance_rows)
        active_mu, sigma_free = _update_population_distribution(
            active_matrix,
            posterior_variance_matrix,
            cfg,
        )
        population_theta[free_indices] = active_mu
        final_rows = iteration_rows

    person_parameters = pd.DataFrame(final_rows)
    recovery = summarize_recovery(
        view,
        model,
        person_parameters,
        population_theta,
        cfg.baseline_reliability,
    )
    target_parameters = set(CONDITIONAL_ORACLE_NATURAL_PARAMETERS[block])
    block_recovery = recovery.loc[
        (recovery["level"] == "person")
        & recovery["parameter"].isin(target_parameters)
    ].copy()
    diagnostics = {
        "training_people_total": float(
            sum(person["split"] == "train" for person in people)
        ),
        "training_people_with_choice": float(len(train_people)),
        "optimizer_success_rate": float(
            person_parameters["optimizer_success"].mean()
        ),
        "median_multistart_theta_max_range": float(
            person_parameters["multistart_theta_max_range"].median()
        ),
        "median_inverse_hessian_condition": float(
            person_parameters["inverse_hessian_condition"].median()
        ),
        "mean_correlation": float(block_recovery["correlation"].mean()),
        "mean_rmse": float(block_recovery["rmse"].mean()),
        "eb_iterations": float(cfg.eb_iterations),
        "multistarts": float(cfg.multistarts),
        "free_theta_count": float(len(free_indices)),
    }
    return person_parameters, block_recovery, diagnostics


def summarize_population_recovery(
    view: PanelView,
    model: str,
    population_theta: np.ndarray,
) -> pd.DataFrame:
    truth = _true_parameter_table(view, model)
    training_ids = set(
        view.people.loc[
            view.people["split"] == "train",
            "focal_id",
        ].astype(int)
    )
    truth = truth.loc[truth["focal_id"].astype(int).isin(training_ids)].copy()
    fitted = unpack_theta(model, population_theta)

    true_tau = truth["true_tau"].to_numpy(dtype=float)
    true_noise = truth["true_noise_s"].to_numpy(dtype=float)
    truth["true_choice_consistency"] = [
        effective_choice_consistency(tau, noise)
        for tau, noise in zip(true_tau, true_noise)
    ]
    fitted["choice_consistency"] = effective_choice_consistency(
        fitted["tau"],
        fitted["noise_s"],
    )

    if model == "tripartite":
        parameters = (
            "w_i",
            "w_e",
            "w_u",
            "alpha_i_pos",
            "alpha_i_neg",
            "alpha_e",
            "alpha_u",
            "social_kappa",
            "choice_consistency",
        )
    elif model == "no_learning":
        parameters = (
            "w_i",
            "w_e",
            "w_u",
            "social_kappa",
            "choice_consistency",
        )
    elif model == "collapsed_reward":
        parameters = (
            "w_i",
            "w_reward",
            "alpha_i_pos",
            "alpha_i_neg",
            "alpha_reward",
            "social_kappa",
            "choice_consistency",
        )
    else:
        raise ValueError(f"Population recovery is not defined for {model}")

    rows = []
    for parameter in parameters:
        true_values = truth[f"true_{parameter}"].to_numpy(dtype=float)
        estimate = float(fitted[parameter])
        true_mean = float(np.mean(true_values))
        rows.append(
            {
                "level": "population",
                "parameter": parameter,
                "n": float(len(true_values)),
                "correlation": np.nan,
                "bias": estimate - true_mean,
                "rmse": abs(estimate - true_mean),
                "true_mean": true_mean,
                "estimate": estimate,
            }
        )
    return pd.DataFrame(rows)


def fit_population_model(
    people: Sequence[Mapping[str, object]],
    model: str,
    cfg: EmpiricalBayesConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if model not in COMPUTATIONAL_MODELS:
        raise ValueError(f"Population fitting is not defined for {model}")
    schema = SCHEMAS[model]
    train_people = [
        person
        for person in people
        if person["split"] == "train" and int(person["n_choice"]) >= 1
    ]
    if not train_people:
        raise ValueError("No training participants have an observed choice")

    noise_index = schema.theta_names.index("noise_logit")
    free_indices = np.asarray(
        [
            index
            for index in range(len(schema.theta_names))
            if index != noise_index
        ],
        dtype=int,
    )
    fixed_theta = _default_theta(model)
    fixed_theta[noise_index] = schema.bounds[noise_index][0]
    free_bounds = tuple(schema.bounds[index] for index in free_indices)
    rng = np.random.default_rng(
        stable_seed(cfg.seed, model, "population_fit")
    )
    starts = [fixed_theta[free_indices].copy()]
    for _ in range(max(0, cfg.multistarts - 1)):
        starts.append(
            np.clip(
                fixed_theta[free_indices] + rng.normal(0.0, 0.35, len(free_indices)),
                [bound[0] for bound in free_bounds],
                [bound[1] for bound in free_bounds],
            )
        )

    solutions = []
    for start_idx, start in enumerate(starts):

        def objective(active_theta: np.ndarray) -> float:
            theta = fixed_theta.copy()
            theta[free_indices] = active_theta
            return float(
                sum(
                    _forward_person(
                        theta,
                        model,
                        person,
                        cfg,
                        collect_predictions=False,
                    )[0]
                    for person in train_people
                )
            )

        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=free_bounds,
            options={"maxiter": cfg.max_iter, "ftol": 1e-9},
        )
        solutions.append((float(result.fun), start_idx, result))

    solutions.sort(key=lambda item: item[0])
    best_value, best_start, best = solutions[0]
    population_theta = fixed_theta.copy()
    population_theta[free_indices] = np.asarray(best.x, dtype=float)
    solution_matrix = np.vstack(
        [np.asarray(result.x, dtype=float) for _, _, result in solutions]
    )
    diagnostics = {
        "training_people_total": float(
            sum(person["split"] == "train" for person in people)
        ),
        "training_people_with_choice": float(len(train_people)),
        "optimizer_success_rate": float(bool(best.success)),
        "optimizer_status": float(best.status),
        "objective": best_value,
        "best_start": float(best_start),
        "multistart_objective_range": float(
            max(value for value, _, _ in solutions)
            - min(value for value, _, _ in solutions)
        ),
        "multistart_theta_max_range": (
            float(np.max(np.ptp(solution_matrix, axis=0)))
            if len(solution_matrix) >= 2
            else 0.0
        ),
        "inverse_hessian_condition": _inverse_hessian_condition(best),
        "multistarts": float(cfg.multistarts),
        "free_theta_count": float(len(free_indices)),
        "latent_value_noise_fixed": 1.0,
    }
    return population_theta, diagnostics


def _predict_computational_test(
    people: Sequence[Mapping[str, object]],
    model: str,
    population_theta: np.ndarray,
    cfg: EmpiricalBayesConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for person in people:
        if person["split"] != "test":
            continue
        _, predictions = _forward_person(
            population_theta,
            model,
            person,
            cfg,
            collect_predictions=True,
        )
        rows.extend(predictions)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["model"] = model
    return frame


def _lagged_features(person: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    events = person["events"]
    features = []
    outcomes = []
    metadata: List[Dict[str, int]] = []
    prior_self_choice = 0.5
    prior_behavior = 0.5
    prior_enjoyment = 0.0
    prior_utility = 0.0
    event_counter = 0
    for _, event in events.iterrows():
        context_idx = int(event["context_idx"])
        context_one_hot = np.zeros(len(CONTEXTS), dtype=float)
        context_one_hot[context_idx] = 1.0
        elapsed = float(event["elapsed_since_prior_observed"])
        if not np.isfinite(elapsed):
            elapsed = 0.0
        x = np.concatenate(
            [
                context_one_hot,
                np.array(
                    [
                        float(event["suggestion_approach"] - event["suggestion_avoid"]),
                        float(event["relationship_warmth"]),
                        float(event["relationship_receptivity"]),
                        np.log1p(max(0.0, elapsed)),
                        prior_self_choice,
                        prior_behavior,
                        prior_enjoyment,
                        prior_utility,
                        np.log1p(event_counter),
                    ]
                ),
            ]
        )
        if event["role"] == "self":
            features.append(x)
            outcomes.append(int(event["behavior_idx"]))
            metadata.append(
                {
                    "focal_id": int(person["focal_id"]),
                    "event_id": int(event["event_id"]),
                    "eval_common": int(bool(event["eval_common"])),
                }
            )
            prior_self_choice = float(event["behavior_idx"])
        prior_behavior = float(event["behavior_idx"])
        prior_enjoyment = float(event["enjoyment_out"])
        prior_utility = float(event["utility_out"])
        event_counter += 1
    if features:
        return np.vstack(features), np.asarray(outcomes, dtype=int), metadata
    return np.empty((0, len(CONTEXTS) + 9)), np.empty(0, dtype=int), metadata


def fit_lagged_model(
    people: Sequence[Mapping[str, object]],
    cfg: EmpiricalBayesConfig,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    train_x = []
    train_y = []
    for person in people:
        if person["split"] != "train":
            continue
        x, y, _ = _lagged_features(person)
        if len(y):
            train_x.append(x)
            train_y.append(y)
    if not train_y:
        raise ValueError("No training choices are available for the lagged model")
    x_train = np.vstack(train_x)
    y_train = np.concatenate(train_y)
    prevalence = float(np.mean(y_train))

    classifier = None
    if len(np.unique(y_train)) >= 2:
        classifier = LogisticRegression(
            C=cfg.lagged_c,
            penalty="l2",
            max_iter=max(200, cfg.max_iter),
            random_state=cfg.seed,
        )
        classifier.fit(x_train, y_train)

    prediction_rows: List[Dict[str, float]] = []
    for person in people:
        if person["split"] != "test":
            continue
        x, y, metadata = _lagged_features(person)
        if not len(y):
            continue
        if classifier is None:
            probabilities = np.full(len(y), prevalence)
        else:
            probabilities = classifier.predict_proba(x)[:, 1]
        for outcome, probability, meta in zip(y, probabilities, metadata):
            if not meta["eval_common"]:
                continue
            prediction_rows.append(
                {
                    "focal_id": meta["focal_id"],
                    "event_id": meta["event_id"],
                    "y_true": int(outcome),
                    "probability": float(probability),
                    "model": "lagged",
                }
            )
    diagnostics = {
        "training_people_total": float(sum(person["split"] == "train" for person in people)),
        "training_choice_count": float(len(y_train)),
        "training_approach_rate": prevalence,
        "optimizer_success_rate": 1.0,
        "coefficient_count": float(x_train.shape[1] + 1),
    }
    population = {"training_approach_rate": prevalence}
    if classifier is not None:
        population["intercept"] = float(classifier.intercept_[0])
        for idx, value in enumerate(classifier.coef_[0]):
            population[f"coefficient_{idx}"] = float(value)
    return pd.DataFrame(prediction_rows), diagnostics, population


def null_predictions(people: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    train_outcomes = []
    for person in people:
        if person["split"] != "train":
            continue
        events = person["events"]
        train_outcomes.extend(events.loc[events["role"] == "self", "behavior_idx"].astype(int).tolist())
    prevalence = float(np.mean(train_outcomes)) if train_outcomes else 0.5
    rows = []
    for person in people:
        if person["split"] != "test":
            continue
        events = person["events"]
        targets = events.loc[(events["role"] == "self") & events["eval_common"]]
        for _, event in targets.iterrows():
            rows.append(
                {
                    "focal_id": int(person["focal_id"]),
                    "event_id": int(event["event_id"]),
                    "y_true": int(event["behavior_idx"]),
                    "probability": prevalence,
                    "model": "null",
                }
            )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame) -> Dict[str, float]:
    empty = {
        "n_predictions": 0.0,
        "approach_rate": np.nan,
        "log_loss": np.nan,
        "brier": np.nan,
        "auc": np.nan,
        "pr_auc": np.nan,
        "balanced_accuracy": np.nan,
        "mcc": np.nan,
        "ece": np.nan,
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
    }
    if predictions.empty:
        return empty
    y = predictions["y_true"].to_numpy(dtype=int)
    probability = np.clip(predictions["probability"].to_numpy(dtype=float), 1e-7, 1.0 - 1e-7)
    predicted = (probability >= 0.5).astype(int)
    metrics = dict(empty)
    metrics.update(
        {
            "n_predictions": float(len(y)),
            "approach_rate": float(np.mean(y)),
            "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
            "brier": float(brier_score_loss(y, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "mcc": float(matthews_corrcoef(y, predicted)) if len(np.unique(y)) >= 2 else np.nan,
            "ece": _expected_calibration_error(y, probability),
        }
    )
    if len(np.unique(y)) >= 2:
        metrics["auc"] = float(roc_auc_score(y, probability))
        metrics["pr_auc"] = float(average_precision_score(y, probability))
        calibration = LogisticRegression(C=1e6, max_iter=500)
        calibration.fit(np.log(probability / (1.0 - probability)).reshape(-1, 1), y)
        metrics["calibration_intercept"] = float(calibration.intercept_[0])
        metrics["calibration_slope"] = float(calibration.coef_[0, 0])
    return metrics


def _expected_calibration_error(
    y: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (probability >= edges[idx]) & (probability <= edges[idx + 1])
        else:
            mask = (probability >= edges[idx]) & (probability < edges[idx + 1])
        if not np.any(mask):
            continue
        value += np.mean(mask) * abs(float(np.mean(y[mask])) - float(np.mean(probability[mask])))
    return float(value) if total else np.nan


def _true_parameter_table(
    view: PanelView,
    model: str,
) -> pd.DataFrame:
    rows = []
    for _, row in view.people_truth.iterrows():
        params = {
            "w_i": float(row["true_w_i"]),
            "w_e": float(row["true_w_e"]),
            "w_u": float(row["true_w_u"]),
            "alpha_i_pos": float(row["true_alpha_i_pos"]),
            "alpha_i_neg": float(row["true_alpha_i_neg"]),
            "alpha_e": float(row["true_alpha_e"]),
            "alpha_u": float(row["true_alpha_u"]),
            "social_kappa": float(row["true_social_kappa"]),
            "tau": float(row["true_tau"]),
            "noise_s": float(row["true_noise_s"]),
        }
        natural = unpack_theta(model, natural_to_theta(model, params))
        rows.append({"focal_id": int(row["focal_id"]), **{f"true_{k}": v for k, v in natural.items()}})
    return pd.DataFrame(rows)


def summarize_recovery(
    view: PanelView,
    model: str,
    person_parameters: pd.DataFrame,
    population_theta: np.ndarray,
    baseline_reliability: float,
) -> pd.DataFrame:
    if person_parameters.empty:
        return pd.DataFrame()
    truth = _true_parameter_table(view, model)
    merged = person_parameters.merge(truth, on="focal_id", how="inner")
    natural_population = unpack_theta(model, population_theta)
    parameter_names = [
        name
        for name in natural_population
        if f"fit_{name}" in merged and f"true_{name}" in merged
    ]
    rows = []
    for name in parameter_names:
        true = merged[f"true_{name}"].to_numpy(dtype=float)
        fitted = merged[f"fit_{name}"].to_numpy(dtype=float)
        correlation = np.nan
        if len(true) >= 3 and np.std(true) > 1e-10 and np.std(fitted) > 1e-10:
            correlation = float(np.corrcoef(true, fitted)[0, 1])
        rows.append(
            {
                "level": "person",
                "parameter": name,
                "n": float(len(true)),
                "correlation": correlation,
                "bias": float(np.mean(fitted - true)),
                "rmse": float(np.sqrt(np.mean((fitted - true) ** 2))),
                "true_mean": float(np.mean(true)),
                "estimate": float(np.mean(fitted)),
            }
        )
        rows.append(
            {
                "level": "population",
                "parameter": name,
                "n": float(len(true)),
                "correlation": np.nan,
                "bias": float(natural_population[name] - np.mean(true)),
                "rmse": abs(float(natural_population[name] - np.mean(true))),
                "true_mean": float(np.mean(true)),
                "estimate": float(natural_population[name]),
            }
        )
        rows.append(
            {
                "level": "population_dispersion",
                "parameter": name,
                "n": float(len(true)),
                "correlation": np.nan,
                "bias": float(np.std(fitted, ddof=1) - np.std(true, ddof=1)),
                "rmse": abs(float(np.std(fitted, ddof=1) - np.std(true, ddof=1))),
                "true_mean": float(np.std(true, ddof=1)),
                "estimate": float(np.std(fitted, ddof=1)),
            }
        )

    fitted_ids = set(person_parameters["focal_id"].astype(int))
    observed_people = view.people.loc[
        view.people["focal_id"].astype(int).isin(fitted_ids)
    ].set_index("focal_id")
    true_people = view.people_truth.loc[
        view.people_truth["focal_id"].astype(int).isin(fitted_ids)
    ].set_index("focal_id")
    for state_name in STATE_NAMES:
        true_values = []
        estimated_values = []
        for context in CONTEXTS:
            for behavior in BEHAVIORS:
                true_column = f"initial_{state_name}_{context}_{behavior}"
                observed_column = f"baseline_{state_name}_{context}_{behavior}"
                common_ids = observed_people.index.intersection(true_people.index)
                true_values.extend(
                    true_people.loc[common_ids, true_column].to_numpy(dtype=float)
                )
                estimated_values.extend(
                    baseline_reliability
                    * observed_people.loc[
                        common_ids,
                        observed_column,
                    ].to_numpy(dtype=float)
                )
        true_array = np.asarray(true_values, dtype=float)
        fitted_array = np.asarray(estimated_values, dtype=float)
        if len(true_array):
            rows.append(
                {
                    "level": "initial_state",
                    "parameter": f"initial_{state_name}",
                    "n": float(len(true_array)),
                    "correlation": (
                        float(np.corrcoef(true_array, fitted_array)[0, 1])
                        if np.std(true_array) > 1e-10
                        and np.std(fitted_array) > 1e-10
                        else np.nan
                    ),
                    "bias": float(np.mean(fitted_array - true_array)),
                    "rmse": float(
                        np.sqrt(np.mean((fitted_array - true_array) ** 2))
                    ),
                    "true_mean": float(np.mean(true_array)),
                    "estimate": float(np.mean(fitted_array)),
                }
            )
    return pd.DataFrame(rows)


def fit_and_evaluate_model(
    view: PanelView,
    model: str,
    cfg: EmpiricalBayesConfig,
    estimator: str = "empirical_bayes",
) -> ModelFitResult:
    people = prepare_people(view)
    if model == "lagged":
        predictions, diagnostics, population = fit_lagged_model(people, cfg)
        return ModelFitResult(
            model=model,
            metrics=evaluate_predictions(predictions),
            diagnostics=diagnostics,
            predictions=predictions,
            person_parameters=pd.DataFrame(),
            recovery=pd.DataFrame(),
            population_parameters=population,
        )
    if model not in COMPUTATIONAL_MODELS:
        raise ValueError(f"Unknown candidate model: {model}")

    if estimator == "population":
        population_theta, diagnostics = fit_population_model(
            people,
            model,
            cfg,
        )
        predictions = _predict_computational_test(
            people,
            model,
            population_theta,
            cfg,
        )
        recovery = summarize_population_recovery(
            view,
            model,
            population_theta,
        )
        population = unpack_theta(model, population_theta)
        population["choice_consistency"] = effective_choice_consistency(
            population.pop("tau"),
            population.pop("noise_s"),
        )
        return ModelFitResult(
            model=model,
            metrics=evaluate_predictions(predictions),
            diagnostics=diagnostics,
            predictions=predictions,
            person_parameters=pd.DataFrame(),
            recovery=recovery,
            population_parameters=population,
        )
    if estimator != "empirical_bayes":
        raise ValueError(
            "estimator must be either 'empirical_bayes' or 'population'"
        )

    mu, sigma, person_parameters, diagnostics = fit_empirical_bayes(people, model, cfg)
    predictions = _predict_computational_test(people, model, mu, cfg)
    recovery = summarize_recovery(
        view,
        model,
        person_parameters,
        mu,
        cfg.baseline_reliability,
    )
    population = unpack_theta(model, mu)
    for name, value in zip(SCHEMAS[model].theta_names, sigma):
        population[f"theta_sd_{name}"] = float(value)
    return ModelFitResult(
        model=model,
        metrics=evaluate_predictions(predictions),
        diagnostics=diagnostics,
        predictions=predictions,
        person_parameters=person_parameters,
        recovery=recovery,
        population_parameters=population,
    )


def assert_prediction_targets_match(
    results: Sequence[ModelFitResult],
) -> None:
    target_sets = []
    for result in results:
        target_sets.append(
            set(
                zip(
                    result.predictions.get("focal_id", pd.Series(dtype=int)).astype(int),
                    result.predictions.get("event_id", pd.Series(dtype=int)).astype(int),
                )
            )
        )
    if target_sets and any(target != target_sets[0] for target in target_sets[1:]):
        counts = [len(target) for target in target_sets]
        raise AssertionError(f"Candidate models were scored on different targets: {counts}")


def flatten_result(
    result: ModelFitResult,
    sample_size: int,
    missing_rate: float,
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "model": result.model,
        "sample_size": float(sample_size),
        "missing_rate": float(missing_rate),
    }
    row.update({f"metric_{name}": value for name, value in result.metrics.items()})
    row.update({f"diagnostic_{name}": value for name, value in result.diagnostics.items()})
    if not result.recovery.empty:
        person = result.recovery.loc[result.recovery["level"] == "person"]
        population = result.recovery.loc[result.recovery["level"] == "population"]
        dispersion = result.recovery.loc[
            result.recovery["level"] == "population_dispersion"
        ]
        initial = result.recovery.loc[result.recovery["level"] == "initial_state"]
        row["recovery_mean_correlation"] = float(person["correlation"].mean())
        row["recovery_mean_rmse"] = float(person["rmse"].mean())
        row["population_mean_abs_bias"] = float(population["bias"].abs().mean())
        row["population_dispersion_mean_abs_bias"] = float(
            dispersion["bias"].abs().mean()
        )
        row["initial_state_mean_correlation"] = float(
            initial["correlation"].mean()
        )
        row["initial_state_mean_rmse"] = float(initial["rmse"].mean())
        parameter_classes = {
            "decision_weight": {"w_i", "w_e", "w_u", "w_reward"},
            "learning_rate": {
                "alpha_i_pos",
                "alpha_i_neg",
                "alpha_e",
                "alpha_u",
                "alpha_reward",
            },
            "social_influence": {"social_kappa"},
            "choice_consistency": {"tau", "noise_s"},
        }
        for class_name, parameters in parameter_classes.items():
            subset = person.loc[person["parameter"].isin(parameters)]
            row[f"{class_name}_mean_correlation"] = float(
                subset["correlation"].mean()
            )
            row[f"{class_name}_mean_rmse"] = float(subset["rmse"].mean())
    else:
        row["recovery_mean_correlation"] = np.nan
        row["recovery_mean_rmse"] = np.nan
        row["population_mean_abs_bias"] = np.nan
        row["population_dispersion_mean_abs_bias"] = np.nan
        row["initial_state_mean_correlation"] = np.nan
        row["initial_state_mean_rmse"] = np.nan
        for class_name in (
            "decision_weight",
            "learning_rate",
            "social_influence",
            "choice_consistency",
        ):
            row[f"{class_name}_mean_correlation"] = np.nan
            row[f"{class_name}_mean_rmse"] = np.nan
    return row
