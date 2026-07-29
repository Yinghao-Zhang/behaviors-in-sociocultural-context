"""Shared social-influence equations for behavioral selection and learning."""

from __future__ import annotations

from typing import Union

import numpy as np


Numeric = Union[float, np.ndarray]


def _bounded_weight(value: float, name: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return weight


def qualify_social_signal(signal: Numeric, receptivity: float) -> Numeric:
    """Qualify a raw social signal by relationship-specific receptivity."""
    receptivity_value = float(receptivity)
    if not np.isfinite(receptivity_value) or not -1.0 <= receptivity_value <= 1.0:
        raise ValueError(
            "receptivity must be finite and in [-1, 1], "
            f"got {receptivity!r}"
        )
    qualified = np.asarray(signal, dtype=float) * receptivity_value
    if qualified.ndim == 0:
        return float(qualified)
    return qualified


def observational_learning_gain(observer_penalty: float) -> float:
    """Convert an observational-learning penalty into an update multiplier."""
    return 1.0 - _bounded_weight(observer_penalty, "observer_penalty")


def blend_personal_and_social(
    personal_value: Numeric,
    qualified_social_value: Numeric,
    social_weight: float,
    *,
    social_input_present: bool = True,
) -> Numeric:
    """Blend a personal value with an already qualified social value."""
    personal = np.asarray(personal_value, dtype=float)
    if not social_input_present:
        result = personal.copy()
    else:
        weight = _bounded_weight(social_weight, "social_weight")
        social = np.asarray(qualified_social_value, dtype=float)
        result = (1.0 - weight) * personal + weight * social
    if result.ndim == 0:
        return float(result)
    return result


def integrate_suggestion(
    personal_intention: Numeric,
    raw_suggestion: Numeric,
    receptivity: float,
    suggestion_weight: float,
    *,
    suggestion_present: bool = True,
) -> Numeric:
    """Integrate relationship-qualified suggestion with personal intention."""
    if not suggestion_present:
        personal = np.asarray(personal_intention, dtype=float)
        return float(personal) if personal.ndim == 0 else personal.copy()
    qualified = qualify_social_signal(raw_suggestion, receptivity)
    return blend_personal_and_social(
        personal_intention,
        qualified,
        suggestion_weight,
    )


def integrate_feedback(
    direct_utility: Numeric,
    raw_feedback: Numeric,
    receptivity: float,
    feedback_weight: float,
    *,
    feedback_present: bool = True,
) -> Numeric:
    """Integrate relationship-qualified feedback with directly experienced utility."""
    if not feedback_present:
        direct = np.asarray(direct_utility, dtype=float)
        return float(direct) if direct.ndim == 0 else direct.copy()
    qualified = qualify_social_signal(raw_feedback, receptivity)
    return blend_personal_and_social(
        direct_utility,
        qualified,
        feedback_weight,
    )
