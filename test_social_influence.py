import unittest

import numpy as np

from agent import Agent, Individual
from behavior import Behavior
from setup import Setup
from social_influence import (
    integrate_feedback,
    integrate_suggestion,
    observational_learning_gain,
    qualify_social_signal,
)


class SocialInfluenceTests(unittest.TestCase):
    def setUp(self):
        Agent._registry = {}
        Agent._debug_log = []
        Behavior._registry = {}
        Setup._registry = {}

    def test_receptivity_qualifies_social_signal(self):
        signal = np.array([-0.4, 0.4])
        np.testing.assert_allclose(
            qualify_social_signal(signal, 0.5),
            np.array([-0.2, 0.2]),
        )

    def test_suggestion_weight_blends_intention_and_suggestion(self):
        result = integrate_suggestion(
            np.array([-0.1, 0.3]),
            np.array([-0.4, 0.4]),
            receptivity=0.5,
            suggestion_weight=0.25,
        )
        np.testing.assert_allclose(result, np.array([-0.125, 0.275]))

    def test_absent_suggestion_leaves_intention_unchanged(self):
        intention = np.array([-0.1, 0.3])
        result = integrate_suggestion(
            intention,
            np.array([1.0, -1.0]),
            receptivity=1.0,
            suggestion_weight=0.9,
            suggestion_present=False,
        )
        np.testing.assert_allclose(result, intention)

    def test_feedback_weight_can_differ_from_suggestion_weight(self):
        result = integrate_feedback(
            direct_utility=0.4,
            raw_feedback=-0.6,
            receptivity=0.5,
            feedback_weight=0.75,
        )
        self.assertAlmostEqual(result, -0.125)

    def test_absent_feedback_leaves_direct_utility_unchanged(self):
        result = integrate_feedback(
            direct_utility=0.4,
            raw_feedback=-0.6,
            receptivity=0.5,
            feedback_weight=0.75,
            feedback_present=False,
        )
        self.assertAlmostEqual(result, 0.4)

    def test_observer_penalty_converts_to_update_gain(self):
        self.assertAlmostEqual(observational_learning_gain(0.25), 0.75)

    def test_social_weights_and_observer_penalty_are_bounded(self):
        with self.assertRaises(ValueError):
            integrate_suggestion(0.0, 1.0, 0.5, 1.1)
        with self.assertRaises(ValueError):
            observational_learning_gain(-0.1)

    def test_core_agent_combines_receptivity_and_observer_penalty(self):
        setup = Setup("conflict")
        behavior = Behavior(
            "approach",
            difficulty=0.5,
            base_outcome=0.0,
            outcome_volatility=0.0,
        )
        observer = Individual(name="observer", setups=[setup])
        actor = Individual(name="actor")
        observer.add_behavior(
            behavior.id,
            setup.id,
            instinct=0.0,
            utility=0.0,
            enjoyment=0.0,
            alpha_instinct_plus=1.0,
            alpha_instinct_minus=1.0,
            alpha_utility=1.0,
            alpha_enjoyment=1.0,
            w_enjoyment=0.5,
            w_utility=0.5,
            bias_scaling_factor=1.0,
        )
        observer.relationships[actor.id] = {
            "distance": 0.5,
            "receptivity": 0.5,
            "power": 0.0,
            "connection": 0.0,
        }

        observer.update_behavior(
            behavior,
            setup,
            performed=False,
            perceived_utility=1.0,
            perceived_enjoyment=1.0,
            observer_penalty=0.25,
            observed_agent=actor,
        )

        state = observer.behaviors[behavior][setup]
        self.assertAlmostEqual(state["instinct"], 0.75)
        self.assertAlmostEqual(state["utility"], 0.375)
        self.assertAlmostEqual(state["enjoyment"], 0.375)


if __name__ == "__main__":
    unittest.main()
