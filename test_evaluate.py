import unittest

import numpy as np

import negatives
from evaluate import Scores, bootstrap, choose_threshold, conversation_data, rank, split_conversations, summarize


class EvaluationTest(unittest.TestCase):
    def test_split_is_disjoint_and_independent_of_input_order(self):
        rows = [{"sample_id": f"conv-{index}"} for index in range(5, 0, -1)]
        calibration, test = split_conversations(rows)
        self.assertEqual([row["sample_id"] for row in calibration], ["conv-1", "conv-3", "conv-5"])
        self.assertEqual([row["sample_id"] for row in test], ["conv-2", "conv-4"])
        self.assertEqual(len(set(negatives.CALIBRATION + negatives.TEST)), 60)

    def test_only_resolved_single_evidence_is_positive(self):
        row = {
            "conversation": {
                "session_1": [{"dia_id": "a", "speaker": "Ann", "text": "First"}, {"dia_id": "b", "speaker": "Ann", "text": "Second"}],
                "session_1_date_time": "Monday",
            },
            "qa": [
                {"category": 1, "question": "Valid?", "evidence": ["a"]},
                {"category": 2, "question": "Two?", "evidence": ["a", "b"]},
                {"category": 3, "question": "Missing?", "evidence": ["c"]},
                {"category": 4, "question": "Unlabeled?"},
                {"category": 5, "question": "Adversarial?", "evidence": ["a"]},
            ],
        }
        self.assertEqual(conversation_data(row), ({"a": "Ann: First", "b": "Ann: Second"}, [("Valid?", "a")]))

    def test_rank_uses_same_winner_for_both_scores(self):
        winners, scores = rank(np.array([[.3, .7, .4], [.8, .8, .6]]))
        np.testing.assert_array_equal(winners, [1, 0])
        np.testing.assert_allclose(scores, [[.7, .3], [.8, 0]])

    def test_threshold_obeys_budget_and_excludes_wrong_winners(self):
        positive = np.array([.700, .690, .300])
        correct = np.array([True, True, False])
        negative = np.array([.9, .5] + [.1] * 8)
        self.assertEqual(choose_threshold(positive, correct, negative), .689)
        self.assertEqual(choose_threshold(positive, np.zeros(3, dtype=bool), negative), 1.0)

    def test_recall_counts_wrong_and_suppressed_winners_as_misses(self):
        row = Scores(
            np.array([[.8, .1], [.9, .4], [.7, .3]]),
            np.array([True, False, True]),
            np.array([[.8, .2], [.7, .3]]),
        )
        np.testing.assert_allclose(summarize([row], np.array([.7, .2])), [[100 / 3, 100 / 3], [50, 50]])

    def test_paired_bootstrap_of_identical_policies_has_zero_difference(self):
        rows = [
            Scores(np.array([[.9, .9]]), np.array([True]), np.array([[.9, .9], [.1, .1]])),
            Scores(np.array([[.2, .2]]), np.array([True]), np.array([[.1, .1], [.1, .1]])),
        ]
        intervals, difference = bootstrap(rows, np.array([.5, .5]))
        self.assertEqual(intervals.shape, (2, 2, 2))
        np.testing.assert_array_equal(difference, np.zeros((2, 2)))
        self.assertGreater(intervals[1, 0, 0], intervals[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
