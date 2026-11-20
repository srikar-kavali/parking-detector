import unittest
from collections import defaultdict

from evaluate_blend import GROUPS, WEATHERS, metrics, rows, select_weight


class BlendEvaluationTest(unittest.TestCase):
    def test_metrics(self):
        result = metrics([90, 10, 20, 80])
        self.assertAlmostEqual(result["accuracy"], 0.85)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.85)
        self.assertAlmostEqual(result["occupied_f1"], 160 / 190)
        self.assertIsNone(metrics([90, 10, 0, 0])["balanced_accuracy"])

    def test_validation_weight(self):
        counts = defaultdict(lambda: [9, 1, 1, 9])
        for weather in WEATHERS:
            counts[(weather, "Blend", 0.5)] = [10, 0, 0, 10]
        self.assertEqual(select_weight(counts), 0.5)
        self.assertEqual(select_weight(defaultdict(lambda: [9, 1, 1, 9])), 1)

    def test_validation_without_rain(self):
        counts = defaultdict(lambda: [0, 0, 0, 0])
        for weather in ("SUNNY", "OVERCAST"):
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                counts[(weather, "Blend", weight)] = [9, 1, 1, 9]
            counts[(weather, "Blend", 0.25)] = [10, 0, 0, 10]
        self.assertEqual(select_weight(counts), 0.25)

    def test_output_weather_and_counts(self):
        counts = defaultdict(lambda: [9, 1, 1, 9])
        frames = dict.fromkeys(GROUPS, 1)
        output = list(rows("test", (counts, frames), [("Blend", 0.5)]))
        self.assertEqual([row["weather"] for row in output], list(GROUPS))
        self.assertTrue(all(row["slots"] == 20 for row in output))
        self.assertTrue(all(row["fp"] == row["fn"] == 1 for row in output))


if __name__ == "__main__":
    unittest.main()
