import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import auroc, best_f1_threshold, quantile, threshold_metrics

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class RankingMetricTests(unittest.TestCase):
    def test_auroc_handles_perfect_random_and_tied_scores(self):
        self.assertEqual(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)
        self.assertAlmostEqual(auroc([1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1]), 2 / 3)
        self.assertIsNone(auroc([1, 2], [0, 0]))
        with self.assertRaises(ValueError):
            auroc([], [])

    def test_threshold_metrics_quantile_and_best_f1(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        labels = [0, 0, 0, 0, 1, 1, 0, 1]
        report = threshold_metrics(scores, labels, 0.5)
        self.assertEqual((report["true_positive"], report["false_positive"], report["false_negative"], report["true_negative"]), (3, 1, 0, 4))
        self.assertEqual(report["false_positive_rate"], 0.2)
        self.assertEqual(quantile([5, 1, 3, 2, 4], 0.95), 5)
        self.assertEqual(quantile([5, 1, 3, 2, 4], 0.5), 3)
        self.assertEqual(quantile(list(range(1, 101)), 0.95), 95)
        best = best_f1_threshold(scores, labels)
        self.assertEqual(best["f1"], 0.8571)
        with self.assertRaises(ValueError):
            quantile([1, 2], 1.5)


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class AnomalyTests(unittest.TestCase):
    def setUp(self):
        from anomaly import MemoryBank

        rng = np.random.default_rng(0)
        self.good = rng.normal(0, 1, (40 * 16, 8)).astype(np.float32)   # 40 images of a 4x4 grid
        self.bank = MemoryBank.fit(self.good, (4, 4), {"backbone": "test"}, fraction=0.25, seed=1)

    def test_coreset_keeps_the_requested_fraction_and_spreads_out(self):
        from anomaly import coreset

        picked = coreset(self.good, 0.25, seed=1)
        self.assertEqual(len(picked), 160)
        self.assertEqual(len(set(picked)), 160)
        self.assertEqual(coreset(self.good[:5], 1.0), [0, 1, 2, 3, 4])
        self.assertEqual(len(self.bank.bank), 160)

    def test_scores_rise_with_distance_and_heatmap_localises(self):
        normal = self.good[:16]
        score_normal, heat_normal, _ = self.bank.score(normal, (64, 64))
        shifted = normal.copy()
        shifted[5] += 12.0                      # one anomalous patch in row 1, column 1
        score_shifted, heat, grid = self.bank.score(shifted, (64, 64))
        self.assertGreater(score_shifted, score_normal * 3)
        self.assertEqual(heat.shape, (64, 64))
        self.assertEqual(grid.shape, (4, 4))
        self.assertEqual(np.unravel_index(np.argmax(grid), grid.shape), (1, 1))
        y, x = np.unravel_index(np.argmax(heat), heat.shape)
        self.assertTrue(8 <= y < 40 and 8 <= x < 40)

    def test_calibration_threshold_comes_from_good_scores_only(self):
        good_scores = [float(v) for v in np.linspace(1.0, 2.0, 20)]
        report = self.bank.calibrate(good_scores, target_false_alarm=0.05, defect_scores=[2.5, 3.0, 1.5])
        self.assertEqual(report["threshold"], sorted(good_scores)[18])   # ceil(20 * 0.95) = 19th value: one good part in 20 may exceed it
        self.assertEqual(report["at_threshold"]["false_positive"], 1)
        self.assertEqual(report["at_threshold"]["recall"], 0.6667)
        self.assertGreater(report["calibration_auroc"], 0.8)
        self.assertTrue(self.bank.verdict(2.4)["defect"])
        self.assertFalse(self.bank.verdict(1.9)["defect"])
        with self.assertRaises(ValueError):
            self.bank.calibrate(good_scores[:5])
        with self.assertRaises(ValueError):
            self.bank.calibrate(good_scores, target_false_alarm=1.5)

    def test_bank_round_trips_through_disk(self):
        from anomaly import MemoryBank

        self.bank.calibrate([1.0] * 12, 0.1)
        with tempfile.TemporaryDirectory() as folder:
            self.bank.save(folder)
            loaded = MemoryBank.load(folder)
            self.assertEqual(loaded.threshold, self.bank.threshold)
            self.assertEqual(loaded.grid_shape, (4, 4))
            self.assertTrue(np.array_equal(loaded.bank, self.bank.bank))
            meta = json.loads((Path(folder) / "bank.json").read_text())
            self.assertEqual(meta["coreset"]["kept"], 160)

    def test_upsample_and_nearest_distances(self):
        from anomaly import nearest_distances, upsample

        grid = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32)
        up = upsample(grid, (4, 4))
        self.assertEqual(up.shape, (4, 4))
        self.assertAlmostEqual(float(up[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(up[3, 3]), 3.0, places=5)
        self.assertTrue(np.all(np.diff(up, axis=1) >= -1e-6))
        bank = np.array([[0, 0], [10, 10]], np.float32)
        distances = nearest_distances(np.array([[1, 0], [9, 10], [5, 5]], np.float32), bank)
        self.assertAlmostEqual(float(distances[0]), 1.0, places=4)
        self.assertAlmostEqual(float(distances[1]), 1.0, places=4)
        self.assertAlmostEqual(float(distances[2]), 50 ** 0.5, places=4)


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class MaskMetricTests(unittest.TestCase):
    def test_iou_dice_and_pixel_auroc(self):
        from metrics import mask_metrics, pixel_auroc

        truth = np.zeros((10, 10), bool)
        truth[2:6, 2:6] = True
        predicted = np.zeros((10, 10), bool)
        predicted[3:7, 3:7] = True
        m = mask_metrics(predicted, truth)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (9, 7, 7))
        self.assertAlmostEqual(m["iou"], 9 / 23)
        self.assertAlmostEqual(m["dice"], 18 / 32)
        self.assertIsNone(mask_metrics(np.zeros((2, 2), bool), np.zeros((2, 2), bool))["iou"])
        score_map = truth.astype(np.float32) * 0.9 + 0.05
        self.assertEqual(pixel_auroc([score_map], [truth]), 1.0)
        with self.assertRaises(ValueError):
            mask_metrics(np.zeros((2, 2), bool), np.zeros((3, 3), bool))


if __name__ == "__main__":
    unittest.main()
