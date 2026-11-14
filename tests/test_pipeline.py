"""End-to-end anomaly pipeline with a stand-in feature extractor, so the test needs numpy and Pillow but not torch."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    from PIL import Image
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


class PatchStatistics:
    """Cheap stand-in for the CNN: mean colour and contrast of every 16x16 patch of a 128 px image."""

    def __init__(self):
        self.grid_shape = (8, 8)
        self.dim = 6

    def extract(self, image):
        array = np.asarray(image.convert("RGB").resize((128, 128)), np.float32) / 255.0
        patches = []
        for y in range(8):
            for x in range(8):
                block = array[y * 16:(y + 1) * 16, x * 16:(x + 1) * 16]
                patches.append(np.concatenate([block.mean(axis=(0, 1)), block.std(axis=(0, 1))]))
        return np.asarray(patches, np.float32)

    def info(self):
        return {"backbone": "patch statistics (test stand-in)", "layers": ["mean", "std"], "input_size": 128, "grid_shape": [8, 8], "dim": 6}


@unittest.skipUnless(HAVE_DEPS, "needs numpy and Pillow")
class PipelineTests(unittest.TestCase):
    def test_benchmark_on_a_small_synthetic_set_separates_defects(self):
        import inspect_anomaly
        from synthetic_parts import write_anomaly_dataset

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_anomaly_dataset(root / "data", train_good=12, calib_good=10, test_good=6, test_defect_each=3, seed=4)
            inspect_anomaly.make_extractor = lambda args: PatchStatistics()
            report = inspect_anomaly.benchmark(inspect_anomaly.main.__globals__["argparse"].Namespace(
                data=str(root / "data"), output=str(root / "out"), coreset=0.5, seed=0, sigma=2.0, false_alarm=0.1, threads=1,
                input_size=128, cache_dir=None))
            for category, result in report["categories"].items():
                self.assertGreaterEqual(result["image_auroc"], 0.6, category)
                self.assertIsNotNone(result["pixel_auroc"])
                self.assertEqual(result["at_threshold"]["true_positive"] + result["at_threshold"]["false_negative"], len(result["per_defect"]) * 3)
                self.assertTrue((root / "out" / category / "bank.npy").exists())
                self.assertTrue((root / "out" / category / result["records"][0]["heatmap"]).exists())
            self.assertTrue((root / "out" / "dashboard.html").exists())
            saved = json.loads((root / "out" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(set(saved["categories"]), {"plate", "cap", "tile"})
            # score command on a saved bank
            from anomaly import MemoryBank

            bank = MemoryBank.load(root / "out" / "plate")
            self.assertIsNotNone(bank.threshold)
            results = inspect_anomaly.score_paths(PatchStatistics(), bank, inspect_anomaly.list_images(root / "data" / "plate" / "test")[:3])
            records = inspect_anomaly.write_scored(results, bank, root / "scored")
            self.assertEqual(len(records), 3)
            self.assertIn("defect", records[0])

    def test_random_images_are_flagged_by_a_bank_of_good_parts(self):
        from anomaly import MemoryBank
        from synthetic_parts import render_part

        extractor = PatchStatistics()
        rng = random.Random(2)
        features = np.concatenate([extractor.extract(render_part("cap", rng)[0]) for _ in range(10)])
        bank = MemoryBank.fit(features, extractor.grid_shape, extractor.info(), fraction=0.5)
        good_scores = [bank.score(extractor.extract(render_part("cap", rng)[0]), (64, 64))[0] for _ in range(10)]
        bank.calibrate(good_scores, 0.1)
        noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8), "RGB")
        self.assertTrue(bank.verdict(bank.score(extractor.extract(noise), (64, 64))[0])["defect"])


if __name__ == "__main__":
    unittest.main()
