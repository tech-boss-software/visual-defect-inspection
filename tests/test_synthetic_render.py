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


@unittest.skipUnless(HAVE_DEPS, "needs numpy and Pillow")
class SyntheticTests(unittest.TestCase):
    def test_every_defect_has_a_mask_and_changes_the_image(self):
        from synthetic_parts import CATEGORIES, DEFECTS, SIZE, render_part

        for category in CATEGORIES:
            rng = random.Random(7)
            good, mask = render_part(category, rng)
            self.assertEqual(good.size, (SIZE, SIZE))
            self.assertEqual(int(mask.sum()), 0)
            for defect in DEFECTS[category]:
                image, mask = render_part(category, random.Random(7), defect)
                self.assertGreater(int(mask.sum()), 40, f"{category}/{defect}")
                self.assertLess(mask.mean(), 0.1, f"{category}/{defect} should be a local defect")
                diff = np.abs(np.asarray(image, np.float32) - np.asarray(good, np.float32)).mean(axis=2)
                self.assertGreater(float(diff[mask].mean()), float(diff[~mask].mean()), f"{category}/{defect} should differ most inside its mask")

    def test_two_good_parts_differ_but_share_the_design(self):
        from synthetic_parts import render_part

        a, _ = render_part("plate", random.Random(1))
        b, _ = render_part("plate", random.Random(2))
        self.assertGreater(float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean()), 1.0)
        dark_a = np.asarray(a.convert("L")) < 60
        dark_b = np.asarray(b.convert("L")) < 60
        self.assertGreater(float((dark_a & dark_b).sum()) / max(1, float(dark_a.sum())), 0.6)   # the four holes are in the same places

    def test_concrete_cracks_and_dataset_layout(self):
        from synthetic_parts import render_concrete, write_anomaly_dataset, write_crack_dataset

        image, mask = render_concrete(random.Random(3), True)
        self.assertGreater(int(mask.sum()), 100)
        self.assertLess(float(mask.mean()), 0.05)
        _, clean = render_concrete(random.Random(3), False)
        self.assertEqual(int(clean.sum()), 0)
        with tempfile.TemporaryDirectory() as folder:
            index = write_anomaly_dataset(Path(folder) / "a", train_good=3, calib_good=2, test_good=2, test_defect_each=1, seed=1)
            self.assertEqual(set(index["categories"]), {"plate", "cap", "tile"})
            plate = Path(folder) / "a" / "plate"
            self.assertEqual(len(list((plate / "train_good").glob("*.png"))), 3)
            test_entries = index["categories"]["plate"]["test"]
            self.assertEqual(sum(e["label"] for e in test_entries), 4)
            for entry in test_entries:
                if entry["label"]:
                    with Image.open(plate / "test" / entry["mask"]) as m:
                        self.assertEqual((np.asarray(m) > 127).sum(), entry["mask_pixels"])
            cracks = write_crack_dataset(Path(folder) / "c", train=4, val=2, test=2, seed=1)
            self.assertEqual({k: len(v) for k, v in cracks["splits"].items()}, {"train": 4, "val": 2, "test": 2})


@unittest.skipUnless(HAVE_DEPS, "needs numpy and Pillow")
class RenderTests(unittest.TestCase):
    def test_overlay_colormap_and_svgs(self):
        from render import colormap, curve_svg, overlay, score_histogram_svg, side_by_side

        colours = colormap(np.array([0.0, 0.5, 1.0]))
        self.assertEqual(colours.shape, (3, 3))
        self.assertGreater(int(colours[2, 0]), int(colours[0, 0]))     # red rises
        self.assertGreater(int(colours[0, 2]), int(colours[2, 2]))     # blue falls
        image = Image.new("RGB", (32, 32), (100, 100, 100))
        heat = np.zeros((32, 32), np.float32)
        heat[10:20, 10:20] = 1.0
        mask = heat > 0.5
        out = overlay(image, heat, 0.0, 1.0, mask=mask)
        self.assertEqual(out.size, (32, 32))
        self.assertEqual(out.getpixel((10, 10)), (255, 255, 255))      # contour
        self.assertNotEqual(out.getpixel((15, 15)), (100, 100, 100))   # heat applied
        self.assertEqual(out.getpixel((2, 2)), (100, 100, 100))        # untouched background
        panel = side_by_side([image, image], ["a", "b"])
        self.assertEqual(panel.width, 72)
        svg = score_histogram_svg([0.1, 0.2, 0.3], [0.8, 0.9], 0.5)
        self.assertIn("threshold 0.500", svg)
        self.assertIn("<svg", curve_svg([(0, 0), (1, 1)], "x", "y", marker=(0.5, 0.5)))


if __name__ == "__main__":
    unittest.main()
