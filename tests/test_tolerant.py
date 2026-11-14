import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class TolerantMetricTests(unittest.TestCase):
    def test_one_pixel_offset_is_forgiven(self):
        from metrics import dilate, mask_metrics, tolerant_mask_metrics

        truth = np.zeros((20, 20), bool)
        truth[5:15, 10] = True              # a vertical 1-px line
        shifted = np.zeros((20, 20), bool)
        shifted[5:15, 11] = True            # the same line one pixel to the right
        self.assertEqual(mask_metrics(shifted, truth)["iou"], 0.0)
        tolerant = tolerant_mask_metrics(shifted, truth, radius=1)
        self.assertEqual((tolerant["precision"], tolerant["recall"], tolerant["f1"]), (1.0, 1.0, 1.0))
        self.assertEqual(int(dilate(truth, 1).sum()), 36)
        far = np.zeros((20, 20), bool)
        far[5:15, 15] = True
        self.assertEqual(tolerant_mask_metrics(far, truth, radius=2)["f1"], 0.0)
        empty = tolerant_mask_metrics(np.zeros((20, 20), bool), truth)
        self.assertIsNone(empty["precision"])
        self.assertEqual(empty["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
