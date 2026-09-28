import unittest
import tempfile
from pathlib import Path

from source_aware_report import compare, frontier, joint_choices


class RateDistortionTests(unittest.TestCase):
    def test_constant_rate_saving(self):
        base = [(1, 4), (2, 3), (3, 2), (4, 1)]
        candidate = [(d, r * 0.8) for d, r in base]
        self.assertAlmostEqual(compare(base, candidate)["bd_rate_percent"], -20)
        self.assertAlmostEqual(compare(candidate, base)["bd_rate_percent"], 25)

    def test_identity_and_affine_metric_units(self):
        base = [(1, 4), (2, 3), (3, 2), (4, 1)]
        self.assertAlmostEqual(compare(base, base)["bd_rate_percent"], 0)
        candidate = [(d, r * 0.9) for d, r in base]
        transformed = lambda c: [(17 + d * 12, r) for d, r in c]
        self.assertAlmostEqual(compare(transformed(base), transformed(candidate))["bd_rate_percent"], -10)

    def test_no_extrapolation_and_overlap_coverage(self):
        self.assertIsNone(compare([(1, 4), (2, 3)], [(3, 2), (4, 1)])["bd_rate_percent"])
        result = compare([(1, 4), (2, 3), (3, 2)], [(2, 3), (3, 2)])
        self.assertEqual(result["baseline_coverage"], 0.5)
        self.assertEqual(result["candidate_coverage"], 1.0)
        self.assertTrue(result["sparse"])

    def test_duplicate_rate_and_dominated_points(self):
        self.assertEqual(frontier([(4, 1), (3, 2), (3.5, 2), (5, 3), (2, 4)]), [(2, 4), (3, 2), (4, 1)])

    def test_joint_gate_rejects_one_metric_regression(self):
        def point(variant, size, ssim2, butter):
            return dict(variant=variant, bytes=size, quality=80, jpeg=f"{variant}.jpg",
                        generation=dict(ssim2=ssim2, butteraugli=butter),
                        cumulative=dict(ssim2=ssim2, butteraugli=butter))
        group = ("test", "classic", 80, "jpegli", "420")
        variants = {"generic": [point("generic", 100, 90, 1)],
                    "good": [point("good", 95, 91, 0.9)],
                    "bad_butter": [point("bad_butter", 80, 99, 1.1)],
                    "bad_ssim": [point("bad_ssim", 70, 89, 0.8)],
                    "oversize": [point("oversize", 110, 99, 0.1)]}
        with tempfile.TemporaryDirectory() as d:
            rows = joint_choices(Path(d), {group: variants})
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["winner"], "good")
            self.assertAlmostEqual(row["savings_percent"], 5)


if __name__ == "__main__":
    unittest.main()
