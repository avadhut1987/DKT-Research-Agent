import tempfile
import unittest
from pathlib import Path

import numpy as np

from dkt_agent.classical_analysis import (
    _sample_admissible_pair_distances,
    correlation_dimension,
    occupied_cell_scaling,
    equal_area_measure,
    empirical_measure_comparison,
    load_trajectory,
)


def make_sphere_points(n=300, seed=1234):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    return points


class TestClassicalAnalysis(unittest.TestCase):

    def setUp(self):
        self.points = make_sphere_points()

    def test_admissible_pair_sampling_is_reproducible(self):
        first = _sample_admissible_pair_distances(
            self.points, theiler=5, n_pairs=500, seed=42
        )
        second = _sample_admissible_pair_distances(
            self.points, theiler=5, n_pairs=500, seed=42
        )

        np.testing.assert_allclose(first, second)
        self.assertEqual(len(first), 500)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(first > 0))

    def test_correlation_dimension_returns_expected_fields(self):
        result = correlation_dimension(
            self.points,
            theiler=5,
            n_pairs=2000,
            seed=42,
            radii=np.logspace(-2, np.log10(2.0), 24),
        )

        self.assertIn("radii", result)
        self.assertIn("correlation_integral", result)
        self.assertIn("local_slopes", result)
        self.assertIn("fit", result)

        self.assertEqual(len(result["radii"]), 24)
        self.assertEqual(len(result["correlation_integral"]), 24)
        self.assertEqual(len(result["local_slopes"]), 24)

        self.assertTrue(
            np.all(np.diff(result["correlation_integral"]) >= 0)
        )

    def test_occupied_cell_scaling_returns_valid_counts(self):
        result = occupied_cell_scaling(
            self.points,
            epsilons=np.array([0.1, 0.2, 0.3, 0.5]),
            n_shifts=3,
            seed=42,
        )

        counts = result["mean_occupied_cells"]

        self.assertEqual(len(counts), 4)
        self.assertTrue(np.all(np.isfinite(counts)))
        self.assertTrue(np.all(counts > 0))
        self.assertEqual(result["n_shifts"], 3)

    def test_equal_area_measure_is_normalized(self):
        measure = equal_area_measure(
            self.points, n_z=12, n_phi=24
        )

        self.assertEqual(len(measure), 12 * 24)
        self.assertAlmostEqual(float(measure.sum()), 1.0)
        self.assertTrue(np.all(measure >= 0))

    def test_empirical_measure_comparison_is_symmetric(self):
        trajectories = {
            "first": self.points,
            "second": make_sphere_points(seed=5678),
        }

        result = empirical_measure_comparison(
            trajectories, n_z=12, n_phi=24
        )

        matrix = result["js_divergence"]

        self.assertEqual(matrix.shape, (2, 2))
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 0.0)
        self.assertTrue(np.all(matrix >= 0))
        self.assertTrue(np.all(matrix <= 1.0 + 1e-12))

    def test_load_trajectory_reads_unit_sphere_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.csv"

            np.savetxt(
                path,
                self.points,
                delimiter=",",
                header="X,Y,Z",
                comments="",
            )

            loaded = load_trajectory(path)

        np.testing.assert_allclose(loaded, self.points)

    def test_load_trajectory_rejects_non_unit_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.csv"

            invalid_points = self.points.copy()
            invalid_points[0] *= 2.0

            np.savetxt(
                path,
                invalid_points,
                delimiter=",",
                header="X,Y,Z",
                comments="",
            )

            with self.assertRaises(ValueError):
                load_trajectory(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
