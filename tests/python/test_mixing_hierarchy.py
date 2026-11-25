import unittest

import numpy as np

from dkt_agent.mixing_hierarchy import (
    AREA_MEANS,
    cartesian_observables,
    finite_partition_alpha,
    mean_ergodic_ensemble_diagnostic,
    run_mixing_hierarchy_from_trajectories,
    strong_mixing_partition_diagnostic,
    weak_mixing_cesaro_diagnostic,
)


def _sphere_points(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    radius = np.sqrt(1.0 - z*z)
    return np.column_stack((radius*np.cos(phi), radius*np.sin(phi), z))


class TestObservableDefinitions(unittest.TestCase):
    def test_uniform_sphere_moments(self):
        points = _sphere_points(2026, 200000)
        means = cartesian_observables(points).mean(axis=0)
        np.testing.assert_allclose(means, AREA_MEANS, atol=4e-3)

    def test_observable_order_and_shape(self):
        values = cartesian_observables(np.array([[1.0, 0.0, 0.0]]))
        self.assertEqual(values.shape, (1, 9))
        np.testing.assert_allclose(values[0], [1, 0, 0, 1, 0, 0, 0, 0, 0])


class TestMeanErgodicDiagnostic(unittest.TestCase):
    def test_identity_like_orbits_retain_ensemble_variance(self):
        points = np.repeat(_sphere_points(11, 8)[:, None, :], 128, axis=1)
        result = mean_ergodic_ensemble_diagnostic(list(points), transient=0)
        final_rows = [row for row in result["rows"] if row["analyzed_steps"] == 128]
        x_row = next(row for row in final_rows if row["observable"] == "X")
        self.assertGreater(x_row["ensemble_variance"], 0.05)
        self.assertEqual(result["n_trajectories"], 8)

    def test_variance_decreases_for_identical_orbit_copies(self):
        base = _sphere_points(9, 256)
        ensemble = [base.copy() for _ in range(5)]
        result = mean_ergodic_ensemble_diagnostic(ensemble)
        self.assertTrue(all(row["ensemble_variance"] == 0.0 for row in result["rows"]))


class TestWeakMixingDiagnostic(unittest.TestCase):
    def test_periodic_observable_has_nonzero_cesaro_envelope(self):
        t = np.arange(4096)
        series = np.column_stack((np.cos(2*np.pi*t/16), np.sin(2*np.pi*t/16),
                                  np.zeros_like(t)))
        # Embed periodic coordinates on the unit sphere.
        points = np.column_stack((np.cos(2*np.pi*t/16), np.sin(2*np.pi*t/16),
                                  np.zeros_like(t)))
        result = weak_mixing_cesaro_diagnostic([points, points.copy()], max_lag=512)
        final = result["observables"]["X"]["final_cesaro_value"]
        self.assertIsNotNone(final)
        self.assertGreater(final, 0.05)

    def test_constant_observable_is_reported_not_misclassified(self):
        points = np.tile(np.array([[0.0, 0.0, 1.0]]), (128, 1))
        # A second orbit is required; choose another constant orbit.
        other = np.tile(np.array([[0.0, 0.0, -1.0]]), (128, 1))
        result = weak_mixing_cesaro_diagnostic([points, other], max_lag=32)
        self.assertEqual(
            result["observables"]["Z"]["finite_record_status"],
            "constant_or_unresolved",
        )


class TestStrongMixingFinitePartition(unittest.TestCase):
    def test_alpha_coefficient_for_independent_joint_table(self):
        delta = np.zeros((2, 2), dtype=float)
        self.assertAlmostEqual(finite_partition_alpha(delta), 0.0)

    def test_alpha_coefficient_for_perfect_two_cell_correlation(self):
        # Joint law is concentrated on (0,0) and (1,1), with equal weights.
        # Delta = [[1/4,-1/4],[-1/4,1/4]], and alpha = 1/4.
        delta = np.array([[0.25, -0.25], [-0.25, 0.25]])
        self.assertAlmostEqual(finite_partition_alpha(delta), 0.25)

    def test_partition_diagnostic_reports_finite_partition_only(self):
        ensemble = [_sphere_points(seed, 512) for seed in range(6)]
        result = strong_mixing_partition_diagnostic(
            ensemble, n_z=2, n_phi=2, max_lag=8
        )
        self.assertEqual(result["partition"]["n_cells"], 4)
        self.assertTrue(result["lags"])
        self.assertIn("finite_partition_alpha", result["lags"][0])
        self.assertIn("cannot establish strong mixing", result["interpretation"])


class TestIntegratedHierarchy(unittest.TestCase):
    def test_returns_separate_diagnostics_without_verdict(self):
        ensemble = [_sphere_points(seed, 256) for seed in range(5)]
        result = run_mixing_hierarchy_from_trajectories(
            ensemble, max_lag=32, n_z=2, n_phi=2
        )
        self.assertEqual(result["status"], "diagnostic_only")
        self.assertIsNone(result["hierarchy_verdict"])
        levels = result["finite_data_diagnostics"]
        self.assertIn("ergodicity_related", levels)
        self.assertIn("weak_mixing_related", levels)
        self.assertIn("strong_mixing_related", levels)


if __name__ == "__main__":
    unittest.main()
