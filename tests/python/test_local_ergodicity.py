import tempfile
import unittest
from pathlib import Path

import numpy as np

from dkt_agent.local_ergodicity import (
    dkt_inverse_step,
    region_membership,
    run_direction_dependent_local_ergodicity,
    sample_region,
)
from dkt_agent.mean_ergodicity import dkt_step


class TestRegionConditionedDirectionalErgodicity(unittest.TestCase):
    def test_inverse_map_round_trip(self):
        rng = np.random.default_rng(12)
        z = rng.uniform(-1.0, 1.0, 300)
        phi = rng.uniform(-np.pi, np.pi, 300)
        radius = np.sqrt(1.0 - z * z)
        points = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
        for p in (np.pi / 2, 0.73):
            x, y, zz = dkt_step(points[:, 0], points[:, 1], points[:, 2], 2.4, -1.7, p)
            evolved = np.column_stack((x, y, zz))
            recovered = dkt_inverse_step(evolved, 2.4, -1.7, p)
            np.testing.assert_allclose(recovered, points, atol=2e-12, rtol=0.0)

    def test_region_samplers_respect_region_and_sphere(self):
        rng = np.random.default_rng(5)
        cases = [
            ("spherical_cap", dict(center_theta=0.9, center_phi=-0.8, angular_radius=0.25)),
            ("latitude_band", dict(theta_min=0.4, theta_max=1.1)),
            ("longitude_sector", dict(phi_min=2.5, phi_max=-2.5)),
            ("full_sphere", {}),
        ]
        for region_type, kwargs in cases:
            points = sample_region(region_type, 2000, rng, **kwargs)
            np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=2e-12)
            self.assertTrue(region_membership(points, region_type, **kwargs).all())

    def test_identity_map_passes_region_screen_but_is_not_mislabelled_ergodic(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_direction_dependent_local_ergodicity(
                k_r=0.0, k_theta=0.0, p=0.0,
                region_type="spherical_cap", center_theta=0.8,
                center_phi=-0.4, angular_radius=0.3,
                observables=["x", "y", "z"],
                n_kicks=32, n_initial_conditions=300,
                n_reference_samples=2000, n_invariance_samples=1000,
                invariance_tolerance=0.01, seed=4, results_root=directory,
            )
            self.assertEqual(
                result["region_invariance_screen"]["status"],
                "finite_sample_invariance_screen_passed",
            )
            self.assertEqual(
                result["local_measure_interpretation"],
                "candidate_region_only; conditional area measure is a working hypothesis pending stronger invariance validation",
            )
            for name in ("x", "y", "z"):
                values = result["observables"][name]["D_N_over_regional_variance"]
                self.assertTrue(all(abs(value - 1.0) < 0.15 for value in values))
                self.assertNotIn("ergodic", result["observables"][name]["finite_data_note"].lower())
            self.assertTrue(Path(result["artifacts"]["summary_json"]).is_file())
            self.assertTrue(Path(result["artifacts"]["directional_birkhoff_csv"]).is_file())

    def test_noninvariant_cap_blocks_local_measure_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_direction_dependent_local_ergodicity(
                k_r=0.0, k_theta=0.0, p=np.pi / 2,
                region_type="spherical_cap", center_theta=0.8,
                center_phi=-0.4, angular_radius=0.1,
                observables=["x", "y", "z"],
                n_kicks=32, n_initial_conditions=200,
                n_reference_samples=1500, n_invariance_samples=1000,
                invariance_tolerance=0.01, seed=7, results_root=directory,
            )
            self.assertEqual(
                result["region_invariance_screen"]["status"],
                "invariance_not_supported_at_selected_tolerance",
            )
            self.assertEqual(
                result["local_measure_interpretation"],
                "blocked_by_region_leakage; only descriptive conditional-start Birkhoff statistics are reported",
            )
            self.assertTrue(all(
                "Descriptive Birkhoff-average spread only" in value["finite_data_note"]
                for value in result["observables"].values()
            ))

    def test_invalid_region_and_observable_are_rejected(self):
        with self.assertRaises(ValueError):
            sample_region("spherical_cap", 10, np.random.default_rng(1), angular_radius=0.0)
        with self.assertRaises(ValueError):
            run_direction_dependent_local_ergodicity(
                k_r=1.0, region_type="spherical_cap", observables=["q"],
                n_kicks=32, n_initial_conditions=100,
                n_reference_samples=1000, n_invariance_samples=500,
            )


if __name__ == "__main__":
    unittest.main()
