import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dkt_agent.mixing_tools as mixing_tools


def _grid_result(grid_size=5, nsteps=2):
    points = []
    for i in range(1, grid_size + 1):
        theta0 = (i - 0.5) * 3.141592653589793 / grid_size
        for j in range(1, grid_size + 1):
            phi0 = -3.141592653589793 + (j - 1) * 2 * 3.141592653589793 / grid_size
            for kick in range(nsteps + 1):
                points.append({
                    "grid_i": i,
                    "grid_j": j,
                    "kick": kick,
                    "theta0": theta0,
                    "phi0": phi0,
                    "X": 0.1 * kick,
                    "Y": 0.2 * kick,
                    "Z": 1.0 - 0.1 * kick,
                    "norm": 1.0,
                })
    return {
        "status": "success",
        "phase_space_csv": "/tmp/mock_phase_space.csv",
        "max_norm_error": 1e-15,
        "orbit_points": points,
    }


class TestSavedDktEnsembleGeneration(unittest.TestCase):
    def test_generates_parameter_named_csvs_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(mixing_tools, "PROJECT_ROOT", root),
                patch.object(mixing_tools, "TRAJECTORY_ROOT", root / "results" / "trajectories"),
                patch.object(mixing_tools, "simulate_dkt_grid", return_value=_grid_result()),
            ):
                result = mixing_tools.generate_dkt_ensemble(
                    kr=1.2, ktheta=0.5, p=1.5707963267948966,
                    grid_size=5, nsteps=2,
                )

                self.assertEqual(result["n_trajectories"], 25)
                manifest_path = Path(result["manifest_path"])
                self.assertTrue(manifest_path.is_file())
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["parameters"]["kr"], 1.2)
                self.assertEqual(manifest["parameters"]["ktheta"], 0.5)
                self.assertEqual(len(manifest["trajectory_files"]), 25)

                first = manifest["trajectory_files"][0]
                csv_path = root / first["file"]
                self.assertTrue(csv_path.is_file())
                self.assertIn("kr_1p2", csv_path.name)
                self.assertIn("ktheta_0p5", csv_path.name)
                with csv_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0]["step"], "0")

                discovered = mixing_tools.discover_saved_ensembles()
                self.assertEqual(len(discovered), 1)
                saved = next(iter(discovered.values()))
                self.assertEqual(saved["manifest"]["parameters"]["n_trajectories"], 25)

    def test_discovers_nested_single_trajectory_runs_by_exact_map_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory_root = root / "results" / "trajectories"
            for index, (theta0, phi0) in enumerate(((0.4, -0.8), (1.2, 0.6))):
                run_dir = trajectory_root / f"dkt_run_{index}"
                run_dir.mkdir(parents=True)
                csv_path = run_dir / (
                    f"trajectory_kr_1_ktheta_0_p_1p570796_"
                    f"theta0_{theta0}_phi0_{phi0}_nsteps_10.csv"
                )
                csv_path.write_text(
                    "step,X,Y,Z,norm\\n0,0,0,1,1\\n1,0.1,0.1,0.99,1\\n",
                    encoding="utf-8",
                )
                metadata = {
                    "parameters": {
                        "kr": 1.0, "ktheta": 0.0, "p": 1.5707963267948966,
                        "theta0": theta0, "phi0": phi0, "nsteps": 1,
                    },
                    "trajectory_csv": str(csv_path.relative_to(root)),
                    "trajectory_rows": 2,
                }
                (run_dir / "trajectory_metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )

            with (
                patch.object(mixing_tools, "PROJECT_ROOT", root),
                patch.object(mixing_tools, "TRAJECTORY_ROOT", trajectory_root),
            ):
                ensembles = mixing_tools.discover_saved_ensembles()
                self.assertEqual(len(ensembles), 1)
                label, ensemble = next(iter(ensembles.items()))
                self.assertIn("Existing simulations", label)
                self.assertEqual(ensemble["manifest"]["parameters"]["n_trajectories"], 2)
                self.assertTrue(Path(ensemble["manifest_path"]).is_file())

    def test_rejects_grid_sizes_outside_supported_range(self):
        with self.assertRaises(ValueError):
            mixing_tools.generate_dkt_ensemble(
                kr=1.0, grid_size=4, nsteps=10
            )


if __name__ == "__main__":
    unittest.main()
