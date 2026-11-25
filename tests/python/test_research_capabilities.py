import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dkt_agent.research_tools import (
    TRAJECTORY_ROOT,
    _resolve_trajectory_csv,
    compare_trajectory_measures,
)


class TestResearchCapabilities(unittest.TestCase):
    def test_trajectory_path_must_be_inside_results_trajectories(self):
        with self.assertRaises(ValueError):
            _resolve_trajectory_csv(str(Path(__file__).resolve()))

    def test_empirical_measure_comparison_is_symmetric(self):
        TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="test_measure_", dir=TRAJECTORY_ROOT))
        output_dir = None
        try:
            rng = np.random.default_rng(2026)
            first = rng.normal(size=(120, 3))
            first /= np.linalg.norm(first, axis=1)[:, None]
            second = rng.normal(size=(120, 3))
            second /= np.linalg.norm(second, axis=1)[:, None]

            paths = []
            for name, points in (("first.csv", first), ("second.csv", second)):
                path = run_dir / name
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["X", "Y", "Z"])
                    writer.writerows(points)
                paths.append(str(path))

            result = json.loads(
                compare_trajectory_measures.invoke(
                    {"trajectory_csvs": paths, "n_z": 8, "n_phi": 16}
                )
            )
            matrix = np.asarray(result["jensen_shannon_divergence_base2"])

            self.assertEqual(result["status"], "success")
            self.assertEqual(matrix.shape, (2, 2))
            np.testing.assert_allclose(matrix, matrix.T)
            np.testing.assert_allclose(np.diag(matrix), 0.0)
            self.assertTrue(np.all(matrix >= 0.0))
            self.assertTrue(np.all(matrix <= 1.0 + 1e-12))
            self.assertTrue(Path(result["csv_path"]).is_file())
            output_dir = Path(result["csv_path"]).parent
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
            if output_dir is not None:
                shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
