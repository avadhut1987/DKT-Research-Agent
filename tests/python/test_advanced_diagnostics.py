import unittest
import numpy as np
from dkt_agent.research_tools import _autocorrelation_fft


class TestAdvancedDiagnostics(unittest.TestCase):
    def test_autocorrelation_normalization_and_constant_series(self):
        t = np.arange(2048, dtype=float)
        values = np.sin(2.0 * np.pi * t / 32.0)
        rho, status = _autocorrelation_fft(values, 100)
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(float(rho[0]), 1.0, places=12)
        self.assertGreater(float(rho[32]), 0.8)
        self.assertLess(float(rho[16]), -0.8)

        constant, constant_status = _autocorrelation_fft(np.ones(100), 20)
        self.assertEqual(constant_status, "constant_observable")
        self.assertAlmostEqual(float(constant[0]), 1.0)
        self.assertTrue(np.all(np.isnan(constant[1:])))

    def test_autocorrelation_rejects_nonfinite_series(self):
        with self.assertRaises((ValueError, FloatingPointError)):
            _autocorrelation_fft(np.array([1.0, np.nan, 2.0]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
