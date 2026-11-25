import unittest

from dkt_agent.tools import run_classical_quantum_comparison


class TestClassicalQuantumComparisonValidation(unittest.TestCase):
    def call(self, **overrides):
        values = {
            "s": 2,
            "kr": 1.0,
            "ktheta": 0.0,
            "theta0": 0.75,
            "phi0": -1.0,
            "nsteps": 10,
        }
        values.update(overrides)
        return run_classical_quantum_comparison(**values)

    def test_rejects_non_half_integer_spin(self):
        with self.assertRaisesRegex(ValueError, "integer or half-integer"):
            self.call(s=0.3)

    def test_rejects_nonpositive_spin(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.call(s=0)

    def test_rejects_theta_outside_sphere(self):
        with self.assertRaisesRegex(ValueError, "theta0"):
            self.call(theta0=3.2)

    def test_rejects_phi_outside_range(self):
        with self.assertRaisesRegex(ValueError, "phi0"):
            self.call(phi0=4.0)

    def test_rejects_negative_kick_count(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.call(nsteps=-1)

    def test_rejects_boolean_kick_count(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.call(nsteps=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
