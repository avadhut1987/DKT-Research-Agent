import unittest

from dkt_agent.tools import simulate_dkt


class TestDKTSimulation(unittest.TestCase):

    def test_basic_simulation(self):
        result = simulate_dkt(
            kr=1.0,
            ktheta=0.0,
            theta0=0.75,
            phi0=-1.0,
            nsteps=1000,
            include_trajectory=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["trajectory_rows"], 1001)

        self.assertAlmostEqual(
            result["parameters"]["k"], 1.0
        )
        self.assertAlmostEqual(
            result["parameters"]["kprime"], 1.0
        )

        self.assertLess(result["max_norm_error"], 1e-10)

        self.assertEqual(
            result["trajectory"][0]["step"], 0
        )
        self.assertEqual(
            result["trajectory"][-1]["step"], 1000
        )

        print("\nDKT integration test result:")
        print("Status:", result["status"])
        print("Trajectory rows:", result["trajectory_rows"])
        print("Initial state:", result["initial_state"])
        print("Final state:", result["final_state"])
        print("Maximum norm error:", result["max_norm_error"])
        print("CSV:", result["trajectory_csv"])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            simulate_dkt(
                kr=float("nan"),
                ktheta=0.0,
                theta0=0.75,
                phi0=-1.0,
                nsteps=100,
            )

    def test_invalid_step_count(self):
        with self.assertRaises(ValueError):
            simulate_dkt(
                kr=1.0,
                ktheta=0.0,
                theta0=0.75,
                phi0=-1.0,
                nsteps=0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
