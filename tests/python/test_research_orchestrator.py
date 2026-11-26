import unittest
from dkt_agent.research_orchestrator import build_research_plan

class TestResearchOrchestrator(unittest.TestCase):
    def test_fixed_kr_ktheta_geometry_guardrail(self):
        p = build_research_plan("Compare ktheta=0 and 10000 at fixed kr=1 for trajectory geometry.")
        self.assertEqual(p["research_intent"], "fixed_kr_ktheta_geometry")
        self.assertEqual(p["workflow"][0]["tool"], "run_dkt_comparison_study")
        self.assertTrue(any("not changed chaoticity" in x for x in p["scope_and_controls"]))
        self.assertFalse(p["execution_performed"])

    def test_periodic_orbit_question_uses_direct_orbit_comparison(self):
        p = build_research_plan("Compare periodic-orbit geometry at fixed kr=1 for ktheta=0 and 10000.")
        self.assertEqual(p["research_intent"], "fixed_kr_ktheta_geometry")
        self.assertEqual([x["tool"] for x in p["workflow"]], ["compare_dkt_periodic_orbits"])

    def test_mean_ergodicity_distinguished(self):
        p = build_research_plan("Investigate von Neumann mean ergodicity and Birkhoff variance.")
        self.assertEqual(p["research_intent"], "mean_ergodicity")
        self.assertIn("mean_ergodicity_langchain_tool", [x["tool"] for x in p["workflow"]])
        self.assertTrue(any("pointwise" in x for x in p["scope_and_controls"]))

    def test_hierarchy_requires_same_map_ensemble(self):
        p = build_research_plan("Study weak and strong mixing hierarchy", existing_data_available=False)
        self.assertEqual(p["research_intent"], "ergodic_hierarchy")
        self.assertTrue(any("new ensemble" in x.lower() for x in p["clarifications_or_gates"]))

    def test_new_calculation_requires_authorization(self):
        p = build_research_plan("Estimate Lyapunov exponents and KS entropy.")
        self.assertEqual(p["execution_authorization"], "not_authorized")
        self.assertTrue(any("not authorized" in x.lower() for x in p["clarifications_or_gates"]))
        self.assertFalse(p["execution_performed"])

    def test_authorization_is_recorded_but_never_executes(self):
        p = build_research_plan("Estimate Lyapunov exponents.", execution_authorized=True)
        self.assertEqual(p["execution_authorization"], "authorized_by_user")
        self.assertFalse(p["execution_performed"])

    def test_classical_quantum_route(self):
        p = build_research_plan("Study classical-quantum correspondence.")
        self.assertEqual(p["research_intent"], "classical_quantum_correspondence")
        self.assertEqual(p["workflow"][0]["tool"], "run_classical_quantum_comparison_tool")

if __name__ == "__main__":
    unittest.main()
