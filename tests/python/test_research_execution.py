import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dkt_agent.research_orchestrator import build_research_plan
from dkt_agent import research_execution as execution


class TestResearchExecution(unittest.TestCase):
    def test_campaign_persists_plan_and_pending_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Compare ktheta=0 and 10000 at fixed kr=1 for trajectory geometry."
                )
                campaign = execution.create_campaign(plan)
                self.assertEqual(campaign["status"], "planned")
                self.assertEqual(campaign["stages"][0]["status"], "pending")
                loaded = execution.load_campaign(campaign["campaign_id"])
                self.assertEqual(loaded["plan"]["research_intent"], "fixed_kr_ktheta_geometry")
                self.assertEqual(execution.list_campaigns()[0]["campaign_id"], campaign["campaign_id"])

    def test_numerical_stage_cannot_run_without_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Compare ktheta=0 and 10000 at fixed kr=1 for trajectory geometry."
                )
                campaign = execution.create_campaign(plan)
                with self.assertRaises(PermissionError):
                    execution.execute_campaign_stage(
                        campaign, 0, {"kr": 1.0}, execution_authorized=False
                    )
                self.assertEqual(campaign["stages"][0]["status"], "pending")

    def test_unknown_tool_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan("Find DKT fixed points.")
                campaign = execution.create_campaign(plan)
                campaign["stages"][0]["tool"] = "arbitrary_python"
                campaign["stages"][0]["execution_mode"] = "new_numerical_data"
                with self.assertRaises(ValueError):
                    execution.execute_campaign_stage(
                        campaign, 0, {}, execution_authorized=True
                    )


    def test_matched_local_cloud_geometry_diagnostic(self):
        import json
        from dkt_agent import research_tools

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(research_tools, "_new_output_dir", return_value=Path(directory)):
                raw = research_tools.run_dkt_geometry_ensemble_comparison.invoke({
                    "kr": 1.0,
                    "ktheta_values": [0.0, 3.0],
                    "theta0": 0.75,
                    "phi0": -1.0,
                    "delta_theta": 0.05,
                    "delta_phi": 0.05,
                    "grid_size": 3,
                    "nsteps": 8,
                    "p": 1.5707963267948966,
                })
                result = json.loads(raw)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["parameters"]["n_initial_conditions"], 9)
                self.assertLess(result["max_norm_error"], 1e-10)
                self.assertGreater(result["final_matched_cloud_separation"]["rms"], 0.0)
                self.assertTrue(Path(result["artifacts"]["geometry_by_kick_csv"]).is_file())
                self.assertIn("local cloud is an initial-condition probe", " ".join(result["interpretation_limits"]))


if __name__ == "__main__":
    unittest.main()
