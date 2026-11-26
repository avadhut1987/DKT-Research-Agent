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


if __name__ == "__main__":
    unittest.main()
