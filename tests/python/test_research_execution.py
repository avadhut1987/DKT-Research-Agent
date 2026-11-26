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


    def test_late_authorization_is_persisted_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Compare ktheta=0 and 10000 at fixed kr=1 for trajectory geometry."
                )
                campaign = execution.create_campaign(plan)
                campaign["stages"][0]["tool"] = "run_dkt_comparison_study"
                campaign["stages"][0]["execution_mode"] = "new_numerical_data"
                self.assertEqual(campaign["execution_authorization"], "not_authorized")

                execution.record_execution_authorization(campaign, authorized=True)
                loaded = execution.load_campaign(campaign["campaign_id"])

                self.assertEqual(loaded["execution_authorization"], "authorized_by_user")
                self.assertIsNotNone(loaded["execution_authorized_at"])
                self.assertTrue(any(
                    event["event"] == "execution_authorization_recorded"
                    and event["decision"] == "authorized_by_user"
                    for event in loaded["events"]
                ))

                class FakeTool:
                    def invoke(self, arguments):
                        return '{"status":"success"}'

                with patch.object(execution, "_resolve_tool", return_value=FakeTool()):
                    updated = execution.execute_campaign_stage(
                        loaded, 0, {"kr": 1.0}, execution_authorized=True
                    )
                self.assertEqual(updated["stages"][0]["status"], "completed")


    def test_authorization_can_be_revoked_even_if_original_plan_was_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Compare ktheta=0 and 10000 at fixed kr=1 for trajectory geometry.",
                    execution_authorized=True,
                )
                campaign = execution.create_campaign(plan)
                self.assertEqual(campaign["execution_authorization"], "authorized_by_user")

                execution.record_execution_authorization(campaign, authorized=False)
                loaded = execution.load_campaign(campaign["campaign_id"])
                self.assertEqual(loaded["execution_authorization"], "not_authorized")
                self.assertIsNone(loaded["execution_authorized_at"])

                loaded["stages"][0]["tool"] = "run_dkt_comparison_study"
                loaded["stages"][0]["execution_mode"] = "new_numerical_data"
                with self.assertRaises(PermissionError):
                    execution.execute_campaign_stage(
                        loaded, 0, {"kr": 1.0}, execution_authorized=True
                    )

    def test_boolean_flag_alone_cannot_bypass_campaign_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Compare ktheta=0 and 10000 at fixed kr=1 for trajectory geometry."
                )
                campaign = execution.create_campaign(plan)
                campaign["stages"][0]["tool"] = "run_dkt_comparison_study"
                campaign["stages"][0]["execution_mode"] = "new_numerical_data"
                with self.assertRaises(PermissionError):
                    execution.execute_campaign_stage(
                        campaign, 0, {"kr": 1.0}, execution_authorized=True
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


    def test_evidence_review_is_required_before_numerical_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Review existing evidence and compare ktheta=0 and 10000 at fixed kr=1 for geometry."
                )
                campaign = execution.create_campaign(plan)
                self.assertEqual(campaign["stages"][0]["tool"], "search_research_evidence")
                campaign["stages"][0]["status"] = "completed"
                campaign["stages"][0]["result"] = {"matches": []}
                execution.save_campaign(campaign)
                with self.assertRaises(PermissionError):
                    execution.execute_campaign_stage(
                        campaign, 1, {"kr": 1.0}, execution_authorized=True
                    )

                execution.record_evidence_review(
                    campaign, proceed_with_new_calculations=True
                )
                execution.record_execution_authorization(campaign, authorized=True)

                class FakeTool:
                    def invoke(self, arguments):
                        return '{"status":"success","checked_arguments":true}'

                with patch.object(execution, "_resolve_tool", return_value=FakeTool()):
                    updated = execution.execute_campaign_stage(
                        campaign, 1, {"kr": 1.0}, execution_authorized=True
                    )
                self.assertEqual(updated["stages"][1]["status"], "completed")
                self.assertEqual(
                    updated["evidence_review"]["decision"],
                    "proceed_with_new_calculations",
                )

    def test_evidence_review_can_stop_new_calculations(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Review existing evidence and compare ktheta=0 and 10000 at fixed kr=1 for geometry."
                )
                campaign = execution.create_campaign(plan)
                campaign["stages"][0]["status"] = "completed"
                execution.save_campaign(campaign)
                execution.record_evidence_review(
                    campaign, proceed_with_new_calculations=False
                )
                self.assertEqual(campaign["status"], "completed_existing_evidence")
                with self.assertRaises(PermissionError):
                    execution.execute_campaign_stage(
                        campaign, 1, {"kr": 1.0}, execution_authorized=True
                    )



    def test_stage_level_evidence_review_can_reuse_trajectory_and_run_geometry_only(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Review existing evidence first. Compare ktheta=0 and 10000 at fixed kr=1 "
                    "for trajectory geometry and matched local clouds."
                )
                campaign = execution.create_campaign(plan)
                self.assertEqual(campaign["stages"][0]["tool"], "search_research_evidence")
                campaign["stages"][0]["status"] = "completed"
                campaign["stages"][0]["result"] = {"matches": ["prior matched trajectory"]}
                execution.save_campaign(campaign)
                execution.record_evidence_review(
                    campaign,
                    stage_decisions={
                        1: {"action": "reuse", "note": "Prior matched comparison: results/old_pair.csv"},
                        2: {"action": "execute", "note": ""},
                    },
                )
                loaded = execution.load_campaign(campaign["campaign_id"])
                self.assertEqual(loaded["stages"][1]["status"], "reused_existing_evidence")
                self.assertEqual(loaded["stages"][2]["status"], "pending")
                self.assertEqual(loaded["evidence_review"]["decision"], "selective_stage_review")

                class FakeTool:
                    def invoke(self, arguments):
                        return '{"status":"success","geometry_only":true}'

                with patch.object(execution, "_resolve_tool", return_value=FakeTool()):
                    updated = execution.execute_campaign_stage(
                        loaded, 2, {"kr": 1.0}, execution_authorized=True
                    )
                self.assertEqual(updated["stages"][2]["status"], "completed")
                self.assertEqual(updated["status"], "completed")
                self.assertTrue(any(
                    event["event"] == "stage_level_evidence_review_recorded"
                    for event in updated["events"]
                ))

    def test_stage_reuse_requires_a_provenance_note(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Review existing evidence first. Compare ktheta=0 and 10000 at fixed kr=1."
                )
                campaign = execution.create_campaign(plan)
                campaign["stages"][0]["status"] = "completed"
                execution.save_campaign(campaign)
                with self.assertRaises(ValueError):
                    execution.record_evidence_review(
                        campaign,
                        stage_decisions={
                            1: {"action": "reuse", "note": ""},
                            2: {"action": "execute", "note": ""},
                        },
                    )

    def test_reused_stage_cannot_be_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution, "CAMPAIGNS_ROOT", Path(directory)):
                plan = build_research_plan(
                    "Review existing evidence first. Compare ktheta=0 and 10000 at fixed kr=1."
                )
                campaign = execution.create_campaign(plan)
                campaign["stages"][0]["status"] = "completed"
                execution.save_campaign(campaign)
                execution.record_evidence_review(
                    campaign,
                    stage_decisions={
                        1: {"action": "reuse", "note": "Prior run 2026-10-01"},
                        2: {"action": "execute", "note": ""},
                    },
                )
                with self.assertRaises(ValueError):
                    execution.execute_campaign_stage(
                        campaign, 1, {"kr": 1.0}, execution_authorized=True
                    )

if __name__ == "__main__":
    unittest.main()
