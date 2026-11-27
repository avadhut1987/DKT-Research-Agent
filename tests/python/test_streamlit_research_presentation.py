"""Tests for researcher-facing Research Studio presentation helpers."""
import tempfile
import unittest
from pathlib import Path

from dkt_agent.research_studio import (
    _duration_seconds,
    _format_duration,
    _result_artifacts,
)


class ResearchStudioPresentationTests(unittest.TestCase):
    def test_duration_is_computed_from_saved_timestamps(self):
        duration = _duration_seconds(
            "2026-11-26T20:14:24.426559+00:00",
            "2026-11-26T20:14:31.903092+00:00",
        )
        self.assertAlmostEqual(duration, 7.476533, places=5)

    def test_invalid_or_missing_timestamps_are_safe(self):
        self.assertIsNone(_duration_seconds(None, None))
        self.assertIsNone(_duration_seconds("not-a-date", "also-not-a-date"))

    def test_duration_formatting_is_readable(self):
        self.assertEqual(_format_duration(None), "Not available")
        self.assertEqual(_format_duration(2.5), "2.50 s")
        self.assertEqual(_format_duration(125), "2 min 5 s")

    def test_artifact_discovery_finds_existing_nested_csv_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "series.csv"
            json_path = Path(directory) / "summary.json"
            csv_path.write_text("kick,value\\n0,1\\n", encoding="utf-8")
            json_path.write_text('{"status":"success"}', encoding="utf-8")
            found = _result_artifacts({
                "artifacts": {
                    "series": str(csv_path),
                    "summary": str(json_path),
                    "missing": str(Path(directory) / "missing.csv"),
                }
            })
            self.assertEqual({Path(path).name for _, path in found}, {"series.csv", "summary.json"})
            self.assertEqual(len(found), 2)

    def test_artifact_discovery_deduplicates_repeated_file_references(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "trajectory.csv"
            csv_path.write_text("step,distance\\n0,0\\n", encoding="utf-8")
            found = _result_artifacts({
                "artifacts": {"primary": str(csv_path)},
                "files": [str(csv_path)],
                "paths": {"copy": str(csv_path)},
            })
            self.assertEqual(len(found), 1)
            self.assertEqual(Path(found[0][1]).name, "trajectory.csv")




class StructuredRenderingDataTests(unittest.TestCase):
    def test_scalar_rows_format_booleans_none_and_numeric_values(self):
        from dkt_agent.streamlit_rendering import scalar_rows

        rows = scalar_rows({
            "status": "success",
            "authorized": True,
            "optional": None,
            "max_norm_error": 9.547918011776346e-15,
            "ktheta_values": [0, 10000],
        })
        self.assertEqual(rows[0], {"Field": "Status", "Value": "success"})
        self.assertEqual(rows[1], {"Field": "Authorized", "Value": "Yes"})
        self.assertEqual(rows[2], {"Field": "Optional", "Value": "—"})
        self.assertEqual(rows[3]["Value"], "9.547918e-15")
        self.assertEqual(rows[4]["Value"], "0, 10000")

    def test_record_lists_flatten_nested_parameters_for_tables(self):
        from dkt_agent.streamlit_rendering import records_to_frame

        frame = records_to_frame([
            {"stage": 1, "parameters": {"kr": 1, "ktheta": 0}},
            {"stage": 2, "parameters": {"kr": 1, "ktheta": 10000}},
        ])
        self.assertEqual(list(frame.columns), ["Stage", "Parameters · kr", "Parameters · ktheta"])
        self.assertEqual(frame.iloc[0]["Parameters · Kr"], 1)
        self.assertEqual(frame.iloc[1]["Parameters · Ktheta"], 10000)


if __name__ == "__main__":
    unittest.main()
