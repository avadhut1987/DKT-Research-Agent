from __future__ import annotations

import pytest

from dkt_agent.research_director import (
    _extract_json,
    SYSTEM_DECISION_RULES,
    execute_approved_step,
    parse_tool_result,
    validate_decision,
)


def test_extract_json_accepts_fenced_json():
    parsed = _extract_json('Result:\n```json\n{"status":"SUFFICIENT"}\n```')
    assert parsed == {"status": "SUFFICIENT"}


def test_validate_sufficient_decision_stops_without_tool():
    result = validate_decision({
        "status": "SUFFICIENT",
        "conclusion": "The available evidence supports the stated finite-data conclusion.",
        "evidence_assessment": "Existing evidence addresses the requested comparison.",
        "limitations": ["Finite data do not prove an asymptotic property."],
        "next_action": "Record the conclusion.",
        "tool_name": None,
        "arguments": {},
        "execution_mode": "none",
    })
    assert result["status"] == "SUFFICIENT"
    assert result["tool_name"] is None
    assert result["arguments"] == {}


def test_validate_rejects_unapproved_tool_name():
    with pytest.raises(ValueError, match="allow-list"):
        validate_decision({
            "status": "MORE_EVIDENCE_REQUIRED",
            "conclusion": "The evidence is incomplete.",
            "tool_name": "run_arbitrary_python",
            "arguments": {},
            "execution_mode": "numerical",
        })


def test_validate_numerical_proposal_requires_more_evidence_status():
    with pytest.raises(ValueError, match="MORE_EVIDENCE_REQUIRED"):
        validate_decision({
            "status": "SUFFICIENT",
            "conclusion": "Sufficient.",
            "tool_name": "mean_ergodicity_langchain_tool",
            "arguments": {"k_r": 1.0},
            "execution_mode": "numerical",
        })


def test_parse_tool_result_preserves_json_structure():
    result = parse_tool_result('{"status":"success","rows":4}')
    assert result == {"status": "success", "rows": 4}


def test_parse_tool_result_preserves_plain_text():
    result = parse_tool_result("diagnostic completed")
    assert result == {"text": "diagnostic completed"}


def test_numerical_execution_requires_explicit_approval():
    with pytest.raises(PermissionError, match="explicit user approval"):
        execute_approved_step("mean_ergodicity_langchain_tool", {"k_r": 1.0})



def test_director_prioritizes_controlled_mean_ergodicity_robustness_check():
    assert "doubling" in SYSTEM_DECISION_RULES
    assert "This isolates the effect of trajectory length" in SYSTEM_DECISION_RULES
    assert "Do not substitute find_dkt_fixed_points" in SYSTEM_DECISION_RULES


def test_director_must_reconsider_when_user_declines_proposal():
    assert "do not execute\n the same proposal unchanged" in SYSTEM_DECISION_RULES or "do not execute it or repeat" in SYSTEM_DECISION_RULES


def test_director_requires_read_only_retrieval_when_evidence_is_truncated():
    assert "A truncated search preview is not evidence that a result is absent." in SYSTEM_DECISION_RULES
    assert "LYAPUNOV/KS REUSE GATE" in SYSTEM_DECISION_RULES
    assert "AUTONOMOUS INVESTIGATION LOOP" in SYSTEM_DECISION_RULES



def test_reader_character_limit_is_capped_before_schema_validation():
    result = validate_decision({
        "status": "MORE_EVIDENCE_REQUIRED",
        "conclusion": "A matching artifact needs full inspection.",
        "evidence_assessment": "The search result is only a preview.",
        "tool_name": "read_research_evidence_file",
        "arguments": {
            "relative_path": "results/example.json",
            "max_chars": 250000,
        },
        "execution_mode": "read_only",
    })
    assert result["arguments"]["max_chars"] == 40000
