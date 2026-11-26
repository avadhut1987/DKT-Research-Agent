"""Controlled, resumable execution of approved DKT research-plan stages.

The execution layer deliberately exposes a small allow-list of validated tools.
Read-only discovery is always permitted; numerical tools require explicit
authorization and an explicit stage execution action in the Research Studio.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS_ROOT = PROJECT_ROOT / "results" / "research_campaigns"

# Keep the execution surface explicit. Never resolve arbitrary tool names from
# user/model text or dynamically import a module based on a plan field.
TOOL_MODES = {
    "search_research_evidence": "read_only",
    "read_research_csv_tool": "read_only",
    "summarize_research_csv": "read_only",
    "run_dkt_comparison_study": "new_numerical_data",
    "run_dkt_geometry_ensemble_comparison": "new_numerical_data",
    "mean_ergodicity_langchain_tool": "new_numerical_data",
    "run_ergodicity_variance_study": "new_numerical_data",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_path(campaign_id: str) -> Path:
    if not isinstance(campaign_id, str) or not campaign_id.startswith("dkt-"):
        raise ValueError("Invalid research campaign identifier.")
    if Path(campaign_id).name != campaign_id:
        raise ValueError("Invalid research campaign identifier.")
    return CAMPAIGNS_ROOT / f"{campaign_id}.json"


def _save(campaign: dict[str, Any]) -> None:
    CAMPAIGNS_ROOT.mkdir(parents=True, exist_ok=True)
    path = _campaign_path(campaign["campaign_id"])
    fd, temporary = tempfile.mkstemp(prefix=".campaign-", suffix=".tmp", dir=CAMPAIGNS_ROOT)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(campaign, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_campaign(plan: dict[str, Any]) -> dict[str, Any]:
    """Persist an immutable plan snapshot and a pending stage ledger."""
    if not isinstance(plan, dict) or plan.get("status") != "plan_created":
        raise ValueError("A valid research plan is required.")
    campaign_id = "dkt-" + uuid.uuid4().hex[:12]
    campaign = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "planned",
        "plan": plan,
        "execution_authorization": (
            "authorized_by_user"
            if plan.get("execution_authorization") == "authorized_by_user"
            else "not_authorized"
        ),
        "execution_authorized_at": _now()
        if plan.get("execution_authorization") == "authorized_by_user"
        else None,
        "stages": [
            {
                "index": i,
                "tool": step["tool"],
                "label": step["action"],
                "execution_mode": step["execution_mode"],
                "status": "pending",
                "attempts": 0,
                "input_arguments": None,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
            }
            for i, step in enumerate(plan.get("workflow", []))
        ],
        "events": [{"at": _now(), "event": "campaign_created"}],
    }
    _save(campaign)
    return campaign


def load_campaign(campaign_id: str) -> dict[str, Any]:
    path = _campaign_path(campaign_id)
    with path.open("r", encoding="utf-8") as handle:
        campaign = json.load(handle)
    if campaign.get("schema_version") != 1:
        raise ValueError("Unsupported research campaign schema version.")
    return campaign


def list_campaigns(limit: int = 20) -> list[dict[str, str]]:
    """Return compact campaign metadata, newest first."""
    if not CAMPAIGNS_ROOT.is_dir():
        return []
    rows = []
    for path in sorted(CAMPAIGNS_ROOT.glob("dkt-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open("r", encoding="utf-8") as handle:
                item = json.load(handle)
            rows.append({
                "campaign_id": item["campaign_id"],
                "status": item.get("status", "unknown"),
                "research_question": item.get("plan", {}).get("research_question", ""),
                "execution_authorization": item.get(
                    "execution_authorization",
                    item.get("plan", {}).get("execution_authorization", "not_authorized"),
                ),
                "evidence_review": item.get("evidence_review", {}).get("decision", "pending"),
                "updated_at": item.get("updated_at", ""),
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(rows) >= limit:
            break
    return rows


def _resolve_tool(tool_name: str):
    """Return one explicitly supported LangChain tool."""
    if tool_name not in TOOL_MODES:
        raise ValueError(f"Tool is not enabled for Research Studio execution: {tool_name}")
    if tool_name in {"search_research_evidence", "read_research_csv_tool",
                     "run_dkt_comparison_study"}:
        from dkt_agent import langchain_tools
        return getattr(langchain_tools, tool_name)
    if tool_name in {"summarize_research_csv", "run_ergodicity_variance_study",
                     "run_dkt_geometry_ensemble_comparison"}:
        from dkt_agent import research_tools
        return getattr(research_tools, tool_name)
    if tool_name == "mean_ergodicity_langchain_tool":
        from dkt_agent.mean_ergodicity_tools import mean_ergodicity_langchain_tool
        return mean_ergodicity_langchain_tool
    raise ValueError(f"No execution adapter registered for {tool_name}")


def execute_campaign_stage(
    campaign: dict[str, Any],
    stage_index: int,
    arguments: dict[str, Any],
    *,
    execution_authorized: bool = False,
) -> dict[str, Any]:
    """Execute exactly the next pending stage, recording inputs, result and provenance."""
    if not isinstance(campaign, dict) or not isinstance(arguments, dict):
        raise ValueError("Campaign and stage arguments must be dictionaries.")
    stages = campaign.get("stages", [])
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or not 0 <= stage_index < len(stages):
        raise ValueError("Invalid research stage index.")
    stage = stages[stage_index]
    if stage["status"] == "completed":
        raise ValueError("This research stage is already complete.")
    if stage["status"] == "running":
        raise ValueError("This research stage is already marked as running.")
    if any(s["status"] != "completed" for s in stages[:stage_index]):
        raise ValueError("Complete earlier workflow stages before executing this stage.")
    tool_name = stage["tool"]
    mode = TOOL_MODES.get(tool_name)
    evidence_was_searched = any(
        item.get("tool") == "search_research_evidence" and item.get("status") == "completed"
        for item in stages[:stage_index]
    )
    if mode == "new_numerical_data" and evidence_was_searched:
        review = campaign.get("evidence_review", {}).get("decision")
        if review != "proceed_with_new_calculations":
            raise PermissionError("Review the read-only evidence and explicitly decide whether new calculations are still needed.")
    if mode is None or mode != stage["execution_mode"]:
        raise ValueError("The planned tool/mode is not enabled by the execution policy.")
    campaign_authorized = (
        campaign.get("execution_authorization") == "authorized_by_user"
        or campaign.get("plan", {}).get("execution_authorization") == "authorized_by_user"
    )
    if mode == "new_numerical_data" and (
        not execution_authorized or not campaign_authorized
    ):
        raise PermissionError(
            "Explicit user authorization must be recorded in the campaign before "
            "new numerical data can be generated."
        )
    if mode == "read_only" and tool_name not in TOOL_MODES:
        raise ValueError("Read-only tool is not allow-listed.")

    tool = _resolve_tool(tool_name)
    stage.update({
        "status": "running", "attempts": int(stage.get("attempts", 0)) + 1,
        "input_arguments": arguments, "started_at": _now(), "error": None,
    })
    campaign["status"] = "running"
    campaign["updated_at"] = _now()
    campaign.setdefault("events", []).append({
        "at": _now(), "event": "stage_started", "stage_index": stage_index,
        "tool": tool_name, "execution_mode": mode,
    })
    _save(campaign)
    try:
        raw_result = tool.invoke(arguments)
        if isinstance(raw_result, str):
            try:
                result: Any = json.loads(raw_result)
            except json.JSONDecodeError:
                result = raw_result
        else:
            result = raw_result
        stage.update({"status": "completed", "completed_at": _now(), "result": result})
        campaign["status"] = "completed" if all(s["status"] == "completed" for s in stages) else "paused"
        campaign["updated_at"] = _now()
        campaign["events"].append({
            "at": _now(), "event": "stage_completed", "stage_index": stage_index,
            "tool": tool_name,
        })
        _save(campaign)
        return campaign
    except Exception as exc:
        stage.update({"status": "failed", "completed_at": _now(), "error": f"{type(exc).__name__}: {exc}"})
        campaign["status"] = "failed"
        campaign["updated_at"] = _now()
        campaign["events"].append({
            "at": _now(), "event": "stage_failed", "stage_index": stage_index,
            "tool": tool_name, "error": stage["error"],
        })
        _save(campaign)
        raise


def record_execution_authorization(
    campaign: dict[str, Any], *, authorized: bool
) -> None:
    """Persist an explicit user authorization decision in the campaign audit log."""
    if not isinstance(authorized, bool):
        raise ValueError("Execution authorization must be boolean.")
    if not isinstance(campaign, dict) or "campaign_id" not in campaign:
        raise ValueError("A valid research campaign is required.")

    decision = "authorized_by_user" if authorized else "not_authorized"
    previous = campaign.get("execution_authorization", "not_authorized")
    if previous == decision:
        return

    campaign["execution_authorization"] = decision
    campaign["execution_authorized_at"] = _now() if authorized else None
    campaign.setdefault("events", []).append({
        "at": _now(),
        "event": "execution_authorization_recorded",
        "decision": decision,
    })
    campaign["updated_at"] = _now()
    _save(campaign)


def record_evidence_review(campaign: dict[str, Any], *, proceed_with_new_calculations: bool) -> None:
    """Persist a human review decision after read-only evidence discovery."""
    if not isinstance(proceed_with_new_calculations, bool):
        raise ValueError("Evidence-review decision must be boolean.")
    evidence_stages = [
        stage for stage in campaign.get("stages", [])
        if stage.get("tool") == "search_research_evidence"
    ]
    if not evidence_stages or any(stage.get("status") != "completed" for stage in evidence_stages):
        raise ValueError("Complete read-only evidence discovery before recording its review.")
    decision = (
        "proceed_with_new_calculations"
        if proceed_with_new_calculations else "stop_existing_evidence_sufficient"
    )
    campaign["evidence_review"] = {"decision": decision, "reviewed_at": _now()}
    campaign.setdefault("events", []).append({
        "at": _now(), "event": "evidence_review_recorded", "decision": decision,
    })
    campaign["status"] = "paused" if proceed_with_new_calculations else "completed_existing_evidence"
    _save(campaign)


def save_campaign(campaign: dict[str, Any]) -> None:
    """Persist UI-side campaign metadata changes without changing stage results."""
    if not isinstance(campaign, dict) or "campaign_id" not in campaign:
        raise ValueError("A valid campaign is required.")
    campaign["updated_at"] = _now()
    _save(campaign)
