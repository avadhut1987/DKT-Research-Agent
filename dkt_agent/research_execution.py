"""Controlled, resumable execution of approved DKT research-plan stages.

The execution layer deliberately exposes a small allow-list of validated tools.
Read-only discovery is always permitted; numerical tools require explicit
authorization and an explicit stage execution action in the Research Studio.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS_ROOT = PROJECT_ROOT / "results" / "research_campaigns"

# Streamlit keeps its ScriptRunner thread alive between reruns. Thread liveness
# therefore cannot tell us whether a stage tool is still executing. Track the
# actual in-process tool invocation by attempt ID instead.
_ACTIVE_ATTEMPTS: set[str] = set()
_ACTIVE_ATTEMPTS_LOCK = threading.Lock()


def _mark_attempt_active(attempt_id: str) -> None:
    with _ACTIVE_ATTEMPTS_LOCK:
        _ACTIVE_ATTEMPTS.add(attempt_id)


def _mark_attempt_inactive(attempt_id: str) -> None:
    with _ACTIVE_ATTEMPTS_LOCK:
        _ACTIVE_ATTEMPTS.discard(attempt_id)


def _is_attempt_active(attempt_id: str | None) -> bool:
    if not attempt_id:
        return False
    with _ACTIVE_ATTEMPTS_LOCK:
        return attempt_id in _ACTIVE_ATTEMPTS

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


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def is_stage_terminal(status: str) -> bool:
    """Statuses that satisfy workflow dependencies without another execution."""
    return status in {"completed", "reused_existing_evidence"}


def derive_campaign_status(campaign: dict[str, Any]) -> str:
    """Derive lifecycle status from stage execution and post-execution review."""
    stages = campaign.get("stages", [])
    statuses = [stage.get("status", "pending") for stage in stages]
    if "running" in statuses:
        return "running"
    if "failed" in statuses:
        return "failed"
    if "interrupted" in statuses:
        return "interrupted"
    if not all(is_stage_terminal(status) for status in statuses):
        evidence_done = any(
            stage.get("tool") == "search_research_evidence"
            and stage.get("status") == "completed"
            for stage in stages
        )
        numerical_pending = any(
            stage.get("execution_mode") == "new_numerical_data"
            and stage.get("status") in {"pending", "failed", "interrupted"}
            for stage in stages
        )
        if evidence_done and numerical_pending and not campaign.get("evidence_review"):
            return "awaiting_evidence_review"
        return "planned" if all(status == "pending" for status in statuses) else "paused"

    has_numerical_work = any(
        stage.get("execution_mode") == "new_numerical_data" for stage in stages
    )
    if not has_numerical_work:
        return "completed"
    review = campaign.get("scientific_review") or {}
    decision = review.get("decision")
    if decision == "accepted":
        return "completed"
    if decision == "more_work_required":
        return "review_requires_follow_up"
    if decision == "inconclusive":
        return "review_inconclusive"
    return "awaiting_scientific_review"




def inspect_stage_liveness(
    stage: dict[str, Any], *, stale_after_seconds: int = 600
) -> dict[str, Any]:
    """Distinguish an active stage from a process/thread that has disappeared.

    New records carry host, PID and Streamlit script-thread identity. Legacy
    records fall back to an age-based check because they have no owner metadata.
    """
    if stage.get("status") != "running":
        return {"running": False, "stale": False, "reason": "stage_not_running"}

    host = stage.get("runner_host")
    pid = stage.get("runner_pid")
    attempt_id = stage.get("active_attempt_id")
    if host == socket.gethostname() and pid == os.getpid():
        # Streamlit reuses its script thread across reruns. Only an actual
        # tool invocation, not a live thread, means the stage is still active.
        active = _is_attempt_active(attempt_id)
        return {
            "running": True,
            "stale": not active,
            "reason": "tool_invocation_active" if active else "tool_invocation_not_active",
        }

    started = _parse_timestamp(stage.get("started_at"))
    age = (datetime.now(timezone.utc) - started).total_seconds() if started else None
    if host == socket.gethostname() and isinstance(pid, int) and pid != os.getpid():
        try:
            os.kill(pid, 0)
            process_alive = True
        except ProcessLookupError:
            process_alive = False
        except PermissionError:
            process_alive = True
        if not process_alive:
            return {"running": True, "stale": True, "reason": "runner_process_missing", "age_seconds": age}
        return {"running": True, "stale": False, "reason": "runner_process_still_alive", "age_seconds": age}
    stale = age is not None and age >= max(1, stale_after_seconds)
    return {
        "running": True, "stale": stale,
        "reason": "legacy_or_remote_runner_timeout" if stale else "runner_may_still_be_active",
        "age_seconds": age,
    }


def recover_interrupted_stage(
    campaign: dict[str, Any], stage_index: int, *, stale_after_seconds: int = 600
) -> dict[str, Any]:
    """Record a dead/stale running attempt as interrupted; never rerun it."""
    stages = campaign.get("stages", [])
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or not 0 <= stage_index < len(stages):
        raise ValueError("Invalid research stage index.")
    stage = stages[stage_index]
    liveness = inspect_stage_liveness(stage, stale_after_seconds=stale_after_seconds)
    if not liveness["running"]:
        raise ValueError("This stage is not marked as running.")
    if not liveness["stale"]:
        raise RuntimeError("The stage runner may still be active; recovery is not safe yet.")

    recovered_at = _now()
    attempt_id = stage.get("active_attempt_id")
    for attempt in stage.get("attempt_history", []):
        if attempt.get("attempt_id") == attempt_id and attempt.get("status") == "running":
            attempt.update({
                "status": "interrupted", "completed_at": recovered_at,
                "error": "Runner disappeared before the stage completion was persisted.",
            })
    stage.update({
        "status": "interrupted", "completed_at": recovered_at,
        "error": "Execution was interrupted before completion could be recorded. Review the attempt history before retrying.",
        "runner_pid": None, "runner_host": None, "runner_thread_id": None,
        "active_attempt_id": None,
    })
    campaign["status"] = "interrupted"
    campaign["updated_at"] = recovered_at
    campaign.setdefault("events", []).append({
        "at": recovered_at, "event": "stage_interruption_recovered",
        "stage_index": stage_index, "liveness": liveness,
    })
    _save(campaign)
    return campaign


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
        "scientific_review": None,
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
                "attempt_history": [],
                "active_attempt_id": None,
                "runner_pid": None,
                "runner_host": None,
                "runner_thread_id": None,
                "input_arguments": None,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
            }
            for i, step in enumerate(plan.get("workflow", []))
        ],
        "events": [
            {"at": _now(), "event": "campaign_created"},
            *(
                [{
                    "at": _now(),
                    "event": "execution_authorization_recorded",
                    "decision": "authorized_by_user",
                }]
                if plan.get("execution_authorization") == "authorized_by_user"
                else []
            ),
        ],
    }
    _save(campaign)
    return campaign


def load_campaign(campaign_id: str) -> dict[str, Any]:
    path = _campaign_path(campaign_id)
    with path.open("r", encoding="utf-8") as handle:
        campaign = json.load(handle)
    if campaign.get("schema_version") != 1:
        raise ValueError("Unsupported research campaign schema version.")
    # Normalize legacy campaigns whose stage ledger was complete while the
    # separate scientific review had never been recorded.
    derived_status = derive_campaign_status(campaign)
    if campaign.get("status") != derived_status:
        campaign["status"] = derived_status
        campaign["updated_at"] = _now()
        _save(campaign)
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
                "status": derive_campaign_status(item),
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
    """Execute the next pending stage, respecting stage-level evidence decisions."""
    if not isinstance(campaign, dict) or not isinstance(arguments, dict):
        raise ValueError("Campaign and stage arguments must be dictionaries.")
    stages = campaign.get("stages", [])
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or not 0 <= stage_index < len(stages):
        raise ValueError("Invalid research stage index.")
    stage = stages[stage_index]
    terminal_statuses = {"completed", "reused_existing_evidence"}
    if stage["status"] in terminal_statuses:
        raise ValueError("This research stage is already complete or covered by existing evidence.")
    if stage["status"] == "running":
        raise ValueError("This research stage is already marked as running.")
    if any(s["status"] not in terminal_statuses for s in stages[:stage_index]):
        raise ValueError("Complete or review earlier workflow stages before executing this stage.")
    if stage.get("review_decision", {}).get("action") == "reuse":
        raise PermissionError("This stage was marked as covered by existing evidence and cannot be executed.")
    tool_name = stage["tool"]
    mode = TOOL_MODES.get(tool_name)
    evidence_was_searched = any(
        item.get("tool") == "search_research_evidence" and item.get("status") == "completed"
        for item in stages[:stage_index]
    )
    if mode == "new_numerical_data" and evidence_was_searched:
        review = campaign.get("evidence_review", {}).get("decision")
        stage_review = stage.get("review_decision", {})
        if stage_review:
            if stage_review.get("action") != "execute":
                raise PermissionError("This stage was not approved for new numerical execution in the evidence review.")
        elif review != "proceed_with_new_calculations":
            raise PermissionError("Review the read-only evidence and explicitly decide whether new calculations are still needed.")
    if mode is None or mode != stage["execution_mode"]:
        raise ValueError("The planned tool/mode is not enabled by the execution policy.")
    campaign_authorized = (
        campaign.get("execution_authorization") == "authorized_by_user"
        if "execution_authorization" in campaign
        else campaign.get("plan", {}).get("execution_authorization") == "authorized_by_user"
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
    attempt_number = int(stage.get("attempts", 0)) + 1
    attempt_id = uuid.uuid4().hex
    started_at = _now()
    attempt = {
        "attempt_id": attempt_id, "attempt_number": attempt_number,
        "status": "running", "started_at": started_at,
        "completed_at": None, "input_arguments": arguments,
        "runner_pid": os.getpid(), "runner_host": socket.gethostname(),
        "runner_thread_id": threading.get_ident(), "error": None,
    }
    stage.setdefault("attempt_history", []).append(attempt)
    stage.update({
        "status": "running", "attempts": attempt_number,
        "active_attempt_id": attempt_id,
        "runner_pid": os.getpid(), "runner_host": socket.gethostname(),
        "runner_thread_id": threading.get_ident(),
        "input_arguments": arguments, "started_at": started_at,
        "completed_at": None, "result": None, "error": None,
    })
    campaign["status"] = "running"
    campaign["updated_at"] = _now()
    campaign.setdefault("events", []).append({
        "at": _now(), "event": "stage_started", "stage_index": stage_index,
        "tool": tool_name, "execution_mode": mode,
    })
    _save(campaign)
    _mark_attempt_active(attempt_id)
    try:
        raw_result = tool.invoke(arguments)
        if isinstance(raw_result, str):
            try:
                result: Any = json.loads(raw_result)
            except json.JSONDecodeError:
                result = raw_result
        else:
            result = raw_result
        if isinstance(result, dict) and (
            str(result.get("status", "")).lower() in {"error", "failed", "failure"}
            or result.get("success") is False
        ):
            detail = result.get("error") or result.get("message") or result.get("status")
            raise RuntimeError(f"Research tool reported failure: {detail}")
        completed_at = _now()
        attempt.update({"status": "completed", "completed_at": completed_at, "result": result})
        stage.update({
            "status": "completed", "completed_at": completed_at, "result": result,
            "error": None, "runner_pid": None, "runner_host": None,
            "runner_thread_id": None, "active_attempt_id": None,
        })
        campaign["status"] = derive_campaign_status(campaign)
        campaign["updated_at"] = _now()
        campaign["events"].append({
            "at": _now(), "event": "stage_completed", "stage_index": stage_index,
            "tool": tool_name,
        })
        _save(campaign)
        return campaign
    except Exception as exc:
        failed_at = _now()
        failure = f"{type(exc).__name__}: {exc}"
        attempt.update({"status": "failed", "completed_at": failed_at, "error": failure})
        stage.update({
            "status": "failed", "completed_at": failed_at, "error": failure,
            "runner_pid": None, "runner_host": None,
            "runner_thread_id": None, "active_attempt_id": None,
        })
        campaign["status"] = "failed"
        campaign["updated_at"] = _now()
        campaign["events"].append({
            "at": _now(), "event": "stage_failed", "stage_index": stage_index,
            "tool": tool_name, "error": stage["error"],
        })
        _save(campaign)
        raise
    finally:
        _mark_attempt_inactive(attempt_id)


def record_execution_authorization(
    campaign: dict[str, Any], *, authorized: bool
) -> None:
    """Persist an explicit user authorization decision in the campaign audit log."""
    if not isinstance(authorized, bool):
        raise ValueError("Execution authorization must be boolean.")
    if not isinstance(campaign, dict) or "campaign_id" not in campaign:
        raise ValueError("A valid research campaign is required.")

    decision = "authorized_by_user" if authorized else "not_authorized"
    previous = campaign.get(
        "execution_authorization",
        campaign.get("plan", {}).get("execution_authorization", "not_authorized"),
    )
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


def record_evidence_review(
    campaign: dict[str, Any],
    *,
    proceed_with_new_calculations: bool | None = None,
    stage_decisions: dict[int, dict[str, str]] | None = None,
) -> None:
    """Persist either a legacy campaign-wide or a selective per-stage evidence review."""
    evidence_stages = [
        stage for stage in campaign.get("stages", [])
        if stage.get("tool") == "search_research_evidence"
    ]
    if not evidence_stages or any(stage.get("status") != "completed" for stage in evidence_stages):
        raise ValueError("Complete read-only evidence discovery before recording its review.")

    numerical_stages = [
        stage for stage in campaign.get("stages", [])
        if stage.get("execution_mode") == "new_numerical_data"
    ]
    if stage_decisions is not None:
        expected = {stage["index"] for stage in numerical_stages}
        if set(stage_decisions) != expected:
            raise ValueError("Provide an evidence decision for every planned numerical stage.")
        normalized: dict[str, dict[str, str]] = {}
        for stage in numerical_stages:
            choice = stage_decisions[stage["index"]]
            action = choice.get("action")
            note = str(choice.get("note", "")).strip()
            if action not in {"execute", "reuse"}:
                raise ValueError("Each stage decision must be 'execute' or 'reuse'.")
            if action == "reuse" and not note:
                raise ValueError("A source/reference note is required when reusing existing evidence.")
            stage["review_decision"] = {"action": action, "note": note, "reviewed_at": _now()}
            if action == "reuse":
                stage.update({
                    "status": "reused_existing_evidence",
                    "completed_at": _now(),
                    "result": {"evidence_reuse_note": note, "generated_by_this_campaign": False},
                    "error": None,
                })
            normalized[str(stage["index"])] = {"action": action, "note": note}

        execute_any = any(item["action"] == "execute" for item in normalized.values())
        decision = "selective_stage_review" if execute_any else "stop_existing_evidence_sufficient"
        campaign["evidence_review"] = {
            "decision": decision, "stage_decisions": normalized, "reviewed_at": _now()
        }
        campaign.setdefault("events", []).append({
            "at": _now(), "event": "stage_level_evidence_review_recorded",
            "stage_decisions": normalized,
        })
        campaign["status"] = "paused" if execute_any else "completed_existing_evidence"
    else:
        if not isinstance(proceed_with_new_calculations, bool):
            raise ValueError("Evidence-review decision must be boolean.")
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


def build_campaign_validation_report(campaign: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic record/provenance checks without judging scientific truth."""
    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, str]] = []
    stages = campaign.get("stages", [])
    if not stages:
        blockers.append("The campaign has no planned stages.")
    for stage in stages:
        number = stage.get("index", 0) + 1
        status = stage.get("status", "unknown")
        label = stage.get("label", f"Stage {number}")
        if not is_stage_terminal(status):
            blockers.append(f"Stage {number} ({label}) is not complete: {status}.")
            continue
        if status == "reused_existing_evidence":
            note = stage.get("review_decision", {}).get("note") or stage.get("result", {}).get("evidence_reuse_note")
            if not str(note or "").strip():
                blockers.append(f"Stage {number} is marked as reused but has no evidence reference.")
            else:
                checks.append({"Check": f"Stage {number} evidence reuse", "Status": "Pass", "Detail": "A provenance note is recorded."})
            continue
        result = stage.get("result")
        if result is None:
            blockers.append(f"Stage {number} is marked completed but has no saved result.")
            continue
        if isinstance(result, dict) and (
            str(result.get("status", "")).lower() in {"error", "failed", "failure"}
            or result.get("success") is False
        ):
            blockers.append(f"Stage {number} contains a tool-reported failure payload.")
            continue
        checks.append({"Check": f"Stage {number} saved result", "Status": "Pass", "Detail": "A result is present and does not report failure."})
        if stage.get("execution_mode") == "new_numerical_data":
            inputs = stage.get("input_arguments")
            if not isinstance(inputs, dict) or not inputs:
                blockers.append(f"Stage {number} has no recorded numerical inputs.")
            else:
                checks.append({"Check": f"Stage {number} input provenance", "Status": "Pass", "Detail": "The exact submitted inputs are saved."})
            history = stage.get("attempt_history", [])
            if history and history[-1].get("status") != "completed":
                warnings.append(f"Stage {number} latest attempt-history entry is not marked completed.")
            params = None
            if isinstance(result, dict):
                params = result.get("parameters") or result.get("study_parameters")
            if isinstance(inputs, dict) and isinstance(params, dict):
                mismatches = [
                    key for key, value in inputs.items()
                    if key in params and params[key] != value
                ]
                if mismatches:
                    blockers.append(
                        f"Stage {number} returned parameters disagree with submitted inputs for: {', '.join(mismatches)}."
                    )
                elif any(key in params for key in inputs):
                    checks.append({"Check": f"Stage {number} parameter cross-check", "Status": "Pass", "Detail": "Overlapping submitted and reported parameters agree exactly."})
                else:
                    warnings.append(f"Stage {number} did not report parameter fields that can be cross-checked against its inputs.")
            elif isinstance(inputs, dict):
                warnings.append(f"Stage {number} result does not include a parameter block for independent input cross-checking.")

    def collect_artifacts(value: Any, found: set[str]) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect_artifacts(child, found)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_artifacts(child, found)
        elif isinstance(value, str) and value.lower().endswith((".csv", ".json")):
            path = Path(value).expanduser()
            if path.is_file():
                found.add(str(path.resolve()))
            else:
                warnings.append(f"Reported artifact path is not accessible in this runtime: {value}")

    artifacts: set[str] = set()
    for stage in stages:
        collect_artifacts(stage.get("result"), artifacts)
    checks.append({
        "Check": "Artifact references",
        "Status": "Pass" if artifacts else "Review",
        "Detail": f"{len(artifacts)} distinct reported CSV/JSON artifact(s) are accessible." if artifacts else "No accessible CSV/JSON artifacts were discovered; inspect stage outputs and provenance.",
    })
    if not artifacts and any(s.get("execution_mode") == "new_numerical_data" for s in stages):
        warnings.append("No accessible numerical output files were found by the automatic artifact scan.")
    return {
        "status": "ready_for_scientific_review" if not blockers else "blocked",
        "ready_for_review": not blockers,
        "checks": checks,
        "blockers": blockers,
        "warnings": list(dict.fromkeys(warnings)),
        "accessible_artifacts": sorted(artifacts),
        "scope_note": (
            "These are deterministic record, status, parameter and file-access checks. "
            "They do not establish ergodicity, mixing, chaoticity, convergence, or the scientific sufficiency of the evidence."
        ),
    }


def record_scientific_review(
    campaign: dict[str, Any], *, decision: str, notes: str
) -> None:
    """Record an explicit post-execution scientific review decision and rationale."""
    allowed = {"accepted", "more_work_required", "inconclusive"}
    if decision not in allowed:
        raise ValueError("Scientific review decision must be accepted, more_work_required, or inconclusive.")
    note = str(notes or "").strip()
    if not note:
        raise ValueError("A review rationale is required.")
    report = build_campaign_validation_report(campaign)
    if not report["ready_for_review"]:
        raise ValueError("Resolve campaign validation blockers before recording scientific review.")
    if not all(is_stage_terminal(stage.get("status", "")) for stage in campaign.get("stages", [])):
        raise ValueError("All planned stages must be complete before scientific review.")
    reviewed_at = _now()
    campaign["scientific_review"] = {
        "decision": decision,
        "notes": note,
        "reviewed_at": reviewed_at,
        "validation_status": report["status"],
    }
    campaign.setdefault("events", []).append({
        "at": reviewed_at,
        "event": "scientific_review_recorded",
        "decision": decision,
        "notes": note,
        "validation_status": report["status"],
    })
    campaign["status"] = derive_campaign_status(campaign)
    campaign["updated_at"] = reviewed_at
    _save(campaign)


def save_campaign(campaign: dict[str, Any]) -> None:
    """Persist UI-side campaign metadata changes without changing stage results."""
    if not isinstance(campaign, dict) or "campaign_id" not in campaign:
        raise ValueError("A valid campaign is required.")
    campaign["updated_at"] = _now()
    _save(campaign)
