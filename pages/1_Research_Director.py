"""Streamlit page for the semi-autonomous DKT Research Director."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import streamlit as st

from dkt_agent.presentation import format_json, normalize_math_notation

from dkt_agent.research_director import (
    decide_next_step,
    execute_approved_step,
    parse_tool_result,
    search_existing_evidence,
    MAX_DECISION_STEPS,
)


st.set_page_config(
    page_title="DKT Research Director",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 DKT Research Director")
st.caption(
    "Evidence first · one justified next step at a time · supervised numerical execution"
)

CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / ".streamlit" / "research_director_checkpoint.json"
DRAFT_PATH = Path(__file__).resolve().parents[1] / ".streamlit" / "research_director_question_draft.json"


def _render_math(value: object) -> str:
    """Return prose with DKT mathematical notation rendered as LaTeX."""
    return normalize_math_notation(value)


def _json_text(value: object) -> str:
    """Serialize a strict, standards-compliant JSON download."""
    return format_json(value)


def _looks_like_json(value: str) -> bool:
    candidate = value.strip()
    if not candidate.startswith(("{", "[")):
        return False
    try:
        json.loads(candidate)
        return True
    except (ValueError, TypeError):
        return False


def _display_value(value: object) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        if _looks_like_json(value):
            return "Structured machine-readable content (hidden in the normal view)."
        return value
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_display_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_display_value(item) for item in value)
    return str(value)


def _render_parameter_table(parameters: object) -> None:
    """Present tool inputs as a readable parameter table, never as raw JSON."""
    if not isinstance(parameters, dict) or not parameters:
        st.caption("No additional settings.")
        return
    rows = [
        {"Parameter": str(key).replace("_", " "), "Value": _display_value(value)}
        for key, value in parameters.items()
    ]
    st.table(rows)


def _render_readable_result(value: object, heading: str = "Result") -> None:
    """Render tool results as prose/tables without exposing serialized JSON."""
    if isinstance(value, str):
        st.markdown(_render_math(value))
        return
    if not isinstance(value, dict):
        st.write(value)
        return

    preferred_text = ("text", "summary", "interpretation", "message", "note")
    rendered = set()
    for key in preferred_text:
        text_value = value.get(key)
        if isinstance(text_value, str) and text_value.strip() and not _looks_like_json(text_value):
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            st.markdown(_render_math(text_value))
            rendered.add(key)

    table_rows = []
    for key, item in value.items():
        if key in rendered or item is None:
            continue
        if isinstance(item, list) and item and all(isinstance(row, dict) for row in item):
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            safe_rows = [
                {str(column): _display_value(cell) for column, cell in row.items()}
                for row in item
            ]
            st.dataframe(safe_rows, use_container_width=True, hide_index=True)
        elif isinstance(item, dict):
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            _render_parameter_table(item)
        elif isinstance(item, list):
            table_rows.append({
                "Item": key.replace("_", " ").title(),
                "Value": _display_value(item),
            })
        else:
            table_rows.append({
                "Item": key.replace("_", " ").title(),
                "Value": _display_value(item),
            })
    if table_rows:
        st.markdown(f"**{heading} details**")
        st.table(table_rows)


def _save_checkpoint() -> None:
    """Persist the active investigation locally so it survives page reloads."""
    payload = {
        "director_version": "1.0.0",
        "research_question": st.session_state.get("director_active_question")
        or st.session_state.get("director_question", ""),
        "evidence": st.session_state.get("director_evidence", ""),
        "history": st.session_state.get("director_history", []),
        "decision": st.session_state.get("director_decision"),
        "started": st.session_state.get("director_started", False),
        "status": st.session_state.get("director_status", "not_started"),
        "updated": st.session_state.get("director_updated"),
    }
    if not payload["research_question"]:
        return
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, CHECKPOINT_PATH)


def _load_checkpoint() -> dict | None:
    """Load the last saved investigation, rejecting malformed checkpoint data."""
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("research_question"), str):
            return None
        if not isinstance(payload.get("history", []), list):
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None


def _save_question_draft(question: str) -> None:
    """Save an editable question draft separately from the active investigation."""
    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DRAFT_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"research_question": question, "updated": datetime.now(timezone.utc).isoformat()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, DRAFT_PATH)


def _load_question_draft() -> str | None:
    """Restore a separately saved question draft without replacing investigation history."""
    try:
        payload = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
        question = payload.get("research_question") if isinstance(payload, dict) else None
        return question if isinstance(question, str) else None
    except (OSError, ValueError, TypeError):
        return None


def _initialize() -> None:
    defaults = {
        "director_question": "",
        "director_evidence": "",
        "director_history": [],
        "director_decision": None,
        "director_started": False,
        "director_status": "not_started",
        "director_error": None,
        "director_updated": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.get("director_checkpoint_loaded"):
        checkpoint = _load_checkpoint()
        if checkpoint:
            st.session_state.director_question = checkpoint["research_question"]
            st.session_state.director_active_question = checkpoint["research_question"]
            st.session_state.director_evidence = checkpoint.get("evidence", "")
            st.session_state.director_history = checkpoint.get("history", [])
            st.session_state.director_decision = checkpoint.get("decision")
            st.session_state.director_started = bool(checkpoint.get("started", False))
            st.session_state.director_status = checkpoint.get("status", "not_started")
            st.session_state.director_updated = checkpoint.get("updated")
            st.session_state.director_checkpoint_restored = True
        draft_question = _load_question_draft()
        if draft_question is not None:
            st.session_state.director_question = draft_question
            st.session_state.director_question_draft_restored = True
        st.session_state.director_checkpoint_loaded = True


def _record(tool_name: str, mode: str, arguments: dict, result: str) -> None:
    st.session_state.director_history.append({
        "step": len(st.session_state.director_history) + 1,
        "tool_name": tool_name,
        "execution_mode": mode,
        "arguments": arguments,
        "result": result,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


def _run_until_decision_gate() -> None:
    """Auto-run reasoning and allow-listed read-only steps; stop at numerical gate."""
    question = st.session_state.director_question.strip()
    if not question:
        st.error("Enter a research question first.")
        return
    try:
        if not st.session_state.director_started:
            with st.spinner("Searching existing research evidence (read-only)…"):
                evidence = search_existing_evidence(question)
            st.session_state.director_evidence = evidence
            st.session_state.director_started = True

        for _ in range(5):
            if len(st.session_state.director_history) >= MAX_DECISION_STEPS:
                st.session_state.director_status = "step_limit_reached"
                st.session_state.director_decision = {
                    "status": "PAUSED",
                    "conclusion": "The investigation reached its configured decision-step limit.",
                    "evidence_assessment": "A further review is required before any additional work.",
                    "limitations": ["The director stopped at the safety limit; this is not a scientific conclusion."],
                    "tool_name": None,
                    "execution_mode": "none",
                    "arguments": {},
                    "next_action": "Review the history and start a new focused investigation if justified.",
                }
                _save_checkpoint()
                return

            with st.spinner("Research Director is assessing the evidence and selecting the next step…"):
                decision = decide_next_step(
                    question,
                    evidence=st.session_state.director_evidence,
                    history=st.session_state.director_history,
                )
            st.session_state.director_decision = decision
            st.session_state.director_updated = datetime.now(timezone.utc).isoformat()

            if decision["tool_name"] is None:
                st.session_state.director_status = decision["status"].lower()
                _save_checkpoint()
                return

            if decision["execution_mode"] == "numerical":
                st.session_state.director_status = "awaiting_numerical_approval"
                _save_checkpoint()
                return

            # Read-only actions can be carried out automatically after the user
            # started this investigation. Each result is added to the audit history.
            mode = decision["execution_mode"]
            with st.spinner(f"Automatically performing {mode.replace('_', ' ')} step: {decision['tool_name']}…"):
                result = execute_approved_step(decision["tool_name"], decision["arguments"])
            _record(
                decision["tool_name"],
                mode,
                decision["arguments"],
                result,
            )
            result_label = "SYMBOLIC ANALYSIS RESULT" if mode == "symbolic" else "READ-ONLY FOLLOW-UP RESULT"
            st.session_state.director_evidence += (
                f"\n\n{result_label} — "
                + decision["tool_name"] + ":\n" + result
            )
            _save_checkpoint()
        st.session_state.director_status = "ready_to_continue"
        _save_checkpoint()
    except Exception as exc:
        st.session_state.director_error = f"{type(exc).__name__}: {exc}"
        st.session_state.director_status = "error"
        _save_checkpoint()


_initialize()

with st.sidebar:
    st.markdown("### Investigation controls")
    st.markdown(
        "**Automatic:** project-evidence inspection, scholarly literature lookup, "
        "symbolic algebra, evidence assessment, and selection of the next justified direction."
    )
    st.markdown(
        "**Requires your click:** every new numerical calculation. "
        "One approval runs only the displayed calculation."
    )
    st.caption(
        "The director stops when the available evidence is sufficient, the claim "
        "remains unresolved, a scientific input is missing, or a safety limit is reached."
    )

if st.session_state.get("director_checkpoint_restored"):
    st.info("Resumed your saved investigation. Its original research question, evidence, results, and pending decision have been restored.")
    st.session_state.director_checkpoint_restored = False
if st.session_state.get("director_question_draft_restored"):
    st.info("A separately saved research-question draft has been restored. It has not replaced the saved investigation history.")
    st.session_state.director_question_draft_restored = False

question = st.text_area(
    "Research question",
    key="director_question",
    height=130,
    placeholder=(
        "Example: At fixed $k_r=1$ and $p=\\pi/2$, assess what the existing evidence "
        "establishes about the effect of changing $k_\\theta$ from 0 to 10^4."
    ),
    help="Enter the scientific objective. The director chooses diagnostics only when a specific evidence gap justifies them.",
)
if question.strip():
    with st.container(border=True):
        st.caption("Mathematical notation preview")
        st.markdown(_render_math(question))

controls = st.columns([1.7, 1, 1.15, 1.15])
if controls[0].button(
    "Start / continue investigation",
    type="primary",
    use_container_width=True,
    disabled=not bool(question.strip()),
):
    # A new question resets prior investigation state.
    if question.strip() != st.session_state.get("director_active_question"):
        st.session_state.director_history = []
        st.session_state.director_evidence = ""
        st.session_state.director_started = False
        st.session_state.director_decision = None
        st.session_state.director_error = None
        st.session_state.director_active_question = question.strip()
    _run_until_decision_gate()
    _save_checkpoint()
    try:
        DRAFT_PATH.unlink(missing_ok=True)
    except OSError:
        pass
if controls[1].button("Save question", use_container_width=True, disabled=not bool(question.strip())):
    _save_question_draft(question.strip())
    st.success("Research question draft saved on this app's local filesystem.")
if controls[2].button("Pause / stop here", use_container_width=True):
    st.session_state.director_status = "paused_by_user"
    st.session_state.director_decision = {
        "status": "PAUSED",
        "conclusion": "The investigation was paused by the user. No further calculations will run unless the user resumes it.",
        "evidence_assessment": "The current evidence and completed steps remain available in the investigation history.",
        "limitations": ["A user pause is a workflow decision, not a scientific conclusion."],
        "tool_name": None,
        "execution_mode": "none",
        "arguments": {},
        "next_action": "Resume only if a further research step is justified.",
    }
    _save_checkpoint()
if controls[3].button("Reset investigation", use_container_width=True):
    for key in (
        "director_history", "director_evidence", "director_started",
        "director_decision", "director_error", "director_updated",
    ):
        st.session_state[key] = [] if key == "director_history" else "" if key == "director_evidence" else False if key == "director_started" else None
    st.session_state.director_status = "not_started"
    st.session_state.director_active_question = ""
    try:
        CHECKPOINT_PATH.unlink(missing_ok=True)
        DRAFT_PATH.unlink(missing_ok=True)
    except OSError as exc:
        st.warning(f"Could not remove the saved checkpoint: {exc}")
    st.rerun()

if st.session_state.director_error:
    st.error(st.session_state.director_error)

decision = st.session_state.director_decision
if decision:
    st.divider()
    left, right = st.columns([3, 1])
    with left:
        st.subheader("Current research decision")
        status = decision.get("status", "UNRESOLVED")
        if status == "SUFFICIENT":
            st.success("Evidence sufficient for the stated, limited conclusion")
        elif status == "MORE_EVIDENCE_REQUIRED":
            st.warning("A specific evidence gap remains")
        elif status == "PAUSED":
            st.info("Investigation paused")
        else:
            st.info("Conclusion remains unresolved")
    with right:
        st.metric("Completed steps", len(st.session_state.director_history))
        st.caption(f"Updated: {st.session_state.director_updated or '—'}")

    st.markdown("**Current conclusion**")
    st.markdown(_render_math(decision.get("conclusion", "")))
    st.markdown("**Evidence assessment**")
    st.markdown(_render_math(decision.get("evidence_assessment", "")))
    limitations = decision.get("limitations") or []
    if limitations:
        st.markdown("**Scientific and numerical limitations**")
        for item in limitations:
            st.markdown(f"- {_render_math(item)}")

    if decision.get("tool_name"):
        st.markdown("### Proposed next step")
        st.markdown(f"**{_render_math(decision.get('next_action', ''))}**")
        st.markdown(_render_math(decision.get("why_needed", "")))
        st.markdown(
            "**Expected impact:** " + _render_math(decision.get("expected_decision_impact", "Not specified"))
        )
        st.markdown(f"**Diagnostic:** {decision['tool_name'].replace('_', ' ').title()}")
        st.markdown("**Proposed settings**")
        _render_parameter_table(decision.get("arguments", {}))

        if decision.get("execution_mode") == "numerical":
            st.warning(
                "This step creates new numerical data. Nothing will run until you approve it below. "
                "Approval applies only to this one calculation."
            )
            approval_col, reconsider_col = st.columns(2)
            if approval_col.button(
                "Approve and run this numerical calculation",
                type="primary",
                key="approve_director_numerical",
                use_container_width=True,
            ):
                try:
                    with st.spinner(f"Running approved calculation: {decision['tool_name']}…"):
                        result = execute_approved_step(
                            decision["tool_name"], decision["arguments"], user_approved=True
                        )
                    _record(
                        decision["tool_name"],
                        "numerical",
                        decision["arguments"],
                        result,
                    )
                    st.session_state.director_evidence += (
                        "\n\nNUMERICAL RESULT — " + decision["tool_name"] + ":\n" + result
                    )
                    st.session_state.director_decision = None
                    st.session_state.director_status = "continuing"
                    _save_checkpoint()
                    _run_until_decision_gate()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Approved calculation failed: {type(exc).__name__}: {exc}")
            if reconsider_col.button(
                "Do not run — ask Director to reconsider",
                key="reject_and_reconsider_director_step",
                use_container_width=True,
            ):
                st.session_state.director_evidence += (
                    "\n\nUSER FEEDBACK ON PROPOSED STEP: The user declined the proposed "
                    + decision["tool_name"]
                    + " calculation. Do not execute it or simply repeat the same proposal. "
                    "Reassess the evidence and choose a different, simpler justified action, "
                    "a read-only inspection, or conclude that no further work is justified."
                )
                st.session_state.director_decision = None
                st.session_state.director_status = "continuing"
                _save_checkpoint()
                _run_until_decision_gate()
                st.rerun()
        elif decision.get("execution_mode") in {"read_only", "symbolic"}:
            if decision.get("execution_mode") == "symbolic":
                st.info("This symbolic analysis does not launch a physical simulation and will run automatically when you continue.")
            else:
                st.info("This read-only step will be performed automatically when you continue the investigation.")
        else:
            st.caption("No further tool execution is proposed.")
    else:
        if decision.get("user_question"):
            st.markdown(_render_math(decision["user_question"]))
        st.markdown("**Next action**")
        st.markdown(_render_math(decision.get("next_action", "No further action proposed.")))

with st.expander("Investigation history and results", expanded=bool(st.session_state.director_history)):
    if not st.session_state.director_history:
        st.caption("No diagnostic or analysis has been executed yet.")
    for item in st.session_state.director_history:
        st.markdown(
            f"**Step {item['step']} · {item['tool_name']}** "
            f"({item['execution_mode'].replace('_', ' ')})"
        )
        st.caption(item.get("completed_at", ""))
        with st.expander("Step settings and result", expanded=False):
            st.markdown("**Settings**")
            _render_parameter_table(item.get("arguments", {}))
            _render_readable_result(parse_tool_result(item.get("result", "")))

record = {
    "director_version": "1.0.0",
    "research_question": st.session_state.director_question,
    "status": st.session_state.director_status,
    "evidence_search": st.session_state.director_evidence,
    "decision": decision,
    "history": st.session_state.director_history,
}
with st.expander("Optional: export research record", expanded=False):
    st.caption(
        "The normal research view contains no raw JSON. If you need a machine-readable "
        "archive for reproducibility or external analysis, you can download it here. No raw JSON is displayed on this page."
    )
    st.download_button(
        "Download technical archive",
        data=_json_text(record).encode("utf-8"),
        file_name="dkt_research_director_record.json",
        mime="application/json",
    )
