"""Streamlit page for the semi-autonomous DKT Research Director."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import streamlit as st

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


def _math_preview(value: str) -> str:
    """Normalize DKT notation for Streamlit Markdown/LaTeX rendering."""
    if not isinstance(value, str):
        return str(value)

    protected: list[str] = []

    def hold(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00MATH{len(protected) - 1}\x00"

    # Preserve already-delimited math, including \( ... \), while normalizing
    # common plain-text forms emitted by the language model.
    text = re.sub(r"\$\$.*?\$\$|\$[^$]+\$|\\\(.*?\\\)", hold, value, flags=re.S)
    patterns = (
        (r"(?<![\w])k\s*[_\s-]*\{?\s*theta\s*\}?(?![\w])", r"$k_\theta$"),
        (r"(?<![\w])k\s*[_\s-]*\{?\s*r\s*\}?(?![\w])", r"$k_r$"),
        (r"(?<![\w])theta[_\s-]*0(?![\w])", r"$\theta_0$"),
        (r"(?<![\w])phi[_\s-]*0(?![\w])", r"$\phi_0$"),
        (r"(?<![\w])pi\s*/\s*2(?![\w])", r"$\pi/2$"),
        (r"(?<![\w])Delta[_\s-]*theta(?![\w])", r"$\Delta\theta$"),
        (r"(?<![\w])Delta[_\s-]*phi(?![\w])", r"$\Delta\phi$"),
    )
    for pattern, replacement in patterns:
        text = re.sub(pattern, lambda _m, rep=replacement: rep, text, flags=re.I)

    for index, original in enumerate(protected):
        if original.startswith(r"\(") and original.endswith(r"\)"):
            original = "$" + original[2:-2] + "$"
        text = text.replace(f"\x00MATH{index}\x00", original)
    return text


def _render_json(value: object) -> None:
    """Render structured payloads as valid, copyable JSON text."""
    st.code(json.dumps(value, indent=2, ensure_ascii=False, default=str), language="json")


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
                return

            if decision["execution_mode"] == "numerical":
                st.session_state.director_status = "awaiting_numerical_approval"
                return

            # Read-only actions can be carried out automatically after the user
            # started this investigation. Each result is added to the audit history.
            with st.spinner(f"Automatically performing read-only step: {decision['tool_name']}…"):
                result = execute_approved_step(decision["tool_name"], decision["arguments"])
            _record(
                decision["tool_name"],
                "read_only",
                decision["arguments"],
                result,
            )
            st.session_state.director_evidence += (
                "\n\nREAD-ONLY FOLLOW-UP RESULT — "
                + decision["tool_name"] + ":\n" + result
            )
        st.session_state.director_status = "ready_to_continue"
    except Exception as exc:
        st.session_state.director_error = f"{type(exc).__name__}: {exc}"
        st.session_state.director_status = "error"


_initialize()

with st.sidebar:
    st.markdown("### Investigation controls")
    st.markdown(
        "**Automatic:** evidence search, evidence assessment, read-only inspection, "
        "and selection of the next justified direction."
    )
    st.markdown(
        "**Requires your click:** every new numerical calculation. "
        "One approval runs only the displayed calculation."
    )
    st.caption(
        "The director stops when the available evidence is sufficient, the claim "
        "remains unresolved, a scientific input is missing, or a safety limit is reached."
    )

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
        st.markdown(_math_preview(question))

controls = st.columns([1, 1, 2])
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
if controls[1].button("Reset investigation", use_container_width=True):
    for key in (
        "director_history", "director_evidence", "director_started",
        "director_decision", "director_error", "director_updated",
    ):
        st.session_state[key] = [] if key == "director_history" else "" if key == "director_evidence" else False if key == "director_started" else None
    st.session_state.director_status = "not_started"
    st.session_state.director_active_question = ""
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

    with st.expander("Complete research decision (JSON)", expanded=False):
        _render_json(decision)

    st.markdown("**Current conclusion**")
    st.markdown(_math_preview(decision.get("conclusion", "")))
    st.markdown("**Evidence assessment**")
    st.markdown(_math_preview(decision.get("evidence_assessment", "")))
    limitations = decision.get("limitations") or []
    if limitations:
        st.markdown("**Scientific and numerical limitations**")
        for item in limitations:
            st.markdown(f"- {_math_preview(item)}")

    if decision.get("tool_name"):
        st.markdown("### Proposed next step")
        st.markdown(f"**{_math_preview(decision.get('next_action', ''))}**")
        st.markdown(_math_preview(decision.get("why_needed", "")))
        st.caption(
            "Expected impact: " + _math_preview(decision.get("expected_decision_impact", "Not specified"))
        )
        st.code(decision["tool_name"], language="text")
        st.markdown("**Exact tool arguments (validated JSON)**")
        _render_json(decision.get("arguments", {}))

        if decision.get("execution_mode") == "numerical":
            st.warning(
                "This step creates new numerical data. Nothing will run until you approve it below. "
                "Approval applies only to this one calculation."
            )
            if st.button(
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
                    _run_until_decision_gate()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Approved calculation failed: {type(exc).__name__}: {exc}")
        elif decision.get("execution_mode") == "read_only":
            st.info("This read-only step will be performed automatically when you continue the investigation.")
        else:
            st.caption("No further tool execution is proposed.")
    else:
        if decision.get("user_question"):
            st.info(_math_preview(decision["user_question"]))
        st.markdown("**Next action**")
        st.markdown(_math_preview(decision.get("next_action", "No further action proposed.")))

with st.expander("Investigation history and results", expanded=bool(st.session_state.director_history)):
    if not st.session_state.director_history:
        st.caption("No diagnostic or analysis has been executed yet.")
    for item in st.session_state.director_history:
        st.markdown(
            f"**Step {item['step']} · {item['tool_name']}** "
            f"({item['execution_mode'].replace('_', ' ')})"
        )
        st.caption(item.get("completed_at", ""))
        st.markdown("Arguments")
        _render_json(item.get("arguments", {}))
        parsed = parse_tool_result(item.get("result", ""))
        if isinstance(parsed, (dict, list)):
            _render_json(parsed)
        else:
            st.text(str(parsed))

with st.expander("Machine-readable investigation record (JSON)", expanded=False):
    record = {
        "director_version": "1.0.0",
        "research_question": st.session_state.director_question,
        "status": st.session_state.director_status,
        "evidence_search": st.session_state.director_evidence,
        "decision": decision,
        "history": st.session_state.director_history,
    }
    st.code(json.dumps(record, indent=2, ensure_ascii=False, default=str), language="json")
    st.download_button(
        "Download investigation record (JSON)",
        data=json.dumps(record, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
        file_name="dkt_research_director_record.json",
        mime="application/json",
    )
