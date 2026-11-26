"""Streamlit Research Studio: evidence-led planning and guarded execution."""
from __future__ import annotations

import json
import math
import re

import streamlit as st

from dkt_agent.research_orchestrator import build_research_plan
from dkt_agent.research_execution import (
    create_campaign,
    execute_campaign_stage,
    list_campaigns,
    load_campaign,
    record_evidence_review,
    record_execution_authorization,
)


def _render_scientific_text(value: str) -> str:
    """Convert common plain-text DKT notation to Streamlit-rendered LaTeX.

    This is presentation-only: it never changes stored questions, plan data,
    tool arguments, campaign records, or numerical values.
    """
    if not isinstance(value, str) or not value:
        return value
    protected: list[str] = []

    def protect_math(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00MATH{len(protected) - 1}\x00"

    rendered = re.sub(r"\$[^$]+\$", protect_math, value)
    replacements = (
        (r"\bk[_\s-]*theta\b", r"$k_\theta$"),
        (r"\bk[_\s-]*r\b", r"$k_r$"),
        (r"\btheta[_\s-]*0\b", r"$\theta_0$"),
        (r"\bphi[_\s-]*0\b", r"$\phi_0$"),
        (r"\bpi\s*/\s*2\b", r"$\pi/2$"),
        (r"\blambda[_\s-]*1\b", r"$\lambda_1$"),
        (r"\blambda[_\s-]*2\b", r"$\lambda_2$"),
        (r"\bL\s*\^?\s*2\b", r"$L^2$"),
        (r"\bDelta[_\s-]*theta\b", r"$\Delta\theta$"),
        (r"\bDelta[_\s-]*phi\b", r"$\Delta\phi$"),
    )
    for pattern, replacement in replacements:
        rendered = re.sub(pattern, replacement, rendered, flags=re.IGNORECASE)
    for index, original in enumerate(protected):
        rendered = rendered.replace(f"\x00MATH{index}\x00", original)
    return rendered


def _question_defaults(question: str) -> dict[str, float | int]:
    """Extract only explicit scalar values; otherwise use documented tool defaults."""
    values: dict[str, float | int] = {
        "kr": 1.0, "ktheta_a": 0.0, "ktheta_b": 10000.0,
        "theta0": 0.75, "phi0": -1.0, "nsteps": 1000,
        "p": math.pi / 2,
    }
    kr = re.search(r"k[_\s-]?r\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.I)
    if kr:
        values["kr"] = float(kr.group(1))
    kt = re.search(r"k[_\s-]?theta\s*(?:from|between)\s*([-+]?\d+(?:\.\d+)?)\s*(?:to|and|[-–])\s*([-+]?\d+(?:\.\d+)?)", question, re.I)
    if kt:
        values["ktheta_a"], values["ktheta_b"] = float(kt.group(1)), float(kt.group(2))
    elif re.search(r"k[_\s-]?theta\s*=\s*[-+]?\d", question, re.I):
        vals = re.findall(r"k[_\s-]?theta\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.I)
        if len(vals) >= 2:
            values["ktheta_a"], values["ktheta_b"] = float(vals[0]), float(vals[1])
    for key, pattern in (
        ("theta0", r"(?:theta[_\s]?0|θ[_\s]?0)\s*=\s*([-+]?\d+(?:\.\d+)?)"),
        ("phi0", r"(?:phi[_\s]?0|φ[_\s]?0)\s*=\s*([-+]?\d+(?:\.\d+)?)"),
        ("p", r"\bp\s*=\s*([-+]?\d+(?:\.\d+)?)"),
    ):
        match = re.search(pattern, question, re.I)
        if match:
            values[key] = float(match.group(1))
    nsteps = re.search(r"(?:nsteps|number of kicks|kicks)\s*(?:=|of)?\s*(\d+)", question, re.I)
    if nsteps:
        values["nsteps"] = int(nsteps.group(1))
    return values


def _campaign_is_authorized(campaign: dict) -> bool:
    """Read current and legacy campaign authorization fields."""
    if "execution_authorization" in campaign:
        return campaign.get("execution_authorization") == "authorized_by_user"
    return campaign.get("plan", {}).get("execution_authorization") == "authorized_by_user"


def _record_authorization_if_requested(campaign: dict, authorized: bool) -> None:
    """Persist the explicit checkbox decision before a numerical stage starts."""
    if authorized and not _campaign_is_authorized(campaign):
        record_execution_authorization(campaign, authorized=True)


def _run_stage(campaign: dict, index: int, arguments: dict, authorized: bool = False) -> None:
    try:
        updated = execute_campaign_stage(
            campaign, index, arguments, execution_authorized=authorized
        )
        st.session_state["research_studio_campaign"] = updated
        st.success(f"Completed: {updated['stages'][index]['label']}")
    except Exception as exc:
        st.session_state["research_studio_campaign"] = campaign
        st.error(f"Stage could not be completed: {exc}")


def _render_stage_result(stage: dict) -> None:
    if stage.get("result") is not None:
        with st.expander(f"View result · {stage['label']}", expanded=False):
            st.json(stage["result"])
            st.download_button(
                "Download stage result (JSON)",
                data=json.dumps(stage["result"], indent=2, ensure_ascii=False, default=str),
                file_name=f"{stage['tool']}_result.json",
                mime="application/json",
                key=f"stage_result_{stage['index']}",
            )
    if stage.get("error"):
        st.error(stage["error"])


def _render_execution(campaign: dict) -> None:
    plan = campaign["plan"]
    stages = campaign.get("stages", [])
    st.markdown("### Research execution")
    st.caption(
        "Stages run in the planned order. Evidence discovery is read-only. "
        "Each numerical stage requires recorded authorization and a separate execution action."
    )
    if not stages:
        st.info("This plan has no executable stages.")
        return

    for stage in stages:
        status = stage["status"].replace("_", " ").title()
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{stage['index'] + 1}. {stage['label']}**")
                st.caption(f"{stage['tool']} · {stage['execution_mode'].replace('_', ' ')}")
            with right:
                st.caption(status)
            _render_stage_result(stage)

    evidence_complete = any(
        s["tool"] == "search_research_evidence" and s["status"] == "completed"
        for s in stages
    )
    numerical_pending = any(
        s["execution_mode"] == "new_numerical_data" and s["status"] in {"pending", "failed"}
        for s in stages
    )
    if evidence_complete and numerical_pending and not campaign.get("evidence_review"):
        with st.container(border=True):
            st.markdown("**Evidence sufficiency checkpoint**")
            st.write(
                "Review the read-only search results above. Decide whether they already answer "
                "the question, or whether the planned numerical work is still necessary."
            )
            left, right = st.columns(2)
            with left:
                if st.button("Evidence is sufficient · stop here", key=f"evidence_stop_{campaign['campaign_id']}"):
                    record_evidence_review(campaign, proceed_with_new_calculations=False)
                    st.rerun()
            with right:
                if st.button("Evidence is insufficient · proceed to numerical work", type="primary",
                             key=f"evidence_proceed_{campaign['campaign_id']}"):
                    record_evidence_review(campaign, proceed_with_new_calculations=True)
                    st.rerun()
        return
    if campaign.get("evidence_review", {}).get("decision") == "stop_existing_evidence_sufficient":
        st.success("The campaign stopped after evidence review. No new numerical data were generated.")
        return

    next_stage = next((s for s in stages if s["status"] in {"pending", "failed"}), None)
    if next_stage is None:
        st.success("All planned workflow stages have completed. Review the evidence and limitations before drawing conclusions.")
        return
    index, tool_name = next_stage["index"], next_stage["tool"]
    if any(s["status"] != "completed" for s in stages[:index]):
        st.info("Complete the earlier stage before proceeding.")
        return

    if tool_name == "search_research_evidence":
        st.markdown("**Stage input · evidence discovery**")
        st.caption("Search is read-only and will not generate or modify numerical research data.")
        if st.button("Run read-only evidence discovery", type="primary", key=f"run_evidence_{index}"):
            _run_stage(campaign, index, {"query": plan["research_question"], "max_files": 15})
            st.rerun()
        return

    defaults = _question_defaults(plan.get("research_question", ""))
    st.markdown("**Review numerical inputs**")
    st.caption("These values are shown before execution and are saved with the campaign record.")
    authorized = _campaign_is_authorized(campaign)
    if authorized:
        if st.button(
            "Revoke numerical-execution authorization",
            key=f"revoke_authorization_{campaign['campaign_id']}",
        ):
            record_execution_authorization(campaign, authorized=False)
            st.session_state.pop(
                f"authorize_campaign_{campaign['campaign_id']}", None
            )
            st.rerun()
    else:
        authorized = st.checkbox(
            "I authorize this campaign to generate new numerical data",
            key=f"authorize_campaign_{campaign['campaign_id']}",
        )
    if tool_name == "run_dkt_comparison_study":
        with st.form(f"comparison_inputs_{campaign['campaign_id']}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                kr = st.number_input(r"$k_r$", value=float(defaults["kr"]), format="%.8g")
                kt_a = st.number_input(r"$k_{\theta,a}$", value=float(defaults["ktheta_a"]), format="%.8g")
            with c2:
                kt_b = st.number_input(r"$k_{\theta,b}$", value=float(defaults["ktheta_b"]), format="%.8g")
                theta0 = st.number_input(r"$\theta_0$", value=float(defaults["theta0"]), format="%.8g")
            with c3:
                phi0 = st.number_input(r"$\phi_0$", value=float(defaults["phi0"]), format="%.8g")
                p = st.number_input(r"$p$", value=float(defaults["p"]), format="%.8g")
            nsteps = st.number_input("Number of kicks", min_value=1, max_value=500000,
                                     value=int(defaults["nsteps"]), step=100)
            submitted = st.form_submit_button(
                "Execute controlled comparison", type="primary", disabled=not authorized,
                use_container_width=True,
            )
        if submitted:
            _record_authorization_if_requested(campaign, authorized)
            args = {"kr": kr, "ktheta_a": kt_a, "ktheta_b": kt_b,
                    "theta0": theta0, "phi0": phi0, "p": p,
                    "nsteps": int(nsteps), "comparison_mode": "fixed_kr"}
            with st.spinner("Running the two matched DKT trajectories and comparison…"):
                _run_stage(campaign, index, args, authorized=True)
            st.rerun()
    elif tool_name == "run_dkt_geometry_ensemble_comparison":
        with st.form(f"geometry_ensemble_inputs_{campaign['campaign_id']}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                kr = st.number_input(r"$k_r$", value=float(defaults["kr"]), format="%.8g")
                kt_a = st.number_input(r"$k_{\theta,a}$", value=float(defaults["ktheta_a"]), format="%.8g")
                theta0 = st.number_input(r"$\theta_0$", value=float(defaults["theta0"]), format="%.8g")
            with c2:
                kt_b = st.number_input(r"$k_{\theta,b}$", value=float(defaults["ktheta_b"]), format="%.8g")
                phi0 = st.number_input(r"$\phi_0$", value=float(defaults["phi0"]), format="%.8g")
                delta_theta = st.number_input(r"Patch half-width $\Delta\theta$", min_value=0.001, max_value=0.5, value=0.08, format="%.3f")
            with c3:
                p = st.number_input(r"$p$", value=float(defaults["p"]), format="%.8g")
                delta_phi = st.number_input(r"Patch half-width $\Delta\phi$", min_value=0.001, max_value=3.14159, value=0.08, format="%.3f")
                grid_size = st.slider("Grid points per angular direction", 3, 31, 7, 2)
            nsteps = st.number_input("Kicks per trajectory", min_value=1, max_value=50000, value=2000, step=500)
            st.caption("Both maps receive the same local angular grid of initial conditions. The cloud is a geometric probe, not an invariant phase-space region.")
            submitted = st.form_submit_button(
                "Execute matched-cloud geometry comparison",
                type="primary", disabled=not authorized, use_container_width=True,
            )
        if submitted:
            _record_authorization_if_requested(campaign, authorized)
            args = {"kr": kr, "ktheta_values": [kt_a, kt_b],
                    "theta0": theta0, "phi0": phi0,
                    "delta_theta": delta_theta, "delta_phi": delta_phi,
                    "grid_size": grid_size, "nsteps": int(nsteps), "p": p}
            with st.spinner("Evolving the matched local clouds and measuring their geometry…"):
                _run_stage(campaign, index, args, authorized=True)
            st.rerun()
    elif tool_name == "mean_ergodicity_langchain_tool":
        with st.form(f"mean_ergodicity_inputs_{campaign['campaign_id']}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                kr = st.number_input(r"$k_r$", value=float(defaults["kr"]), format="%.8g")
                kt = st.number_input(r"$k_{\theta}$", value=float(defaults["ktheta_a"]), format="%.8g")
            with c2:
                p = st.number_input(r"$p$", value=float(defaults["p"]), format="%.8g")
                seed = st.number_input("Random seed", min_value=0, value=0, step=1)
            with c3:
                n_kicks = st.select_slider("Maximum kicks $N$", [1024, 2048, 4096, 8192, 16384],
                                           value=16384)
                n_initial = st.slider("Initial conditions $M$", 500, 5000, 3000, 500)
            observables = st.multiselect(
                "Test observables independently",
                ["x", "y", "z", "x2", "y2", "z2", "xy", "yz", "xz"],
                default=["x", "y", "z", "x2", "y2", "z2", "xy", "yz", "xz"],
            )
            submitted = st.form_submit_button(
                "Execute observable-resolved mean-ergodicity test",
                type="primary", disabled=(not authorized or not observables),
                use_container_width=True,
            )
        if submitted:
            _record_authorization_if_requested(campaign, authorized)
            args = {"k_r": kr, "k_theta": kt, "observables": observables,
                    "n_kicks": int(n_kicks), "n_initial_conditions": int(n_initial),
                    "p": p, "seed": int(seed)}
            with st.spinner("Estimating the observable-wise finite-data L2 diagnostics…"):
                _run_stage(campaign, index, args, authorized=True)
            st.rerun()
    elif tool_name == "run_ergodicity_variance_study":
        with st.form(f"birkhoff_variance_inputs_{campaign['campaign_id']}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                kr = st.number_input(r"$k_r$", value=float(defaults["kr"]), format="%.8g")
                kt = st.number_input(r"$k_{\theta}$", value=float(defaults["ktheta_a"]), format="%.8g")
            with c2:
                p = st.number_input(r"$p$", value=float(defaults["p"]), format="%.8g")
                seed = st.number_input("Random seed", min_value=0, value=20261124, step=1)
            with c3:
                n_trajectories = st.slider("Initial conditions", 3, 24, 8)
                nsteps = st.number_input("Total kicks", min_value=100, max_value=50000, value=10000, step=1000)
            transient = st.number_input("Discarded transient", min_value=0, max_value=int(nsteps)-2,
                                        value=min(1000, int(nsteps)-2), step=100)
            submitted = st.form_submit_button(
                "Execute observable-resolved Birkhoff-variance study",
                type="primary", disabled=not authorized, use_container_width=True,
            )
        if submitted:
            _record_authorization_if_requested(campaign, authorized)
            args = {"kr": kr, "ktheta": kt, "p": p, "n_trajectories": n_trajectories,
                    "nsteps": int(nsteps), "transient": int(transient), "seed": int(seed)}
            with st.spinner("Generating the matched ensemble and analysing nine observables…"):
                _run_stage(campaign, index, args, authorized=True)
            st.rerun()
    else:
        st.warning(
            f"The planned stage '{tool_name}' is not yet connected to the Research Studio execution interface. "
            "No calculation has been launched."
        )


def render_research_studio() -> None:
    """Render the plan-first and campaign-execution research workspace."""
    st.subheader("Research Studio")
    st.caption(
        "Frame a scientific question, discover existing evidence, then execute an approved "
        "workflow with saved inputs, results and provenance."
    )

    saved = list_campaigns()
    with st.expander("Resume a saved research campaign", expanded=False):
        if not saved:
            st.caption("No saved campaigns yet.")
        else:
            labels = {
                f"{item['campaign_id']} · {item['status']} · "
                f"{'authorized' if item.get('execution_authorization') == 'authorized_by_user' else 'not authorized'}": item["campaign_id"]
                for item in saved
            }
            selected = st.selectbox("Saved campaigns", list(labels), key="research_campaign_resume")
            if st.button("Load selected campaign", key="load_research_campaign"):
                st.session_state["research_studio_campaign"] = load_campaign(labels[selected])
                st.session_state["research_studio_latest_plan"] = st.session_state["research_studio_campaign"]["plan"]
                st.rerun()

    question = st.text_area(
        "Research question", height=125, key="research_studio_question",
        placeholder=(
            "Example: At fixed average kicking strength 1, compare asymmetry "
            "values 0 and 10^4. "
            "Assess X, Y and Z separately, and distinguish observable-wise "
            "ergodicity from global or region-restricted claims."
        ),
        help="Enter the question in plain text. A rendered mathematical-notation preview appears below; the saved question remains unchanged.",
    )
    if question.strip():
        with st.container(border=True):
            st.caption("Mathematical-notation preview")
            st.markdown(_render_scientific_text(question))
    c1, c2 = st.columns(2)
    with c1:
        data_state = st.selectbox(
            "Existing data",
            ["Not sure / discover first", "Relevant data already available", "No suitable data available"],
            key="research_studio_existing_data",
        )
    with c2:
        authorize = st.checkbox(
            "I explicitly authorize new numerical calculations if required",
            value=False, key="research_studio_initial_authorization",
            help="Authorization is recorded; numerical work still requires a separate execution action.",
        )
    submitted = st.button("Create research plan", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a research question before creating a plan.")
        else:
            existing = True if data_state == "Relevant data already available" else False if data_state == "No suitable data available" else None
            plan = build_research_plan(
                question, existing_data_available=existing, execution_authorized=authorize
            )
            campaign = create_campaign(plan)
            st.session_state["research_studio_latest_plan"] = plan
            st.session_state["research_studio_campaign"] = campaign

    campaign = st.session_state.get("research_studio_campaign")
    plan = st.session_state.get("research_studio_latest_plan")
    if not campaign or not plan:
        with st.container(border=True):
            st.markdown("**How the research workflow works**")
            st.markdown(
                "1. Frame the question and controlled variables.\n"
                "2. Discover existing evidence before repeating calculations.\n"
                "3. Select validated diagnostics matched to the claim.\n"
                "4. Authorize and execute numerical stages explicitly.\n"
                "5. Preserve results, limitations and provenance in a resumable campaign."
            )
        return

    st.divider()
    overview, execution, record = st.tabs(["Investigation overview", "Execute workflow", "Campaign record"])
    with overview:
        auth_columns = st.columns(3)
        is_authorized = _campaign_is_authorized(campaign)
        auth_columns[0].metric(
            "Numerical execution authorization",
            "Authorized" if is_authorized else "Not authorized",
        )
        evidence_decision = campaign.get("evidence_review", {}).get("decision")
        evidence_label = {
            "proceed_with_new_calculations": "Proceed approved",
            "stop_existing_evidence_sufficient": "Stopped after review",
        }.get(evidence_decision, "Review pending")
        auth_columns[1].metric("Evidence sufficiency review", evidence_label)
        auth_columns[2].metric("Campaign status", campaign.get("status", "unknown").replace("_", " ").title())
        if is_authorized:
            authorized_at = campaign.get("execution_authorized_at")
            if authorized_at:
                st.caption(f"User authorization recorded at {authorized_at}.")
            else:
                st.caption("User authorization is present in this saved campaign.")
        else:
            st.warning(
                "New numerical calculations are not authorized in this campaign. "
                "Read-only evidence discovery remains available."
            )

        st.markdown("**Research question**")
        st.markdown(_render_scientific_text(plan.get("research_question", "")))
        st.markdown(f"**Research intent:** `{plan['research_intent']}`")
        st.markdown("**Scope and controlled variables**")
        for item in plan["scope_and_controls"]:
            st.markdown(f"- {_render_scientific_text(item)}")
        st.markdown("**Proposed workflow**")
        for index, step in enumerate(plan["workflow"], start=1):
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{index}. {_render_scientific_text(step['action'])}**")
                    st.caption(_render_scientific_text(step["why_this_step"]))
                with right:
                    st.caption(step["execution_mode"].replace("_", " ").title())
                with st.expander("Method, required inputs and scientific caveat", expanded=False):
                    st.caption(f"Tool: `{step['tool']}`")
                    st.markdown("**Required inputs**")
                    for item in step["required_inputs"]:
                        st.markdown(f"- {_render_scientific_text(item)}")
                    st.info(_render_scientific_text(step["scientific_caveat"]))
        if plan["clarifications_or_gates"]:
            st.markdown("**Scientific and data gates**")
            for item in plan["clarifications_or_gates"]:
                st.warning(_render_scientific_text(item))

    with execution:
        st.markdown(f"**Campaign:** `{campaign['campaign_id']}`")
        if _campaign_is_authorized(campaign):
            st.success("Numerical execution authorization is recorded for this campaign.")
        else:
            st.warning(
                "Numerical execution is not authorized. Read-only evidence discovery is still permitted."
            )
        _render_execution(campaign)

    with record:
        st.markdown("**Persistent campaign record**")
        st.caption("The campaign stores the plan snapshot, each stage's exact inputs, timestamps, outputs, errors and execution history.")
        st.code(json.dumps(campaign, indent=2, ensure_ascii=False, default=str), language="json")
        st.download_button(
            "Download complete campaign record",
            data=json.dumps(campaign, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name=f"{campaign['campaign_id']}.json",
            mime="application/json",
            use_container_width=True,
        )
