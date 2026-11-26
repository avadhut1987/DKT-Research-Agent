"""Streamlit Research Studio: plan-first interface for DKT investigations."""
from __future__ import annotations
import json
import streamlit as st
from dkt_agent.research_orchestrator import build_research_plan

def render_research_studio() -> None:
    """Render the plan-first research workspace."""
    st.subheader("Research Studio")
    st.caption("Translate a scientific question into an evidence-led workflow. Planning does not run simulations or modify existing research data.")
    with st.form("research_studio_plan_form"):
        question = st.text_area("Research question", height=130,
            placeholder="Example: At fixed k_r=1, how does changing k_theta from 0 to 10^4 alter the detected periodic-orbit geometry?",
            help="State the physical question, parameter constraints, observables and relevant existing data.")
        c1, c2 = st.columns(2)
        with c1:
            data_state = st.selectbox("Existing data",
                ["Not sure / discover first", "Relevant data already available", "No suitable data available"])
        with c2:
            authorize = st.checkbox("I explicitly authorize new numerical calculations if required",
                value=False, help="This records authorization in the plan. The planner itself never executes calculations.")
        submitted = st.form_submit_button("Create research plan", type="primary", width="stretch")
    if not submitted:
        with st.expander("Research workflow philosophy", expanded=False):
            st.markdown(
                "1. **Frame** the scientific question and controlled variables.\n"
                "2. **Discover** existing evidence before repeating calculations.\n"
                "3. **Plan** a method matched to the claim and available data.\n"
                "4. **Authorize** new numerical execution explicitly.\n"
                "5. **Verify and interpret** outputs with limitations and provenance."
            )
        return
    if not question.strip():
        st.warning("Enter a research question before creating a plan.")
        return
    existing = True if data_state == "Relevant data already available" else False if data_state == "No suitable data available" else None
    plan = build_research_plan(question, existing_data_available=existing, execution_authorized=authorize)
    st.session_state["research_studio_latest_plan"] = plan
    st.divider()
    st.markdown("#### Investigation overview")
    st.markdown(f"**Research intent:** {plan['research_intent']}")
    st.markdown("**Scope and controlled variables**")
    for item in plan["scope_and_controls"]:
        st.markdown(f"- {item}")
    st.markdown("#### Proposed workflow")
    for index, step in enumerate(plan["workflow"], start=1):
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{index}. {step['action']}**")
                st.caption(step["why_this_step"])
            with c2:
                st.caption(step["execution_mode"].replace("_", " ").title())
            with st.expander("Required inputs, method and scientific caveat"):
                st.caption("Technical tool")
                st.code(step["tool"], language="text")
                st.markdown("**Required inputs**")
                for item in step["required_inputs"]:
                    st.markdown(f"- {item}")
                st.info(step["scientific_caveat"])
    st.markdown("#### Authorization and next steps")
    if plan["execution_authorization"] == "authorized_by_user":
        st.success("New numerical calculations may be included in the investigation. No calculation has been executed by the planner.")
    elif any(s["execution_mode"] == "new_numerical_data" for s in plan["workflow"]):
        st.warning("New numerical work is identified but not authorized. The planner has not launched any calculation.")
    else:
        st.info("No new numerical execution is proposed in this plan.")
    for item in plan["clarifications_or_gates"]:
        st.markdown(f"- {item}")
    with st.expander("Full machine-readable plan", expanded=False):
        st.json(plan)
        st.download_button("Download research plan (JSON)",
            data=json.dumps(plan, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="dkt_research_plan.json", mime="application/json",
            key="research_studio_plan_download", width="stretch")
