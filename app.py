from __future__ import annotations

import json
from typing import Any

import streamlit as st
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dkt_agent.agent import build_agent
from dkt_agent.tools import simulate_dkt, simulate_dkt_grid
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DKT Research Assistant",
    page_icon="⚛️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stChatMessage"] {
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 14px;
    padding: .7rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("⚛️ DKT Research Assistant")
st.caption(
    "A research interface for Double Kicked Top simulations, "
    "analysis, and evidence retrieval."
)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False, default=str))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


with st.sidebar:
    st.header("DKT parameters")
    st.markdown(r"**Average kicking strength**  $k_r=\frac{k+k'}{2}$")
    st.caption(
        "Controls the degree of classical chaos and the "
        "regular-to-chaotic transition."
    )
    st.markdown(r"**Asymmetry parameter**  $k_\theta=\frac{k-k'}{2}$")
    st.caption(
        "Changes phase-space geometry and its orientation/structure; "
        "it does not control the degree of classical chaos."
    )


    with st.expander("Interactive DKT simulation", expanded=True):
        with st.form("dkt_simulation_form"):
            st.caption("Run the classical DKT Julia simulation.")

            sim_kr = st.number_input(
                r"Average kicking strength $k_r$",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
            )

            sim_ktheta = st.number_input(
                r"Asymmetry parameter $k_\theta$",
                min_value=-1000000.0,
                max_value=1000000.0,
                value=0.0,
                step=100.0,
                format="%.4f",
            )

            sim_theta0 = st.number_input(
                "Initial theta (radians)",
                min_value=0.0,
                max_value=3.141592653589793,
                value=0.75,
                step=0.05,
            )

            sim_phi0 = st.number_input(
                "Initial phi (radians)",
                min_value=-3.141592653589793,
                max_value=3.141592653589793,
                value=-1.0,
                step=0.05,
            )

            sim_nsteps = st.number_input(
                "Number of kicks",
                min_value=1,
                max_value=50000,
                value=1000,
                step=100,
            )

            sim_p = st.number_input(
                r"Rotation angle $p$ (radians)",
                min_value=0.0,
                max_value=6.283185307179586,
                value=1.5707963267948966,
                step=0.05,
            )

            sim_timeout = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=1800,
                value=300,
                step=30,
            )

            run_simulation = st.form_submit_button(
                "Run DKT Simulation",
                use_container_width=True,
            )

        if run_simulation:
            try:
                with st.spinner("Running DKT simulation using Julia..."):
                    result = simulate_dkt(
                        kr=float(sim_kr),
                        ktheta=float(sim_ktheta),
                        theta0=float(sim_theta0),
                        phi0=float(sim_phi0),
                        nsteps=int(sim_nsteps),
                        p=float(sim_p),
                        timeout=int(sim_timeout),
                        include_trajectory=True,
                    )

                st.session_state["simulation_result"] = result
                st.session_state.pop("simulation_error", None)

            except Exception as exc:
                st.session_state["simulation_error"] = str(exc)

    with st.expander("Theta-phi phase-space grid", expanded=False):
        st.caption(
            "Evolve a spherical grid of initial conditions and display "
            "all orbit points from kick 0 through the final kick."
        )

        with st.form("dkt_grid_simulation_form"):
            grid_resolution = st.number_input(
                "Grid resolution (N × N)",
                min_value=5,
                max_value=50,
                value=15,
                step=1,
                help="Enter the number of initial points along each angular direction (5–50).",
            )

            grid_kr = st.number_input(
                r"Average kicking strength $k_r$",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="grid_kr",
            )

            grid_ktheta = st.number_input(
                r"Asymmetry parameter $k_\theta$",
                min_value=-1000000.0,
                max_value=1000000.0,
                value=0.0,
                step=100.0,
                format="%.4f",
                key="grid_ktheta",
            )

            grid_p = st.number_input(
                r"Rotation angle $p$ (radians)",
                min_value=0.0,
                max_value=6.283185307179586,
                value=1.5707963267948966,
                step=0.05,
                key="grid_p",
            )

            grid_nsteps = st.number_input(
                "Number of kicks",
                min_value=1,
                max_value=10000,
                value=120,
                step=10,
                key="grid_nsteps",
            )

            run_grid_simulation = st.form_submit_button(
                "Run phase-space grid",
                use_container_width=True,
            )

        if run_grid_simulation:
            try:
                with st.spinner(
                    "Evolving the phase-space grid using Julia..."
                ):
                    grid_result = simulate_dkt_grid(
                        kr=float(grid_kr),
                        ktheta=float(grid_ktheta),
                        grid_size=int(grid_resolution),
                        nsteps=int(grid_nsteps),
                        p=float(grid_p),
                        timeout=300,
                    )

                st.session_state["grid_simulation_result"] = grid_result
                st.session_state.pop("grid_simulation_error", None)

            except Exception as exc:
                st.session_state["grid_simulation_error"] = str(exc)


    st.divider()
    st.subheader("DKT Research Capabilities")
    st.caption(
        "Physics-oriented tools and analysis capabilities for "
        "Double Kicked Top research."
    )

    with st.expander("1. Simulate", expanded=True):
        st.caption("AVAILABLE")
        st.markdown(
            "**Classical DKT trajectory** — Run a single classical "
            "trajectory using the Julia simulation backend."
        )
        st.caption("Manual interface: Interactive DKT simulation above.")

        st.markdown(
            "**Phase-space grid evolution** — Evolve an N × N grid "
            "of initial conditions and record every kick."
        )
        st.caption("Manual interface: Theta-phi phase-space grid above.")

        st.caption("PLANNED")
        st.markdown(
            "Fixed-point analysis, stability analysis and systematic "
            "parameter scans."
        )

    with st.expander("2. Diagnose"):
        st.caption("PLANNED")
        st.markdown(
            "**Classical chaos:** Lyapunov exponents, stability and "
            "Kolmogorov-Sinai entropy."
        )
        st.markdown(
            "**Ergodicity:** Temporal averages, observable fluctuations, "
            "variance scaling and operational ergodicity diagnostics."
        )
        st.markdown(
            "**Mixing and transport:** Correlation functions, "
            "decay behaviour and transport diagnostics."
        )
        st.caption(
            "These diagnostics will be implemented and validated "
            "before being presented as available tools."
        )

    with st.expander("3. Compare"):
        st.caption("AVAILABLE")

        st.markdown(
            "**Trajectory comparison** — Compare two supplied "
            "classical trajectory CSV datasets."
        )
        st.markdown(
            "**Configured comparison study** — Run and compare "
            "two classical trajectories with selected parameter values."
        )

        st.caption("POSSIBLE COMPARISON STUDIES")
        st.markdown(
            "- Different $k_r$: regular-to-chaotic behaviour."
            "\n- Different $k_\\theta$ at fixed $k_r$: phase-space "
            "geometry and orientation."
            "\n- Different initial conditions: trajectory dependence."
            "\n- Different observables: temporal statistics."
            "\n- Different dynamical regimes: diagnostic behaviour."
        )

        st.caption("PLANNED")
        st.markdown(
            "Automated comparisons of phase-space distributions, "
            "ergodicity diagnostics, correlation functions and "
            "classical-quantum results."
        )

    with st.expander("4. Research Data"):
        st.caption("AVAILABLE")
        st.markdown(
            "**Research evidence search** — Search local research "
            "files for relevant evidence."
        )
        st.markdown(
            "**CSV reader** — Inspect numerical datasets and "
            "tabular research results."
        )

        st.caption("PLANNED")
        st.markdown(
            "Integrated plotting, statistical analysis, "
            "reproducible comparison reports and data export."
        )

    with st.expander("5. Explain"):
        st.caption("AGENT GUIDANCE")
        st.markdown(
            "Ask physics questions about DKT dynamics, parameters, "
            "regular and chaotic behaviour, ergodicity, mixing, "
            "quantum dynamics and classical-quantum correspondence."
        )

        st.caption("PLANNED")
        st.markdown(
            "Structured diagnostic interpretation, evidence-linked "
            "scientific reports and automated assessment of numerical "
            "limitations."
        )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.agent_messages = []
        st.session_state.chat_history = []
        st.rerun()


if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



if st.session_state.get("simulation_error"):
    st.error(
        "DKT simulation failed: "
        + st.session_state["simulation_error"]
    )

if "simulation_result" in st.session_state:
    result = st.session_state["simulation_result"]

    st.divider()
    st.header("DKT Simulation Results")

    metrics = st.columns(3)
    metrics[0].metric(
        "Trajectory points",
        result["trajectory_rows"],
    )
    metrics[1].metric(
        "Initial norm",
        f'{result["initial_norm"]:.12f}',
    )
    metrics[2].metric(
        "Maximum norm error",
        f'{result["max_norm_error"]:.3e}',
    )

    with st.expander("Simulation parameters", expanded=False):
        st.json(result["parameters"])

    state_columns = st.columns(2)

    with state_columns[0]:
        st.markdown("**Initial state**")
        st.json(result["initial_state"])

    with state_columns[1]:
        st.markdown("**Final state**")
        st.json(result["final_state"])

    trajectory = pd.DataFrame(result.get("trajectory", []))

    if not trajectory.empty:
        st.subheader("Classical trajectory")
        st.caption("Evolution of the Cartesian components X, Y and Z.")

        plot_data = trajectory.set_index("step")[["X", "Y", "Z"]]
        st.line_chart(plot_data)

        with st.expander("Trajectory data", expanded=False):
            st.dataframe(
                trajectory.head(20),
                use_container_width=True,
            )
            st.caption(
                f"Showing the first 20 of {len(trajectory)} trajectory rows."
            )

        csv_data = trajectory.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download trajectory CSV",
            data=csv_data,
            file_name="dkt_trajectory.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("The simulation returned no trajectory data.")

    st.caption(
        "Scientific interpretation: kr controls the degree of classical "
        "chaos; ktheta changes phase-space geometry, not the degree of chaos."
    )

if st.session_state.get("grid_simulation_error"):
    st.error(
        "Phase-space grid simulation failed: "
        + st.session_state["grid_simulation_error"]
    )

if "grid_simulation_result" in st.session_state:
    grid_result = st.session_state["grid_simulation_result"]
    grid_parameters = grid_result["parameters"]
    grid_data = pd.DataFrame(grid_result["orbit_points"])

    st.divider()
    st.header("Final theta-phi phase-space portrait")
    st.success("Phase-space grid simulation completed successfully.")

    grid_metrics = st.columns(3)
    grid_metrics[0].metric(
        "Grid resolution",
        f'{grid_parameters["grid_size"]} × '
        f'{grid_parameters["grid_size"]}',
    )
    grid_metrics[1].metric(
        "Number of kicks",
        grid_parameters["nsteps"],
    )
    grid_metrics[2].metric(
        "Maximum norm error",
        f'{grid_result["max_norm_error"]:.3e}',
    )

    with st.expander("Grid simulation parameters", expanded=False):
        st.json(grid_parameters)

    if not grid_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            grid_data["phi"],
            grid_data["theta"],
            c=grid_data["kick"],
            cmap="viridis",
            s=2,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )

        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Kick number")

        ax.set_xlim(-3.141592653589793, 3.141592653589793)
        ax.set_ylim(0.0, 3.141592653589793)
        ax.set_xlabel(r"$\phi$", fontsize=13)
        ax.set_ylabel(r"$\theta$", fontsize=13)
        ax.set_title("Phase-space evolution: all orbit points")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Each point represents the final state of one initial "
            "condition. The initial grid and intermediate trajectories "
            "are not plotted."
        )

        csv_data = grid_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download phase-space CSV",
            data=csv_data,
            file_name="dkt_phase_space_grid.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("The grid simulation returned no final states.")

    st.caption(
        "Scientific interpretation: kr controls the degree of classical "
        "chaos; ktheta changes phase-space geometry, not the degree of chaos."
    )


for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

        if item.get("tools"):
            with st.expander("Tool activity", expanded=False):
                for tool in item["tools"]:
                    st.markdown(f"**{tool['name']}**")
                    if tool.get("args"):
                        st.code(
                            json.dumps(
                                tool["args"],
                                indent=2,
                                ensure_ascii=False,
                                default=str,
                            ),
                            language="json",
                        )
                    if tool.get("result"):
                        st.text(content_to_text(tool["result"]))


prompt = st.chat_input(
    "Ask about DKT dynamics, simulations, diagnostics, or research data…"
)

if prompt:
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing your request…"):
                agent = get_agent()
                response = agent.invoke({
                    "messages": (
                        st.session_state.agent_messages
                        + [HumanMessage(content=prompt)]
                    )
                })

            all_messages = response.get("messages", [])
            new_messages = all_messages[
                len(st.session_state.agent_messages) + 1:
            ]

            tool_calls = []

            for message in new_messages:
                if isinstance(message, AIMessage):
                    for call in (getattr(message, "tool_calls", None) or []):
                        tool_calls.append({
                            "name": call.get("name", "Tool"),
                            "args": call.get("args", {}),
                            "result": "",
                        })

                elif isinstance(message, ToolMessage):
                    name = getattr(message, "name", "Tool") or "Tool"
                    match = next(
                        (
                            item for item in reversed(tool_calls)
                            if item["name"] == name and not item["result"]
                        ),
                        None,
                    )
                    if match:
                        match["result"] = message.content
                    else:
                        tool_calls.append({
                            "name": name,
                            "args": {},
                            "result": message.content,
                        })

            final_answer = ""

            for message in reversed(all_messages):
                if (
                    isinstance(message, AIMessage)
                    and not (getattr(message, "tool_calls", None) or [])
                ):
                    final_answer = content_to_text(
                        message.content
                    ).strip()
                    if final_answer:
                        break

            if not final_answer:
                final_answer = (
                    "The agent completed the request but did not "
                    "return a text response."
                )

            st.session_state.agent_messages = all_messages
            st.markdown(final_answer)

            if tool_calls:
                with st.expander("Tool activity", expanded=False):
                    for tool in tool_calls:
                        st.markdown(f"**{tool['name']}**")
                        if tool["args"]:
                            st.code(
                                json.dumps(
                                    tool["args"],
                                    indent=2,
                                    ensure_ascii=False,
                                    default=str,
                                ),
                                language="json",
                            )
                        if tool["result"]:
                            st.text(content_to_text(tool["result"]))

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_answer,
                "tools": tool_calls,
            })

        except Exception as exc:
            error_text = (
                "I couldn't complete that request.\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "Check your Gemini API key, dependencies, "
                "network connection, and API quota."
            )
            st.error(error_text)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_text,
            })
