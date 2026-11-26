from __future__ import annotations
from datetime import datetime
from io import StringIO

import json
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dkt_agent.agent import build_agent
from dkt_agent.langchain_tools import (
    find_dkt_fixed_points,
    run_dkt_comparison_study,
    compare_dkt_simulations,
    run_classical_quantum_comparison_tool,
    search_research_evidence,
    read_research_csv_tool,
)
from dkt_agent.research_tools import (
    diagnose_classical_trajectory,
    diagnose_observable_timeseries,
    diagnose_lyapunov_ks,
    diagnose_mixing_correlations,
    run_ergodicity_variance_study,
    compare_trajectory_measures,
    summarize_research_csv,
)
from dkt_agent.tools import (simulate_dkt, simulate_dkt_grid, run_entanglement_dynamics, run_entanglement_heatmap)
from dkt_agent.streamlit_mixing_dashboard import render_mixing_hierarchy_dashboard
from dkt_agent.periodic_orbit_dashboard import render_periodic_orbit_panel
import matplotlib.pyplot as plt

def filename_token(value: float | int) -> str:
    """Format numerical parameters safely for filenames."""
    return (
        format(float(value), ".8g")
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def csv_with_metadata(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> bytes:
    """Create a CSV download with reproducibility metadata comments."""
    buffer = StringIO()

    buffer.write("# Double Kicked Top research data\n")
    buffer.write(
        f"# Generated: "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
    )

    for key, value in metadata.items():
        buffer.write(f"# {key}: {value}\n")

    buffer.write("#\n")
    frame.to_csv(buffer, index=False)

    return buffer.getvalue().encode("utf-8")

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

def render_tool_result(tool: dict[str, Any]) -> None:
    """Render structured, user-friendly results from DKT research tools."""
    raw = tool.get("result", "")
    name = tool.get("name", "unknown")

    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        payload = None

    if not isinstance(payload, dict):
        st.text(content_to_text(raw))
        return

    def number(value, digits=6):
        if isinstance(value, (int, float)):
            return f"{value:.{digits}g}"
        return str(value) if value is not None else "—"

    def parameter_line(parameters, include_initial=False):
        if not parameters:
            return
        terms = [
            rf"k_r={number(parameters.get('kr'))}",
            rf"k_\theta={number(parameters.get('ktheta'))}",
        ]
        if "k" in parameters:
            terms.extend([
                rf"k={number(parameters.get('k'))}",
                rf"k'={number(parameters.get('kprime'))}",
            ])
        if "p" in parameters:
            terms.append(rf"p={number(parameters.get('p'))}")
        if include_initial:
            terms.extend([
                rf"\theta_0={number(parameters.get('theta0'))}",
                rf"\phi_0={number(parameters.get('phi0'))}",
            ])
        st.markdown("$" + r",\quad ".join(terms) + "$")

    def metric_row(items):
        columns = st.columns(len(items))
        for column, (label, value) in zip(columns, items):
            column.metric(label, value)

    def show_limits(items):
        if items:
            with st.expander("Interpretation and limitations"):
                for item in items:
                    st.markdown(f"- {item}")

    def download(path, label, key):
        if path and Path(path).is_file():
            st.download_button(
                label,
                data=Path(path).read_bytes(),
                file_name=Path(path).name,
                mime="text/csv" if str(path).endswith(".csv")
                else "application/octet-stream",
                key=f"{key}_{id(tool)}",
            )

    def show_parameters(parameters):
        if parameters:
            st.markdown("**Physical parameters**")
            parameter_line(parameters, include_initial=True)

    # -----------------------------------------------------
    # Fixed points
    # -----------------------------------------------------
    if name == "find_dkt_fixed_points":
        st.markdown("### Fixed-point analysis")
        parameters = payload.get("parameters", {})
        points = payload.get("fixed_points", [])

        metric_row([
            ("Converged seeds", payload.get("converged_seeds", "—")),
            ("Unique fixed points", payload.get(
                "unique_fixed_points", len(points)
            )),
            ("Search status", payload.get("status", "unknown")),
        ])

        show_parameters(parameters)

        settings = pd.DataFrame([
            {"Setting": "Initial seeds", "Value": parameters.get("nseeds")},
            {"Setting": "Tolerance", "Value": parameters.get("tolerance")},
            {"Setting": "Maximum iterations",
             "Value": parameters.get("max_iterations")},
        ])
        st.dataframe(settings, width="stretch", hide_index=True)

        if points:
            st.markdown("**Fixed-point coordinates and stability**")
            frame = pd.DataFrame(points)
            preferred = [
                "fixed_point", "X", "Y", "Z", "residual", "stability",
                "iterations", "lambda1_real", "lambda1_imag",
                "lambda2_real", "lambda2_imag", "determinant",
            ]
            columns = [c for c in preferred if c in frame.columns]
            st.dataframe(
                frame[columns],
                width="stretch",
                hide_index=True,
            )

        download(
            payload.get("fixed_points_csv"),
            "Download fixed-points CSV",
            "fixed_points",
        )
        st.caption(
            "Stability classifications are numerical tangent-map diagnostics. "
            "The number of unique points found depends on the seed search "
            "and its tolerances."
        )
        return

    # -----------------------------------------------------
    # Controlled two-trajectory parameter comparison
    # -----------------------------------------------------
    if name == "run_dkt_comparison_study":
        sim_a = payload.get("simulation_a", {})
        sim_b = payload.get("simulation_b", {})
        comparison = payload.get("comparison", {})
        study = payload.get("study_parameters", {})

        st.markdown("### Controlled trajectory comparison")

        mode = study.get("comparison_mode", "fixed_kr")
        if mode == "fixed_ktheta":
            st.caption(
                "Comparison at fixed asymmetry parameter, with "
                "the average kicking strength varied."
            )
        else:
            st.caption(
                "Comparison at fixed average kicking strength, with "
                "the asymmetry parameter varied."
            )

        metric_row([
            ("Initial separation", number(
                comparison.get("initial_separation")
            )),
            ("Final separation", number(
                comparison.get("final_separation")
            )),
            ("Maximum separation", number(
                comparison.get("maximum_separation")
            )),
            ("RMS separation", number(
                comparison.get("rms_separation")
            )),
        ])

        st.caption(
            f"{comparison.get('number_of_steps', '—')} kicks · "
            f"{comparison.get('comparison_rows', '—')} recorded points"
        )

        col_a, col_b = st.columns(2)
        for column, label, simulation in [
            (col_a, "Simulation A", sim_a),
            (col_b, "Simulation B", sim_b),
        ]:
            with column:
                st.markdown(f"**{label}**")
                parameter_line(simulation.get("parameters", {}))
                st.caption(
                    f"Maximum norm error: "
                    f"{number(simulation.get('max_norm_error'))}"
                )

        statistics_rows = []
        for label, simulation in [
            ("A", sim_a), ("B", sim_b)
        ]:
            path = simulation.get("trajectory_csv")
            if not path or not Path(path).is_file():
                continue
            frame = pd.read_csv(path)
            if not {"X", "Y", "Z"}.issubset(frame.columns):
                continue
            row = {"Simulation": label}
            for axis in ["X", "Y", "Z"]:
                values = pd.to_numeric(
                    frame[axis], errors="coerce"
                ).dropna()
                row[f"Mean {axis}"] = values.mean()
                row[f"Mean {axis}²"] = (values ** 2).mean()
                row[f"Std. {axis}"] = values.std(ddof=0)
            statistics_rows.append(row)

        if statistics_rows:
            st.markdown("**Trajectory-averaged observables**")
            st.dataframe(
                pd.DataFrame(statistics_rows),
                width="stretch",
                hide_index=True,
            )

        comparison_path = (
            comparison.get("comparison_csv")
            or payload.get("csv_paths", {}).get("comparison")
        )
        if comparison_path and Path(comparison_path).is_file():
            frame = pd.read_csv(comparison_path)
            if "distance" in frame.columns:
                st.markdown("**Phase-space separation versus kick**")
                chart = frame.set_index("step")[["distance"]]
                st.line_chart(chart, height=300)
            download(
                comparison_path,
                "Download comparison CSV",
                "comparison",
            )

        for label, simulation in [("A", sim_a), ("B", sim_b)]:
            download(
                simulation.get("trajectory_csv"),
                f"Download trajectory {label}",
                f"trajectory_{label}",
            )

        st.caption(
            "Changing kθ changes the phase-space geometry and trajectory. "
            "It does not change the degree of classical chaos. "
            "Changing kr can change the degree of classical chaos."
        )
        return

    # -----------------------------------------------------
    # Direct comparison of existing trajectories
    # -----------------------------------------------------
    if name == "compare_dkt_simulations":
        st.markdown("### Trajectory comparison")

        metric_row([
            ("Initial separation", number(
                payload.get("initial_separation")
            )),
            ("Final separation", number(
                payload.get("final_separation")
            )),
            ("Maximum separation", number(
                payload.get("maximum_separation")
            )),
            ("RMS separation", number(
                payload.get("rms_separation")
            )),
        ])

        path = (
            payload.get("comparison_csv")
            or payload.get("csv_path")
        )
        if path and Path(path).is_file():
            frame = pd.read_csv(path)
            if "distance" in frame.columns:
                st.markdown("**Separation versus kick number**")
                chart = frame.set_index("step")[["distance"]]
                st.line_chart(chart, height=300)

            if {"delta_X", "delta_Y", "delta_Z"}.issubset(frame.columns):
                with st.expander("Cartesian component differences"):
                    st.line_chart(
                        frame.set_index("step")[
                            ["delta_X", "delta_Y", "delta_Z"]
                        ],
                        height=260,
                    )
            download(path, "Download comparison CSV", "direct_compare")

        return

    # -----------------------------------------------------
    # Classical-quantum correspondence
    # -----------------------------------------------------
    if name == "run_classical_quantum_comparison_tool":
        st.markdown("### Classical–quantum correspondence")

        parameters = payload.get("parameters", {})
        show_parameters(parameters)

        if parameters:
            st.caption(
                f"Spin s = {number(parameters.get('s'))} · "
                f"Hilbert dimension = "
                f"{parameters.get('hilbert_dimension', '—')} · "
                f"{parameters.get('nsteps', '—')} kicks"
            )

        st.markdown("**Vector correspondence errors**")
        metric_row([
            ("Final difference", number(
                payload.get("final_vector_difference")
            )),
            ("Maximum difference", number(
                payload.get("maximum_vector_difference")
            )),
            ("Mean difference", number(
                payload.get("mean_vector_difference")
            )),
            ("Vector RMSE", number(payload.get("vector_rmse"))),
        ])

        component_metrics = payload.get("component_metrics", {})
        if component_metrics:
            st.markdown("**Component-wise errors**")
            rows = []
            for axis in ["X", "Y", "Z"]:
                values = component_metrics.get(axis, {})
                rows.append({
                    "Component": axis,
                    "MAE": values.get("mae"),
                    "RMSE": values.get("rmse"),
                    "Maximum absolute error": values.get("max_abs_error"),
                })
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

        st.markdown("**Kick-resolved spin expectations**")
        path = payload.get("csv_path")
        if path and Path(path).is_file():
            frame = pd.read_csv(path)

            expected = [
                "classical_X", "quantum_X",
                "classical_Y", "quantum_Y",
                "classical_Z", "quantum_Z",
            ]
            if set(expected).issubset(frame.columns):
                st.line_chart(
                    frame.set_index("kick")[expected],
                    height=350,
                )

            delta_columns = ["delta_X", "delta_Y", "delta_Z"]
            if set(delta_columns).issubset(frame.columns):
                st.markdown(
                    "**Classical minus quantum expectation values**"
                )
                st.line_chart(
                    frame.set_index("kick")[delta_columns],
                    height=280,
                )

            st.markdown("**Numerical consistency checks**")
            norm_rows = []
            for key, label in [
                ("initial_alignment_error", "Initial alignment error"),
                ("max_classical_norm_error", "Maximum classical norm error"),
                ("max_quantum_state_norm_error",
                 "Maximum quantum state norm error"),
            ]:
                if key in payload:
                    norm_rows.append({
                        "Check": label,
                        "Value": payload[key],
                    })
            if norm_rows:
                st.dataframe(
                    pd.DataFrame(norm_rows),
                    width="stretch",
                    hide_index=True,
                )

            download(
                path,
                "Download classical–quantum comparison CSV",
                "cq",
            )
        else:
            st.warning("The classical–quantum CSV is unavailable.")

        st.caption(
            "Quantum curves are normalized spin expectations. Their "
            "difference from one classical orbit is a finite-time "
            "correspondence diagnostic, not a measure of classical "
            "chaoticity. Entanglement entropy is a separate observable."
        )
        return

    # -----------------------------------------------------
    # Classical trajectory geometry
    # -----------------------------------------------------
    if name == "diagnose_classical_trajectory":
        st.markdown("### Classical trajectory geometry")

        metric_row([
            ("Trajectory points", payload.get("n_points", "—")),
            ("Correlation dimension", number(
                payload.get("correlation_dimension")
            )),
            ("Box-counting dimension", number(
                payload.get("box_dimension")
            )),
        ])

        parameters = payload.get("parameters", {})
        if parameters:
            st.markdown("**Diagnostic settings**")
            st.json(parameters)

        artifacts = payload.get("artifacts", {})
        for key, title in [
            ("correlation_plot", "Correlation integral and local slopes"),
            ("box_plot", "Shifted-grid occupied-cell scaling"),
        ]:
            path = artifacts.get(key)
            if path and Path(path).is_file():
                st.markdown(f"**{title}**")
                st.image(str(path), width="stretch")

        for key, label in [
            ("correlation_curve_csv", "Download correlation curve CSV"),
            ("box_curve_csv", "Download box-counting curve CSV"),
            ("summary_json", "Download diagnostic summary JSON"),
        ]:
            download(artifacts.get(key), label, key)

        show_limits(payload.get("interpretation_limits"))
        return

    # -----------------------------------------------------
    # Observable time-series diagnostics
    # -----------------------------------------------------
    if name == "diagnose_observable_timeseries":
        st.markdown("### Observable time-series diagnostics")

        metric_row([
            ("Samples analyzed", payload.get("samples_analyzed", "—")),
            ("Transient discarded", payload.get("transient_discarded", "—")),
            ("Maximum lag", payload.get(
                "autocorrelation", {}
            ).get("max_lag", "—")),
        ])

        averages = payload.get("birkhoff_time_averages", {})
        if averages:
            st.markdown("**Finite-record Birkhoff averages**")
            rows = []
            for observable, values in averages.items():
                rows.append({
                    "Observable": observable,
                    "Mean": values.get("mean"),
                    "Standard deviation": values.get("standard_deviation"),
                    "Samples": values.get("samples"),
                })
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

        fluctuations = payload.get("block_fluctuations", {})
        fluctuation_rows = []
        for observable, records in fluctuations.items():
            for record in records:
                fluctuation_rows.append({
                    "Observable": observable,
                    **record,
                })

        if fluctuation_rows:
            st.markdown("**Block-mean fluctuations**")
            fluctuation_frame = pd.DataFrame(fluctuation_rows)
            st.dataframe(
                fluctuation_frame,
                width="stretch",
                hide_index=True,
            )

            selected = st.selectbox(
                "Observable for block-variance plot",
                list(fluctuations.keys()),
                key=f"observable_block_{id(tool)}",
            )
            selected_frame = fluctuation_frame[
                fluctuation_frame["Observable"] == selected
            ]
            if not selected_frame.empty:
                st.line_chart(
                    selected_frame.set_index("block_size")[
                        ["variance_of_block_means"]
                    ],
                    height=280,
                )

        correlations = payload.get(
            "autocorrelation", {}
        ).get("observables", {})
        if correlations:
            st.markdown("**Autocorrelation functions**")
            corr_frame = pd.DataFrame(correlations)
            corr_frame.index.name = "Lag"
            st.line_chart(corr_frame[["X", "Y", "Z"]], height=300)

            with st.expander("All nine observable autocorrelations"):
                st.line_chart(corr_frame, height=350)

        show_limits(payload.get("interpretation_limits"))
        return

    # -----------------------------------------------------
    # Lyapunov exponents and finite-time KS estimate
    # -----------------------------------------------------
    if name == "diagnose_lyapunov_ks":
        st.markdown("### Lyapunov and finite-time KS diagnostics")

        parameters = payload.get("parameters", {})
        show_parameters(parameters)

        exponents = payload.get(
            "lyapunov_exponents_nats_per_kick", []
        )
        if len(exponents) >= 2:
            metric_row([
                ("λ₁ (nats/kick)", number(exponents[0])),
                ("λ₂ (nats/kick)", number(exponents[1])),
                ("Sum of exponents", number(
                    payload.get("sum_of_exponents_nats_per_kick")
                )),
                ("Finite-time Pesin estimate", number(
                    payload.get(
                        "finite_time_pesin_ks_estimate_nats_per_kick"
                    )
                )),
            ])

        check = payload.get(
            "area_preservation_consistency_check", {}
        )
        if check:
            st.markdown("**Area-preservation consistency check**")
            metric_row([
                ("Expected sum", number(check.get("expected_sum"))),
                ("Measured sum", number(check.get("measured_sum"))),
                ("Absolute residual", number(
                    check.get("absolute_residual")
                )),
            ])

        checkpoints = payload.get("finite_time_checkpoints", [])
        if checkpoints:
            st.markdown("**Finite-time checkpoint estimates**")
            frame = pd.DataFrame(checkpoints)
            st.dataframe(
                frame,
                width="stretch",
                hide_index=True,
            )

            if "analyzed_steps" in frame.columns:
                columns = [
                    c for c in ["lambda1", "lambda2"]
                    if c in frame.columns
                ]
                if columns:
                    st.line_chart(
                        frame.set_index("analyzed_steps")[columns],
                        height=300,
                    )

        artifacts = payload.get("artifacts", {})
        download(artifacts.get("csv"), "Download Lyapunov checkpoints CSV",
                 "lyapunov")
        show_limits(payload.get("interpretation_limits"))
        return

    # -----------------------------------------------------
    # Mixing / autocorrelation diagnostics
    # -----------------------------------------------------
    if name == "diagnose_mixing_correlations":
        st.markdown("### Correlation-decay diagnostics")

        metric_row([
            ("Samples analyzed", payload.get("samples_analyzed", "—")),
            ("Maximum lag", payload.get("max_lag", "—")),
            ("Transient discarded", payload.get("transient_discarded", "—")),
        ])

        summaries = payload.get("correlation_decay_summaries", {})
        if summaries:
            rows = []
            for observable, values in summaries.items():
                rows.append({
                    "Observable": observable,
                    "Status": values.get("status"),
                    "First zero-crossing lag":
                        values.get("first_zero_crossing_lag"),
                    "Threshold-crossing lag":
                        values.get("first_five_lag_threshold_crossing"),
                    "Threshold": values.get("threshold"),
                })
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

        artifacts = payload.get("artifacts", {})
        path = artifacts.get("curves_csv")
        if path and Path(path).is_file():
            frame = pd.read_csv(path)
            if "lag" in frame.columns:
                st.markdown("**Autocorrelation versus lag**")
                selected = st.multiselect(
                    "Observables to display",
                    [c for c in frame.columns if c != "lag"],
                    default=["X", "Y", "Z"],
                    key=f"mixing_obs_{id(tool)}",
                )
                if selected:
                    st.line_chart(
                        frame.set_index("lag")[selected],
                        height=320,
                    )
            download(path, "Download autocorrelation curves CSV",
                     "mixing_curves")

        show_limits(payload.get("interpretation_limits"))
        return

    # -----------------------------------------------------
    # Finite-time Birkhoff variance study
    # -----------------------------------------------------
    if name == "run_ergodicity_variance_study":
        st.markdown("### Birkhoff-average variance study")

        parameters = payload.get("parameters", {})
        show_parameters(parameters)

        metric_row([
            ("Trajectories", parameters.get("n_trajectories", "—")),
            ("Total kicks", parameters.get("nsteps", "—")),
            ("Transient discarded",
             parameters.get("transient_discarded", "—")),
            ("Seed", parameters.get("seed", "—")),
        ])

        data = payload.get("variance_scaling", [])
        if data:
            frame = pd.DataFrame(data)
            st.markdown("**Variance across initial conditions**")

            observables = list(frame["observable"].drop_duplicates())
            selected = st.multiselect(
                "Observables",
                observables,
                default=["X", "Y", "Z"],
                key=f"variance_obs_{id(tool)}",
            )
            if selected:
                selected_frame = frame[
                    frame["observable"].isin(selected)
                ]
                chart = selected_frame.pivot(
                    index="analyzed_steps",
                    columns="observable",
                    values="ensemble_variance",
                )
                st.line_chart(chart, height=320)

            with st.expander("Full variance-scaling data"):
                st.dataframe(
                    frame,
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("**Reference spherical-area averages**")
            reference = (
                frame.groupby("observable", as_index=False)
                [["reference_area_mean"]].first()
            )
            st.dataframe(
                reference,
                width="stretch",
                hide_index=True,
            )

        artifacts = payload.get("artifacts", {})
        download(artifacts.get("csv"), "Download Birkhoff variance CSV",
                 "birkhoff_variance")
        show_limits(payload.get("interpretation_limits"))
        return

    # -----------------------------------------------------
    # Empirical phase-space measure comparison
    # -----------------------------------------------------
    if name == "compare_trajectory_measures":
        st.markdown("### Empirical phase-space measure comparison")

        labels = payload.get("labels", [])
        matrix = payload.get("jensen_shannon_divergence_base2", [])

        partition = payload.get("partition", {})
        if partition:
            st.caption(
                f"Equal-area partition: n_z={partition.get('n_z', '—')}, "
                f"n_φ={partition.get('n_phi', '—')}"
            )

        if labels and matrix:
            frame = pd.DataFrame(matrix, index=labels, columns=labels)
            frame.index.name = "Trajectory"
            st.dataframe(frame, width="stretch")

            values = [
                matrix[i][j]
                for i in range(len(matrix))
                for j in range(i + 1, len(matrix))
            ]
            if values:
                metric_row([
                    ("Pairwise comparisons", len(values)),
                    ("Minimum divergence", number(min(values))),
                    ("Maximum divergence", number(max(values))),
                    ("Mean divergence", number(sum(values) / len(values))),
                ])

        download(
            payload.get("csv_path"),
            "Download measure-comparison CSV",
            "measure_comparison",
        )
        show_limits(payload.get("interpretation_limits"))
        return

    # -----------------------------------------------------
    # Evidence search
    # -----------------------------------------------------
    if name == "search_research_evidence":
        st.markdown("### Research evidence search")

        matches = payload.get("matches", [])
        metric_row([
            ("Files scanned", payload.get("files_scanned", "—")),
            ("Files matched", payload.get("files_matched", len(matches))),
            ("Files returned", payload.get("files_returned", len(matches))),
        ])

        if payload.get("query"):
            st.caption(f"Search query: {payload['query']}")

        if matches:
            for index, match in enumerate(matches, start=1):
                title = (
                    f"{index}. {match.get('path', 'Unknown file')} "
                    f"· score {match.get('score', '—')}"
                )
                with st.expander(title):
                    st.caption(
                        f"Match type: {match.get('match_type', '—')}"
                    )
                    st.caption(
                        "Matched terms: "
                        + ", ".join(
                            match.get("content_matched_terms", [])
                            + match.get("filename_matched_terms", [])
                        )
                    )
                    for line in match.get("matching_lines", []):
                        st.markdown(
                            f"`L{line.get('line', '—')}` "
                            f"{line.get('text', '')}"
                        )
        else:
            st.info("No matching research files were found.")

        show_limits(payload.get("limitations"))
        return

    # -----------------------------------------------------
    # Read CSV
    # -----------------------------------------------------
    if name == "read_research_csv_tool":
        st.markdown("### Research CSV inspection")

        rows = payload.get("rows", [])
        columns = payload.get("columns", [])

        metric_row([
            ("Rows returned", payload.get("rows_returned", len(rows))),
            ("Columns", len(columns)),
            ("Possibly truncated",
             "Yes" if payload.get("possibly_truncated") else "No"),
        ])

        st.caption(payload.get("path", ""))

        if columns:
            st.markdown("**CSV columns**")
            st.code(", ".join(columns))

        if rows:
            records = []
            for row in rows:
                record = {"CSV line": row.get("line")}
                record.update(row.get("values", {}))
                records.append(record)
            st.dataframe(
                pd.DataFrame(records),
                width="stretch",
                hide_index=True,
            )

        st.caption(payload.get("interpretation_note", ""))
        return

    # -----------------------------------------------------
    # CSV descriptive summary
    # -----------------------------------------------------
    if name == "summarize_research_csv":
        st.markdown("### Research CSV summary")

        metric_row([
            ("Rows analyzed", payload.get("rows_analyzed", "—")),
            ("Columns", len(payload.get("columns", []))),
            ("Possibly truncated",
             "Yes" if payload.get("possibly_truncated") else "No"),
        ])

        st.caption(payload.get("path", ""))

        missing = payload.get("missing_or_nonfinite_values", {})
        non_numeric = payload.get("non_numeric_values", {})
        statistics = payload.get("numeric_statistics", {})

        rows = []
        for column in payload.get("columns", []):
            stat = statistics.get(column, {})
            rows.append({
                "Column": column,
                "Numeric values": stat.get("numeric_values", 0),
                "Missing/nonfinite": missing.get(column, 0),
                "Non-numeric": non_numeric.get(column, 0),
                "Mean": stat.get("mean"),
                "Std. dev.": stat.get("standard_deviation"),
                "Minimum": stat.get("minimum"),
                "Median": stat.get("median"),
                "Maximum": stat.get("maximum"),
            })

        if rows:
            st.markdown("**Column-wise descriptive statistics**")
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

        show_limits(payload.get("limitations"))
        return

    # -----------------------------------------------------
    # Quantum entanglement dynamics
    # -----------------------------------------------------
    if name == "run_quantum_entanglement_dynamics":
        frame = (
            pd.DataFrame(payload["data"])
            if "data" in payload else
            pd.read_csv(payload["csv_path"])
            if payload.get("csv_path") and Path(payload["csv_path"]).is_file()
            else pd.DataFrame()
        )

        st.markdown("### Quantum entanglement dynamics")
        if not frame.empty and "normalized_entropy" in frame.columns:
            kick_col = "kick" if "kick" in frame.columns else None
            chart = (
                frame.set_index(kick_col)[["normalized_entropy"]]
                if kick_col else frame[["normalized_entropy"]]
            )
            st.line_chart(chart, height=320)
            metric_row([
                ("Initial entropy", number(
                    frame["normalized_entropy"].iloc[0]
                )),
                ("Final entropy", number(
                    frame["normalized_entropy"].iloc[-1]
                )),
                ("Maximum entropy", number(
                    frame["normalized_entropy"].max()
                )),
            ])
            st.caption(
                "Normalized von Neumann entanglement entropy versus kick."
            )
            st.download_button(
                "Download entanglement dynamics CSV",
                frame.to_csv(index=False).encode("utf-8"),
                "quantum_entanglement_dynamics.csv",
                "text/csv",
                key=f"entropy_{id(tool)}",
            )
        return

    # -----------------------------------------------------
    # Quantum entanglement heatmap
    # -----------------------------------------------------
    if name == "run_quantum_entanglement_heatmap":
        frame = (
            pd.DataFrame(payload["data"])
            if "data" in payload else
            pd.read_csv(payload["csv_path"])
            if payload.get("csv_path") and Path(payload["csv_path"]).is_file()
            else pd.DataFrame()
        )

        st.markdown("### Quantum entanglement phase-space scan")
        if not frame.empty and {
            "status", "theta", "phi", "normalized_average_entropy"
        }.issubset(frame.columns):
            good = frame[frame["status"] == "success"]
            if not good.empty:
                matrix = good.pivot(
                    index="theta",
                    columns="phi",
                    values="normalized_average_entropy",
                )
                fig, ax = plt.subplots(figsize=(8, 4.5))
                mesh = ax.pcolormesh(
                    matrix.columns.to_numpy(),
                    matrix.index.to_numpy(),
                    matrix.to_numpy(),
                    shading="auto",
                    vmin=0,
                    vmax=1,
                )
                fig.colorbar(
                    mesh, ax=ax,
                    label="Normalized average entropy",
                )
                ax.set_xlabel(r"$\phi_0$")
                ax.set_ylabel(r"$\theta_0$")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            metric_row([
                ("Successful points", payload.get("successful_points", 0)),
                ("Failed points", payload.get("failed_points", 0)),
            ])

            st.download_button(
                "Download entanglement heatmap CSV",
                frame.to_csv(index=False).encode("utf-8"),
                "quantum_entanglement_heatmap.csv",
                "text/csv",
                key=f"heatmap_{id(tool)}",
            )
        return

    # -----------------------------------------------------
    # Generic structured fallback
    # -----------------------------------------------------
    st.warning(
        f"No dedicated presentation is defined yet for `{name}`. "
        "Showing the structured result below."
    )
    st.json({
        key: value for key, value in payload.items()
        if key not in {"julia_stdout"}
    })


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


with st.sidebar:
    st.header("Quick simulations")

    with st.expander("DKT parameter guide", expanded=False):
        st.markdown("**Average kicking strength**")
        st.latex(r"k_r = \frac{k+k'}{2}")
        st.caption(
            "Controls the degree of classical chaos and the "
            "regular-to-chaotic transition."
        )
        st.markdown("**Asymmetry parameter**")
        st.latex(r"k_\theta = \frac{k-k'}{2}")
        st.caption(
            "Changes phase-space geometry and its orientation/structure; "
            "it does not control the degree of classical chaos."
        )
        st.latex(r"k = k_r+k_\theta, \qquad k\prime = k_r-k_\theta")

    with st.expander("Interactive DKT simulation", expanded=False):
        st.caption("Evolve one classical initial condition under the DKT map.")

        with st.form("dkt_simulation_form"):
            st.caption("Run the classical DKT Julia simulation.")

            sim_kr = st.number_input(
                r"Average kicking strength $k_r$",
                min_value=0.0, max_value=100.0,
                value=1.25, step=0.1,
            )
            sim_ktheta = st.number_input(
                r"Asymmetry parameter $k_\theta$",
                min_value=-1000000.0, max_value=1000000.0,
                value=0.0, step=100.0, format="%.4f",
            )
            sim_theta0 = st.number_input(
                "Initial theta (radians)",
                min_value=0.0, max_value=3.141592653589793,
                value=1.5707963268, step=0.05,
            )
            sim_phi0 = st.number_input(
                "Initial phi (radians)",
                min_value=-3.141592653589793,
                max_value=3.141592653589793,
                value=-1.5707963268, step=0.05,
            )
            sim_nsteps = st.number_input(
                "Number of kicks",
                min_value=1, max_value=50000,
                value=500, step=10,
            )
            sim_p = st.number_input(
                r"Rotation angle $p$ (radians)",
                min_value=0.0, max_value=6.283185307179586,
                value=1.5707963267948966, step=0.05,
            )
            sim_timeout = st.number_input(
                "Timeout (seconds)",
                min_value=1, max_value=1800,
                value=300, step=30,
            )

            run_simulation = st.form_submit_button(
                "Run DKT Simulation",
                width="stretch",
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

    with st.expander("θ–φ phase grid", expanded=False):
        st.caption(
            "Evolve a spherical grid of initial conditions and display "
            "all orbit points from kick 0 through the final kick."
        )

        with st.form("dkt_grid_simulation_form"):
            grid_resolution = st.number_input(
                "Grid resolution (N × N)",
                min_value=5, max_value=50,
                value=25, step=1,
                help="Number of initial points along each angular direction (5–50).",
            )
            grid_kr = st.number_input(
                r"Average kicking strength $k_r$",
                min_value=0.0, max_value=100.0,
                value=1.0, step=0.1, key="grid_kr",
            )
            grid_ktheta = st.number_input(
                r"Asymmetry parameter $k_\theta$",
                min_value=-1000000.0, max_value=1000000.0,
                value=0.0, step=100.0,
                format="%.4f", key="grid_ktheta",
            )
            grid_p = st.number_input(
                r"Rotation angle $p$ (radians)",
                min_value=0.0, max_value=6.283185307179586,
                value=1.5707963267948966,
                step=0.05, key="grid_p",
            )
            grid_nsteps = st.number_input(
                "Number of kicks",
                min_value=1, max_value=10000,
                value=500, step=10, key="grid_nsteps",
            )

            run_grid_simulation = st.form_submit_button(
                "Run phase-space grid",
                width="stretch",
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

    with st.expander("Quantum entanglement dynamics", expanded=False):
        st.caption(
            "Evolve a spin-coherent state and study normalized "
            "von Neumann entanglement entropy versus kick number."
        )
        with st.form("entanglement_dynamics_form"):
            ed_s = st.number_input(
                "Spin s", min_value=1.0, max_value=100.0,
                value=5.0, step=0.5, key="ed_s"
            )
            ed_kr = st.number_input(
                r"$k_r$", value=1.0, step=0.1, key="ed_kr"
            )
            ed_kt = st.number_input(
                r"$k_\theta$", value=0.0, step=100.0, key="ed_kt"
            )
            ed_theta = st.number_input(
                r"Initial $\theta_0$", min_value=0.0,
                max_value=3.141592653589793,
                value=0.75, key="ed_theta"
            )
            ed_phi = st.number_input(
                r"Initial $\phi_0$", min_value=-3.141592653589793,
                max_value=3.141592653589793,
                value=-1.0, key="ed_phi"
            )
            ed_q = st.number_input(
                "Subsystem size Q (rdm_dim)", min_value=1,
                max_value=99, value=2, step=1, key="ed_q"
            )
            ed_n = st.number_input(
                "Number of kicks T", min_value=1,
                max_value=50000, value=1000, step=100, key="ed_n"
            )
            ed_run = st.form_submit_button(
                "Run entropy dynamics", width="stretch"
            )

        if ed_run:
            try:
                with st.spinner(
                    "Computing quantum entanglement dynamics in Julia..."
                ):
                    st.session_state["entanglement_dynamics_result"] = (
                        run_entanglement_dynamics(
                            s=float(ed_s), kr=float(ed_kr),
                            ktheta=float(ed_kt),
                            theta0=float(ed_theta), phi0=float(ed_phi),
                            nsteps=int(ed_n), rdm_dim=int(ed_q)
                        )
                    )
                    st.session_state.pop("entanglement_dynamics_error", None)
            except Exception as exc:
                st.session_state["entanglement_dynamics_error"] = str(exc)

    with st.expander("Long-time-averaged entropy heatmap", expanded=False):
        st.caption(
            "Calculate the long-time average of normalized entanglement "
            "entropy over a grid of initial coherent-state orientations."
        )
        with st.form("entanglement_heatmap_form"):
            hm_s = st.number_input(
                "Spin s", min_value=1.0, max_value=100.0,
                value=5.0, step=0.5, key="hm_s"
            )
            hm_kr = st.number_input(
                r"$k_r$", value=1.0, step=0.1, key="hm_kr"
            )
            hm_kt = st.number_input(
                r"$k_\theta$", value=0.0, step=100.0, key="hm_kt"
            )
            hm_q = st.number_input(
                "Subsystem size Q (rdm_dim)", min_value=1,
                max_value=99, value=2, step=1, key="hm_q"
            )
            hm_n = st.number_input(
                "Final kick T", min_value=2, max_value=50000,
                value=1000, step=100, key="hm_n"
            )
            hm_t0 = st.number_input(
                "Discard transient T0", min_value=0,
                max_value=49999, value=100, step=10, key="hm_t0"
            )
            hm_grid = st.number_input(
                "Grid resolution", min_value=2, max_value=100,
                value=15, step=1, key="hm_grid"
            )
            hm_run = st.form_submit_button(
                "Run averaged entropy heatmap", width="stretch"
            )

        if hm_run:
            try:
                with st.spinner(
                    "Computing long-time entropy heatmap in Julia..."
                ):
                    st.session_state["entanglement_heatmap_result"] = (
                        run_entanglement_heatmap(
                            s=float(hm_s), kr=float(hm_kr),
                            ktheta=float(hm_kt),
                            nsteps=int(hm_n), transient=int(hm_t0),
                            rdm_dim=int(hm_q), grid_size=int(hm_grid)
                        )
                    )
                    st.session_state.pop("entanglement_heatmap_error", None)
            except Exception as exc:
                st.session_state["entanglement_heatmap_error"] = str(exc)


    st.divider()


    if st.button("Clear conversation", width="stretch"):
        st.session_state.agent_messages = []
        st.session_state.chat_history = []
        st.rerun()


if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------------
# Functional research workspaces
# ------------------------------------------------------------------

def run_workspace_tool(tool, arguments: dict, workspace: str):
    """Execute a registered research tool directly, without calling Gemini."""
    try:
        with st.spinner(f"Running {tool.name}..."):
            result = tool.invoke(arguments)

        st.session_state["workspace_results"][workspace] = {
            "answer": "",
            "tools": [{
                "name": tool.name,
                "content": content_to_text(result),
            }],
            "error": None,
        }

    except Exception as exc:
        st.session_state["workspace_results"][workspace] = {
            "answer": "",
            "tools": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_workspace_request(prompt: str, workspace: str):
    """Run one supervised workspace request without altering chat history."""
    try:
        with st.spinner("Research agent is working..."):
            response = get_agent().invoke({
                "messages": [HumanMessage(content=prompt)]
            })

        messages = response.get("messages", [])
        tool_results = []
        final_answer = ""

        for message in messages:
            if isinstance(message, ToolMessage):
                tool_results.append({
                    "name": getattr(message, "name", "Research tool"),
                    "content": content_to_text(message.content),
                })
            elif isinstance(message, AIMessage) and not getattr(
                message, "tool_calls", []
            ):
                final_answer = content_to_text(message.content)

        st.session_state["workspace_results"][workspace] = {
            "answer": final_answer,
            "tools": tool_results,
            "error": None,
        }

    except Exception as exc:
        st.session_state["workspace_results"][workspace] = {
            "answer": "",
            "tools": [],
            "error": str(exc),
        }


def show_workspace_result(workspace: str):
    result = st.session_state.get("workspace_results", {}).get(workspace)

    if not result:
        return

    if result.get("error"):
        st.error(result["error"])
        return

    if result.get("answer"):
        st.markdown("**Agent response**")
        st.markdown(result["answer"])

    for index, tool_result in enumerate(result.get("tools", [])):
        with st.expander(
            f"Tool result {index + 1}: {tool_result['name']}",
            expanded=index == 0,
        ):
            render_tool_result({
                "name": tool_result["name"],
                "result": tool_result["content"],
            })


if "workspace_results" not in st.session_state:
    st.session_state["workspace_results"] = {}


st.divider()
st.header("Research workspaces")
st.caption(
    "Diagnose, Compare and Research Data execute selected tools directly. "
    "Explain and the main chat use the Gemini research agent."
)

diagnose_tab, compare_tab, data_tab, explain_tab = st.tabs([
    "Diagnostics", "Comparisons", "Data & evidence", "Research assistant",
])

# ------------------------------------------------------------------
# 1. DIAGNOSE
# ------------------------------------------------------------------

with diagnose_tab:
    st.subheader("Classical-dynamics diagnostics")
    st.caption(
        "Choose one of the seven diagnostic workflows. The controls below "
        "are the actual inputs passed to the selected analysis."
    )

    diagnostic = st.selectbox(
        "Diagnostic",
        [
            "Fixed points and local stability",
            "Periodic orbits (all searched periods)",
            "Trajectory geometry",
            "Observable time series",
            "Finite-time Lyapunov exponents and KS estimate",
            "Autocorrelation and mixing diagnostics",
            "Birkhoff-average variance study",
        ],
        key="ws_diagnostic",
    )

    if diagnostic == "Periodic orbits (all searched periods)":
        render_periodic_orbit_panel(in_tab=True)

    elif diagnostic == "Fixed points and local stability":
        with st.form("ws_fixed_points_form"):
            c1, c2 = st.columns(2)
            with c1:
                d_kr = st.number_input(
                    r"Average kicking parameter $k_r$",
                    value=1.0, step=0.1,
                    key="ws_fp_kr",
                )
                d_kt = st.number_input(
                    r"Asymmetry parameter $k_\theta$",
                    value=0.0, step=100.0,
                    key="ws_fp_kt",
                )
            with c2:
                d_p = st.number_input(
                    r"Rotation angle $p$",
                    value=1.5707963267948966,
                    key="ws_fp_p",
                )
                d_seeds = st.number_input(
                    "Number of initial seeds",
                    min_value=10, max_value=5000,
                    value=200, step=10,
                    key="ws_fp_seeds",
                )

            diagnose_submit = st.form_submit_button(
                "Run fixed-point search",
                width="stretch",
            )

        if diagnose_submit:
            run_workspace_tool(
                find_dkt_fixed_points,
                {
                    "kr": float(d_kr),
                    "ktheta": float(d_kt),
                    "p": float(d_p),
                    "nseeds": int(d_seeds),
                },
                "diagnose",
            )

    else:
        with st.form("ws_diagnostic_form"):
            trajectory_path = ""
            if diagnostic in {"Trajectory geometry", "Observable time series", "Autocorrelation and mixing diagnostics"}:
                trajectory_path = st.text_input(
                    "Existing trajectory CSV path",
                    placeholder="Paste the exact path returned by a simulation or evidence search",
                    key="ws_diag_path",
                )
            if diagnostic == "Trajectory geometry":
                st.markdown("**Geometric diagnostic settings**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    d_theiler = st.number_input("Theiler window", min_value=0, value=20, step=1, key="ws_geom_theiler")
                with c2:
                    d_pairs = st.number_input("Random point pairs", min_value=1000, max_value=2000000, value=200000, step=10000, key="ws_geom_pairs")
                with c3:
                    d_geom_seed = st.number_input("Random seed", min_value=0, value=1234, step=1, key="ws_geom_seed")
            elif diagnostic == "Observable time series":
                st.markdown("**Time-series settings**")
                c1, c2 = st.columns(2)
                with c1:
                    d_ts_transient = st.number_input("Transient kicks to discard", min_value=0, value=0, step=10, key="ws_ts_transient")
                with c2:
                    d_ts_lag = st.number_input("Maximum autocorrelation lag", min_value=1, max_value=5000, value=200, step=10, key="ws_ts_lag")
            elif diagnostic == "Finite-time Lyapunov exponents and KS estimate":
                st.markdown("**Map and initial-condition parameters**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    d_lyap_kr = st.number_input(r"Average kicking strength $k_r$", value=1.0, step=0.1, key="ws_lyap_kr")
                    d_lyap_kt = st.number_input(r"Asymmetry parameter $k_\theta$", value=0.0, step=100.0, key="ws_lyap_kt")
                with c2:
                    d_lyap_theta = st.number_input(r"Initial polar angle $\theta_0$", value=0.75, step=0.05, key="ws_lyap_theta")
                    d_lyap_phi = st.number_input(r"Initial azimuth $\phi_0$", value=-1.0, step=0.05, key="ws_lyap_phi")
                with c3:
                    d_lyap_p = st.number_input(r"Rotation angle $p$", value=1.5707963267948966, step=0.05, key="ws_lyap_p")
                st.markdown("**Evolution settings**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    d_lyap_steps = st.number_input("Total kicks", min_value=100, max_value=500000, value=20000, step=1000, key="ws_lyap_steps")
                with c2:
                    d_lyap_transient = st.number_input("Transient kicks to discard", min_value=0, max_value=499900, value=1000, step=100, key="ws_lyap_transient")
                with c3:
                    d_lyap_timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=3600, value=900, step=30, key="ws_lyap_timeout")
            elif diagnostic == "Autocorrelation and mixing diagnostics":
                st.markdown("**Correlation settings**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    d_mix_transient = st.number_input("Transient kicks to discard", min_value=0, value=0, step=10, key="ws_mix_transient")
                with c2:
                    d_mix_lag = st.number_input("Maximum lag", min_value=1, max_value=5000, value=2000, step=50, key="ws_mix_lag")
                with c3:
                    d_mix_threshold = st.number_input("Correlation threshold", min_value=0.001, max_value=0.999, value=0.05, step=0.01, format="%.3f", key="ws_mix_threshold")
            elif diagnostic == "Birkhoff-average variance study":
                st.markdown("**Map parameters**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    d_birk_kr = st.number_input(r"Average kicking strength $k_r$", value=1.0, step=0.1, key="ws_birk_kr")
                with c2:
                    d_birk_kt = st.number_input(r"Asymmetry parameter $k_\theta$", value=0.0, step=100.0, key="ws_birk_kt")
                with c3:
                    d_birk_p = st.number_input(r"Rotation angle $p$", value=1.5707963267948966, step=0.05, key="ws_birk_p")
                st.markdown("**Ensemble settings**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    d_birk_count = st.number_input("Initial conditions", min_value=3, max_value=24, value=8, step=1, key="ws_birk_count")
                with c2:
                    d_birk_steps = st.number_input("Total kicks", min_value=100, max_value=50000, value=10000, step=500, key="ws_birk_steps")
                with c3:
                    d_birk_transient = st.number_input("Transient kicks to discard", min_value=0, max_value=49999, value=1000, step=100, key="ws_birk_transient")
                d_birk_seed = st.number_input("Random seed", min_value=0, value=20261124, step=1, key="ws_birk_seed")
            diagnose_submit = st.form_submit_button("Run diagnostic", type="primary", width="stretch")
        if diagnose_submit:
            if diagnostic in {"Trajectory geometry", "Observable time series", "Autocorrelation and mixing diagnostics"} and not trajectory_path.strip():
                st.error("Please provide an existing trajectory CSV path.")
            elif diagnostic == "Trajectory geometry":
                run_workspace_tool(diagnose_classical_trajectory, {
                    "trajectory_csv": trajectory_path.strip(),
                    "theiler": int(d_theiler), "n_pairs": int(d_pairs), "seed": int(d_geom_seed),
                }, "diagnose")
            elif diagnostic == "Observable time series":
                run_workspace_tool(diagnose_observable_timeseries, {
                    "trajectory_csv": trajectory_path.strip(),
                    "transient": int(d_ts_transient), "max_lag": int(d_ts_lag),
                }, "diagnose")
            elif diagnostic == "Finite-time Lyapunov exponents and KS estimate":
                run_workspace_tool(diagnose_lyapunov_ks, {
                    "kr": float(d_lyap_kr), "ktheta": float(d_lyap_kt),
                    "theta0": float(d_lyap_theta), "phi0": float(d_lyap_phi),
                    "p": float(d_lyap_p), "nsteps": int(d_lyap_steps),
                    "transient": int(d_lyap_transient), "timeout": int(d_lyap_timeout),
                }, "diagnose")
            elif diagnostic == "Autocorrelation and mixing diagnostics":
                run_workspace_tool(diagnose_mixing_correlations, {
                    "trajectory_csv": trajectory_path.strip(),
                    "transient": int(d_mix_transient), "max_lag": int(d_mix_lag),
                    "threshold": float(d_mix_threshold),
                }, "diagnose")
            elif diagnostic == "Birkhoff-average variance study":
                run_workspace_tool(run_ergodicity_variance_study, {
                    "kr": float(d_birk_kr), "ktheta": float(d_birk_kt),
                    "n_trajectories": int(d_birk_count), "nsteps": int(d_birk_steps),
                    "transient": int(d_birk_transient), "seed": int(d_birk_seed),
                    "p": float(d_birk_p),
                }, "diagnose")

    show_workspace_result("diagnose")

    with st.expander("Ensemble-based ergodic-hierarchy diagnostics", expanded=False):
        st.caption(
            "A separate multi-trajectory workflow for finite-data Birkhoff, "
            "autocorrelation/Cesàro, and finite-partition alpha diagnostics. "
            "These tests complement the single-trajectory tools above; they do "
            "not establish mathematical ergodicity or mixing."
        )
        render_mixing_hierarchy_dashboard()


# ------------------------------------------------------------------
# 2. COMPARE
# ------------------------------------------------------------------

with compare_tab:
    st.subheader("Controlled comparisons")
    st.caption("Vary one map parameter at a time, compare existing trajectories, or examine classical–quantum correspondence.")


    comparison_type = st.selectbox(
        "Comparison type",
        [
            "Compare kθ values at fixed kr",
            "Compare kr values at fixed kθ",
            "Compare two existing trajectories",
            "Compare empirical phase-space measures",
            "Classical–quantum correspondence",
        ],
        key="ws_comparison_type",
    )

    if comparison_type == "Compare kθ values at fixed kr":
        st.markdown(
            r"Compare trajectories at fixed $k_r$, varying $k_\theta$. "
            r"This examines phase-space geometry at a fixed degree of classical chaos."
        )

        with st.form("ws_ktheta_comparison_form"):
            c1, c2 = st.columns(2)

            with c1:
                c_kr = st.number_input(
                    r"Fixed $k_r$", value=1.0, step=0.1,
                    key="ws_cmp_kr",
                )
                c_ka = st.number_input(
                    r"First $k_\theta$", value=0.0, step=100.0,
                    key="ws_cmp_ka",
                )
                c_kb = st.number_input(
                    r"Second $k_\theta$", value=10000.0, step=100.0,
                    key="ws_cmp_kb",
                )

            with c2:
                c_theta = st.number_input(
                    r"Initial $\theta$", value=0.75,
                    key="ws_cmp_theta",
                )
                c_phi = st.number_input(
                    r"Initial $\phi$", value=-1.0,
                    key="ws_cmp_phi",
                )
                c_n = st.number_input(
                    "Number of kicks", min_value=1,
                    max_value=100000, value=1000, step=100,
                    key="ws_cmp_n",
                )

            c_run = st.form_submit_button(
                "Run controlled comparison",
                width="stretch",
            )

        if c_run:
            run_workspace_tool(
                run_dkt_comparison_study,
                {
                    "comparison_mode": "fixed_kr",
                    "kr": float(c_kr),
                    "ktheta_a": float(c_ka),
                    "ktheta_b": float(c_kb),
                    "theta0": float(c_theta),
                    "phi0": float(c_phi),
                    "nsteps": int(c_n),
                    "p": 1.5707963267948966,
                },
                "compare",
            )

    elif comparison_type == "Compare kr values at fixed kθ":
        st.markdown(
            r"Compare trajectories at fixed $k_\theta$, varying $k_r$. "
            r"This allows comparisons across different classical dynamical regimes."
        )

        with st.form("ws_kr_comparison_form"):
            c1, c2 = st.columns(2)

            with c1:
                c_kt_fixed = st.number_input(
                    r"Fixed $k_\theta$", value=0.0, step=100.0,
                    key="ws_cmp_fixed_kt",
                )
                c_kr_a = st.number_input(
                    r"First $k_r$", value=1.0, step=0.1,
                    key="ws_cmp_kr_a",
                )
                c_kr_b = st.number_input(
                    r"Second $k_r$", value=3.0, step=0.1,
                    key="ws_cmp_kr_b",
                )

            with c2:
                c_theta_kr = st.number_input(
                    r"Initial $\theta$", value=0.75,
                    key="ws_cmp_kr_theta",
                )
                c_phi_kr = st.number_input(
                    r"Initial $\phi$", value=-1.0,
                    key="ws_cmp_kr_phi",
                )
                c_n_kr = st.number_input(
                    "Number of kicks", min_value=1,
                    max_value=100000, value=1000, step=100,
                    key="ws_cmp_kr_n",
                )

            c_run_kr = st.form_submit_button(
                "Run controlled comparison",
                width="stretch",
            )

        if c_run_kr:
            run_workspace_tool(
                run_dkt_comparison_study,
                {
                    "comparison_mode": "fixed_ktheta",
                    "kr_a": float(c_kr_a),
                    "kr_b": float(c_kr_b),
                    "ktheta_fixed": float(c_kt_fixed),
                    "theta0": float(c_theta_kr),
                    "phi0": float(c_phi_kr),
                    "nsteps": int(c_n_kr),
                    "p": 1.5707963267948966,
                },
                "compare",
            )


    elif comparison_type == "Compare two existing trajectories":
        with st.form("ws_trajectory_comparison_form"):
            path_a = st.text_input(
                "First trajectory CSV path",
                key="ws_path_a",
            )
            path_b = st.text_input(
                "Second trajectory CSV path",
                key="ws_path_b",
            )
            c_run = st.form_submit_button(
                "Compare trajectories",
                width="stretch",
            )

        if c_run:
            if not path_a.strip() or not path_b.strip():
                st.error("Please provide both trajectory CSV paths.")
            else:
                run_workspace_tool(
                    compare_dkt_simulations,
                    {
                        "trajectory_csv_1": path_a.strip(),
                        "trajectory_csv_2": path_b.strip(),
                    },
                    "compare",
                )

    elif comparison_type == "Compare empirical phase-space measures":
        with st.form("ws_measure_comparison_form"):
            paths = st.text_area(
                "Existing trajectory CSV paths",
                placeholder="Enter 2–10 exact paths, one per line",
                height=120,
                key="ws_measure_paths",
            )
            c_run = st.form_submit_button(
                "Compare empirical measures",
                width="stretch",
            )

        if c_run:
            trajectory_paths = [
                path.strip()
                for path in paths.splitlines()
                if path.strip()
            ]

            if not 2 <= len(trajectory_paths) <= 10:
                st.error("Please provide between 2 and 10 CSV paths.")
            else:
                run_workspace_tool(
                    compare_trajectory_measures,
                    {
                        "trajectory_csvs": trajectory_paths,
                        "n_z": 50,
                        "n_phi": 100,
                    },
                    "compare",
                )

    else:
        with st.form("ws_cq_form"):
            c1, c2 = st.columns(2)
            with c1:
                cq_s = st.number_input(
                    "Spin s", min_value=1.0,
                    max_value=1000.0, value=10.0,
                    step=1.0, key="ws_cq_s",
                )
                cq_kr = st.number_input(
                    "kr", value=1.0, step=0.1,
                    key="ws_cq_kr",
                )
                cq_kt = st.number_input(
                    "ktheta", value=0.0, step=100.0,
                    key="ws_cq_kt",
                )
            with c2:
                cq_theta = st.number_input(
                    "Initial theta", value=0.75,
                    key="ws_cq_theta",
                )
                cq_phi = st.number_input(
                    "Initial phi", value=-1.0,
                    key="ws_cq_phi",
                )
                cq_n = st.number_input(
                    "Number of kicks", min_value=1,
                    max_value=100000, value=1000, step=100,
                    key="ws_cq_n",
                )

            cq_run = st.form_submit_button(
                "Run classical–quantum comparison",
                width="stretch",
            )

        if cq_run:
            run_workspace_tool(
                run_classical_quantum_comparison_tool,
                {
                    "s": int(cq_s),
                    "kr": float(cq_kr),
                    "ktheta": float(cq_kt),
                    "theta0": float(cq_theta),
                    "phi0": float(cq_phi),
                    "nsteps": int(cq_n),
                    "p": 1.5707963267948966,
                },
                "compare",
            )

    show_workspace_result("compare")


# ------------------------------------------------------------------
# 3. RESEARCH DATA
# ------------------------------------------------------------------

with data_tab:
    st.subheader("Research evidence and datasets")
    st.caption("Search existing project outputs and inspect or summarize saved CSV data.")

    data_action = st.radio(
        "Choose an operation",
        [
            "Search research evidence",
            "Inspect an existing CSV",
            "Summarize an existing CSV",
        ],
        key="ws_data_action",
    )

    with st.form("ws_research_data_form"):
        if data_action == "Search research evidence":
            data_query = st.text_area(
                "Evidence search query",
                placeholder=(
                    "For example: kr=1 ktheta=10000 phase-space "
                    "geometry trajectory stretching"
                ),
                height=100,
                key="ws_data_query",
            )
            data_path = ""
            data_limit = st.number_input(
                "Maximum files", min_value=1, max_value=50,
                value=10, key="ws_data_max_files",
            )
        else:
            data_path = st.text_input(
                "Exact relative CSV path under results/",
                placeholder="Use the exact path returned by evidence search",
                key="ws_data_path",
            )
            data_query = ""
            data_limit = st.number_input(
                "Maximum rows", min_value=1, max_value=200000,
                value=200, step=100, key="ws_data_max_rows",
            )

        data_run = st.form_submit_button(
            "Run research-data operation",
            width="stretch",
        )

    if data_run:
        if data_action == "Search research evidence":
            if not data_query.strip():
                st.error("Please enter an evidence search query.")
            else:
                run_workspace_tool(
                    search_research_evidence,
                    {
                        "query": data_query.strip(),
                        "max_files": int(data_limit),
                    },
                    "research_data",
                )

        elif data_action == "Inspect an existing CSV":
            if not data_path.strip():
                st.error("Please provide the exact relative CSV path.")
            else:
                run_workspace_tool(
                    read_research_csv_tool,
                    {
                        "relative_path": data_path.strip(),
                        "max_rows": int(data_limit),
                    },
                    "research_data",
                )

        else:
            if not data_path.strip():
                st.error("Please provide the exact relative CSV path.")
            else:
                run_workspace_tool(
                    summarize_research_csv,
                    {
                        "relative_path": data_path.strip(),
                        "max_rows": int(data_limit),
                    },
                    "research_data",
                )

    show_workspace_result("research_data")


# ------------------------------------------------------------------
# 4. EXPLAIN
# ------------------------------------------------------------------

with explain_tab:
    st.subheader("Explain DKT physics and research results")
    st.write(
        "Ask the research agent to interpret a physical concept, "
        "numerical result, diagnostic or comparison."
    )

    with st.form("ws_explain_form"):
        explanation_question = st.text_area(
            "Your research question",
            placeholder=(
                "For example: At fixed kr=1, how should I interpret "
                "the phase-space stretching observed when ktheta "
                "increases from 0 to 10000?"
            ),
            height=150,
            key="ws_explain_question",
        )
        use_evidence = st.checkbox(
            "Retrieve existing research evidence before answering",
            value=False,
            key="ws_explain_evidence",
        )
        explain_run = st.form_submit_button(
            "Explain",
            width="stretch",
        )

    if explain_run:
        evidence_instruction = (
            "First use search_research_evidence to retrieve relevant "
            "existing results, and inspect relevant CSV records before "
            "making quantitative claims. "
            if use_evidence else
            "Answer conceptually using established physics and clearly "
            "identify claims that would require numerical evidence. "
        )

        prompt = (
            evidence_instruction
            + "Answer this research question:\n"
            + explanation_question
            + "\nDistinguish numerical observations, physical interpretation "
            "and unresolved claims. Preserve the DKT distinction that kr "
            "controls classical chaoticity while ktheta changes phase-space "
            "geometry at fixed kr. Where appropriate, organize the answer "
            "under Evidence, What it establishes, What it does not establish, "
            "Remaining uncertainty and Current research decision."
        )
        run_workspace_request(prompt, "explain")

    show_workspace_result("explain")


with st.expander("Interactive DKT simulation results", expanded=False):
    if st.session_state.get("simulation_error"):
        st.error(
            "DKT simulation failed: "
            + st.session_state["simulation_error"]
        )

    if "simulation_result" in st.session_state:
        result = st.session_state["simulation_result"]

        st.header("DKT Simulation Results")

        metrics = st.columns(3)
        metrics[0].metric(
            "Trajectory points", result["trajectory_rows"]
        )
        metrics[1].metric(
            "Initial norm", f'{result["initial_norm"]:.12f}'
        )
        metrics[2].metric(
            "Maximum norm error", f'{result["max_norm_error"]:.3e}'
        )

        with st.expander("Simulation parameters", expanded=False):
            parameters = result["parameters"]

            parameter_columns = st.columns(4)
            parameter_columns[0].metric(
                r"$k_r$", f'{parameters["kr"]:g}'
            )
            parameter_columns[1].metric(
                r"$k_\theta$", f'{parameters["ktheta"]:g}'
            )
            parameter_columns[2].metric(
                r"$p$", f'{parameters["p"]:.6f}'
            )
            parameter_columns[3].metric(
                "Kicks", f'{parameters["nsteps"]}'
            )

            st.caption(
                rf'$k={parameters["k"]:g},\quad '
                rf"k'={parameters['kprime']:g},\quad "
                rf'\theta_0={parameters["theta0"]:g},\quad '
                rf'\phi_0={parameters["phi0"]:g}$'
            )


        trajectory = pd.DataFrame(result.get("trajectory", []))

        if not trajectory.empty:
            st.subheader("Classical trajectory")
            st.caption("Evolution of the Cartesian components X, Y and Z.")

            plot_data = trajectory.set_index("step")[["X", "Y", "Z"]]
            st.line_chart(plot_data)

            with st.expander("Trajectory data", expanded=False):
                st.dataframe(
                    trajectory.head(20),
                    width="stretch",
                )
                st.caption(
                    f"Showing the first 20 of {len(trajectory)} trajectory rows."
                )

            parameters = result["parameters"]

            trajectory_metadata = {
                "Data type": "Single classical trajectory",
                "kr": parameters["kr"],
                "ktheta": parameters["ktheta"],
                "k": parameters["k"],
                "kprime": parameters["kprime"],
                "p": parameters["p"],
                "theta0": parameters["theta0"],
                "phi0": parameters["phi0"],
                "Total spin": "Not applicable (classical)",
                "Number of kicks": parameters["nsteps"],
                "Saved kicks": f"0 through {parameters['nsteps']}",
                "Transient discarded": "None",
                "Number of data rows": len(trajectory),
            }

            csv_data = csv_with_metadata(trajectory, trajectory_metadata)

            trajectory_filename = (
                f"dkt_trajectory"
                f"_kr_{filename_token(parameters['kr'])}"
                f"_ktheta_{filename_token(parameters['ktheta'])}"
                f"_theta_{filename_token(parameters['theta0'])}"
                f"_phi_{filename_token(parameters['phi0'])}.csv"
            )

            st.download_button(
                "Download trajectory CSV",
                data=csv_data,
                file_name=trajectory_filename,
                mime="text/csv",
                width="stretch",
            )
        else:
            st.warning("The simulation returned no trajectory data.")

        st.caption(
            "Scientific interpretation: kr controls the degree of classical "
            "chaos; ktheta changes phase-space geometry, not the degree of chaos."
        )


with st.expander("θ–φ phase grid results", expanded=False):
    if st.session_state.get("grid_simulation_error"):
        st.error(
            "Phase-space grid simulation failed: "
            + st.session_state["grid_simulation_error"]
        )

    if "grid_simulation_result" in st.session_state:
        grid_result = st.session_state["grid_simulation_result"]
        grid_parameters = grid_result["parameters"]
        grid_data = pd.DataFrame(grid_result["orbit_points"])

        st.header("Final θ–φ phase-space portrait")
        st.success("Phase-space grid simulation completed successfully.")

        grid_metrics = st.columns(3)
        grid_metrics[0].metric(
            "Grid resolution",
            f'{grid_parameters["grid_size"]} × '
            f'{grid_parameters["grid_size"]}',
        )
        grid_metrics[1].metric(
            "Number of kicks", grid_parameters["nsteps"]
        )
        grid_metrics[2].metric(
            "Maximum norm error",
            f'{grid_result["max_norm_error"]:.3e}',
        )

        with st.expander("Grid simulation parameters", expanded=False):
            st.json(grid_parameters)

        if not grid_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.scatter(
                grid_data["phi"],
                grid_data["theta"],
                color="black",
                s=0.5,
                marker=".",
                alpha=1.0,
                linewidths=0,
                rasterized=True,
            )

            ax.set_xlim(-3.141592653589793, 3.141592653589793)
            ax.set_ylim(0.0, 3.141592653589793)
            ax.set_xlabel(r"$\phi$", fontsize=13)
            ax.set_ylabel(r"$\theta$", fontsize=13)
            ax.set_title(r"Final $\theta$–$\phi$ phase-space portrait")
            ax.grid(False)
            fig.tight_layout()

            st.pyplot(fig, width="stretch")
            plt.close(fig)

            st.caption(
                "Each point represents the state of one initial condition "
                "at a given kick. Intermediate orbit points are included."
            )

            grid_parameters = grid_result["parameters"]

            grid_metadata = {
                "Data type": "Full classical phase-space portrait",
                "kr": grid_parameters["kr"],
                "ktheta": grid_parameters["ktheta"],
                "k": grid_parameters["k"],
                "kprime": grid_parameters["kprime"],
                "p": grid_parameters["p"],
                "Total spin": "Not applicable (classical)",
                "Grid resolution": (
                    f"{grid_parameters['grid_size']} x "
                    f"{grid_parameters['grid_size']}"
                ),
                "Initial conditions": grid_parameters["initial_conditions"],
                "Number of kicks": grid_parameters["nsteps"],
                "Saved kicks": f"0 through {grid_parameters['nsteps']}",
                "Transient discarded": "None",
                "Total orbit points": len(grid_data),
            }

            csv_data = csv_with_metadata(grid_data, grid_metadata)

            grid_filename = (
                f"dkt_phase_space"
                f"_kr_{filename_token(grid_parameters['kr'])}"
                f"_ktheta_{filename_token(grid_parameters['ktheta'])}"
                f"_grid_{grid_parameters['grid_size']}"
                f"_T_{grid_parameters['nsteps']}.csv"
            )

            st.download_button(
                "Download phase-space CSV",
                data=csv_data,
                file_name=grid_filename,
                mime="text/csv",
                width="stretch",
            )
        else:
            st.warning("The grid simulation returned no orbit points.")

        st.caption(
            "Scientific interpretation: kr controls the degree of classical "
            "chaos; ktheta changes phase-space geometry, not the degree of chaos."
        )


with st.expander("Entropy dynamics results", expanded=False):
    if st.session_state.get("entanglement_dynamics_error"):
        st.error(
            "Entanglement dynamics failed: "
            + st.session_state["entanglement_dynamics_error"]
        )

    if "entanglement_dynamics_result" in st.session_state:
        er = st.session_state["entanglement_dynamics_result"]
        edf = pd.DataFrame(er.get("data", []))

        st.subheader("Quantum entanglement dynamics")
        st.caption(
            "Normalized von Neumann entropy of the evolving "
            "spin-coherent state, including the initial product state."
        )

        metrics = st.columns(3)
        metrics[0].metric("Time samples", er.get("rows", len(edf)))
        metrics[1].metric(
            "Initial normalized entropy",
            f'{er.get("initial_entropy", float("nan")):.5g}',
        )
        metrics[2].metric(
            "Maximum norm error",
            f'{er.get("max_norm_error", float("nan")):.3e}',
        )

        if not edf.empty and {
            "kick", "normalized_entropy"
        }.issubset(edf.columns):
            st.markdown("**Entropy evolution**")
            st.line_chart(
                edf.set_index("kick")[["normalized_entropy"]],
                height=350,
            )
        else:
            st.warning("Entropy time-series data are unavailable.")

        with st.expander("Calculation parameters", expanded=False):
            parameters = er.get("parameters", {})
            if parameters:
                parameter_table = pd.DataFrame(
                    list(parameters.items()),
                    columns=["Parameter", "Value"],
                )
                st.dataframe(
                    parameter_table,
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("Entropy time-series data", expanded=False):
            st.dataframe(edf, width="stretch")

        if not edf.empty:
            ed_parameters = er["parameters"]

            entropy_dynamics_metadata = {
                "Data type": "Quantum entanglement entropy dynamics",
                "Spin s": ed_parameters["s"],
                "N = 2s": ed_parameters["N"],
                "kr": ed_parameters["kr"],
                "ktheta": ed_parameters["ktheta"],
                "k": ed_parameters["k"],
                "kprime": ed_parameters["kprime"],
                "p": ed_parameters["p"],
                "Initial theta": ed_parameters["theta0"],
                "Initial phi": ed_parameters["phi0"],
                "Subsystem dimension": ed_parameters["rdm_dim"],
                "Number of kicks": ed_parameters["nsteps"],
                "Saved kicks": f"0 through {ed_parameters['nsteps']}",
                "Transient discarded": "None (complete time series)",
                "Number of data rows": len(edf),
            }

            csv_data = csv_with_metadata(edf, entropy_dynamics_metadata)

            entropy_dynamics_filename = (
                f"dkt_entropy_dynamics"
                f"_kr_{filename_token(ed_parameters['kr'])}"
                f"_ktheta_{filename_token(ed_parameters['ktheta'])}"
                f"_theta_{filename_token(ed_parameters['theta0'])}"
                f"_phi_{filename_token(ed_parameters['phi0'])}"
                f"_s_{filename_token(ed_parameters['s'])}"
                f"_Q_{ed_parameters['rdm_dim']}.csv"
            )

            st.download_button(
                "Download entropy dynamics CSV",
                data=csv_data,
                file_name=entropy_dynamics_filename,
                mime="text/csv",
                key="download_entropy_dynamics",
            )

        st.caption(
            "Entanglement entropy characterizes quantum-state entanglement; "
            "it is not itself a measure of classical chaoticity."
        )


with st.expander("Long-time-averaged entropy heatmap results", expanded=False):
    if st.session_state.get("entanglement_heatmap_error"):
        st.error(
            "Entanglement heatmap failed: "
            + st.session_state["entanglement_heatmap_error"]
        )

    if "entanglement_heatmap_result" in st.session_state:
        hr = st.session_state["entanglement_heatmap_result"]
        parameters = hr.get("parameters", {})

        hdf = pd.DataFrame(hr.get("data", []))

        if hdf.empty and hr.get("csv_path"):
            csv_path = Path(hr["csv_path"])
            if csv_path.is_file():
                hdf = pd.read_csv(csv_path)

        st.subheader("Long-time-averaged quantum entanglement")
        st.caption(
            "Normalized von Neumann entropy averaged over the specified "
            "post-transient kick interval for each initial coherent-state "
            "orientation."
        )

        successful = int(hr.get("successful_points", 0))
        failed = int(hr.get("failed_points", 0))
        total = int(hr.get("grid_points", successful + failed))

        good = pd.DataFrame()
        if not hdf.empty and "status" in hdf.columns:
            good = hdf[hdf["status"] == "success"].copy()

        metric_columns = st.columns(4)
        metric_columns[0].metric("Grid points", total)
        metric_columns[1].metric("Successful", successful)
        metric_columns[2].metric("Failed", failed)
        metric_columns[3].metric(
            "Averaging kicks",
            parameters.get("post_transient_samples", "—"),
        )

        if not good.empty and "normalized_average_entropy" in good.columns:
            entropy_values = pd.to_numeric(
                good["normalized_average_entropy"],
                errors="coerce",
            ).dropna()

            if not entropy_values.empty:
                st.markdown("**Entropy statistics across successful grid points**")

                stats = st.columns(3)
                stats[0].metric(
                    "Mean entropy",
                    f"{entropy_values.mean():.5f}",
                )
                stats[1].metric(
                    "Minimum entropy",
                    f"{entropy_values.min():.5f}",
                )
                stats[2].metric(
                    "Maximum entropy",
                    f"{entropy_values.max():.5f}",
                )

        required_columns = {
            "theta", "phi", "normalized_average_entropy"
        }

        if not good.empty and required_columns.issubset(good.columns):
            pivot = good.pivot(
                index="theta",
                columns="phi",
                values="normalized_average_entropy",
            ).sort_index().sort_index(axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))

            mesh = ax.pcolormesh(
                pivot.columns.to_numpy(),
                pivot.index.to_numpy(),
                pivot.to_numpy(),
                shading="auto",
                cmap="jet",
                vmin=0,
                vmax=1,
            )

            colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            colorbar.set_label("Normalized average entropy")

            ax.set_xlabel(r"Initial azimuth $\phi_0$")
            ax.set_ylabel(r"Initial polar angle $\theta_0$")
            ax.set_title(
                "Long-time-averaged normalized entanglement entropy"
            )
            fig.tight_layout()

            st.pyplot(fig, width="stretch")
            plt.close(fig)

        else:
            st.warning(
                "No valid heatmap data are available. "
                "Failed grid points have not been replaced by zero."
            )

        with st.expander("Heatmap calculation parameters", expanded=False):
            if parameters:
                parameter_labels = {
                    "s": "Spin",
                    "N": "Twice the spin",
                    "kr": "Average kicking strength",
                    "ktheta": "Asymmetry parameter",
                    "k": "First kicking strength",
                    "kprime": "Second kicking strength",
                    "p": "Rotation angle",
                    "nsteps": "Final kick",
                    "transient_discarded": "Discarded transient kicks",
                    "averaging_kicks": "Averaging kick interval",
                    "post_transient_samples": "Samples per grid point",
                    "rdm_dim": "Subsystem dimension",
                    "grid_size": "Grid resolution",
                }

                parameter_table = pd.DataFrame([
                    {
                        "Parameter": parameter_labels.get(key, key),
                        "Value": value,
                    }
                    for key, value in parameters.items()
                ])

                st.dataframe(
                    parameter_table,
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("Heatmap data and diagnostics", expanded=False):
            st.markdown("**Calculation status**")

            status = hr.get("status", "unknown")
            if status == "success" and failed == 0:
                st.success("All grid points completed successfully.")
            elif successful > 0:
                st.warning(
                    "The calculation completed with some failed grid points."
                )
            else:
                st.error("No grid points completed successfully.")

            diagnostic_columns = st.columns(3)
            diagnostic_columns[0].metric("Total grid points", total)
            diagnostic_columns[1].metric("Successful points", successful)
            diagnostic_columns[2].metric("Failed points", failed)

            if not hdf.empty:
                st.markdown("**Grid-resolved data**")
                st.dataframe(
                    hdf,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("The grid-resolved CSV data are unavailable.")

            if hr.get("csv_path"):
                st.caption(f"CSV output: {hr['csv_path']}")

        if not hdf.empty:
            heatmap_metadata = {
                "Data type": "Long-time-averaged quantum entropy heatmap",
                "Spin s": parameters["s"],
                "N = 2s": parameters["N"],
                "kr": parameters["kr"],
                "ktheta": parameters["ktheta"],
                "k": parameters["k"],
                "kprime": parameters["kprime"],
                "p": parameters["p"],
                "Grid resolution": (
                    f"{parameters['grid_size']} x {parameters['grid_size']}"
                ),
                "Number of kicks": parameters["nsteps"],
                "Transient discarded": parameters["transient_discarded"],
                "Averaging kick interval": parameters["averaging_kicks"],
                "Post-transient samples per initial condition": (
                    parameters["post_transient_samples"]
                ),
                "Subsystem dimension": parameters["rdm_dim"],
                "Total grid points": hr["grid_points"],
                "Successful grid points": hr["successful_points"],
                "Failed grid points": hr["failed_points"],
            }

            csv_data = csv_with_metadata(hdf, heatmap_metadata)

            heatmap_filename = (
                f"dkt_entropy_heatmap"
                f"_kr_{filename_token(parameters['kr'])}"
                f"_ktheta_{filename_token(parameters['ktheta'])}"
                f"_s_{filename_token(parameters['s'])}"
                f"_Q_{parameters['rdm_dim']}"
                f"_grid_{parameters['grid_size']}"
                f"_T_{parameters['nsteps']}"
                f"_transient_{parameters['transient_discarded']}.csv"
            )

            st.download_button(
                "Download entropy heatmap CSV",
                data=csv_data,
                file_name=heatmap_filename,
                mime="text/csv",
                key="download_entropy_heatmap",
            )

        st.caption(
            "Interpretation: this is a finite-time, post-transient "
            "entropy average over initial conditions. It characterizes "
            "quantum entanglement patterns and does not by itself establish "
            "classical chaos, ergodicity, or mixing. The asymmetry parameter "
            "changes phase-space geometry, not the degree of classical chaos."
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
                        render_tool_result(tool)


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
                            render_tool_result(tool)

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
