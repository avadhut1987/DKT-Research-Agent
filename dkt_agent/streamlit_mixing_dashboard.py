"""Streamlit visual workspace for finite-data ergodic-hierarchy diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dkt_agent.mixing_tools import diagnose_mixing_hierarchy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = PROJECT_ROOT / "results" / "trajectories"


def _load_result(raw):
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError("The mixing diagnostic returned an unsupported result type.")


def render_mixing_hierarchy_dashboard() -> None:
    """Render the interactive finite-data ergodic-hierarchy dashboard."""
    st.divider()
    st.header("Ergodic Hierarchy — Visual Dashboard")
    st.caption(
        "Finite-data diagnostics for selected-observable Birkhoff averages, "
        "autocorrelation/Cesàro behaviour, and finite-partition alpha coefficients."
    )

    files = sorted(TRAJECTORY_ROOT.glob("*.csv")) if TRAJECTORY_ROOT.is_dir() else []
    if len(files) < 2:
        st.info(
            f"Only {len(files)} trajectory CSV file(s) found. At least two trajectories "
            "from distinct initial conditions under the same DKT map are required to run "
            "the ensemble diagnostics. The four diagnostic sections remain visible below."
        )

    path_by_label = {f"{p.name}": str(p) for p in files}
    with st.expander("Select ensemble and diagnostic settings", expanded=True):
        st.warning(
            "Trajectory CSV coordinates do not prove which map parameters generated them. "
            "Select trajectories from distinct initial conditions that share exactly the "
            "same $k_r$, $k_\\theta$, and $p$. The parameter values below are recorded "
            "as your declared ensemble metadata; they are not inferred from the CSV files."
        )
        with st.form("mixing_hierarchy_dashboard_form"):
            selected_names = st.multiselect(
                "Existing trajectory files (same map, distinct initial conditions)",
                options=list(path_by_label),
                default=[],
                help="Only files in results/trajectories are listed.",
            )
            p1, p2, p3 = st.columns(3)
            with p1:
                kr = st.number_input(r"Common average strength $k_r$", value=1.0, step=0.1, key="mix_dash_kr")
            with p2:
                ktheta = st.number_input(r"Common asymmetry $k_\theta$", value=0.0, step=1.0, key="mix_dash_ktheta")
            with p3:
                kick_angle = st.number_input(r"Common rotation angle $p$", value=1.5707963267948966, step=0.05, format="%.8f", key="mix_dash_p")
            q1, q2, q3 = st.columns(3)
            with q1:
                transient = st.number_input("Transient kicks to discard", min_value=0, value=0, step=10, key="mix_dash_transient")
            with q2:
                max_lag = st.number_input("Maximum autocorrelation lag", min_value=2, max_value=5000, value=500, step=50, key="mix_dash_lag")
            with q3:
                n_z = st.selectbox("Partition bins in z", options=[1, 2, 3, 4], index=1, key="mix_dash_nz")
                n_phi = st.selectbox("Partition bins in azimuth", options=[1, 2, 3, 4, 5, 6, 8], index=2, key="mix_dash_nphi")
            confirmed = st.checkbox(
                "I confirm that all selected trajectories use the same map parameters and distinct initial conditions.",
                value=False,
                key="mix_dash_confirm",
            )
            submitted = st.form_submit_button("Run ergodic-hierarchy diagnostics", type="primary", use_container_width=True)

    if submitted:
        if len(selected_names) < 2:
            st.error("Select at least two trajectory CSV files.")
        elif len(selected_names) > 24:
            st.error("Select no more than 24 trajectory files.")
        elif n_z * n_phi > 16:
            st.error("The exact finite-partition alpha calculation is limited to 16 cells.")
        elif not confirmed:
            st.error("Confirm that the ensemble uses identical map parameters and distinct initial conditions.")
        else:
            paths = [path_by_label[name] for name in selected_names]
            try:
                with st.spinner("Computing finite-data ergodic-hierarchy diagnostics..."):
                    raw = diagnose_mixing_hierarchy.invoke({
                        "trajectory_csvs": paths,
                        "transient": int(transient),
                        "max_lag": int(max_lag),
                        "n_z": int(n_z),
                        "n_phi": int(n_phi),
                    })
                result = _load_result(raw)
                result["dashboard_declared_map_parameters"] = {
                    "kr": float(kr),
                    "ktheta": float(ktheta),
                    "p": float(kick_angle),
                }
                st.session_state["mixing_hierarchy_dashboard_result"] = result
                st.session_state["mixing_hierarchy_dashboard_files"] = selected_names
            except Exception as exc:
                st.error(f"Diagnostic failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("mixing_hierarchy_dashboard_result")
    if not isinstance(result, dict):
        result = {}
        st.caption(
            "No diagnostic result is loaded yet. The four sections below are ready; "
            "select at least two matching trajectory files and run the diagnostics to populate them."
        )

    has_result = bool(result)
    diagnostics = result.get("finite_data_diagnostics", {})
    ergodic = diagnostics.get("ergodicity_related", {})
    weak = diagnostics.get("weak_mixing_related", {})
    strong = diagnostics.get("strong_mixing_related", {})
    parameters = result.get("dashboard_declared_map_parameters", {})
    artifacts = result.get("artifacts", {})

    if has_result:
        st.success("Finite-data diagnostics completed and saved.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trajectories", ergodic.get("n_trajectories", "—"))
        m2.metric("Common samples / trajectory", ergodic.get("common_samples_per_trajectory", "—"))
        m3.metric("Maximum lag used", weak.get("max_lag_used", "—"))
        m4.metric("Partition cells", strong.get("partition", {}).get("n_cells", "—"))
        st.markdown(
            rf"**Declared common map:** $k_r={parameters.get('kr', '—')}$, "
            rf"$k_\theta={parameters.get('ktheta', '—')}$, "
            rf"$p={parameters.get('p', '—')}$"
        )
        st.caption("Ensemble files: " + ", ".join(result.get("input_trajectory_csvs", [])))
    else:
        st.info(
            "Dashboard ready. Choose at least two matching trajectory files above "
            "to calculate and display numerical results in the four tabs."
        )

    tab_ergodic, tab_weak, tab_strong, tab_summary = st.tabs([
        "Birkhoff averages",
        "Weak-mixing indicators",
        "Finite-partition alpha",
        "Numerical tables & files",
    ])

    with tab_ergodic:
        st.subheader("Selected-observable Birkhoff-average variance")
        rows = ergodic.get("rows", [])
        if rows:
            frame = pd.DataFrame(rows)
            pivot = frame.pivot(index="analyzed_steps", columns="observable", values="ensemble_variance")
            st.line_chart(pivot, height=360)
            observable = st.selectbox("Inspect observable", options=list(frame["observable"].unique()), key="mix_dash_erg_obs")
            chosen = frame[frame["observable"] == observable]
            st.dataframe(chosen, use_container_width=True, hide_index=True)
        if not rows:
            st.caption("No Birkhoff-average data yet. Run the diagnostics with at least two trajectory files.")
        st.info(
            "This is a finite-ensemble, selected-observable consistency diagnostic against "
            "uniform-sphere moments. It is not a general von Neumann mean-ergodicity test "
            "and does not prove ergodicity."
        )

    with tab_weak:
        st.subheader("Autocorrelation and Cesàro mean of squared correlations")
        observables = weak.get("observables", {})
        if observables:
            selected_observable = st.selectbox(
                "Cartesian observable",
                options=list(observables),
                key="mix_dash_weak_obs",
            )
            record = observables[selected_observable]
            rho = record.get("normalized_autocorrelation", [])
            cesaro = record.get("cesaro_mean_squared_autocorrelation", [])
            corr_frame = pd.DataFrame({
                "lag": range(len(rho)),
                "normalized autocorrelation": rho,
            }).dropna()
            cesaro_frame = pd.DataFrame({
                "lag": range(1, len(cesaro) + 1),
                "Cesàro mean of squared correlation": cesaro,
            }).dropna()
            st.markdown("**Normalized autocorrelation**")
            st.line_chart(corr_frame.set_index("lag"), height=280)
            st.markdown("**Cesàro diagnostic**")
            st.line_chart(cesaro_frame.set_index("lag"), height=280)
            st.metric("Final finite-window Cesàro value", record.get("final_cesaro_value", "—"))
            st.dataframe(pd.DataFrame({
                "lag": range(len(rho)),
                "normalized_autocorrelation": rho,
                "cesaro_mean_squared_autocorrelation": [None] + list(cesaro),
            }), use_container_width=True, hide_index=True)
        else:
            st.caption("No autocorrelation or Cesàro data yet. Run the diagnostics to populate these plots.")
        st.info(
            "A declining finite-window Cesàro average is consistent with weak-mixing "
            "behaviour for the tested observable; it is not a proof. Only nine observables "
            "and a finite lag range are examined."
        )

    with tab_strong:
        st.subheader("Finite-partition alpha-coefficient estimates")
        alpha_rows = strong.get("lags", [])
        if alpha_rows:
            alpha_frame = pd.DataFrame(alpha_rows)
            st.line_chart(alpha_frame.set_index("lag")[["finite_partition_alpha"]], height=320)
            st.dataframe(alpha_frame, use_container_width=True, hide_index=True)
            st.caption(
                f"Partition: {strong.get('partition', {}).get('type', '—')} · "
                f"{strong.get('partition', {}).get('n_cells', '—')} cells"
            )
        else:
            st.caption("No finite-partition alpha data yet. Run the diagnostics to populate this plot.")
        st.info(
            "These estimates maximize over unions of cells in the selected finite partition "
            "and sampled lags. They do not take the supremum over all measurable sets and "
            "cannot establish strong mixing of the full dynamical system."
        )

    with tab_summary:
        st.subheader("Input and diagnostic summary")
        st.dataframe(pd.DataFrame([
            {"Item": "Trajectory files", "Value": len(result.get("input_trajectory_csvs", []))},
            {"Item": "Transient discarded", "Value": result.get("parameters", {}).get("transient_discarded")},
            {"Item": "Maximum lag requested", "Value": result.get("parameters", {}).get("max_lag")},
            {"Item": "z partition bins", "Value": result.get("parameters", {}).get("n_z")},
            {"Item": "azimuth partition bins", "Value": result.get("parameters", {}).get("n_phi")},
            {"Item": "Diagnostic status", "Value": result.get("status")},
            {"Item": "Hierarchy verdict", "Value": "Not assigned (finite-data diagnostics only)"},
        ]), use_container_width=True, hide_index=True)
        st.markdown("**Scientific limitations**")
        st.write(result.get("scientific_limit", ""))
        st.write(ergodic.get("interpretation", ""))
        st.write(weak.get("interpretation", ""))
        st.write(strong.get("interpretation", ""))
        st.markdown("**Downloadable research artifacts**")
        for key, label, mime in [
            ("summary_json", "Download complete diagnostic JSON", "application/json"),
            ("birkhoff_variance_csv", "Download Birkhoff variance CSV", "text/csv"),
            ("weak_mixing_cesaro_csv", "Download autocorrelation and Cesàro CSV", "text/csv"),
            ("strong_mixing_partition_alpha_csv", "Download finite-partition alpha CSV", "text/csv"),
        ]:
            path = artifacts.get(key)
            if path and Path(path).is_file():
                st.download_button(
                    label,
                    data=Path(path).read_bytes(),
                    file_name=Path(path).name,
                    mime=mime,
                    key=f"mixdash_{key}",
                )
        st.download_button(
            "Download full result JSON",
            data=json.dumps(result, indent=2).encode("utf-8"),
            file_name="mixing_hierarchy_dashboard_result.json",
            mime="application/json",
            key="mixdash_full_result",
        )
