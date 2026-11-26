"""Streamlit workspace for finite numerical searches of primitive DKT periodic orbits."""

from __future__ import annotations

import math


def _render_orbit_run(run: dict, suffix: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import streamlit as st

    frame = pd.DataFrame(run["orbits"])
    if frame.empty:
        st.info(
            "No primitive cycles were detected with this search configuration. "
            "This finite-seed result does not establish that no such cycles exist."
        )
        return

    st.markdown("**Detected primitive cycles**")
    st.dataframe(frame, width="stretch", hide_index=True)

    points = pd.read_csv(run["artifacts"]["points_csv"])
    periods = sorted(points["primitive_period"].unique().tolist())
    selected = st.selectbox(
        "Primitive period to plot",
        periods,
        key="po_plot_period_" + suffix,
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 20)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        alpha=0.08,
        linewidth=0,
    )

    subset = points[points["primitive_period"] == selected]
    for _, cycle in subset.groupby("orbit"):
        cycle = cycle.sort_values("point_index")
        xyz = cycle[["X", "Y", "Z"]].to_numpy()
        closed = np.vstack([xyz, xyz[0]])
        ax.plot(
            closed[:, 0], closed[:, 1], closed[:, 2],
            marker="o", linewidth=1,
        )

    ax.set(
        xlabel=r"$X$",
        ylabel=r"$Y$",
        zlabel=r"$Z$",
        title=f"Detected primitive period {selected} cycles",
    )
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    c1, c2 = st.columns(2)
    for column, key, label in [
        (c1, "summary_csv", "Download orbit summary (CSV)"),
        (c2, "points_csv", "Download orbit points (CSV)"),
    ]:
        path = run["artifacts"][key]
        with open(path, "rb") as handle:
            column.download_button(
                label,
                handle.read(),
                file_name=key + ".csv",
                mime="text/csv",
                key="po_" + suffix + "_" + key,
                width="stretch",
            )


def render_periodic_orbit_panel(
    title: str = "Periodic orbits",
    *,
    in_tab: bool = False,
) -> None:
    from contextlib import nullcontext

    import pandas as pd
    import streamlit as st

    from dkt_agent.periodic_orbit_tools import (
        compare_periodic_orbit_searches,
        format_periodic_orbit_result,
        run_periodic_orbit_search,
    )

    panel = nullcontext() if in_tab else st.expander(title, expanded=False)
    with panel:
        st.markdown(
            r"Search for primitive cycles satisfying "
            r"$F^n(\mathbf R)=\mathbf R$ on the unit sphere."
        )
        st.caption(
            "The search is numerical and uses a finite set of initial seeds. "
            "Detected cycles are candidates, not an exhaustive catalogue."
        )

        st.markdown("**Map parameters**")
        a, b, c = st.columns(3)
        with a:
            kr = st.number_input(r"$k_r$", value=1.0, step=0.1, key="po_kr")
        with b:
            kt = st.number_input(r"$k_\theta$", value=0.0, step=0.5, key="po_kt")
        with c:
            p = st.number_input(
                r"Rotation angle $p$",
                value=float(math.pi / 2),
                step=0.05,
                key="po_p",
            )

        st.markdown("**Search settings**")
        d, e, f = st.columns(3)
        with d:
            nmax = st.slider(
                "Maximum period", 1, 12, 6, key="po_nmax"
            )
        with e:
            seeds = st.select_slider(
                "Seeds per family",
                [32, 64, 128, 256, 512, 1024],
                value=256,
                key="po_seeds",
            )
        with f:
            tol = st.selectbox(
                "Closure residual tolerance",
                [1e-8, 1e-10, 1e-12],
                index=1,
                key="po_tol",
            )

        d, e = st.columns(2)
        with d:
            it = st.number_input(
                "Maximum Newton iterations",
                20, 500, 100, 20,
                key="po_iter",
            )
        with e:
            seed = st.number_input(
                "Random seed",
                min_value=0,
                value=20261126,
                step=1,
                key="po_seed",
            )

        compare = st.checkbox(
            r"Compare $k_\theta=0$ and $10^4$ at fixed $k_r$",
            value=False,
            key="po_compare",
        )

        if st.button(
            "Search periodic orbits",
            type="primary",
            width="stretch",
            key="po_run",
        ):
            with st.spinner(
                "Searching periodic points and checking primitive periods…"
            ):
                try:
                    if compare:
                        st.session_state["periodic_orbit_comparison"] = (
                            compare_periodic_orbit_searches(
                                kr,
                                (0.0, 10000.0),
                                p=p,
                                max_period=nmax,
                                n_seeds=seeds,
                                tolerance=tol,
                                max_iterations=int(it),
                                seed=int(seed),
                            )
                        )
                        st.session_state.pop("periodic_orbit_result", None)
                    else:
                        st.session_state["periodic_orbit_result"] = (
                            run_periodic_orbit_search(
                                kr, kt, p, nmax, seeds, tol,
                                int(it), int(seed),
                            )
                        )
                        st.session_state.pop("periodic_orbit_comparison", None)
                except Exception as exc:
                    st.error(f"Periodic-orbit search failed: {exc}")

        comparison = st.session_state.get("periodic_orbit_comparison")
        if comparison:
            st.divider()
            st.subheader("Fixed-$k_r$ asymmetry comparison")
            for run in comparison["runs"]:
                parameters = run["parameters"]
                with st.expander(
                    rf"Run with $k_\theta={parameters['k_theta']:g}$",
                    expanded=True,
                ):
                    st.markdown(format_periodic_orbit_result(run))
                    _render_orbit_run(
                        run, "cmp_" + str(parameters["k_theta"])
                    )

        result = st.session_state.get("periodic_orbit_result")
        if result:
            st.divider()
            st.subheader("Search result")
            st.markdown(format_periodic_orbit_result(result))
            _render_orbit_run(result, "single")
