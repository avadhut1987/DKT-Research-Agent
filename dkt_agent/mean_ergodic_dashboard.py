"""Streamlit panel for the finite-sampling von Neumann mean-ergodicity diagnostic."""

from __future__ import annotations

import math
from contextlib import nullcontext

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dkt_agent.mean_ergodicity import DEFAULT_BATTERY, mean_ergodic_battery, render
from dkt_agent.mean_ergodicity_tools import CAVEAT, save_mean_ergodic_artifacts


def render_mean_ergodic_panel(
    title: str = "Mean ergodicity (von Neumann L2 diagnostic)",
    *,
    in_tab: bool = False,
) -> None:
    """Render the independent L² mean-ergodicity sampling workflow."""
    panel = nullcontext() if in_tab else st.expander(title, expanded=False)

    with panel:
        st.markdown(
            r"Estimates **$D_N = \left\|A_N f - \langle f\rangle\right\|^2$**, "
            "the mean-square distance between a finite-time average and the "
            "uniform-sphere average, over sampled initial conditions."
        )
        st.markdown(
            r"- $D_N \to 0$: consistent with a constant $L^2$ limit "
            r"$\langle f\rangle$ for the tested observable."
            "\n- A nonzero plateau may indicate a non-constant invariant component "
            "or finite-data effects."
        )
        st.caption(
            "This is a finite-sampling diagnostic, not a proof of ergodicity. "
            "Mean ergodicity is weaker than Birkhoff (pointwise) ergodicity."
        )

        st.markdown("**Map parameters**")
        c1, c2, c3 = st.columns(3)
        with c1:
            kr = st.number_input(
                r"$k_r$", value=1.0, step=0.05, format="%.4f", key="me_kr"
            )
        with c2:
            kt = st.number_input(
                r"$k_\theta$", value=10000.0, step=0.25,
                format="%.4f", key="me_kt"
            )
        with c3:
            p = st.number_input(
                r"Rotation angle $p$", value=float(math.pi / 2),
                format="%.6f", key="me_p"
            )

        obs = st.multiselect(
            "Observables",
            list(DEFAULT_BATTERY),
            default=list(DEFAULT_BATTERY),
            key="me_obs",
            help="Use several observables: a single observable can be insensitive to some invariant structure.",
        )

        st.markdown("**Sampling and reproducibility**")
        c1, c2, c3 = st.columns(3)
        with c1:
            N = st.select_slider(
                "Number of kicks $N$",
                [2 ** n for n in range(10, 18)],
                value=2 ** 14,
                key="me_N",
            )
        with c2:
            M = st.slider(
                "Initial conditions $M$", 500, 10000, 3000, 500,
                key="me_M"
            )
        with c3:
            seed = st.number_input(
                "Random seed", value=0, step=1, key="me_seed"
            )

        save = st.checkbox(
            "Save reproducibility artifacts under results/diagnostics/mean_ergodicity/",
            value=True,
            key="me_save",
        )

        if st.button(
            "Run mean-ergodicity diagnostic",
            type="primary",
            width="stretch",
            key="me_run",
        ):
            if not obs:
                st.warning("Select at least one observable.")
            else:
                with st.spinner("Iterating the map and estimating the L² diagnostic…"):
                    results, summary = mean_ergodic_battery(
                        kr=kr,
                        ktheta=kt,
                        observables=obs,
                        N=int(N),
                        M=int(M),
                        seed=int(seed),
                        p=p,
                    )
                st.session_state["me_results"] = (
                    results, summary, (kr, kt, p, N, M, seed)
                )
                if save:
                    saved = save_mean_ergodic_artifacts(
                        float(kr), float(kt), obs, int(N), int(M),
                        float(p), int(seed), results, summary,
                    )
                    st.success(
                        "Saved diagnostic artifacts to "
                        f"{saved['artifacts']['directory']}"
                    )

        if "me_results" not in st.session_state:
            st.info("Run the diagnostic to display the observable-wise decay estimates.")
            return

        results, summary, parameters = st.session_state["me_results"]
        st.divider()
        st.markdown(
            rf"**Run:** $k_r={parameters[0]:g}$, "
            rf"$k_\theta={parameters[1]:g}$, $p={parameters[2]:.4f}$, "
            rf"$N={parameters[3]}$, $M={parameters[4]}$, seed={parameters[5]}"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Worst tested observable", summary["worst_observable"])
        m2.metric("Largest relative $D_N$", f"{summary['worst_rel_D']:.2e}")
        m3.metric("All tested observables decay", "Yes" if summary["all_decay"] else "No")

        rows = [
            {
                "Observable": observable,
                r"Relative $D_N$ at largest $N$": summary["rel_D"][observable],
                r"Tail exponent $\alpha$": result.alpha,
                r"Uncertainty in $\alpha$": result.alpha_err,
                "Finite-data indicator": result.verdict.split()[0],
            }
            for observable, result in results.items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        selected = st.selectbox(
            "Detailed decay curve",
            list(results),
            key="me_sel",
        )
        fig = render(results[selected], return_fig=True)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        selected_result = results[selected]
        frame = pd.DataFrame({
            "N": selected_result.Ns,
            "D_direct": selected_result.D,
            "D_standard_error": selected_result.D_se,
            "D_from_autocorrelation": selected_result.D_corr,
        })
        st.download_button(
            "Download selected-observable $D_N$ data (CSV)",
            frame.to_csv(index=False).encode("utf-8"),
            file_name=f"D_N_{selected}.csv",
            mime="text/csv",
            key="me_dl",
            width="stretch",
        )
        st.caption(CAVEAT)
