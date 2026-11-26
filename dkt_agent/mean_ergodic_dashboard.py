"""
Streamlit panel for the mean-ergodicity diagnostic.

Usage in app.py (one line, anywhere inside the page body):

    from dkt_agent.mean_ergodic_dashboard import render_mean_ergodic_panel
    render_mean_ergodic_panel()
"""
from __future__ import annotations

import math

from dkt_agent.mean_ergodicity import DEFAULT_BATTERY, mean_ergodic_battery, render
from dkt_agent.mean_ergodicity_tools import CAVEAT, save_mean_ergodic_artifacts


def render_mean_ergodic_panel(title: str = "Mean ergodicity (von Neumann L² diagnostic)", *, in_tab: bool = False):
    from contextlib import nullcontext

    import pandas as pd
    import streamlit as st

    panel = nullcontext() if in_tab else st.expander(title, expanded=False)
    with panel:
        st.markdown(
            r"""Estimates **$D_N = \left\|A_N f - \langle f\rangle\right\|^2$**
            (mean-square distance of the time average from the phase-space average)
            over uniformly sampled initial conditions.

            - $D_N \to 0$: the $L^2$ limit is the constant $\langle f\rangle$.
            - A nonzero plateau: the limit may be a non-constant invariant function.

            Mean ergodicity is *weaker* than Birkhoff (pointwise) ergodicity."""
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown(r"$k_r$")
        kr = c1.number_input("k_r", value=1.0, step=0.05, format="%.4f",
                              key="me_kr", label_visibility="collapsed")
        c2.markdown(r"$k_\theta$")
        kt = c2.number_input("k_theta", value=0.0, step=0.25, format="%.4f",
                              key="me_kt", label_visibility="collapsed")
        p = c3.number_input("p (rotation about y)", value=float(math.pi / 2),
                            format="%.6f", key="me_p")
        obs = st.multiselect("Observables (use several: single observables can be "
                             "blind by symmetry)", list(DEFAULT_BATTERY),
                             default=list(DEFAULT_BATTERY), key="me_obs")
        c4, c5, c6 = st.columns(3)
        N = c4.select_slider("Kicks N", [2 ** n for n in range(10, 18)], value=2 ** 14, key="me_N")
        M = c5.slider("Initial conditions M", 500, 10000, 3000, 500, key="me_M")
        seed = c6.number_input("Seed", value=0, step=1, key="me_seed")
        save = st.checkbox("Save artifacts under results/diagnostics/mean_ergodicity/",
                           value=True, key="me_save")

        if st.button("Run mean-ergodicity diagnostic", key="me_run"):
            if not obs:
                st.warning("Select at least one observable.")
                return
            with st.spinner("Iterating the map…"):
                results, summary = mean_ergodic_battery(
                    kr=kr, ktheta=kt, observables=obs, N=int(N), M=int(M),
                    seed=int(seed), p=p)
            st.session_state["me_results"] = (results, summary, (kr, kt, p, N, M, seed))
            if save:
                saved = save_mean_ergodic_artifacts(
                    float(kr), float(kt), obs, int(N), int(M), float(p), int(seed),
                    results, summary,
                )
                st.success(f"Saved diagnostic artifacts to {saved['artifacts']['directory']}")

        if "me_results" in st.session_state:
            results, summary, pr = st.session_state["me_results"]
            st.markdown(
                rf"Showing $k_r={pr[0]:g}$, $k_\theta={pr[1]:g}$, "
                rf"$p={pr[2]:.4f}$, $N={pr[3]}$, $M={pr[4]}$, seed={pr[5]}"
            )
            rows = [{"observable": o, "D_N/Var(f) at largest N": summary["rel_D"][o],
                     "tail exponent α": r.alpha, "± α": r.alpha_err,
                     "indicator": r.verdict.split()[0]} for o, r in results.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.info(f"Worst observable: **{summary['worst_observable']}** "
                    f"(D_N/Var = {summary['worst_rel_D']:.2e}); "
                    f"all observables decay: **{summary['all_decay']}**")
            sel = st.selectbox("Detailed view for observable", list(results), key="me_sel")
            fig = render(results[sel], return_fig=True)
            st.pyplot(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            r = results[sel]
            df = pd.DataFrame({"N": r.Ns, "D_direct": r.D, "D_standard_error": r.D_se,
                               "D_from_autocorrelation": r.D_corr})
            st.download_button("Download D_N table (CSV)", df.to_csv(index=False),
                               file_name=f"D_N_{sel}.csv", mime="text/csv", key="me_dl")
            st.caption(CAVEAT)
