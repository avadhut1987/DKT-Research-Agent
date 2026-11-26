"""Streamlit UI panel for numerical primitive periodic-orbit searches."""
import math


def _render_orbit_run(run, suffix):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import streamlit as st

    frame = pd.DataFrame(run["orbits"])
    if frame.empty:
        st.info("No primitive cycles were detected with this search configuration.")
        return
    st.dataframe(frame, use_container_width=True)
    points = pd.read_csv(run["artifacts"]["points_csv"])
    periods = sorted(points["primitive_period"].unique().tolist())
    selected = st.selectbox("Plot primitive period", periods, key="po_plot_period_" + suffix)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 20)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)), alpha=0.08, linewidth=0)
    subset = points[points["primitive_period"] == selected]
    for _, cycle in subset.groupby("orbit"):
        cycle = cycle.sort_values("point_index")
        xyz = cycle[["X", "Y", "Z"]].to_numpy()
        closed = np.vstack([xyz, xyz[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], marker="o", linewidth=1)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=f"Detected primitive period {selected} cycles")
    ax.set_box_aspect((1, 1, 1))
    st.pyplot(fig)
    plt.close(fig)
    for key, label in (("summary_csv", "Download orbit summary"), ("points_csv", "Download orbit points")):
        with open(run["artifacts"][key], "rb") as handle:
            st.download_button(label, handle.read(), file_name=key + ".csv", mime="text/csv",
                                key="po_" + suffix + "_" + key)

def render_periodic_orbit_panel(title="Periodic orbits",*,in_tab=False):
    from contextlib import nullcontext
    import pandas as pd
    import streamlit as st
    from dkt_agent.periodic_orbit_tools import (run_periodic_orbit_search,compare_periodic_orbit_searches,format_periodic_orbit_result)
    panel=nullcontext() if in_tab else st.expander(title,expanded=False)
    with panel:
        st.markdown(r"Searches for primitive cycles satisfying $F^n(\mathbf R)=\mathbf R$ on the unit sphere.")
        st.caption("Finite numerical search: detected orbits only, not a proof of completeness.")
        a,b,c=st.columns(3)
        kr=a.number_input(r"$k_r$",value=1.0,step=0.1,key="po_kr")
        kt=b.number_input(r"$k_\theta$",value=0.0,step=0.5,key="po_kt")
        p=c.number_input("p",value=float(math.pi/2),format="%.8f",key="po_p")
        d,e,f=st.columns(3)
        nmax=d.slider("Maximum period",1,12,6,key="po_nmax")
        seeds=e.select_slider("Seeds per family",[32,64,128,256,512,1024],value=256,key="po_seeds")
        tol=f.selectbox("Residual tolerance",[1e-8,1e-10,1e-12],index=1,key="po_tol")
        it=st.number_input("Maximum iterations",20,500,100,20,key="po_iter")
        seed=st.number_input("Random seed",min_value=0,value=20261126,key="po_seed")
        compare=st.checkbox("Compare ktheta=0 and ktheta=10000 at fixed kr",value=False,key="po_compare")
        if st.button("Search periodic orbits",key="po_run"):
            with st.spinner("Searching periodic points and checking primitive periods..."):
                try:
                    if compare:
                        st.session_state["periodic_orbit_comparison"]=compare_periodic_orbit_searches(
                          kr,(0.0,10000.0),p=p,max_period=nmax,n_seeds=seeds,
                          tolerance=tol,max_iterations=int(it),seed=int(seed))
                        st.session_state.pop("periodic_orbit_result",None)
                    else:
                        st.session_state["periodic_orbit_result"]=run_periodic_orbit_search(
                          kr,kt,p,nmax,seeds,tol,int(it),int(seed))
                        st.session_state.pop("periodic_orbit_comparison",None)
                except Exception as exc: st.error(f"Search failed: {exc}")
        comparison=st.session_state.get("periodic_orbit_comparison")
        if comparison:
            for run in comparison["runs"]:
                st.subheader("k_theta="+str(run["parameters"]["k_theta"]))
                st.markdown(format_periodic_orbit_result(run))
                _render_orbit_run(run, "cmp_" + str(run["parameters"]["k_theta"]))
        result=st.session_state.get("periodic_orbit_result")
        if result:
            st.markdown(format_periodic_orbit_result(result))
            _render_orbit_run(result, "single")
