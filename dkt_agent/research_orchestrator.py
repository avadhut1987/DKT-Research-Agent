"""Plan-first research orchestration for DKT. Planning never executes calculations."""
from __future__ import annotations
import json
import re
from typing import Any

PLANNER_VERSION = "0.1.0"
CAPABILITIES = {
    "search_research_evidence": ("Search existing research evidence", "read_only"),
    "read_research_csv_tool": ("Inspect an existing research CSV", "read_only"),
    "summarize_research_csv": ("Summarize an existing research CSV", "read_only"),
    "run_dkt_simulation": ("Run one classical DKT trajectory", "new_numerical_data"),
    "find_dkt_fixed_points": ("Fixed-point and local-stability search", "new_numerical_data"),
    "diagnose_classical_trajectory": ("Finite-trajectory geometric diagnostics", "existing_data_required"),
    "diagnose_observable_timeseries": ("Finite-record observable time-series diagnostics", "existing_data_required"),
    "run_dkt_parameter_sweep": ("Controlled DKT parameter sweep", "new_numerical_data"),
    "search_dkt_periodic_orbits": ("Primitive periodic-orbit search", "new_numerical_data"),
    "compare_dkt_simulations": ("Compare two existing trajectories kick by kick", "existing_data_required"),
    "run_dkt_comparison_study": ("Controlled two-trajectory comparison", "new_numerical_data"),
    "compare_dkt_periodic_orbits": ("Fixed-k_r periodic-orbit geometry comparison", "new_numerical_data"),
    "mean_ergodicity_langchain_tool": ("von Neumann L² mean-ergodicity diagnostic", "new_numerical_data"),
    "run_ergodicity_variance_study": ("Birkhoff-average ensemble variance", "new_numerical_data"),
    "diagnose_mixing_hierarchy": ("Combined ergodic-hierarchy diagnostics", "existing_data_required"),
    "diagnose_mixing_correlations": ("Finite-record correlation diagnostics", "existing_data_required"),
    "diagnose_lyapunov_ks": ("Finite-time Lyapunov and KS diagnostics", "new_numerical_data"),
    "run_lyapunov_parameter_scan": ("Lyapunov/KS parameter scan", "new_numerical_data"),
    "run_quantum_entanglement_dynamics": ("Evolving-state entanglement dynamics", "new_numerical_data"),
    "run_quantum_entanglement_heatmap": ("Quantum entanglement phase-space heatmap", "new_numerical_data"),
    "run_classical_quantum_comparison_tool": ("Classical–quantum correspondence", "new_numerical_data"),
    "compare_trajectory_measures": ("Empirical phase-space measure comparison", "existing_data_required"),
}
def _has(text: str, *terms: str) -> bool:
    return any(re.search(term, text) for term in terms)
def _step(stage: str, tool: str, reason: str, inputs: list[str], caveat: str) -> dict[str, Any]:
    label, mode = CAPABILITIES[tool]
    return {"stage": stage, "action": label, "tool": tool, "execution_mode": mode,
            "why_this_step": reason, "required_inputs": inputs, "scientific_caveat": caveat}
def build_research_plan(research_question: str, *, existing_data_available: bool | None = None,
                        execution_authorized: bool = False) -> dict[str, Any]:
    """Create a transparent workflow proposal without running any tool."""
    if not isinstance(research_question, str) or not research_question.strip():
        raise ValueError("A non-empty research question is required.")
    if len(research_question) > 6000:
        raise ValueError("Research question must be 6000 characters or fewer.")
    q, t = research_question.strip(), research_question.strip().lower()
    steps, scope, gates = [], [], []
    intent = "evidence_led_diagnostic"
    if _has(t, r"existing", r"previous", r"our results", r"evidence", r"csv", r"data", r"results"):
        steps.append(_step("1 · Evidence review", "search_research_evidence",
            "Locate existing records before repeating calculations or quoting numerical results.",
            ["Focused keywords: parameters, observables, diagnostic and run identifier."],
            "Search results locate evidence; they do not validate a method or prove an asymptotic property."))
    if _has(t, r"k[_\s-]?theta", r"kθ", r"asymmetry", r"phase.?space geometry") and _has(t, r"fixed\s*k[_\s-]?r", r"same\s*k[_\s-]?r", r"k[_\s-]?r\s*=", r"vary", r"compare", r"geometry"):
        intent = "fixed_kr_ktheta_geometry"
        scope = ["Hold k_r, p, initial-condition protocol and numerical settings fixed.",
                 "Vary k_theta only; interpret differences as phase-space/trajectory geometry, not changed chaoticity.",
                 "Do not attribute changes in Lyapunov exponents, KS entropy or the regular-to-chaotic transition to k_theta."]
        if _has(t, r"periodic", r"orbit"):
            steps.append(_step("2 · Orbit-structure comparison", "compare_dkt_periodic_orbits",
                "Compare detected periodic-orbit geometry directly at fixed k_r.",
                ["k_r", "p", "maximum searched period", "seed count and tolerance"],
                "Finite multistart searches are non-exhaustive; non-detection is not non-existence."))
        else:
            steps.append(_step("2 · Controlled trajectory comparison", "run_dkt_comparison_study",
                "A matched comparison isolates the change in k_theta at fixed k_r.",
                ["k_r", "two k_theta values", "theta_0 and phi_0", "number of kicks", "p (default pi/2)"],
                "Trajectory separation alone does not establish changed chaos, Lyapunov behaviour or ergodicity."))
    elif _has(t, r"mean.?ergod", r"von neumann", r"l²", r"l\^2", r"d_n"):
        intent = "mean_ergodicity"
        scope = ["Treat mean ergodicity separately from Birkhoff/pointwise ergodicity and mixing.",
                 "Use multiple observables and report finite-sampling uncertainty."]
        steps.append(_step("2 · Mean-ergodicity test", "mean_ergodicity_langchain_tool",
            "The requested quantity is the finite-sampling von Neumann L² distance D_N.",
            ["k_r, k_theta, p", "observable battery", "maximum N", "initial-condition count M", "random seed"],
            "Finite N and M estimates are indicators, not a proof of an asymptotic L² limit or pointwise ergodicity."))
        if _has(t, r"birkhoff", r"pointwise", r"variance", r"ensemble"):
            steps.append(_step("3 · Time-average follow-up", "run_ergodicity_variance_study",
                "Compare finite-time Birkhoff-average variance across sampled initial conditions.",
                ["Map parameters", "ensemble size", "time-window checkpoints", "seed"],
                "Finite-ensemble variance decay does not establish pointwise ergodicity."))
    elif _has(t, r"weak.?mix", r"strong.?mix", r"mixing", r"ergodic hierarchy", r"hierarchy"):
        intent = "ergodic_hierarchy"
        scope = ["Keep mean ergodicity, Birkhoff behaviour, weak-mixing indicators and strong-mixing indicators distinct.",
                 "Verify that ensemble trajectories share identical k_r, k_theta and p."]
        steps.append(_step("2 · Ensemble hierarchy", "diagnose_mixing_hierarchy",
            "The combined workflow reports complementary finite-data diagnostics without assigning an asymptotic verdict.",
            ["2–24 exact trajectory CSV paths", "distinct initial conditions", "verified common map parameters",
             "transient", "maximum lag", "partition resolution"],
            "Finite Cesàro data do not prove weak mixing; finite-partition alpha is not the supremum over all measurable sets."))
        steps.append(_step("3 · Correlation follow-up", "diagnose_mixing_correlations",
            "Inspect finite-record autocorrelation behaviour for selected observables.",
            ["Exact trajectory CSV path(s)", "transient", "maximum lag"],
            "Finite-lag correlation decay is not proof of mixing; stationarity and sampling matter."))
        if existing_data_available is False:
            gates.append("The hierarchy diagnostic requires an existing same-map ensemble. Authorize generation of a new ensemble if needed.")
    elif _has(t, r"lyapunov", r"ks entropy", r"kolmogorov", r"chaos", r"regular.?to.?chaotic"):
        intent = "chaos_and_lyapunov"
        scope = ["Vary k_r to investigate classical chaoticity; hold k_theta fixed unless geometry is a separate question.",
                 "Use multiple initial conditions or parameter samples before making global claims."]
        steps.append(_step("2 · Finite-time stability", "diagnose_lyapunov_ks",
            "Estimate finite-time Lyapunov exponents and the finite-time Pesin sum for a specified orbit.",
            ["k_r, k_theta, p", "initial condition", "kick count", "transient", "timeout"],
            "A single-orbit finite-time estimate is not a global phase-space classification or proof of KS entropy."))
        if _has(t, r"scan", r"transition", r"threshold", r"parameter", r"critical"):
            steps.append(_step("3 · Parameter dependence", "run_lyapunov_parameter_scan",
                "A controlled scan over k_r is appropriate for studying the regular-to-chaotic transition.",
                ["k_r range and sampling", "fixed k_theta", "p", "initial condition(s)", "kick count"],
                "A finite scan does not locate an exact global transition by itself."))
    elif _has(t, r"entanglement", r"entropy heatmap", r"heatmap"):
        intent = "quantum_entanglement"
        scope = ["Keep evolving-state entanglement distinct from Floquet-eigenstate entanglement.",
                 "Report the transient convention and exact number of averaged samples."]
        tool = "run_quantum_entanglement_heatmap" if _has(t, r"heatmap", r"grid", r"phase.?space") else "run_quantum_entanglement_dynamics"
        steps.append(_step("2 · Entanglement calculation", tool,
            "Use a coherent-state grid for a heatmap request; otherwise use single-state evolution.",
            ["Spin s", "k_r, k_theta, p", "initial angle(s)", "subsystem size", "kick count", "transient where applicable"],
            "Finite-time entropy averages are not asymptotic limits and do not establish classical chaos or mixing."))
    elif _has(t, r"classical.?quantum", r"correspondence", r"quantum expectation", r"spin coherent"):
        intent = "classical_quantum_correspondence"
        scope = ["Match classical and quantum initial orientations, Floquet parameters and kick indices.",
                 "Treat wavepacket spreading and finite-time expectation-value differences explicitly."]
        steps.append(_step("2 · Matched classical–quantum calculation", "run_classical_quantum_comparison_tool",
            "Compare classical X,Y,Z with normalized quantum spin expectations.",
            ["Spin s", "k_r, k_theta, p", "theta_0, phi_0", "number of kicks"],
            "Finite-time mismatch is not itself a measure of classical chaos; entanglement is separate."))
    elif _has(t, r"fixed point", r"fixed-point", r"stability point"):
        intent = "fixed_point_analysis"
        scope = ["Interpret local stability only for the numerically detected points.",
                 "A finite seed search does not establish completeness or global ergodic properties."]
        steps.append(_step("2 · Fixed-point search", "find_dkt_fixed_points",
            "Use the tangent-space numerical search to locate candidate fixed points and assess local multipliers.",
            ["k_r, k_theta, p", "seed count", "tolerance", "iteration limit"],
            "A finite multistart search may miss roots; local stability is not a global classification."))
    elif _has(t, r"periodic orbit", r"periodic-orbit", r"primitive period"):
        intent = "periodic_orbit_search"
        steps.append(_step("2 · Periodic-orbit search", "search_dkt_periodic_orbits",
            "Search numerically for primitive cycles over the requested period range.",
            ["k_r, k_theta, p", "maximum period", "seed count", "tolerance", "random seed"],
            "The finite multistart search is non-exhaustive; failure to detect an orbit does not show it does not exist."))
    elif _has(t, r"correlation dimension", r"box.?count", r"trajectory geometry", r"fractal dimension"):
        intent = "trajectory_geometry"
        steps.append(_step("2 · Trajectory geometry", "diagnose_classical_trajectory",
            "Estimate finite-trajectory geometric scaling from an existing orbit.",
            ["Exact trajectory CSV path", "Theiler window", "pair count", "random seed"],
            "Scaling fits depend on the selected range and finite record; neither estimate proves ergodicity or mixing."))
    elif _has(t, r"time series", r"timeseries", r"birkhoff average", r"observable average", r"block variance"):
        intent = "observable_timeseries"
        steps.append(_step("2 · Observable time-series analysis", "diagnose_observable_timeseries",
            "Calculate finite-record averages, block fluctuations and selected autocorrelations.",
            ["Exact trajectory CSV path", "transient", "maximum lag", "optional block sizes"],
            "Finite-record averages and correlations are not infinite-time Birkhoff limits or proofs of mixing."))
    elif _has(t, r"empirical measure", r"jensen.?shannon", r"phase.?space distribution"):
        intent = "empirical_measure_comparison"
        steps.append(_step("2 · Empirical measure comparison", "compare_trajectory_measures",
            "Compare finite sampled distributions on a common equal-area spherical partition.",
            ["2–10 exact trajectory CSV paths", "matched observation windows", "partition resolution"],
            "A finite-partition divergence does not establish distinct invariant measures or changed chaoticity."))
        gates.append("Confirm that the selected trajectory files use compatible parameters, transients and observation windows.")
    elif _has(t, r"inspect.*csv", r"read.*csv", r"csv.*column", r"csv.*record"):
        intent = "research_csv_inspection"
        steps.append(_step("2 · Read-only CSV inspection", "read_research_csv_tool",
            "Inspect original column names and records from the exact path found in the evidence search.",
            ["Exact relative CSV path returned by evidence search", "maximum rows to inspect"],
            "A partial CSV read must be identified as partial; retrieved records do not validate the underlying method."))
        gates.append("Use the exact path returned by search_research_evidence; do not reconstruct filenames.")
    elif _has(t, r"summarize.*csv", r"csv.*summary", r"csv.*statistics"):
        intent = "research_csv_summary"
        steps.append(_step("2 · Read-only CSV summary", "summarize_research_csv",
            "Summarize schema, missingness and descriptive statistics from an existing results CSV.",
            ["Exact path relative to results/", "maximum rows to inspect"],
            "Descriptive statistics do not validate the scientific method or establish dynamical properties."))
    elif _has(t, r"compare.*existing trajectories", r"compare.*trajectory csv", r"two existing trajectories"):
        intent = "trajectory_comparison"
        steps.append(_step("2 · Existing trajectory comparison", "compare_dkt_simulations",
            "Compare two existing trajectories kick by kick using their exact recorded paths.",
            ["Two exact trajectory CSV paths", "matching step indices"],
            "Trajectory separation alone does not establish chaos, ergodicity or mixing."))
    elif _has(t, r"simulate.*trajectory", r"run.*simulation", r"generate.*trajectory"):
        intent = "single_trajectory_simulation"
        steps.append(_step("2 · Classical trajectory simulation", "run_dkt_simulation",
            "Generate a single classical DKT trajectory with explicitly supplied parameters.",
            ["k_r, k_theta, p", "theta_0, phi_0", "number of kicks"],
            "A trajectory by itself does not establish chaos, ergodicity or mixing."))
    elif _has(t, r"parameter sweep", r"sweep.*parameter", r"scan.*phase space"):
        intent = "parameter_sweep"
        steps.append(_step("2 · Controlled parameter sweep", "run_dkt_parameter_sweep",
            "Compare finite empirical phase-space measures while varying one parameter at a time.",
            ["Sweep k_r or k_theta", "2–8 parameter values", "fixed remaining parameters", "initial-condition protocol", "kick count"],
            "A finite sweep compares sampled measures; it does not prove invariant-measure or hierarchy claims."))
    else:
        scope = ["Start with the narrowest relevant existing evidence.",
                 "Choose diagnostics only after the target claim, observable and parameter regime are clear."]
        if not steps:
            steps.append(_step("1 · Evidence discovery", "search_research_evidence",
                "Establish what numerical evidence already exists before designing new work.",
                ["Focused query with parameters, observables and diagnostic names"],
                "Retrieved evidence may be incomplete or partially inspected."))
        gates.extend(["Which scientific claim or observable should be prioritized?",
                      "Should the investigation use existing data only, or may it generate new numerical data?"])
    new_work = any(s["execution_mode"] == "new_numerical_data" for s in steps)
    authorization = "authorized_by_user" if execution_authorized and new_work else "not_authorized"
    if new_work and not execution_authorized:
        gates.append("New numerical execution is proposed but not authorized. Confirm before running calculations that generate data.")
    if any(s["execution_mode"] == "existing_data_required" for s in steps) and existing_data_available is not True:
        gates.append("Provide or locate exact trajectory CSV paths and verify identical map parameters before ensemble analysis.")
    return {
        "planner_version": PLANNER_VERSION, "status": "plan_created",
        "research_question": q, "research_intent": intent,
        "scope_and_controls": scope, "workflow": steps,
        "execution_authorization": authorization, "execution_performed": False,
        "clarifications_or_gates": list(dict.fromkeys(gates)),
        "scientific_guardrails": [
            "A recommended tool is not evidence that it has run.",
            "Finite-data diagnostics do not prove asymptotic ergodicity, weak mixing or strong mixing.",
            "k_r controls classical chaoticity; k_theta changes phase-space geometry, not chaoticity.",
            "Report observations, interpretation, limitations, unresolved claims and decisions separately."
        ],
        "available_capabilities": {k: {"label": v[0], "execution_mode": v[1]} for k, v in CAPABILITIES.items()}
    }
try:
    from langchain_core.tools import tool
    @tool
    def plan_dkt_research(research_question: str, existing_data_available: bool | None = None,
                          execution_authorized: bool = False) -> str:
        """Create a structured DKT research plan without executing calculations.
        Set execution_authorized=True only when the user explicitly authorized new numerical execution."""
        return json.dumps(build_research_plan(research_question,
            existing_data_available=existing_data_available,
            execution_authorized=execution_authorized), indent=2, ensure_ascii=False)
except ImportError:
    plan_dkt_research = None
