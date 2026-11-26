# DKT Research Agent: handoff context

## Convention
k = k_r + k_theta, k' = k_r - k_theta (Purohit & Bhosale, PRE 112, 014217, Eq. 5).
Classical map = paper Eqs. (2)-(4); verified identical to dkt_step in dkt_agent/mean_ergodicity.py.
Parameter p = rotation angle about y (default pi/2).

## Done
- Added dkt_agent/mean_ergodicity.py: von Neumann L2 mean-ergodicity, D_N = ||A_N f - <f>||^2,
  Monte-Carlo over uniform initial conditions with standard errors, autocorrelation cross-check,
  tail-exponent fit, nine-observable battery, 4-panel rendering, parameter scan.
- dkt_agent/mean_ergodicity_tools.py (run_mean_ergodic_diagnostic + LangChain tool),
  dkt_agent/mean_ergodic_dashboard.py (Streamlit panel), tests/python/test_mean_ergodicity.py.
  12 tests pass locally.

## Findings (finite-data indicators, not proofs)
- Single observables can be blind by symmetry: at k_r=0.75, k_theta=0, f=z looks ergodic
  (D_N ~ 1e-9) but f=y plateaus (~0.33). Always use the battery.
- At k_r=1, k_theta=1e4: x, y, z decay, but x^2, y^2, z^2 plateau (D_N/Var = 0.147, 0.037, 0.037).
  Round-off ruled out (float64 vs longdouble, several seeds). Verify with the repo's own tools.

## Caveats
Existing tools analyse saved trajectory CSVs from grid-sampled ICs; the new module generates a
uniformly sampled ensemble (needed for its error bars). No CSV-input version yet.

## Integration status
- Registered mean_ergodicity_langchain_tool in dkt_agent/agent.py.
- Added render_mean_ergodic_panel() to app.py as a separate Streamlit expander.
- The dashboard saves artifacts from its in-memory results instead of repeating the full calculation.
- The correlation plot now uses the Cesàro mean of squared normalized correlations as the finite-data weak-mixing indicator.
- The existing mixing-hierarchy dashboard retains its separate finite-partition alpha-coefficient diagnostic for strong mixing.

## Next tasks (in order)
1. Periodic-orbit finder: Newton on T^n(x)=x on the sphere, with stability from the Jacobian.
2. Husimi-scar module: Husimi Q of Floquet eigenstates, symmetry-resolved first; score the
   overlap with classical period-n orbits (scar measure, coherent-state overlap).
3. Optional: mean-ergodicity from saved trajectory CSVs.
Scientific wording: report finite-data indicators, never proofs.
