## Numerical periodic-orbit search

The general solver searches for primitive cycles of the classical DKT map by solving
\(F^n(\mathbf R)-\mathbf R=0\) with \(\|\mathbf R\|=1\). It excludes roots that
close at a proper divisor of \(n\), deduplicates cyclic rotations, and reports
Floquet multipliers from the \(n\)-step tangent-space monodromy matrix.

Run from the repository root with. On first use, instantiate the small Julia environment (the Python/Streamlit wrapper does this automatically):

julia --project=scripts/julia/periodic_orbit_env -e 'using Pkg; Pkg.instantiate()'

Then run:
julia --project=scripts/julia/periodic_orbit_env scripts/julia/periodic_orbit_search.jl 1.0 0.0 1.5707963267948966 6 256 1e-10 100 20261126 results/periodic_orbits.csv

Arguments: kr, ktheta, p, maximum period, seeds per family, tolerance,
maximum Newton iterations, random seed, output CSV. Two CSVs are written:
one orbit summary and one containing every point in each detected cycle.
Maximum supported period is 20; practical costs can grow rapidly.

Python wrapper: dkt_agent.periodic_orbit_tools.run_periodic_orbit_search.
Streamlit panel: dkt_agent.periodic_orbit_dashboard.render_periodic_orbit_panel.

The search is finite and non-exhaustive. Newton basins, seed density, numerical
tolerance and parameter values affect which roots are detected. A zero count
does not establish nonexistence. Increase seed density, vary seeds, tighten
tolerances and use parameter continuation before making completeness claims.

At fixed k_r, varying k_theta can reveal changes in orbit geometry and
organization. Do not interpret k_theta as controlling classical chaoticity,
Lyapunov behaviour, or the regular-to-chaotic transition.
