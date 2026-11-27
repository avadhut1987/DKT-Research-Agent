# DKT Research Agent

**An evidence-led computational research environment for the classical and quantum dynamics of the Double Kicked Top.**

The **DKT Research Agent** integrates Python analysis tools, Julia-based numerical computation, a LangGraph research assistant, and a Streamlit interface. It supports controlled computational studies of nonlinear dynamics, phase-space geometry, the ergodic hierarchy, quantum entanglement, and classical–quantum correspondence.

The platform is intended as a research instrument. Numerical calculations are performed by explicit computational tools; research workflows preserve inputs and provenance; and scientific conclusions must remain within the scope of the data and methods actually used. The language model assists with planning, tool selection, evidence discovery, and interpretation. It is not a substitute for numerical validation or mathematical proof.

---

## Contents

- [Research scope](#research-scope)
- [The Double Kicked Top model](#the-double-kicked-top-model)
- [Scientific interpretation and guardrails](#scientific-interpretation-and-guardrails)
- [Computational capabilities](#computational-capabilities)
- [Research Studio and reproducibility](#research-studio-and-reproducibility)
- [Application interface](#application-interface)
- [Technology stack](#technology-stack)
- [Installation](#installation)
- [Running the application and agent](#running-the-application-and-agent)
- [Julia workflows](#julia-workflows)
- [Testing](#testing)
- [Data and provenance](#data-and-provenance)
- [Recommended research practice](#recommended-research-practice)
- [Scientific limitations](#scientific-limitations)
- [Repository organization](#repository-organization)

---

## Research scope

The platform brings together six related but distinct areas:

1. **Classical nonlinear dynamics:** trajectories, phase-space portraits, fixed points, periodic orbits, stability, Lyapunov exponents, and KS-entropy estimates.
2. **Ergodic hierarchy:** finite-data investigations of mean ergodicity, Birkhoff-average behaviour, correlation decay, weak-mixing-related indicators, and finite-partition strong-mixing indicators.
3. **Phase-space geometry:** controlled comparisons of trajectories and matched local clouds under different DKT maps.
4. **Quantum dynamics:** evolving-state entanglement entropy and long-time-averaged entropy maps.
5. **Classical–quantum correspondence:** comparisons between classical spin coordinates and normalized quantum spin expectation values.
6. **Research orchestration:** evidence discovery, workflow planning, explicit execution authorization, persistent campaign records, validation, and scientific review.

The scientific question should determine which diagnostic is appropriate. The availability of a computational tool should not determine the research question.

## The Double Kicked Top model

### Quantum Floquet operator

The quantum DKT is represented by the Floquet operator

$$
U =
\exp\left(-i\frac{k'}{2j}J_x^2\right)
\exp\left(-i\frac{k}{2j}J_z^2\right)
\exp(-ipJ_y),
$$

where $J_x$, $J_y$, and $J_z$ are angular-momentum operators in the spin-$j$ representation, $p$ is the rotation parameter, and $k$ and $k'$ are the nonlinear kicking strengths. The Hilbert-space dimension is $D=2j+1$.

The average and asymmetry parameters are

$$
k_r = \frac{k+k'}{2},
\qquad
k_\theta = \frac{k-k'}{2},
$$

so that

$$
k = k_r+k_\theta,
\qquad
k' = k_r-k_\theta.
$$

Here $k_r$ is the average kicking strength and $k_\theta$ is the asymmetry between the two kicks.

### Classical phase space

The classical dynamics is represented on the unit sphere, with normalized spin coordinates satisfying

$$
X^2+Y^2+Z^2=1.
$$

Angular coordinates $(\theta,\phi)$ specify initial conditions on the sphere. Classical tools evolve these initial conditions under the DKT map and can save kick-resolved coordinates together with run metadata. For reproducibility, retain the map parameters, initial conditions, number of kicks, and numerical settings recorded with each run.

## Scientific interpretation and guardrails

The distinction between $k_r$ and $k_\theta$ is essential.

- **The average kicking strength $k_r$ controls the degree of classical chaos.** The regular-to-chaotic transition, Lyapunov behaviour, and KS-entropy behaviour are associated with changes in $k_r$.
- **The asymmetry parameter $k_\theta$ changes phase-space geometry.** At fixed $k_r$, it can change the orientation, structure, stretching, and rotation of phase-space features and can break time-reversal symmetry.
- A change in $k_\theta$ at fixed $k_r$ must not be described as an increase or decrease in classical chaoticity, a change in Lyapunov behaviour, or a shift of the regular-to-chaotic transition.
- Separation between two trajectories is a property of that matched pair over a finite observation time. It is not, by itself, a Lyapunov-exponent estimate.
- A local initial-condition cloud is a finite-time geometric probe. It is not automatically an invariant region, an invariant measure, or a test of local ergodicity.

A controlled geometry comparison holds $k_r$, $p$, the initial-condition protocol, and the number of kicks fixed while varying $k_\theta$. For example, comparing $k_\theta=0$ and $k_\theta=10^4$ at the same $k_r$ isolates effects associated with the asymmetry parameter.

### Ergodic hierarchy: distinct questions, distinct diagnostics

| Concept | Mathematical question | Numerical support in this project |
|---|---|---|
| Mean ergodicity | Do time-averaged observables approach their phase-space averages in the relevant Hilbert-space sense? | Finite-data von Neumann / $L^2$ mean-ergodicity indicators |
| Birkhoff or pointwise ergodicity | Do time averages along almost every trajectory approach the appropriate space average? | Finite-time trajectory averages and ensemble variance |
| Weak mixing | Do correlations of suitable observables vanish in the required long-time averaged sense? | Finite-record autocorrelations and Cesàro averages of squared correlations |
| Strong mixing | Do correlations between suitable measurable events become asymptotically independent under time separation? | Finite-partition alpha-coefficient estimates for sampled lags |

These diagnostics are complementary, not interchangeable:

- Mean ergodicity is not the same as pointwise ergodicity.
- A finite-time Birkhoff-average variance trend is not a proof of almost-everywhere convergence.
- A finite-window correlation indicator is not a proof of weak mixing.
- A strong-mixing alpha estimate on one finite partition and a finite set of lags is not the supremum over all measurable sets in the mathematical definition.
- A result for selected observables does not automatically establish a property for every observable.
- Observable-dependent behaviour under full-sphere sampling does not establish ergodicity restricted to a spatial subregion.

The software therefore reports finite-data diagnostics and their scope rather than assigning an unconditional mathematical label to the dynamics.

## Computational capabilities

### 1. Classical DKT dynamics

The classical toolset supports:

- Single-trajectory simulation and kick-resolved data export.
- Grid-based trajectory generation and phase-space visualization.
- Comparisons between selected DKT parameter configurations.
- Fixed-point searches and local-stability analysis.
- Primitive periodic-orbit searches with closure and stability diagnostics.
- Parameter sweeps and selected trajectory/measure comparisons.
- Finite-time Lyapunov-exponent and KS-entropy estimates.

Fixed-point and periodic-orbit searches are numerical searches over finite initial seed sets. Failure to find an orbit in a finite search is not proof that no such orbit exists.

### 2. Mean ergodicity and Birkhoff averages

The mean-ergodicity workflow evaluates a finite-data von Neumann / $L^2$ indicator for selected observables over sampled initial conditions. The available observable set includes

$$
X, Y, Z, X^2, Y^2, Z^2, XY, YZ, ZX.
$$

The complementary Birkhoff-variance workflow calculates finite-time trajectory averages and examines their variance across an ensemble of distinct initial conditions.

These workflows allow observable-resolved comparisons. For example, outputs can be examined separately for $X$, $Y$, and $Z$, rather than compressed into one global verdict. Results remain conditional on the chosen observables, sampling protocol, number of initial conditions, transient, and observation window.

### 3. Correlation and mixing-hierarchy diagnostics

The correlation tools calculate finite-record observable correlations and related time-averaged indicators. The combined mixing-hierarchy diagnostic includes:

- **Weak-mixing-related indicator:** finite-window Cesàro averages of squared autocorrelations for the implemented Cartesian observables.
- **Strong-mixing-related indicator:** finite-partition alpha-coefficient estimates over selected time lags, using sampled trajectory ensembles and time origins.

The strong-mixing estimate depends on the chosen spherical partition, lag range, trajectory ensemble, and sampling procedure. It is not a computation of the supremum over all measurable sets.

### 4. Direction-dependent and spatially local questions

The research planner can formulate investigations in which numerical behaviour differs across observables or Cartesian directions. Such questions should be evaluated observable by observable using the implemented mean-ergodicity and Birkhoff-variance diagnostics. Mixing diagnostics can be added when requested and when suitable same-map trajectory data are available.

**Region-conditioned direction-dependent ergodicity is now available as a finite-data investigation.** The tool `diagnose_direction_dependent_local_ergodicity` samples a specified region with respect to spherical area and evaluates $X$, $Y$, $Z$, their squares, and cross-observables separately. Supported region geometries are the full sphere, a spherical cap, a polar-angle band, and a longitude sector.

The analysis first screens whether the proposed region is dynamically retained:
- Forward one-step escape from the region.
- Backward one-step escape, using the inverse DKT map.
- Multi-kick retention of the sampled initial-condition ensemble.

Each escape rate is accompanied by a 95% Wilson upper confidence bound. The user-selected tolerance is a finite-sample screening threshold, not a mathematical invariance tolerance. A region that fails the screen does **not** receive a local invariant-measure interpretation; its Birkhoff statistics are reported only as descriptive averages from initial conditions selected inside that geometric region.

If the screen passes, the tool estimates the conditional spherical-area mean and variance from an independent reference sample, then computes observable-resolved finite-time Birkhoff averages and the normalized spread

$$
\frac{D_N^{(R)}(f)}{\operatorname{Var}_R(f)}
=
\frac{
\int_R |A_N f(x)-\langle f\rangle_R|^2\,d\mu_R(x)
}{
\operatorname{Var}_R(f)
},
\qquad
A_N f(x)=\frac{1}{N}\sum_{n=0}^{N-1}f(F^n x).
$$
This conditional $L^2$ interpretation requires $R$ to be invariant under the map so that the normalized restricted area measure $\mu_R$ is invariant. The finite-sample screen can identify only a **candidate region**; it cannot prove invariance. Even for a candidate region, finite-$N$ decay or a plateau is evidence for the tested observable and sampling range, not a proof of mean or pointwise ergodicity. Direction-dependent differences across $X$, $Y$, $Z$ and higher-order observables are therefore reported as observable-resolved finite-data behaviour, not as an unconditional claim that the map is ergodic in some directions and non-ergodic in others.

The Research Studio exposes region geometry, map parameters, observables, sample sizes and leakage tolerance in the same supervised execution workflow used by the other research tools. The LangGraph assistant can also invoke the diagnostic directly. This workflow is distinct from matched local-cloud geometry comparisons: a cloud's deformation is not a local-ergodicity test.

### 5. Matched phase-space geometry comparisons

Two complementary comparison workflows are available:

- **Controlled two-trajectory comparison:** evolves the same initial condition under two specified maps and records the kick-by-kick separation.
- **Matched local-cloud comparison:** constructs the same small angular grid of initial conditions around a specified centre and evolves that grid under both maps.

The local-cloud workflow records finite-time geometric quantities such as covariance eigenvalues, principal axes, cloud radii, anisotropy, axis alignment, and matched-point separations. Principal-axis alignment accounts for the sign ambiguity of eigenvectors.

These measurements characterize the deformation and orientation of the selected finite-time cloud. They do not establish ergodicity, mixing, invariant-region structure, or a change in classical chaoticity.

### 6. Quantum entanglement dynamics

The Julia-backed quantum workflows provide:

- Kick-resolved entanglement entropy for evolving quantum states.
- Long-time-averaged entanglement entropy heatmaps over grids of spin-coherent initial states.
- Transient-discarded averaging.
- State-norm and calculation-status diagnostics.
- CSV outputs and visualizations.

For a bipartition with subsystem size $Q$ and total spin parameter $N=2j$, the normalized subsystem entropy is

$$
s_A =
\frac{S_A}
{\log_2\left[\min(Q+1,N-Q+1)\right]}.
$$

Evolving-state entanglement calculations are distinct from Floquet-eigenstate entanglement analysis. For the heatmap workflow, the transient convention is to discard kicks $0:T_0$ and average kicks $T_0+1:T$. Failed calculations remain missing values; they are not interpreted as zero entropy.

### 7. Classical–quantum correspondence

The classical–quantum comparison workflow evaluates the correspondence between a classical DKT trajectory and normalized quantum spin expectation values:

$$
\frac{\langle J_x\rangle}{j},
\qquad
\frac{\langle J_y\rangle}{j},
\qquad
\frac{\langle J_z\rangle}{j}.
$$

The output includes kick-resolved components and numerical error summaries such as mean absolute error, root mean square error, maximum component-wise error, and vector differences. The initial normalized quantum spin vector is checked against the specified classical initial point.

Finite-time disagreement between a single classical trajectory and quantum expectation values is not, by itself, a diagnostic of classical chaos. Entanglement entropy is treated as a separate observable.

### 8. Scholarly literature and symbolic mathematics

The research assistant and Research Director include read-only scholarly discovery through **arXiv** and **Crossref**. Search results provide bibliographic metadata, abstracts when available, DOI/arXiv identifiers, and source links. Available arXiv HTML full text can be inspected in bounded chunks. Search ranking is only a discovery aid: it does not establish correctness, relevance, or peer-review status. The agent must distinguish preprints from published work and verify theorem statements and technical equations against primary sources.

A bounded **SymPy** tool supports:
- Algebraic simplification.
- Symbolic differentiation.
- Jacobian construction for lists of expressions.

The symbolic parser accepts a restricted expression grammar rather than arbitrary Python. Variables are treated as real unless a later tool explicitly supports stronger assumptions. These calculations can check algebra and support derivations, but they are not a theorem prover and do not establish dynamical-systems results without a complete mathematical argument.

### Theory-to-tool development

The Research Director can return a structured, design-only research-tool proposal when a genuine scientific need is not covered by the registered tools. Such a proposal includes the scientific gap, mathematical definition, algorithm, validation plan, expected outputs, and limitations. The agent can identify a mathematical question, retrieve relevant literature, formulate assumptions, derive candidate expressions, and propose an algorithm or tool specification. It does **not** autonomously rewrite its own source code, install arbitrary packages, or execute unreviewed code. A new research tool follows a controlled development cycle:

1. State the mathematical object, assumptions, intended output, and scientific use.
2. Derive the algorithm and identify edge cases and failure modes.
3. Implement it as a separate, bounded module.
4. Add analytic test cases, numerical cross-checks, and regression tests.
5. Review the implementation and scientific interpretation before merging it into the research agent.

This separation prevents a language-model-generated method from being treated as validated merely because the model proposed or executed it.

## Research Studio and reproducibility

The **Research Studio** is a plan-first, human-supervised workflow for structured computational investigations. It makes the path from a scientific question to a numerical result explicit and reviewable.

### Campaign lifecycle

A typical investigation follows these stages:

1. **Frame the question.** Define the scientific claim, parameters, observables, and scope.
2. **Discover existing evidence.** Search existing records before repeating calculations. Evidence search is read-only; keyword matches are leads for inspection, not confirmation of a claim.
3. **Plan the workflow.** The deterministic research planner maps the question to supported tools, required inputs, execution modes, and scientific caveats. Planning itself does not run calculations.
4. **Review the proposed stages.** Decide which stages are already covered by existing evidence and provide a provenance note for reused results.
5. **Authorize new numerical work.** Numerical execution requires explicit campaign-level authorization.
6. **Execute stages explicitly.** Each numerical stage is started through a separate execution action. Stages run in the planned order.
7. **Inspect and validate results.** The campaign record preserves inputs, outputs, attempts, errors, timestamps, and artifact references. Record-integrity validation checks that the saved record is internally consistent; it is not a proof of scientific correctness.
8. **Record scientific review.** A researcher records an outcome and rationale, distinguishing accepted results from work requiring follow-up or inconclusive outcomes.

### Persistent campaign records

Research campaigns are saved under the results/research_campaigns directory. A campaign records, as applicable:

- The original research question and plan snapshot.
- Ordered workflow stages and execution modes.
- Exact submitted inputs and parameter values.
- Authorization and evidence-review decisions.
- Attempt history, timestamps, and execution status.
- Tool outputs, errors, and artifact paths.
- Validation results and post-execution scientific review.

Campaigns can be resumed through the interface. Failed or interrupted attempts remain distinguishable from completed calculations; a tool error must not be represented as a successful scientific result.

### Evidence reuse and authorization

Existing evidence can be reused only when the researcher records where it came from and why it covers the planned stage. A reused stage is not represented as a newly executed calculation.

New numerical work is guarded by two distinct actions: campaign-level authorization and an explicit stage-execution action. Revoking campaign authorization overrides an earlier authorization. The execution registry restricts numerical execution to registered tools; the planner does not dynamically execute arbitrary tool names produced by a language model.

### Result presentation

Research Studio presents stage summaries and structured numerical outputs in readable sections and tables. Detailed machine-readable JSON remains available through an explicit advanced view and downloadable records for reproducibility. The interface distinguishes campaign execution status, the pre-execution evidence gate, and post-execution scientific review.

For the design principles and workflow rationale, see [Phase 3 product and interface principles](docs/PHASE3_PRODUCT_DESIGN.md).

## Application interface

Launch the Streamlit application with **streamlit run app.py**. The interface organizes the research environment around the following workspaces:

- **Quick simulations:** classical and quantum simulation entry points.
- **Simulation results:** collected outputs from simulation runs.
- **Diagnostics:** classical dynamical diagnostics, ergodic-hierarchy analyses, mixing diagnostics, and periodic-orbit investigations.
- **Comparisons:** controlled trajectory, geometry, and classical–quantum comparisons.
- **Data and evidence:** inspection and analysis of existing research data.
- **Research Studio:** question framing, evidence discovery, workflow planning, guarded campaign execution, campaign records, and scientific review.
- **Research Director:** iterative evidence-led investigation, automatic read-only inspection, scholarly literature retrieval, symbolic algebra support, and approval-gated numerical proposals.
- **Research assistant:** natural-language interaction with the LangGraph research agent, including DKT numerical tools, scholarly discovery, and symbolic mathematics.

Exact tab labels may evolve as the interface develops. The scientific tools and their recorded inputs, rather than the visual arrangement of tabs, define the calculation performed.

## Technology stack

| Component | Role |
|---|---|
| Python | Agent orchestration, analysis, data handling, and application logic |
| LangGraph / LangChain | Research-agent workflow and tool integration |
| Google Gemini | Language-model reasoning for the research assistant |
| Julia | Numerical simulation and quantum-dynamics computation |
| Streamlit | Interactive research interface and dashboards |
| NumPy / SciPy | Numerical arrays and scientific computation |
| Pandas | Tabular data processing and CSV workflows |
| Matplotlib | Scientific plotting |
| Pytest | Automated Python tests |
| Git submodules | Versioned Julia simulation package integration |

Python dependencies are listed in requirements.txt. The Julia package is maintained in the QuantumKickedTop.jl submodule.

## Installation

### Prerequisites

- Git with submodule support.
- Python 3.10 or later.
- Julia (the project has been developed and tested with Julia 1.12).
- A Google Gemini API key for the language-model research assistant.

### 1. Clone the repository

~~~bash
git clone --recurse-submodules https://github.com/avadhut1987/DKT-Research-Agent.git
cd DKT-Research-Agent
~~~

If the repository has already been cloned without its submodule, initialize it with:

~~~bash
git submodule update --init --recursive
~~~

### 2. Create a Python environment

~~~bash
python -m venv .venv
source .venv/bin/activate
~~~

On Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

### 3. Install Python dependencies

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### 4. Instantiate the Julia environment

From the repository root:

~~~bash
julia --project=QuantumKickedTop.jl -e 'using Pkg; Pkg.instantiate()'
~~~

### 5. Configure the Gemini API key

Set the key in the current shell session:

~~~bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
~~~

Do not place a real API key in source files, commit it to Git, or include it in campaign records. Keep credentials in your local environment or an appropriate secrets manager.

## Running the application and agent

With the Python environment active and the repository root as the working directory, launch the Streamlit interface:

~~~bash
streamlit run app.py
~~~

Streamlit prints a local URL in the terminal. Open that address in a web browser.

The LangGraph research agent can also be launched from the terminal:

~~~bash
PYTHONPATH=. python -m dkt_agent.agent
~~~

Ensure that the virtual environment is active and the GOOGLE_API_KEY environment variable is available. The agent can use registered research tools for simulations, comparisons, data analysis, evidence discovery, and supported diagnostics. Tool availability does not remove the need to specify valid inputs or interpret outputs within their scientific limits.

## Julia workflows

The Julia simulation package is located in QuantumKickedTop.jl/. The Python application invokes Julia-backed workflows for supported classical and quantum calculations.

Standalone Julia scripts are maintained under scripts/julia/. They cover selected tasks such as classical trajectory generation, grid-based simulations, fixed-point analysis, Lyapunov/KS diagnostics, entanglement dynamics, entropy heatmaps, and classical–quantum comparisons.

For a particular script, inspect its source for the expected command-line arguments, parameter conventions, and output paths. Initialize the Julia project before running package-dependent scripts.

## Testing

Run the Python test suite from the repository root:

~~~bash
PYTHONPATH=. python -m pytest tests/python/
~~~

The tests cover agent/tool integration, classical analysis, classical–quantum comparisons, numerical diagnostics, research planning and execution, campaign persistence and authorization, result presentation, and mathematical-notation rendering.

Some CI workflows also run Julia smoke tests. A passing test suite verifies the tested software behaviour; it does not validate every possible physical interpretation or establish asymptotic dynamical properties.

## Data and provenance

The application generates CSV files, plots, JSON metadata, campaign records, and downloadable archives depending on the selected workflow. Generated numerical data and research campaign records are stored under the repository's results area unless a tool or user specifies another output location.

For reproducible analysis, retain:

- The exact DKT parameters $k_r$, $k_\theta$, and $p$.
- Initial conditions and the coordinate convention.
- The number of kicks and any transient-discard interval.
- Ensemble size, grid resolution, random seed, and sampling procedure, where applicable.
- Observable definitions and normalization conventions.
- Partition resolution and lag range for mixing estimates.
- Software version / commit and relevant Julia environment.
- Original CSV files, metadata manifests, campaign records, and derived artifacts.

Do not infer missing parameters from a filename alone when the associated metadata are unavailable. Do not silently replace missing, failed, or uncomputed values with zero. Keep raw data separate from derived summaries so that analyses can be repeated.

## Recommended research practice

A robust computational investigation should:

1. State a precise, falsifiable research question.
2. Identify the observable and the mathematical property being investigated.
3. Fix control parameters and vary only the parameter relevant to the question.
4. Search and inspect existing evidence before generating new data.
5. Record all numerical inputs, sampling assumptions, and software provenance.
6. Use independent initial conditions or ensembles where the diagnostic requires them.
7. Check numerical constraints such as unit-sphere normalization and state norms where applicable.
8. Examine sensitivity to observation time, ensemble size, grid resolution, partition resolution, and random seed.
9. Separate direct numerical observations from physical interpretation.
10. State limitations and unresolved questions explicitly.

For a fixed-$k_r$ asymmetry study, keep $k_r$, $p$, initial conditions, and kick count fixed while varying $k_\theta$. For an ergodic-hierarchy investigation, do not use one diagnostic as a proxy for another: report mean-ergodicity, Birkhoff-average, weak-mixing-related, and strong-mixing-related results separately.

## Scientific limitations

This software supports controlled numerical investigation. It does not, by itself, provide mathematical proofs of ergodicity, mixing, invariant measures, or asymptotic convergence.

Interpret all finite-data results in relation to:

- The finite observation time and transient treatment.
- The number and distribution of sampled initial conditions.
- The observables actually evaluated.
- The numerical precision and stability checks performed.
- The chosen partition and lag range for mixing estimates.
- The finite seed set used in fixed-point and periodic-orbit searches.
- The assumptions and scope of each numerical method.

In particular, finite-time trajectory separation is not a Lyapunov exponent; finite-partition alpha estimates are not a proof of strong mixing; full-sphere sampling is not a local-region test; and a local-cloud geometry comparison is not a test of ergodicity. These distinctions are part of the intended scientific use of the platform.

## Repository organization

The principal components are organized as follows. The exact file set may expand as the project develops.

~~~text
DKT-Research-Agent/
├── app.py                         # Main Streamlit application
├── README.md
├── requirements.txt
├── dkt_agent/
│   ├── agent.py                   # LangGraph research agent
│   ├── tools.py                   # Core simulation tools
│   ├── langchain_tools.py         # Agent-facing tools
│   ├── research_orchestrator.py   # Deterministic research planner
│   ├── research_execution.py      # Campaign persistence and guarded execution
│   ├── research_studio.py         # Research Studio interface
│   ├── streamlit_rendering.py     # Structured result presentation
│   └── ...                        # Diagnostics and domain-specific modules
├── scripts/
│   └── julia/                     # Julia workflow entry points
├── QuantumKickedTop.jl/           # Julia simulation package (submodule)
├── docs/
│   └── PHASE3_PRODUCT_DESIGN.md   # Research workflow and interface principles
├── tests/
│   └── python/                    # Python regression and scientific-workflow tests
└── results/                       # Generated data and campaign records (runtime)
~~~

## Acknowledgements and citation

This repository is a computational research project for the Double Kicked Top model. If you use the software, numerical outputs, or workflows in academic work, cite the repository at the commit or release corresponding to the version used and describe the relevant numerical methods, parameters, and sampling assumptions in your methods section.

Repository: [github.com/avadhut1987/DKT-Research-Agent](https://github.com/avadhut1987/DKT-Research-Agent)
