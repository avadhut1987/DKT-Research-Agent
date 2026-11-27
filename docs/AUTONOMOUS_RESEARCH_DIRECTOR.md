# Autonomous DKT Research Director

The Research Director is a semi-autonomous, evidence-led layer over the existing
DKT research tools. It does not replace the original chat assistant, manual
simulation panels, Research Studio, or Julia physics engine.

## User experience

Open the Streamlit page **Research Director**.

1. Enter a scientific objective in ordinary language.
2. Select **Start / continue investigation**.
3. The director searches existing research evidence first.
4. It assesses whether that evidence is sufficient for the stated claim.
5. It may automatically perform additional allow-listed read-only inspection.
6. If new numerical data are required, it presents exactly one proposed
   calculation, its JSON arguments, why it is needed, and what result could
   change the conclusion.
7. The calculation runs only after the user clicks **Approve and run this
   numerical calculation**. Approval applies to that single calculation.
8. The director evaluates the returned result and repeats the decision process.
9. It stops at a defensible finite-data conclusion, an unresolved question,
   a missing scientific input, a failure, or the configured decision-step cap.

## Scientific guardrails

- Evidence search precedes repeated numerical work.
- Only tools in `dkt_agent/research_director.py::TOOL_REGISTRY` may be selected.
- Model output is parsed and validated as JSON; tool schemas validate arguments
  again at invocation.
- Each decision proposes at most one next step.
- Numerical tools require an explicit per-step user click.
- The director is not allowed to claim mathematical proof from finite numerical
  evidence.
- Mean ergodicity, Birkhoff/pointwise ergodicity, weak mixing, strong mixing,
  local ergodicity, Lyapunov exponents, and geometric diagnostics remain distinct.
- At fixed $k_r$, changing $k_\theta$ is interpreted as a phase-space geometry
  change, not a change in chaoticity. Classical chaoticity is controlled by
  $k_r$.
- The director should not choose local or direction-dependent ergodicity unless
  the research question requires a spatially restricted or observable-resolved
  claim and simpler evidence is inadequate.

## JSON and mathematical rendering

The decision payload and tool arguments are displayed as structured JSON using
Streamlit's JSON renderer. A complete machine-readable investigation record can
be downloaded as `dkt_research_director_record.json`. Scientific prose uses
Streamlit Markdown with LaTeX math rendering; the presentation preview does not
modify the stored research question or numerical arguments.

## Current boundaries

- The director makes one decision at a time and caps a single automatic
  read-only pass at five actions; the total investigation is capped at ten
  recorded tool steps before pausing for review.
- Numerical execution is supervised one calculation at a time.
- It is not an unrestricted autonomous agent and cannot silently launch
  arbitrary code, modify research files, or change the DKT physics model.
- The director's conclusion is a structured research assessment, not a
  substitute for the researcher's scientific judgment.
