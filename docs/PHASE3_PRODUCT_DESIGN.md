# Phase 3 product and interface principles

## Product direction
The DKT Research Agent is a scientific research environment, not a chat wrapper around an expanding tool list. It should guide a researcher from a well-defined question to a reproducible and critically interpreted result.

## Research workflow
1. **Frame** the question, target claim, observables and controlled parameters.
2. **Discover** existing evidence before repeating calculations.
3. **Plan** methods matched to the question and expose required inputs.
4. **Authorize** new numerical execution explicitly.
5. **Execute** validated tools with recorded parameters and provenance.
6. **Verify** outputs, numerical checks, completeness and method limitations.
7. **Interpret** observations, physical interpretation, unresolved claims and decisions separately.
8. **Report** a reproducible record with figures, tables, methods and artifacts.

## Scientific invariants
- \(k_r\) controls classical chaoticity, Lyapunov behaviour and the regular-to-chaotic transition.
- \(k_\theta\) changes phase-space geometry and may break time-reversal symmetry; it must not be described as changing classical chaoticity.
- Mean ergodicity, Birkhoff/pointwise ergodicity, weak mixing and strong mixing are separate concepts.
- Finite-data indicators are not mathematical proofs of asymptotic properties.
- Missing is not zero; non-detection in a finite search is not proof of non-existence.
- New numerical work must be explicitly authorized; evidence retrieval is read-only.

## Interface design philosophy

### Information architecture
- Organize around researcher tasks and stages, not internal Python module names.
- Keep simulation, diagnostics, comparisons, evidence, research planning and conversation clearly separated.
- Make the primary action obvious; group advanced settings using progressive disclosure.
- Avoid duplicate workflows and controls whose purpose is unclear.

### Visual hierarchy and consistency
- Use restrained styling, consistent spacing, concise subtitles and predictable heading levels.
- Use mathematical notation consistently for physical parameters and observables.
- Use metric cards only for genuinely important run summaries.
- Align controls, tables, charts and downloads consistently.
- Prefer descriptive labels and help text over cryptic abbreviations.
- Keep tables compact, units explicit and numerical precision meaningful.

### Scientific result presentation
- Present run configuration and provenance with numerical results.
- Separate **Observation**, **Interpretation**, **Limitations** and **Decision**.
- Label finite-time, finite-sample and finite-partition quantities explicitly.
- Display missing, failed and not-computed values distinctly; never silently coerce them to zero.
- Put detailed tool activity and exact arguments in secondary disclosure.
- Downloads should have stable filenames and enough metadata to identify the calculation.

### Interaction and accessibility
- Collect related inputs in forms and validate before execution.
- Do not ask users to re-enter known information.
- Show execution mode and authorization requirements before expensive work.
- Preserve selections and results across Streamlit reruns.
- Provide actionable errors while retaining technical details.
- Maintain contrast, descriptive labels, keyboard access and responsive layouts.
- Do not rely on colour alone to communicate status.

## Phase 3 acceptance criteria
A feature must preserve scientific algorithms and defaults unless explicitly intended otherwise; have focused tests; expose inputs, outputs, provenance and limitations; fit the information architecture; use consistent terminology; avoid duplicate workflows; pass relevant CI; and be reviewed in the running Streamlit application before merge.


## Phase 3B execution architecture

Research Studio campaigns persist a plan snapshot and an ordered stage ledger. Each stage records its execution mode, exact arguments, attempt count, timestamps, result or error. Campaigns are stored under `results/research_campaigns/` and can be resumed. Execution is restricted to an explicit allow-list of validated tools; arbitrary tool names from model output are never dynamically resolved.

Read-only evidence discovery may be run independently. Numerical stages require both recorded user authorization and a separate execution action. Stages run in order, and later stages cannot start until earlier stages complete. Failed stages retain their inputs and error record and may be retried. This is a human-supervised execution foundation; it is not yet an unrestricted autonomous agent or an automatic scientific reviewer.

## Observable-resolved and local ergodicity

For direction-dependent questions, preserve separate results for (X,Y,Z) and the available quadratic/cross observables. Combine the existing nine-observable mean-ergodicity battery with the Birkhoff-average ensemble-variance study where appropriate. Do not compress mixed outcomes into a single global verdict.

The current samplers draw initial conditions uniformly over the full sphere. Observable-wise differences under that sampling support only finite-data statements about the tested observables and ensemble. They do not establish ergodicity on a spatially restricted region. A future local-region workflow must first define a dynamically meaningful invariant region or component and a justified conditional invariant measure; merely sampling initial points inside an arbitrary patch is not sufficient to define a measure-preserving restricted system.

## Matched-ensemble geometry comparison

At fixed (k_r) and (p), the geometry comparison uses the same reproducible uniform-sphere initial conditions for both selected (k_\theta) values. It reports pooled empirical-measure divergence and per-initial-condition divergences on a common equal-area spherical partition, together with all simulation paths and settings. This is a finite-window distributional comparison; it does not establish distinct invariant measures or changed classical chaoticity.
