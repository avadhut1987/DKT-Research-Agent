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
