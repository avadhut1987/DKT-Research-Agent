# Scientific review workflow

Research Studio separates three milestones:

1. **Execution authorization** — user permission to generate new numerical data.
2. **Pre-execution evidence sufficiency review** — stage-level decision to reuse existing evidence or run new calculations.
3. **Post-execution scientific review** — an explicit assessment after all planned stages are complete.

The automatic campaign validation checks saved results, submitted inputs, overlapping reported parameters, attempt records and accessible CSV/JSON artifacts. It is a reproducibility and record-integrity screen only. It does not prove ergodicity, mixing, chaoticity, convergence, or scientific sufficiency.

A user records one of three review outcomes: accepted for the stated question, further work required, or inconclusive. The rationale is saved in the campaign record and audit events. The campaign lifecycle status is derived from stage execution and the recorded scientific review rather than treating successful tool return as scientific completion.
