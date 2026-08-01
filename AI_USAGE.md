# AI_USAGE.md

## How AI Tools Were Used

This project was built with AI coding assistance (Gemini/Claude) used as a collaborative
pair-programmer, not an autopilot. Below is an honest account of what was accepted,
what was rejected, and how correctness was verified.

## Suggestion Accepted

**LangGraph state schema design.** The AI-suggested state schema — separating
`filtered_trials`, `evidence`, `evaluations`, and `report` into distinct typed fields
rather than one flat dictionary — was accepted as proposed. It kept each node's
input/output contract explicit and made the graph easy to inspect at any stage,
which directly supported the assignment's requirement for a transparent, testable
state object.

## Suggestion Rejected / Changed

**Conditional branching (dynamic graph edges).** The AI suggested adding conditional
routing — e.g., short-circuiting `structured_filtering` straight to `report_generation`
when zero trials pass the filter, or bypassing the HITL interrupt when nothing is
flagged for review. This was rejected. The assignment explicitly states it does not
require selective re-execution or complex routing, and reviewers were told to
prioritize a clear, explainable graph over a maximally clever one. Adding branching
this late also risked destabilizing an already-verified 4-node pipeline with no time
left to re-run the full 900-assessment benchmark if something broke. The linear graph
was kept as-is.

## Additions Beyond the Minimum Requirement (My Design Decisions)

The Human-in-the-Loop (HITL) interrupt node and LangSmith tracing were my own
additions, not something the assignment required (the spec explicitly says an
interactive approval workflow and production monitoring are "not required"). I chose
to add them anyway to demonstrate a realistic reliability workflow for a
regulatory/healthcare context, where a coordinator sign-off step and execution
traceability are practically valuable even if not graded requirements.

## How Final Behavior Was Verified

- Ran `python evals.py` across all 15 synthetic patients (900 criterion assessments)
  to confirm the accepted state schema didn't silently drop or misroute data between
  nodes.
- Verified `Unknown Avoidance Rate = 0.0000` — confirming no criterion was ever
  resolved to a false SUPPORTED/NOT_SUPPORTED when the required patient fact
  (lab value or medication) was actually missing from the record.
- Verified `Citation Accuracy = 1.0000` — every evaluated criterion links back to
  a real `source_id` in the original dataset, confirming no invented evidence.
- Verified HbA1c exclusion handling against protocol `NCT05181449` to ensure exclusion rules (e.g., `HbA1c > 11%`) evaluate correctly as `SUPPORTED` when patient is within safety limits and `NOT_SUPPORTED` when exceeded.
- Verified that `calculate_score` aggregates exclusively the 4 clinical criteria (`age`, `hba1c`, `medications`, `egfr`), keeping operational recruiting status separated per specification.
- Manually inspected the report output for `--patient_index 0` and `--patient_index 3`
  (a case with a missing eGFR value) to confirm the UNKNOWN state and unanswered
  questions appeared correctly rather than being silently skipped.