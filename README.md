# Type 2 Diabetes Clinical Trial Pre-Screening Agent

## Library Desk Analogy
Imagine the pre-screening agent as a librarian organizing index cards. On one desk, there are cards detailing the patient's history (diagnoses, medications, lab results). On another desk, there are cards detailing trial requirements. The librarian first quickly filters out the trial cards that are closed for recruitment or strictly age-incompatible (Structured Filtering). Then, for the remaining trials, they retrieve specific relevant fact cards from the patient's desk and criteria cards from the trial's desk (Evidence Retrieval). Next, they compare these pairs of cards meticulously, categorizing the match as Supported, Not Supported, Unknown, Conflicting, or requiring a head librarian's review (Criterion Evaluation). Finally, they compile a shortlist of the top three most promising trials, along with the evidence cards attached, and leave them in a neat folder for the clinical research coordinator to make the final call (Report Generation).

## Architecture & Design Choices
- **Framework**: LangGraph is used to define an explicit, step-by-step state machine with four clear stages: Structured Filtering, Evidence Retrieval, Criterion Evaluation, and Report Generation.
- **State Schema**: The agent maintains a state dictionary tracking `patient_id`, `patient_data`, `trials`, `filtered_trials`, `evidence`, `evaluations`, and `report`.
- **Retrieval (RAG)**: A mock in-memory retrieval is set up to preserve source identifiers. It matches patient facts with trial criteria instead of dumping the whole JSON into the context.
- **Decoupled Evaluation**: Clinical fit and recruiting status are evaluated separately, preserving the specific states (SUPPORTED, NOT_SUPPORTED, etc.).

## Setup Instructions
1. Ensure Python 3.10+ is installed.
2. Run `pip install -r requirements.txt`.
3. To run the single patient end-to-end vertical slice: `python agent.py`
4. To run the evaluations: `python evals.py`

## Known Limitations
- The current implementation provides a structural vertical slice using rule-based mocks for the LLM evaluation to ensure it runs immediately without an API key. To fully utilize an LLM, the `evaluate_criteria` node would need a real LangChain LLM instance injected.
- The evidence retrieval relies on basic metadata matching rather than a full vector database due to time constraints, but demonstrates the architectural boundary.
