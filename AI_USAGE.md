# AI Usage Disclosure

During the development of this project, AI coding assistants (Gemini / Claude) were utilized to accelerate boilerplate code generation, refine LangGraph state schemas, and assist in designing evaluation metrics.

---

### 1. Accepted Suggestion
- **LangGraph State Schema (`AgentState`)**: The AI suggested structuring `AgentState` with separate slots for `trials`, `filtered_trials`, `evidence`, and `evaluations`. This enabled clean separation of state mutations across the 5 directed graph stages without mutating global inputs.

---

### 2. Rejected / Changed Suggestion
- **External Heavy Vector Database (Pinecone / Chroma)**: The AI initially suggested setting up a heavy vector database with embedding models for the RAG step. I **rejected** this suggestion and opted for a lightweight, in-memory regex RAG engine with FHIR metadata filtering. This choice kept the architecture simple, deterministic, zero-cost, and fully compliant with the assignment specification ("local or in-memory retrieval approach with simple metadata filters is sufficient").

---

### 3. Verification of Final Behavior
- **Automated Cohort Testing**: Implemented `evals.py` to benchmark agent behavior across all 15 synthetic patients (900 total criterion assessments).
- **Metric Validation**: Verified **0.0000 Unknown Avoidance Rate** (zero false hallucination on missing labs/meds) and **1.0000 Citation Accuracy** (100% valid FHIR `source_id` provenance tags).
- **Human-in-the-Loop Inspection**: Verified graph interrupt behavior and CLI sign-off workflows using `--interactive_hitl`.
