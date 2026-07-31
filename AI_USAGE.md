# AI Usage

During the development of this project, AI coding assistants (like Copilot and Gemini) were used to accelerate boilerplate generation and help design the state schema for LangGraph.

- **Accepted Suggestion**: I used AI to generate the boilerplate TypedDict for the LangGraph state. It correctly identified the need for a list to hold intermediate filtered trials before evaluation.
- **Rejected/Changed Suggestion**: The AI suggested using a large vector store (like Pinecone or Chroma) for the Evidence Retrieval step. I rejected this and opted for a simple in-memory dictionary and heuristic retrieval, as the assignment explicitly stated that a local or in-memory retrieval approach with simple metadata filters is sufficient, and hybrid search/reranking is not required.
- **Verification**: I verified the behavior by manually inspecting the output of the `agent.py` script to ensure the required criterion states (`SUPPORTED`, `NOT_SUPPORTED`, `UNKNOWN`, `CONFLICTING_EVIDENCE`, `REQUIRES_CLINICAL_REVIEW`) were exclusively used and properly assigned in the vertical slice.
