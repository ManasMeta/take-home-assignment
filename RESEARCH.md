# Research

## Problem Framing
Clinical trial pre-screening is time-consuming and error-prone when done manually by research coordinators. The core issue is not simply finding overlapping keywords like "Type 2 Diabetes", but rather matching structured criteria (like age and recruiting status) alongside complex unstructured criteria (like medical history, eGFR, HbA1c ranges) that require temporal and clinical reasoning.

## Approach
Our goal is to build an agentic workflow that does not make final decisions, but rather highlights evidence, identifies missing information (UNKNOWN), and flags conditions needing expert review (REQUIRES_CLINICAL_REVIEW). 

The decision to use an explicit directed graph (LangGraph) rather than a single monolithic prompt allows us to separate concerns:
1. **Filtering** offloads deterministic tasks.
2. **Retrieval** limits hallucination by restricting context.
3. **Evaluation** forces the LLM to output specific categorized states.
4. **Generation** ensures the output is tailored for a coordinator, not a patient.
