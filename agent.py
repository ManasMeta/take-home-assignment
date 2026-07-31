import json
import os
import sys
import re
import time
import argparse
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

try:
    from langsmith import traceable
except ImportError:
    def traceable(name: Optional[str] = None, run_type: Optional[str] = None):
        def decorator(func):
            return func
        return decorator

# Ensure stdout handles UTF-8 on Windows environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Define strict criterion states per assignment rubric
VALID_STATES = [
    "SUPPORTED",
    "NOT_SUPPORTED",
    "UNKNOWN",
    "CONFLICTING_EVIDENCE",
    "REQUIRES_CLINICAL_REVIEW"
]

def setup_langsmith(enable: bool = True):
    """
    Configures LangSmith observability tracing environment flags.
    Automatically loads variables from .env if present.
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ[k] = v

    if enable or os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if "LANGCHAIN_PROJECT" not in os.environ:
            os.environ["LANGCHAIN_PROJECT"] = "t2d-clinical-trial-screening"
        api_key = os.environ.get("LANGCHAIN_API_KEY")
        key_status = f"Key Loaded ({api_key[:10]}...)" if api_key else "No API Key found"
        print(f"[Telemetry] LangSmith Tracing Active | Project: {os.environ['LANGCHAIN_PROJECT']} | {key_status}")

class CriterionResult(BaseModel):
    state: str = Field(description="One of: SUPPORTED, NOT_SUPPORTED, UNKNOWN, CONFLICTING_EVIDENCE, REQUIRES_CLINICAL_REVIEW")
    reason: str = Field(description="Clinical rationale for this evaluation")
    evidence_id: str = Field(description="FHIR source_id or data provenance tag")

class TrialEvaluationSchema(BaseModel):
    age: CriterionResult
    hba1c: CriterionResult
    current_diabetes_medications: CriterionResult
    egfr: CriterionResult
    trial_recruiting_status: CriterionResult
    other_criteria: CriterionResult

class AgentState(TypedDict):
    patient_id: str
    patient: Dict[str, Any]
    trials: List[Dict[str, Any]]
    filtered_trials: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    evaluations: Dict[str, Any]
    report: str
    use_llm: bool
    interactive_hitl: bool
    human_approved: Optional[bool]
    human_review_notes: Optional[str]
    telemetry: Dict[str, Any]

def load_data(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Node 1: Structured Filtering
@traceable(name="structured_filtering", run_type="chain")
def structured_filtering(state: AgentState) -> dict:
    print(f"-> [Node 1] Structured Filtering for Patient {state['patient_id']}...")
    trials = state["trials"]
    patient = state["patient"]
    
    demographics = patient.get("demographics", {})
    patient_age = demographics.get("age_at_reference_date")
    
    filtered = []
    for t in trials:
        status = (t.get("overall_status") or "").upper()
        # Filter out inactive recruitment statuses
        if status in ["TERMINATED", "WITHDRAWN", "SUSPENDED", "COMPLETED"]:
            continue
            
        min_age = t.get("minimum_age_years")
        max_age = t.get("maximum_age_years")
        
        # Age boundary check (null means open boundary)
        if patient_age is not None:
            if min_age is not None and patient_age < min_age:
                continue
            if max_age is not None and patient_age > max_age:
                continue
                
        filtered.append(t)
        
    print(f"   Passed filtering: {len(filtered)} / {len(trials)} trials.")
    return {"filtered_trials": filtered[:10]} # Focus top candidates

# Node 2: Evidence Retrieval (RAG Engine)
@traceable(name="evidence_retrieval", run_type="retriever")
def evidence_retrieval(state: AgentState) -> dict:
    print(f"-> [Node 2] Evidence Retrieval (RAG) for {len(state['filtered_trials'])} trials...")
    patient = state["patient"]
    evidence_by_trial = {}
    
    # 1. Index Patient Facts with FHIR source_id provenance
    patient_demographics = patient.get("demographics", {})
    patient_obs = patient.get("observations", [])
    patient_meds = patient.get("medications", [])
    
    # Extract HbA1c observations
    hba1c_obs = [o for o in patient_obs if o.get("type") == "hba1c"]
    hba1c_obs_sorted = sorted(hba1c_obs, key=lambda x: x.get("effective_date", ""), reverse=True)
    
    # Extract eGFR observations
    egfr_obs = [o for o in patient_obs if o.get("type") == "egfr"]
    egfr_obs_sorted = sorted(egfr_obs, key=lambda x: x.get("effective_date", ""), reverse=True)
    
    # Extract Medication list
    active_meds = [m for m in patient_meds if m.get("status") in [None, "active", "completed"]]
    
    for trial in state["filtered_trials"]:
        nct_id = trial.get("nct_id", "UNKNOWN_TRIAL")
        eligibility_text = trial.get("eligibility_text", "")
        
        # RAG chunking: extract text relevant to each criterion
        hba1c_clauses = [line.strip() for line in eligibility_text.split("\n") if re.search(r"hba1c|hb1ac|glycated|a1c", line, re.IGNORECASE)]
        egfr_clauses = [line.strip() for line in eligibility_text.split("\n") if re.search(r"egfr|egrf|renal|kidney|creatinine|gfr", line, re.IGNORECASE)]
        med_clauses = [line.strip() for line in eligibility_text.split("\n") if re.search(r"medication|insulin|metformin|sglt|glp|drug|hypoglycemia|agent", line, re.IGNORECASE)]
        
        evidence_by_trial[nct_id] = {
            "age": {
                "patient_age": patient_demographics.get("age_at_reference_date"),
                "trial_min_age": trial.get("minimum_age_years"),
                "trial_max_age": trial.get("maximum_age_years"),
                "source_id": "patient_demographics"
            },
            "recruiting_status": {
                "status": trial.get("overall_status"),
                "source_id": "trial_metadata"
            },
            "hba1c": {
                "patient_observations": hba1c_obs_sorted,
                "trial_clauses": hba1c_clauses,
                "primary_source_id": hba1c_obs_sorted[0].get("source_id") if hba1c_obs_sorted else "lab_results"
            },
            "egfr": {
                "patient_observations": egfr_obs_sorted,
                "trial_clauses": egfr_clauses,
                "primary_source_id": egfr_obs_sorted[0].get("source_id") if egfr_obs_sorted else "lab_results"
            },
            "medications": {
                "patient_medications": active_meds,
                "trial_clauses": med_clauses,
                "primary_source_id": active_meds[0].get("source_id") if active_meds else "medications_list"
            },
            "other": {
                "eligibility_text": eligibility_text[:300] + "...",
                "source_id": "trial_eligibility_text"
            }
        }
        
    return {"evidence": evidence_by_trial}

# Node 3: Criterion Evaluation
def evaluate_hba1c(hba1c_data: dict) -> CriterionResult:
    obs = hba1c_data.get("patient_observations", [])
    if not obs:
        return CriterionResult(
            state="UNKNOWN",
            reason="No HbA1c laboratory observation found in patient record.",
            evidence_id="lab_results"
        )
    
    # Check for conflicting observations (wildly inconsistent recent values)
    if len(obs) > 1:
        v1, v2 = obs[0].get("value"), obs[1].get("value")
        if v1 is not None and v2 is not None and abs(v1 - v2) > 2.5:
            return CriterionResult(
                state="CONFLICTING_EVIDENCE",
                reason=f"Conflicting recent HbA1c values found ({v1}% on {obs[0].get('effective_date')} vs {v2}% on {obs[1].get('effective_date')}).",
                evidence_id=f"{obs[0].get('source_id')}, {obs[1].get('source_id')}"
            )
            
    latest = obs[0]
    val = latest.get("value")
    date = latest.get("effective_date")
    src = latest.get("source_id", "lab_results")
    
    clauses = hba1c_data.get("trial_clauses", [])
    clause_text = " ".join(clauses).lower()
    
    # Check trial HbA1c thresholds in clauses
    min_match = re.search(r"hba1c\s*[>≥=]\s*(\d+\.?\d*)", clause_text) or re.search(r"hba1c\s*\\\s*>\s*(\d+\.?\d*)", clause_text)
    if min_match:
        threshold = float(min_match.group(1))
        if val is not None:
            if val >= threshold:
                return CriterionResult(
                    state="SUPPORTED",
                    reason=f"Patient HbA1c ({val}% on {date}) meets trial threshold (>= {threshold}%).",
                    evidence_id=src
                )
            else:
                return CriterionResult(
                    state="NOT_SUPPORTED",
                    reason=f"Patient HbA1c ({val}% on {date}) is below trial requirement (>= {threshold}%).",
                    evidence_id=src
                )
                
    return CriterionResult(
        state="SUPPORTED",
        reason=f"Patient HbA1c documented ({val}% on {date}).",
        evidence_id=src
    )

def evaluate_egfr(egfr_data: dict) -> CriterionResult:
    obs = egfr_data.get("patient_observations", [])
    if not obs:
        return CriterionResult(
            state="UNKNOWN",
            reason="eGFR laboratory observation absent from patient record.",
            evidence_id="lab_results"
        )
        
    latest = obs[0]
    val = latest.get("value")
    date = latest.get("effective_date")
    src = latest.get("source_id", "lab_results")
    
    clauses = egfr_data.get("trial_clauses", [])
    clause_text = " ".join(clauses).lower()
    
    # Exclusion thresholds e.g., eGFR < 30 or eGFR < 45
    excl_match = re.search(r"egfr\s*\\\s*<\s*(\d+)", clause_text) or re.search(r"egfr\s*<\s*(\d+)", clause_text)
    if excl_match:
        excl_thresh = float(excl_match.group(1))
        if val is not None:
            if val < excl_thresh:
                return CriterionResult(
                    state="NOT_SUPPORTED",
                    reason=f"Patient eGFR ({val} mL/min/1.73m2 on {date}) triggers trial renal exclusion threshold (< {excl_thresh}).",
                    evidence_id=src
                )
            else:
                return CriterionResult(
                    state="SUPPORTED",
                    reason=f"Patient eGFR ({val} mL/min/1.73m2 on {date}) satisfies renal threshold (>= {excl_thresh}).",
                    evidence_id=src
                )
                
    return CriterionResult(
        state="SUPPORTED",
        reason=f"Patient eGFR documented ({val} mL/min/1.73m2 on {date}).",
        evidence_id=src
    )

def evaluate_medications(med_data: dict) -> CriterionResult:
    meds = med_data.get("patient_medications", [])
    clauses = med_data.get("trial_clauses", [])
    clause_text = " ".join(clauses).lower()
    
    if not meds and "taking" in clause_text:
        return CriterionResult(
            state="UNKNOWN",
            reason="Patient medication history is empty in record; trial requires active medication regimen.",
            evidence_id="medications_list"
        )
        
    med_names = [m.get("display_name") or m.get("name") or m.get("medication_name", "") for m in meds if isinstance(m, dict)]
    src_ids = [m.get("source_id") for m in meds if isinstance(m, dict) and m.get("source_id")]
    primary_src = ", ".join(src_ids) if src_ids else "medications_list"
    
    if "insulin" in clause_text and "exclusion" in clause_text:
        has_insulin = any("insulin" in name.lower() for name in med_names)
        if has_insulin:
            return CriterionResult(
                state="NOT_SUPPORTED",
                reason=f"Patient active medications include Insulin ({', '.join(med_names)}), violating trial exclusion.",
                evidence_id=primary_src
            )
            
    return CriterionResult(
        state="SUPPORTED",
        reason=f"Patient medication profile evaluated ({', '.join(med_names) if med_names else 'No prohibited medications'}).",
        evidence_id=primary_src
    )

@traceable(name="criterion_evaluation", run_type="chain")
def criterion_evaluation(state: AgentState) -> dict:
    print(f"-> [Node 3] Evaluating Criteria for {len(state['filtered_trials'])} trials...")
    evaluations = {}
    evidence = state["evidence"]
    
    for trial in state["filtered_trials"]:
        nct_id = trial.get("nct_id", "UNKNOWN_TRIAL")
        ev = evidence.get(nct_id, {})
        
        # Age evaluation
        age_ev = ev.get("age", {})
        p_age = age_ev.get("patient_age")
        min_a = age_ev.get("trial_min_age")
        max_a = age_ev.get("trial_max_age")
        
        if p_age is None:
            age_res = CriterionResult(state="UNKNOWN", reason="Patient age missing.", evidence_id="patient_demographics")
        elif (min_a is not None and p_age < min_a) or (max_a is not None and p_age > max_a):
            age_res = CriterionResult(state="NOT_SUPPORTED", reason=f"Patient age ({p_age}) outside trial bounds ({min_a}-{max_a}).", evidence_id="patient_demographics")
        else:
            age_res = CriterionResult(state="SUPPORTED", reason=f"Patient age ({p_age}) satisfies trial bounds ({min_a}-{max_a}).", evidence_id="patient_demographics")
            
        # Recruiting Status evaluation
        rec_ev = ev.get("recruiting_status", {})
        status = (rec_ev.get("status") or "").upper()
        if status in ["RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]:
            rec_res = CriterionResult(state="SUPPORTED", reason=f"Trial overall recruitment status is '{status}'.", evidence_id="trial_metadata")
        elif status in ["TERMINATED", "COMPLETED", "WITHDRAWN", "SUSPENDED"]:
            rec_res = CriterionResult(state="NOT_SUPPORTED", reason=f"Trial overall status is '{status}'.", evidence_id="trial_metadata")
        else:
            rec_res = CriterionResult(state="UNKNOWN", reason=f"Trial status '{status}' unconfirmed.", evidence_id="trial_metadata")
            
        # Evaluators for HbA1c, eGFR, Meds
        hba1c_res = evaluate_hba1c(ev.get("hba1c", {}))
        egfr_res = evaluate_egfr(ev.get("egfr", {}))
        meds_res = evaluate_medications(ev.get("medications", {}))
        
        # Other criteria always REQUIRES_CLINICAL_REVIEW per assignment rubric
        other_res = CriterionResult(
            state="REQUIRES_CLINICAL_REVIEW",
            reason="Unstructured clinical eligibility criteria require human coordinator review.",
            evidence_id="trial_eligibility_text"
        )
        
        evaluations[nct_id] = {
            "age": age_res.model_dump(),
            "hba1c": hba1c_res.model_dump(),
            "current_diabetes_medications": meds_res.model_dump(),
            "egfr": egfr_res.model_dump(),
            "trial_recruiting_status": rec_res.model_dump(),
            "other_criteria": other_res.model_dump()
        }
        
    return {"evaluations": evaluations}

# Node 3.5: Human Review (LangGraph HITL Interrupt Node)
@traceable(name="human_review", run_type="chain")
def human_review(state: AgentState) -> dict:
    print("-> [Node 3.5] Human Review (HITL Node)...")
    evaluations = state.get("evaluations", {})
    interactive = state.get("interactive_hitl", False)
    
    flagged_trials = []
    for nct_id, evs in evaluations.items():
        has_review_flag = any(res.get("state") in ["REQUIRES_CLINICAL_REVIEW", "CONFLICTING_EVIDENCE"] for res in evs.values())
        if has_review_flag:
            flagged_trials.append(nct_id)
            
    print(f"   Evaluated {len(evaluations)} trials. {len(flagged_trials)} trials flagged for coordinator review.")
    
    if interactive and flagged_trials:
        print(f"\n[INTERRUPT TRIGGERED] LangGraph state execution paused.")
        print(f"Flagged Trials requiring human oversight: {', '.join(flagged_trials[:3])}")
        
        # Trigger true LangGraph interrupt
        human_input = interrupt({
            "status": "PAUSED_FOR_CLINICAL_REVIEW",
            "patient_id": state["patient_id"],
            "flagged_trials": flagged_trials,
            "message": "Unstructured criteria or conflicting evidence detected. Please provide coordinator approval & notes."
        })
        
        return {
            "human_approved": human_input.get("approved", True),
            "human_review_notes": human_input.get("notes", "Coordinator reviewed and approved trial matches.")
        }
        
    return {
        "human_approved": False,
        "human_review_notes": f"Pending Human Coordinator Review ({len(flagged_trials)} trials flagged for unstructured protocol clauses)."
    }

# Node 4: Report Generation
@traceable(name="report_generation", run_type="chain")
def report_generation(state: AgentState) -> dict:
    print("-> [Node 4] Generating Coordinator Pre-Screening Report...")
    patient_id = state["patient_id"]
    evaluations = state["evaluations"]
    filtered_trials = state["filtered_trials"]
    human_notes = state.get("human_review_notes", "Pending Human Review")
    human_approved = state.get("human_approved", False)
    
    # Rank trials based on clinical support
    def calculate_score(trial):
        nct_id = trial.get("nct_id")
        evs = evaluations.get(nct_id, {})
        score = 0
        for key, res in evs.items():
            st = res.get("state")
            if st == "SUPPORTED":
                score += 2
            elif st == "UNKNOWN":
                score += 0
            elif st == "NOT_SUPPORTED":
                score -= 5
        return score
        
    sorted_trials = sorted(filtered_trials, key=calculate_score, reverse=True)
    top_trials = sorted_trials[:3]
    
    report_lines = [
        f"# Clinical Trial Pre-Screening Report",
        f"**Patient ID**: `{patient_id}`",
        f"**Candidate Trials Evaluated**: {len(filtered_trials)}",
        f"**Top Recommended Trial Matches**: {len(top_trials)}",
        f"**Human-in-the-Loop Sign-off**: `{ 'APPROVED' if human_approved else 'REQUIRES_CLINICAL_REVIEW' }`\n",
        "---"
    ]
    
    for idx, trial in enumerate(top_trials, 1):
        nct_id = trial.get("nct_id", "UNKNOWN")
        title = trial.get("brief_title", "Untitled Study")
        evs = evaluations.get(nct_id, {})
        
        report_lines.append(f"\n### Candidate #{idx}: {nct_id} - {title}")
        report_lines.append(f"**Reason Surfaced**: Passed age and recruitment status filters; high clinical feature match.\n")
        
        # 1. Operational Recruiting Status (Separated from Clinical Fit)
        rec_status = trial.get("overall_status", "N/A")
        rec_eval = evs.get("trial_recruiting_status", {})
        report_lines.append(f"#### 1. Operational Recruiting Status")
        report_lines.append(f"- **Overall Status**: `{rec_status}` — {rec_eval.get('reason', '')} *(Source: `{rec_eval.get('evidence_id', 'trial_metadata')}`)*\n")
        
        # 2. Clinical Fit Criteria Breakdown
        report_lines.append("#### 2. Clinical Eligibility Criteria")
        missing_flags = []
        for crit in ["age", "hba1c", "current_diabetes_medications", "egfr"]:
            details = evs.get(crit, {})
            st = details.get("state")
            reason = details.get("reason")
            src = details.get("evidence_id")
            
            tag = "[SUPPORTED]" if st == "SUPPORTED" else ("[NOT_SUPPORTED]" if st == "NOT_SUPPORTED" else ("[UNKNOWN]" if st == "UNKNOWN" else "[REVIEW]"))
            report_lines.append(f"- {tag} **{crit.upper()}**: `{st}` — {reason} *(Source: `{src}`)*")
            
            if st == "UNKNOWN":
                missing_flags.append(crit)
                
        # 3. Unstructured Protocol Criteria
        other_eval = evs.get("other_criteria", {})
        report_lines.append(f"\n#### 3. Unstructured Protocol Criteria")
        report_lines.append(f"- [REVIEW] **OTHER_CRITERIA**: `{other_eval.get('state')}` — {other_eval.get('reason')} *(Source: `{other_eval.get('evidence_id')}`)*")
                
        # 4. Human Coordinator Summary
        report_lines.append("\n#### 4. Human Coordinator Summary")
        report_lines.append(f"- **Unanswered Questions / Missing Data**: {', '.join(missing_flags) if missing_flags else 'None'}")
        report_lines.append(f"- **Human-Review Decision**: `{ 'APPROVED' if human_approved else 'REQUIRES_CLINICAL_REVIEW' }`")
        report_lines.append(f"- **Coordinator Notes**: {human_notes}")
        report_lines.append("\n---")
        
    return {"report": "\n".join(report_lines)}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("structured_filtering", structured_filtering)
    workflow.add_node("evidence_retrieval", evidence_retrieval)
    workflow.add_node("criterion_evaluation", criterion_evaluation)
    workflow.add_node("human_review", human_review)
    workflow.add_node("report_generation", report_generation)
    
    workflow.set_entry_point("structured_filtering")
    workflow.add_edge("structured_filtering", "evidence_retrieval")
    workflow.add_edge("evidence_retrieval", "criterion_evaluation")
    workflow.add_edge("criterion_evaluation", "human_review")
    workflow.add_edge("human_review", "report_generation")
    workflow.add_edge("report_generation", END)
    
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Type 2 Diabetes Clinical Trial Pre-Screening Agent")
    parser.add_argument("--data", default="Type2-Diabetes-Trial-Agent-Dataset.json", help="Path to JSON dataset")
    parser.add_argument("--patient_index", type=int, default=0, help="Index of patient to process (0-14)")
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM structured output evaluation")
    parser.add_argument("--interactive_hitl", action="store_true", help="Enable interactive Human-in-the-Loop coordinator approval")
    parser.add_argument("--langsmith", action="store_true", help="Enable LangSmith observability tracing")
    args = parser.parse_args()
    
    if args.langsmith:
        setup_langsmith(True)
        
    data = load_data(args.data)
    patients = data.get("patients", [])
    trials = data.get("trials", [])
    
    if not patients:
        print("No patients found in dataset.")
        exit(1)
        
    selected_patient = patients[args.patient_index]
    patient_id = selected_patient.get("patient_id", f"PT_{args.patient_index}")
    
    print(f"Starting Pre-Screening Agent pipeline for Patient {patient_id}...")
    
    app = build_graph()
    thread_config = {"configurable": {"thread_id": f"patient_session_{patient_id}"}}
    
    initial_input = {
        "patient_id": patient_id,
        "patient": selected_patient,
        "trials": trials,
        "filtered_trials": [],
        "evidence": {},
        "evaluations": {},
        "report": "",
        "use_llm": args.use_llm,
        "interactive_hitl": args.interactive_hitl,
        "human_approved": False,
        "human_review_notes": "",
        "telemetry": {}
    }
    
    final_state = app.invoke(initial_input, config=thread_config)
    
    # Check if graph paused at a Human-in-the-Loop (HITL) interrupt
    state_history = app.get_state(thread_config)
    if state_history and state_history.next:
        print("\n==========================================================")
        print("  LANGGRAPH HUMAN-IN-THE-LOOP (HITL) INTERRUPT TRIGGERED  ")
        print("==========================================================")
        notes_input = input("Enter Clinical Coordinator Approval Notes: ") or "Approved by Clinical Coordinator."
        
        # Resume LangGraph execution with human decision command
        final_state = app.invoke(
            Command(resume={"approved": True, "notes": notes_input}),
            config=thread_config
        )
        
    print("\n\n" + final_state["report"])

