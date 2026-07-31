import json
import argparse
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# Define our Agent's State
class AgentState(TypedDict):
    patient_id: str
    patient: Dict[str, Any]
    trials: List[Dict[str, Any]]
    filtered_trials: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    evaluations: Dict[str, Any]
    report: str

def load_data(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

# Node 1: Structured Filtering
def structured_filtering(state: AgentState) -> dict:
    print("-> Running Structured Filtering...")
    trials = state["trials"]
    patient = state["patient"]
    filtered = []
    
    # Very basic mock filtering - assume patient age is 50 for the sake of the vertical slice
    # and we only want Recruiting trials.
    for t in trials:
        # In a real scenario, we parse t['eligibility']['minimum_age'] etc.
        status = t.get("overall_status", "Recruiting")
        if status in ["Recruiting", "Not yet recruiting", "Enrolling by invitation"]:
            filtered.append(t)
            
    return {"filtered_trials": filtered[:10]} # Limit to 10 for processing

# Node 2: Evidence Retrieval
def evidence_retrieval(state: AgentState) -> dict:
    print("-> Retrieving Evidence...")
    # Mock retrieval: in a real app, this would use a vector store or BM25
    evidence = {
        "patient_facts": state["patient"],
        "trial_criteria": {t.get("nct_id", f"TRIAL_{i}"): t.get("eligibility", {}) for i, t in enumerate(state["filtered_trials"])}
    }
    return {"evidence": evidence}

# Node 3: Criterion Evaluation
def criterion_evaluation(state: AgentState) -> dict:
    print("-> Evaluating Criteria (Mock LLM)...")
    evaluations = {}
    
    # Mocking the evaluation process for the vertical slice
    for trial in state["filtered_trials"]:
        nct_id = trial.get("nct_id", "UNKNOWN_TRIAL")
        evaluations[nct_id] = {
            "age": {"state": "SUPPORTED", "reason": "Patient age falls within trial boundaries.", "evidence_id": "patient_demographics"},
            "hba1c": {"state": "UNKNOWN", "reason": "HbA1c data not recent enough.", "evidence_id": "lab_results"},
            "current_diabetes_medications": {"state": "SUPPORTED", "reason": "Patient is on Metformin.", "evidence_id": "medications_list"},
            "egfr": {"state": "NOT_SUPPORTED", "reason": "eGFR below trial threshold.", "evidence_id": "lab_results"},
            "trial_recruiting_status": {"state": "SUPPORTED", "reason": "Trial is actively recruiting.", "evidence_id": "trial_metadata"},
            "other_criteria": {"state": "REQUIRES_CLINICAL_REVIEW", "reason": "Additional specific cardiovascular history needed.", "evidence_id": "trial_inclusion_criteria"}
        }
        
    return {"evaluations": evaluations}

# Node 4: Report Generation
def report_generation(state: AgentState) -> dict:
    print("-> Generating Report...")
    report_lines = [f"=== Pre-Screening Report for Patient {state['patient_id']} ==="]
    
    # Sort and pick top 3 (Mock scoring)
    top_trials = state["filtered_trials"][:3]
    
    for trial in top_trials:
        nct_id = trial.get("nct_id", "UNKNOWN_TRIAL")
        title = trial.get("brief_title", "Untitled Trial")
        evals = state["evaluations"].get(nct_id, {})
        
        report_lines.append(f"\nTrial: {nct_id} - {title}")
        report_lines.append("Reason surfaced: Passed initial age and recruiting filters. Partial clinical match.")
        report_lines.append("--- Criteria Breakdown ---")
        
        clinical_fit_flags = []
        for crit, details in evals.items():
            report_lines.append(f"  * {crit.upper()}: {details['state']} - {details['reason']} (Source: {details['evidence_id']})")
            if details["state"] == "UNKNOWN":
                clinical_fit_flags.append(f"Missing info for {crit}")
                
        report_lines.append("--- Summary ---")
        report_lines.append(f"Unanswered Questions: {', '.join(clinical_fit_flags) if clinical_fit_flags else 'None'}")
        report_lines.append("Human-Review Status: REQUIRES_CLINICAL_REVIEW (Due to complex exclusion criteria)")
        
    return {"report": "\n".join(report_lines)}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("structured_filtering", structured_filtering)
    workflow.add_node("evidence_retrieval", evidence_retrieval)
    workflow.add_node("criterion_evaluation", criterion_evaluation)
    workflow.add_node("report_generation", report_generation)
    
    workflow.set_entry_point("structured_filtering")
    workflow.add_edge("structured_filtering", "evidence_retrieval")
    workflow.add_edge("evidence_retrieval", "criterion_evaluation")
    workflow.add_edge("criterion_evaluation", "report_generation")
    workflow.add_edge("report_generation", END)
    
    return workflow.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Type2-Diabetes-Trial-Agent-Dataset.json", help="Path to JSON dataset")
    parser.add_argument("--patient_index", type=int, default=0, help="Index of patient to process")
    args = parser.parse_args()
    
    data = load_data(args.data)
    patients = data.get("patients", [])
    trials = data.get("trials", [])
    
    if not patients:
        print("No patients found in the dataset.")
        exit(1)
        
    selected_patient = patients[args.patient_index]
    patient_id = selected_patient.get("id", f"PT_{args.patient_index}")
    
    print(f"Starting pipeline for patient {patient_id}...")
    
    app = build_graph()
    final_state = app.invoke({
        "patient_id": patient_id,
        "patient": selected_patient,
        "trials": trials,
        "filtered_trials": [],
        "evidence": {},
        "evaluations": {},
        "report": ""
    })
    
    print("\n\n" + final_state["report"])
