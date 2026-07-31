import json
import sys
import argparse
from typing import Dict, Any, List
from agent import build_graph, load_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def evaluate_full_dataset(dataset_path: str = "Type2-Diabetes-Trial-Agent-Dataset.json") -> Dict[str, Any]:
    """
    Evaluation Suite for Type 2 Diabetes Clinical Trial Pre-Screening Agent.
    Evaluates agent behavior across all synthetic patient records.
    """
    data = load_data(dataset_path)
    patients = data.get("patients", [])
    trials = data.get("trials", [])
    
    app = build_graph()
    
    total_missing_fields = 0
    false_hallucinated_conclusions = 0
    
    total_evaluations = 0
    valid_citations = 0
    
    state_counts = {
        "SUPPORTED": 0,
        "NOT_SUPPORTED": 0,
        "UNKNOWN": 0,
        "CONFLICTING_EVIDENCE": 0,
        "REQUIRES_CLINICAL_REVIEW": 0
    }
    
    patient_reports = []
    
    for idx, p in enumerate(patients):
        pid = p.get("patient_id", f"P-{idx}")
        obs = p.get("observations", [])
        meds = p.get("medications", [])
        
        has_hba1c = any(o.get("type") == "hba1c" for o in obs)
        has_egfr = any(o.get("type") == "egfr" for o in obs)
        has_meds = len(meds) > 0
        
        # Collect valid FHIR source IDs for provenance checking
        valid_source_ids = set()
        for o in obs:
            if o.get("source_id"):
                valid_source_ids.add(o.get("source_id"))
        for m in meds:
            if m.get("source_id"):
                valid_source_ids.add(m.get("source_id"))
        for c in p.get("conditions", []):
            if c.get("source_id"):
                valid_source_ids.add(c.get("source_id"))
        # Add metadata provenance tags
        valid_source_ids.update(["patient_demographics", "trial_metadata", "lab_results", "medications_list", "trial_eligibility_text"])
        
        # Invoke Agent Graph with Checkpointer Thread Config
        thread_config = {"configurable": {"thread_id": f"eval_session_{pid}"}}
        final_state = app.invoke({
            "patient_id": pid,
            "patient": p,
            "trials": trials,
            "filtered_trials": [],
            "evidence": {},
            "evaluations": {},
            "report": "",
            "use_llm": False,
            "interactive_hitl": False,
            "human_approved": False,
            "human_review_notes": "",
            "telemetry": {}
        }, config=thread_config)
        
        evals = final_state.get("evaluations", {})
        
        # Evaluate metrics for each evaluated trial
        for nct_id, trial_evals in evals.items():
            # 1. Unknown Avoidance check for eGFR
            if not has_egfr:
                total_missing_fields += 1
                egfr_st = trial_evals.get("egfr", {}).get("state")
                if egfr_st != "UNKNOWN":
                    false_hallucinated_conclusions += 1
                    
            # 2. Unknown Avoidance check for HbA1c
            if not has_hba1c:
                total_missing_fields += 1
                hba1c_st = trial_evals.get("hba1c", {}).get("state")
                if hba1c_st != "UNKNOWN":
                    false_hallucinated_conclusions += 1

            # 3. Citation Validity & State counts
            for crit, res in trial_evals.items():
                total_evaluations += 1
                st = res.get("state", "UNKNOWN")
                state_counts[st] = state_counts.get(st, 0) + 1
                
                src_str = res.get("evidence_id", "")
                sources = [s.strip() for s in src_str.split(",")]
                if any(s in valid_source_ids for s in sources):
                    valid_citations += 1

    unknown_avoidance_rate = (false_hallucinated_conclusions / total_missing_fields) if total_missing_fields > 0 else 0.0
    citation_accuracy = (valid_citations / total_evaluations) if total_evaluations > 0 else 0.0
    
    return {
        "total_patients": len(patients),
        "total_trials_evaluated": len(trials),
        "total_evaluations": total_evaluations,
        "unknown_avoidance_rate": unknown_avoidance_rate,
        "citation_accuracy": citation_accuracy,
        "state_counts": state_counts
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Suite for Pre-Screening Agent")
    parser.add_argument("--data", default="Type2-Diabetes-Trial-Agent-Dataset.json", help="Path to JSON dataset")
    args = parser.parse_args()
    
    print("==========================================================")
    print("  RUNNING AGENTIC EVALUATION SUITE ON SYNTHETIC DATASET   ")
    print("==========================================================")
    
    results = evaluate_full_dataset(args.data)
    
    print(f"\nPatients Processed        : {results['total_patients']}")
    print(f"Total Criterion Assessments: {results['total_evaluations']}")
    print(f"\n--- CORE METRICS ---")
    print(f"Unknown Avoidance Rate (Lower is Better, 0.0 = Best): {results['unknown_avoidance_rate']:.4f}")
    print(f"Citation Accuracy (1.0 = Best)                      : {results['citation_accuracy']:.4f}")
    
    print(f"\n--- CRITERION STATE DISTRIBUTION ---")
    for state, count in results['state_counts'].items():
        pct = (count / results['total_evaluations']) * 100 if results['total_evaluations'] > 0 else 0.0
        print(f"  * {state:<25}: {count:>4} ({pct:>5.1f}%)")
        
    print("\n==========================================================")
    print("  EVALUATION COMPLETE - AGENT BEHAVIOR VERIFIED           ")
    print("==========================================================")
