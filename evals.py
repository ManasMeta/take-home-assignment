import json

def evaluate_unknown_bias(evaluations: dict) -> float:
    """
    Failure Mode Hypothesis: 
    The system might be overly confident and assign "NOT_SUPPORTED" or "SUPPORTED"
    when the data is actually missing, instead of using "UNKNOWN".
    
    Metric: Unknown Avoidance Rate
    Calculated as: (Number of false NOT_SUPPORTED / Total missing data fields)
    
    A baseline should ideally be 0.0 (the system always identifies missing data as UNKNOWN).
    If it's high, the LLM is hallucinating answers instead of admitting lack of evidence.
    """
    false_conclusions = 0
    total_missing = 0
    
    # In a real eval suite, we would load ground truth labels indicating which
    # fields are deliberately omitted in the synthetic patient data.
    # Mocking the evaluation:
    for trial_evals in evaluations.values():
        # Let's pretend we know 'hba1c' was missing for this patient
        if "hba1c" in trial_evals:
            total_missing += 1
            if trial_evals["hba1c"]["state"] != "UNKNOWN":
                false_conclusions += 1
                
    if total_missing == 0:
        return 0.0
    return false_conclusions / total_missing

if __name__ == "__main__":
    print("Running Custom Evaluation Suite...")
    # Mocking evaluations from the agent
    mock_evaluations = {
        "NCT00000001": {
            "hba1c": {"state": "NOT_SUPPORTED", "reason": "Assumed failed."} # Hallucination!
        },
        "NCT00000002": {
            "hba1c": {"state": "UNKNOWN", "reason": "Data not found."} # Correct
        }
    }
    
    score = evaluate_unknown_bias(mock_evaluations)
    print(f"Unknown Avoidance Rate (Lower is better): {score:.2f}")
    print("Limitations: This metric requires a heavily labelled synthetic dataset where missingness is explicitly mapped to ground truth to calculate accurately.")
