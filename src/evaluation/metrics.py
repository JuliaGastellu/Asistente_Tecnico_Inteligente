from src.config import get_settings

def check_thresholds(report: dict) -> dict:
    metrics = report.get("metrics", {})
    
    # Targets (could be moved to config)
    targets = {
        "tool_accuracy": 0.90,
        "error_rate": 0.10,
        "faithfulness": 0.90,
        "answer_relevancy": 0.85
    }
    
    results = {}
    for metric, target in targets.items():
        value = metrics.get(metric, 0)
        passed = False
        
        if metric == "error_rate":
            passed = value <= target
        else:
            passed = value >= target
            
        results[metric] = {
            "value": value,
            "target": target,
            "passed": passed
        }
        
    return results
