import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import json
from src.agent.agent import get_assistant
from src.evaluation.evaluator import SystemEvaluator
from src.config import get_settings, validate_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
validate_config()
settings = get_settings()

def save_baseline():
    assistant = get_assistant()
    evaluator = SystemEvaluator(assistant)
    
    report = evaluator.run_full_evaluation(settings.golden_dataset_path)

    baseline_path = settings.reports_dir / "baseline_metrics.json"

    # Reutiliza failed_case_ids que ya calcula run_full_evaluation (fuente única).
    baseline_data = {
        "metrics": report["metrics"],
        "failed_case_ids": report.get("failed_case_ids", [])
    }

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Baseline metrics saved to {baseline_path}")

if __name__ == "__main__":
    save_baseline()
