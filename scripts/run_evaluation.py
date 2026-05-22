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

def run():
    assistant = get_assistant()
    evaluator = SystemEvaluator(assistant)
    
    dataset_path = settings.golden_dataset_path
    report_md_path = settings.reports_dir / "baseline_evaluation.md"
    report_json_path = settings.reports_dir / "baseline_evaluation.json"
    
    logger.info(f"Running full evaluation on {dataset_path}...")
    report = evaluator.run_full_evaluation(dataset_path)
    
    # Print metrics
    metrics = report["metrics"]
    print("\n--- Resultados de la Evaluación ---")
    print(f"Tool Accuracy: {metrics['tool_accuracy']:.2%} {'✅' if metrics['tool_accuracy'] >= 0.90 else '❌'}")
    print(f"Error Rate: {metrics['error_rate']:.2%} {'✅' if metrics['error_rate'] < 0.10 else '❌'}")
    print(f"Faithfulness: {metrics['faithfulness']:.2%} {'✅' if metrics['faithfulness'] >= 0.90 else '❌'}")
    print(f"Answer Relevancy: {metrics['answer_relevancy']:.2%} {'✅' if metrics['answer_relevancy'] >= 0.85 else '❌'}")
    print(f"Avg Latency: {metrics['avg_latency']:.2f}s")
    
    # Save reports
    evaluator.generate_markdown_report(report, report_md_path)
    
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Reports saved to {settings.reports_dir}")

if __name__ == "__main__":
    run()
