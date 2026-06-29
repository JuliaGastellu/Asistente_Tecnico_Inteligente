import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import json
from datetime import datetime
from src.agent.agent import get_assistant
from src.evaluation.evaluator import SystemEvaluator
from src.config import get_settings, validate_config
from src.utils.logger import setup_logger

# La consola de Windows usa cp1252 por defecto; al redirigir stdout (background) los
# emojis crashean (UnicodeEncodeError). Forzamos UTF-8 si es posible (ERR-025).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = setup_logger(__name__)
validate_config()
settings = get_settings()


def _mark(ok: bool) -> str:
    """Marcador ASCII-safe para no depender de la codificación de la consola."""
    return "[PASS]" if ok else "[FAIL]"


def run():
    assistant = get_assistant()
    evaluator = SystemEvaluator(assistant)

    dataset_path = settings.golden_dataset_path
    # Timestamp en el nombre para NO sobrescribir corridas anteriores (histórico).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_md_path = settings.reports_dir / f"evaluation_{ts}.md"
    report_json_path = settings.reports_dir / f"evaluation_{ts}.json"

    logger.info(f"Running full evaluation on {dataset_path}...")
    report = evaluator.run_full_evaluation(dataset_path)

    # GUARDAR PRIMERO: una falla al imprimir (p.ej. encoding) no debe perder el report
    # tras una evaluación costosa (ERR-025).
    evaluator.generate_markdown_report(report, report_md_path)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    logger.info(f"Reports saved to {settings.reports_dir}")

    # Resumen por consola (ASCII-safe).
    metrics = report["metrics"]
    print("\n--- Resultados de la Evaluacion ---")
    print(f"Tool Accuracy: {metrics['tool_accuracy']:.2%} {_mark(metrics['tool_accuracy'] >= 0.90)}")
    print(f"Error Rate: {metrics['error_rate']:.2%} {_mark(metrics['error_rate'] < 0.10)}")
    print(f"Faithfulness: {metrics['faithfulness']:.2%} {_mark(metrics['faithfulness'] >= 0.90)}")
    print(f"Answer Relevancy: {metrics['answer_relevancy']:.2%} {_mark(metrics['answer_relevancy'] >= 0.85)}")
    print(f"Avg Latency: {metrics['avg_latency']:.2f}s")
    print(f"Reports: {report_json_path.name} / {report_md_path.name}")


if __name__ == "__main__":
    run()
