import json
import time
from typing import List, Dict, Any
from pathlib import Path
from datasets import Dataset
# Compat shim: debe ejecutarse ANTES de importar ragas (ver _ragas_compat.py).
from src.evaluation import _ragas_compat  # noqa: F401
from ragas import evaluate
# Usamos las métricas legacy: son instancias de ragas.metrics.base.Metric, lo que la
# función evaluate() espera. Los nombres en ragas.metrics.collections son MÓDULOS
# (no métricas) en 0.4.3, incompatibles con esta API. La deprecación recién se hace
# efectiva en ragas v1.0.
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


def _case_failed(r: Dict[str, Any]) -> bool:
    """Un caso falla si hubo error o si no usó todas las herramientas esperadas."""
    if r.get("error"):
        return True
    expected = set(r.get("expected_tools", []))
    used = set(r.get("tools_used", []))
    return not expected.issubset(used)


def failed_case_ids(results: List[Dict[str, Any]]) -> List[str]:
    """IDs de los casos fallidos (clave estable para comparación de regresiones)."""
    return [r.get("case_id") for r in results if _case_failed(r)]


def _mean_score(value: Any) -> float:
    """Reduce el resultado de una métrica RAGAS a un float.

    ragas 0.4.x devuelve una lista de scores por fila (y puede contener NaN o
    np.float64). Promediamos los valores válidos; si no hay ninguno, 0.0.
    """
    import math

    if isinstance(value, (list, tuple)):
        valores = []
        for v in value:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(f):
                valores.append(f)
        return sum(valores) / len(valores) if valores else 0.0

    try:
        f = float(value)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


class SystemEvaluator:
    def __init__(self, agent):
        self.agent = agent
        self.llm = ChatOpenAI(
            model=settings.evaluation_model,
            openai_api_key=settings.openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )

    def load_golden_dataset(self, path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_answers(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for case in test_cases:
            logger.info(f"Evaluating case {case['id']}: {case['query'][:50]}...")
            try:
                # Reset session for clean evaluation
                self.agent.reset_session("eval_session")
                
                start_time = time.time()
                response = self.agent.query(case['query'], session_id="eval_session")
                latency = time.time() - start_time
                
                results.append({
                    "case_id": case['id'],
                    "question": case['query'],
                    "answer": response["answer"],
                    "tools_used": response["tools_used"],
                    "sources": response["sources"],
                    "contexts": response.get("contexts", []),
                    "expected_tools": case["expected_tools"],
                    "ground_truth": case.get("ground_truth", ""),
                    "category": case.get("category", "general"),
                    "difficulty": case.get("difficulty", "medium"),
                    "latency": latency,
                    "error": response.get("error", False)
                })
            except Exception as e:
                logger.error(f"Error evaluating case {case['id']}: {e}")
                results.append({
                    "case_id": case['id'],
                    "question": case['query'],
                    "answer": "",
                    "error": True,
                    "error_message": str(e),
                    "expected_tools": case["expected_tools"],
                    "category": case.get("category", "general")
                })
        return results

    def calculate_tool_accuracy(self, results: List[Dict[str, Any]]) -> float:
        valid_cases = [r for r in results if not r.get("error")]
        if not valid_cases:
            return 0.0
            
        correct = 0
        for r in valid_cases:
            expected = set(r["expected_tools"])
            used = set(r["tools_used"])
            if expected.issubset(used):
                correct += 1
                
        return correct / len(valid_cases)

    def calculate_error_rate(self, results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        errors = sum(1 for r in results if r.get("error"))
        return errors / len(results)

    def evaluate_with_ragas(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        # Filter RAG cases
        rag_results = [r for r in results if "search_documents" in r["expected_tools"] and not r.get("error")]
        
        if not rag_results:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}
            
        # RAGAS necesita el TEXTO real recuperado como contexto. Usamos r["contexts"]
        # (propagado desde el agente). Fallback a sources solo si no hay contexto, para
        # no romper el cálculo, aunque degrada la métrica.
        data = {
            "question": [r["question"] for r in rag_results],
            "answer": [r["answer"] for r in rag_results],
            "contexts": [
                r.get("contexts") or r.get("sources", []) for r in rag_results
            ],
            "ground_truth": [r["ground_truth"] for r in rag_results]
        }

        try:
            dataset = Dataset.from_dict(data)
            score = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=self.llm,
                embeddings=self.embeddings
            )
            # ragas 0.4.x devuelve UNA LISTA de scores por fila (p.ej. [1.0]),
            # no un float agregado. Promediamos a un float, ignorando NaN.
            return {
                "faithfulness": _mean_score(score["faithfulness"]),
                "answer_relevancy": _mean_score(score["answer_relevancy"]),
            }
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    def run_full_evaluation(self, dataset_path: Path) -> Dict[str, Any]:
        test_cases = self.load_golden_dataset(dataset_path)
        results = self.generate_answers(test_cases)
        
        tool_accuracy = self.calculate_tool_accuracy(results)
        error_rate = self.calculate_error_rate(results)
        ragas_scores = self.evaluate_with_ragas(results)
        
        avg_latency = sum(r.get("latency", 0) for r in results) / len(results) if results else 0

        report = {
            "metrics": {
                "tool_accuracy": tool_accuracy,
                "error_rate": error_rate,
                "faithfulness": ragas_scores["faithfulness"],
                "answer_relevancy": ragas_scores["answer_relevancy"],
                "avg_latency": avg_latency
            },
            "results": results,
            # IDs de casos fallidos: necesario para que compare_evaluations.py detecte
            # regresiones (ERR-009). Un caso falla si hay error o si no usó las tools esperadas.
            "failed_case_ids": failed_case_ids(results),
            "pass_fail": (
                ragas_scores["faithfulness"] >= 0.90 and
                tool_accuracy >= 0.90 and
                error_rate < 0.10
            )
        }
        return report

    def generate_markdown_report(self, report: Dict[str, Any], output_path: Path):
        metrics = report["metrics"]
        results = report["results"]
        
        md = f"# Reporte de Evaluación del Sistema\n\n"
        md += f"## Métricas Generales\n"
        md += f"- **Tool Accuracy:** {metrics['tool_accuracy']:.2%}\n"
        md += f"- **Error Rate:** {metrics['error_rate']:.2%}\n"
        md += f"- **Faithfulness (RAGAS):** {metrics['faithfulness']:.2%}\n"
        md += f"- **Answer Relevancy (RAGAS):** {metrics['answer_relevancy']:.2%}\n"
        md += f"- **Latencia Promedio:** {metrics['avg_latency']:.2f}s\n\n"
        
        status = "✅ PASSED" if report["pass_fail"] else "❌ FAILED"
        md += f"## Estado Final: {status}\n\n"
        
        md += "## Casos Fallidos (Top 5)\n"
        failed_cases = [r for r in results if _case_failed(r)]
        
        if not failed_cases:
            md += "No se encontraron fallos.\n"
        else:
            md += "| ID | Pregunta | Error | Herramientas Esperadas | Herramientas Usadas |\n"
            md += "|---|---|---|---|---|\n"
            for case in failed_cases[:5]:
                md += f"| {case['case_id']} | {case['question']} | {case.get('error', False)} | {case.get('expected_tools')} | {case.get('tools_used')} |\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"Markdown report generated at {output_path}")
