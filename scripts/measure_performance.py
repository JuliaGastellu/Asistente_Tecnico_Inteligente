import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import time
import statistics
from src.config import validate_config
from src.agent.agent import get_assistant
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
validate_config()

def measure():
    assistant = get_assistant()
    queries = [
        "¿Cómo crear un endpoint en FastAPI?",
        "Calcula el 15% de 2500 y busca el clima en Madrid.",
        "Dime la diferencia entre LangChain y LlamaIndex según la web."
    ]
    
    runs = 5
    all_results = {}
    
    print(f"Midiendo performance ({runs} runs por query)...\n")
    
    for query in queries:
        latencies = []
        print(f"Query: {query}")
        for i in range(runs):
            assistant.cache.clear() # Clear cache to measure real execution
            start = time.time()
            assistant.query(query)
            lat = time.time() - start
            latencies.append(lat)
            print(f"  Run {i+1}: {lat:.2f}s")
            
        all_results[query] = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "p95": sorted(latencies)[int(0.95 * len(latencies))]
        }
        
    print("\n--- Resultados de Latencia ---")
    for q, res in all_results.items():
        print(f"\nQuery: {q[:50]}...")
        print(f"  Media: {res['mean']:.2f}s")
        print(f"  Mediana: {res['median']:.2f}s")
        print(f"  P95: {res['p95']:.2f}s")
        print(f"  Min/Max: {res['min']:.2f}s / {res['max']:.2f}s")

    # Estimated cost breakdown (dummy values for example)
    print("\n--- Costo Estimado por Componente ---")
    print("  RAG Retrieval: $0.0001")
    print("  LLM Reasoning (GPT-4): $0.01")
    print("  LLM Mini (Rerank/Comp): $0.001")
    print("  Total estimado por query compleja: ~$0.0111")

if __name__ == "__main__":
    measure()
