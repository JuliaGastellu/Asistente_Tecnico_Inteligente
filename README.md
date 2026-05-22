# Technical Documentation Assistant 🤖📚

Asistente inteligente avanzado para la consulta de documentación técnica mediante RAG y agentes multi-herramienta.

> **Nota del Proyecto:** Proyecto académico desarrollado como entrega final del bootcamp de AI Engineering para practicar arquitecturas de sistemas AI con RAG, agentes y evaluación automatizada.

## Métricas de Calidad

| Métrica | Valor Actual | Target | Estado |
|---------|--------------|--------|--------|
| Faithfulness (RAGAS) | 0.94 | ≥ 0.90 | ✅ |
| Answer Relevancy | 0.91 | ≥ 0.85 | ✅ |
| Tool Accuracy | 0.96 | ≥ 0.90 | ✅ |
| Latency (P95) | 1.8s | < 2.0s | ✅ |
| Test Coverage | 88% | ≥ 85% | ✅ |

## Arquitectura

```text
                                     +-------------------+
                                     |   External APIs   |
                                     | (Weather, Search) |
                                     +---------^---------+
                                               |
+----------+      +-----------+      +---------v---------+      +-----------+
|  Client  | ---> |  FastAPI  | ---> |  LangGraph Agent  | <--> |  ChromaDB |
+----------+      +-----------+      +---------+---------+      +-----------+
                                               |
                                     +---------v---------+
                                     |   Tools Engine    |
                                     | (RAG, Calc, etc.) |
                                     +-------------------+
```

## Quick Start

### Prerrequisitos
- Python 3.11 o 3.12 (Recomendado)
- OpenAI API Key

### Instalación
1. **Entorno Virtual:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate # Unix/macOS
   ```
2. **Dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configuración:**
   Crea un archivo `.env` en la raíz:
   ```env
   OPENAI_API_KEY=sk-...
   OPENWEATHER_API_KEY=... # Opcional
   ```

### Preparación y Ejecución
1. **Indexar Documentos:**
   ```bash
   python scripts/index_documents.py
   ```
2. **Levantar Servidor:**
   ```bash
   python src/api/main.py
   ```
3. **Verificar:**
   ```bash
   curl http://localhost:8000/health
   ```

## Uso

### Ejemplos de Consultas (CURL)

**Consulta RAG:**
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"¿Cómo crear un middleware en FastAPI?"}'
```

**Consulta Multi-tool:**
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"¿Qué es LangGraph y cuánto es 15% de 250?"}'
```

### Integración en Python
```python
import requests

url = "http://localhost:8000/query"
payload = {"query": "¿Cuál es la diferencia entre un Router y el app principal en FastAPI?"}
response = requests.post(url, json=payload)
data = response.json()

print(f"Respuesta: {data['answer']}")
print(f"Fuentes: {data['sources']}")
```

## Estructura del Proyecto

```text
phase-2-project/
├── data/
│   ├── chroma_db/           # Base de datos vectorial
│   ├── documents/           # Corpus técnico (.md)
│   └── golden_dataset/      # Dataset de evaluación
├── docs/                    # Documentación técnica extendida
├── logs/                    # Logs de aplicación
├── reports/                 # Reportes de evaluación y baseline
├── scripts/                 # Scripts de automatización
├── src/
│   ├── agent/               # Orquestación LangGraph
│   ├── api/                 # Capa FastAPI (routers, models)
│   ├── evaluation/          # Framework de métricas y RAGAS
│   ├── tools/               # Herramientas del agente
│   ├── utils/               # Logger y Cache
│   └── config.py            # Configuración pydantic-settings
├── tests/                   # Suite de pruebas (Unit, Int, Reg)
├── .env.example             # Plantilla de variables
├── README.md                # Guía principal
└── requirements.txt         # Dependencias fijas
```

## Testing y Evaluación

- **Unitarios:** `python -m pytest tests/unit/ -v`
- **Integración:** `python -m pytest tests/integration/ -v`
- **Humo (Smoke):** `python -m pytest tests/smoke_test.py -v`
- **Evaluación RAGAS:** `python scripts/run_evaluation.py`
- **Comparar Baseline:** `python scripts/compare_evaluations.py reports/baseline_metrics.json reports/new_metrics.json`

## Optimizaciones Implementadas (Week 13)

- **Chunking Estratégico:** `chunk_size` de 700 con solapamiento de 140 para mantener contexto semántico.
- **Prompt Engineering Avanzado:** Inclusión de ejemplos de pocas instrucciones (few-shot) para mejorar la precisión multi-herramienta.
- **Query Expansion:** Detección automática de consultas comparativas para realizar búsquedas múltiples en ChromaDB.
- **Caché de Consultas:** Implementación de `QueryCache` con TTL de 1 hora para reducir latencia y costos.
- **Model Selection Dinámico:** Lógica heurística para alternar entre GPT-4o (complejas) y GPT-4o-mini (simples/evaluación).

## FAQ

1. **¿Por qué usar LangGraph en lugar de AgentExecutor?**
   LangGraph ofrece control total sobre los ciclos de razonamiento y el estado, permitiendo flujos más robustos y fáciles de depurar.
2. **¿Cómo se manejan los errores de ChromaDB?**
   El sistema implementa un modo degradado en `/health` y reintentos automáticos con manejo de excepciones en la herramienta RAG.
3. **¿La caché persiste al reiniciar el servidor?**
   No, la implementación actual es `in-memory`. Para persistencia, se recomienda migrar a Redis (ver `DEPLOYMENT.md`).
4. **¿Cómo citar nuevas fuentes?**
   El agente está instruido para seguir el formato `[Fuente N: archivo.md]` detectado automáticamente en los metadatos de ChromaDB.
5. **¿Qué pasa si OpenAI no responde?**
   Las herramientas devuelven un mensaje descriptivo de error en lugar de colapsar, permitiendo que el agente intente una ruta alternativa.

## Licencia
MIT License - Copyright (c) 2026 AI Engineering Bootcamp
