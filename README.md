# Technical Documentation Assistant 

Este proyecto es una implementación avanzada de un **Asistente Técnico Inteligente** diseñado para el bootcamp de AI Engineering (Weeks 12-13). El sistema utiliza una arquitectura RAG (Retrieval-Augmented Generation) optimizada, agentes multi-herramienta y un framework de evaluación automatizada.

## Arquitectura del Sistema

```text
+----------------+      +-------------------+      +-------------------+
|   User Query   | ---> |  Model Selector   | ---> |   Agent Executor  |
+----------------+      +---------+---------+      +---------+---------+
                                  |                          |
                                  v                          v
                        +-------------------+      +-------------------+
                        | GPT-4 (Complex)   |      | Tools:            |
                        | GPT-4o-mini (Simp)|      | - RAG Tool        |
                        +-------------------+      | - Calculator      |
                                                   | - Weather API     |
                                                   | - Web Search      |
                                                   +---------+---------+
                                                             |
                                                             v
                                                   +-------------------+
                                                   |    ChromaDB       |
                                                   +-------------------+
```

## Características Principales

- **RAG Avanzado:** Pipeline con k-adaptativo, re-ranking con LLM, compresión de contexto y expansión de queries comparativas.
- **Agentes Multi-herramienta:** Orquestación con `create_openai_functions_agent` para manejar consultas complejas que requieren múltiples pasos.
- **Optimización Data-Driven:** Scripts para medir performance, latencia y costos.
- **Evaluación con RAGAS:** Métricas automáticas de `faithfulness` y `answer_relevancy`.
- **API Robusta:** Construida con FastAPI, inyección de dependencias y validación estricta con Pydantic.

## Instalación

1. Clonar el repositorio.
2. Crear un entorno virtual: `python -m venv venv`.
3. Activar el entorno:
   - Windows: `.\venv\Scripts\activate`
   - Unix/macOS: `source venv/bin/activate`
4. Instalar dependencias: `pip install -r requirements.txt`.
5. Configurar el archivo `.env` basándose en `.env.example`.

## Uso

### 1. Indexar Documentos
Primero, prepara la base de conocimientos:
```bash
python scripts/index_documents.py
```

### 2. Levantar la API
```bash
python src/api/main.py
```
Accede a la documentación interactiva en: `http://localhost:8000/docs`

### 3. Ejecutar Evaluación
```bash
python scripts/run_evaluation.py
```

### 4. Guardar y Comparar Baseline
```bash
# Guardar estado actual
python scripts/save_baseline.py
# Comparar con una nueva versión
python scripts/compare_evaluations.py reports/baseline_metrics.json reports/new_metrics.json
```

## Pruebas
Ejecutar la suite de pruebas:
```bash
# Unitarias e Integración (sin APIs externas)
python -m pytest tests/unit/ tests/integration/ -v

# Cobertura
python -m pytest --cov=src --cov-report=term
```

## Métricas Objetivo (KPIs)
- **Faithfulness:** ≥ 0.90
- **Answer Relevancy:** ≥ 0.85
- **Tool Accuracy:** ≥ 0.90
- **Latencia (P95):** < 2.0s
- **Cobertura de Tests:** ≥ 85%

## Estructura del Proyecto
- `src/`: Código fuente (agentes, tools, api, evaluation).
- `data/`: Documentación técnica y dataset de evaluación.
- `scripts/`: Automatización de tareas e indexado.
- `tests/`: Suite completa de pruebas unitarias, integración y regresión.
- `reports/`: Resultados de evaluaciones y comparativas.
