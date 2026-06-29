# Technical Documentation Assistant

Asistente inteligente avanzado para la consulta de documentación técnica mediante RAG y agentes multi-herramienta.

> **Nota del Proyecto:** Proyecto académico desarrollado como entrega final del bootcamp de AI Engineering para practicar arquitecturas de sistemas AI con RAG, agentes y evaluación automatizada.

## 🚀 Producción

**URL de la API:** https://tu-app.onrender.com  <!-- reemplazar por la URL real de Render -->

| Endpoint | Descripción |
|----------|-------------|
| GET `/health` | Estado del sistema (agente + Pinecone) |
| POST `/query` | Consulta al agente |

## Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| API | FastAPI + Uvicorn |
| Agente | LangGraph |
| Vectores | **Pinecone** (cloud) — fallback local: ChromaDB |
| LLM | **OpenRouter** (gateway) — fallback: OpenAI directo |
| Embeddings | OpenAI (`text-embedding-3-small`) |
| Evaluación | RAGAS |
| Config | pydantic-settings |
| Tests prod | `test_production.py` |
| Deploy | Render |

> El backend de vectores y el LLM se eligen por configuración: si `PINECONE_API_KEY` /
> `OPENROUTER_API_KEY` están definidas se usa el stack cloud; si no, el proyecto sigue
> funcionando en local con ChromaDB + OpenAI directo.

## Testing de Producción

Editá `BASE_URL` en `test_production.py` con la URL real de Render y ejecutá:

```bash
python test_production.py
```

## Métricas de Calidad

| Métrica | Valor Actual (baseline v2) | Target | Estado |
|---------|----------------------------|--------|--------|
| Tool Accuracy | 1.00 | ≥ 0.90 | ✅ |
| Error Rate | 0.00 | < 0.10 | ✅ |
| Answer Relevancy (RAGAS) | 0.86 | ≥ 0.85 | ✅ |
| Faithfulness (RAGAS) | 0.35 | ≥ 0.90 | ❌ |
| Latency (promedio) | 4.01s | < 2.0s | ❌ |
| Test Coverage | no medido en baseline v2 | ≥ 85% | — |

> **Nota:** Métricas medidas con el golden dataset real (v2, 15 casos).
> Faithfulness y Answer Relevancy requieren dataset con ground_truth del corpus real.

> **Nota metodológica:** Faithfulness medida contra corpus de 60 documentos técnicos.
> El techo actual (~35%) refleja que el agente enriquece respuestas con conocimiento
> propio (Starlette, Pydantic, OpenAPI) no presente en los chunks recuperados.
> Para elevar esta métrica se requiere ampliar el corpus con documentación más detallada.

## Arquitectura

```text
                                     +-------------------+
                                     |   External APIs   |
                                     | (Weather, Search) |
                                     +---------^---------+
                                               |
+----------+      +-----------+      +---------v---------+      +-----------------+
|  Client  | ---> |  FastAPI  | ---> |  LangGraph Agent  | <--> | Pinecone (cloud)|
+----------+      +-----------+      +---------+---------+      +-----------------+
                                               |              (fallback: ChromaDB)
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
   Copiá la plantilla y completá los valores reales:
   ```bash
   cp .env.example .env
   ```
   ```env
   OPENAI_API_KEY=sk-...
   OPENWEATHER_API_KEY=... # Opcional
   ENVIRONMENT=local
   ```
   > ⚠️ **Nota:** `OPENWEATHER_API_KEY` es una clave de
   > [OpenWeatherMap](https://openweathermap.org/api), **distinta** de la de OpenAI.
   > No reutilices la `OPENAI_API_KEY` aquí: la herramienta de clima fallaría con 401.

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

## Endpoints

| Método | Path | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado de los componentes (agente, Pinecone/ChromaDB) |
| `POST` | `/query` | Consulta al agente multi-herramienta |
| `GET` | `/test-rag?query=...` | Test directo de recuperación vectorial |
| `GET` | `/docs` | UI interactiva Swagger |

## Uso

### Ejemplos de Consultas (CURL)

**Consulta RAG:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Cómo crear un middleware en FastAPI?"}'
```

**Consulta Multi-tool:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Qué es LangGraph y cuánto es 15% de 250?"}'
```

**Con session_id (conversación continua):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Cómo añadir autenticación?", "session_id":"user_123"}'
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
│   ├── chroma_db/           # Base de datos vectorial (generada por index_documents.py)
│   ├── documents/           # Corpus técnico (.md) — 60 documentos
│   └── golden_dataset/      # Dataset de evaluación RAGAS
├── docs/                    # Documentación técnica extendida
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── RUNBOOK.md
├── logs/                    # Logs de aplicación (creado automáticamente)
├── reports/                 # Reportes de evaluación y baseline
├── scripts/
│   ├── index_documents.py   # Indexa data/documents/ en ChromaDB
│   ├── run_evaluation.py    # Ejecuta evaluación RAGAS
│   ├── compare_evaluations.py
│   ├── deploy_localstack.py # Deploy completo en LocalStack
│   └── cleanup_localstack.py# Limpieza de recursos LocalStack
├── src/
│   ├── agent/               # Orquestación LangGraph (agent.py, prompts, model_selector)
│   ├── api/                 # FastAPI: main.py, routers/, models/, dependencies.py
│   ├── evaluation/          # Framework RAGAS y métricas custom
│   ├── tools/               # Herramientas: RAG, calculadora, clima, búsqueda web
│   ├── utils/               # Logger (con fallback a /tmp) y QueryCache
│   └── config.py            # Configuración pydantic-settings (patrón Camaleón)
├── tests/                   # Unit, Integration, Smoke
├── requirements.txt         # Dependencias completas (dev + eval)
├── requirements-lambda.txt  # Dependencias de producción (sin ragas/pytest)
└── .env.example             # Plantilla de variables de entorno
```

## Variables de Entorno

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `OPENAI_API_KEY` | Sí | Token de acceso a la API de OpenAI |
| `OPENWEATHER_API_KEY` | No | Token para la herramienta de clima |
| `ENVIRONMENT` | No | `local` (default), `localstack`, o `aws` |

## Despliegue en LocalStack

Requiere Docker y LocalStack corriendo. Ver `docs/DEPLOYMENT.md` para instrucciones completas.

**Inicio rápido:**
```bash
# 1. Iniciar LocalStack con límite de tamaño ampliado (necesario: el ZIP descomprimido supera 250 MB)
#    Community edition:
docker run --rm -d --name localstack-main \
  -p 127.0.0.1:4566:4566 \
  -e LAMBDA_LIMITS_TOTAL_CODE_SIZE=1073741824 \
  localstack/localstack

#    Pro edition (agregar LOCALSTACK_AUTH_TOKEN si es necesario):
docker run --rm -d --name localstack-main \
  -p 127.0.0.1:443:443 \
  -p 127.0.0.1:4510-4560:4510-4560 \
  -p 127.0.0.1:4566:4566 \
  -e LAMBDA_LIMITS_TOTAL_CODE_SIZE=1073741824 \
  localstack/localstack-pro

# 2. Esperar a que esté healthy
docker inspect localstack-main --format='{{.State.Health.Status}}'
# → debe mostrar "healthy"

# 3. Configurar .env
echo "ENVIRONMENT=localstack" >> .env

# 4. Desplegar (el script intenta ampliar el límite via config API antes de deployar)
python scripts/deploy_localstack.py
```

**Si LocalStack ya está corriendo sin el flag de tamaño:**
```bash
# Opción A — Ampliar dinámicamente sin reiniciar (LocalStack Pro)
curl -s -X PATCH http://localhost:4566/_localstack/config \
  -H "Content-Type: application/json" \
  -d '{"LAMBDA_LIMITS_TOTAL_CODE_SIZE": 1073741824}'

# Opción B — Reiniciar el container con el flag
docker inspect localstack-main --format='{{range .Config.Env}}{{println .}}{{end}}'  # anotar env vars
docker stop localstack-main
docker run --rm -d --name localstack-main \
  -p 127.0.0.1:443:443 \
  -p 127.0.0.1:4510-4560:4510-4560 \
  -p 127.0.0.1:4566:4566 \
  -e LAMBDA_LIMITS_TOTAL_CODE_SIZE=1073741824 \
  localstack/localstack-pro   # agregar -e LOCALSTACK_AUTH_TOKEN=... si es necesario
```

## Testing y Evaluación

```bash
# Tests unitarios
python -m pytest tests/unit/ -v

# Tests de integración
python -m pytest tests/integration/ -v

# Smoke test
python -m pytest tests/smoke_test.py -v

# Evaluación RAGAS
python scripts/run_evaluation.py

# Comparar con baseline
python scripts/compare_evaluations.py reports/baseline_metrics.json reports/new_metrics.json
```

## Optimizaciones Implementadas

- **Chunking Estratégico:** `chunk_size=700`, `overlap=140` para mantener contexto semántico.
- **Prompt Engineering:** Few-shot prompting para precisión multi-herramienta.
- **Query Expansion:** Detección automática de consultas comparativas (búsqueda paralela en ChromaDB).
- **Caché de Consultas:** `QueryCache` in-memory con TTL de 1 hora (hash MD5 de query).
- **Model Selection Dinámico:** GPT-4o para consultas complejas, GPT-4o-mini para simples.
- **Pipeline RAG Optimizado:** Expansion → Retrieval → Reranking → Compression → Deduplication.

## FAQ

1. **¿Por qué usar LangGraph en lugar de AgentExecutor?**
   LangGraph ofrece control total sobre los ciclos de razonamiento y el estado, permitiendo flujos más robustos y fáciles de depurar.
2. **¿Cómo se manejan los errores de ChromaDB?**
   El sistema implementa un modo degradado en `/health` y reintentos automáticos con manejo de excepciones en la herramienta RAG.
3. **¿La caché persiste al reiniciar el servidor?**
   No, la implementación actual es `in-memory`. Para persistencia, se recomienda migrar a Redis.
4. **¿Qué pasa si OpenAI no responde?**
   Las herramientas devuelven un mensaje descriptivo de error en lugar de colapsar, permitiendo que el agente intente una ruta alternativa.
5. **¿Por qué el paquete Lambda excluye tantos paquetes?**
   Lambda tiene un límite de 250 MB descomprimido. Se excluyen `onnxruntime`, `datasets`, `grpcio`, etc. que no son necesarios en runtime con OpenAI Embeddings y ChromaDB embedded.

## Licencia
MIT License - Copyright (c) 2026 AI Engineering Bootcamp
