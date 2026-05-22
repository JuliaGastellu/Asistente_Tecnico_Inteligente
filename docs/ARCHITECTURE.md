# Arquitectura del Sistema 🏗️

Este documento detalla las decisiones de diseño y la estructura interna del **Technical Documentation Assistant**.

## Principios de Diseño
- **Modularidad:** Cada herramienta y componente de la API está desacoplado, facilitando el mantenimiento y la extensión.
- **Observabilidad:** Logging centralizado y estructurado para trazabilidad completa de cada consulta.
- **Confiabilidad:** Manejo de excepciones en el nivel de herramienta para evitar fallos en cascada del agente.
- **Performance:** Pipeline RAG optimizado con re-ranking y caché de consultas para minimizar latencia.
- **Testabilidad:** Arquitectura diseñada para facilitar el mocking de servicios externos y pruebas automatizadas.

## Componentes del Sistema

### 1. Capa de API (FastAPI)
- **Responsabilidad:** Exponer endpoints REST, validar payloads y gestionar la inyección de dependencias.
- **Archivos Clave:** `src/api/main.py`, `src/api/routers/`.
- **Flujo:** Recibe el `QueryRequest`, valida con Pydantic e invoca al singleton del Agente.

### 2. Orquestador del Agente (LangGraph)
- **Responsabilidad:** Gestionar el ciclo de razonamiento (Reasoning Loop) y el estado de la conversación.
- **Archivos Clave:** `src/agent/agent.py`.
- **Flujo:** Decide si usar herramientas basadas en la intención del usuario y consolida la respuesta final.

### 3. Herramienta RAG (Optimized Retriever)
Pipeline de 5 pasos para máxima relevancia:
```text
1. Expansion -> 2. Retrieval -> 3. Reranking -> 4. Compression -> 5. Deduplication
```
- **Responsabilidad:** Recuperar contexto técnico preciso de ChromaDB.
- **Archivos Clave:** `src/tools/rag_tool.py`.

### 4. External Tools Engine
- **Responsabilidad:** Ejecutar lógica no-RAG (cálculos, clima, búsqueda web).
- **Archivos Clave:** `src/tools/calculator_tool.py`, `src/tools/weather_tool.py`.

### 5. Framework de Evaluación
- **Responsabilidad:** Medir la calidad de las respuestas usando el dataset "golden" y métricas RAGAS.
- **Archivos Clave:** `src/evaluation/evaluator.py`.

## Flujo de Datos (End-to-End)

1. **Recepción:** El cliente envía una consulta al endpoint `/query`.
2. **Caché Check (Path Rápido):** 
   - Se genera un hash MD5 de la consulta.
   - Si existe en `QueryCache` y no ha expirado -> **Cache HIT**: Se devuelve el resultado inmediatamente.
3. **Procesamiento (Path Estándar):**
   - **Model Selector:** Se estima la complejidad para elegir entre GPT-4o o GPT-4o-mini.
   - **Graph Execution:** Se inicia el grafo de LangGraph.
   - **Tool Calling:** El modelo solicita herramientas (RAG, Calc, etc.) si es necesario.
   - **Observation:** Las herramientas devuelven strings con el resultado/contexto.
4. **Respuesta:** El modelo genera la respuesta final citando fuentes.
5. **Finalización:** Se guarda en caché y se devuelve el JSON al cliente.

## Decisiones Tecnológicas
- **FastAPI:** Por su soporte nativo de asincronía y generación automática de OpenAPI.
- **LangGraph:** Supera a `AgentExecutor` al permitir definir grafos de estado cíclicos y personalizados.
- **ChromaDB:** Base de datos vectorial ligera y persistente ideal para prototipado rápido y entornos locales.
- **GPT-4o + GPT-4o-mini:** Estrategia de costo/beneficio usando el modelo potente solo para tareas complejas.

## Modos de Fallo y Mitigación
- **ChromaDB Inaccesible:** El health check marca el sistema como "degraded"; las herramientas RAG devuelven error descriptivo sin tumbar la API.
- **Límite de Rate de OpenAI:** El agente implementa reintentos exponenciales y manejo de excepciones de API.
- **Leak de Memoria (Caché):** El `QueryCache` tiene un límite de tamaño y TTL para evitar el crecimiento indefinido.

## Roadmap de Escalabilidad
- **Fase 1 (<100 RPS):** Arquitectura actual con caché in-memory y persistencia local de ChromaDB.
- **Fase 2 (100-1000 RPS):** Migrar caché a Redis, persistencia de ChromaDB en volumen Docker compartido y balanceador de carga.
- **Fase 3 (>1000 RPS):** Despliegue en Kubernetes (K8s), ChromaDB en modo cliente-servidor (cloud), y base de datos relacional para historial de chats.
