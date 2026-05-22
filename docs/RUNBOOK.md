# Runbook Operacional 📖

Guía de referencia para la operación, mantenimiento y resolución de problemas del sistema.

## Monitoreo de Salud

Endpoint: `GET /health`

**Campos de respuesta:**
- `status`: "healthy" (todo OK) o "degraded" (ChromaDB no responde).
- `components.agent`: Estado del orquestador.
- `components.chromadb`: Estado de la base vectorial.
- `tools_count`: Cantidad de herramientas cargadas.

## Mantenimiento de Datos

### Re-indexar Documentos
Se debe ejecutar cada vez que se agreguen, eliminen o modifiquen archivos `.md` en `data/documents/`.
```bash
python scripts/index_documents.py
```
**Verificación:** Revisar los logs para confirmar que el conteo de chunks coincide con el volumen de documentos actualizado.

### Limpieza de Caché
La caché es en memoria y se limpia automáticamente al reiniciar el servidor. Para forzar una limpieza programática:
- Usar el método `get_cache().clear()`.
- **Cuándo hacerlo:** Tras cambios críticos en la lógica de las herramientas o actualizaciones masivas de documentación que invaliden las respuestas guardadas.

## Evaluación y Calidad

Para validar que los cambios no degradan la performance:
1. **Ejecutar evaluación:** `python scripts/run_evaluation.py`
2. **Comparar con baseline:** `python scripts/compare_evaluations.py reports/baseline_metrics.json reports/latest_metrics.json`

## Troubleshooting

| Síntoma | Causa Probable | Solución |
|---------|----------------|----------|
| `PermissionError` al indexar | Carpeta `chroma_db` bloqueada por otro proceso. | Cerrar el servidor FastAPI u otras terminales y reintentar. |
| `OpenAIError: API Key missing` | Variable de entorno no cargada. | Verificar que el archivo `.env` existe y tiene la key correcta. |
| `Port 8000 already in use` | Instancia previa de la API corriendo. | Matar el proceso: `taskkill /F /IM python.exe` (Windows). |
| `ModuleNotFoundError` | Entorno virtual no activado o dependencias faltantes. | Ejecutar `.\venv\Scripts\activate` y `pip install -r requirements.txt`. |
| `RAGAS Evaluation failed` | Problemas de conexión o formato de datos. | Verificar logs en `logs/app_YYYYMMDD.log` para el error específico del dataset. |

## Guías de Extensión

### Agregar un nuevo documento
1. Copiar el archivo `.md` a `data/documents/{categoria}/`.
2. Ejecutar `python scripts/index_documents.py`.
3. Probar con una consulta en `/test-rag`.

### Agregar una nueva herramienta
1. Crear el archivo en `src/tools/nombre_tool.py`.
2. Definir la función y decorarla con `@tool`.
3. Registrar la herramienta en el constructor de `TechnicalAssistant` en `src/agent/agent.py`.
4. Agregar casos de prueba al `golden_dataset.json`.
5. Ejecutar la evaluación completa.
