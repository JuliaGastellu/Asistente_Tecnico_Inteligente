# Guía de Despliegue

Instrucciones para poner el **Technical Documentation Assistant** en producción.

---

## Desarrollo Local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar .env
OPENAI_API_KEY=sk-...
ENVIRONMENT=local

# 3. Indexar documentos en ChromaDB
python scripts/index_documents.py

# 4. Levantar servidor
python src/api/main.py
# → http://localhost:8000
```

El directorio `logs/` se crea automáticamente al arrancar. No es necesario crearlo manualmente.

---

## LocalStack (simulación AWS Lambda)

### Requisitos previos
- Docker Desktop corriendo
- `ENVIRONMENT=localstack` en `.env`

### Paso 1 — Iniciar LocalStack

LocalStack tiene un límite de 250 MB descomprimido por defecto (igual que AWS). El paquete Lambda de este proyecto supera ese límite por las dependencias transitivas de ChromaDB y LangChain. Se debe arrancar con el límite ampliado a 1 GB:

```bash
docker run --rm -p 4566:4566 \
  -e LAMBDA_LIMITS_TOTAL_CODE_SIZE=1073741824 \
  localstack/localstack
```

### Paso 2 — Configurar `.env`

```env
OPENAI_API_KEY=sk-...
OPENWEATHER_API_KEY=...   # Opcional
ENVIRONMENT=localstack
```

### Paso 3 — Desplegar

```bash
python scripts/deploy_localstack.py
```

El script es **idempotente**: se puede re-ejecutar sin efectos secundarios. Internamente:

1. Crea el bucket S3 `ai-documents` y sube los 60 documentos.
2. Construye `function.zip` instalando `requirements-lambda.txt` en un directorio temporal, excluyendo paquetes innecesarios.
3. Sube el ZIP a S3 y crea/actualiza la función Lambda.
4. Crea o reutiliza el API Gateway REST con proxy `{proxy+}` → Lambda.

### Paso 4 — Verificar

```bash
# La URL base se imprime al final del deploy. Ejemplo:
curl http://localhost:4566/restapis/<api-id>/dev/_user_request_/health
```

### Limpieza

```bash
python scripts/cleanup_localstack.py
```

---

## Estructura del paquete Lambda

El deploy usa `requirements-lambda.txt` (sin `ragas`, `datasets`, `pytest`) y excluye los siguientes paquetes del ZIP para mantenerse bajo el límite de 250 MB descomprimido:

| Paquete(s) excluido(s) | Motivo |
|------------------------|--------|
| `onnxruntime`, `onnxruntime_common` | Embedding por defecto de ChromaDB; usamos `OpenAIEmbeddings` |
| `datasets`, `pyarrow`, `multiprocess`, `dill`, `xxhash` | Dependencias transitivas de `langchain-community`, no usadas en runtime |
| `huggingface_hub`, `tokenizers` | Ecosistema HuggingFace; no necesario con OpenAI |
| `grpcio`, `grpcio_tools` | ChromaDB en modo embedded (`PersistentClient`) no usa gRPC |
| `posthog` | Telemetría de ChromaDB; innecesaria en producción |
| `botocore/data/<servicio>` | Se conservan solo los modelos de `s3`, `lambda`, `apigateway`, `apigatewayv2`, `sts`, `iam` |

Si el deploy sigue fallando por tamaño (al actualizar dependencias), reiniciar LocalStack con `LAMBDA_LIMITS_TOTAL_CODE_SIZE=1073741824`.

---

## Variables de Entorno del Sistema

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `OPENAI_API_KEY` | Sí | Token de acceso a la API de OpenAI |
| `OPENWEATHER_API_KEY` | No | Token para la herramienta de clima (opcional) |
| `ENVIRONMENT` | No | `local` (default), `localstack`, `aws` |
| `AWS_ENDPOINT_URL` | Auto | Autoconfigured a `http://localhost:4566` en `localstack` |
| `AWS_REGION` | No | Default: `us-east-1` |
| `S3_BUCKET_NAME` | No | Default: `ai-documents` |

En entornos Lambda (`localstack` / `aws`):
- `chroma_db_dir` se redirige a `/tmp/chroma_db` (único directorio escribible).
- `logs_dir` se redirige a `/tmp/logs`; si falla la escritura, el logger funciona solo por consola (CloudWatch).

---

## Contenedorización con Docker

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Construir
docker build -t tech-assistant:latest .

# Ejecutar
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -e OPENWEATHER_API_KEY=your-key \
  tech-assistant:latest
```

---

## Consideraciones de Producción (AWS real)

1. **Persistencia ChromaDB:** Montar EFS en `/tmp/chroma_db` o pre-poblar el índice en S3 y cargarlo en cold start.
2. **Cache Externo:** Reemplazar `QueryCache` en memoria por **Redis** (ElastiCache) para soportar múltiples instancias.
3. **Seguridad:**
   - Habilitar HTTPS mediante API Gateway + ACM.
   - Implementar autenticación vía API Key o OAuth2/JWT en los endpoints.
   - Configurar Rate Limiting en API Gateway para prevenir abuso.
4. **Logs:** CloudWatch captura automáticamente stdout del Lambda; el `FileHandler` se desactiva silenciosamente en `/tmp` no escribible.

---

## Integración Continua (CI)

`.github/workflows/test.yml`:
```yaml
name: Python Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run unit tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}
      run: |
        python -m pytest tests/unit/ -v
```
