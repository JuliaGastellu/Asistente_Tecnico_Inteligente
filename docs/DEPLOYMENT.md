# Guía de Despliegue 🚀

Instrucciones para poner el **Technical Documentation Assistant** en producción.

## Contenedorización con Docker

### Dockerfile
```dockerfile
FROM python:3.11-slim

# Evitar la generación de archivos .pyc y habilitar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y datos necesarios
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Comandos de Docker
```bash
# Construir la imagen
docker build -t tech-assistant:latest .

# Ejecutar el contenedor (pasando las variables de entorno)
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -e OPENWEATHER_API_KEY=your-key \
  tech-assistant:latest
```

## Variables de Entorno

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `OPENAI_API_KEY` | Sí | Token de acceso a la API de OpenAI. |
| `OPENWEATHER_API_KEY` | No | Token para la herramienta de clima (opcional). |
| `LOG_LEVEL` | No | Nivel de logging (DEBUG, INFO, etc. Default: INFO). |

## Consideraciones de Producción

1. **Persistencia de Datos:** Montar un volumen para la carpeta `data/chroma_db/` para evitar perder el índice al reiniciar el contenedor.
2. **Cache Externo:** Reemplazar el `QueryCache` en memoria por **Redis** para soportar múltiples instancias de la API.
3. **Seguridad:**
   - Habilitar **HTTPS** mediante un proxy inverso (Nginx, Traefik).
   - Implementar un sistema de autenticación (OAuth2/JWT) en los endpoints.
   - Configurar **Rate Limiting** para prevenir abusos de la API de OpenAI.
4. **Logs:** Enviar los logs a un sistema centralizado (ELK Stack, CloudWatch) en lugar de solo archivos locales.

## Integración Continua (CI)

Archivo `.github/workflows/test.yml`:
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
