import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.config import validate_config, get_settings
from src.api.routers import health, rag, agent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate config at startup so Lambda init failures surface in CloudWatch
    validate_config()
    yield


app = FastAPI(
    title="Technical Documentation Assistant",
    description="Proyecto académico para el bootcamp de AI Engineering. Sistema RAG avanzado con agentes multi-herramienta.",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Endpoints de estado del sistema"},
        {"name": "RAG", "description": "Endpoints para probar la recuperación de documentos"},
        {"name": "Agent", "description": "Endpoints principales del asistente inteligente"}
    ]
)

# CORS: la spec prohíbe allow_origins=["*"] junto con allow_credentials=True.
# Solo habilitamos credenciales cuando hay orígenes explícitos (no wildcard).
_cors_origins = get_settings().cors_origins_list
_allow_credentials = _cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Method: {request.method} | Path: {request.url.path} | Status: {response.status_code} | Duration: {duration:.2f}s")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)} at {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": request.url.path
        }
    )


app.include_router(health.router)
app.include_router(rag.router)
app.include_router(agent.router)

# Mangum adapter — Lambda usa `handler`; Uvicorn usa el bloque __main__
# lifespan="off" porque Lambda gestiona el ciclo de vida del contenedor,
# no el protocolo ASGI lifespan.
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
