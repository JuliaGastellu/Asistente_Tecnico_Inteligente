from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ------------------------------------------------------------------ #
    #  LLM / OpenAI
    # ------------------------------------------------------------------ #
    # No es required a nivel de schema: si lo fuera, Settings() lanzaría
    # ValidationError en tiempo de IMPORT (vía el logger), antes de que
    # validate_config() pueda dar un mensaje claro. La validación real ocurre
    # en validate_config() (ERR-018).
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openweather_api_key: str = Field("", alias="OPENWEATHER_API_KEY")

    default_model: str = "gpt-4o"
    evaluation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ------------------------------------------------------------------ #
    #  LLM / OpenRouter (gateway opcional; si está, reemplaza a OpenAI)
    # ------------------------------------------------------------------ #
    openrouter_api_key: Optional[str] = Field(None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_model: str = Field("openai/gpt-4o-mini", alias="OPENROUTER_MODEL")

    # ------------------------------------------------------------------ #
    #  Pinecone (vector DB cloud opcional; si está, reemplaza a ChromaDB)
    # ------------------------------------------------------------------ #
    pinecone_api_key: Optional[str] = Field(None, alias="PINECONE_API_KEY")
    pinecone_index_name: Optional[str] = Field(None, alias="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field("us-east-1-aws", alias="PINECONE_ENVIRONMENT")

    # ------------------------------------------------------------------ #
    #  App
    # ------------------------------------------------------------------ #
    app_env: str = Field("local", alias="APP_ENV")          # "local" | "production"
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ------------------------------------------------------------------ #
    #  API / CORS
    # ------------------------------------------------------------------ #
    # Orígenes permitidos, separados por coma. "*" = todos (sin credenciales,
    # por restricción de la spec CORS). Para permitir credenciales, listar
    # orígenes explícitos: CORS_ORIGINS=https://midominio.com,https://otro.com
    cors_origins: str = Field("*", alias="CORS_ORIGINS")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ------------------------------------------------------------------ #
    #  RAG / ChromaDB
    # ------------------------------------------------------------------ #
    chunk_size: int = 700
    chunk_overlap: int = 140
    retrieval_k: int = 5

    # Reranking y compresión vía LLM: alta calidad pero ~1 llamada LLM POR documento
    # (hasta ~15 llamadas secuenciales por consulta). Desactivados por defecto para
    # cumplir el target Latency P95 < 2.0s; el orden por similitud vectorial ya es bueno.
    # Activar solo si se prioriza calidad sobre latencia (p.ej. evaluación offline).
    rag_use_llm_rerank: bool = Field(False, alias="RAG_USE_LLM_RERANK")
    rag_use_compression: bool = Field(False, alias="RAG_USE_COMPRESSION")

    # ------------------------------------------------------------------ #
    #  Paths (resueltos en validación para soportar /tmp en Lambda)
    # ------------------------------------------------------------------ #
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    docs_dir: Path = data_dir / "documents"
    chroma_db_dir: Path = data_dir / "chroma_db"
    golden_dataset_path: Path = data_dir / "golden_dataset" / "golden_dataset.json"
    logs_dir: Path = base_dir / "logs"
    reports_dir: Path = base_dir / "reports"

    # ------------------------------------------------------------------ #
    #  Entorno Cloud — Patrón Camaleón
    # ------------------------------------------------------------------ #
    # ENVIRONMENT controla toda la lógica cloud. Valores válidos:
    #   "local"      → sin LocalStack, sin boto3 (desarrollo puro)
    #   "localstack" → LocalStack en localhost:4566
    #   "aws"        → AWS real (usa IAM Role, sin endpoint_url)
    environment: Literal["local", "localstack", "aws"] = Field(
        "local", alias="ENVIRONMENT"
    )

    # Estos campos se pueden sobreescribir desde .env; si no están
    # presentes, el @model_validator los rellena automáticamente según
    # el valor de `environment`.
    is_local: bool = Field(True, alias="IS_LOCAL")
    aws_endpoint_url: Optional[str] = Field(None, alias="AWS_ENDPOINT_URL")
    aws_region: str = Field("us-east-1", alias="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, alias="AWS_SECRET_ACCESS_KEY")
    s3_bucket_name: str = Field("ai-documents", alias="S3_BUCKET_NAME")

    @model_validator(mode="after")
    def _resolve_cloud_config(self) -> "Settings":
        """Autoconfigura variables cloud según el entorno detectado."""
        if self.environment == "localstack":
            self.is_local = True
            if self.aws_endpoint_url is None:
                self.aws_endpoint_url = "http://localhost:4566"
            if self.aws_access_key_id is None:
                self.aws_access_key_id = "test"
            if self.aws_secret_access_key is None:
                self.aws_secret_access_key = "test"
            # En Lambda/LocalStack los paths efímeros van a /tmp (único dir escribible)
            self.chroma_db_dir = Path("/tmp/chroma_db")
            self.logs_dir = Path("/tmp/logs")

        elif self.environment == "aws":
            self.is_local = False
            self.aws_endpoint_url = None   # SDK usa endpoints regionales nativos
            self.aws_access_key_id = None  # IAM Execution Role, sin credenciales explícitas
            self.aws_secret_access_key = None
            self.chroma_db_dir = Path("/tmp/chroma_db")
            self.logs_dir = Path("/tmp/logs")

        else:  # "local"
            self.is_local = True
            self.aws_endpoint_url = None

        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Instancia compartida a nivel de módulo (conveniencia: `from src.config import settings`).
# Es el mismo objeto que devuelve get_settings() por el lru_cache.
settings = get_settings()


def validate_config():
    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key == "sk-...":
        raise ValueError("OPENAI_API_KEY must be set in .env file")

    if settings.environment == "local":
        # En desarrollo local los documentos deben estar en disco
        if not settings.docs_dir.exists():
            raise ValueError(f"Documents directory not found at {settings.docs_dir}")
        # logs/ y reports/ solo son relevantes en local; en Lambda no son escribibles
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        settings.reports_dir.mkdir(parents=True, exist_ok=True)

    # /tmp/chroma_db es escribible tanto en Lambda como en local
    settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
