import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openweather_api_key: str = Field("", alias="OPENWEATHER_API_KEY")
    
    default_model: str = "gpt-4"
    evaluation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    chunk_size: int = 700
    chunk_overlap: int = 140
    retrieval_k: int = 5
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    docs_dir: Path = data_dir / "documents"
    chroma_db_dir: Path = data_dir / "chroma_db"
    golden_dataset_path: Path = data_dir / "golden_dataset" / "golden_dataset.json"
    logs_dir: Path = base_dir / "logs"
    reports_dir: Path = base_dir / "reports"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

def validate_config():
    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key == "sk-...":
        raise ValueError("OPENAI_API_KEY must be set in .env file")
    
    if not settings.docs_dir.exists():
        raise ValueError(f"Documents directory not found at {settings.docs_dir}")
    
    # Ensure directories exist
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
