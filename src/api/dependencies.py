import io
import tarfile
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import get_settings
from src.agent.agent import get_assistant as get_agent_assistant
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_S3_CHROMA_KEY = "chroma_db/index.tar.gz"


def _try_load_chroma_from_s3() -> bool:
    """
    En entornos Lambda (localstack/aws), descarga el índice ChromaDB desde S3
    y lo extrae en /tmp/chroma_db si el directorio está vacío (cold start).

    El índice se genera localmente y se sube a S3 mediante:
        python scripts/index_documents.py --upload
    o automáticamente durante:
        python scripts/deploy_localstack.py

    En warm containers el directorio ya existe y la función retorna inmediatamente.
    """
    settings = get_settings()
    if settings.environment not in ("localstack", "aws"):
        return False

    chroma_dir = settings.chroma_db_dir
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        logger.info("ChromaDB already loaded in /tmp — skipping S3 download.")
        return True

    logger.info(f"Cold start: downloading ChromaDB index from S3 to {chroma_dir}...")
    try:
        import boto3

        boto_kwargs: dict = {"service_name": "s3", "region_name": settings.aws_region}
        if settings.aws_endpoint_url:
            boto_kwargs["endpoint_url"] = settings.aws_endpoint_url
        if settings.aws_access_key_id:
            boto_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            boto_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        s3 = boto3.client(**boto_kwargs)
        obj = s3.get_object(Bucket=settings.s3_bucket_name, Key=_S3_CHROMA_KEY)
        data = obj["Body"].read()

        chroma_dir.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(path=chroma_dir.parent)

        file_count = sum(1 for f in chroma_dir.rglob("*") if f.is_file())
        logger.info(f"ChromaDB loaded from S3 ({file_count} files in {chroma_dir})")
        return True

    except Exception as exc:
        logger.warning(
            f"Could not load ChromaDB from S3: {exc}. "
            "RAG queries will return empty results. "
            "Fix: run `python scripts/index_documents.py --upload` and redeploy."
        )
        return False


@lru_cache()
def get_vectorstore() -> Chroma:
    """
    Devuelve la instancia compartida de ChromaDB.
    En Lambda (cold start), descarga el índice desde S3 antes de inicializar.
    El @lru_cache garantiza que esto ocurre una sola vez por contenedor.
    """
    _try_load_chroma_from_s3()
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return Chroma(
        persist_directory=str(settings.chroma_db_dir),
        embedding_function=embeddings,
    )


def get_vectorstore_or_none():
    """Variante segura para /health: si la CONSTRUCCIÓN del vectorstore falla
    (cold start, descarga S3, embeddings), devuelve None en lugar de propagar la
    excepción. Así /health puede reportar 'degraded' (200) en vez de 500 (ERR-015),
    y sigue siendo overrideable en tests vía dependency_overrides.
    """
    try:
        return get_vectorstore()
    except Exception as exc:
        logger.warning(f"Vectorstore no disponible en /health: {exc}")
        return None


def get_assistant():
    return get_agent_assistant()
