"""
Indexa los documentos del corpus en ChromaDB local y, opcionalmente,
sube el índice comprimido a S3 para que la Lambda pueda descargarlo
en cold start.

Uso:
    python scripts/index_documents.py             # solo indexación local
    python scripts/index_documents.py --upload    # indexa y sube a S3
"""

import sys
import os
import io
import tarfile
import shutil
import argparse
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import get_settings, validate_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# El índice siempre se genera en disco local (data/chroma_db), no en /tmp.
# /tmp/chroma_db es el path de la Lambda en runtime — diferente entorno.
LOCAL_CHROMA_DIR = root_path / "data" / "chroma_db"
S3_CHROMA_KEY = "chroma_db/index.tar.gz"


def _remove_readonly(func, path, _):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


def index_documents(chroma_dir: Path = LOCAL_CHROMA_DIR) -> int:
    """
    Indexa data/documents/ en ChromaDB.
    Devuelve el número de chunks generados.
    """
    settings = get_settings()

    logger.info("Starting document indexing...")

    if chroma_dir.exists():
        logger.info(f"Clearing existing index at {chroma_dir}")
        shutil.rmtree(chroma_dir, onerror=_remove_readonly)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading documents from {settings.docs_dir}")
    loader = DirectoryLoader(
        str(settings.docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    logger.info("Generating embeddings and indexing in ChromaDB...")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
    )

    # Verificación rápida
    vs = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
    results = vs.similarity_search("FastAPI", k=1)
    if results:
        logger.info(f"Test search OK: {results[0].metadata.get('source')}")
    else:
        logger.warning("Test search returned no results — revisar los documentos.")

    logger.info(f"Indexing complete — {len(chunks)} chunks indexed in {chroma_dir}")
    return len(chunks)


def upload_index_to_s3(chroma_dir: Path = LOCAL_CHROMA_DIR):
    """
    Comprime chroma_dir/ en tar.gz y lo sube a S3.
    La Lambda descarga este archivo en cold start para tener el índice disponible.
    """
    settings = get_settings()

    if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
        raise FileNotFoundError(
            f"ChromaDB directory not found or empty at {chroma_dir}. "
            "Run indexing first."
        )

    logger.info(f"Compressing ChromaDB index ({chroma_dir})...")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(chroma_dir, arcname="chroma_db")
    size_mb = buf.tell() / 1024 / 1024
    buf.seek(0)
    logger.info(f"Compressed index: {size_mb:.1f} MB")

    import boto3
    from botocore.exceptions import ClientError

    boto_kwargs: dict = {"service_name": "s3", "region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        boto_kwargs["endpoint_url"] = settings.aws_endpoint_url
    if settings.aws_access_key_id:
        boto_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        boto_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    s3 = boto3.client(**boto_kwargs)

    try:
        s3.create_bucket(Bucket=settings.s3_bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise

    s3.upload_fileobj(buf, settings.s3_bucket_name, S3_CHROMA_KEY)
    logger.info(
        f"ChromaDB index uploaded → s3://{settings.s3_bucket_name}/{S3_CHROMA_KEY}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index documents into ChromaDB. Use --upload to push the index to S3."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the index to S3 after indexing (requires ENVIRONMENT=localstack or aws).",
    )
    args = parser.parse_args()

    validate_config()
    index_documents()

    if args.upload:
        settings = get_settings()
        if settings.environment == "local":
            logger.warning(
                "--upload requires ENVIRONMENT=localstack or aws. "
                "Set ENVIRONMENT in your .env and try again."
            )
        else:
            upload_index_to_s3()
