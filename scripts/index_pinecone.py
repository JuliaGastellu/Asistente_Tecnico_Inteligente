#!/usr/bin/env python3
"""
scripts/index_pinecone.py — Indexa data/documents/ en Pinecone Cloud.

Uso:
    python scripts/index_pinecone.py

Requiere en .env:
    PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY (para los embeddings).

Mismo chunking que el pipeline local (chunk_size=700, overlap=140) y el mismo
modelo de embeddings que usa src/tools/rag_tool.py. Es idempotente: los IDs son
deterministas (hash de source+posición+texto), así que re-ejecutar hace upsert
sobre los mismos vectores en lugar de duplicar.
"""

import sys
import hashlib
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_settings
from src.tools.rag_tool import _get_embedder, _get_pinecone_index
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Dimensión de text-embedding-3-small (por si hay que crear el índice).
_EMBED_DIM = 1536
_UPSERT_BATCH = 100


def _ensure_index():
    """Devuelve el índice; si no existe, lo crea (serverless aws/us-east-1)."""
    settings = get_settings()
    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        raise ValueError("PINECONE_API_KEY y PINECONE_INDEX_NAME deben estar en .env")

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info(f"Creando índice Pinecone '{settings.pinecone_index_name}' (dim={_EMBED_DIM})...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=_EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(settings.pinecone_index_name)


def _chunk_id(source: str, position: int, text: str) -> str:
    raw = f"{source}::{position}::{text}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def index_documents_to_pinecone() -> int:
    """Indexa data/documents/ en Pinecone. Devuelve el número de chunks subidos."""
    settings = get_settings()
    index = _ensure_index()
    embedder = _get_embedder()

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
    logger.info(f"Created {len(chunks)} chunks. Generando embeddings y subiendo...")

    total = 0
    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start:start + _UPSERT_BATCH]
        texts = [c.page_content for c in batch]
        vectors = embedder.embed_documents(texts)

        items = []
        for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
            source = chunk.metadata.get("source", "desconocido")
            filename = str(source).split("\\")[-1].split("/")[-1]
            position = start + offset
            items.append({
                "id": _chunk_id(filename, position, chunk.page_content),
                "values": vector,
                "metadata": {"text": chunk.page_content, "source": filename},
            })

        index.upsert(vectors=items)
        total += len(items)
        logger.info(f"  upsert {total}/{len(chunks)} chunks...")

    logger.info(f"Indexing complete — {total} chunks en Pinecone '{settings.pinecone_index_name}'.")
    return total


if __name__ == "__main__":
    index_documents_to_pinecone()
