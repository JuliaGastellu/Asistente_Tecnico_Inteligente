"""
Initializes ChromaDB by indexing all documents in data/documents/.
Run inside the container before starting the API:
    docker compose run --rm rag-api python setup_data.py
"""
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import get_settings, validate_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    validate_config()
    settings = get_settings()

    # Clear existing index
    if settings.chroma_db_dir.exists():
        logger.info(f"Removing existing index at {settings.chroma_db_dir}")
        try:
            shutil.rmtree(settings.chroma_db_dir)
        except Exception as exc:
            logger.warning(f"shutil.rmtree failed ({exc}), clearing contents instead")
            for root, dirs, files in os.walk(settings.chroma_db_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass

    settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)

    # Load .md documents
    logger.info(f"Loading documents from {settings.docs_dir}")
    loader = DirectoryLoader(
        str(settings.docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Embed and index
    logger.info("Generating embeddings and writing to ChromaDB...")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_db_dir),
    )

    # Smoke-test the index
    results = vectorstore.similarity_search("FastAPI", k=1)
    if results:
        logger.info(f"Index verified — sample hit: {results[0].metadata.get('source')}")
    else:
        logger.warning("Index built but test search returned no results")

    logger.info("setup_data complete")


if __name__ == "__main__":
    main()
