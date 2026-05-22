import os
import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import get_settings, validate_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
validate_config()
settings = get_settings()

def remove_readonly(func, path, excinfo):
    """Clear the readonly bit and reattempt the removal."""
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

def index_documents():
    logger.info("Starting document indexing...")
    
    # 1. Clear existing index
    if settings.chroma_db_dir.exists():
        logger.info(f"Removing existing index at {settings.chroma_db_dir}")
        try:
            shutil.rmtree(settings.chroma_db_dir, onerror=remove_readonly)
        except Exception as e:
            logger.warning(f"Could not remove directory {settings.chroma_db_dir} using shutil: {e}")
            logger.info("Attempting to clear directory contents instead...")
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
    
    # Ensure directory exists after cleanup
    settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load documents
    logger.info(f"Loading documents from {settings.docs_dir}")
    loader = DirectoryLoader(
        str(settings.docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents.")
    
    # 3. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    
    # 4. Generate embeddings and index
    logger.info("Generating embeddings and indexing in ChromaDB...")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_db_dir)
    )
    
    logger.info("Indexing complete.")
    
    # 5. Test search
    logger.info("Performing test search...")
    results = vectorstore.similarity_search("FastAPI", k=1)
    if results:
        logger.info(f"Test search successful. Found: {results[0].metadata.get('source')}")
    else:
        logger.warning("Test search returned no results.")

if __name__ == "__main__":
    index_documents()
