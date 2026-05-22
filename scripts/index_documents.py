import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

def index_documents():
    logger.info("Starting document indexing...")
    
    # 1. Clear existing index
    if settings.chroma_db_dir.exists():
        logger.info(f"Removing existing index at {settings.chroma_db_dir}")
        shutil.rmtree(settings.chroma_db_dir)
    
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
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_db_dir)
    )
    # Chroma in 0.4.x+ persists automatically, but call persist() if needed in older versions
    # vectorstore.persist() 
    
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
