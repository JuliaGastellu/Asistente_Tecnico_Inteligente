from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import get_settings
from src.agent.agent import get_assistant as get_agent_assistant

settings = get_settings()

@lru_cache()
def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )
    return Chroma(
        persist_directory=str(settings.chroma_db_dir),
        embedding_function=embeddings
    )

def get_assistant():
    return get_agent_assistant()
