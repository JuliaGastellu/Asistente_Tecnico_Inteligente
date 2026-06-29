from fastapi import APIRouter, Depends
from src.api.dependencies import get_assistant, get_vectorstore_or_none
from src.config import get_settings

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
async def health_check(
    assistant=Depends(get_assistant),
    vectorstore=Depends(get_vectorstore_or_none)
):
    settings = get_settings()
    components = {
        "agent": "ok",
        "tools_count": len(assistant.tools)
    }

    status = "healthy"

    if settings.pinecone_api_key:
        # Backend cloud: verificar Pinecone (nunca lanza; devuelve estado).
        from src.tools.rag_tool import pinecone_status
        ok, detail = pinecone_status()
        components["pinecone"] = detail
        if not ok:
            status = "degraded"
    else:
        # Backend local ChromaDB. vectorstore is None => construcción falló
        # (cold start/S3/embeddings). Un fallo => 'degraded' (200), nunca 500 (ERR-015).
        components["chromadb"] = "ok"
        if vectorstore is None:
            components["chromadb"] = "error"
            status = "degraded"
        else:
            try:
                # similarity_search is the standard LangChain Chroma public API
                vectorstore.similarity_search("test", k=1)
            except Exception:
                components["chromadb"] = "error"
                status = "degraded"

    return {
        "status": status,
        "version": "1.0.0",
        "components": components
    }
