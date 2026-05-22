from fastapi import APIRouter, Depends
from src.api.dependencies import get_assistant, get_vectorstore

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
async def health_check(
    assistant=Depends(get_assistant),
    vectorstore=Depends(get_vectorstore)
):
    components = {
        "agent": "ok",
        "chromadb": "ok",
        "tools_count": len(assistant.tools)
    }
    
    status = "healthy"
    
    try:
        # Check if vectorstore is accessible
        vectorstore.get(limit=1)
    except Exception:
        components["chromadb"] = "error"
        status = "degraded"
        
    return {
        "status": status,
        "version": "1.0.0",
        "components": components
    }
