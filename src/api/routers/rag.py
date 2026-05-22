from fastapi import APIRouter, Depends, Query
from src.api.dependencies import get_vectorstore

router = APIRouter(prefix="/test-rag", tags=["RAG"])

@router.get("")
async def test_rag(
    query: str = Query(..., min_length=1),
    vectorstore=Depends(get_vectorstore)
):
    docs = vectorstore.similarity_search(query, k=3)
    
    results = []
    for doc in docs:
        results.append({
            "preview": doc.page_content[:200] + "...",
            "source": doc.metadata.get("source", "desconocido").split("\\")[-1].split("/")[-1]
        })
        
    return {"query": query, "results": results}
