import time
from fastapi import APIRouter, Depends, HTTPException
from src.api.models.query import QueryRequest, QueryResponse
from src.api.dependencies import get_assistant
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(tags=["Agent"])

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    assistant=Depends(get_assistant)
):
    """
    Procesa una consulta técnica usando el asistente inteligente.
    
    - **query**: La pregunta o instrucción técnica.
    - **session_id**: ID opcional para mantener el contexto de la conversación.
    """
    logger.info(f"Received query: {request.query}")
    start_time = time.time()
    
    try:
        result = assistant.query(request.query, request.session_id)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["answer"])
            
        latency = time.time() - start_time
        logger.info(f"Query processed in {latency:.2f}s")
        
        return QueryResponse(
            answer=result["answer"],
            tools_used=result["tools_used"],
            sources=result["sources"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
