from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., example="¿Cómo crear un endpoint en FastAPI?")
    session_id: str = Field("default", example="user_123")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "¿Cómo crear un endpoint en FastAPI?",
                    "session_id": "default"
                }
            ]
        }
    }

class QueryResponse(BaseModel):
    answer: str
    tools_used: List[str] = []
    sources: List[str] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Para crear un endpoint en FastAPI, usas el decorador @app.get()...",
                    "tools_used": ["search_documents"],
                    "sources": ["fastapi_intro.md"]
                }
            ]
        }
    }
