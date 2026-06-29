from functools import lru_cache
from langchain_openai import ChatOpenAI
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=8)
def _get_client(model: str) -> ChatOpenAI:
    """Cachea el cliente ChatOpenAI por modelo: evitar reconstruirlo en cada
    consulta (ERR-023). bind_tools() crea un runnable nuevo sin mutar este cliente,
    por lo que es seguro compartirlo.

    Si OPENROUTER_API_KEY está definida, apunta al gateway de OpenRouter (compatible
    con la API de OpenAI vía base_url); si no, usa OpenAI directo (fallback)."""
    if settings.openrouter_api_key:
        return ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
        )
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

def estimate_query_complexity(query: str) -> str:
    query_lower = query.lower()
    words = query_lower.split()
    
    # Rules for complex query
    is_long = len(words) >= 15
    has_logical_connectors = " y " in query_lower and "?" in query_lower
    has_comparison = any(w in query_lower for w in ["comparar", "diferencia", "vs"])
    has_explanation_request = "explicar" in query_lower and "ejemplo" in query_lower
    
    complexity_score = sum([is_long, has_logical_connectors, has_comparison, has_explanation_request])
    
    complexity = "complex" if complexity_score >= 2 else "simple"
    logger.info(f"Query complexity estimated: {complexity} (score: {complexity_score})")
    return complexity

def get_llm_for_query(query: str) -> ChatOpenAI:
    # Con OpenRouter usamos el modelo configurado (su naming difiere del de OpenAI,
    # p.ej. "openai/gpt-4o-mini"); la selección dinámica gpt-4o/mini aplica a OpenAI.
    if settings.openrouter_api_key:
        model = settings.openrouter_model
        logger.info(f"Selected OpenRouter model: {model}")
        return _get_client(model)

    complexity = estimate_query_complexity(query)
    model = settings.default_model if complexity == "complex" else settings.evaluation_model

    logger.info(f"Selected model: {model} for complexity: {complexity}")
    return _get_client(model)
