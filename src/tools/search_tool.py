from langchain_core.tools import tool
from duckduckgo_search import DDGS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """Busca información en tiempo real en internet.
    Usa esta herramienta solo si no encuentras la respuesta en la documentación técnica 
    o si la información requerida es muy reciente.
    """
    logger.info(f"Searching web for: {query}")
    try:
        results = []
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, max_results=max_results)
            for r in ddgs_gen:
                results.append(f"Título: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}")
        
        if not results:
            return "No se encontraron resultados en la web para esta consulta."
            
        return "\n\n---\n\n".join(results)
    except Exception as e:
        logger.error(f"Error in web search tool: {e}")
        return f"Error al buscar en la web: {str(e)}"
