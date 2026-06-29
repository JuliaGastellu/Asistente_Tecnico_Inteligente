import re
import hashlib
from typing import List, Tuple, Optional
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


# ------------------------------------------------------------------ #
#  Backend cloud: Pinecone (se usa cuando PINECONE_API_KEY está definida).
#  Si no, se mantiene el backend local ChromaDB (OptimizedRetriever).
# ------------------------------------------------------------------ #

def _get_embedder() -> OpenAIEmbeddings:
    """Embedder compartido (mismo modelo en indexación y consulta)."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def _get_pinecone_index():
    """Inicializa el índice Pinecone. Falla claro si faltan variables.
    Import perezoso: `pinecone` no es necesario en modo local/ChromaDB."""
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY no configurada")
    if not settings.pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME no configurada")
    from pinecone import Pinecone  # lazy import
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index_name)


def pinecone_status() -> Tuple[bool, str]:
    """Estado de Pinecone para /health. Nunca lanza: devuelve (ok, detalle)."""
    try:
        index = _get_pinecone_index()
        stats = index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        if total is None and isinstance(stats, dict):
            total = stats.get("total_vector_count")
        return True, f"connected ({total} vectors)"
    except Exception as e:
        return False, f"unavailable: {e}"


def _search_pinecone(query: str) -> str:
    """Búsqueda RAG contra Pinecone. Mantiene Query Expansion (comparaciones) y el
    mismo formato de salida que el backend ChromaDB: '[Fuente N: archivo]\\n{texto}'."""
    index = _get_pinecone_index()
    embedder = _get_embedder()

    is_comp, concept_a, concept_b = OptimizedRetriever.detect_comparison_query(query)
    queries = [concept_a, concept_b] if (is_comp and concept_a and concept_b) else [query]

    seen_hashes = set()
    collected = []
    for q in queries:
        vector = embedder.embed_query(q)
        res = index.query(vector=vector, top_k=settings.retrieval_k, include_metadata=True)
        for match in res.matches:
            text = (match.metadata or {}).get("text", "")
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            collected.append(match)

    if not collected:
        return "No se encontraron documentos relevantes para esta consulta."

    chunks = []
    for i, match in enumerate(collected[:settings.retrieval_k]):
        md = match.metadata or {}
        source = str(md.get("source", "desconocido"))
        filename = source.split("\\")[-1].split("/")[-1]
        chunks.append(f"[Fuente {i+1}: {filename}]\n{md.get('text', '')}")

    return "\n\n---\n\n".join(chunks)


class OptimizedRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.llm_mini = ChatOpenAI(
            model=settings.evaluation_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

    def estimate_complexity(self, query: str) -> int:
        query_lower = query.lower()
        words = query_lower.split()
        
        if len(words) <= 3:
            return 3
        if any(w in query_lower for w in ["vs", "diferencia", "comparar", "comparación"]):
            return 10
        if any(w in query_lower for w in ["cómo", "how", "pasos", "guía"]):
            return 7
        return settings.retrieval_k

    def retrieve_base(self, query: str, k: int) -> List:
        return self.vectorstore.similarity_search(query, k=k)

    def rerank_documents(self, query: str, documents: List) -> List:
        if not documents:
            return []
            
        logger.info(f"Reranking {len(documents)} documents...")
        scored_docs = []
        for doc in documents:
            try:
                prompt = f"Puntúa del 1 al 10 qué tan relevante es este documento para la consulta: '{query}'\n\nDocumento: {doc.page_content[:500]}\n\nResponde solo con el número."
                response = self.llm_mini.invoke(prompt)
                score_match = re.search(r'\d+', response.content)
                score = int(score_match.group()) if score_match else 5
            except Exception as e:
                logger.error(f"Error reranking document: {e}")
                score = 5
            scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]

    def compress_context(self, query: str, documents: List) -> List:
        # Simplificación de compresión usando prompt directo para evitar dependencias obsoletas
        compressed_docs = []
        for doc in documents:
            try:
                prompt = f"Extrae solo la información relevante para responder a: '{query}' del siguiente texto:\n\n{doc.page_content[:1000]}"
                response = self.llm_mini.invoke(prompt)
                doc.page_content = response.content
                compressed_docs.append(doc)
            except Exception as e:
                logger.error(f"Error compressing document: {e}")
                compressed_docs.append(doc)
        return compressed_docs

    def deduplicate_documents(self, documents: List) -> List:
        seen_hashes = set()
        unique_docs = []
        for doc in documents:
            doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)
        return unique_docs

    @staticmethod
    def detect_comparison_query(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        # Regex más flexible para detectar comparaciones
        pattern_full = r"(?:diferencia|comparar|comparación) (?:entre|de)?\s*(.+?)\s*(?:y|vs)\s*(.+)"
        pattern_vs = r"(.+?)\s+vs\s+(.+)"
        
        query_lower = query.lower()
        match = re.search(pattern_full, query_lower)
        if match:
            return True, match.group(1).strip(), match.group(2).strip()
            
        match = re.search(pattern_vs, query_lower)
        if match:
            return True, match.group(1).strip(), match.group(2).strip()
            
        return False, None, None

    def retrieve_with_expansion(self, query: str) -> List:
        is_comparison, concept_a, concept_b = self.detect_comparison_query(query)
        if is_comparison and concept_a and concept_b:
            logger.info(f"Comparison detected: {concept_a} vs {concept_b}")
            docs_a = self.retrieve_base(concept_a, k=5)
            docs_b = self.retrieve_base(concept_b, k=5)
            return self.deduplicate_documents(docs_a + docs_b)
        
        k = self.estimate_complexity(query)
        return self.retrieve_base(query, k=k)

    def retrieve(self, query: str) -> List:
        logger.info(f"Starting RAG pipeline for query: {query}")
        docs = self.retrieve_with_expansion(query)
        if not docs:
            return []

        # Deduplicar antes de truncar para no desperdiciar slots con duplicados.
        docs = self.deduplicate_documents(docs)

        # Rerank/compress por LLM solo si se habilitan explícitamente (ver config:
        # impactan fuerte la latencia). Por defecto se usa el orden por similitud.
        if settings.rag_use_llm_rerank:
            docs = self.rerank_documents(query, docs)

        docs = docs[:settings.retrieval_k]

        if settings.rag_use_compression:
            docs = self.compress_context(query, docs)

        return docs

@tool
def search_documents(query: str) -> str:
    """Busca en la documentación técnica de FastAPI, LangChain y Python.
    Usa esta herramienta para responder preguntas técnicas sobre estas tecnologías.
    """
    if not query or len(query.strip()) == 0:
        return "Error: La consulta de búsqueda no puede estar vacía."
    
    if len(query) > 500:
        query = query[:500]

    try:
        # Backend cloud Pinecone si está configurado; si no, ChromaDB local.
        if settings.pinecone_api_key:
            return _search_pinecone(query)

        from src.api.dependencies import get_vectorstore
        vectorstore = get_vectorstore()
        retriever = OptimizedRetriever(vectorstore)
        docs = retriever.retrieve(query)

        if not docs:
            return "No se encontraron documentos relevantes para esta consulta."

        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "desconocido")
            filename = source.split("\\")[-1].split("/")[-1]
            formatted_docs.append(f"[Fuente {i+1}: {filename}]\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_docs)
    except Exception as e:
        logger.error(f"Error in search_documents tool: {e}")
        return f"Error al buscar en los documentos: {str(e)}"
