import re
import hashlib
from typing import List, Tuple, Optional
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

class OptimizedRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.llm_mini = ChatOpenAI(model=settings.evaluation_model, temperature=0)
        self.compressor = LLMChainExtractor.from_llm(self.llm_mini)

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
        compressed_docs = []
        for doc in documents:
            try:
                compressed_content = self.compressor.compress_documents([doc], query)
                if compressed_content:
                    compressed_docs.append(compressed_content[0])
                else:
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

    def detect_comparison_query(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        pattern = r"(?:diferencia|vs|comparar|comparación) (?:entre|de)?\s*(.+?)\s*(?:y|vs)\s*(.+)"
        match = re.search(pattern, query.lower())
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
            
        docs = self.rerank_documents(query, docs)
        docs = docs[:5]
        docs = self.compress_context(query, docs)
        docs = self.deduplicate_documents(docs)
        return docs

@tool
def search_documents(query: str, k: int = 5) -> str:
    """Busca en la documentación técnica de FastAPI, LangChain y Python.
    Usa esta herramienta para responder preguntas técnicas sobre estas tecnologías.
    """
    if not query or len(query.strip()) == 0:
        return "Error: La consulta de búsqueda no puede estar vacía."
    
    if len(query) > 500:
        query = query[:500]
        
    try:
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
