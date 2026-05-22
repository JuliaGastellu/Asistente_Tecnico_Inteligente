import pytest
from unittest.mock import MagicMock
from src.tools.rag_tool import search_documents, OptimizedRetriever

def test_search_documents_empty():
    result = search_documents.invoke("")
    assert "Error" in result

def test_detect_comparison_query():
    retriever = OptimizedRetriever(MagicMock())
    is_comp, a, b = retriever.detect_comparison_query("diferencia entre FastAPI y Flask")
    assert is_comp is True
    assert a == "fastapi"
    assert b == "flask"
    
    is_comp, a, b = retriever.detect_comparison_query("FastAPI vs Django")
    assert is_comp is True
    assert a == "fastapi"
    assert b == "django"

def test_deduplicate_documents():
    retriever = OptimizedRetriever(MagicMock())
    doc1 = MagicMock(page_content="mismo contenido")
    doc2 = MagicMock(page_content="mismo contenido")
    doc3 = MagicMock(page_content="otro contenido")
    
    unique = retriever.deduplicate_documents([doc1, doc2, doc3])
    assert len(unique) == 2
