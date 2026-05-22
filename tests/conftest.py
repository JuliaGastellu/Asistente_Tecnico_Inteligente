import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.dependencies import get_vectorstore, get_assistant

@pytest.fixture
def mock_vectorstore():
    mock = MagicMock()
    # Mock similarity_search to return dummy docs
    doc1 = MagicMock()
    doc1.page_content = "Contenido de prueba 1"
    doc1.metadata = {"source": "test1.md"}
    
    doc2 = MagicMock()
    doc2.page_content = "Contenido de prueba 2"
    doc2.metadata = {"source": "test2.md"}
    
    mock.similarity_search.return_value = [doc1, doc2]
    mock.get.return_value = {"ids": ["1"]}
    return mock

@pytest.fixture
def mock_assistant():
    mock = MagicMock()
    mock.query.return_value = {
        "answer": "Esta es una respuesta de prueba.",
        "tools_used": ["search_documents"],
        "sources": ["test1.md"],
        "error": False
    }
    mock.tools = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    return mock

@pytest.fixture
def client(mock_vectorstore, mock_assistant):
    app.dependency_overrides[get_vectorstore] = lambda: mock_vectorstore
    app.dependency_overrides[get_assistant] = lambda: mock_assistant
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
