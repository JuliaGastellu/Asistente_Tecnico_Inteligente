import pytest

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data

def test_health_degraded_when_vectorstore_down(mock_assistant):
    """Si el vectorstore no está disponible, /health responde 200 'degraded', no 500."""
    from src.api.main import app
    from src.api.dependencies import get_assistant, get_vectorstore_or_none
    from fastapi.testclient import TestClient

    app.dependency_overrides[get_assistant] = lambda: mock_assistant
    app.dependency_overrides[get_vectorstore_or_none] = lambda: None  # construcción falló
    try:
        with TestClient(app) as c:
            response = c.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["chromadb"] == "error"
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint(client):
    payload = {"query": "¿Qué es FastAPI?", "session_id": "test_session"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "tools_used" in data
    assert "sources" in data

def test_rag_endpoint(client):
    response = client.get("/test-rag?query=FastAPI")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0

def test_calculator_via_agent(client, mock_assistant):
    mock_assistant.query.return_value = {
        "answer": "El resultado es 30.",
        "tools_used": ["calculate"],
        "sources": [],
        "error": False
    }
    payload = {"query": "¿Cuánto es 15 * 2?"}
    response = client.post("/query", json=payload)
    assert "calculate" in response.json()["tools_used"]

def test_multi_tool_via_agent(client, mock_assistant):
    mock_assistant.query.return_value = {
        "answer": "El 15% de 200 es 30 y en Madrid hace sol.",
        "tools_used": ["calculate", "get_weather"],
        "sources": [],
        "error": False
    }
    payload = {"query": "¿Cuánto es 15% de 200 y qué tiempo hace en Madrid?"}
    response = client.post("/query", json=payload)
    assert "calculate" in response.json()["tools_used"]
    assert "get_weather" in response.json()["tools_used"]
