import pytest
from unittest.mock import MagicMock, patch
from src.agent.agent import TechnicalAssistant
from src.utils.cache import get_cache
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture(autouse=True)
def clear_cache():
    """El cache es un singleton global: limpiar antes y después para aislar tests."""
    get_cache().clear()
    yield
    get_cache().clear()


def _mock_app(answer: str = "Respuesta de prueba", tools_used=None):
    """Devuelve un grafo compilado simulado con un estado final válido."""
    mock_app = MagicMock()
    mock_app.invoke.return_value = {
        "messages": [HumanMessage(content="q"), AIMessage(content=answer)],
        "tools_used": tools_used or [],
    }
    return mock_app


def test_query_empty_no_crash():
    assistant = TechnicalAssistant()
    # Mockear _build_app (método real) evita cualquier llamada a OpenAI.
    with patch.object(assistant, "_build_app", return_value=_mock_app()):
        result = assistant.query("")
    assert "answer" in result
    assert result.get("error") is not True


def test_query_too_long_no_crash():
    assistant = TechnicalAssistant()
    with patch.object(assistant, "_build_app", return_value=_mock_app()):
        result = assistant.query("a" * 1000)
    assert "answer" in result
    assert result.get("error") is not True


def test_tool_failure_graceful():
    assistant = TechnicalAssistant()
    # El grafo lanza una excepción al invocar → la rama de error debe capturarla.
    mock_app = MagicMock()
    mock_app.invoke.side_effect = Exception("Tool failed")

    with patch.object(assistant, "_build_app", return_value=mock_app):
        result = assistant.query("consulta que fallará")

    assert result["error"] is True
    assert "Lo siento" in result["answer"]
