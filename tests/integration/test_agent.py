import pytest
from unittest.mock import MagicMock, patch
from src.agent.agent import TechnicalAssistant

@pytest.mark.integration
@patch("src.agent.agent.get_llm_for_query")
def test_agent_query_structure(mock_llm):
    # Mock LLM and Agent response
    mock_llm_instance = MagicMock()
    mock_llm.return_value = mock_llm_instance
    
    assistant = TechnicalAssistant()
    
    # Mock _build_executor to avoid complex agent setup in integration test
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": "Respuesta de prueba [Fuente 1: doc.md]",
        "intermediate_steps": [(MagicMock(tool="search_documents"), "obs")]
    }
    assistant._build_executor = MagicMock(return_value=mock_executor)
    
    result = assistant.query("¿Cómo usar FastAPI?")
    
    assert "answer" in result
    assert "tools_used" in result
    assert "sources" in result
    assert "search_documents" in result["tools_used"]
    assert "doc.md" in result["sources"]

@pytest.mark.integration
def test_agent_memory_persistence():
    assistant = TechnicalAssistant()
    assistant._build_executor = MagicMock()
    
    # First query
    assistant.query("Hola, soy Juan", session_id="user_1")
    memory = assistant._get_or_create_memory("user_1")
    assert len(memory.chat_memory.messages) > 0
    
    # Second query same session
    assistant.query("¿Cómo me llamo?", session_id="user_1")
    assert len(memory.chat_memory.messages) > 2
