import pytest
from unittest.mock import MagicMock, patch
from src.agent.agent import TechnicalAssistant
from langchain_core.messages import AIMessage, HumanMessage

def test_agent_query_structure():
    assistant = TechnicalAssistant()
    
    # Mock the compiled app instead of _build_executor
    mock_app = MagicMock()
    mock_app.invoke.return_value = {
        "messages": [
            HumanMessage(content="¿Cómo usar FastAPI?"),
            AIMessage(content="Respuesta de prueba [Fuente 1: doc.md]")
        ],
        "tools_used": ["search_documents"]
    }
    
    with patch.object(assistant, "_build_app", return_value=mock_app):
        result = assistant.query("¿Cómo usar FastAPI?")
        
        assert "answer" in result
        assert "tools_used" in result
        assert "sources" in result
        assert "search_documents" in result["tools_used"]
        assert "doc.md" in result["sources"]

def test_agent_memory_persistence():
    assistant = TechnicalAssistant()
    
    mock_app = MagicMock()
    mock_app.invoke.return_value = {
        "messages": [
            HumanMessage(content="Hola"),
            AIMessage(content="Hola")
        ],
        "tools_used": []
    }
    
    with patch.object(assistant, "_build_app", return_value=mock_app):
        # First query
        assistant.query("Hola", session_id="user_1")
        assert "user_1" in assistant.sessions
        assert len(assistant.sessions["user_1"]) > 0
        
        # Second query same session
        assistant.query("¿Cómo estás?", session_id="user_1")
        assert len(assistant.sessions["user_1"]) > 1
