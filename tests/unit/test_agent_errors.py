import pytest
from unittest.mock import MagicMock, patch
from src.agent.agent import TechnicalAssistant

def test_query_empty_no_crash():
    assistant = TechnicalAssistant()
    # Mocking _build_executor to avoid actual LLM calls
    assistant._build_executor = MagicMock()
    result = assistant.query("")
    # Cache should return None, then it goes to executor
    assert "answer" in result

def test_query_too_long_no_crash():
    assistant = TechnicalAssistant()
    assistant._build_executor = MagicMock()
    result = assistant.query("a" * 1000)
    assert "answer" in result

@patch("src.agent.agent.get_llm_for_query")
def test_tool_failure_graceful(mock_llm):
    assistant = TechnicalAssistant()
    # Mock executor to raise exception
    mock_executor = MagicMock()
    mock_executor.invoke.side_effect = Exception("Tool failed")
    assistant._build_executor = MagicMock(return_value=mock_executor)
    
    result = assistant.query("consulta que fallará")
    assert result["error"] is True
    assert "Lo siento" in result["answer"]
