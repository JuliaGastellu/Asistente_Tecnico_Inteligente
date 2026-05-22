import pytest
from src.evaluation.evaluator import SystemEvaluator

def test_calculate_tool_accuracy():
    evaluator = SystemEvaluator(None)
    results = [
        {"expected_tools": ["a", "b"], "tools_used": ["a", "b", "c"], "error": False},
        {"expected_tools": ["a"], "tools_used": ["b"], "error": False},
        {"expected_tools": ["a"], "tools_used": ["a"], "error": True} # Should be ignored
    ]
    # Only 2 non-error cases. 1 correct. Accuracy = 0.5
    assert evaluator.calculate_tool_accuracy(results) == 0.5

def test_calculate_error_rate():
    evaluator = SystemEvaluator(None)
    results = [
        {"error": False},
        {"error": True},
        {"error": False},
        {"error": False}
    ]
    assert evaluator.calculate_error_rate(results) == 0.25
