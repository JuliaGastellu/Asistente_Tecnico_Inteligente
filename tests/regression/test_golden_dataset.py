import pytest
import json
from pathlib import Path
from src.agent.agent import get_assistant
from src.config import get_settings

settings = get_settings()

def load_golden_cases():
    path = settings.golden_dataset_path
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.mark.regression
@pytest.mark.parametrize("case", load_golden_cases())
def test_golden_case_tools(case):
    # This test would require a live environment or heavy mocking.
    # For now, we skip it if no API key is present or just mark it as regression.
    # In a real scenario, we would run this against a test environment.
    pytest.skip("Requiere entorno real con APIs configuradas")
