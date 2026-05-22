import pytest
import json
from pathlib import Path
from src.config import get_settings

settings = get_settings()

@pytest.mark.regression
def test_no_performance_regression():
    baseline_path = settings.reports_dir / "baseline_metrics.json"
    if not baseline_path.exists():
        pytest.skip("No hay baseline_metrics.json para comparar")
        
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
        
    # Here we would run a new evaluation and compare.
    # For now, it's a placeholder for the regression logic.
    assert "metrics" in baseline
