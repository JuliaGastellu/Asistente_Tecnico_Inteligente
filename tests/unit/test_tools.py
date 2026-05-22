import pytest
from unittest.mock import MagicMock, patch
from src.tools.rag_tool import OptimizedRetriever
from src.tools.calculator_tool import calculate
from src.tools.weather_tool import get_weather

def test_estimate_complexity():
    retriever = OptimizedRetriever(MagicMock())
    assert retriever.estimate_complexity("hola") == 3
    assert retriever.estimate_complexity("diferencia entre x e y") == 10
    assert retriever.estimate_complexity("cómo hacer un endpoint") == 7
    assert retriever.estimate_complexity("esto es una consulta normal de prueba") == 5

def test_calculate_percentage():
    assert calculate.invoke("15% de 200") == "30"

def test_calculate_sqrt():
    assert calculate.invoke("sqrt(144)") == "12"

def test_calculate_div_zero():
    assert calculate.invoke("10 / 0") == "Error: División por cero."

def test_calculate_invalid():
    assert "Error" in calculate.invoke("invalid expression")

@patch("requests.get")
def test_get_weather_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "main": {"temp": 25, "humidity": 50},
        "weather": [{"description": "cielo claro"}]
    }
    mock_get.return_value = mock_response
    
    with patch("src.tools.weather_tool.get_settings") as mock_settings:
        mock_settings.return_value.openweather_api_key = "test_key"
        result = get_weather.invoke("Madrid")
        assert "25°C" in result
        assert "Madrid" in result

@patch("requests.get")
def test_get_weather_not_found(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value.raise_for_status.side_effect = Exception("404 Client Error")
    mock_get.return_value.status_code = 404
    
    # Simulating requests.exceptions.HTTPError
    import requests
    mock_get.side_effect = requests.exceptions.HTTPError(response=mock_response)
    
    with patch("src.tools.weather_tool.get_settings") as mock_settings:
        mock_settings.return_value.openweather_api_key = "test_key"
        result = get_weather.invoke("CiudadInexistente")
        assert "No se encontró la ciudad" in result
