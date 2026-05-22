import requests
from langchain_core.tools import tool
from src.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@tool
def get_weather(city: str) -> str:
    """Obtiene el clima actual de una ciudad. 
    Usa esta herramienta cuando el usuario pregunte por el tiempo o la temperatura en un lugar específico.
    """
    settings = get_settings()
    api_key = settings.openweather_api_key
    
    if not api_key:
        return "Error: La API key de OpenWeather no está configurada."
        
    logger.info(f"Getting weather for city: {city}")
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "es"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        
        return f"El clima en {city} es {desc} con una temperatura de {temp}°C y una humedad del {humidity}%."
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: No se encontró la ciudad '{city}'."
        return f"Error de conexión con el servicio de clima: {str(e)}"
    except requests.exceptions.Timeout:
        return "Error: Tiempo de espera agotado al consultar el clima."
    except Exception as e:
        logger.error(f"Error in weather tool: {e}")
        return f"Error al obtener el clima: {str(e)}"
