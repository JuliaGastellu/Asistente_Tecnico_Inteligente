import re
import math
from langchain_core.tools import tool
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@tool
def calculate(expression: str) -> str:
    """Calcula el resultado de una expresión matemática. 
    Usa esta herramienta para cualquier cálculo numérico, porcentajes o raíces cuadradas.
    Formato esperado: '2 + 2', '15% de 200', 'sqrt(16)'.
    """
    logger.info(f"Calculating expression: {expression}")
    expr = expression.lower().strip()
    
    # Handle percentage: "X% de Y" -> "Y * (X/100)"
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*de\s*(\d+(?:\.\d+)?)', expr)
    if percent_match:
        x, y = percent_match.groups()
        expr = f"{y} * ({x} / 100)"
    
    # Replace sqrt with math.sqrt
    expr = expr.replace('sqrt', 'math.sqrt')
    
    # Safe eval namespace
    allowed_names = {
        "__builtins__": {},
        "math": math
    }
    
    try:
        # Check for division by zero before eval if possible or handle in eval
        result = eval(expr, allowed_names)
        
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            return f"{result:.2f}"
        return str(result)
        
    except ZeroDivisionError:
        return "Error: División por cero."
    except Exception as e:
        logger.error(f"Error in calculator: {e}")
        return f"Error al calcular: Expresión inválida o no soportada."
