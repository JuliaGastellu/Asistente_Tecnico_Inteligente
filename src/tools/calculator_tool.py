import re
import ast
import math
import operator
from langchain_core.tools import tool
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Evaluador matemático seguro basado en AST (reemplaza eval(), ERR-019).
# Solo se permiten operadores aritméticos, un set acotado de funciones de `math`
# y constantes. Cualquier otro nodo (nombres arbitrarios, atributos, llamadas no
# whitelisteadas, etc.) se rechaza.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil, "pow": math.pow,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Constante no numérica")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise ValueError("Argumentos por palabra clave no soportados")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("Expresión no soportada")


def _safe_eval(expr: str):
    return _eval_node(ast.parse(expr, mode="eval").body)


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

    try:
        result = _safe_eval(expr)

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
