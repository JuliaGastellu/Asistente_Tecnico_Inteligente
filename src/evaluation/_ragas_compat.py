"""
Shim de compatibilidad para ragas 0.4.3 sobre langchain-community 0.4.2.

ragas/llms/base.py hace `from langchain_community.chat_models.vertexai import ChatVertexAI`,
pero ese módulo fue eliminado de langchain-community 0.4.x (Vertex se movió a
langchain-google-vertexai). Como este proyecto usa OpenAI exclusivamente y nunca
instancia Vertex, registramos un módulo stub con un `ChatVertexAI` placeholder para que
el import de ragas no falle.

IMPORTANTE: importar este módulo ANTES que cualquier import de ragas.
Mantiene todas las versiones pinneadas intactas (no requiere reinstalar nada).
"""

import sys
import types
import importlib


def install() -> None:
    module_name = "langchain_community.chat_models.vertexai"

    # Si el módulo real existe (otra versión de langchain-community), no tocar nada.
    try:
        importlib.import_module(module_name)
        return
    except ImportError:
        pass

    stub = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: D401 - placeholder no funcional
        """Stub: Vertex no se usa en este proyecto (solo OpenAI).

        Existe únicamente para satisfacer el import de ragas. Instanciarlo es un error.
        """

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "ChatVertexAI no está soportado en este entorno. "
                "Este proyecto usa OpenAI; el stub solo permite importar ragas."
            )

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = stub


install()
