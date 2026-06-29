import re
import time
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from src.agent.model_selector import get_llm_for_query, _get_client
from src.agent.prompts import SYSTEM_PROMPT
from src.tools.rag_tool import search_documents
from src.tools.calculator_tool import calculate
from src.tools.weather_tool import get_weather
from src.tools.search_tool import search_web
from src.utils.logger import setup_logger
from src.utils.cache import get_cache
from src.config import get_settings

logger = setup_logger(__name__)


def _build_llm(model_name: str = None):
    """Construye el LLM según el entorno:
    - Si OPENROUTER_API_KEY está definida -> usa OpenRouter (gateway).
    - Si no -> usa OpenAI directo (comportamiento original).
    Delega en model_selector._get_client (cacheado y consciente de OpenRouter).
    """
    settings = get_settings()
    if settings.openrouter_api_key:
        return _get_client(model_name or settings.openrouter_model)
    return _get_client(model_name or settings.default_model)

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    tools_used: List[str]
    sources: List[str]

class TechnicalAssistant:
    def __init__(self):
        self.tools = [search_documents, calculate, get_weather, search_web]
        self.tool_node = ToolNode(self.tools)
        self.sessions: Dict[str, List[BaseMessage]] = {}
        self.cache = get_cache()
        
    def _get_llm(self, query: str):
        return get_llm_for_query(query).bind_tools(self.tools)

    def _should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # getattr defensivo: solo los AIMessage traen tool_calls; un ToolMessage u
        # otro tipo no lo tienen y romperían el routing del grafo.
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    def _call_model(self, state: AgentState, config: Optional[RunnableConfig] = None):
        messages = state["messages"]
        # Use the last HumanMessage for complexity estimation; messages[0] is SystemMessage
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        query = user_msgs[-1].content if user_msgs else ""
        llm = self._get_llm(query)
        response = llm.invoke(messages, config=config)
        
        # Track tools used
        tools_used = state.get("tools_used", [])
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] not in tools_used:
                    tools_used.append(tc["name"])
        
        return {"messages": [response], "tools_used": tools_used}

    def _build_app(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    def query(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        # La clave incluye session_id: el cache es por sesión, evitando que la
        # respuesta de una conversación se sirva a otra (cross-session bleed).
        cache_key = f"{session_id}::{user_input}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Cache HIT")
            return cached_result

        logger.info(f"Processing query for session {session_id}: {user_input[:50]}...")
        
        if session_id not in self.sessions:
            self.sessions[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
            
        history = self.sessions[session_id]
        history.append(HumanMessage(content=user_input))
        
        app = self._build_app()
        
        try:
            start_time = time.time()
            state = {"messages": history, "tools_used": [], "sources": []}
            result_state = app.invoke(state)
            latency = time.time() - start_time
            
            final_answer = result_state["messages"][-1].content
            tools_used = result_state.get("tools_used", [])
            
            # Extract sources (filenames) and contexts (texto real recuperado) de los
            # outputs de search_documents. `contexts` es lo que RAGAS necesita para
            # faithfulness/answer_relevancy (ver evaluator.py); `sources` son solo nombres.
            sources: List[str] = []
            contexts: List[str] = []
            for msg in result_state["messages"]:
                content = str(getattr(msg, "content", ""))
                if not content:
                    continue
                sources.extend(re.findall(r'\[Fuente \d+: (.+?)\]', content))
                # Solo los ToolMessage de búsqueda documental aportan contexto RAG
                is_rag_tool = isinstance(msg, ToolMessage) and (
                    getattr(msg, "name", None) == "search_documents" or "[Fuente" in content
                )
                if is_rag_tool:
                    for block in content.split("\n\n---\n\n"):
                        block = block.strip()
                        if block:
                            contexts.append(block)

            sources = list(set(sources))

            # Update history
            self.sessions[session_id] = result_state["messages"]

            result = {
                "answer": final_answer,
                "tools_used": tools_used,
                "sources": sources,
                "contexts": contexts,
                "latency": latency
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}",
                "tools_used": [],
                "sources": [],
                "error": True
            }

    def reset_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} reset.")

# Singleton de proceso. Suficiente para un único worker; bajo concurrencia real
# (varios contenedores/invocaciones Lambda) cada proceso tiene su propia instancia
# y sus propias sesiones/caché en memoria — aceptado como límite conocido (ERR-022).
_assistant = None

def get_assistant() -> TechnicalAssistant:
    global _assistant
    if _assistant is None:
        _assistant = TechnicalAssistant()
    return _assistant
