import re
import time
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent.model_selector import get_llm_for_query
from src.agent.prompts import SYSTEM_PROMPT
from src.tools.rag_tool import search_documents
from src.tools.calculator_tool import calculate
from src.tools.weather_tool import get_weather
from src.tools.search_tool import search_web
from src.utils.logger import setup_logger
from src.utils.cache import get_cache

logger = setup_logger(__name__)

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
        if last_message.tool_calls:
            return "tools"
        return END

    def _call_model(self, state: AgentState, config: Optional[Any] = None):
        messages = state["messages"]
        query = messages[0].content if messages else ""
        llm = self._get_llm(query)
        response = llm.invoke(messages, config=config)
        
        # Track tools used
        tools_used = state.get("tools_used", [])
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] not in tools_used:
                    tools_used.append(tc["name"])
        
        return {"messages": [response], "tools_used": tools_used}

    def _build_app(self, query: str):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    def query(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        # Check cache
        cached_result = self.cache.get(user_input)
        if cached_result:
            logger.info("Cache HIT")
            return cached_result

        logger.info(f"Processing query for session {session_id}: {user_input[:50]}...")
        
        if session_id not in self.sessions:
            self.sessions[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
            
        history = self.sessions[session_id]
        history.append(HumanMessage(content=user_input))
        
        app = self._build_app(user_input)
        
        try:
            start_time = time.time()
            state = {"messages": history, "tools_used": [], "sources": []}
            result_state = app.invoke(state)
            latency = time.time() - start_time
            
            final_answer = result_state["messages"][-1].content
            tools_used = result_state.get("tools_used", [])
            
            # Extract sources from all messages (tool outputs)
            sources = []
            for msg in result_state["messages"]:
                if hasattr(msg, "content"):
                    found = re.findall(r'\[Fuente \d+: (.+?)\]', str(msg.content))
                    sources.extend(found)
            
            sources = list(set(sources))
            
            # Update history
            self.sessions[session_id] = result_state["messages"]
            
            result = {
                "answer": final_answer,
                "tools_used": tools_used,
                "sources": sources,
                "latency": latency
            }
            
            self.cache.set(user_input, result)
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

_assistant = None

def get_assistant() -> TechnicalAssistant:
    global _assistant
    if _assistant is None:
        _assistant = TechnicalAssistant()
    return _assistant
