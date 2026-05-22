import re
import time
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
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

class TechnicalAssistant:
    def __init__(self):
        self.tools = [search_documents, calculate, get_weather, search_web]
        self.sessions: Dict[str, ConversationBufferMemory] = {}
        self.cache = get_cache()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def _get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.sessions[session_id]

    def _build_executor(self, session_id: str, query: str) -> AgentExecutor:
        llm = get_llm_for_query(query)
        memory = self._get_or_create_memory(session_id)
        
        agent = create_openai_functions_agent(llm, self.tools, self.prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            max_iterations=5,
            handle_parsing_errors=True,
            verbose=True
        )

    def query(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        # Check cache
        cached_result = self.cache.get(user_input)
        if cached_result:
            logger.info("Cache HIT")
            return cached_result

        logger.info(f"Processing query for session {session_id}: {user_input[:50]}...")
        executor = self._build_executor(session_id, user_input)
        
        try:
            start_time = time.time()
            response = executor.invoke({"input": user_input})
            latency = time.time() - start_time
            
            answer = response["output"]
            
            # Extract tools used and sources
            tools_used = []
            # In AgentExecutor, we can look at intermediate_steps if return_intermediate_steps=True
            # But the user didn't specify return_intermediate_steps=True.
            # I'll enable it to extract tools_used.
            
            # Re-build with intermediate steps for extraction
            # Wait, I'll just look for tool names in the answer or better, 
            # I'll modify _build_executor to return intermediate steps.
            
            # Let's just extract sources with regex as requested
            sources = re.findall(r'\[Fuente \d+: (.+?)\]', answer)
            
            # For tools_used, I'll check if they are mentioned in the output or 
            # better yet, I'll use a callback or just check intermediate steps.
            # Actually, let's use intermediate steps.
            
            # Modifying _build_executor to include intermediate steps if I want to extract tools_used
            # But I can also just check which tools were called during the execution.
            
            # Let's refine the query method to get tools_used.
            
            # To get tools_used, I'll use return_intermediate_steps=True
            executor.return_intermediate_steps = True
            response = executor.invoke({"input": user_input})
            
            tools_used = list(set([step[0].tool for step in response["intermediate_steps"]]))
            answer = response["output"]
            sources = list(set(re.findall(r'\[Fuente \d+: (.+?)\]', str(response["intermediate_steps"]))))
            
            result = {
                "answer": answer,
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
