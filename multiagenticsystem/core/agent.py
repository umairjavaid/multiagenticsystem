"""
Core agent implementation with LLM provider abstraction.
"""

import uuid
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .tool import Tool

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback base class
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def Field(**kwargs):
        return kwargs.get('default', None)

from ..llm.providers import LLMProvider, get_llm_provider
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    description: str = ""
    system_prompt: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = 10
    memory_enabled: bool = True
    tools: List[str] = Field(default_factory=list)


class Agent:
    """
    A multi-agent system agent with pluggable LLM backend support.
    
    Each agent can be configured with different LLM providers and has
    access to a hierarchical tool system (local, shared, global).
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        system_prompt: str = "",
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        llm_config: Optional[Dict[str, Any]] = None,
        max_iterations: int = 10,
        memory_enabled: bool = True,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize an agent.
        
        Args:
            name: Unique name for the agent
            description: Description of the agent's purpose
            system_prompt: System prompt to guide the agent's behavior
            llm_provider: LLM provider (openai, anthropic, aws, azure, etc.)
            llm_model: Specific model to use
            llm_config: Additional LLM configuration parameters
            max_iterations: Maximum iterations for complex tasks
            memory_enabled: Whether to maintain conversation memory
            agent_id: Optional custom agent ID
        """
        self.id = agent_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm_provider_name = llm_provider
        self.llm_model = llm_model
        self.llm_config = llm_config or {}
        self.max_iterations = max_iterations
        self.memory_enabled = memory_enabled
        
        # Tool access tracking
        self.local_tools: List[str] = []
        self.shared_tools: List[str] = []
        self.global_tools: List[str] = []
        
        # Runtime state
        self.memory: List[Dict[str, Any]] = []
        self.execution_context: Dict[str, Any] = {}
        self._llm_provider: Optional[LLMProvider] = None
        
        logger.info(f"Created agent '{name}' with {llm_provider}/{llm_model}")
    
    @property
    def llm_provider(self) -> LLMProvider:
        """Get the LLM provider instance."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider(
                provider=self.llm_provider_name,
                model=self.llm_model,
                **self.llm_config
            )
        return self._llm_provider
    
    def add_to_memory(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the agent's memory."""
        if not self.memory_enabled:
            return
            
        self.memory.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": logger.name  # Using logger.name as a placeholder
        })
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory.clear()
        logger.debug(f"Cleared memory for agent '{self.name}'")
    
    def get_available_tools(self, tool_registry: Dict[str, Any]) -> List[str]:
        """Get all tools available to this agent."""
        available = []
        
        # Add local tools
        available.extend(self.local_tools)
        
        # Add shared tools where this agent has access
        for tool_name in self.shared_tools:
            if tool_name in tool_registry:
                tool = tool_registry[tool_name]
                if hasattr(tool, 'shared_agents') and self.name in tool.shared_agents:
                    available.append(tool_name)
        
        # Add global tools
        available.extend(self.global_tools)
        
        return list(set(available))  # Remove duplicates
    
    """
    Fixed Agent execution method that handles tool calls
    """

    async def execute(
        self, 
        input_text: str, 
        context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None,
        tool_registry: Optional[Dict[str, "Tool"]] = None
    ) -> Dict[str, Any]:
        """Execute a task with tool support."""
        import time
        start_time = time.time()
        
        # Log the agent action start
        logger.log_agent_action(
            agent_name=self.name,
            action="execute",
            input_data=input_text,
            context={
                "available_tools": available_tools,
                "context": context
            }
        )
        
        try:
            # Add input to memory
            self.add_to_memory("user", input_text)
            
            # Prepare context with available tools
            execution_context = context or {}
            if available_tools:
                execution_context["available_tools"] = available_tools
            
            # Prepare messages for LLM
            messages = []
            
            # Add system prompt
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            
            # Add conversation history
            for msg in self.memory[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Log the LLM call details
            logger.log_system_event("agent_llm_call", {
                "agent": self.name,
                "provider": self.llm_provider_name,
                "model": self.llm_model,
                "message_count": len(messages),
                "context_keys": list(execution_context.keys())
            })
            
            # Execute with LLM provider
            response = await self.llm_provider.execute(
                messages=messages,
                context=execution_context
            )
            
            # Handle tool calls if present
            tool_results = []
            if hasattr(response, 'metadata') and 'tool_calls' in response.metadata:
                tool_calls = response.metadata.get('tool_calls', [])
                
                for tool_call in tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['arguments']
                    
                    # Log tool call
                    logger.log_tool_execution(
                        tool_name=tool_name,
                        agent_name=self.name,
                        parameters=tool_args
                    )
                    
                    # Execute the tool if available
                    if tool_registry and tool_name in tool_registry:
                        tool = tool_registry[tool_name]
                        tool_start_time = time.time()
                        
                        try:
                            # Execute tool with arguments
                            if tool_name == "Terminal":
                                result = await tool.execute(
                                    self,
                                    command=tool_args.get('command'),
                                    working_dir=tool_args.get('working_dir')
                                )
                            elif tool_name == "FileManager":
                                result = await tool.execute(
                                    self,
                                    file_path=tool_args.get('file_path'),
                                    content=tool_args.get('content')
                                )
                            else:
                                result = await tool.execute(self, **tool_args)
                            
                            tool_execution_time = time.time() - tool_start_time
                            
                            # Log tool result
                            logger.log_tool_execution(
                                tool_name=tool_name,
                                agent_name=self.name,
                                parameters=tool_args,
                                result=result,
                                execution_time=tool_execution_time
                            )
                            
                            tool_results.append({
                                "tool": tool_name,
                                "result": result
                            })
                            
                            # Add tool result to conversation
                            self.add_to_memory(
                                "tool",
                                f"Tool {tool_name} result: {result.get('result', 'completed')}"
                            )
                            
                        except Exception as tool_error:
                            logger.log_system_event("tool_error", {
                                "tool": tool_name,
                                "agent": self.name,
                                "error": str(tool_error)
                            }, level="ERROR")
                            
                            tool_results.append({
                                "tool": tool_name,
                                "error": str(tool_error)
                            })
                    else:
                        logger.log_system_event("tool_not_found", {
                            "tool": tool_name,
                            "agent": self.name,
                            "available_tools": list(tool_registry.keys()) if tool_registry else []
                        }, level="WARNING")
                
                # If tools were called, get a final response
                if tool_results:
                    # Add tool results to messages
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    for tool_result in tool_results:
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_result)
                        })
                    
                    # Get final response after tool execution
                    final_response = await self.llm_provider.execute(
                        messages=messages,
                        context=execution_context
                    )
                    
                    response_content = final_response.content
                else:
                    response_content = response.content
            else:
                response_content = response.content
            
            # Add response to memory
            self.add_to_memory("assistant", response_content)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = {
                "agent_id": self.id,
                "agent_name": self.name,
                "input": input_text,
                "output": response_content,
                "metadata": response.metadata if hasattr(response, 'metadata') else {},
                "tool_results": tool_results,
                "execution_time": execution_time,
                "success": True
            }
            
            # Log successful completion
            logger.log_agent_action(
                agent_name=self.name,
                action="execute_complete",
                input_data=input_text,
                output_data=response_content,
                context={
                    "execution_time": execution_time,
                    "tools_used": [tr["tool"] for tr in tool_results]
                }
            )
            
            logger.log_system_event("agent_execution_success", {
                "agent": self.name,
                "execution_time": execution_time,
                "tools_used": len(tool_results),
                "output_length": len(response_content)
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            logger.log_agent_action(
                agent_name=self.name,
                action="execute_error",
                input_data=input_text,
                context={
                    "error": str(e),
                    "execution_time": execution_time
                }
            )
            
            logger.log_system_event("agent_execution_error", {
                "agent": self.name,
                "error": str(e),
                "execution_time": execution_time
            }, level="ERROR")
            
            return {
                "agent_id": self.id,
                "agent_name": self.name,
                "input": input_text,
                "output": "",
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "llm_provider": self.llm_provider_name,
            "llm_model": self.llm_model,
            "llm_config": self.llm_config,
            "max_iterations": self.max_iterations,
            "memory_enabled": self.memory_enabled,
            "local_tools": self.local_tools,
            "shared_tools": self.shared_tools,
            "global_tools": self.global_tools
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create agent from dictionary representation."""
        agent = cls(
            name=data["name"],
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            llm_provider=data.get("llm_provider", "openai"),
            llm_model=data.get("llm_model", "gpt-3.5-turbo"),
            llm_config=data.get("llm_config", {}),
            max_iterations=data.get("max_iterations", 10),
            memory_enabled=data.get("memory_enabled", True),
            agent_id=data.get("id")
        )
        
        # Restore tool assignments
        agent.local_tools = data.get("local_tools", [])
        agent.shared_tools = data.get("shared_tools", [])
        agent.global_tools = data.get("global_tools", [])
        
        return agent
    
    @classmethod
    def from_config(cls, config: AgentConfig) -> "Agent":
        """Create agent from configuration object."""
        return cls.from_dict(config.model_dump())
    
    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', llm='{self.llm_provider_name}/{self.llm_model}')"
