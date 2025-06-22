"""
Main system orchestrator that manages agents, tools, tasks, and automations.
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .agent import Agent
from .tool import Tool, create_logger_tool, create_memory_tool
from .task import Task, Collaboration
from .trigger import Trigger
from .automation import Automation
from ..utils.logger import get_logger

logger = get_logger(__name__)


class System:
    """
    The main orchestration system for multi-agent workflows.
    
    The System class manages:
    - Agent registry and lifecycle
    - Tool registry with hierarchical sharing
    - Task execution and coordination
    - Event-driven automations
    - Configuration loading and management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the system.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Core registries
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}
        self.tasks: Dict[str, Task] = {}
        self.triggers: Dict[str, Trigger] = {}
        self.automations: Dict[str, Automation] = {}
        self.collaborations: Dict[str, Collaboration] = {}
        
        # Runtime state
        self.running = False
        self.event_queue: List[Dict[str, Any]] = []
        self.execution_context: Dict[str, Any] = {}
        
        # Add built-in tools
        self._add_builtin_tools()
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        logger.info("MultiAgenticSystem initialized")
    
    def _add_builtin_tools(self) -> None:
        """Add built-in utility tools to the system."""
        logger_tool = create_logger_tool()
        memory_tool = create_memory_tool()
        
        self.tools[logger_tool.name] = logger_tool
        self.tools[memory_tool.name] = memory_tool
        
        logger.debug("Added built-in tools: Logger, StoreMemory")
    
    # Agent Management
    def register_agent(self, agent: Agent) -> None:
        """Register a single agent."""
        self.agents[agent.name] = agent
        self._update_agent_tools(agent)
        logger.info(f"Registered agent: {agent.name}")
    
    def register_agents(self, *agents: Agent) -> None:
        """Register multiple agents."""
        for agent in agents:
            self.register_agent(agent)
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agents.keys())
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the system."""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Removed agent: {name}")
            return True
        return False
    
    # Tool Management
    def register_tool(self, tool: Tool) -> None:
        """Register a single tool."""
        self.tools[tool.name] = tool
        self._update_all_agent_tools()
        logger.info(f"Registered tool: {tool.name} (scope: {tool.scope.value})")
    
    def register_tools(self, *tools: Tool) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register_tool(tool)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the system."""
        if name in self.tools:
            del self.tools[name]
            self._update_all_agent_tools()
            logger.info(f"Removed tool: {name}")
            return True
        return False
    
    def _update_agent_tools(self, agent: Agent) -> None:
        """Update an agent's tool access lists."""
        agent.local_tools.clear()
        agent.shared_tools.clear()
        agent.global_tools.clear()
        
        for tool_name, tool in self.tools.items():
            if tool.is_global:
                agent.global_tools.append(tool_name)
            elif tool.local_agent == agent.name:
                agent.local_tools.append(tool_name)
            elif agent.name in tool.shared_agents:
                agent.shared_tools.append(tool_name)
    
    def _update_all_agent_tools(self) -> None:
        """Update tool access for all agents."""
        for agent in self.agents.values():
            self._update_agent_tools(agent)
    
    # Task Management
    def register_task(self, task: Task) -> None:
        """Register a single task."""
        self.tasks[task.name] = task
        logger.info(f"Registered task: {task.name}")
    
    def register_tasks(self, *tasks: Task) -> None:
        """Register multiple tasks."""
        for task in tasks:
            self.register_task(task)
    
    def get_task(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        return self.tasks.get(name)
    
    def list_tasks(self) -> List[str]:
        """List all registered task names."""
        return list(self.tasks.keys())
    
    def remove_task(self, name: str) -> bool:
        """Remove a task from the system."""
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Removed task: {name}")
            return True
        return False
    
    # Trigger Management
    def register_trigger(self, trigger: Trigger) -> None:
        """Register a single trigger."""
        self.triggers[trigger.name] = trigger
        logger.info(f"Registered trigger: {trigger.name}")
    
    def register_triggers(self, *triggers: Trigger) -> None:
        """Register multiple triggers."""
        for trigger in triggers:
            self.register_trigger(trigger)
    
    def get_trigger(self, name: str) -> Optional[Trigger]:
        """Get a trigger by name."""
        return self.triggers.get(name)
    
    def list_triggers(self) -> List[str]:
        """List all registered trigger names."""
        return list(self.triggers.keys())
    
    def remove_trigger(self, name: str) -> bool:
        """Remove a trigger from the system."""
        if name in self.triggers:
            del self.triggers[name]
            logger.info(f"Removed trigger: {name}")
            return True
        return False
    
    # Automation Management
    def register_automation(self, automation: Automation) -> None:
        """Register a single automation."""
        self.automations[automation.name] = automation
        logger.info(f"Registered automation: {automation.name}")
    
    def register_automations(self, *automations: Automation) -> None:
        """Register multiple automations."""
        for automation in automations:
            self.register_automation(automation)
    
    def get_automation(self, name: str) -> Optional[Automation]:
        """Get an automation by name."""
        return self.automations.get(name)
    
    def list_automations(self) -> List[str]:
        """List all registered automation names."""
        return list(self.automations.keys())
    
    def remove_automation(self, name: str) -> bool:
        """Remove an automation from the system."""
        if name in self.automations:
            del self.automations[name]
            logger.info(f"Removed automation: {name}")
            return True
        return False
    
    # Collaboration Management
    def register_collaboration(self, collaboration: Collaboration) -> None:
        """Register a collaboration."""
        self.collaborations[collaboration.name] = collaboration
        logger.info(f"Registered collaboration: {collaboration.name}")
    
    def register_collaborations(self, *collaborations: Collaboration) -> None:
        """Register multiple collaborations."""
        for collaboration in collaborations:
            self.register_collaboration(collaboration)
    
    # Event Processing
    def emit_event(self, event: Dict[str, Any]) -> None:
        """Emit an event to the system."""
        self.event_queue.append(event)
        logger.debug(f"Emitted event: {event.get('type', 'unknown')}")
    
    async def process_events(self) -> None:
        """Process pending events and trigger automations."""
        while self.event_queue:
            event = self.event_queue.pop(0)
            await self._process_single_event(event)
    
    async def _process_single_event(self, event: Dict[str, Any]) -> None:
        """Process a single event."""
        logger.debug(f"Processing event: {event}")
        
        # Check all triggers
        for trigger in self.triggers.values():
            if trigger.evaluate(event):
                trigger.fire(event)
                
                # Find automations for this trigger
                for automation in self.automations.values():
                    if automation.trigger_name == trigger.name:
                        await automation.execute(event, self.execution_context, self.tasks)
    
    # Execution Methods
    async def execute_task(
        self,
        task_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a specific task."""
        task = self.get_task(task_name)
        if not task:
            raise ValueError(f"Task '{task_name}' not found")
        
        logger.info(f"Executing task: {task_name}")
        
        results = []
        task.status = task.TaskStatus.RUNNING if hasattr(task, 'TaskStatus') else "running"
        
        for step in task.steps:
            agent = self.get_agent(step.agent)
            if not agent:
                raise ValueError(f"Agent '{step.agent}' not found for task '{task_name}'")
            
            # Get available tools for this agent
            available_tools = agent.get_available_tools(self.tools)
            
            # Execute the step
            try:
                result = await agent.execute(
                    input_text=step.input_data,
                    context=context or {},
                    available_tools=available_tools
                )
                
                results.append(result)
                task.mark_step_completed(result)
                
            except Exception as e:
                task.mark_step_failed(str(e))
                logger.error(f"Task step failed: {e}")
                break
        
        task.status = "completed" if task.is_completed() else "failed"
        
        return {
            "task_name": task_name,
            "status": task.status,
            "results": results,
            "success": task.is_completed()
        }
    
    async def execute_agent(
        self,
        agent_name: str,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a single agent."""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        available_tools = agent.get_available_tools(self.tools)
        return await agent.execute(input_text, context or {}, available_tools)
    
    # Configuration Management
    def load_config(self, config_path: str) -> None:
        """Load system configuration from file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            self._apply_config(config)
            logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration to the system."""
        # Load agents
        for agent_config in config.get('agents', []):
            agent = Agent.from_dict(agent_config)
            self.register_agent(agent)
        
        # Load tools (function implementations need to be provided separately)
        for tool_config in config.get('tools', []):
            tool = Tool.from_dict(tool_config)
            self.register_tool(tool)
        
        # Load tasks
        for task_config in config.get('tasks', []):
            task = Task.from_dict(task_config)
            self.register_task(task)
        
        # Load triggers
        for trigger_config in config.get('triggers', []):
            trigger = Trigger.from_dict(trigger_config)
            self.register_trigger(trigger)
        
        # Load automations
        for automation_config in config.get('automations', []):
            automation = Automation.from_dict(automation_config, self.triggers)
            self.register_automation(automation)
    
    def save_config(self, config_path: str) -> None:
        """Save current system configuration to file."""
        config = {
            'agents': [agent.to_dict() for agent in self.agents.values()],
            'tools': [tool.to_dict() for tool in self.tools.values()],
            'tasks': [task.to_dict() for task in self.tasks.values()],
            'triggers': [trigger.to_dict() for trigger in self.triggers.values()],
            'automations': [automation.to_dict() for automation in self.automations.values()]
        }
        
        config_file = Path(config_path)
        with open(config_file, 'w') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to: {config_path}")
    
    # System Control
    def run(self) -> None:
        """Run the system in blocking mode."""
        asyncio.run(self.run_async())
    
    async def run_async(self) -> None:
        """Run the system asynchronously."""
        self.running = True
        logger.info("Starting MultiAgenticSystem...")
        
        try:
            while self.running:
                await self.process_events()
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        self.running = False
        logger.info("MultiAgenticSystem shut down")
    
    # System Information
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "running": self.running,
            "agents": len(self.agents),
            "tools": len(self.tools),
            "tasks": len(self.tasks),
            "triggers": len(self.triggers),
            "automations": len(self.automations),
            "collaborations": len(self.collaborations),
            "pending_events": len(self.event_queue)
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information."""
        return {
            "status": self.get_system_status(),
            "agents": {name: agent.to_dict() for name, agent in self.agents.items()},
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "tasks": {name: task.to_dict() for name, task in self.tasks.items()},
            "triggers": {name: trigger.to_dict() for name, trigger in self.triggers.items()},
            "automations": {name: automation.to_dict() for name, automation in self.automations.items()}
        }
    
    @classmethod
    def from_config(cls, config_path: str) -> "System":
        """Create a system instance from configuration file."""
        system = cls()
        system.load_config(config_path)
        return system
    
    def __repr__(self) -> str:
        return f"System(agents={len(self.agents)}, tools={len(self.tools)}, tasks={len(self.tasks)})"
