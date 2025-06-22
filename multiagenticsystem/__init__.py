"""
MultiAgenticSystem - A powerful LangGraph-based multi-agent system
with dynamic configuration and hierarchical tool sharing.
"""

from .core.agent import Agent
from .core.tool import Tool
from .core.task import Task, Collaboration
from .core.trigger import Trigger
from .core.automation import Automation
from .core.system import System

__version__ = "0.1.0"
__author__ = "MultiAgenticSystem Team"
__email__ = "contact@multiagenticsystem.dev"

__all__ = [
    "Agent",
    "Tool", 
    "Task",
    "Collaboration",
    "Trigger",
    "Automation",
    "System",
]
