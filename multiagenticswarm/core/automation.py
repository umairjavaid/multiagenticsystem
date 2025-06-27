"""
Automation system that connects triggers to tasks.
"""

import uuid
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .trigger import Trigger
from .task import Task
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AutomationStatus(str, Enum):
    """Automation execution status."""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AutomationMode(str, Enum):
    """Automation execution mode."""
    IMMEDIATE = "immediate"    # Execute immediately when triggered
    QUEUED = "queued"         # Queue for later execution
    SCHEDULED = "scheduled"   # Schedule for specific time
    CONDITIONAL = "conditional" # Execute based on additional conditions


class Automation:
    """
    An automation connects triggers to tasks and manages their execution.
    
    When a trigger fires, the automation executes the associated task
    or sequence of tasks with the specified configuration.
    """
    
    def __init__(
        self,
        trigger: Union[Trigger, str],
        sequence: Union[Task, List[Task], str, List[str]],
        name: Optional[str] = None,
        description: str = "",
        mode: AutomationMode = AutomationMode.IMMEDIATE,
        conditions: Optional[Dict[str, Any]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        automation_id: Optional[str] = None
    ):
        """
        Initialize an automation.
        
        Args:
            trigger: Trigger that activates this automation
            sequence: Task or list of tasks to execute
            name: Name for the automation (auto-generated if None)
            description: Description of the automation
            mode: Execution mode (immediate, queued, scheduled, conditional)
            conditions: Additional conditions for execution
            retry_policy: Retry policy configuration
            automation_id: Optional custom automation ID
        """
        self.id = automation_id or str(uuid.uuid4())
        
        # Handle trigger
        if isinstance(trigger, str):
            self.trigger_name = trigger
            self.trigger: Optional[Trigger] = None
        else:
            self.trigger = trigger
            self.trigger_name = trigger.name
        
        # Handle sequence
        if isinstance(sequence, (str, Task)):
            self.task_sequence = [sequence]
        else:
            self.task_sequence = sequence
        
        # Convert task names to actual references later
        self.task_names = []
        self.tasks: List[Task] = []
        
        for task in self.task_sequence:
            if isinstance(task, str):
                self.task_names.append(task)
            else:
                self.tasks.append(task)
                self.task_names.append(task.name)
        
        self.name = name or f"Auto_{self.trigger_name}_{len(self.task_names)}"
        self.description = description
        self.mode = mode
        self.conditions = conditions or {}
        self.retry_policy = retry_policy or {"max_retries": 3, "delay": 1.0}
        
        # Execution tracking
        self.status = AutomationStatus.WAITING
        self.execution_count = 0
        self.last_execution: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Created automation '{self.name}' with trigger '{self.trigger_name}' and {len(self.task_names)} tasks")
    
    def can_execute(self, event: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Check if automation can execute based on conditions.
        
        Args:
            event: Triggering event
            context: Execution context
            
        Returns:
            True if automation can execute
        """
        if self.status == AutomationStatus.RUNNING:
            return False
        
        # Check additional conditions
        if self.conditions:
            for condition_name, condition_value in self.conditions.items():
                if condition_name in context:
                    if context[condition_name] != condition_value:
                        return False
                elif condition_name in event:
                    if event[condition_name] != condition_value:
                        return False
                else:
                    return False
        
        # Check mode-specific conditions
        if self.mode == AutomationMode.CONDITIONAL:
            # Additional conditional logic could be implemented here
            pass
        
        return True
    
    async def execute(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any],
        task_registry: Dict[str, Task]
    ) -> Dict[str, Any]:
        """
        Execute the automation.
        
        Args:
            event: Triggering event
            context: Execution context
            task_registry: Registry of available tasks
            
        Returns:
            Execution results
        """
        if not self.can_execute(event, context):
            return {
                "automation_id": self.id,
                "automation_name": self.name,
                "status": "skipped",
                "reason": "conditions not met"
            }
        
        self.status = AutomationStatus.RUNNING
        self.execution_count += 1
        self.last_execution = logger.name  # Placeholder for timestamp
        
        try:
            results = []
            
            # Execute tasks in sequence
            for task_name in self.task_names:
                task = task_registry.get(task_name)
                if not task:
                    raise ValueError(f"Task '{task_name}' not found in registry")
                
                # Execute task (this would integrate with the task execution system)
                task_result = {
                    "task_name": task_name,
                    "status": "simulated",  # Placeholder
                    "result": f"Executed task '{task_name}' in automation '{self.name}'"
                }
                
                results.append(task_result)
                
                logger.debug(f"Automation '{self.name}' executed task '{task_name}'")
            
            self.status = AutomationStatus.COMPLETED
            self.last_result = {
                "success": True,
                "tasks_executed": len(results),
                "results": results
            }
            
            # Add to execution history
            self.execution_history.append({
                "timestamp": self.last_execution,
                "event": str(event)[:100],
                "status": "completed",
                "tasks_executed": len(results),
                "success": True
            })
            
            logger.info(f"Automation '{self.name}' completed successfully")
            
            return {
                "automation_id": self.id,
                "automation_name": self.name,
                "status": "completed",
                "execution_count": self.execution_count,
                "results": results,
                "success": True
            }
            
        except Exception as e:
            self.status = AutomationStatus.FAILED
            self.error_count += 1
            self.last_error = str(e)
            
            # Add to execution history
            self.execution_history.append({
                "timestamp": self.last_execution,
                "event": str(event)[:100],
                "status": "failed",
                "error": str(e),
                "success": False
            })
            
            logger.error(f"Automation '{self.name}' failed: {e}")
            
            return {
                "automation_id": self.id,
                "automation_name": self.name,
                "status": "failed",
                "error": str(e),
                "success": False
            }
        
        finally:
            if self.status == AutomationStatus.RUNNING:
                self.status = AutomationStatus.WAITING
    
    def reset(self) -> None:
        """Reset automation to waiting state."""
        self.status = AutomationStatus.WAITING
        self.last_error = None
        logger.debug(f"Reset automation '{self.name}'")
    
    def cancel(self) -> None:
        """Cancel the automation."""
        self.status = AutomationStatus.CANCELLED
        logger.debug(f"Cancelled automation '{self.name}'")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get automation execution statistics."""
        return {
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.execution_count - self.error_count) / self.execution_count
                if self.execution_count > 0 else 0.0
            ),
            "last_execution": self.last_execution,
            "last_error": self.last_error,
            "current_status": self.status.value
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert automation to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_name": self.trigger_name,
            "task_names": self.task_names,
            "mode": self.mode.value,
            "conditions": self.conditions,
            "retry_policy": self.retry_policy,
            "status": self.status.value,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "last_execution": self.last_execution,
            "last_result": self.last_result,
            "last_error": self.last_error
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        trigger_registry: Optional[Dict[str, Trigger]] = None
    ) -> "Automation":
        """Create automation from dictionary representation."""
        # Resolve trigger
        trigger_name = data.get("trigger", data.get("trigger_name", ""))
        if not trigger_name:
            raise ValueError("Automation must have a trigger specified")
            
        trigger = None
        if trigger_registry and trigger_name in trigger_registry:
            trigger = trigger_registry[trigger_name]
        else:
            trigger = trigger_name  # Use name as placeholder
        
        # Get task sequence
        task_sequence = data.get("task", data.get("sequence", []))
        if isinstance(task_sequence, str):
            task_sequence = [task_sequence]
        
        automation = cls(
            trigger=trigger,
            sequence=task_sequence,
            name=data.get("name"),
            description=data.get("description", ""),
            mode=AutomationMode(data.get("mode", "immediate")),
            conditions=data.get("conditions", {}),
            retry_policy=data.get("retry_policy", {}),
            automation_id=data.get("id")
        )
        
        # Restore execution state
        if "status" in data:
            automation.status = AutomationStatus(data["status"])
        automation.execution_count = data.get("execution_count", 0)
        automation.error_count = data.get("error_count", 0)
        automation.last_execution = data.get("last_execution")
        automation.last_result = data.get("last_result")
        automation.last_error = data.get("last_error")
        
        return automation
    
    def __repr__(self) -> str:
        return f"Automation(name='{self.name}', trigger='{self.trigger_name}', tasks={len(self.task_names)})"


# Built-in automation factories
def create_email_auto_response(
    response_template: str,
    agent_name: str = "EmailResponder"
) -> Automation:
    """Create an automation that auto-responds to emails."""
    from .trigger import create_email_trigger
    from .task import Task, TaskStep
    
    trigger = create_email_trigger("EmailAutoResponse")
    
    task = Task(
        name="EmailAutoResponseTask",
        description="Automatically respond to emails",
        steps=[
            TaskStep(
                agent=agent_name,
                tool="EmailSender",
                input_data=response_template
            )
        ]
    )
    
    return Automation(
        trigger=trigger,
        sequence=task,
        name="EmailAutoResponse",
        description="Automatically respond to incoming emails"
    )


def create_data_processing_automation(
    schedule: str,
    processing_agent: str = "DataProcessor"
) -> Automation:
    """Create an automation for scheduled data processing."""
    from .trigger import create_time_trigger
    from .task import Task, TaskStep
    
    trigger = create_time_trigger("DataProcessingSchedule", schedule)
    
    task = Task(
        name="DataProcessingTask",
        description="Process data on schedule",
        steps=[
            TaskStep(
                agent=processing_agent,
                tool="DataFetcher",
                input_data="fetch latest data"
            ),
            TaskStep(
                agent=processing_agent,
                tool="DataProcessor",
                input_data="process and analyze data"
            ),
            TaskStep(
                agent=processing_agent,
                tool="ReportGenerator",
                input_data="generate summary report"
            )
        ]
    )
    
    return Automation(
        trigger=trigger,
        sequence=task,
        name="ScheduledDataProcessing",
        description=f"Process data on schedule: {schedule}"
    )
