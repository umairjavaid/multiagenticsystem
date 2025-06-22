"""
Web interface for MultiAgenticSystem using FastAPI.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..core.system import System
from ..core.agent import Agent
from ..core.tool import Tool
from ..core.task import Task
from ..core.trigger import Trigger
from ..core.automation import Automation


# Request/Response Models
class AgentRequest(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_config: Dict[str, Any] = {}


class TaskExecutionRequest(BaseModel):
    task_name: str
    context: Optional[Dict[str, Any]] = None


class AgentExecutionRequest(BaseModel):
    agent_name: str
    input_text: str
    context: Optional[Dict[str, Any]] = None


class EventRequest(BaseModel):
    event: Dict[str, Any]


def create_app(system: System) -> FastAPI:
    """Create FastAPI application with system integration."""
    
    app = FastAPI(
        title="MultiAgenticSystem",
        description="Web interface for multi-agent system management",
        version="0.1.0"
    )
    
    # Dashboard HTML
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MultiAgenticSystem Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .stat { background: #3498db; color: white; padding: 15px; border-radius: 5px; text-align: center; }
            .stat h3 { margin: 0; font-size: 24px; }
            .stat p { margin: 5px 0 0 0; }
            .section { margin: 20px 0; }
            .agents, .tools, .tasks { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
            .item { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
            .item h4 { margin: 0 0 10px 0; color: #2c3e50; }
            .item p { margin: 5px 0; color: #7f8c8d; font-size: 14px; }
            .controls { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 15px 0; }
            .btn { background: #27ae60; color: white; border: none; padding: 10px 15px; border-radius: 3px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #219a52; }
            .btn.danger { background: #e74c3c; }
            .btn.danger:hover { background: #c0392b; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 MultiAgenticSystem Dashboard</h1>
                <p>Manage your multi-agent system with hierarchical tool sharing and event-driven automation</p>
            </div>
            
            <div class="stats" id="stats">
                <!-- Stats will be loaded here -->
            </div>
            
            <div class="controls">
                <button class="btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <button class="btn" onclick="executeTask()">▶️ Execute Task</button>
                <button class="btn" onclick="sendEvent()">📡 Send Event</button>
                <button class="btn danger" onclick="clearSystem()">🗑️ Clear System</button>
            </div>
            
            <div class="section">
                <div class="card">
                    <h2>🤖 Agents</h2>
                    <div class="agents" id="agents">
                        <!-- Agents will be loaded here -->
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="card">
                    <h2>🔧 Tools</h2>
                    <div class="tools" id="tools">
                        <!-- Tools will be loaded here -->
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="card">
                    <h2>📋 Tasks</h2>
                    <div class="tasks" id="tasks">
                        <!-- Tasks will be loaded here -->
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="card">
                    <h2>⚡ Automations</h2>
                    <div class="automations" id="automations">
                        <!-- Automations will be loaded here -->
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function fetchSystemInfo() {
                try {
                    const response = await fetch('/api/system/info');
                    return await response.json();
                } catch (error) {
                    console.error('Error fetching system info:', error);
                    return null;
                }
            }
            
            function updateStats(status) {
                const statsHtml = `
                    <div class="stat">
                        <h3>${status.agents}</h3>
                        <p>Agents</p>
                    </div>
                    <div class="stat">
                        <h3>${status.tools}</h3>
                        <p>Tools</p>
                    </div>
                    <div class="stat">
                        <h3>${status.tasks}</h3>
                        <p>Tasks</p>
                    </div>
                    <div class="stat">
                        <h3>${status.automations}</h3>
                        <p>Automations</p>
                    </div>
                    <div class="stat">
                        <h3>${status.pending_events}</h3>
                        <p>Pending Events</p>
                    </div>
                `;
                document.getElementById('stats').innerHTML = statsHtml;
            }
            
            function updateAgents(agents) {
                const agentsHtml = Object.values(agents).map(agent => `
                    <div class="item">
                        <h4>${agent.name}</h4>
                        <p>${agent.description}</p>
                        <p><strong>LLM:</strong> ${agent.llm_provider}/${agent.llm_model}</p>
                        <p><strong>Tools:</strong> ${agent.local_tools.length + agent.shared_tools.length + agent.global_tools.length}</p>
                    </div>
                `).join('');
                document.getElementById('agents').innerHTML = agentsHtml || '<p>No agents registered</p>';
            }
            
            function updateTools(tools) {
                const toolsHtml = Object.values(tools).map(tool => `
                    <div class="item">
                        <h4>${tool.name}</h4>
                        <p>${tool.description}</p>
                        <p><strong>Scope:</strong> ${tool.scope}</p>
                        <p><strong>Usage:</strong> ${tool.usage_count} times</p>
                    </div>
                `).join('');
                document.getElementById('tools').innerHTML = toolsHtml || '<p>No tools registered</p>';
            }
            
            function updateTasks(tasks) {
                const tasksHtml = Object.values(tasks).map(task => `
                    <div class="item">
                        <h4>${task.name}</h4>
                        <p>${task.description}</p>
                        <p><strong>Steps:</strong> ${task.steps.length}</p>
                        <p><strong>Status:</strong> ${task.status}</p>
                    </div>
                `).join('');
                document.getElementById('tasks').innerHTML = tasksHtml || '<p>No tasks registered</p>';
            }
            
            function updateAutomations(automations) {
                const automationsHtml = Object.values(automations).map(automation => `
                    <div class="item">
                        <h4>${automation.name}</h4>
                        <p>${automation.description}</p>
                        <p><strong>Trigger:</strong> ${automation.trigger_name}</p>
                        <p><strong>Executions:</strong> ${automation.execution_count}</p>
                    </div>
                `).join('');
                document.getElementById('automations').innerHTML = automationsHtml || '<p>No automations registered</p>';
            }
            
            async function refreshDashboard() {
                const info = await fetchSystemInfo();
                if (info) {
                    updateStats(info.status);
                    updateAgents(info.agents);
                    updateTools(info.tools);
                    updateTasks(info.tasks);
                    updateAutomations(info.automations);
                }
            }
            
            async function executeTask() {
                const taskName = prompt('Enter task name to execute:');
                if (taskName) {
                    try {
                        const response = await fetch('/api/tasks/execute', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({task_name: taskName, context: {}})
                        });
                        const result = await response.json();
                        alert(`Task executed: ${result.status}`);
                    } catch (error) {
                        alert(`Error executing task: ${error.message}`);
                    }
                }
            }
            
            async function sendEvent() {
                const eventType = prompt('Enter event type:');
                if (eventType) {
                    try {
                        const response = await fetch('/api/events/emit', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({event: {type: eventType, timestamp: new Date().toISOString()}})
                        });
                        const result = await response.json();
                        alert(`Event sent: ${result.message}`);
                    } catch (error) {
                        alert(`Error sending event: ${error.message}`);
                    }
                }
            }
            
            function clearSystem() {
                if (confirm('Are you sure you want to clear the system? This will remove all components.')) {
                    // Implementation would call clear endpoint
                    alert('Clear system functionality would be implemented here');
                }
            }
            
            // Initial load
            refreshDashboard();
            
            // Auto-refresh every 10 seconds
            setInterval(refreshDashboard, 10000);
        </script>
    </body>
    </html>
    """
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the main dashboard."""
        return dashboard_html
    
    @app.get("/api/system/status")
    async def get_system_status():
        """Get system status."""
        return system.get_system_status()
    
    @app.get("/api/system/info")
    async def get_system_info():
        """Get detailed system information."""
        return system.get_system_info()
    
    @app.get("/api/agents")
    async def list_agents():
        """List all agents."""
        return {name: agent.to_dict() for name, agent in system.agents.items()}
    
    @app.post("/api/agents")
    async def create_agent(agent_request: AgentRequest):
        """Create a new agent."""
        try:
            agent = Agent(
                name=agent_request.name,
                description=agent_request.description,
                system_prompt=agent_request.system_prompt,
                llm_provider=agent_request.llm_provider,
                llm_model=agent_request.llm_model,
                llm_config=agent_request.llm_config
            )
            system.register_agent(agent)
            return {"message": f"Agent '{agent.name}' created successfully", "agent": agent.to_dict()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.delete("/api/agents/{agent_name}")
    async def delete_agent(agent_name: str):
        """Delete an agent."""
        if system.remove_agent(agent_name):
            return {"message": f"Agent '{agent_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    @app.get("/api/tools")
    async def list_tools():
        """List all tools."""
        return {name: tool.to_dict() for name, tool in system.tools.items()}
    
    @app.get("/api/tasks")
    async def list_tasks():
        """List all tasks."""
        return {name: task.to_dict() for name, task in system.tasks.items()}
    
    @app.post("/api/tasks/execute")
    async def execute_task(request: TaskExecutionRequest, background_tasks: BackgroundTasks):
        """Execute a task."""
        try:
            result = await system.execute_task(request.task_name, request.context)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/agents/execute")
    async def execute_agent(request: AgentExecutionRequest):
        """Execute an agent with input."""
        try:
            result = await system.execute_agent(
                request.agent_name,
                request.input_text,
                request.context
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/api/triggers")
    async def list_triggers():
        """List all triggers."""
        return {name: trigger.to_dict() for name, trigger in system.triggers.items()}
    
    @app.get("/api/automations")
    async def list_automations():
        """List all automations."""
        return {name: automation.to_dict() for name, automation in system.automations.items()}
    
    @app.post("/api/events/emit")
    async def emit_event(request: EventRequest, background_tasks: BackgroundTasks):
        """Emit an event to the system."""
        try:
            system.emit_event(request.event)
            background_tasks.add_task(system.process_events)
            return {"message": "Event emitted successfully", "event": request.event}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/config/load")
    async def load_config(config_path: str):
        """Load configuration from file."""
        try:
            system.load_config(config_path)
            return {"message": f"Configuration loaded from {config_path}"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/config/save")
    async def save_config(config_path: str):
        """Save configuration to file."""
        try:
            system.save_config(config_path)
            return {"message": f"Configuration saved to {config_path}"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    return app
