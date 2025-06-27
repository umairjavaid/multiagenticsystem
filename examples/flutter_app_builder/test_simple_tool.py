#!/usr/bin/env python3
"""
Simple test for tool execution
"""
import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
root_dir = Path(__file__).parent.parent.parent
env_file = root_dir / ".env"
load_dotenv(env_file)

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multiagenticsystem.core.system import System
from multiagenticsystem.core.agent import Agent
from multiagenticsystem.core.tool import Tool

def create_file_simple(file_path: str, content: str) -> str:
    """Simple file creation function."""
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully created file: {file_path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

async def test_tool_execution():
    """Test tool execution with the multi-agent system."""
    
    print("🧪 Testing Tool Execution")
    print("=" * 30)
    
    # Create system
    system = System()
    
    # Create agent
    agent = Agent(
        name="TestAgent",
        description="Test agent for file creation",
        system_prompt="You are a helpful assistant that creates files. Use the FileCreator tool to create files.",
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022",
        max_iterations=5
    )
    
    # Create tool
    file_tool = Tool(
        name="FileCreator",
        func=create_file_simple,
        description="Create files with specified content",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to create"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }
    )
    
    # Register agent and tool
    system.register_agent(agent)
    file_tool.set_local(agent.name)
    system.register_tool(file_tool)
    
    print(f"✅ System initialized")
    print(f"📁 Available tools: {list(system.tools.keys())}")
    print(f"🤖 Agent tools: {agent.get_available_tools(system.tools)}")
    
    # Test tool execution
    test_file_path = "/tmp/test_output.txt"
    test_content = "Hello, World! This is a test file."
    
    task = f"Create a file at '{test_file_path}' with the content: '{test_content}'. Use the FileCreator tool."
    
    print(f"\n🎯 Executing task: {task}")
    
    try:
        result = await system.execute_agent(
            agent_name=agent.name,
            input_text=task,
            context={}
        )
        
        print(f"📝 Result: {result}")
        
        # Check if file was created
        if Path(test_file_path).exists():
            content = Path(test_file_path).read_text()
            print(f"✅ File created successfully!")
            print(f"📄 Content: {content}")
        else:
            print(f"❌ File not created")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_execution())
