#!/usr/bin/env python3
"""
Flutter App Builder Example

This example demonstrates a multi-agent system for building Flutter applications.
It features two specialized agents:
- ImplementationAgent: Handles app development, creating files, and implementing features
- TestingAgent: Focuses on testing, quality assurance, and validation

Both agents share access to a terminal tool for running Flutter commands.
"""

import asyncio
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file in the root directory
root_dir = Path(__file__).parent.parent.parent
env_file = root_dir / ".env"
load_dotenv(env_file)

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multiagenticsystem.core.system import System
from multiagenticsystem.core.agent import Agent
from multiagenticsystem.core.tool import Tool
from multiagenticsystem.core.task import Task
from multiagenticsystem.utils.logger import get_logger, setup_logging

# Set up logging for the flutter builder example
setup_logging(
    level="INFO",
    log_dir="./flutter_builder_logs",
    session_id=f"flutter_builder_{int(asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0)}",
    format_type="detailed"
)

logger = get_logger("flutter_builder")


def create_terminal_tool() -> Tool:
    """Create a shared terminal tool for both agents."""
    
    async def run_command(command: str, working_dir: Optional[str] = None) -> str:
        """Execute a shell command and return the output."""
        try:
            # Set working directory if provided
            cwd = working_dir if working_dir else os.getcwd()
            
            # Run the command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await process.communicate()
            
            output = ""
            if stdout:
                output += f"STDOUT:\n{stdout.decode()}\n"
            if stderr:
                output += f"STDERR:\n{stderr.decode()}\n"
            
            output += f"Return code: {process.returncode}"
            
            logger.info(f"Executed command: {command}")
            return output
            
        except Exception as e:
            error_msg = f"Error executing command '{command}': {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    terminal_tool = Tool(
        name="Terminal",
        func=run_command,
        description="Execute shell commands and Flutter CLI operations",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command"
                }
            },
            "required": ["command"]
        }
    )
    
    return terminal_tool


def create_file_tool() -> Tool:
    """Create a file management tool."""
    
    def create_file(file_path: str, content: str) -> str:
        """Create or update a file with the given content."""
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result = f"Successfully created/updated file: {file_path}"
            logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"Error creating file '{file_path}': {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def read_file(file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Successfully read file: {file_path}")
            return content
            
        except Exception as e:
            error_msg = f"Error reading file '{file_path}': {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    file_tool = Tool(
        name="FileManager",
        func=create_file,
        description="Create, update, and read files in the project",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }
    )
    
    return file_tool


def create_implementation_agent() -> Agent:
    """Create the implementation agent for Flutter development."""
    
    system_prompt = """You are a Flutter Implementation Agent, an expert Flutter developer responsible for:

1. **Project Setup**: Creating new Flutter projects in the correct directory structure
2. **Feature Development**: Implementing app features, UI components, business logic
3. **Code Organization**: Structuring code with proper architecture patterns
4. **Dependencies**: Managing pubspec.yaml and package dependencies
5. **Platform Integration**: Implementing platform-specific features

You have access to:
- Terminal: For running Flutter commands (flutter create, flutter pub get, flutter build, etc.)
  - IMPORTANT: When creating projects, always use the working_dir parameter to specify where to run commands
  - For 'flutter create', use the apps_dir from context as the working_dir
- FileManager: For creating and modifying Dart files, YAML configs, etc.

Directory Structure Rules:
- Flutter projects should be created in the apps directory provided in context
- Use Terminal tool with working_dir parameter to ensure commands run in correct location
- Never create projects in the root workspace directory

Best Practices:
- Follow Flutter/Dart conventions and best practices
- Use proper state management (Provider, Bloc, etc.)
- Implement responsive design principles
- Write clean, well-documented code
- Structure projects with clear separation of concerns

When implementing features:
1. Start with project setup if needed
2. Plan the app architecture
3. Implement core functionality first
4. Add UI and user experience elements
5. Ensure proper error handling

Always explain what you're doing and why."""

    # Check which API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Use available provider, prefer OpenAI if both are available
    if openai_key and openai_key != "your-openai-api-key-here":
        llm_provider = "openai"
        llm_model = "gpt-4"
    elif anthropic_key:
        llm_provider = "anthropic"
        llm_model = "claude-3-5-sonnet-20241022"
    else:
        raise ValueError("No valid API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in the .env file.")

    return Agent(
        name="ImplementationAgent",
        description="Expert Flutter developer for feature implementation and app development",
        system_prompt=system_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_iterations=15,
        memory_enabled=True
    )


def create_testing_agent() -> Agent:
    """Create the testing agent for quality assurance."""
    
    system_prompt = """You are a Flutter Testing Agent, an expert in Flutter testing and quality assurance responsible for:

1. **Test Strategy**: Planning comprehensive testing approaches
2. **Unit Testing**: Writing and executing unit tests for business logic
3. **Widget Testing**: Creating widget tests for UI components
4. **Integration Testing**: Implementing end-to-end integration tests
5. **Code Quality**: Reviewing code for best practices and potential issues
6. **Performance**: Analyzing app performance and optimization opportunities

You have access to:
- Terminal: For running test commands (flutter test, flutter drive, dart analyze)
- FileManager: For creating test files and reviewing implementation code

Testing Focus Areas:
- Functionality verification
- Edge case handling
- Performance validation
- Code coverage analysis
- Accessibility compliance
- Platform compatibility

When testing:
1. Review the implementation code first
2. Identify critical paths and edge cases
3. Write comprehensive test suites
4. Execute tests and analyze results
5. Provide detailed feedback and recommendations
6. Suggest improvements for code quality

Always provide clear test reports and actionable feedback."""

    # Check which API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Use available provider, prefer Anthropic for testing agent
    if anthropic_key:
        llm_provider = "anthropic"
        llm_model = "claude-3-5-sonnet-20241022"
    elif openai_key and openai_key != "your-openai-api-key-here":
        llm_provider = "openai"
        llm_model = "gpt-4"
    else:
        raise ValueError("No valid API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in the .env file.")

    return Agent(
        name="TestingAgent", 
        description="Expert Flutter testing specialist for quality assurance and validation",
        system_prompt=system_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_iterations=15,
        memory_enabled=True
    )


def create_flutter_tasks() -> List[Task]:
    """Create tasks for the Flutter app building workflow."""
    
    # Task 1: Project Setup and Planning
    setup_task = Task(
        name="ProjectSetup",
        description="Initialize Flutter project and plan app architecture",
        steps=[
            {
                "agent": "ImplementationAgent",
                "input": "Create new Flutter project with proper structure",
                "context": {"expected_output": "Initialized Flutter project with basic architecture"}
            }
        ]
    )
    
    # Task 2: Feature Implementation
    implementation_task = Task(
        name="FeatureImplementation", 
        description="Implement core app features and functionality",
        steps=[
            {
                "agent": "ImplementationAgent",
                "input": "Implement requested app features with proper Flutter patterns",
                "context": {"expected_output": "Working Flutter app with implemented features"}
            }
        ]
    )
    
    # Task 3: Testing and Quality Assurance
    testing_task = Task(
        name="TestingAndQA",
        description="Create comprehensive tests and validate app quality",
        steps=[
            {
                "agent": "TestingAgent",
                "input": "Review implementation and create comprehensive test suite",
                "context": {"expected_output": "Complete test suite with quality assessment"}
            },
            {
                "agent": "TestingAgent",
                "input": "Execute tests and provide quality report",
                "context": {"expected_output": "Test results and quality recommendations"}
            }
        ]
    )
    
    # Task 4: Final Integration and Deployment Prep
    integration_task = Task(
        name="IntegrationAndDeployment",
        description="Final integration testing and deployment preparation",
        steps=[
            {
                "agent": "TestingAgent",
                "input": "Run integration tests and performance validation",
                "context": {"expected_output": "Integration test results and performance metrics"}
            },
            {
                "agent": "ImplementationAgent",
                "input": "Prepare app for deployment (build optimization, documentation)",
                "context": {"expected_output": "Deployment-ready Flutter app with documentation"}
            }
        ]
    )
    
    return [setup_task, implementation_task, testing_task, integration_task]


async def main():
    """Main function to demonstrate the Flutter app builder system."""
    
    logger.info("Starting Flutter App Builder - Multi-Agent System", extra={
        "component": "main",
        "operation": "startup"
    })
    
    print("🚀 Flutter App Builder - Multi-Agent System")
    print("=" * 50)
    
    # Check if API keys are loaded
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    logger.debug("Checking API key configuration", extra={
        "component": "main",
        "operation": "auth_check",
        "metadata": {
            "openai_key_present": bool(openai_key and openai_key != 'your-openai-api-key-here'),
            "anthropic_key_present": bool(anthropic_key)
        }
    })
    
    print(f"🔑 OPENAI_API_KEY: {'✅ Set' if openai_key and openai_key != 'your-openai-api-key-here' else '❌ Missing/Invalid'}")
    print(f"🔑 ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Missing'}")
    
    if not ((openai_key and openai_key != "your-openai-api-key-here") or anthropic_key):
        logger.error("No valid API keys found", extra={
            "component": "main",
            "operation": "auth_check",
            "metadata": {"error": "missing_api_keys"}
        })
        print("\n❌ No valid API keys found! Please check your .env file.")
        print("   Make sure to set either OPENAI_API_KEY or ANTHROPIC_API_KEY with actual values.")
        print("   Current OPENAI_API_KEY appears to be a placeholder.")
        return
    
    logger.info("Creating multi-agent system", extra={
        "component": "main",
        "operation": "system_creation"
    })
    
    # Create the system
    system = System()
    
    # Create and register agents
    try:
        logger.info("Creating implementation agent", extra={
            "component": "main",
            "operation": "agent_creation",
            "agent_type": "implementation"
        })
        impl_agent = create_implementation_agent()
        
        logger.info("Creating testing agent", extra={
            "component": "main", 
            "operation": "agent_creation",
            "agent_type": "testing"
        })
        test_agent = create_testing_agent()
        
        system.register_agents(impl_agent, test_agent)
        print(f"✅ Successfully created agents:")
        print(f"  - ImplementationAgent using {impl_agent.llm_provider_name}/{impl_agent.llm_model}")
        print(f"  - TestingAgent using {test_agent.llm_provider_name}/{test_agent.llm_model}")
        
    except Exception as e:
        print(f"❌ Failed to create agents: {e}")
        return
    
    # Create and register shared tools
    terminal_tool = create_terminal_tool()
    file_tool = create_file_tool()
    
    # Set terminal as shared between both agents
    terminal_tool.set_shared(impl_agent, test_agent)
    file_tool.set_shared(impl_agent, test_agent)
    
    system.register_tools(terminal_tool, file_tool)
    
    # Create and register tasks
    tasks = create_flutter_tasks()
    for task in tasks:
        system.register_task(task)
    
    # Interactive mode
    print("\nAvailable commands:")
    print("1. 'build [app_name] [description]' - Start building a Flutter app")
    print("2. 'status' - Show system status")
    print("3. 'agents' - List all agents")
    print("4. 'tools' - List all tools")
    print("5. 'tasks' - List all tasks")
    print("6. 'quit' - Exit the system")
    print()
    
    while True:
        try:
            user_input = input("🎯 Enter command: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'status':
                print(f"📊 System Status:")
                print(f"  Agents: {len(system.agents)}")
                print(f"  Tools: {len(system.tools)}")
                print(f"  Tasks: {len(system.tasks)}")
                
            elif user_input.lower() == 'agents':
                print("🤖 Registered Agents:")
                for name, agent in system.agents.items():
                    print(f"  - {name}: {agent.description}")
                    
            elif user_input.lower() == 'tools':
                print("🔧 Registered Tools:")
                for name, tool in system.tools.items():
                    print(f"  - {name}: {tool.description} (scope: {tool.scope.value})")
                    
            elif user_input.lower() == 'tasks':
                print("📋 Registered Tasks:")
                for name, task in system.tasks.items():
                    print(f"  - {name}: {task.description}")
                    
            elif user_input.startswith('build'):
                parts = user_input.split(' ', 2)
                if len(parts) < 3:
                    print("❌ Usage: build [app_name] [description]")
                    continue
                    
                app_name = parts[1]
                app_description = parts[2]
                
                print(f"🏗️ Starting Flutter app build: {app_name}")
                print(f"📝 Description: {app_description}")
                
                # Create project directory in the apps folder relative to this script
                script_dir = Path(__file__).parent
                apps_dir = script_dir / "apps"
                project_dir = apps_dir / app_name
                os.makedirs(project_dir, exist_ok=True)
                
                # Execute the workflow
                print("\n🚀 Starting development workflow...")
                
                # Step 1: Project Setup
                print("\n📦 Step 1: Project Setup")
                setup_context = {
                    "app_name": app_name,
                    "description": app_description,
                    "project_dir": str(project_dir),
                    "apps_dir": str(apps_dir)
                }
                
                setup_result = await impl_agent.execute(
                    f"Create a new Flutter project named '{app_name}' with description '{app_description}'. "
                    f"Use the Terminal tool with working_dir='{apps_dir}' to run 'flutter create {app_name}'. "
                    f"This will create the project in the correct apps directory. "
                    f"Then set up basic project structure and update pubspec.yaml with the app description.",
                    context=setup_context,
                    available_tools=["Terminal", "FileManager"]
                )
                
                print(f"✅ Setup Result: {setup_result.get('output', 'Completed')[:200]}...")
                
                # Step 2: Feature Implementation
                print("\n🔨 Step 2: Feature Implementation")
                impl_result = await impl_agent.execute(
                    f"Implement a basic Flutter app for '{app_description}'. "
                    f"Create a simple but functional UI with proper Flutter widgets and state management. "
                    f"Focus on core functionality and good code organization.",
                    context=setup_context,
                    available_tools=["Terminal", "FileManager"]
                )
                
                print(f"✅ Implementation Result: {impl_result.get('output', 'Completed')[:200]}...")
                
                # Step 3: Testing
                print("\n🧪 Step 3: Testing and Quality Assurance")
                test_result = await test_agent.execute(
                    f"Review the Flutter app implementation for '{app_name}' and create comprehensive tests. "
                    f"Write unit tests, widget tests, and run flutter analyze to check code quality. "
                    f"Provide a detailed quality assessment and recommendations.",
                    context=setup_context,
                    available_tools=["Terminal", "FileManager"]
                )
                
                print(f"✅ Testing Result: {test_result.get('output', 'Completed')[:200]}...")
                
                print(f"\n🎉 Flutter app '{app_name}' build workflow completed!")
                print(f"📁 Project location: {project_dir}")
                
            else:
                print("❌ Unknown command. Type 'quit' to exit or try one of the available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
