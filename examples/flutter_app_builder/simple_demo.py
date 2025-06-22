#!/usr/bin/env python3
"""
Simple Flutter App Builder Demo

This is a simplified version that demonstrates the core concepts
without the complexity of the full workflow system.
"""

import asyncio
import subprocess
from multiagenticsystem.core.system import System
from multiagenticsystem.core.agent import Agent
from multiagenticsystem.core.tool import Tool


def create_simple_terminal_tool() -> Tool:
    """Create a simple terminal tool."""
    
    def run_flutter_command(command: str) -> str:
        """Run a Flutter command and return output."""
        try:
            # Simulate Flutter command execution
            if command.startswith("flutter create"):
                return f"✅ Created Flutter project: {command.split()[-1]}"
            elif command == "flutter pub get":
                return "✅ Dependencies installed successfully"
            elif command == "flutter test":
                return "✅ All tests passed (2 tests, 0 failures)"
            elif command == "flutter build apk":
                return "✅ APK built successfully at build/app/outputs/flutter-apk/app-release.apk"
            else:
                return f"✅ Executed: {command}"
        except Exception as e:
            return f"❌ Error: {e}"
    
    terminal_tool = Tool(
        name="Terminal",
        func=run_flutter_command,
        description="Execute Flutter CLI commands"
    )
    
    return terminal_tool


async def simple_demo():
    """Simple demonstration of the Flutter app builder."""
    
    print("🚀 Simple Flutter App Builder Demo")
    print("=" * 40)
    
    # Create the system
    system = System()
    
    # Create agents
    developer = Agent(
        name="Developer",
        description="Flutter developer",
        system_prompt="You are a Flutter developer. Create and build Flutter apps.",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    tester = Agent(
        name="Tester", 
        description="Flutter tester",
        system_prompt="You are a Flutter tester. Test Flutter apps for quality.",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Create shared terminal tool
    terminal = create_simple_terminal_tool()
    terminal.set_shared(developer, tester)  # Both agents can use it
    
    # Register everything
    system.register_agents(developer, tester)
    system.register_tools(terminal)
    
    print(f"✅ System ready with {len(system.agents)} agents and {len(system.tools)} tools")
    print(f"🤖 Agents: {list(system.agents.keys())}")
    print(f"🔧 Tools: {list(system.tools.keys())}")
    
    # Demo workflow
    app_name = "demo_app"
    
    print(f"\n🏗️ Building Flutter app: {app_name}")
    
    # Step 1: Developer creates project
    print("\n📦 Step 1: Project Creation")
    dev_result = await developer.execute(
        f"Create a new Flutter project named '{app_name}' using the Terminal tool. "
        f"Run 'flutter create {app_name}' and then 'flutter pub get'.",
        available_tools=["Terminal"]
    )
    print(f"Developer: {dev_result.get('output', 'Created project')[:100]}...")
    
    # Step 2: Tester validates project
    print("\n🧪 Step 2: Testing")
    test_result = await tester.execute(
        f"Test the Flutter project '{app_name}' using the Terminal tool. "
        f"Run 'flutter test' to validate the project.",
        available_tools=["Terminal"]
    )
    print(f"Tester: {test_result.get('output', 'Tests completed')[:100]}...")
    
    # Step 3: Developer builds app
    print("\n🔨 Step 3: Building")
    build_result = await developer.execute(
        f"Build the Flutter app '{app_name}' for release using the Terminal tool. "
        f"Run 'flutter build apk'.",
        available_tools=["Terminal"]
    )
    print(f"Developer: {build_result.get('output', 'Build completed')[:100]}...")
    
    print(f"\n🎉 Flutter app '{app_name}' completed successfully!")
    print("📱 Ready for deployment!")


if __name__ == "__main__":
    asyncio.run(simple_demo())
