#!/usr/bin/env python3
"""
Flutter App Builder - Configuration-Driven Version

This version loads the multi-agent system from a YAML configuration file,
demonstrating how to use the MultiAgenticSystem with declarative configuration.
"""

import asyncio
import os
from pathlib import Path

from multiagenticsystem.core.system import System
from multiagenticsystem.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Main function using configuration-driven approach."""
    
    print("🚀 Flutter App Builder - Configuration-Driven")
    print("=" * 50)
    
    # Get configuration file path
    config_path = Path(__file__).parent / "flutter_config.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    try:
        # Create system from configuration
        system = System(config_path=str(config_path))
        
        print(f"✅ System loaded from configuration")
        print(f"📊 Loaded: {len(system.agents)} agents, {len(system.tools)} tools, {len(system.tasks)} tasks")
        
        # Interactive mode
        print("\nAvailable commands:")
        print("1. 'build [app_name] [description]' - Start building a Flutter app")
        print("2. 'info' - Show system information")
        print("3. 'test [app_name]' - Run tests for an existing app")
        print("4. 'quit' - Exit the system")
        print()
        
        while True:
            try:
                user_input = input("🎯 Enter command: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'info':
                    print(f"📊 System Information:")
                    print(f"  🤖 Agents: {list(system.agents.keys())}")
                    print(f"  🔧 Tools: {list(system.tools.keys())}")
                    print(f"  📋 Tasks: {list(system.tasks.keys())}")
                    if system.collaborations:
                        print(f"  🤝 Collaborations: {list(system.collaborations.keys())}")
                    
                elif user_input.startswith('build'):
                    parts = user_input.split(' ', 2)
                    if len(parts) < 3:
                        print("❌ Usage: build [app_name] [description]")
                        continue
                        
                    app_name = parts[1]
                    app_description = parts[2]
                    
                    print(f"🏗️ Building Flutter app: {app_name}")
                    print(f"📝 Description: {app_description}")
                    
                    # Create project directory
                    project_dir = f"./apps/{app_name}"
                    os.makedirs(project_dir, exist_ok=True)
                    
                    # Execute tasks sequentially
                    context = {
                        "app_name": app_name,
                        "description": app_description,
                        "project_dir": project_dir
                    }
                    
                    # Execute the project setup task
                    if "ProjectSetup" in system.tasks:
                        print("\n📦 Executing ProjectSetup task...")
                        result = await system.execute_task("ProjectSetup", context)
                        print(f"✅ Setup completed: {result.get('status', 'Done')}")
                    
                    # Execute feature implementation
                    if "FeatureImplementation" in system.tasks:
                        print("\n🔨 Executing FeatureImplementation task...")
                        impl_context = {**context, "features": app_description}
                        result = await system.execute_task("FeatureImplementation", impl_context)
                        print(f"✅ Implementation completed: {result.get('status', 'Done')}")
                    
                    # Execute testing
                    if "TestingAndQA" in system.tasks:
                        print("\n🧪 Executing TestingAndQA task...")
                        result = await system.execute_task("TestingAndQA", context)
                        print(f"✅ Testing completed: {result.get('status', 'Done')}")
                    
                    print(f"\n🎉 Flutter app '{app_name}' build completed!")
                    print(f"📁 Project location: {project_dir}")
                    
                elif user_input.startswith('test'):
                    parts = user_input.split(' ', 1)
                    if len(parts) < 2:
                        print("❌ Usage: test [app_name]")
                        continue
                        
                    app_name = parts[1]
                    project_dir = f"./apps/{app_name}"
                    
                    if not os.path.exists(project_dir):
                        print(f"❌ App '{app_name}' not found in {project_dir}")
                        continue
                    
                    print(f"🧪 Running tests for: {app_name}")
                    
                    context = {
                        "app_name": app_name,
                        "project_dir": project_dir
                    }
                    
                    if "TestingAndQA" in system.tasks:
                        result = await system.execute_task("TestingAndQA", context)
                        print(f"✅ Testing completed: {result.get('status', 'Done')}")
                    else:
                        print("❌ TestingAndQA task not found")
                    
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Error in main loop: {e}")
    
    except Exception as e:
        print(f"❌ Failed to load system: {e}")
        logger.error(f"Failed to load system: {e}")


if __name__ == "__main__":
    asyncio.run(main())
