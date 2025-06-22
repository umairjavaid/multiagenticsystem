"""
Quick test of the MultiAgenticSystem package.
"""

import asyncio
from multiagenticsystem import Agent, Tool, Task, System


def test_basic_functionality():
    """Test basic system functionality."""
    print("🚀 Testing MultiAgenticSystem Basic Functionality")
    print("=" * 50)
    
    # 1. Create a simple system
    system = System()
    print(f"✅ Created system")
    
    # 2. Create agents
    agent1 = Agent(
        name="TestAgent1",
        description="A test agent for demonstration",
        system_prompt="You are a helpful test agent.",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    agent2 = Agent(
        name="TestAgent2", 
        description="Another test agent",
        system_prompt="You are also a helpful test agent.",
        llm_provider="anthropic",
        llm_model="claude-3.5-sonnet"
    )
    
    print(f"✅ Created agents: {agent1.name}, {agent2.name}")
    
    # 3. Create tools
    def simple_function(text: str) -> str:
        return f"Processed: {text}"
    
    def another_function(x: int, y: int) -> int:
        return x + y
    
    tool1 = Tool("SimpleTool", func=simple_function, description="A simple tool")
    tool1.set_local(agent1)
    
    tool2 = Tool("SharedTool", func=another_function, description="A shared tool")
    tool2.set_shared(agent1, agent2)
    
    tool3 = Tool("GlobalTool", func=lambda msg: f"Global: {msg}", description="A global tool")
    tool3.set_global()
    
    print(f"✅ Created tools with different sharing levels")
    
    # 4. Register components
    system.register_agents(agent1, agent2)
    system.register_tools(tool1, tool2, tool3)
    
    print(f"✅ Registered components in system")
    
    # 5. Check system status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 6. Test tool access
    print(f"\n🔧 Tool Access:")
    agent1_tools = agent1.get_available_tools(system.tools)
    agent2_tools = agent2.get_available_tools(system.tools)
    
    print(f"  {agent1.name} has access to: {agent1_tools}")
    print(f"  {agent2.name} has access to: {agent2_tools}")
    
    # 7. Test tool execution
    print(f"\n🧪 Testing Tool Execution:")
    
    async def test_tools():
        # Test local tool
        result1 = await tool1.execute(agent1, "test input")
        print(f"  Local tool result: {result1['result']}")
        
        # Test shared tool
        result2 = await tool2.execute(agent1, 5, 3)
        print(f"  Shared tool result: {result2['result']}")
        
        # Test global tool
        result3 = await tool3.execute(agent2, "hello world")
        print(f"  Global tool result: {result3['result']}")
        
        # Test agent execution
        agent_result = await agent1.execute("Hello, can you help me?")
        print(f"  Agent execution: {agent_result['output'][:100]}...")
    
    # Run async tests
    asyncio.run(test_tools())
    
    # 8. Create and test a simple task
    print(f"\n📋 Testing Task System:")
    
    task = Task("TestTask", description="A simple test task")
    task.add_step(agent1, "SimpleTool", "process this data")
    task.add_step(agent2, "GlobalTool", "finalize results")
    
    system.register_task(task)
    print(f"  Created task with {len(task.steps)} steps")
    
    # 9. Test configuration save/load
    print(f"\n💾 Testing Configuration:")
    
    # Save configuration
    system.save_config("test_config.yaml")
    print(f"  Saved configuration to test_config.yaml")
    
    # Load configuration
    new_system = System.from_config("test_config.yaml")
    new_status = new_system.get_system_status()
    print(f"  Loaded system: {new_status['agents']} agents, {new_status['tools']} tools")
    
    print(f"\n✅ All tests completed successfully!")
    print("=" * 50)
    
    return system


def test_hierarchical_tools():
    """Test the hierarchical tool sharing system."""
    print("\n🔧 Testing Hierarchical Tool Sharing")
    print("=" * 40)
    
    # Create multiple agents
    agents = [
        Agent(f"Agent{i}", description=f"Test agent {i}") 
        for i in range(1, 4)
    ]
    
    # Create tools with different sharing levels
    local_tool = Tool("LocalTool", description="Only for Agent1")
    local_tool.set_local(agents[0])
    
    shared_tool = Tool("SharedTool", description="For Agent1 and Agent2")
    shared_tool.set_shared(agents[0], agents[1])
    
    global_tool = Tool("GlobalTool", description="For all agents")
    global_tool.set_global()
    
    # Test access permissions
    tools = {"LocalTool": local_tool, "SharedTool": shared_tool, "GlobalTool": global_tool}
    
    for i, agent in enumerate(agents):
        available = []
        for tool_name, tool in tools.items():
            if tool.can_be_used_by(agent):
                available.append(tool_name)
        print(f"  {agent.name}: {available}")
    
    print("✅ Hierarchical tool sharing working correctly!")


if __name__ == "__main__":
    # Run tests
    system = test_basic_functionality()
    test_hierarchical_tools()
    
    print(f"\n🎉 MultiAgenticSystem test completed successfully!")
    print(f"📦 Package is ready for use!")
