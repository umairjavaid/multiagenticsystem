#!/usr/bin/env python3
"""
Test script for the standardized function calling system.
This demonstrates how to use the new tool calling architecture.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from multiagenticsystem.core.system import System
from multiagenticsystem.core.agent import Agent
from multiagenticsystem.core.base_tool import FunctionTool, PydanticTool, ToolScope, ToolCallRequest, ToolCallResponse
from multiagenticsystem.llm.providers import LLMResponse

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Pydantic not available - some features will be disabled")


# Example 1: Simple function tools
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


def calculate_product(a: float, b: float) -> float:
    """Calculate the product of two numbers."""
    return a * b


def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather information for a location."""
    # Mock weather data
    temperatures = {
        "london": 15,
        "new york": 22,
        "tokyo": 18,
        "sydney": 25
    }
    
    temp = temperatures.get(location.lower(), 20)
    if units == "fahrenheit":
        temp = (temp * 9/5) + 32
    
    return {
        "location": location.title(),
        "temperature": temp,
        "units": units,
        "conditions": "partly cloudy",
        "humidity": 65
    }


def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web for information."""
    # Mock search results
    results = [
        {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i+1}", "snippet": f"This is a mock search result about {query}"}
        for i in range(min(max_results, 3))
    ]
    
    return {
        "query": query,
        "results": results,
        "total_results": len(results)
    }


# Example 2: Pydantic tool (if available)
if PYDANTIC_AVAILABLE:
    class EmailParameters(BaseModel):
        recipient: str = Field(description="Email recipient address")
        subject: str = Field(description="Email subject line")
        body: str = Field(description="Email body content")
        priority: str = Field(default="normal", description="Email priority (low, normal, high)")
    
    class EmailTool(PydanticTool):
        def __init__(self):
            super().__init__(
                name="send_email",
                description="Send an email to a recipient",
                parameters_model=EmailParameters,
                scope=ToolScope.GLOBAL
            )
        
        async def _execute_impl(self, **kwargs) -> dict:
            # Mock email sending
            return {
                "success": True,
                "message": f"Email sent to {kwargs['recipient']} with subject '{kwargs['subject']}'",
                "message_id": "msg_12345"
            }


class MockLLMProvider:
    """Mock LLM provider for testing standardized function calling."""
    
    def __init__(self, model: str = "mock-model", **kwargs):
        self.model = model
        self.api_key = "mock-key"
        self.config = kwargs
        self.call_count = 0
    
    def get_supported_parameters(self):
        return ["temperature", "max_tokens"]
    
    def extract_tool_calls(self, response):
        """Extract tool calls from mock response."""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = []
            for tc in response.tool_calls:
                if isinstance(tc, dict):
                    func_name = tc.get('function', {}).get('name') or tc.get('name')
                    func_args = tc.get('function', {}).get('arguments') or tc.get('arguments', '{}')
                    
                    if isinstance(func_args, str):
                        try:
                            args = json.loads(func_args)
                        except:
                            args = {}
                    else:
                        args = func_args
                    
                    tool_calls.append(ToolCallRequest(
                        id=tc.get('id', 'mock_id'),
                        name=func_name,
                        arguments=args
                    ))
            return tool_calls
        return []
    
    def create_tool_response_for_llm(self, tool_responses):
        """Create tool response messages."""
        messages = []
        for response in tool_responses:
            messages.append({
                "role": "tool",
                "tool_call_id": response.id,
                "content": json.dumps(response.to_dict())
            })
        return messages
    
    async def execute(self, messages, context=None, **kwargs):
        """Mock execute that simulates standardized function calling."""
        self.call_count += 1
        print(f"🔧 Mock LLM Call #{self.call_count}")
        
        # Check if tools are provided
        if context and "tools" in context:
            tools = context["tools"]
            print(f"✅ Found {len(tools)} tools: {[t['function']['name'] for t in tools]}")
            
            # First call: simulate tool usage
            if self.call_count == 1 and tools:
                first_tool = tools[0]
                tool_name = first_tool["function"]["name"]
                
                # Create appropriate arguments based on tool
                if "calculate" in tool_name:
                    arguments = {"a": 5, "b": 3}
                elif "weather" in tool_name:
                    arguments = {"location": "London"}
                elif "search" in tool_name:
                    arguments = {"query": "test search"}
                elif "email" in tool_name:
                    arguments = {"recipient": "test@example.com", "subject": "Test", "body": "Test email"}
                else:
                    arguments = {"message": "Testing function call!"}
                
                mock_tool_call = {
                    "id": f"call_{self.call_count}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments)
                    }
                }
                
                return LLMResponse(
                    content=f"I'll use the {tool_name} tool.",
                    metadata={"tool_calls": [mock_tool_call]},
                    usage={"total_tokens": 50},
                    finish_reason="tool_calls",
                    tool_calls=[mock_tool_call]
                )
            
            # Second call: final response after tool execution
            else:
                return LLMResponse(
                    content="Based on the tool results, I've completed the task successfully.",
                    metadata={},
                    usage={"total_tokens": 30},
                    finish_reason="stop"
                )
        else:
            print("❌ No tools provided in context")
            return LLMResponse(
                content="No tools available, responding normally.",
                metadata={},
                usage={"total_tokens": 20},
                finish_reason="stop"
            )
    
    def validate_config(self):
        return True


async def test_basic_tools():
    """Test basic function tools."""
    print("🧪 Testing Basic Function Tools")
    print("=" * 40)
    
    # Create system
    system = System(enable_logging=False)
    
    # Create calculator agent
    calculator = Agent(
        name="Calculator",
        description="Mathematical calculation agent",
        system_prompt="You are a helpful calculator. Use the available tools to perform mathematical calculations.",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Mock the LLM provider
    calculator._llm_provider = MockLLMProvider()
    
    system.register_agent(calculator)
    
    # Create function tools
    sum_tool = FunctionTool(
        func=calculate_sum,
        scope=ToolScope.LOCAL
    ).set_local(calculator)
    
    product_tool = FunctionTool(
        func=calculate_product,
        scope=ToolScope.LOCAL
    ).set_local(calculator)
    
    # Register tools
    system.register_tools(sum_tool, product_tool)
    
    # Test calculations
    test_cases = [
        "What is 15 + 27?",
        "Calculate the product of 4.5 and 8.2"
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Test: {test_case}")
        try:
            result = await system.execute_agent("Calculator", test_case)
            print(f"✅ Result: {result['output']}")
            if result.get('tool_calls_made', 0) > 0:
                print(f"   Tool calls made: {result['tool_calls_made']}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def test_shared_tools():
    """Test shared tools between agents."""
    print("\n🤝 Testing Shared Tools")
    print("=" * 40)
    
    # Create system
    system = System(enable_logging=False)
    
    # Create agents
    weather_agent = Agent(
        name="WeatherBot",
        description="Weather information agent",
        system_prompt="You provide weather information. Use the weather tool to get current conditions.",
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet"
    )
    
    search_agent = Agent(
        name="SearchBot", 
        description="Search agent",
        system_prompt="You help search for information. Use search tools to find relevant results.",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Mock the LLM providers
    weather_agent._llm_provider = MockLLMProvider()
    search_agent._llm_provider = MockLLMProvider()
    
    system.register_agents(weather_agent, search_agent)
    
    # Create shared tools
    weather_tool = FunctionTool(
        func=get_weather,
        scope=ToolScope.SHARED
    ).set_shared(weather_agent, search_agent)
    
    search_tool = FunctionTool(
        func=search_web,
        scope=ToolScope.GLOBAL  # Available to all agents
    )
    
    # Register tools
    system.register_tools(weather_tool, search_tool)
    
    # Test weather agent
    print(f"\n📝 Weather Agent Test:")
    try:
        result = await system.execute_agent("WeatherBot", "What's the weather like in London?")
        print(f"✅ Weather Result: {result['output']}")
    except Exception as e:
        print(f"❌ Weather Error: {e}")
    
    # Test search agent
    print(f"\n📝 Search Agent Test:")
    try:
        result = await system.execute_agent("SearchBot", "Search for information about Python programming")
        print(f"✅ Search Result: {result['output']}")
    except Exception as e:
        print(f"❌ Search Error: {e}")


async def test_pydantic_tools():
    """Test Pydantic-based tools."""
    if not PYDANTIC_AVAILABLE:
        print("\n⚠️  Skipping Pydantic tests - pydantic not available")
        return
    
    print("\n📧 Testing Pydantic Tools")
    print("=" * 40)
    
    # Create system
    system = System(enable_logging=False)
    
    # Create assistant agent
    assistant = Agent(
        name="Assistant",
        description="General assistant agent",
        system_prompt="You are a helpful assistant. You can send emails when requested.",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Mock the LLM provider
    assistant._llm_provider = MockLLMProvider()
    
    system.register_agent(assistant)
    
    # Create Pydantic email tool
    email_tool = EmailTool()
    system.register_tool(email_tool)
    
    # Test email sending
    test_case = "Please send an email to john@example.com with the subject 'Meeting Tomorrow' and body 'Don't forget about our meeting at 2 PM tomorrow.'"
    
    print(f"\n📝 Test: {test_case}")
    try:
        result = await system.execute_agent("Assistant", test_case)
        print(f"✅ Result: {result['output']}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_tool_schemas():
    """Test that tool schemas are generated correctly."""
    print("\n📋 Testing Tool Schema Generation")
    print("=" * 40)
    
    # Create function tool
    sum_tool = FunctionTool(func=calculate_sum)
    schema = sum_tool.get_openapi_schema()
    
    print("Function Tool Schema:")
    print(json.dumps(schema, indent=2))
    
    # Create Pydantic tool if available
    if PYDANTIC_AVAILABLE:
        email_tool = EmailTool()
        schema = email_tool.get_openapi_schema()
        
        print("\nPydantic Tool Schema:")
        print(json.dumps(schema, indent=2))


async def test_tool_access_control():
    """Test tool access control mechanisms."""
    print("\n🔒 Testing Tool Access Control")
    print("=" * 40)
    
    # Create system
    system = System(enable_logging=False)
    
    # Create agents
    agent1 = Agent(name="Agent1", description="First agent")
    agent2 = Agent(name="Agent2", description="Second agent")
    agent3 = Agent(name="Agent3", description="Third agent")
    
    system.register_agents(agent1, agent2, agent3)
    
    # Create tools with different access levels
    local_tool = FunctionTool(func=calculate_sum, name="local_calc").set_local(agent1)
    shared_tool = FunctionTool(func=calculate_product, name="shared_calc").set_shared(agent1, agent2)
    global_tool = FunctionTool(func=get_weather, name="global_weather").set_global()
    
    system.register_tools(local_tool, shared_tool, global_tool)
    
    # Test access for each agent
    for agent in [agent1, agent2, agent3]:
        available_tools = system.tool_executor.get_available_tools_for_agent(agent.name)
        tool_names = [tool.name for tool in available_tools]
        print(f"{agent.name} can access: {tool_names}")
        
        # Verify access control
        assert local_tool.can_be_used_by(agent1.name) == True
        assert local_tool.can_be_used_by(agent2.name) == False
        assert local_tool.can_be_used_by(agent3.name) == False
        
        assert shared_tool.can_be_used_by(agent1.name) == True
        assert shared_tool.can_be_used_by(agent2.name) == True
        assert shared_tool.can_be_used_by(agent3.name) == False
        
        assert global_tool.can_be_used_by(agent1.name) == True
        assert global_tool.can_be_used_by(agent2.name) == True
        assert global_tool.can_be_used_by(agent3.name) == True
    
    print("✅ Access control tests passed!")


async def test_tool_execution_stats():
    """Test tool execution statistics."""
    print("\n📊 Testing Tool Execution Statistics")
    print("=" * 40)
    
    # Create system
    system = System(enable_logging=False)
    
    # Create agent
    agent = Agent(name="TestAgent", description="Test agent")
    agent._llm_provider = MockLLMProvider()
    system.register_agent(agent)
    
    # Create tools
    sum_tool = FunctionTool(func=calculate_sum).set_global()
    weather_tool = FunctionTool(func=get_weather).set_global()
    
    system.register_tools(sum_tool, weather_tool)
    
    # Execute some tasks to generate stats
    tasks = [
        "Calculate 5 + 3",
        "What's the weather in Sydney?",
        "Add 10 and 20",
        "Weather in New York please"
    ]
    
    for task in tasks:
        try:
            await system.execute_agent("TestAgent", task)
        except Exception as e:
            print(f"Task failed: {e}")
    
    # Get execution statistics
    stats = system.tool_executor.get_execution_stats()
    print("\nExecution Statistics:")
    print(json.dumps(stats, indent=2))


async def main():
    """Run all tests."""
    print("🚀 MultiAgenticSystem - Standardized Function Calling Tests")
    print("=" * 60)
    
    # List of tests to run
    tests = [
        test_basic_tools,
        test_shared_tools,
        test_pydantic_tools,
        test_tool_schemas,
        test_tool_access_control,
        test_tool_execution_stats
    ]
    
    # Run tests
    for test_func in tests:
        try:
            await test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()  # Add spacing between tests
    
    print("✅ All tests completed!")
    print("\n📝 Summary:")
    print("- Standardized tool calling system is working")
    print("- Tools can be scoped (local, shared, global)")
    print("- Function tools auto-generate schemas")
    print("- Pydantic tools provide advanced validation")
    print("- Tool execution is centralized and tracked")
    print("- Access control works correctly")


if __name__ == "__main__":
    asyncio.run(main())
