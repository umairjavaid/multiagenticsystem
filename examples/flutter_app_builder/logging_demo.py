#!/usr/bin/env python3
"""
Example demonstrating how to use multiagenticsystem logging 
from the flutter_app_builder example.

This shows how to:
1. Set up logging for your multiagenticsystem usage
2. View logs during and after system execution
3. Filter and analyze logs for debugging
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to Python path so we can import multiagenticsystem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import multiagenticsystem with logging capabilities
import multiagenticsystem as mas


def main():
    """Main function demonstrating logging usage."""
    
    print("=== MultiAgenticSystem Logging Demo ===\n")
    
    # 1. Set up logging with custom configuration
    print("1. Setting up logging...")
    mas.setup_logging(
        level="DEBUG",  # Log everything for demo
        log_dir="./demo_logs",  # Custom log directory
        session_id="flutter_demo_session",
        format_type="detailed"  # More verbose format
    )
    print("✓ Logging configured\n")
    
    # 2. Create a simple system to generate some logs
    print("2. Creating and running a simple system...")
    
    # Create a logger for our demo
    logger = mas.get_logger("flutter_demo")
    
    # Log some demo activities
    logger.info("Starting flutter app builder demo")
    logger.debug("This is a debug message with details", extra={
        "component": "demo",
        "operation": "setup",
        "metadata": {"step": 1, "config": "flutter_basic"}
    })
    
    # Simulate some work
    time.sleep(0.5)
    
    logger.info("Processing app configuration", extra={
        "component": "config_processor", 
        "operation": "parse",
        "metadata": {"config_file": "flutter_config.yaml"}
    })
    
    # Simulate an LLM call (this would normally be done by the system)
    logger.info("Making LLM request", extra={
        "component": "llm_provider",
        "operation": "request",
        "llm_provider": "openai",
        "metadata": {
            "model": "gpt-4",
            "prompt": "Generate Flutter app structure for music app",
            "tokens": 150
        }
    })
    
    time.sleep(0.3)
    
    logger.info("Received LLM response", extra={
        "component": "llm_provider",
        "operation": "response", 
        "llm_provider": "openai",
        "metadata": {
            "model": "gpt-4",
            "response_tokens": 500,
            "success": True
        }
    })
    
    logger.warning("Deprecated widget detected in generated code", extra={
        "component": "code_analyzer",
        "operation": "validation",
        "metadata": {"widget": "FlatButton", "replacement": "TextButton"}
    })
    
    logger.info("Flutter app generation completed", extra={
        "component": "app_builder",
        "operation": "complete",
        "metadata": {
            "output_dir": "./generated_app",
            "files_created": 15,
            "duration_seconds": 2.3
        }
    })
    
    print("✓ Demo system execution completed\n")
    
    # 3. Demonstrate log viewing capabilities
    print("3. Viewing logs...")
    
    # View recent logs
    print("\n--- Recent logs (last 5 lines) ---")
    mas.view_logs(lines=5)
    
    # 4. Create a LogViewer for more advanced operations
    print("\n4. Using LogViewer for advanced log analysis...")
    
    log_viewer = mas.LogViewer(log_dir="./demo_logs")
    
    # Show log summary
    print("\n--- Log Summary ---")
    log_viewer.print_summary()
    
    # Filter logs by component
    print("\n--- LLM Provider logs ---")
    llm_logs = log_viewer.filter_logs(component="llm_provider")
    for log in llm_logs[-3:]:  # Show last 3 LLM logs
        print(log.rstrip())
    
    # Search for specific content
    print("\n--- Searching for 'Flutter' ---")
    log_viewer.search_logs("Flutter", context_lines=1)
    
    # Show available sessions and agents
    print(f"\n--- Available sessions: {log_viewer.get_sessions()}")
    print(f"--- Available agents: {log_viewer.get_agents()}")
    
    # 5. Demonstrate programmatic log access
    print("\n5. Programmatic log access...")
    
    # Get logs as data for processing
    logs = mas.get_logs(lines=10, format="json")
    print(f"Retrieved {len(logs)} log entries for processing")
    
    # Count log levels
    level_counts = {}
    for log_entry in logs:
        level = log_entry.get("level", "UNKNOWN")
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("Log level distribution:")
    for level, count in level_counts.items():
        print(f"  {level}: {count}")
    
    print("\n=== Demo completed ===")
    print(f"Check the './demo_logs' directory for all generated log files.")
    print("You can also use the LogViewer class to analyze logs programmatically.")


def demonstrate_real_usage():
    """
    Demonstrate how logging would work in a real flutter_app_builder scenario.
    """
    print("\n=== Real Usage Example ===")
    
    # This is how you'd use logging in your actual flutter_app_builder code
    logger = mas.get_logger("flutter_app_builder")
    
    # Example: Processing a configuration file
    config_file = "flutter_config.yaml"
    logger.info(f"Loading configuration from {config_file}", extra={
        "component": "config_loader",
        "operation": "load",
        "metadata": {"config_file": config_file}
    })
    
    # Example: Agent creation
    agent_id = "ui_designer_agent"
    logger.info(f"Creating agent: {agent_id}", extra={
        "component": "agent_manager",
        "operation": "create_agent",
        "agent_id": agent_id,
        "metadata": {"agent_type": "ui_designer", "specialization": "flutter"}
    })
    
    # Example: Task execution
    task_id = "generate_login_screen"
    logger.info(f"Executing task: {task_id}", extra={
        "component": "task_executor",
        "operation": "execute_task",
        "agent_id": agent_id,
        "task_id": task_id,
        "metadata": {"screen_type": "login", "requirements": ["email", "password", "remember_me"]}
    })
    
    # Example: Tool usage
    tool_name = "flutter_code_generator"
    logger.debug(f"Using tool: {tool_name}", extra={
        "component": "tool_executor",
        "operation": "use_tool",
        "agent_id": agent_id,
        "tool_name": tool_name,
        "metadata": {"input_params": {"widget_type": "Form", "validation": True}}
    })
    
    print("✓ Real usage logging examples added to logs")


if __name__ == "__main__":
    main()
    demonstrate_real_usage()
    
    print("\n" + "="*50)
    print("INTEGRATION GUIDE:")
    print("="*50)
    print("""
To integrate logging into your flutter_app_builder:

1. Import multiagenticsystem:
   import multiagenticsystem as mas

2. Set up logging at the start of your script:
   mas.setup_logging(
       level="INFO",
       log_dir="./flutter_builder_logs",
       session_id="your_session_id"
   )

3. Get a logger for your component:
   logger = mas.get_logger("flutter_app_builder")

4. Log important events:
   logger.info("Generated Flutter widget", extra={
       "component": "widget_generator",
       "operation": "generate",
       "metadata": {"widget_type": "StatefulWidget", "name": "LoginScreen"}
   })

5. View logs during development:
   mas.view_logs(lines=20)  # View last 20 log lines
   
   # Or use LogViewer for advanced analysis:
   viewer = mas.LogViewer()
   viewer.print_summary()
   viewer.filter_logs(component="widget_generator")

6. For real-time monitoring during long operations:
   viewer.tail_logs(lines=10)  # Like 'tail -f'
""")
