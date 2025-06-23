#!/usr/bin/env python3
"""
Enhanced Flutter App Builder with comprehensive logging.

This script demonstrates how to integrate MultiAgenticSystem logging
into the flutter_app_builder example.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import multiagenticsystem
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# Import the multiagenticsystem with logging
try:
    import multiagenticsystem as mas
    from multiagenticsystem.logging import (
        setup_logging, view_logs, search_logs, get_summary,
        get_logs_for_agent, get_llm_logs, export_logs,
        log_info, log_debug, log_warning, log_error
    )
    print("✅ MultiAgenticSystem with logging imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MultiAgenticSystem: {e}")
    sys.exit(1)


def setup_logging_for_flutter_builder():
    """Setup comprehensive logging for flutter app builder."""
    print("🔧 Setting up comprehensive logging for Flutter App Builder...")
    
    # Setup logging with custom directory for flutter builder
    log_config = setup_logging(
        verbose=True,  # Enable debug logging
        log_directory="./flutter_builder_logs",
        enable_json_logs=True  # Enable structured JSON logs
    )
    
    log_info("Flutter App Builder session started", 
             builder_type="config_driven", 
             session_type="demo")
    
    return log_config


def create_flutter_builder_system():
    """Create the flutter builder system with logging."""
    log_info("Creating Flutter Builder system...")
    
    try:
        # Create the main system
        system = mas.System(enable_logging=True, verbose=True)
        
        # Create specialized agents for flutter development
        ui_designer = mas.Agent(
            name="UIDesigner",
            role="Flutter UI Designer",
            system_prompt="""You are a Flutter UI designer agent. You create beautiful, 
            modern Flutter user interfaces based on requirements. You excel at layout 
            design, color schemes, and user experience."""
        )
        
        code_generator = mas.Agent(
            name="CodeGenerator", 
            role="Flutter Code Generator",
            system_prompt="""You are a Flutter code generation agent. You write clean,
            efficient Flutter/Dart code following best practices. You create complete
            widgets, screens, and app structures."""
        )
        
        config_manager = mas.Agent(
            name="ConfigManager",
            role="Configuration Manager", 
            system_prompt="""You are a configuration management agent. You handle
            app configuration, settings, and ensure all components work together
            properly."""
        )
        
        # Create tools for flutter development
        def generate_widget_code(widget_type: str, properties: dict) -> str:
            """Generate Flutter widget code."""
            log_debug("Generating widget code", widget_type=widget_type, properties=properties)
            
            # Simplified code generation for demo
            code = f"""
class {widget_type}Widget extends StatelessWidget {{
  @override
  Widget build(BuildContext context) {{
    return Container(
      // Generated properties: {properties}
      child: Text('Generated {widget_type}'),
    );
  }}
}}"""
            log_info("Widget code generated successfully", 
                    widget_type=widget_type, 
                    lines_of_code=len(code.split('\n')))
            return code
        
        def validate_flutter_config(config: dict) -> bool:
            """Validate Flutter app configuration."""
            log_debug("Validating Flutter configuration", config_keys=list(config.keys()))
            
            required_keys = ["app_name", "package_name", "target_platform"]
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                log_warning("Configuration validation failed", missing_keys=missing_keys)
                return False
            
            log_info("Configuration validation passed", config=config)
            return True
        
        def build_app_structure(app_config: dict) -> dict:
            """Build the Flutter app structure."""
            log_info("Building Flutter app structure", app_name=app_config.get("app_name"))
            
            structure = {
                "lib": {
                    "main.dart": "// Main app entry point",
                    "screens": {},
                    "widgets": {},
                    "models": {},
                    "services": {}
                },
                "pubspec.yaml": f"name: {app_config.get('app_name', 'my_app')}",
                "android": {},
                "ios": {}
            }
            
            log_info("App structure built successfully", 
                    directories=len(structure), 
                    app_name=app_config.get("app_name"))
            return structure
        
        # Create tools
        widget_tool = mas.Tool(
            name="generate_widget_code",
            description="Generate Flutter widget code",
            function=generate_widget_code
        )
        
        config_tool = mas.Tool(
            name="validate_flutter_config", 
            description="Validate Flutter app configuration",
            function=validate_flutter_config
        )
        
        structure_tool = mas.Tool(
            name="build_app_structure",
            description="Build Flutter app directory structure", 
            function=build_app_structure
        )
        
        # Add agents and tools to system
        system.add_agent(ui_designer)
        system.add_agent(code_generator)
        system.add_agent(config_manager)
        
        system.add_tool(widget_tool)
        system.add_tool(config_tool)
        system.add_tool(structure_tool)
        
        # Create a comprehensive task
        build_task = mas.Task(
            name="build_flutter_app",
            description="Build a complete Flutter application from configuration",
            agents=[ui_designer, code_generator, config_manager],
            tools=[widget_tool, config_tool, structure_tool]
        )
        
        system.add_task(build_task)
        
        log_info("Flutter Builder system created successfully",
                agents=len(system.agents) if hasattr(system, 'agents') else 0,
                tools=len(system.tools) if hasattr(system, 'tools') else 0,
                tasks=len(system.tasks) if hasattr(system, 'tasks') else 0)
        
        return system
        
    except Exception as e:
        log_error("Failed to create Flutter Builder system", error=str(e))
        raise


def simulate_flutter_app_building():
    """Simulate the process of building a Flutter app with logging."""
    log_info("Starting Flutter app building simulation...")
    
    try:
        # Simulate configuration
        app_config = {
            "app_name": "awesome_music_app",
            "package_name": "com.example.music",
            "target_platform": "android",
            "theme": {
                "primary_color": "#FF6B35",
                "secondary_color": "#004E89", 
                "theme_mode": "dark"
            },
            "features": ["music_player", "playlist", "search", "favorites"]
        }
        
        log_info("App configuration defined", config=app_config)
        
        # Simulate agent interactions (since we don't have LLM access in demo)
        log_info("UIDesigner agent starting design process...")
        
        # Simulate tool usage
        widget_code = generate_widget_code("MusicPlayer", {
            "color": app_config["theme"]["primary_color"],
            "controls": ["play", "pause", "next", "previous"]
        })
        
        config_valid = validate_flutter_config(app_config)
        
        if config_valid:
            app_structure = build_app_structure(app_config)
            log_info("Flutter app built successfully", 
                    structure_items=len(app_structure),
                    app_name=app_config["app_name"])
        else:
            log_error("App building failed due to invalid configuration")
            
    except Exception as e:
        log_error("Flutter app building simulation failed", error=str(e))


def analyze_building_logs():
    """Analyze the logs from the building process."""
    print("\n" + "="*60)
    print("📊 ANALYZING FLUTTER BUILDER LOGS")
    print("="*60)
    
    # Get overall summary
    print("\n📋 Overall Activity Summary:")
    get_summary()
    
    # View recent activity
    print("\n📄 Recent Activity (last 15 lines):")
    view_logs(lines=15)
    
    # Search for specific patterns
    print("\n🔍 Flutter-specific Activity:")
    search_logs("flutter")
    
    print("\n🔍 Error Analysis:")
    search_logs("error")
    
    # Agent-specific logs
    print("\n🤖 Agent Activity:")
    agents = ["UIDesigner", "CodeGenerator", "ConfigManager"]
    for agent in agents:
        agent_logs = get_logs_for_agent(agent)
        print(f"  {agent}: {len(agent_logs)} log entries")
    
    # LLM usage (if any)
    llm_logs = get_llm_logs()
    print(f"\n🧠 LLM Interactions: {len(llm_logs)} entries")


def export_session_logs():
    """Export the session logs for later analysis."""
    print("\n" + "="*60)
    print("📤 EXPORTING SESSION LOGS")
    print("="*60)
    
    timestamp = mas.utils.logger.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = f"./flutter_builder_logs/session_{timestamp}.json"
    
    success = export_logs(export_file)
    
    if success:
        print(f"✅ Session logs exported to: {export_file}")
        
        # Also create a summary report
        summary_file = f"./flutter_builder_logs/summary_{timestamp}.txt"
        try:
            with open(summary_file, 'w') as f:
                f.write("Flutter App Builder Session Summary\n")
                f.write("="*50 + "\n\n")
                f.write(f"Session Export: {export_file}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                # Get summary data
                from multiagenticsystem.utils.logger import get_log_viewer
                viewer = get_log_viewer("./flutter_builder_logs")
                summary = viewer.generate_summary_report()
                
                f.write(f"Total Events: {sum(summary['event_counts'].values())}\n")
                f.write(f"Agents: {', '.join(summary['agents'])}\n")
                f.write(f"Tools Used: {', '.join(summary['tools_used'])}\n")
                f.write(f"Errors: {summary['errors']}\n")
                f.write(f"Warnings: {summary['warnings']}\n\n")
                
                f.write("Event Breakdown:\n")
                for event_type, count in summary['event_counts'].items():
                    f.write(f"  {event_type}: {count}\n")
                    
            print(f"✅ Summary report created: {summary_file}")
            
        except Exception as e:
            print(f"⚠️  Could not create summary report: {e}")
    else:
        print("❌ Failed to export session logs")


def demonstrate_real_time_monitoring():
    """Show how to monitor logs in real-time during development."""
    print("\n" + "="*60)
    print("⏱️  REAL-TIME MONITORING EXAMPLE")
    print("="*60)
    
    print("For real-time monitoring during Flutter app development:")
    print()
    
    monitoring_code = '''
# Terminal 1 - Run your Flutter builder:
python flutter_app_builder_with_logging.py

# Terminal 2 - Monitor logs in real-time:
tail -f ./flutter_builder_logs/multiagenticsystem_*.log

# Terminal 3 - Monitor JSON logs:
tail -f ./flutter_builder_logs/multiagenticsystem_*.json | jq .

# Python script for programmatic monitoring:
import time
from multiagenticsystem.logging import get_logs_for_session, get_summary

while True:
    print("\\n=== Current Status ===")
    get_summary()
    time.sleep(10)  # Check every 10 seconds
'''
    
    print(monitoring_code)
    print("✅ Real-time monitoring examples provided")


def main():
    """Main demo function."""
    print("🚀 Flutter App Builder with Comprehensive Logging")
    print("="*60)
    
    try:
        # Setup logging
        log_config = setup_logging_for_flutter_builder()
        print(f"📁 Logs directory: {log_config['log_directory']}")
        print(f"🆔 Session ID: {log_config['session_id']}")
        
        # Create the system
        system = create_flutter_builder_system()
        
        # Simulate building process
        simulate_flutter_app_building()
        
        # Analyze what happened
        analyze_building_logs()
        
        # Export session data
        export_session_logs()
        
        # Show monitoring options
        demonstrate_real_time_monitoring()
        
        print("\n" + "="*60)
        print("🎉 FLUTTER BUILDER LOGGING DEMO COMPLETED!")
        print("="*60)
        print("\n📋 What you can do now:")
        print("✅ Examine log files in ./flutter_builder_logs/")
        print("✅ Import the JSON logs for custom analysis")
        print("✅ Use the monitoring examples for real development")
        print("✅ Integrate this logging into your actual flutter_app_builder")
        
    except Exception as e:
        log_error("Demo failed", error=str(e), demo_type="flutter_builder")
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
