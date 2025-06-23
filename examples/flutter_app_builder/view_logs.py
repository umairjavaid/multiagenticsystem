#!/usr/bin/env python3
"""
Simple log viewer for the multiagenticsystem.
Run this script to view logs from your multiagenticsystem usage.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import multiagenticsystem as mas


def main():
    """Main function to view logs."""
    
    print("=== MultiAgenticSystem Log Viewer ===\n")
    
    # Create log viewer
    viewer = mas.LogViewer()
    
    # Check if there are any log files
    log_files = viewer.get_log_files()
    
    if not log_files:
        print("❌ No log files found!")
        print("Make sure you've run some multiagenticsystem code that generates logs.")
        print("Log files are typically stored in a 'logs' directory.")
        return
    
    print(f"📁 Found {len(log_files)} log file(s):")
    for i, log_file in enumerate(log_files[:5], 1):  # Show up to 5 files
        print(f"  {i}. {log_file.name} ({log_file.stat().st_size} bytes)")
    
    print("\n" + "="*50)
    
    # Show summary
    print("📊 Log Summary:")
    viewer.print_summary()
    
    print("\n" + "="*50)
    
    # Show recent logs
    print("📝 Recent Log Entries (last 10):")
    viewer.tail_logs(lines=10)
    
    print("\n" + "="*50)
    
    # Interactive menu
    while True:
        print("\n🔍 What would you like to do?")
        print("1. View recent logs (last N lines)")
        print("2. Search logs")
        print("3. Filter by log level")
        print("4. Filter by component")
        print("5. Show log summary")
        print("6. List sessions")
        print("7. List agents")
        print("8. Quit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            try:
                lines = int(input("How many lines? (default 20): ") or "20")
                print(f"\n--- Last {lines} lines ---")
                viewer.tail_logs(lines=lines)
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == "2":
            search_text = input("Enter text to search for: ").strip()
            if search_text:
                print(f"\n--- Search results for '{search_text}' ---")
                viewer.search_logs(search_text, context_lines=1)
        
        elif choice == "3":
            level = input("Enter log level (DEBUG/INFO/WARNING/ERROR): ").strip().upper()
            if level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
                print(f"\n--- {level} logs ---")
                filtered_logs = viewer.filter_logs(level=level)
                for log in filtered_logs[-10:]:  # Show last 10
                    print(log.rstrip())
            else:
                print("❌ Invalid log level")
        
        elif choice == "4":
            component = input("Enter component name: ").strip()
            if component:
                print(f"\n--- Logs from component '{component}' ---")
                filtered_logs = viewer.filter_logs(component=component)
                for log in filtered_logs[-10:]:  # Show last 10
                    print(log.rstrip())
        
        elif choice == "5":
            print("\n--- Log Summary ---")
            viewer.print_summary()
        
        elif choice == "6":
            sessions = viewer.get_sessions()
            print(f"\n--- Sessions ({len(sessions)}) ---")
            for session in sessions:
                print(f"  • {session}")
        
        elif choice == "7":
            agents = viewer.get_agents()
            print(f"\n--- Agents ({len(agents)}) ---")
            for agent in agents:
                print(f"  • {agent}")
        
        elif choice == "8":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
