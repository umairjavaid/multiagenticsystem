# Flutter App Builder - Multi-Agent System Example

This example demonstrates how to use the MultiAgenticSystem to build Flutter applications with specialized agents working together through shared tools.

## 🏗️ Architecture

The system consists of two specialized agents:

### 🤖 Agents

1. **ImplementationAgent** (GPT-4)
   - Expert Flutter developer
   - Handles project setup and feature implementation
   - Creates and modifies Dart files
   - Manages dependencies and project structure

2. **TestingAgent** (Claude-3.5-Sonnet)
   - Quality assurance specialist
   - Creates comprehensive test suites
   - Performs code analysis and validation
   - Provides quality reports and recommendations

### 🔧 Shared Tools

Both agents have access to:

1. **Terminal Tool**
   - Execute Flutter CLI commands
   - Run tests and builds
   - Perform system operations

2. **FileManager Tool**
   - Create and modify project files
   - Read existing code for analysis
   - Manage project structure

## 📋 Workflow

The development process follows these tasks:

1. **ProjectSetup**: Initialize Flutter project and architecture
2. **FeatureImplementation**: Implement core app functionality
3. **TestingAndQA**: Create tests and validate quality
4. **IntegrationAndDeployment**: Final testing and deployment prep

## 🚀 Usage

### Prerequisites

Make sure you have Flutter installed:
```bash
# Install Flutter (if not already installed)
# Follow instructions at: https://flutter.dev/docs/get-started/install

# Verify installation
flutter doctor
```

### Running the Example

There are two ways to run this example:

#### 1. Programmatic Version (Recommended for learning)

```bash
cd examples/flutter_app_builder
python flutter_builder.py
```

This version shows how to:
- Create agents programmatically
- Define tools with custom functions
- Set up hierarchical tool sharing
- Execute tasks interactively

#### 2. Configuration-Driven Version

```bash
cd examples/flutter_app_builder
python config_driven_builder.py
```

This version demonstrates:
- Loading system from YAML configuration
- Declarative agent and tool definitions
- Configuration-based workflow setup

### 🎯 Interactive Commands

Both versions support these commands:

- `build [app_name] [description]` - Start building a new Flutter app
- `test [app_name]` - Run tests for an existing app (config version)
- `info`/`status` - Show system information
- `agents` - List all agents (programmatic version)
- `tools` - List all tools (programmatic version)
- `tasks` - List all tasks (programmatic version)
- `quit` - Exit the system

### 📝 Example Usage

```bash
🎯 Enter command: build my_todo_app A simple todo list app with add, edit, and delete functionality

🏗️ Building Flutter app: my_todo_app
📝 Description: A simple todo list app with add, edit, and delete functionality

📦 Executing ProjectSetup task...
✅ Setup completed: Done

🔨 Executing FeatureImplementation task...
✅ Implementation completed: Done

🧪 Executing TestingAndQA task...
✅ Testing completed: Done

🎉 Flutter app 'my_todo_app' build completed!
📁 Project location: ./flutter_projects/my_todo_app
```

## 🔧 Customization

### Adding New Tools

You can extend the system with additional tools:

```python
def create_design_tool() -> Tool:
    """Create a tool for design operations."""
    
    def generate_ui_mockup(description: str) -> str:
        # Your implementation here
        return f"Generated UI mockup for: {description}"
    
    return Tool(
        name="Designer",
        func=generate_ui_mockup,
        description="Generate UI mockups and design assets"
    )

# Set tool as shared between agents
design_tool = create_design_tool()
design_tool.set_shared(impl_agent, test_agent)
system.register_tool(design_tool)
```

### Adding New Agents

Add specialized agents for specific tasks:

```python
def create_ui_agent() -> Agent:
    """Create a UI/UX specialized agent."""
    
    return Agent(
        name="UIAgent",
        description="UI/UX specialist for Flutter apps",
        system_prompt="You are a UI/UX expert specializing in Flutter design...",
        llm_provider="openai",
        llm_model="gpt-4"
    )
```

### Modifying Workflows

Customize the task workflow in the configuration file:

```yaml
tasks:
  - name: "CustomTask"
    description: "Your custom task"
    steps:
      - agent: "ImplementationAgent"
        action: "Custom implementation step"
        expected_output: "Expected result"
    dependencies: ["ProjectSetup"]
```

## 🎯 Key Features Demonstrated

1. **Hierarchical Tool Sharing**
   - Terminal tool shared between implementation and testing agents
   - File management accessible to both agents
   - Clear separation of concerns

2. **Multi-LLM Integration**
   - GPT-4 for implementation (code generation strength)
   - Claude-3.5-Sonnet for testing (analytical strength)
   - Different models optimized for different tasks

3. **Configuration-Driven Setup**
   - YAML-based system configuration
   - Declarative agent and tool definitions
   - Easy to modify without code changes

4. **Collaborative Workflows**
   - Sequential task execution
   - Agent handoffs between implementation and testing
   - Shared context and memory

5. **Real-world Application**
   - Actual Flutter development workflow
   - Production-ready patterns
   - Extensible architecture

## 🔍 Generated Output

The system will create:

```
flutter_projects/
└── [app_name]/
    ├── lib/
    │   ├── main.dart
    │   ├── models/
    │   ├── widgets/
    │   └── screens/
    ├── test/
    │   ├── unit/
    │   ├── widget/
    │   └── integration/
    ├── pubspec.yaml
    └── README.md
```

## 🚀 Next Steps

- Extend with additional agents (UI/UX, DevOps, etc.)
- Add more sophisticated tools (API integration, database tools)
- Implement continuous integration workflows
- Add deployment automation
- Create custom Flutter project templates

This example showcases the power of multi-agent collaboration in real-world software development scenarios!
