# 🚀 Flutter App Builder - Quick Start Guide

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **LLM API Keys** (at least one required):
   - OpenAI API key (for GPT-4) - Get from https://platform.openai.com/api-keys
   - Anthropic API key (for Claude) - Get from https://console.anthropic.com/
3. **Flutter** (optional, but recommended for full functionality)

## 🔧 Setup Instructions

### Step 1: Set up API Keys

Edit the `.env` file in the root directory:

```bash
# Open the .env file
nano .env
```

Replace the placeholder values with your actual API keys:
```env
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
```

### Step 2: Run the Setup Script

```bash
# Make sure you're in the project root
cd /workspaces/multiagenticsystem

# Run the setup script
./setup_flutter_builder.sh
```

### Step 3: Choose Your Demo

#### Option 1: Simple Demo (Recommended for first time)
```bash
cd examples/flutter_app_builder
python simple_demo.py
```

This runs a simplified version that shows the core concepts without requiring real API calls.

#### Option 2: Full Interactive Version
```bash
cd examples/flutter_app_builder  
python flutter_builder.py
```

This is the complete implementation with real LLM integration.

#### Option 3: Configuration-Driven Version
```bash
cd examples/flutter_app_builder
python config_driven_builder.py
```

This shows how to set up the system using YAML configuration files.

## 🎯 Example Usage

Once you run any of the demos, you can try these commands:

```bash
# Build a todo app
build my_todo_app A simple todo list with add, edit, and delete functionality

# Build a calculator
build calculator A basic calculator app with arithmetic operations

# Build a weather app
build weather_app A weather app that shows current conditions and forecasts

# Show system information
info

# Exit
quit
```

## 🔍 What You'll See

The system will:

1. **Initialize** the multi-agent system with two specialized agents
2. **Create** a Flutter project structure
3. **Implement** features using the ImplementationAgent (GPT-4)
4. **Test** the implementation using the TestingAgent (Claude-3.5-Sonnet)
5. **Generate** a complete Flutter app with tests

## 🛠️ Troubleshooting

### API Key Issues
```bash
# If you get API key errors, check your .env file:
cat .env

# Make sure the keys are properly set:
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Import Errors
```bash
# Make sure you're in the correct directory:
cd /workspaces/multiagenticsystem

# Activate virtual environment if created:
source venv/bin/activate

# Install dependencies:
pip install -r examples/flutter_app_builder/requirements.txt
```

### Flutter Not Found
```bash
# Check if Flutter is installed:
flutter --version

# If not installed, the demo will still work in simulation mode
```

## 📚 Understanding the Architecture

The Flutter App Builder demonstrates:

- **Multi-Agent Collaboration**: Two agents working together
- **Hierarchical Tool Sharing**: Both agents share Terminal and FileManager tools
- **Multi-LLM Integration**: GPT-4 for coding, Claude for testing
- **Real-World Workflow**: Actual software development process
- **Configuration-Driven Setup**: YAML-based system configuration

## 🎓 Next Steps

After running the demos:

1. **Explore the code** to understand the multi-agent patterns
2. **Modify the agents** to add new capabilities
3. **Create custom tools** for your specific needs
4. **Build your own** multi-agent system for different use cases

## 🆘 Need Help?

- Check the logs in the `logs/` directory
- Review the comprehensive README in `examples/flutter_app_builder/README.md`
- Examine the configuration file `examples/flutter_app_builder/flutter_config.yaml`

Happy building! 🚀
