#!/bin/bash
# Flutter App Builder Setup Script

echo "🚀 Setting up Flutter App Builder"
echo "=" * 50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check environment variables
check_env_var() {
    if [ -z "${!1}" ]; then
        echo -e "${YELLOW}⚠️  $1 is not set${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 is set${NC}"
        return 0
    fi
}

echo "🔍 Checking prerequisites..."

# Check Python
if command_exists python3; then
    echo -e "${GREEN}✅ Python 3 found: $(python3 --version)${NC}"
else
    echo -e "${RED}❌ Python 3 is required but not found${NC}"
    exit 1
fi

# Check pip
if command_exists pip || command_exists pip3; then
    echo -e "${GREEN}✅ pip found${NC}"
else
    echo -e "${RED}❌ pip is required but not found${NC}"
    exit 1
fi

# Check Flutter (optional but recommended)
if command_exists flutter; then
    echo -e "${GREEN}✅ Flutter found: $(flutter --version | head -n1)${NC}"
    echo -e "${BLUE}ℹ️  Running flutter doctor...${NC}"
    flutter doctor --android-licenses > /dev/null 2>&1 || true
else
    echo -e "${YELLOW}⚠️  Flutter not found${NC}"
    echo -e "${BLUE}ℹ️  Install Flutter from: https://flutter.dev/docs/get-started/install${NC}"
    echo -e "${BLUE}ℹ️  The demo will work without Flutter, but some features may be limited${NC}"
fi

echo ""
echo "📦 Installing Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r examples/flutter_app_builder/requirements.txt

echo ""
echo "🔐 Checking environment variables..."

# Source .env file if it exists
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
else
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
fi

# Check required environment variables
at_least_one_api_key=0

if check_env_var "OPENAI_API_KEY"; then
    at_least_one_api_key=1
fi

if check_env_var "ANTHROPIC_API_KEY"; then
    at_least_one_api_key=1
fi

if [ $at_least_one_api_key -eq 0 ]; then
    echo -e "${RED}❌ No API keys found!${NC}"
    echo -e "${YELLOW}Please set at least one of the following in your .env file:${NC}"
    echo "  - OPENAI_API_KEY"
    echo "  - ANTHROPIC_API_KEY"
    echo ""
    echo -e "${BLUE}Example .env file:${NC}"
    echo "OPENAI_API_KEY=sk-your-key-here"
    echo "ANTHROPIC_API_KEY=sk-ant-your-key-here"
    exit 1
fi

echo ""
echo "📁 Creating project directories..."
mkdir -p examples/flutter_app_builder/apps
mkdir -p logs

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${BLUE}🎯 Next steps:${NC}"
echo ""
echo "1. Make sure you have set your API keys in .env file"
echo "2. Choose which demo to run:"
echo ""
echo -e "${GREEN}   # Simple demo (recommended for first time)${NC}"
echo "   cd examples/flutter_app_builder"
echo "   python simple_demo.py"
echo ""
echo -e "${GREEN}   # Full interactive version${NC}"
echo "   cd examples/flutter_app_builder"
echo "   python flutter_builder.py"
echo ""
echo -e "${GREEN}   # Configuration-driven version${NC}"
echo "   cd examples/flutter_app_builder"
echo "   python config_driven_builder.py"
echo ""
echo -e "${BLUE}3. Example commands to try:${NC}"
echo "   build my_todo_app A simple todo list with add/edit/delete tasks"
echo "   build calculator A basic calculator app with arithmetic operations"
echo "   info                    # Show system information"
echo "   quit                    # Exit the demo"
echo ""
echo -e "${YELLOW}💡 Tip: Start with simple_demo.py to understand the concepts!${NC}"
