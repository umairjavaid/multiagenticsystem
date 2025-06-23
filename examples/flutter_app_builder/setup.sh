#!/bin/bash
# Setup script for Flutter App Builder example

echo "🚀 Setting up Flutter App Builder Example"
echo "=" * 40

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if Flutter is available (optional but recommended)
if ! command -v flutter &> /dev/null; then
    echo "⚠️  Flutter is not installed. Some features may not work."
    echo "   Install Flutter from: https://flutter.dev/docs/get-started/install"
else
    echo "✅ Flutter found: $(flutter --version | head -n1)"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating project directories..."
mkdir -p apps
mkdir -p logs

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Set up your LLM API keys:"
echo "   export OPENAI_API_KEY='your-key-here'"
echo "   export ANTHROPIC_API_KEY='your-key-here'"
echo ""
echo "2. Run the examples:"
echo "   python simple_demo.py              # Basic demonstration"
echo "   python flutter_builder.py          # Full interactive version"
echo "   python config_driven_builder.py    # Configuration-driven version"
echo ""
echo "3. Try building an app:"
echo "   Enter: build my_app A simple mobile app with basic functionality"
