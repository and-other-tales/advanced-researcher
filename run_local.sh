#!/bin/bash
set -e

echo "=========================================="
echo "Advanced Researcher - Local Startup Script"
echo "=========================================="

# Create data directory if it doesn't exist
DATA_DIR="./data"
mkdir -p "$DATA_DIR"
DATA_MOUNT_PATH="$(realpath $DATA_DIR)"
echo "Using data directory: $DATA_MOUNT_PATH"

# Create backend directories if they don't exist
mkdir -p backend/utils
mkdir -p backend/static
touch backend/utils/__init__.py
touch backend/static/__init__.py

# Check if we already have essential environment variables set
HAS_ENV_VARS=false
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ] || [ -n "$FIREWORKS_API_KEY" ] || [ "$USE_OLLAMA" = "true" ]; then
    HAS_ENV_VARS=true
    echo "Using environment variables from system environment"
fi

# Create .env file if it doesn't exist and we don't have environment variables set
if [ ! -f .env ] && [ "$HAS_ENV_VARS" = "false" ]; then
    echo "Warning: No environment variables found and .env file not found"
    if [ -f .env.example ]; then
        echo "Creating .env file from .env.example"
        cp .env.example .env
        echo "Please edit .env file to add your API keys and configuration"
    else
        echo "Warning: No .env.example file exists"
    fi
fi

# Set default values for HOST and PORT
export HOST=${HOST:-"127.0.0.1"}
export PORT=${PORT:-"8080"}
export DATA_MOUNT_PATH=${DATA_MOUNT_PATH:-"$DATA_MOUNT_PATH"}
export USE_LOCAL=true

# Run verification script to check for required credentials
echo "Verifying environment setup..."
python -c "
import sys
import os

try:
    # Check API keys
    api_keys = [
        os.environ.get('OPENAI_API_KEY'),
        os.environ.get('ANTHROPIC_API_KEY'),
        os.environ.get('GOOGLE_API_KEY'),
        os.environ.get('FIREWORKS_API_KEY')
    ]
    
    # Check LangSmith configuration
    langsmith_api_key = os.environ.get('LANGSMITH_API_KEY')
    langsmith_tracing = os.environ.get('LANGSMITH_TRACING', '').lower() in ('true', '1', 't', 'yes', 'y')
    if langsmith_api_key and langsmith_tracing:
        print('LangSmith tracing is enabled')
    else:
        print('LangSmith tracing is disabled')
    use_ollama = os.environ.get('USE_OLLAMA', '').lower() in ('true', '1', 't', 'yes', 'y')
    
    if not any(api_keys) and not use_ollama:
        print('Error: No API keys found and local models not enabled')
        print('Please add at least one API key to your .env file:')
        print('- OPENAI_API_KEY for OpenAI models')
        print('- ANTHROPIC_API_KEY for Anthropic Claude models')
        print('- GOOGLE_API_KEY for Google Gemini models')
        print('- FIREWORKS_API_KEY for Fireworks models')
        print('Or set USE_OLLAMA=true for local models via Ollama')
        sys.exit(1)
    else:
        print('Environment verified successfully!')
        
        # Get host and port from environment
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', '8080'))
        print(f'Server will be available at http://{host}:{port}')
except Exception as e:
    print(f'Warning: Verification error: {e}. Continuing anyway.')
"

# Check if frontend static files exist
if [ ! -d "backend/static/app" ] && [ -d "frontend" ]; then
    echo "Frontend static files not found. Do you want to build the frontend now? (y/n)"
    read -r build_frontend
    if [ "$build_frontend" = "y" ]; then
        echo "Building frontend..."
        ./build_and_deploy.sh
    else
        echo "Continuing without frontend build. Note that the web UI may not work correctly."
    fi
fi

# Print startup information
echo "Starting application with configuration:"
echo "- Host: $HOST"
echo "- Port: $PORT"
echo "- Data path: $DATA_MOUNT_PATH"
echo "- Local mode: enabled"

# Run the server with the correct configuration
echo "Starting local server..."
python main.py