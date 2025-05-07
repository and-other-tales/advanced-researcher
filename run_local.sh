#!/bin/bash
set -e

# Check if we already have essential environment variables set
HAS_ENV_VARS=false
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ] || [ -n "$FIREWORKS_API_KEY" ] || [ "$USE_OLLAMA" = "true" ]; then
    HAS_ENV_VARS=true
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
elif [ "$HAS_ENV_VARS" = "true" ]; then
    echo "Using environment variables from system environment"
fi

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
        
        # Get host and port
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', '8000'))
        print(f'Server will be available at http://{host}:{port}')
except Exception as e:
    print(f'Warning: Verification error: {e}. Continuing anyway.')
"

# Run the server with the centralized entrypoint and local flag
echo "Starting local server..."
USE_LOCAL=true DATA_MOUNT_PATH=/tmp/data python -m main