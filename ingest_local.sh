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
from pathlib import Path
# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from backend.utils import get_env, get_bool_env
    api_keys = [
        get_env('OPENAI_API_KEY'),
        get_env('ANTHROPIC_API_KEY'),
        get_env('GOOGLE_API_KEY'),
        get_env('FIREWORKS_API_KEY')
    ]
    
    # Check LangSmith configuration
    langsmith_api_key = get_env('LANGSMITH_API_KEY')
    langsmith_tracing = get_bool_env('LANGSMITH_TRACING', False)
    if langsmith_api_key and langsmith_tracing:
        print('LangSmith tracing is enabled')
    else:
        print('LangSmith tracing is disabled')
    use_ollama = get_env('USE_OLLAMA', '').lower() == 'true'
    
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
except ImportError:
    print('Warning: Could not import backend.utils. Continuing anyway.')
"

# Run the ingestion script
echo "Starting ingestion process for local deployment"
python -c "
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Make sure environment is loaded from main.py first
import main
os.environ['USE_LOCAL'] = 'true'  # Ensure we use local modules

# Then run ingestion
import backend.local_ingest
print('Ingestion completed successfully')
"