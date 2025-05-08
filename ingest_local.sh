#!/bin/bash
set -e

echo "============================================="
echo "Advanced Researcher - Local Ingestion Script"
echo "============================================="

# Create data directory if it doesn't exist
DATA_DIR="./data"
mkdir -p "$DATA_DIR"
DATA_MOUNT_PATH="$(realpath $DATA_DIR)"
echo "Using data directory: $DATA_MOUNT_PATH"

# Create backend directories if they don't exist
mkdir -p backend/utils
mkdir -p backend/static
touch backend/utils/__init__.py

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

# Set default values for environment variables
export HOST=${HOST:-"127.0.0.1"}
export PORT=${PORT:-"8080"}
export DATA_MOUNT_PATH=${DATA_MOUNT_PATH:-"$DATA_MOUNT_PATH"}
export USE_LOCAL=true
export FORCE_UPDATE=${FORCE_UPDATE:-"false"}

# Run verification script to check for required credentials
echo "Verifying environment setup..."
python -c "
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

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
except Exception as e:
    print(f'Warning: Verification error: {e}. Continuing anyway.')
"

# Print status information
echo "Starting ingestion process with configuration:"
echo "- Data path: $DATA_MOUNT_PATH"
echo "- Force update: $FORCE_UPDATE"
echo "- Local mode: enabled"

# Run the ingestion script
echo "Starting ingestion process for local deployment..."
python -c "
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Make sure environment is loaded from main.py first
import main

# Set environment variables
os.environ['USE_LOCAL'] = 'true'
os.environ['DATA_MOUNT_PATH'] = '$DATA_MOUNT_PATH'

# Determine if we should force re-ingestion
force_update = os.environ.get('FORCE_UPDATE', '').lower() in ('true', '1', 't', 'yes', 'y')
if force_update:
    print('Force update enabled - will re-ingest all documents')

try:
    # Then run ingestion
    import backend.local_ingest
    print('Ingestion completed successfully')
except Exception as e:
    print(f'Error during ingestion: {e}')
    sys.exit(1)
"