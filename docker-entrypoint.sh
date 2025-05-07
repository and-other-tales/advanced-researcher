#!/bin/bash
set -e

# Check if we already have essential environment variables set
HAS_ENV_VARS=false
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ] || [ -n "$FIREWORKS_API_KEY" ] || [ "$USE_OLLAMA" = "true" ]; then
    HAS_ENV_VARS=true
fi

# Create .env file if it doesn't exist and we don't have environment variables set
if [ ! -f .env ] && [ "$HAS_ENV_VARS" = "false" ]; then
    echo "No environment variables found. Creating .env file from .env.example"
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        echo "Warning: No environment variables found and no .env.example file exists"
    fi
fi

# Ensure backend/utils/__init__.py exists
if [ ! -f /app/backend/utils/__init__.py ]; then
    echo "Creating backend/utils/__init__.py"
    cat > /app/backend/utils/__init__.py << 'EOF'
"""Utility modules for the Advanced Researcher project."""
EOF
fi

# Check if environment is properly configured
echo "Checking environment configuration..."
HAS_API_KEY=false

# Check for API keys
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ] || [ -n "$FIREWORKS_API_KEY" ]; then
    HAS_API_KEY=true
fi

# Check for Ollama configuration
if [ "$USE_OLLAMA" = "true" ]; then
    echo "Configured to use Ollama for local models"
    HAS_API_KEY=true
fi

if [ "$HAS_API_KEY" = "false" ]; then
    echo "WARNING: No API keys found. Please set at least one of the following:"
    echo "- OPENAI_API_KEY for OpenAI models"
    echo "- ANTHROPIC_API_KEY for Anthropic Claude models"
    echo "- GOOGLE_API_KEY for Google Gemini models"
    echo "- FIREWORKS_API_KEY for Fireworks models"
    echo "Or set USE_OLLAMA=true for local models via Ollama"
    # Continue anyway - we'll use dummy embeddings as fallback
fi

# Check for LangSmith configuration
if [ -n "$LANGSMITH_API_KEY" ] && [ "$LANGSMITH_TRACING" = "true" ]; then
    echo "LangSmith tracing is enabled"
else
    echo "LangSmith tracing is disabled"
fi

# Run the application with the correct command
if [ "$USE_LOCAL" = "true" ]; then
    echo "Starting local application on $HOST:$PORT"
    exec python -m backend.local_main
else
    echo "Starting standard application on $HOST:$PORT"
    exec python -m main
fi