#!/bin/bash
set -e

# Print banner
echo "======================================"
echo "Advanced Researcher - Docker Entrypoint"
echo "======================================"

# This script runs before supervisord starts the services
# It sets up the environment and ensures all required files exist

# Check if essential directories exist
mkdir -p /data
mkdir -p /app/backend/utils

# Ensure required Python modules exist
if [ ! -f /app/backend/utils/__init__.py ]; then
    echo "Creating backend/utils/__init__.py"
    cat > /app/backend/utils/__init__.py << 'EOF'
"""Utility modules for the Advanced Researcher project."""
EOF
fi

# Check if we already have essential environment variables set
HAS_ENV_VARS=false
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ] || [ -n "$FIREWORKS_API_KEY" ] || [ "$USE_OLLAMA" = "true" ]; then
    HAS_ENV_VARS=true
    echo "Using environment variables from container environment"
fi

# Use only OS environment variables, no .env fallback
if [ "$HAS_ENV_VARS" = "false" ]; then
    echo "WARNING: No API keys set in environment variables"
    echo "NOT creating .env file as fallback - using OS environment variables only"
    # Create empty .env to prevent fallback attempts
    touch /app/.env
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

# Check if data directory is mounted
if [ ! -d "$DATA_MOUNT_PATH" ]; then
    echo "Creating data directory at $DATA_MOUNT_PATH"
    mkdir -p "$DATA_MOUNT_PATH"
    chmod 777 "$DATA_MOUNT_PATH"
fi

# Update Nginx configuration if custom ports are specified
if [ -n "$NGINX_PORT" ] && [ "$NGINX_PORT" != "8080" ]; then
    echo "Updating Nginx to listen on port $NGINX_PORT"
    sed -i "s/listen 8080/listen $NGINX_PORT/g" /etc/nginx/conf.d/default.conf
fi

if [ -n "$BACKEND_PORT" ] && [ "$BACKEND_PORT" != "8000" ]; then
    echo "Updating Nginx to proxy to backend port $BACKEND_PORT"
    sed -i "s/127.0.0.1:8000/127.0.0.1:$BACKEND_PORT/g" /etc/nginx/conf.d/default.conf
    # Also update supervisord configuration
    sed -i "s/PORT=8000/PORT=$BACKEND_PORT/g" /etc/supervisord.conf
fi

# Print startup information
echo "Advanced Researcher is configured with:"
echo "- Frontend files served by Nginx on port $NGINX_PORT"
echo "- Backend API running on port $BACKEND_PORT"
echo "- Data directory at $DATA_MOUNT_PATH"
echo "- Local mode: $USE_LOCAL"

# This script is now used for setup only, not for launching the application
# Supervisord will start both Nginx and the Python backend
echo "Setup completed. Starting services with supervisord..."