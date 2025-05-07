#!/usr/bin/env python
"""Entry point for the Advanced Researcher application.

This script loads environment variables from .env file and starts the application
with the configured settings.
"""
import os
import sys
import logging
import uvicorn
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Create utils path if it doesn't exist
utils_path = os.path.join(project_root, "backend", "utils")
if not os.path.exists(utils_path):
    os.makedirs(utils_path, exist_ok=True)

# Create utils/__init__.py if it doesn't exist
utils_init = os.path.join(utils_path, "__init__.py")
if not os.path.exists(utils_init):
    with open(utils_init, "w") as f:
        f.write('"""Utility modules for the Advanced Researcher project."""\n')

# Create utils/env.py if it doesn't exist
utils_env = os.path.join(utils_path, "env.py")
if not os.path.exists(utils_env):
    from shutil import copyfile
    env_template = os.path.join(project_root, "backend", "utils", "env.py.template")
    if os.path.exists(env_template):
        copyfile(env_template, utils_env)
    else:
        # Create basic env.py with essential functions
        with open(utils_env, "w") as f:
            f.write('''"""Environment variable utilities for the Advanced Researcher project."""
import os
from typing import Any, Optional

def get_env(key: str, default: Any = None) -> Any:
    """Get environment variable or default."""
    return os.environ.get(key, default)

def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = get_env(key, default)
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "t", "yes", "y") if isinstance(value, str) else bool(value)

def get_int_env(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    value = get_env(key, default)
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def load_env_file(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if env_path and os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = value.strip()
''')

# Create .env file from template if it doesn't exist, but don't use it if we have environment variables set
env_file = os.path.join(project_root, ".env")
env_example = os.path.join(project_root, ".env.example")

# Check if we already have essential environment variables set
has_env_vars = any([
    os.environ.get("OPENAI_API_KEY"),
    os.environ.get("ANTHROPIC_API_KEY"),
    os.environ.get("GOOGLE_API_KEY"),
    os.environ.get("FIREWORKS_API_KEY"),
    os.environ.get("USE_OLLAMA") in ("true", "1", "t", "yes", "y")
])

if not os.path.exists(env_file) and not has_env_vars:
    if os.path.exists(env_example):
        logger.info(f"No environment variables found. Creating .env file from .env.example")
        from shutil import copyfile
        copyfile(env_example, env_file)
    else:
        logger.warning("No environment variables found and no .env.example file exists")

# Import environment utilities and load .env
try:
    from backend.utils.env import load_env_file, get_env, get_bool_env, get_int_env
    
    # Load environment variables from .env file
    env_path = os.path.join(project_root, ".env")
    load_env_file(env_path)
    if os.path.exists(env_path):
        logger.info(f"Environment loaded from {env_path}")
except ImportError as e:
    logger.warning(f"Error importing environment utilities: {e}")
    logger.warning("Continuing with system environment")

def check_api_keys() -> bool:
    """Check if any API keys are configured.
    
    Returns:
        bool: True if at least one API key is configured
    """
    try:
        # Check for API keys
        api_keys = [
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("ANTHROPIC_API_KEY"), 
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("FIREWORKS_API_KEY")
        ]
        
        # Check for Ollama config
        use_ollama = os.environ.get("USE_OLLAMA", "").lower() in ("true", "1", "t", "yes", "y")
        
        if any(api_keys) or use_ollama:
            return True
            
        logger.warning("No API keys found and local models not enabled")
        logger.warning("Please set at least one of the following:")
        logger.warning("- OPENAI_API_KEY for OpenAI models")
        logger.warning("- ANTHROPIC_API_KEY for Anthropic Claude models")
        logger.warning("- GOOGLE_API_KEY for Google Gemini models")
        logger.warning("- FIREWORKS_API_KEY for Fireworks models")
        logger.warning("Or set USE_OLLAMA=true for local models via Ollama")
        
        return False
    except Exception as e:
        logger.warning(f"Error checking API keys: {e}")
        return False

def check_langsmith() -> bool:
    """Check if LangSmith tracing is configured.
    
    Returns:
        bool: True if LangSmith tracing is enabled
    """
    try:
        langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
        langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1", "t", "yes", "y")
        
        if langsmith_api_key and langsmith_tracing:
            logger.info("LangSmith tracing is enabled")
            return True
        else:
            logger.info("LangSmith tracing is disabled")
            return False
    except Exception as e:
        logger.warning(f"Error checking LangSmith config: {e}")
        return False

def main():
    """Run the main application with settings from environment."""
    # Check API keys
    check_api_keys()
    
    # Check LangSmith
    check_langsmith()
    
    # Load host and port from environment
    host = get_env("HOST", "127.0.0.1")
    port = get_int_env("PORT", 8000)
    reload = get_bool_env("RELOAD", False)
    log_level = get_env("LOG_LEVEL", "info").lower()
    
    # Determine which application to run based on environment
    use_local = get_bool_env("USE_LOCAL", True)  # Default to local deployment for simplicity
    
    try:
        if use_local:
            # Use local application (no external dependencies)
            from backend.local_main import app
            logger.info("Starting local application (no external dependencies)")
        else:
            # Use standard application (with all features)
            from backend.main import app
            logger.info("Starting standard application (with all features)")
            
        # Log available config for debugging
        logger.info(f"Running with host={host}, port={port}, reload={reload}")
        
        # Start the FastAPI application with uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
        )
    except ImportError as e:
        logger.error(f"Error importing application: {e}")
        logger.error("Please make sure the required files exist")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()