"""Environment variable utilities for the Advanced Researcher project."""
import os
import logging
from typing import Any, Optional, Dict, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Dictionary to store environment variables
_env_vars: Dict[str, Any] = {}
_loaded = False

def load_env_file(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file if it exists.
    
    Args:
        env_path: Optional path to .env file. If not provided,
                 looks for .env in current directory and parent directories.
    """
    global _loaded
    
    if _loaded:
        return
    
    if env_path is None:
        # Look for .env in current directory and parent directories
        current_dir = Path.cwd()
        possible_paths = [current_dir / ".env"]
        
        # Check parent directories
        parent = current_dir.parent
        while parent != parent.parent:  # Stop at root
            possible_paths.append(parent / ".env")
            parent = parent.parent
        
        # Try each path
        for path in possible_paths:
            if path.exists():
                env_path = str(path)
                break
    
    if env_path and os.path.exists(env_path):
        logger.info(f"Loading environment variables from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Don't override existing environment variables
                    if key not in os.environ:
                        os.environ[key] = value
    
    _loaded = True


def get_env(key: str, default: Any = None) -> Any:
    """Get environment variable from OS environment or cached dictionary.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Value from environment or default
    """
    # Ensure .env file is loaded
    if not _loaded:
        load_env_file()
    
    # Always get directly from OS environment first (highest priority)
    value = os.environ.get(key, default)
    
    # Update cache with current value
    _env_vars[key] = value
    
    return value


def set_env(key: str, value: Any) -> None:
    """Set environment variable in both cache and OS environment.
    
    Args:
        key: Environment variable key
        value: Value to set
    """
    _env_vars[key] = value
    os.environ[key] = str(value)


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Boolean value from environment
    """
    value = get_env(key, default)
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ("true", "1", "t", "yes", "y")
    
    return bool(value)


def get_int_env(key: str, default: int = 0) -> int:
    """Get integer environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Integer value from environment
    """
    value = get_env(key, default)
    if isinstance(value, int):
        return value
    
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def get_float_env(key: str, default: float = 0.0) -> float:
    """Get float environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Float value from environment
    """
    value = get_env(key, default)
    if isinstance(value, float):
        return value
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def get_list_env(key: str, default: Optional[list] = None, separator: str = ",") -> list:
    """Get list environment variable by splitting a string.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        separator: Separator character to split string
        
    Returns:
        List value from environment
    """
    default = default or []
    value = get_env(key, None)
    
    if value is None:
        return default
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    return default


# Initialize by loading .env file
load_env_file()