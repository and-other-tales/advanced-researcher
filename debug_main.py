#!/usr/bin/env python
import os
import sys

# Set the env vars
os.environ["USE_OLLAMA"] = "true"
os.environ["DATA_MOUNT_PATH"] = "/tmp/data"

# Make sure /tmp/data exists
os.makedirs("/tmp/data", exist_ok=True)

try:
    from backend.local_main import app
    print("Successfully imported backend.local_main.app")
except Exception as e:
    import traceback
    print(f"Error importing app: {e}")
    traceback.print_exc()