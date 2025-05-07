#!/usr/bin/env python
import os
import sys
import traceback

try:
    # Set the env vars
    os.environ["USE_OLLAMA"] = "true"
    os.environ["DATA_MOUNT_PATH"] = "/tmp/data"
    
    # Make sure /tmp/data exists
    os.makedirs("/tmp/data", exist_ok=True)
    
    # Import the specific modules 
    import backend.local_main
    import backend.auto_learn_routes
    import backend.dataset_routes
    import backend.dynamic_routes
    import backend.auto_learn
    import backend.deep_research_routes
    import backend.deep_research
    import backend.local_chain
    
    print("All imports successful!")
except Exception as e:
    print(f"Error during import: {e}")
    traceback.print_exc()