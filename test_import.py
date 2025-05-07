#!/usr/bin/env python

import sys

print("Python version:", sys.version)
print("Modules:")

try:
    from pathlib import Path
    print("Successfully imported Path from pathlib")
except ImportError as e:
    print("Error importing Path:", e)

try:
    import langchain
    print("Successfully imported langchain:", langchain.__version__)
except ImportError as e:
    print("Error importing langchain:", e)

try:
    import langchain_community
    print("Successfully imported langchain_community:", langchain_community.__version__)
except ImportError as e:
    print("Error importing langchain_community:", e)

try:
    import langchain_core
    print("Successfully imported langchain_core:", langchain_core.__version__)
except ImportError as e:
    print("Error importing langchain_core:", e)

try:
    from langchain_community.vectorstores import Chroma
    print("Successfully imported Chroma from langchain_community.vectorstores")
except ImportError as e:
    print("Error importing Chroma:", e)

try:
    from langchain_core.retrievers import BaseRetriever
    print("Successfully imported BaseRetriever from langchain_core.retrievers")
except ImportError as e:
    print("Error importing BaseRetriever:", e)

try:
    from backend.auto_learn_routes import router
    print("Successfully imported router from backend.auto_learn_routes")
except ImportError as e:
    print("Error importing router from backend.auto_learn_routes:", e)

try:
    import backend.auto_learn
    print("Successfully imported backend.auto_learn")
    if hasattr(backend.auto_learn, 'Path'):
        print("  Path is defined in backend.auto_learn")
    else:
        print("  Path is NOT defined in backend.auto_learn")
except ImportError as e:
    print("Error importing backend.auto_learn:", e)