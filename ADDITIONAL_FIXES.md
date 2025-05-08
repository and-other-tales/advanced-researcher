# Additional Fixes Required

Based on the server logs, the following additional issues should be addressed in a future update.

## Protocol Buffers Compatibility Issue

**Issue**: Error initializing Chroma with message "Descriptors cannot be created directly."

```
Error initializing Chroma: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
```

**Recommended Solutions**:

1. Add the following to the Docker environment:
   ```
   ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   ```

2. Or downgrade the protobuf package:
   ```
   pip install protobuf==3.20.3
   ```

## Pydantic Deprecation Warnings

**Issue**: Multiple Pydantic V1 style validator warnings in fireworks library

```
/usr/local/lib/python3.10/site-packages/fireworks/client/image_api.py:131: PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated.
```

**Recommended Solution**:

These warnings are coming from third-party libraries, and may require:

1. Updating the fireworks library if an update is available
2. Creating a fork of the library with updated validators
3. Temporarily suppressing the warnings:
   ```python
   import warnings
   warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
   ```

## WebSockets Deprecation Warnings

**Issue**: Deprecation warnings from websockets library

```
/usr/local/lib/python3.10/site-packages/websockets/legacy/__init__.py:6: DeprecationWarning: websockets.legacy is deprecated
/usr/local/lib/python3.10/site-packages/uvicorn/protocols/websockets/websockets_impl.py:17: DeprecationWarning: websockets.server.WebSocketServerProtocol is deprecated
```

**Recommended Solution**:

These warnings are from uvicorn's use of deprecated websockets features. Solutions include:

1. Wait for uvicorn to update their websockets implementation
2. Pin the uvicorn version until the issue is resolved
3. Suppress these specific warnings as they don't affect functionality

## User Agent Warning

**Issue**: LangChain community warning about missing user agent

```
2025-05-08 06:04:44,977 - langchain_community.utils.user_agent - WARNING - USER_AGENT environment variable not set, consider setting it to identify your requests.
```

**Recommended Solution**:

Set a USER_AGENT environment variable in the Docker container:

```
ENV USER_AGENT="AdvancedResearcher/1.0"
```

## Long-term Maintenance Recommendations

1. **Dependency Version Pinning**: Consider using more specific version pins in requirements.txt and pyproject.toml to prevent unexpected updates.

2. **Testing Environment**: Create a staging environment that closely mirrors production to test updates before deployment.

3. **Monitoring**: Add centralized logging to capture warnings and errors for easier troubleshooting.

4. **Dependency Review**: Regularly review dependencies for updates, especially those with security fixes, and implement a structured update process.

5. **Library Consolidation**: Consider reducing the number of different libraries used for similar functions to reduce dependency complexity.