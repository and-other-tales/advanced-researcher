# Security Updates

This document outlines the security updates implemented to address the 37 vulnerabilities identified by GitHub Dependabot.

## Critical Vulnerabilities Addressed

### 1. Next.js Authentication Bypass (CVE-2025-29927)

**Vulnerability**: Critical authentication bypass in Next.js 13.5.4 allowed attackers to bypass authentication mechanisms by adding a specific HTTP header.

**Fix**:
- Updated Next.js from 13.5.4 to 15.3.0
- Added middleware to explicitly block the `x-middleware-subrequest` header
- Implemented additional header filtering in custom middleware

### 2. LangChain Remote Code Execution Vulnerabilities

**Vulnerability**: Several components in LangChain were vulnerable to remote code execution through prompt injection.

**Fix**:
- Updated all LangChain packages to their latest versions
- Updated LangChain Core to v0.3.59
- Updated frontend LangChain packages

## Other Security Improvements

### 1. Security Headers

Added essential security headers in the Next.js configuration:
- `X-Content-Type-Options` - Prevents MIME type sniffing
- `X-Frame-Options` - Prevents clickjacking
- `X-XSS-Protection` - Additional layer of XSS protection
- `Referrer-Policy` - Controls referrer information

Note: Some advanced security headers like Content-Security-Policy have been temporarily disabled to ensure application functionality.

### 2. Request Filtering Middleware

Created a simplified middleware layer that:
- Blocks requests with the dangerous `x-middleware-subrequest` header
- Only applies to API routes to minimize impact on normal application functionality

### 3. Frontend Dependencies

Updated all frontend dependencies to their latest secure versions:
- React and React DOM (upgraded to v19.0.0 to ensure compatibility with Next.js 15)
- TypeScript (5.3.3)
- ESLint (8.56.0)
- PostCSS (8.4.35)
- Marked (upgraded from 7.0.2 to 9.1.5)
- Tailwind CSS (3.4.1)
- Development dependencies

### 4. Backend Dependencies

Updated Python dependencies:
- Weaviate Client from v3.26.2 to v4.14.1 (major version update)
- LangChain ecosystem packages
- Security-related packages

## Implementation Notes

### Compatibility Fixes

During implementation, we encountered several compatibility issues:

1. **React Version Compatibility**: Next.js 15.3.0 requires React 19 for proper functionality. We upgraded React from 18.2.0 to 19.0.0 to ensure compatibility.

2. **Middleware Configuration**: The initial middleware implementation was too restrictive, causing legitimate requests to be blocked. We simplified the middleware to only block the specific vulnerability pattern and limited its scope to API routes.

3. **Security Headers**: Some security headers, particularly Content-Security-Policy, were causing conflicts with the application's functionality. We implemented a more focused set of essential security headers.

### Backend Issues Identified in Logs

Server logs revealed additional potential issues that should be addressed:

1. **Protocol Buffers Conflict**: Error initializing Chroma with message: "Descriptors cannot be created directly." This suggests a compatibility issue with the Protocol Buffers version. Possible solutions:
   - Downgrade the protobuf package to 3.20.x or lower
   - Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   
2. **Pydantic Deprecation Warnings**: Multiple warnings about Pydantic V1 style validators being deprecated. These should be updated to Pydantic V2 style field validators.

3. **WebSockets Deprecation**: Warnings about websockets.legacy and WebSocketServerProtocol being deprecated.

These backend issues do not directly relate to the security vulnerabilities but should be addressed in future updates to maintain a healthy codebase.

## Ongoing Security Maintenance

To maintain the security posture of this application:

1. **Regular Updates**: Continue to update dependencies regularly
2. **Dependency Scanning**: Use GitHub Dependabot or similar tools to scan for new vulnerabilities
3. **Security Testing**: Implement security testing in CI/CD pipeline
4. **Code Reviews**: Pay special attention to code that handles user input or external data
5. **LangChain Security**: Follow LangChain security best practices, especially for components that execute code

## References

- [Next.js Security Documentation](https://nextjs.org/docs/advanced-features/security-headers)
- [Next.js 15 React Compatibility](https://nextjs.org/docs/app/building-your-application/upgrading/version-15)
- [LangChain Security Policy](https://python.langchain.com/docs/security/)
- [Weaviate Python Client v4 Migration Guide](https://weaviate.io/developers/weaviate/client-libraries/python/v3_v4_migration)
- [Protocol Buffers Python Migration](https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates)
- [Pydantic V2 Migration Guide](https://errors.pydantic.dev/2.11/migration/)