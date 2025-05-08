/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export
  output: 'export',
  distDir: 'out',
  
  // Required for static export
  images: {
    unoptimized: true,
  },
  
  // Simplified security headers to avoid potential conflicts
  async headers() {
    return [
      {
        // Apply these headers to all routes
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'same-origin',
          }
        ],
      },
    ];
  },
  
  // Note: rewrites and static export (output: 'export') cannot be used together
  // This configuration will be ignored when building for export
  async rewrites() {
    // Only used in development mode or when not exporting
    if (process.env.NEXT_PHASE === 'phase-production-build' && process.env.NEXT_EXPORT === 'true') {
      return [];
    }
    
    // Use environment variables to configure the backend URL or default to current origin
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 
                       process.env.BACKEND_URL || 
                       (process.env.NODE_ENV === 'development' ? 'http://localhost:8080' : '');
    
    const routes = ['/api/:path*', '/chat', '/feedback', '/get_trace', '/health'];
    
    return routes.map(route => {
      const path = route.endsWith('*') ? route : route;
      const destination = backendUrl ? 
        `${backendUrl}${path}` : 
        path;
      
      return {
        source: path,
        destination: destination,
      };
    });
  },
  
  // Server runtime config - only used when not exporting
  serverRuntimeConfig: {
    blockHeaderPatterns: ['x-middleware-subrequest'],
  },
};

module.exports = nextConfig;
