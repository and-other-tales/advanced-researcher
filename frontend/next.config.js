/** @type {import('next').NextConfig} */
const nextConfig = {
  // Note: we cannot use export and rewrites together
  distDir: 'out',
  
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
  
  async rewrites() {
    // Use environment variables to configure the backend URL or default to current origin
    // This lets the app work in Cloud Run where backend and frontend run on same port
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 
                       process.env.BACKEND_URL || 
                       (process.env.NODE_ENV === 'development' ? 'http://localhost:8081' : '');
    
    const routes = ['/api/:path*', '/chat', '/feedback', '/get_trace', '/health'];
    
    return routes.map(route => {
      const path = route.endsWith('*') ? route : route;
      const destination = backendUrl ? 
        `${backendUrl}${path}` : 
        path; // Empty backendUrl means same-origin proxy
      
      return {
        source: path,
        destination: destination,
      };
    });
  },
  
  // Server runtime config
  serverRuntimeConfig: {
    // This will be available on the server side
    blockHeaderPatterns: ['x-middleware-subrequest'],
  },
};

module.exports = nextConfig;
