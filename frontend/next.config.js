/** @type {import('next').NextConfig} */
const nextConfig = {
  // Note: we cannot use export and rewrites together
  distDir: 'out',
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
};

module.exports = nextConfig;
