/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export
  output: 'export',
  distDir: 'out',
  
  // Required for static export
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
