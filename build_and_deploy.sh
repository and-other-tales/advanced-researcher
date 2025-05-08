#!/bin/bash
set -e

echo "===== Advanced Researcher Build and Deploy Script ====="
echo "This script will build the frontend and prepare it for deployment"

# Check for required dependencies
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting."; exit 1; }
command -v yarn >/dev/null 2>&1 || { echo "Yarn is required but not installed. Aborting."; exit 1; }

# Build the frontend
echo "===== Building frontend ====="
cd frontend
echo "Installing dependencies..."
yarn install
echo "Building Next.js app..."
yarn build
echo "Frontend build complete!"

# Copy static files to backend
echo "===== Copying static files to backend ====="
mkdir -p ../backend/static
echo "Copying frontend build to backend/static..."
cp -R out/* ../backend/static/

# Create empty __init__.py to ensure the static directory is included in Python packages
touch ../backend/static/__init__.py

echo "===== Build and deploy preparation complete! ====="
echo "You can now run the application with the backend server"
echo "  python -m main"
echo "The frontend will be served directly from the backend at http://localhost:8080"