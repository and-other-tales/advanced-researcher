#!/bin/bash
set -e

echo "===== Advanced Researcher Build and Deploy Script ====="
echo "This script will build the frontend and prepare it for deployment"

# Check for required dependencies
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required but not installed. Aborting."; exit 1; }

# Build the frontend
echo "===== Building frontend ====="
cd frontend
echo "Installing dependencies..."
npm install
echo "Building Next.js app with static export..."
npm run build
echo "Frontend build complete!"

# Verify the export was successful
if [ ! -d "out" ]; then
    echo "Error: Frontend build failed - 'out' directory not found."
    echo "This could be because:"
    echo "1. Your next.config.js is not configured with output: 'export'"
    echo "2. The Next.js build process had errors"
    echo "Trying to fix by creating out directory if needed..."
    mkdir -p out
    # If out is empty, this is still a problem
    if [ -z "$(ls -A out)" ]; then
        echo "Warning: out directory is empty, build may have failed."
        echo "Continuing anyway, but the static site may not work correctly."
    fi
fi

# Copy static files to backend
echo "===== Copying static files to backend ====="
mkdir -p ../backend/static
echo "Copying frontend build to backend/static..."
cp -R out/* ../backend/static/

# Create empty __init__.py to ensure the static directory is included in Python packages
touch ../backend/static/__init__.py

# Create utils directory if it doesn't exist
mkdir -p ../backend/utils
touch ../backend/utils/__init__.py

# Prepare for Docker if needed
if [ "$1" == "--docker" ]; then
    echo "===== Preparing Docker build ====="
    if [ ! -f "../.env" ] && [ -f "../.env.example" ]; then
        echo "Creating .env from .env.example for Docker build"
        cp ../.env.example ../.env
    fi
    
    cd ..
    docker build -t advanced-researcher .
    echo "Docker image built successfully. Run with:"
    echo "  docker run -p 8080:8080 -e OPENAI_API_KEY=your-key advanced-researcher"
    exit 0
fi

echo "===== Build and deploy preparation complete! ====="
echo "You can now run the application with the backend server"
echo "  python main.py"
echo "The frontend will be served directly from the backend at http://localhost:8080"
echo
echo "To build and run with Docker, use:"
echo "  ./build_and_deploy.sh --docker"
