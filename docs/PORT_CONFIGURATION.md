# Port Configuration

## Overview

Advanced Researcher is configured to run with the following port setup:

- **Frontend**: Runs on port 8080
- **Backend**: Runs on port 8081

This separation allows for better development workflow and more flexibility in deployment.

## How It Works

The frontend application (Next.js) includes API route forwarding that proxies all API requests to the backend server. This means:

1. Users interact with the web interface at http://localhost:8080
2. API requests made from the frontend are transparently forwarded to the backend at http://localhost:8081
3. The backend processes these requests and returns responses that get relayed back to the frontend

## Configuration Files

This setup is configured in the following files:

- **Frontend**:
  - `frontend/next.config.js`: Configures the frontend to run on port 8080 and sets up API route forwarding
  - `frontend/vercel.json`: Additional configuration for Vercel deployments
  - `frontend/app/utils/constants.tsx`: Sets the API base URL to point to port 8081

- **Backend**:
  - `backend/main.py` and `backend/local_main.py`: Both configured to run on port 8081
  - `main.py`: Root application entry point that sets the default port to 8081
  - `Dockerfile`: Sets environment variables for port 8081
  - `Procfile`: For deployment platforms that use Procfile

## Running Both Services

### Development Mode

1. Start the backend:
   ```bash
   python -m backend.main
   # or
   python -m backend.local_main
   ```

2. Start the frontend:
   ```bash
   cd frontend
   yarn dev
   ```

3. Access the application at http://localhost:8080

### Production Mode

For production deployments, we recommend running the backend and frontend on separate services, maintaining the same port configuration.

## Modifying the Port Configuration

If you need to change the port configuration:

1. Update the backend port in:
   - `backend/main.py`
   - `backend/local_main.py`
   - `main.py`
   - `Dockerfile`
   - `Procfile`

2. Update the frontend port in:
   - `frontend/next.config.js`

3. Update API references in:
   - `frontend/app/utils/constants.tsx`
   - `frontend/next.config.js` (rewrites section)
   - `frontend/vercel.json` (rewrites section)
   - Any documentation referencing the ports

## Cloud Run Deployment

When deploying to Google Cloud Run:

1. The service will use port 8080 by default, which is now correctly configured in all files.
2. The backend will automatically use the PORT environment variable provided by Cloud Run.
3. The frontend API calls will target the same origin, so no CORS issues will occur.

To prepare the frontend for Cloud Run deployment, you can build it and copy to the backend static directory:

```bash
# Build the frontend
cd frontend
yarn build
yarn export  # Creates the 'out' directory with static assets

# Copy to backend static directory
mkdir -p ../backend/static
cp -R out/* ../backend/static/
```

This ensures the frontend is served from the backend's static files when deployed to Cloud Run.