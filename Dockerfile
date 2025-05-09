# Stage 1: Build frontend server
FROM node:22-alpine AS frontend-builder

WORKDIR /app

# Copy package.json and yarn.lock first for better caching
COPY frontend/package.json frontend/yarn.lock* ./

# Install dependencies
# Use yarn with frozen lockfile for more reliable builds and extended timeout
RUN yarn install --frozen-lockfile || (yarn cache clean && yarn install --network-timeout 600000)

# Copy the rest of the frontend source
COPY frontend/ ./

# Build the application
RUN yarn build

# Stage 2: Set up Python backend
FROM python:3.10-slim AS backend-builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt ./

# Install Python dependencies with retry mechanism
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt || (pip cache purge && pip install --no-cache-dir -r requirements.txt --timeout 300)

# Install additional dependencies for new features
RUN pip install --no-cache-dir lxml

# Create data directory for persistent storage
RUN mkdir -p /data && chmod 777 /data

# Copy application code
COPY backend/ ./backend/
COPY main.py ./

# Ensure utilities directories exist
RUN mkdir -p backend/utils
RUN touch backend/utils/__init__.py

# Stage 3: Final image with Nginx
FROM nginx:alpine

# Copy Nginx configuration
COPY _scripts/nginx.conf /etc/nginx/conf.d/default.conf

# Copy frontend from builder stage
COPY --from=frontend-builder /app/out /usr/share/nginx/html

# Create directory for backend
RUN mkdir -p /app

# Copy backend from builder stage
COPY --from=backend-builder /app /app
COPY docker-entrypoint.sh .env.example /app/

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Default environment variables
ENV DATA_MOUNT_PATH=/data
ENV BACKEND_HOST=localhost
ENV BACKEND_PORT=8000
ENV FRONTEND_PORT=3000
ENV NGINX_PORT=8080
ENV PYTHONUNBUFFERED=1

# Install supervisor to manage processes
RUN apk add --no-cache supervisor python3

# Copy supervisor configuration
COPY _scripts/supervisord.conf /etc/supervisord.conf

# Expose port
EXPOSE 8080

# Start supervisor as the entrypoint
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
