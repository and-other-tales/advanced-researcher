# Stage 1: Build frontend server
FROM node:22-alpine AS frontend-builder

WORKDIR /app

# Copy package.json and yarn.lock first for better caching
COPY frontend/package.json frontend/yarn.lock ./
RUN yarn install --frozen-lockfile --ignore-optional || (yarn cache clean && yarn install --ignore-optional --network-timeout 600000)

# Copy frontend source code
COPY frontend/ ./

# Build frontend (Next.js 15.3.0+ includes export in build)
RUN yarn build

# Stage 2: Set up Python backend with frontend files
FROM python:3.10-slim

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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for new features
RUN pip install --no-cache-dir lxml

# Create data directory for persistent storage
RUN mkdir -p /data && chmod 777 /data

# Copy application code
COPY backend/ ./backend/
# Create static directory if not exists
RUN mkdir -p ./backend/static
# Copy built frontend files from frontend-builder stage
COPY --from=frontend-builder /app/out ./backend/static/
COPY main.py docker-entrypoint.sh .env.example ./

# Expose port
EXPOSE 8080

# Use entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]