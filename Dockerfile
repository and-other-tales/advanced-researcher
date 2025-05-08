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

# Install additional dependencies for new features if not already in requirements.txt
RUN pip install --no-cache-dir lxml

# Create data directory for persistent storage
RUN mkdir -p /data && chmod 777 /data

# Copy application code
COPY backend/ ./backend/
# Create static directory if not exists
RUN mkdir -p ./backend/static
# Copy frontend/out/ directory
COPY frontend/out/ ./backend/static/
COPY main.py docker-entrypoint.sh .env.example ./

# Ensure entrypoint is executable
RUN chmod +x /app/docker-entrypoint.sh

# Ensure utilities directories exist
RUN mkdir -p backend/utils
RUN touch backend/utils/__init__.py
RUN touch backend/static/__init__.py

# Default environment variables
ENV DATA_MOUNT_PATH=/data
ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Use entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
