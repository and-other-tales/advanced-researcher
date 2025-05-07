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
COPY frontend/app/utils/constants.tsx ./frontend/app/utils/constants.tsx
COPY main.py .env.example ./

# Ensure backend/utils directory exists
RUN mkdir -p backend/utils

# Default environment variables
ENV DATA_MOUNT_PATH=/data
ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV USE_LOCAL=true

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["python", "main.py"]