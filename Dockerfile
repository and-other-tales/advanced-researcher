FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==1.6.1 \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev \
    && pip uninstall -y poetry

# Install additional dependencies for new features
RUN pip install --no-cache-dir duckduckgo_search tavily-python beautifulsoup4 lxml

# Create data directory for persistent storage
RUN mkdir -p /data && chmod 777 /data

# Copy application code
COPY backend ./backend
COPY frontend/app/utils/constants.tsx ./frontend/app/utils/constants.tsx

# Default environment variables
ENV DATA_MOUNT_PATH=/data

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["python", "-m", "backend.local_main"]