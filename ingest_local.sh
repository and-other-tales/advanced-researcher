#!/bin/bash
set -e

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please create a .env file by copying .env.example and filling in the values"
    exit 1
fi

# Source environment variables
source .env

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set in .env file"
    exit 1
fi

# Run the ingestion script
echo "Starting ingestion process for local deployment"
python -m backend.local_ingest