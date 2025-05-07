# Deployment Guide with New Features

This guide covers deployment of the enhanced chat-langchain application with all new features:
- Dynamic document ingestion
- Dataset creation
- Deep research
- Auto-learning capabilities

## Prerequisites

1. A Google Cloud account with billing enabled
2. The [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
3. A Weaviate instance for the vector store
4. A Cloud SQL PostgreSQL instance for the record manager
5. API keys for search providers (optional but recommended)

## Environment Variables

The application requires several environment variables for full functionality:

### Core Variables
```
OPENAI_API_KEY=your_openai_api_key
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
RECORD_MANAGER_DB_URL=postgresql://username:password@host:port/database
```

### Search API Keys (for auto-learning and deep research)
```
TAVILY_API_KEY=your_tavily_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
EXA_API_KEY=your_exa_api_key
LINKUP_API_KEY=your_linkup_api_key
```

### Optional Alternative LLM Providers
```
ANTHROPIC_API_KEY=your_anthropic_api_key
FIREWORKS_API_KEY=your_fireworks_api_key
GOOGLE_API_KEY=your_google_api_key
COHERE_API_KEY=your_cohere_api_key
```

### Data Storage Configuration
```
DATA_MOUNT_PATH=/data
```

## Step 1: Set Up Cloud SQL PostgreSQL

```bash
# Create a Cloud SQL PostgreSQL instance
gcloud sql instances create langchain-postgres \
  --database-version=POSTGRES_13 \
  --cpu=1 \
  --memory=4GB \
  --region=us-central1

# Create a database
gcloud sql databases create langchain --instance=langchain-postgres

# Set a password for the postgres user
gcloud sql users set-password postgres \
  --instance=langchain-postgres \
  --password=YOUR_PASSWORD
```

## Step 2: Create a Weaviate Cluster

1. Sign up for a Weaviate account at [https://console.weaviate.cloud/](https://console.weaviate.cloud/)
2. Create a new cluster with sufficient resource allocation (recommended minimum: 2 CPU, 4GB RAM)
3. Make note of your cluster URL and API key

## Step 3: Configure Environment Variables

Create a copy of the example environment file:

```bash
cp .env.example .env
```

For Cloud Run deployment, create a .env.cloud.yaml file:

```yaml
OPENAI_API_KEY: "your_openai_api_key"
WEAVIATE_URL: "your_weaviate_url"
WEAVIATE_API_KEY: "your_weaviate_api_key"
RECORD_MANAGER_DB_URL: "postgresql://postgres:your_password@/langchain?host=/cloudsql/your_instance_connection_name"
DATA_MOUNT_PATH: "/data"
TAVILY_API_KEY: "your_tavily_api_key"
PERPLEXITY_API_KEY: "your_perplexity_api_key"
# Add other environment variables as needed
```

## Step 4: Build and Deploy to Cloud Run

```bash
# Set your Google Cloud project ID
PROJECT_ID=$(gcloud config get-value project)

# Build and push the Docker image
gcloud builds submit --tag gcr.io/$PROJECT_ID/chat-langchain

# Get your Cloud SQL instance connection name
INSTANCE_CONNECTION_NAME=$(gcloud sql instances describe langchain-postgres --format='value(connectionName)')

# Deploy to Cloud Run with volume mounting for /data
gcloud run deploy chat-langchain \
  --image gcr.io/$PROJECT_ID/chat-langchain \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --env-vars-file .env.cloud.yaml \
  --add-cloudsql-instances $INSTANCE_CONNECTION_NAME \
  --execution-environment gen2 \
  --cpu 1 \
  --memory 2Gi \
  --volumes name=data,path=/data
```

## Step 5: Run Initial Data Ingestion

```bash
# Create a one-time job for ingestion
gcloud run jobs create chat-langchain-ingest \
  --image gcr.io/$PROJECT_ID/chat-langchain \
  --command python \
  --args "-m,backend.local_ingest" \
  --env-vars-file .env.cloud.yaml \
  --region us-central1 \
  --add-cloudsql-instances $INSTANCE_CONNECTION_NAME \
  --execution-environment gen2 \
  --cpu 1 \
  --memory 2Gi \
  --volumes name=data,path=/data

# Execute the job
gcloud run jobs execute chat-langchain-ingest
```

## Step 6: Deploy the Frontend

The frontend can be deployed to Vercel by following these steps:

1. Fork the repository on GitHub
2. Create a new project on Vercel linked to your fork
3. Set the environment variables in Vercel:
   - `NEXT_PUBLIC_API_BASE_URL`: URL of your Cloud Run deployment

## Testing the New Features

### Auto-Learning

The auto-learning feature works transparently in the chat interface. When the system doesn't have sufficient knowledge to answer a question, it will automatically:

1. Notify the user that it's searching for more information
2. Search the web and ingest relevant information in the background
3. Store the information in the vector database for future use

To check the status of learning tasks, make a GET request to:
```
GET /api/learning/{task_id}
```

### Dataset Creation

To create a dataset from a web source:

```
POST /api/datasets
{
  "name": "UK Primary Legislation",
  "source_type": "legislation",
  "url": "https://www.legislation.gov.uk/primary+secondary",
  "description": "All UK primary and secondary legislation",
  "max_depth": 8
}
```

### Deep Research

To initiate a deep research task:

```
POST /api/research
{
  "topic": "The impact of quantum computing on cryptography",
  "max_depth": 2,
  "number_of_queries": 2,
  "search_api": "tavily"
}
```

## Troubleshooting

### Volume Mounting Issues

If you encounter issues with volume mounting in Cloud Run:

1. Ensure you're using Cloud Run Gen2 execution environment
2. Verify the volume name and path are correctly specified
3. Check that the service account has appropriate permissions

### Search API Errors

If auto-learning or deep research features fail:

1. Verify your search API keys are correctly set in the environment variables
2. Check the API quotas and limits for your search providers
3. Try using alternative search providers if one is unavailable

### Database Connection Issues

If the application can't connect to the database:

1. Verify the connection string is correctly formatted
2. Ensure the Cloud SQL instance is properly linked to your Cloud Run service
3. Check the database password and permissions

## Monitoring and Maintenance

### Check Application Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=chat-langchain" --limit 50
```

### Update the Deployment

When you make changes to the application:

```bash
# Build the new image
gcloud builds submit --tag gcr.io/$PROJECT_ID/chat-langchain

# Update the deployment
gcloud run deploy chat-langchain \
  --image gcr.io/$PROJECT_ID/chat-langchain \
  --platform managed \
  --region us-central1
```

### Database Management

Periodically check your database size and performance:

```bash
# Connect to your PostgreSQL database
gcloud sql connect langchain-postgres --user=postgres

# Check table sizes
SELECT pg_size_pretty(pg_total_relation_size('record_manager'));
```