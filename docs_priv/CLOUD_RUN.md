# Deploying to Google Cloud Run

This guide walks you through deploying the Chat LangChain application to Google Cloud Run.

## Prerequisites

1. A Google Cloud account with billing enabled
2. The [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
3. A Weaviate instance for the vector store
4. A Cloud SQL PostgreSQL instance for the record manager

## Setup Instructions

### 1. Create a Cloud SQL PostgreSQL Instance

```bash
gcloud sql instances create langchain-postgres \
  --database-version=POSTGRES_13 \
  --cpu=1 \
  --memory=4GB \
  --region=us-central1
```

Then create a database:

```bash
gcloud sql databases create langchain --instance=langchain-postgres
```

And set a password for the postgres user:

```bash
gcloud sql users set-password postgres \
  --instance=langchain-postgres \
  --password=YOUR_PASSWORD
```

### 2. Create a Vector Store

Sign up for a Weaviate account at [https://console.weaviate.cloud/](https://console.weaviate.cloud/) and create a new cluster. Note down the URL and API key.

### 3. Configure Environment Variables

Create a copy of the example environment file:

```bash
cp .env.cloud.yaml .env.cloud.yaml.local
```

Edit `.env.cloud.yaml.local` and fill in all the required values:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_HOST`: Set to `/cloudsql/YOUR_INSTANCE_CONNECTION_NAME` 
  - You can get the instance connection name with: `gcloud sql instances describe langchain-postgres --format='value(connectionName)'`
- `DATABASE_PASSWORD`: The password you set for the postgres user
- `WEAVIATE_URL`: Your Weaviate cluster URL
- `WEAVIATE_API_KEY`: Your Weaviate API key

### 4. Build and Deploy to Cloud Run

First, build the Docker image and push it to Google Container Registry:

```bash
# Set your Google Cloud project ID
PROJECT_ID=$(gcloud config get-value project)

# Build and push the Docker image
gcloud builds submit --tag gcr.io/$PROJECT_ID/chat-langchain
```

Then deploy the application to Cloud Run:

```bash
gcloud run deploy chat-langchain \
  --image gcr.io/$PROJECT_ID/chat-langchain \
  --platform managed \
  --region us-central1 \
  --env-vars-file .env.cloud.yaml.local \
  --allow-unauthenticated \
  --add-cloudsql-instances YOUR_INSTANCE_CONNECTION_NAME
```

### 5. Ingest Documents

After deployment, you need to run the ingestion process to populate the vector store:

```bash
# Create a one-time job for ingestion
gcloud run jobs create chat-langchain-ingest \
  --image gcr.io/$PROJECT_ID/chat-langchain \
  --env-vars-file .env.cloud.yaml.local \
  --command python \
  --args "-m,backend.local_ingest" \
  --region us-central1 \
  --add-cloudsql-instances YOUR_INSTANCE_CONNECTION_NAME

# Execute the job
gcloud run jobs execute chat-langchain-ingest
```

### 6. Deploy the Frontend

The frontend can be deployed to Vercel by following these steps:

1. Fork the repository on GitHub
2. Create a new project on Vercel linked to your fork
3. Set the environment variables in Vercel:
   - `NEXT_PUBLIC_API_BASE_URL`: URL of your Cloud Run deployment

## Monitoring and Maintenance

### Checking Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=chat-langchain" --limit 50
```

### Updating the Deployment

When you make changes to the application, you can update your Cloud Run deployment:

```bash
# Build the new image
gcloud builds submit --tag gcr.io/$PROJECT_ID/chat-langchain

# Update the deployment
gcloud run deploy chat-langchain \
  --image gcr.io/$PROJECT_ID/chat-langchain \
  --platform managed \
  --region us-central1
```

### Scheduled Ingestion

To set up scheduled ingestion (e.g., daily):

```bash
# Create a Cloud Scheduler job
gcloud scheduler jobs create http update-chat-langchain-index \
  --schedule="0 0 * * *" \
  --uri="https://$(gcloud run jobs execute chat-langchain-ingest --format='value(metadata.name)')" \
  --http-method=POST \
  --location=us-central1 \
  --oidc-service-account-email=YOUR_SERVICE_ACCOUNT_EMAIL
```

## Troubleshooting

- **Connection to Cloud SQL fails**: Make sure you've properly set up the connection name and added the Cloud SQL instance to your deployment.
- **API key errors**: Verify all API keys are correctly configured in the environment variables.
- **Container crashes**: Check the logs for error messages and ensure all dependencies are properly installed.
- **Ingestion fails**: Verify the vector store is properly configured and accessible from Cloud Run.