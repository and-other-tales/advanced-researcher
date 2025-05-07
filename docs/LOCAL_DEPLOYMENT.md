# Local Deployment Guide

This guide will help you set up and run Advanced Researcher locally on your machine.

## Requirements

To run locally, you'll need:

1. **Docker** - For running Chroma (vector store) and PostgreSQL (record manager)
2. **Ollama** (optional) - For local LLM inference instead of OpenAI
3. **Python 3.10+** - For running the backend
4. **Node.js & Yarn** - For running the frontend

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/and-other-tales/advanced-researcher.git
cd advanced-researcher
```

### 2. Set Up Docker Containers

#### PostgreSQL

```bash
docker pull postgres
docker run --name postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d postgres
```

Then create a database for the record manager:

```bash
docker exec -it postgres createdb -U postgres langchain
```

#### Chroma

```bash
docker pull chromadb/chroma
docker run -p 8000:8000 -d chromadb/chroma
```

### 3. Environment Variables

Create a `.env` file in the root directory by copying the example:

```bash
cp .env.example .env
```

Then edit the file to add your OpenAI API key and other configuration:

```
OPENAI_API_KEY=your_openai_api_key
COLLECTION_NAME=langchain
DATABASE_HOST=127.0.0.1
DATABASE_PORT=5432
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=mysecretpassword
DATABASE_NAME=langchain
```

### 4. Install Backend Dependencies

```bash
pip install -e .
# OR if using poetry
poetry install
```

### 5. Install Frontend Dependencies

```bash
cd frontend
yarn install
```

### 6. Run the Backend Server

```bash
cd /path/to/advanced-researcher
python -m backend.main
```

The server will start on http://localhost:8080

### 7. Run the Frontend

```bash
cd frontend
yarn dev
```

The frontend will be available at http://localhost:3000

## Using Ollama for Local LLM Inference (Optional)

If you want to use Ollama for local inference instead of OpenAI:

1. Install Ollama from [https://ollama.com/download](https://ollama.com/download)

2. Pull the required models:
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

3. Update your `.env` file to use Ollama:
   ```
   USE_OLLAMA=true
   OLLAMA_BASE_URL=http://localhost:11434
   ```

4. Restart your backend server

## Ingesting Documents

To ingest documents into your local vector store:

```bash
python -m backend.ingest --dir /path/to/your/documents
```

## Troubleshooting

- If you encounter Docker networking issues, make sure ports 5432 (PostgreSQL) and 8000 (Chroma) are not being used by other applications.
- If the backend fails to start, check your `.env` file for proper configuration.
- For any LangChain-specific errors, check the logs for detailed error messages.

## Additional Resources

- [Chroma Documentation](https://docs.trychroma.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Ollama Documentation](https://ollama.com/docs)