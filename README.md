# OtherTales Advanced Researcher

OtherTales Advanced Researcher is an example of how to build an advanced research and knowledge management system using [LangChain](https://github.com/langchain-ai/langchain). View the project repository at [https://github.com/and-other-tales/advanced-researcher](https://github.com/and-other-tales/advanced-researcher).

## Quick Start

This project supports both local deployment and cloud-based deployment.

### For Local Deployment:

1. Clone the repository:
   ```bash
   git clone https://github.com/and-other-tales/advanced-researcher.git
   cd advanced-researcher
   ```

2. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to fill in your API keys and configuration. At minimum, you need one of the following:
   - `OPENAI_API_KEY` for OpenAI models (GPT and embeddings)
   - `ANTHROPIC_API_KEY` for Anthropic Claude models
   - `GOOGLE_API_KEY` for Google Gemini models
   - `FIREWORKS_API_KEY` for Fireworks models
   - Or set `USE_OLLAMA=true` for local models via Ollama
   
   See [Environment Variables](#environment-variables) section for details on all available configuration options.

3. Set up Docker containers for Chroma and PostgreSQL (see [LOCAL_DEPLOYMENT.md](docs/LOCAL_DEPLOYMENT.md) for details).

4. Run the ingestion script to populate the vector store:
   ```bash
   ./ingest_local.sh
   ```

5. Start the backend server:
   ```bash
   ./run_local.sh
   ```

6. In another terminal, run the frontend:
   ```bash
   cd frontend
   yarn install
   yarn dev
   ```

7. Open http://localhost:8080 in your browser.

For more detailed instructions, see [LOCAL_DEPLOYMENT.md](docs/LOCAL_DEPLOYMENT.md).

### For Cloud Deployment:

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions on deploying to Vercel + GCP Cloud Run.

## Features

### Core Capabilities
- **Advanced conversational AI**: Answer questions based on your knowledge bases with contextual understanding.
- **Memory**: The system remembers the conversation history for coherent, contextual interactions.
- **Tracing**: Integration with [LangSmith](https://smith.langchain.com/) for debugging and monitoring.
- **Multi-modal support**: Answer questions about images and PDFs with comprehensive understanding.
- **Model flexibility**: Use different models like OpenAI, Anthropic Claude, or run models locally with Ollama.

### Advanced Research Features
- **Deep Research**: Conduct comprehensive, well-structured research on complex topics by:
  - Automatically planning research approaches for complex queries
  - Performing iterative searches with multiple search providers (Tavily, Perplexity, Exa, etc.)
  - Generating structured reports with proper citations and organization
  - Integrating findings with custom knowledge bases
  - Automatically detecting when deep research is needed vs. standard RAG

### Knowledge Expansion
- **Dynamic Document Ingestion**: Create custom knowledge bases on demand by:
  - Crawling and extracting content from websites and sitemaps
  - Special handling for GOV.UK and legislative content
  - Organizing content into searchable knowledge bases
  - Using custom knowledge bases alongside the default knowledge base

- **Auto-Learning**: Self-improve knowledge over time by:
  - Detecting when the system lacks information on a topic
  - Automatically searching the web for relevant information
  - Processing and storing new knowledge for future use
  - Transparent notification when learning new information

### Dataset Creation
- **Web-to-Dataset**: Generate Hugging Face-compatible datasets from web sources:
  - Create datasets from government websites, legislation sources, and general web content
  - Automatically split data into train/validation/test sets
  - Generate appropriate metadata and documentation
  - Format data for different tasks (text, document, question-answering)

## Documentation

- [LOCAL_DEPLOYMENT.md](docs/LOCAL_DEPLOYMENT.md): Detailed instructions for local deployment
- [DEPLOYMENT.md](docs/DEPLOYMENT.md): Detailed instructions for cloud deployment
- [PORT_CONFIGURATION.md](docs/PORT_CONFIGURATION.md): How the ports are configured for frontend and backend
- [CONCEPTS.md](docs/CONCEPTS.md): Core concepts used in this project
- [MODIFY.md](docs/MODIFY.md): Guide for modifying and extending this system
- [LANGSMITH.md](docs/LANGSMITH.md): How to use LangSmith for tracing and debugging
- [DYNAMIC_INGEST.md](docs/DYNAMIC_INGEST.md): How to use dynamic document ingestion to create custom knowledge bases
- [DATASET_CREATION.md](docs/DATASET_CREATION.md): How to create Hugging Face datasets from web sources
- [DEEP_RESEARCH.md](docs/DEEP_RESEARCH.md): How to use deep research for comprehensive reports
- [AUTO_LEARNING.md](docs/AUTO_LEARNING.md): How the system automatically learns from web sources

## Project Structure

- `backend/`: Python backend code
  - `local_main.py`: Entry point for local deployment
  - `local_chain.py`: Chain implementation for local deployment
  - `local_ingest.py`: Ingestion script for local deployment
  - `local_embeddings.py`: Embeddings configuration for local deployment
  - `dynamic_ingest.py`: Dynamic document ingestion from web sources
  - `dynamic_chain.py`: Chain implementation for multiple knowledge bases
  - `dynamic_routes.py`: API routes for managing knowledge bases
  - `dataset_creator.py`: Dataset creation from web sources
  - `dataset_routes.py`: API routes for dataset management
  - `deep_research.py`: Deep research implementation using open_deep_research
  - `deep_research_routes.py`: API routes for deep research functionality
  - `auto_learn.py`: Automatic learning implementation for missing information
  - `auto_learn_routes.py`: API routes for learning status tracking
- `frontend/`: Next.js frontend
- `_scripts/`: Evaluation scripts
- `test_dynamic_ingest.py`: Test script for dynamic document ingestion
- `test_dataset_creation.py`: Test script for dataset creation

## API Examples

### Deep Research
```bash
curl -X POST http://localhost:8081/api/research \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The impact of quantum computing on cryptography",
    "max_depth": 2,
    "number_of_queries": 2,
    "search_api": "tavily"
  }'
```

### Create Knowledge Base
```bash
curl -X POST http://localhost:8081/api/knowledge_bases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "UK Tax Guidance",
    "source_type": "gov_uk",
    "url": "https://www.gov.uk/hmrc/internal-manuals",
    "max_depth": 8
  }'
```

### Create Dataset
```bash
curl -X POST http://localhost:8081/api/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "UK Primary Legislation",
    "source_type": "legislation",
    "url": "https://www.legislation.gov.uk/primary+secondary",
    "description": "All UK primary and secondary legislation",
    "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1}
  }'
```

## Environment Variables

The application uses a comprehensive set of environment variables to configure its behavior. You can set these in a `.env` file or directly in your environment.

### LLM API Keys

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models and embeddings | No* | None |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | No* | None |
| `GOOGLE_API_KEY` | Google API key for Gemini models | No* | None |
| `FIREWORKS_API_KEY` | Fireworks API key for models | No* | None |
| `COHERE_API_KEY` | Cohere API key for embeddings | No | None |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | No | None |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | No | "false" |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | No | "https://api.smith.langchain.com" |
| `LANGSMITH_PROJECT` | LangSmith project name for traces | No | None |

*At least one of these is required for full functionality, or you must use Ollama locally.

### Search API Keys (for Deep Research and Auto-Learning)

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `TAVILY_API_KEY` | Tavily search API key | No | None |
| `PERPLEXITY_API_KEY` | Perplexity API key | No | None |

### Vector Database Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `WEAVIATE_URL` | URL for Weaviate vector database | No | None |
| `WEAVIATE_API_KEY` | Authentication key for Weaviate | No | None |
| `WEAVIATE_DOCS_INDEX_NAME` | Name of the Weaviate index | No | "LangChain_Combined_Docs_OpenAI_text_embedding_3_small" |
| `COLLECTION_NAME` | Name of the Chroma collection | No | "langchain" |

### PostgreSQL Database Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `RECORD_MANAGER_DB_URL` | Complete PostgreSQL connection string | No | Constructed from individual settings |
| `DATABASE_HOST` | PostgreSQL host | No | "127.0.0.1" |
| `DATABASE_PORT` | PostgreSQL port | No | "5432" |
| `DATABASE_USERNAME` | PostgreSQL username | No | "postgres" |
| `DATABASE_PASSWORD` | PostgreSQL password | No | "mysecretpassword" |
| `DATABASE_NAME` | PostgreSQL database name | No | "langchain" |

### Local Model Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `USE_OLLAMA` | Use Ollama for local models | No | "false" |
| `OLLAMA_BASE_URL` | URL for Ollama service | No | "http://localhost:11434" |

### Deployment Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATA_MOUNT_PATH` | Path for persistent data storage | No | "/data" |
| `FORCE_UPDATE` | Force update during document ingestion | No | "false" |
| `HOST` | Host to bind the server to | No | "127.0.0.1" |
| `PORT` | Port to bind the server to | No | 8081 |
| `RELOAD` | Enable auto-reload for development | No | "false" |
| `LOG_LEVEL` | Logging level (info, debug, warning, error) | No | "info" |
| `USE_LOCAL` | Use local application with minimal dependencies | No | "false" |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.