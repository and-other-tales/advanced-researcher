# Advanced Researcher Docker Setup

This document covers the Docker setup for the Advanced Researcher project.

## Docker Configuration

The project uses Docker to containerize both the frontend and backend for easy deployment.

### Complete Solution Architecture

The Dockerfile uses a multi-stage build:

1. First stage uses Node 22 to prepare the frontend
2. Second stage builds the Python backend and includes the frontend files

```dockerfile
# Stage 1: Build frontend server
FROM node:22-alpine AS frontend-builder

WORKDIR /app

# Install serve - a static server for Next.js apps
RUN npm install -g serve

# Copy pre-built frontend files
COPY frontend/out /app/frontend

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
# Copy pre-built frontend files from frontend-builder stage
COPY --from=frontend-builder /app/frontend ./backend/static/
COPY main.py docker-entrypoint.sh .env.example ./

# Expose port
EXPOSE 8080

# Use entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
```

### Key improvements:

1. **Combined Solution**: Single container running both backend API and frontend interface
2. **Multi-stage Build**: Uses Node 22 to prepare frontend assets, then includes them in the Python backend
3. **Standardized Port**: Serves everything on port 8080
4. **Optimized Size**: Uses pre-built Next.js assets instead of building in the container
5. **Proper Asset Serving**: Backend handles serving the static frontend assets at the root path

## Build and Run Instructions

### Building the image

```bash
docker build -t advanced-researcher:latest .
```

### Running the container

```bash
# Run with environment variables (preferred method)
docker run -d -p 8080:8080 \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e WEAVIATE_URL=your-weaviate-url \
  -e WEAVIATE_API_KEY=your-weaviate-api-key \
  -e USE_LOCAL=true \
  --name advanced-researcher advanced-researcher:latest
```

The container is designed to prioritize OS environment variables over any .env files. 
The entrypoint script checks for the presence of API keys in the environment and uses those directly.
Only if no environment variables are found will it attempt to use the .env file as a fallback.

### Accessing the application

The application will be available at:
- Web UI: http://localhost:8080
- API docs: http://localhost:8080/docs
- API root: http://localhost:8080/api

## Modifications Made

1. **Weaviate Client Updates**: Updated all Weaviate client code to use v4 API instead of v3:
   - Changed `weaviate.Client()` to `weaviate.WeaviateClient()`
   - Added proper imports: `from weaviate.connect import ConnectionParams`
   - Updated connection parameters and authentication

2. **Pydantic V2 Migration**: Updated Pydantic models to use V2 syntax:
   - Added `model_config` dictionaries
   - Changed `default=[]` to `default_factory=list` for all List fields
   - Added field validators using `@field_validator` decorator
   - Updated serialization methods from `.dict()` to `.model_dump()`
   - Added type checking and validation

3. **Import Path Corrections**: Fixed relative imports for better module resolution:
   - Updated `from parser import langchain_docs_extractor` to `from backend.parser import langchain_docs_extractor`
   - Updated `from constants import WEAVIATE_DOCS_INDEX_NAME` to `from backend.constants import WEAVIATE_DOCS_INDEX_NAME`
   - Updated `from ingest import get_embeddings_model` to `from backend.ingest import get_embeddings_model`

4. **Multi-stage Docker Build**: Created a streamlined container with both frontend and backend:
   - Stage 1: Node 22 for preparing frontend assets
   - Stage 2: Python 3.10 for the main backend application
   - Combined solution serves the API and frontend on port 8080

## HTML to Markdown Processing for Weaviate Storage

For all crawl, scrape, and storage operations in Weaviate, it's important to process HTML content into Markdown format. This improves readability and reduces noise in the stored data. We use the `jinaai/ReaderLM-v2` model for this purpose.

### Installation

Add these dependencies to your requirements.txt or install them directly:

```bash
pip install transformers
```

### Implementation

Add the following code to your ingest pipeline:

```python
# 1. Import necessary libraries
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# 2. Load the model
device = "cpu"  # or "cuda" if available
tokenizer = AutoTokenizer.from_pretrained("jinaai/ReaderLM-v2")
model = AutoModelForCausalLM.from_pretrained("jinaai/ReaderLM-v2").to(device)

# 3. HTML cleaning patterns
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )

def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(
        SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html

# 4. Prompt creation
def create_prompt(
    text: str, tokenizer=None, instruction: str = None, schema: str = None
) -> str:
    """
    Create a prompt for the model with optional instruction and JSON schema.
    """
    if not instruction:
        instruction = "Extract the main content from the given HTML and convert it to Markdown format."
    if schema:
        instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
        prompt = f"{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
    else:
        prompt = f"{instruction}\n```html\n{text}\n```"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# 5. HTML to Markdown conversion function
def html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown using the ReaderLM model.
    
    Args:
        html: The HTML string to convert
        
    Returns:
        Markdown formatted content
    """
    html = clean_html(html, clean_svg=True, clean_base64=True)
    input_prompt = create_prompt(html, tokenizer=tokenizer)
    
    inputs = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, max_new_tokens=1024, temperature=0, do_sample=False, repetition_penalty=1.08
    )
    
    markdown_result = tokenizer.decode(outputs[0])
    
    # Extract just the generated markdown from the response
    # This may need adjustment based on actual output format
    try:
        markdown = markdown_result.split("```markdown")[1].split("```")[0].strip()
    except IndexError:
        # Fallback if markdown isn't properly formatted in the output
        markdown = markdown_result.split("ASSISTANT:")[1].strip()
    
    return markdown
```

### Integration with Ingestion Pipeline

Add this step to your document processing flow before storing in Weaviate:

```python
# Example integration in document processing pipeline
def process_documents(docs_with_html):
    processed_docs = []
    
    for doc in docs_with_html:
        # Convert HTML content to Markdown
        if doc.page_content and "<" in doc.page_content and ">" in doc.page_content:
            doc.page_content = html_to_markdown(doc.page_content)
            
        processed_docs.append(doc)
    
    return processed_docs

# Then use the processed documents for indexing
processed_docs = process_documents(raw_documents)
vectorstore.add_documents(processed_docs)
```

## Troubleshooting

If port 8080 is already in use, you can either:
1. Stop the process using that port: `kill -9 $(lsof -t -i:8080)`
2. Use a different port mapping: `docker run -d -p 8081:8080 --name advanced-researcher advanced-researcher:latest`