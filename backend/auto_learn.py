"""Auto-learning module that searches for information and adds it to the vector store."""
import asyncio
import logging
import os
import re
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from langchain_core.indexing import SQLRecordManager, index
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

# Import backend modules with error handling
try:
    from backend.local_embeddings import get_embeddings_model
except ImportError as e:
    logger.warning(f"Error importing local_embeddings: {e}")
    # Define fallback function
    def get_embeddings_model():
        from langchain_openai import OpenAIEmbeddings
        try:
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception:
            # Return a dummy embeddings model if OpenAI is not available
            from langchain_core.embeddings import Embeddings
            class DummyEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    return [[0.0] * 768 for _ in texts]
                def embed_query(self, text):
                    return [0.0] * 768
            return DummyEmbeddings()

# Attempt to import Weaviate or Chroma based on environment
try:
    import weaviate
    try:
        from constants import WEAVIATE_DOCS_INDEX_NAME
    except ImportError:
        # Define a fallback if not available
        WEAVIATE_DOCS_INDEX_NAME = "LangChainDocs"
    from langchain_community.vectorstores import Weaviate
    USING_WEAVIATE = True
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        USING_WEAVIATE = False
    except ImportError:
        logger.error("Neither Weaviate nor Chroma could be imported, vector storage functionality will be limited")
        USING_WEAVIATE = False
        # Define a dummy Chroma class
        class Chroma:
            def __init__(self, *args, **kwargs):
                pass
            def as_retriever(self, *args, **kwargs):
                return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Result from a web search."""
    url: str
    title: str
    snippet: str


class SearchResponse(BaseModel):
    """Response from a web search."""
    query: str
    results: List[SearchResult]


class LearningStatus(BaseModel):
    """Status of the learning process."""
    query: str
    status: str
    documents_found: int = 0
    documents_added: int = 0
    error: Optional[str] = None


def detect_insufficient_information(query: str, response: str) -> bool:
    """Detect if the response indicates insufficient information.
    
    Args:
        query: The user's query
        response: The system's response
        
    Returns:
        True if the response indicates insufficient information
    """
    # Check for phrases that indicate insufficient information
    insufficient_indicators = [
        "I don't have information",
        "I don't have enough information",
        "I don't have specific information",
        "I don't know",
        "I'm not sure",
        "I'm unsure",
        "I'm not aware",
        "I don't have details",
        "I don't have access",
        "I can't find",
        "No information available",
        "not in my knowledge",
        "I wasn't trained on",
        "beyond my training",
        "I couldn't find",
        "I cannot provide",
        "I don't have data",
        "Hmm, I'm not sure",
        "I need more information",
        "I can't access"
    ]
    
    # Check if response contains any of the indicators
    for indicator in insufficient_indicators:
        if indicator.lower() in response.lower():
            return True
    
    # Check if the response is very short (likely non-informative)
    if len(response.split()) < 20:
        # But only if it's not a straightforward answer to a simple question
        simple_question_starters = ["who", "what", "when", "where", "how many", "is", "are", "was", "were"]
        is_simple_question = any(query.lower().startswith(starter) for starter in simple_question_starters)
        if not is_simple_question:
            return True
    
    return False


def generate_search_queries(query: str, llm: LanguageModelLike) -> List[str]:
    """Generate search queries for the given user query using an LLM.
    
    Args:
        query: The user's query
        llm: The language model to use
        
    Returns:
        List of search queries
    """
    # Create a system message to instruct the model
    system_message = SystemMessage(content="""
    Your task is to generate 2-3 effective web search queries based on the user's question.
    Each query should:
    1. Be concise and specific
    2. Use different phrasings to maximize coverage
    3. Include key entities and concepts from the original question
    4. Be written in a way that would yield informative results in a search engine
    
    Return just the search queries, one per line, with no additional explanation or formatting.
    """)
    
    # Create a human message with the query
    human_message = HumanMessage(content=f"Generate search queries for: {query}")
    
    # Generate the search queries
    try:
        response = llm.invoke([system_message, human_message])
        # Extract and clean queries from the response
        queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        return queries
    except Exception as e:
        logger.error(f"Error generating search queries: {e}")
        # Fall back to the original query if there's an error
        return [query]


async def search_web(search_queries: List[str]) -> List[SearchResult]:
    """Search the web for information using multiple search APIs.
    
    Args:
        search_queries: List of search queries
        
    Returns:
        List of search results
    """
    from backend.deep_research import get_search_tool
    from backend.utils import get_env
    
    all_results = []
    
    # Try different search APIs in order of preference
    search_apis = ["tavily", "duckduckgo", "perplexity", "exa"]
    
    for api in search_apis:
        try:
            # Import and check for API key
            if api == "tavily":
                tavily_api_key = get_env("TAVILY_API_KEY")
                if not tavily_api_key:
                    continue
                
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_api_key)
                
                for query in search_queries:
                    response = client.search(query=query, search_depth="basic")
                    for result in response.get("results", []):
                        all_results.append(SearchResult(
                            url=result.get("url", ""),
                            title=result.get("title", ""),
                            snippet=result.get("content", "")
                        ))
                        
                # If we got results, return them
                if all_results:
                    return all_results
                    
            elif api == "duckduckgo":
                from duckduckgo_search import DDGS
                
                ddgs = DDGS()
                for query in search_queries:
                    results = list(ddgs.text(query, max_results=3))
                    for result in results:
                        all_results.append(SearchResult(
                            url=result.get("href", ""),
                            title=result.get("title", ""),
                            snippet=result.get("body", "")
                        ))
                
                # If we got results, return them
                if all_results:
                    return all_results
                    
            elif api == "perplexity":
                perplexity_api_key = get_env("PERPLEXITY_API_KEY")
                if not perplexity_api_key:
                    continue
                
                import httpx
                headers = {
                    "Authorization": f"Bearer {perplexity_api_key}",
                    "Content-Type": "application/json"
                }
                
                for query in search_queries:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.perplexity.ai/search",
                            json={"query": query},
                            headers=headers
                        )
                        data = response.json()
                        
                        for result in data.get("results", []):
                            all_results.append(SearchResult(
                                url=result.get("url", ""),
                                title=result.get("title", ""),
                                snippet=result.get("extract", "")
                            ))
                
                # If we got results, return them
                if all_results:
                    return all_results
        
        except ImportError:
            continue
        except Exception as e:
            logger.error(f"Error using {api} search: {e}")
            continue
    
    # If we've tried all APIs and have no results, try a direct DuckDuckGo search
    # which doesn't require an API key
    if not all_results:
        try:
            from duckduckgo_search import DDGS
            
            ddgs = DDGS()
            for query in search_queries:
                results = list(ddgs.text(query, max_results=3))
                for result in results:
                    all_results.append(SearchResult(
                        url=result.get("href", ""),
                        title=result.get("title", ""),
                        snippet=result.get("body", "")
                    ))
        except Exception as e:
            logger.error(f"Error using fallback DuckDuckGo search: {e}")
    
    return all_results


async def fetch_and_process_urls(search_results: List[SearchResult]) -> List[Document]:
    """Fetch content from search result URLs and process them into documents.
    
    Args:
        search_results: List of search results
        
    Returns:
        List of processed documents
    """
    documents = []
    
    for result in search_results:
        try:
            # Skip if URL is empty or None
            if not result.url:
                continue
                
            # Check if the URL looks like a sitemap
            if result.url.endswith('.xml') or 'sitemap' in result.url.lower():
                # Use SitemapLoader for sitemap URLs
                loader = SitemapLoader(
                    result.url,
                    filter_urls=[result.url.split('/sitemap')[0]],
                )
                docs = loader.load()
                
            else:
                # Use WebBaseLoader for faster loading of individual pages
                loader = WebBaseLoader(
                    [result.url],
                    bs_kwargs={
                        "parse_only": None,  # Parse the entire page
                        "features": "lxml",
                    }
                )
                docs = loader.load()
            
            # Add documents to the list
            documents.extend(docs)
            
        except Exception as e:
            logger.error(f"Error fetching URL {result.url}: {e}")
            # Create a document from the snippet even if we can't fetch the full page
            doc = Document(
                page_content=f"{result.title}\n\n{result.snippet}",
                metadata={
                    "source": result.url,
                    "title": result.title,
                }
            )
            documents.append(doc)
    
    return documents


def simple_extractor(html: str) -> str:
    """Extract text content from HTML."""
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def get_vectorstore():
    """Get the appropriate vector store based on environment."""
    # Import utility for consistent environment variable handling
    from backend.utils import get_env
    
    embedding_model = get_embeddings_model()
    
    if USING_WEAVIATE:
        # Use Weaviate for cloud deployment
        WEAVIATE_URL = get_env("WEAVIATE_URL")
        WEAVIATE_API_KEY = get_env("WEAVIATE_API_KEY")
        
        if not WEAVIATE_URL or not WEAVIATE_API_KEY:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set for Weaviate vectorstore")
        
        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
        )
        
        return Weaviate(
            client=client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding_model,
            by_text=False,
            attributes=["source", "title"],
        )
    else:
        # Use Chroma for local deployment
        collection_name = get_env("COLLECTION_NAME", "langchain")
        
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
        )


def get_record_manager():
    """Get the appropriate record manager based on environment."""
    # Import utility for consistent environment variable handling
    from backend.utils import get_env
    
    # Get record manager DB URL from environment or use default
    RECORD_MANAGER_DB_URL = get_env("RECORD_MANAGER_DB_URL")
    if not RECORD_MANAGER_DB_URL:
        # Use a default PostgreSQL URL for local deployment
        DATABASE_HOST = get_env("DATABASE_HOST", "127.0.0.1")
        DATABASE_PORT = get_env("DATABASE_PORT", "5432")
        DATABASE_USERNAME = get_env("DATABASE_USERNAME", "postgres")
        DATABASE_PASSWORD = get_env("DATABASE_PASSWORD", "mysecretpassword")
        DATABASE_NAME = get_env("DATABASE_NAME", "langchain")
        RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    
    # Create record manager
    if USING_WEAVIATE:
        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
    else:
        collection_name = get_env("COLLECTION_NAME", "langchain")
        record_manager = SQLRecordManager(
            f"chroma/{collection_name}", db_url=RECORD_MANAGER_DB_URL
        )
    
    record_manager.create_schema()
    return record_manager


async def ingest_documents(documents: List[Document], llm: Optional[LanguageModelLike] = None) -> Dict:
    """Ingest documents into the vector store.
    
    Args:
        documents: List of documents to ingest
        llm: Optional language model for advanced verification
        
    Returns:
        Dict with ingestion statistics
    """
    # Import document verification functionality
    from backend.dynamic_ingest import verify_document
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    # Split documents
    docs_transformed = text_splitter.split_documents(documents)
    
    # Verify documents with advanced verification if LLM is provided
    docs_original_count = len(docs_transformed)
    verified_docs = []
    for doc in docs_transformed:
        if len(doc.page_content) > 10 and verify_document(doc, llm):
            verified_docs.append(doc)
    
    docs_transformed = verified_docs
    
    # Log verification results
    docs_filtered_count = docs_original_count - len(docs_transformed)
    if docs_filtered_count > 0:
        logging.info(f"Filtered out {docs_filtered_count} documents during verification ({docs_filtered_count / docs_original_count:.1%})")
    
    # Ensure required metadata fields
    for doc in docs_transformed:
        doc.metadata["source"] = doc.metadata.get("source", "")
        doc.metadata["title"] = doc.metadata.get("title", "")
    
    # Get vector store and record manager
    vectorstore = get_vectorstore()
    record_manager = get_record_manager()
    
    # Index documents
    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="incremental",  # Use incremental to allow adding more docs later
        source_id_key="source",
    )
    
    return {
        "num_added": indexing_stats["num_added"],
        "num_updated": indexing_stats["num_updated"],
        "num_skipped": indexing_stats["num_skipped"],
        "num_deleted": indexing_stats["num_deleted"],
    }


async def learn_from_query(query: str, llm: LanguageModelLike) -> LearningStatus:
    """Learn information related to a query by searching the web and ingesting into vector store.
    
    Args:
        query: The user's query
        llm: Language model for generating search queries
        
    Returns:
        Status of the learning process
    """
    try:
        # Generate search queries
        search_queries = generate_search_queries(query, llm)
        
        # Search the web
        search_results = await search_web(search_queries)
        
        if not search_results:
            return LearningStatus(
                query=query,
                status="failed",
                error="No search results found"
            )
        
        # Fetch and process URLs
        documents = await fetch_and_process_urls(search_results)
        
        if not documents:
            return LearningStatus(
                query=query,
                status="failed",
                documents_found=len(search_results),
                error="Failed to fetch any documents"
            )
        
        # Ingest documents with verification
        ingestion_stats = await ingest_documents(documents, llm)
        
        return LearningStatus(
            query=query,
            status="success",
            documents_found=len(documents),
            documents_added=ingestion_stats["num_added"]
        )
        
    except Exception as e:
        logger.error(f"Error learning from query: {e}")
        return LearningStatus(
            query=query,
            status="failed",
            error=str(e)
        )


# Registry for tracking learning status with persistent storage
import json
import datetime
from pathlib import Path

# Define persistent storage location
from backend.utils import get_env
DATA_MOUNT_PATH = get_env("DATA_MOUNT_PATH", "/data")
LEARNING_DIR = Path(DATA_MOUNT_PATH) / "auto_learning"

# Create directory if it doesn't exist
LEARNING_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache backed by persistent storage
# Initialize with empty dict to avoid reference before assignment
learning_registry = {}

def _get_learning_file_path(task_id: str) -> Path:
    """Get the file path for a learning task."""
    return LEARNING_DIR / f"{task_id}.json"

def _save_learning_task(task_id: str) -> None:
    """Save learning task data to disk."""
    try:
        if task_id not in learning_registry:
            logger.warning(f"Attempted to save non-existent learning task: {task_id}")
            return
            
        # Get the data to save, making a copy to avoid modifying the original
        task_data = dict(learning_registry[task_id])
        
        # Remove non-serializable objects
        if "task" in task_data:
            del task_data["task"]
        if "llm" in task_data:
            del task_data["llm"]
            
        # Save to JSON file
        file_path = _get_learning_file_path(task_id)
        with open(file_path, 'w') as f:
            json.dump(task_data, f)
            
        logger.debug(f"Saved learning task data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving learning task data for {task_id}: {e}")

def _load_learning_tasks() -> dict:
    """Load all learning task data from disk.
    
    Returns:
        Dict containing loaded task data
    """
    loaded_registry = {}
    try:
        # Find all JSON files in the learning directory
        json_files = list(LEARNING_DIR.glob("*.json"))
        
        for file_path in json_files:
            try:
                # Get task ID from filename
                task_id = file_path.stem
                
                # Load JSON data
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                
                # Add to registry
                loaded_registry[task_id] = task_data
                
            except Exception as e:
                logger.error(f"Error loading learning task data from {file_path}: {e}")
        
        logger.info(f"Loaded {len(loaded_registry)} learning tasks from disk")
        return loaded_registry
    except Exception as e:
        logger.error(f"Error loading learning task data: {e}")
        # Return empty registry on error
        return {}

# Initialize the learning registry and load existing tasks
learning_registry = _load_learning_tasks()


async def register_learning_task(query: str, llm: LanguageModelLike) -> str:
    """Register a learning task and start it in the background.
    
    Args:
        query: The user's query
        llm: Language model for generating search queries
        
    Returns:
        task_id: Unique identifier for the learning task
    """
    # Generate a unique ID for this task
    task_id = f"learn_{uuid.uuid4().hex[:8]}"
    
    # Register the task with comprehensive metadata
    learning_registry[task_id] = {
        "query": query,
        "status": "pending",
        "created_at": datetime.datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "result": None,
        "llm": llm  # Store for processing
    }
    
    # Save to persistent storage
    _save_learning_task(task_id)
    
    # Start the task in the background with proper error handling
    task = asyncio.create_task(_run_learning_task(task_id, query, llm))
    
    # Store the task reference and set up callback for completion
    learning_registry[task_id]["task"] = task
    task.add_done_callback(lambda t: _handle_task_completion(task_id, t))
    
    return task_id


def _handle_task_completion(task_id: str, task: asyncio.Task) -> None:
    """Handle the completion of a learning task, including error cases.
    
    Args:
        task_id: The unique identifier for the learning task
        task: The completed asyncio Task object
    """
    if task_id not in learning_registry:
        logger.error(f"Task completed for unknown learning task ID: {task_id}")
        return
        
    try:
        # Check if the task failed with an exception
        if task.exception():
            exc = task.exception()
            logger.error(f"Learning task {task_id} failed with error: {exc}")
            
            # Update the learning registry with the error information
            learning_registry[task_id]["status"] = "failed"
            learning_registry[task_id]["error"] = str(exc)
            learning_registry[task_id]["failed_at"] = datetime.datetime.now().isoformat()
            
            # Clean up task reference
            learning_registry[task_id]["task"] = None
            
            # Save error status
            _save_learning_task(task_id)
        else:
            # Task completed successfully (though result already handled in _run_learning_task)
            logger.info(f"Learning task {task_id} completed successfully")
            
            # Clean up task reference
            learning_registry[task_id]["task"] = None
            
            # Save final status (in case anything was missed)
            if learning_registry[task_id]["status"] == "completed":
                _save_learning_task(task_id)
    except Exception as e:
        # Handle any errors in the error handling itself
        logger.error(f"Error in task completion handler for {task_id}: {e}")
        
        # Ensure task is marked as failed
        learning_registry[task_id]["status"] = "failed"
        learning_registry[task_id]["error"] = f"Error in completion handler: {str(e)}"
        learning_registry[task_id]["task"] = None
        learning_registry[task_id]["failed_at"] = datetime.datetime.now().isoformat()
        
        # Try to save error status
        try:
            _save_learning_task(task_id)
        except Exception as save_err:
            logger.error(f"Failed to save error status for {task_id}: {save_err}")

async def _run_learning_task(task_id: str, query: str, llm: LanguageModelLike) -> None:
    """Run a learning task in the background.
    
    Args:
        task_id: Unique identifier for the learning task
        query: The user's query
        llm: Language model for generating search queries
    """
    try:
        # Update status to running with timestamp
        learning_registry[task_id]["status"] = "running"
        learning_registry[task_id]["started_at"] = datetime.datetime.now().isoformat()
        
        # Save status update
        _save_learning_task(task_id)
        
        # Run the learning process
        result = await learn_from_query(query, llm)
        
        # Update the registry with completion information
        learning_registry[task_id]["status"] = "completed"
        learning_registry[task_id]["result"] = result
        learning_registry[task_id]["completed_at"] = datetime.datetime.now().isoformat()
        
        # Save completed status
        _save_learning_task(task_id)
        
    except Exception as e:
        # Update status to failed with error information
        learning_registry[task_id]["status"] = "failed"
        learning_registry[task_id]["error"] = str(e)
        learning_registry[task_id]["failed_at"] = datetime.datetime.now().isoformat()
        logger.error(f"Error in learning task: {e}")
        
        # Save error status
        _save_learning_task(task_id)


def get_learning_status(task_id: str) -> Optional[Dict]:
    """Get the status of a learning task.
    
    Args:
        task_id: Unique identifier for the learning task
        
    Returns:
        Dict with task status information
    """
    return learning_registry.get(task_id)