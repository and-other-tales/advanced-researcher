"""Dynamic document ingestion from web sources."""
import logging
import os
import re
import uuid
from typing import Dict, List, Optional, Union, Tuple, Any

from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain_community.indexes import SQLRecordManager, index
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Chroma, Weaviate
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from pydantic import BaseModel

from backend.local_embeddings import get_embeddings_model
from backend.verification import get_verification_components

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SiteConfig(BaseModel):
    """Configuration for a web scraping target."""
    name: str
    base_url: str
    sitemap_url: Optional[str] = None
    max_depth: int = 8
    prevent_outside: bool = True
    exclude_dirs: List[str] = []


class IngestRequest(BaseModel):
    """Request to ingest documents from a web source."""
    source_type: str  # "sitemap", "recursive_url", "gov_uk"
    name: str  # User-provided name for the knowledge base
    url: str
    user_id: Optional[str] = None  # For multi-user support
    max_depth: int = 8  # For recursive loaders
    collection_id: Optional[str] = None  # If None, a new one will be generated


class IngestResponse(BaseModel):
    """Response with details about the ingestion process."""
    collection_id: str
    document_count: int
    status: str
    message: str


# Predefined site configurations
GOV_UK_HMRC_CONFIG = SiteConfig(
    name="HMRC Tax Guidance",
    base_url="https://www.gov.uk/hmrc/internal-manuals",
    max_depth=10,
    prevent_outside=True
)


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    """Extract metadata from the BeautifulSoup object."""
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def gov_uk_extractor(html: str) -> str:
    """Custom extractor for GOV.UK pages."""
    soup = BeautifulSoup(html, "lxml")
    
    # Remove navigation, footer, and other non-content elements
    for element in soup.select('nav, footer, aside, .gem-c-contextual-sidebar, .govuk-breadcrumbs, .govuk-header'):
        if element:
            element.decompose()
            
    # Extract the main content
    main_content = soup.select_one('.govuk-main-wrapper, article.manual-body, .hmrc-manual-body, .hmrc-manual-section-body')
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text
    
    # Fallback to simple extraction if specific elements aren't found
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def simple_extractor(html: str) -> str:
    """Simple HTML extraction."""
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def get_collection_name(name: str, user_id: Optional[str] = None, collection_id: Optional[str] = None) -> str:
    """Generate a collection name for the vectorstore."""
    if collection_id:
        return f"{collection_id}"
    
    # Clean the name to make it URL and filesystem friendly
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()
    
    # Add user_id if available
    if user_id:
        return f"{user_id}_{clean_name}_{uuid.uuid4().hex[:8]}"
    
    return f"{clean_name}_{uuid.uuid4().hex[:8]}"


def load_from_sitemap(url: str, filter_urls: Optional[List[str]] = None):
    """Load documents from a sitemap URL."""
    return SitemapLoader(
        url,
        filter_urls=filter_urls if filter_urls else [url.split('/sitemap.xml')[0]],
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_from_recursive_url(url: str, max_depth: int = 8, extractor=simple_extractor, exclude_dirs: List[str] = None):
    """Load documents by recursively crawling from a base URL."""
    return RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=exclude_dirs if exclude_dirs else [],
    ).load()


def load_from_gov_uk_hmrc(base_url: str = "https://www.gov.uk/hmrc/internal-manuals"):
    """Load documents from GOV.UK HMRC manuals."""
    return RecursiveUrlLoader(
        url=base_url,
        max_depth=10,
        extractor=gov_uk_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        check_response_status=True,
    ).load()


def get_or_create_vectorstore(collection_name: str):
    """Get or create a vector store based on environment configuration."""
    if os.environ.get("WEAVIATE_URL") and os.environ.get("WEAVIATE_API_KEY"):
        # Use Weaviate for cloud deployment
        import weaviate
        
        client = weaviate.Client(
            url=os.environ["WEAVIATE_URL"],
            auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
        )
        
        # Check if the class already exists
        try:
            class_properties = {
                "text": "text",
                "source": "string",
                "title": "string",
                "description": "string",
                "language": "string",
            }
            
            if not client.schema.exists(collection_name):
                client.schema.create_class({
                    "class": collection_name,
                    "properties": [
                        {"name": name, "dataType": [data_type]}
                        for name, data_type in class_properties.items()
                    ]
                })
                
            vectorstore = Weaviate(
                client=client,
                index_name=collection_name,
                text_key="text",
                embedding=get_embeddings_model(),
                by_text=False,
                attributes=["source", "title"],
            )
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating Weaviate class: {e}")
            raise e
    else:
        # Use Chroma for local deployment
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings_model(),
        )
        return vectorstore


def extract_document_topic(doc: Document) -> str:
    """Extract the main topic of a document based on title and content."""
    title = doc.metadata.get("title", "")
    
    if title:
        # Use the title if available
        return title
    
    # Try to extract topic from content
    content = doc.page_content.strip()
    
    # Take first paragraph or sentence
    first_para = content.split("\n\n")[0]
    # Limit to reasonable length
    topic = first_para[:50]
    
    if not topic:
        # Fallback to generic topic
        return "this topic"
    
    return topic


def verify_document(doc: Document, llm: Optional[LanguageModelLike] = None) -> bool:
    """Verify document content and metadata for quality and safety.
    
    Args:
        doc: The document to verify
        llm: Optional language model for advanced verification
        
    Returns:
        bool: True if document passes verification, False otherwise.
    """
    # Check for minimum content length
    if len(doc.page_content.strip()) < 20:  # Require at least 20 chars of content
        logger.debug(f"Document too short: {doc.page_content[:30]}...")
        return False
        
    # Check for required metadata
    if not doc.metadata.get("source"):
        logger.debug(f"Document missing source metadata")
        return False
        
    # Check for potentially problematic content
    problematic_patterns = [
        r'<script.*?>.*?</script>',  # Remaining script tags
        r'<iframe.*?>.*?</iframe>',  # iframes
        r'^(\s*\n)+$',  # Only whitespace or newlines
        r'access denied|forbidden|not found|404|403',  # Error messages
        r'javascript:void',  # JavaScript code
        r'error occurred|500|unauthorized',  # Additional error messages
        r'this site requires javascript|enable javascript|please enable cookies',  # Browser requirements
        r'login required|sign in to continue'  # Authentication blocks
    ]
    
    for pattern in problematic_patterns:
        if re.search(pattern, doc.page_content, re.IGNORECASE | re.DOTALL):
            logger.debug(f"Document contains problematic pattern: {pattern}")
            return False
    
    # If LLM is provided, use advanced verification
    if llm:
        try:
            # Get verification components
            verification = get_verification_components(llm)
            
            # Simulate a relevance check with a general question
            # about the topic of the document
            topic = extract_document_topic(doc)
            question = f"What information is available about {topic}?"
            
            # Check if document would pass the relevance filter
            relevant = verification.retrieval_grader.invoke({
                "question": question,
                "document": doc.page_content
            })
            
            if relevant.binary_score.lower() != "yes":
                logger.debug(f"Document failed relevance check: {relevant.reason}")
                return False
            
            return True
        except Exception as e:
            # Log but don't fail the document on verification error
            logger.warning(f"Error during advanced document verification: {e}")
            return True  # Fail open on verification errors
            
    return True

def ingest_documents(docs: List[Document], collection_name: str, llm: Optional[LanguageModelLike] = None) -> Dict[str, Union[str, int]]:
    """Ingest documents into a vector store.
    
    Args:
        docs: The documents to ingest
        collection_name: The name of the collection to store documents in
        llm: Optional language model for advanced verification
        
    Returns:
        Dict with stats about the ingestion process
    """
    # Get record manager DB URL from environment or use default
    if os.environ.get("RECORD_MANAGER_DB_URL"):
        RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
    else:
        # Use a default PostgreSQL URL for local deployment
        DATABASE_HOST = os.environ.get("DATABASE_HOST", "127.0.0.1")
        DATABASE_PORT = os.environ.get("DATABASE_PORT", "5432")
        DATABASE_USERNAME = os.environ.get("DATABASE_USERNAME", "postgres")
        DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "mysecretpassword")
        DATABASE_NAME = os.environ.get("DATABASE_NAME", "langchain")
        RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    
    # Get or create the vector store
    vectorstore = get_or_create_vectorstore(collection_name)
    
    # Create record manager
    if isinstance(vectorstore, Weaviate):
        record_manager = SQLRecordManager(
            f"weaviate/{collection_name}", db_url=RECORD_MANAGER_DB_URL
        )
    else:
        record_manager = SQLRecordManager(
            f"chroma/{collection_name}", db_url=RECORD_MANAGER_DB_URL
        )
    
    record_manager.create_schema()
    
    # Split documents
    docs_transformed = text_splitter.split_documents(docs)
    
    # Apply verification and filtering
    docs_original_count = len(docs_transformed)
    
    # Apply advanced verification if LLM is provided
    verified_docs = []
    for doc in docs_transformed:
        if len(doc.page_content) > 10 and verify_document(doc, llm):
            verified_docs.append(doc)
    
    docs_transformed = verified_docs
    
    # Log verification results
    docs_filtered_count = docs_original_count - len(docs_transformed)
    if docs_filtered_count > 0:
        logger.info(f"Filtered out {docs_filtered_count} documents during verification ({docs_filtered_count / docs_original_count:.1%})")
    
    # Ensure required metadata fields
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""
    
    # Index documents
    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="incremental",  # Use incremental to allow adding more docs later
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )
    
    # Get document count
    if isinstance(vectorstore, Weaviate):
        try:
            num_vecs = vectorstore.client.query.aggregate(collection_name).with_meta_count().do()
            doc_count = num_vecs["data"]["Aggregate"][collection_name][0]["meta"]["count"]
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            doc_count = len(docs_transformed)
    else:
        # For Chroma
        doc_count = len(docs_transformed)
    
    return {
        "collection_name": collection_name,
        "document_count": doc_count,
        "added": indexing_stats["num_added"],
        "updated": indexing_stats["num_updated"],
        "deleted": indexing_stats["num_deleted"],
        "filtered": docs_filtered_count,
    }


async def process_ingestion_request(request: IngestRequest) -> IngestResponse:
    """Process a document ingestion request."""
    try:
        # Generate collection ID if not provided
        collection_id = request.collection_id or get_collection_name(request.name, request.user_id)
        
        # Load documents based on source type
        docs = []
        if request.source_type == "sitemap":
            docs = load_from_sitemap(request.url)
        elif request.source_type == "recursive_url":
            docs = load_from_recursive_url(request.url, request.max_depth)
        elif request.source_type == "gov_uk":
            if "gov.uk/hmrc/internal-manuals" in request.url:
                docs = load_from_gov_uk_hmrc(request.url)
            else:
                docs = load_from_recursive_url(request.url, request.max_depth, extractor=gov_uk_extractor)
        else:
            return IngestResponse(
                collection_id=collection_id,
                document_count=0,
                status="error",
                message=f"Unsupported source type: {request.source_type}"
            )
        
        if not docs:
            return IngestResponse(
                collection_id=collection_id,
                document_count=0,
                status="error",
                message="No documents were loaded from the provided URL"
            )
        
        # Ingest documents
        result = ingest_documents(docs, collection_id)
        
        filtered_msg = f" ({result['filtered']} filtered during verification)" if result.get('filtered', 0) > 0 else ""
        return IngestResponse(
            collection_id=collection_id,
            document_count=result["document_count"],
            status="success",
            message=f"Successfully ingested {result['added']} documents, updated {result['updated']}, and removed {result['deleted']}{filtered_msg}"
        )
        
    except Exception as e:
        logger.error(f"Error processing ingestion request: {e}")
        return IngestResponse(
            collection_id=request.collection_id or "",
            document_count=0,
            status="error",
            message=f"Error processing ingestion request: {str(e)}"
        )


if __name__ == "__main__":
    # For testing
    import asyncio
    
    async def test_ingestion():
        request = IngestRequest(
            source_type="gov_uk",
            name="HMRC Tax Guidance",
            url="https://www.gov.uk/hmrc/internal-manuals",
            max_depth=5
        )
        
        response = await process_ingestion_request(request)
        print(f"Ingestion result: {response}")
    
    asyncio.run(test_ingestion())