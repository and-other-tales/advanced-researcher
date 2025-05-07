"""Dynamic document ingestion API routes."""
import asyncio
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.dynamic_ingest import IngestRequest, IngestResponse, process_ingestion_request

logger = logging.getLogger(__name__)

router = APIRouter()


class KnowledgeBase(BaseModel):
    """Knowledge base metadata."""
    id: str
    name: str
    source: str
    document_count: int
    user_id: Optional[str] = None


class ListKnowledgeBasesResponse(BaseModel):
    """Response containing available knowledge bases."""
    knowledge_bases: List[KnowledgeBase]


class CreateKnowledgeBaseRequest(BaseModel):
    """Request to create a new knowledge base."""
    name: str
    source_type: str  # "sitemap", "recursive_url", "gov_uk"
    url: str
    max_depth: int = 8
    user_id: Optional[str] = None


# Knowledge base metadata storage
# Uses a persistent JSON file for production-grade storage
# while remaining compatible with both local and cloud deployments
import json
import os
from pathlib import Path

# Define the path to the knowledge base metadata file
# Use the DATA_MOUNT_PATH environment variable if available
try:
    DATA_MOUNT_PATH = os.environ.get("DATA_MOUNT_PATH", "/tmp/data")
    KB_METADATA_FILE = Path(DATA_MOUNT_PATH) / "knowledge_bases.json"
    
    # Initialize the global variable
    KNOWLEDGE_BASES = []
    
    # Create the directory if it doesn't exist
    os.makedirs(DATA_MOUNT_PATH, exist_ok=True)
    print(f"Using knowledge base directory: {DATA_MOUNT_PATH}")
except Exception as e:
    print(f"Error setting up knowledge base directory: {e}")
    # Use a fallback directory in /tmp
    DATA_MOUNT_PATH = "/tmp"
    KB_METADATA_FILE = Path(DATA_MOUNT_PATH) / "knowledge_bases.json"
    KNOWLEDGE_BASES = []
    print(f"Using fallback knowledge base directory: {DATA_MOUNT_PATH}")

def save_knowledge_bases():
    """Save knowledge bases to persistent storage."""
    try:
        with open(KB_METADATA_FILE, 'w') as f:
            json.dump([kb.dict() for kb in KNOWLEDGE_BASES], f)
        logger.info(f"Saved {len(KNOWLEDGE_BASES)} knowledge bases to {KB_METADATA_FILE}")
    except Exception as e:
        logger.error(f"Failed to save knowledge bases: {e}")

def load_knowledge_bases():
    """Load knowledge bases from persistent storage."""
    global KNOWLEDGE_BASES
    try:
        if KB_METADATA_FILE.exists():
            with open(KB_METADATA_FILE, 'r') as f:
                kb_dicts = json.load(f)
                KNOWLEDGE_BASES = [KnowledgeBase(**kb_dict) for kb_dict in kb_dicts]
            logger.info(f"Loaded {len(KNOWLEDGE_BASES)} knowledge bases from {KB_METADATA_FILE}")
        else:
            KNOWLEDGE_BASES = []
            logger.info("No knowledge base metadata file found, starting with empty list")
    except Exception as e:
        KNOWLEDGE_BASES = []
        logger.error(f"Failed to load knowledge bases: {e}")
        # Create an empty file to ensure we can write to it later
        save_knowledge_bases()


async def _arun(func, *args, **kwargs):
    """Run a function asynchronously."""
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


def get_knowledge_bases(user_id: Optional[str] = None) -> List[KnowledgeBase]:
    """Get knowledge bases, optionally filtered by user_id."""
    if user_id:
        return [kb for kb in KNOWLEDGE_BASES if kb.user_id == user_id]
    return KNOWLEDGE_BASES


def get_knowledge_base_by_id(kb_id: str) -> Optional[KnowledgeBase]:
    """Get a knowledge base by its ID."""
    for kb in KNOWLEDGE_BASES:
        if kb.id == kb_id:
            return kb
    return None


def load_existing_knowledge_bases():
    """Load existing knowledge bases from the environment."""
    # This would typically be loaded from a database
    # For now, we'll just add the default LangChain knowledge base
    KNOWLEDGE_BASES.append(
        KnowledgeBase(
            id=os.environ.get("WEAVIATE_DOCS_INDEX_NAME", "LangChain_Combined_Docs_OpenAI_text_embedding_3_small"),
            name="LangChain Documentation",
            source="https://python.langchain.com/",
            document_count=1000,  # Placeholder count
            user_id=None  # System-wide knowledge base
        )
    )


@router.get("/knowledge_bases", response_model=ListKnowledgeBasesResponse)
async def list_knowledge_bases(user_id: Optional[str] = None):
    """List all knowledge bases."""
    knowledge_bases = get_knowledge_bases(user_id)
    return ListKnowledgeBasesResponse(knowledge_bases=knowledge_bases)


@router.get("/knowledge_bases/{kb_id}", response_model=KnowledgeBase)
async def get_knowledge_base(kb_id: str):
    """Get a knowledge base by ID."""
    kb = get_knowledge_base_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_id} not found")
    return kb


@router.post("/knowledge_bases", response_model=IngestResponse)
async def create_knowledge_base(request: CreateKnowledgeBaseRequest):
    """Create a new knowledge base by ingesting documents."""
    # Convert to IngestRequest
    ingest_request = IngestRequest(
        source_type=request.source_type,
        name=request.name,
        url=request.url,
        max_depth=request.max_depth,
        user_id=request.user_id
    )
    
    # Process the ingestion
    response = await process_ingestion_request(ingest_request)
    
    if response.status == "success":
        # Add to the list of knowledge bases
        KNOWLEDGE_BASES.append(
            KnowledgeBase(
                id=response.collection_id,
                name=request.name,
                source=request.url,
                document_count=response.document_count,
                user_id=request.user_id
            )
        )
        
        # Persist the changes
        save_knowledge_bases()
    
    return response


@router.delete("/knowledge_bases/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """Delete a knowledge base."""
    kb = get_knowledge_base_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_id} not found")
    
    # Remove from the list
    global KNOWLEDGE_BASES
    KNOWLEDGE_BASES = [kb for kb in KNOWLEDGE_BASES if kb.id != kb_id]
    
    # Persist the changes
    save_knowledge_bases()
    
    # Delete the vector store collection
    try:
        # Check if we're using Weaviate
        if os.environ.get("WEAVIATE_URL") and os.environ.get("WEAVIATE_API_KEY"):
            # Delete Weaviate collection
            import weaviate
            
            weaviate_client = weaviate.Client(
                url=os.environ["WEAVIATE_URL"],
                auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
            )
            
            # Delete the class/collection
            if weaviate_client.schema.exists(kb_id):
                weaviate_client.schema.delete_class(kb_id)
                logger.info(f"Deleted Weaviate collection: {kb_id}")
        else:
            # Delete Chroma collection
            from langchain_community.vectorstores import Chroma
            from backend.local_embeddings import get_embeddings_model
            
            # Initialize Chroma client with the collection name
            chroma_client = Chroma(
                collection_name=kb_id,
                embedding_function=get_embeddings_model(),
            )
            
            # Delete the collection
            chroma_client.delete_collection()
            logger.info(f"Deleted Chroma collection: {kb_id}")
            
        # Also delete the record manager entries for this collection
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
        
        # Delete records for both potential formats (weaviate/chroma)
        from langchain.indexes import SQLRecordManager
        record_manager_weaviate = SQLRecordManager(
            f"weaviate/{kb_id}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager_chroma = SQLRecordManager(
            f"chroma/{kb_id}", db_url=RECORD_MANAGER_DB_URL
        )
        
        # Delete all records
        try:
            record_manager_weaviate.delete_records()
            logger.info(f"Deleted Weaviate record manager entries for: {kb_id}")
        except Exception as e:
            logger.warning(f"Error deleting Weaviate records for {kb_id}: {e}")
            
        try:
            record_manager_chroma.delete_records()
            logger.info(f"Deleted Chroma record manager entries for: {kb_id}")
        except Exception as e:
            logger.warning(f"Error deleting Chroma records for {kb_id}: {e}")
            
        return {"status": "success", "message": f"Knowledge base {kb_id} deleted"}
        
    except Exception as e:
        logger.error(f"Error deleting vector store collection {kb_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete vector store collection: {str(e)}"
        )


# Initialize knowledge bases
load_existing_knowledge_bases()