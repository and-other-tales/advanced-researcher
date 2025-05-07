"""Ingest content for local deployment with Chroma and PostgreSQL."""
import logging
import os
import re
from parser import langchain_docs_extractor

from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain_core.indexing import SQLRecordManager, index
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from backend.local_embeddings import get_embeddings_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
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


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def ingest_docs():
    # Import utility for consistent environment variable handling
    from backend.utils import get_env
    
    # Use environment variables for configuration
    WEAVIATE_URL = get_env("WEAVIATE_URL")
    WEAVIATE_API_KEY = get_env("WEAVIATE_API_KEY")
    
    if WEAVIATE_URL and WEAVIATE_API_KEY:
        # Cloud deployment with Weaviate
        import weaviate
        from constants import WEAVIATE_DOCS_INDEX_NAME
        from langchain_community.vectorstores import Weaviate
        
        RECORD_MANAGER_DB_URL = get_env("RECORD_MANAGER_DB_URL")
        if not RECORD_MANAGER_DB_URL:
            raise ValueError("RECORD_MANAGER_DB_URL must be set for Weaviate deployment")
        
        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
        )
        vectorstore = Weaviate(
            client=client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=get_embeddings_model(),
            by_text=False,
            attributes=["source", "title"],
        )
        
        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
    else:
        # Local deployment with Chroma and PostgreSQL
        DATABASE_HOST = get_env("DATABASE_HOST", "127.0.0.1")
        DATABASE_PORT = get_env("DATABASE_PORT", "5432")
        DATABASE_USERNAME = get_env("DATABASE_USERNAME", "postgres")
        DATABASE_PASSWORD = get_env("DATABASE_PASSWORD", "mysecretpassword")
        DATABASE_NAME = get_env("DATABASE_NAME", "langchain")
        RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
        
        COLLECTION_NAME = get_env("COLLECTION_NAME", "langchain")
        
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings_model(),
        )
        
        record_manager = SQLRecordManager(
            f"chroma/{COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
    
    record_manager.create_schema()
    
    # Common processing code
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    
    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

    docs_transformed = text_splitter.split_documents(
        docs_from_documentation + docs_from_api + docs_from_langsmith
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    
    # Log the number of documents
    if "Weaviate" in str(type(vectorstore)):
        import weaviate
        from constants import WEAVIATE_DOCS_INDEX_NAME
        num_vecs = vectorstore.client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
        logger.info(f"Vector store now has this many vectors: {num_vecs}")
    else:
        logger.info(f"Vector store updated successfully")


if __name__ == "__main__":
    ingest_docs()