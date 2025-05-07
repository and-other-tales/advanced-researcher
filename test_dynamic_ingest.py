#!/usr/bin/env python
"""Test script for dynamic document ingestion."""
import asyncio
import logging
import sys

from backend.dynamic_ingest import IngestRequest, process_ingestion_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_govuk_ingestion():
    """Test ingesting documents from GOV.UK."""
    logger.info("Testing HMRC Tax Guidance ingestion...")
    
    request = IngestRequest(
        source_type="gov_uk",
        name="HMRC Tax Guidance",
        url="https://www.gov.uk/hmrc/internal-manuals",
        max_depth=3  # Limit depth for testing
    )
    
    response = await process_ingestion_request(request)
    logger.info(f"Ingestion result: {response}")
    return response


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_dynamic_ingest.py")
        print("Tests the dynamic document ingestion functionality")
        sys.exit(0)
    
    result = asyncio.run(test_govuk_ingestion())
    if result.status == "success":
        print(f"\nSuccess! Created knowledge base {result.collection_id} with {result.document_count} documents")
        print(f"You can now use this knowledge base by setting knowledge_base_id={result.collection_id} in your chat requests")
    else:
        print(f"\nFailed to create knowledge base: {result.message}")