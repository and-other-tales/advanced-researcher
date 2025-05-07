#!/usr/bin/env python
"""Test script for dataset creation functionality."""
import asyncio
import json
import logging
import sys
import time

from backend.dataset_creator import DatasetRequest, DatasetType, create_dataset, get_dataset_progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_legislation_dataset_creation():
    """Test creating a dataset from legislation.gov.uk."""
    logger.info("Testing legislation dataset creation...")
    
    request = DatasetRequest(
        name="UK Primary Legislation Test",
        source_type=DatasetType.LEGISLATION,
        url="https://www.legislation.gov.uk/ukpga/2020/1/contents",
        description="UK European Union (Withdrawal Agreement) Act 2020",
        max_depth=3  # Limit depth for testing
    )
    
    dataset_id = await create_dataset(request)
    logger.info(f"Dataset creation started with ID: {dataset_id}")
    
    # Poll for progress
    while True:
        progress = get_dataset_progress(dataset_id)
        if not progress:
            logger.error(f"Dataset {dataset_id} not found")
            break
        
        logger.info(f"Status: {progress.status}, Progress: {progress.progress:.1f}%")
        
        if progress.status in ["completed", "failed"]:
            logger.info(f"Final status: {progress.status}")
            if progress.status == "failed" and progress.error_message:
                logger.error(f"Error: {progress.error_message}")
            if progress.status == "completed" and progress.dataset_path:
                logger.info(f"Dataset created at: {progress.dataset_path}")
            break
        
        await asyncio.sleep(5)
    
    return dataset_id


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_dataset_creation.py")
        print("Tests the dataset creation functionality")
        sys.exit(0)
    
    dataset_id = asyncio.run(test_legislation_dataset_creation())
    
    # Print final status
    progress = get_dataset_progress(dataset_id)
    if progress:
        print(f"\nDataset creation {'successful' if progress.status == 'completed' else 'failed'}")
        print(f"Status: {progress.status}")
        print(f"Progress: {progress.progress:.1f}%")
        if progress.dataset_path:
            print(f"Dataset created at: {progress.dataset_path}")
        if progress.error_message:
            print(f"Error: {progress.error_message}")
    else:
        print(f"\nDataset {dataset_id} not found")