"""API routes for dataset creation and management."""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.dataset_creator import (DatasetProgress, DatasetRequest,
                                    create_dataset, get_dataset_progress,
                                    list_datasets)

logger = logging.getLogger(__name__)

router = APIRouter()


class DatasetCreateResponse(BaseModel):
    """Response for dataset creation request."""
    dataset_id: str
    status: str
    message: str


class DatasetProgressResponse(BaseModel):
    """Response containing dataset progress."""
    dataset: DatasetProgress


class ListDatasetsResponse(BaseModel):
    """Response containing a list of datasets."""
    datasets: List[DatasetProgress]


@router.post("/datasets", response_model=DatasetCreateResponse)
async def create_new_dataset(request: DatasetRequest):
    """Create a new dataset from a web source."""
    try:
        dataset_id = await create_dataset(request)
        return DatasetCreateResponse(
            dataset_id=dataset_id,
            status="success",
            message="Dataset creation started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting dataset creation: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start dataset creation: {str(e)}"
        )


@router.get("/datasets", response_model=ListDatasetsResponse)
async def get_all_datasets():
    """Get a list of all datasets."""
    datasets_list = list_datasets()
    return ListDatasetsResponse(datasets=datasets_list)


@router.get("/datasets/{dataset_id}", response_model=DatasetProgressResponse)
async def get_dataset_status(dataset_id: str):
    """Get the status of a dataset creation process."""
    dataset = get_dataset_progress(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )
    return DatasetProgressResponse(dataset=dataset)


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    """Get a download link for a dataset."""
    dataset = get_dataset_progress(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )
    
    if dataset.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} is not yet completed (current status: {dataset.status})"
        )
    
    # Return a path to the dataset
    if dataset.dataset_path:
        return {"dataset_path": dataset.dataset_path}
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset path not available for {dataset_id}"
        )