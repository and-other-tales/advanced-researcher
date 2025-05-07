"""API routes for automatic learning and knowledge base updates."""
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.auto_learn import (
    LearningStatus, get_learning_status
)

logger = logging.getLogger(__name__)

router = APIRouter()


class LearningStatusResponse(BaseModel):
    """Response containing the status of a learning task."""
    task_id: str
    status: str
    query: Optional[str] = None
    documents_found: Optional[int] = None
    documents_added: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


@router.get("/learning/{task_id}", response_model=LearningStatusResponse)
async def get_learning_task_status(task_id: str):
    """Get the status of a learning task."""
    task_data = get_learning_status(task_id)
    
    if not task_data:
        raise HTTPException(
            status_code=404,
            detail=f"Learning task {task_id} not found"
        )
    
    # Convert the task data to the response model
    response = LearningStatusResponse(
        task_id=task_id,
        status=task_data.get("status", "unknown"),
        query=task_data.get("query"),
        start_time=task_data.get("start_time"),
        end_time=task_data.get("end_time")
    )
    
    # Add result information if available
    result = task_data.get("result")
    if result:
        response.documents_found = result.documents_found
        response.documents_added = result.documents_added
        
    # Add error information if available
    if "error" in task_data:
        response.error = task_data["error"]
    elif result and hasattr(result, "error") and result.error:
        response.error = result.error
        
    return response