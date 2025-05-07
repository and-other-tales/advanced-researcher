"""API routes for deep research functionality."""
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.deep_research import (
    ResearchRequest, ResearchResponse, research, get_research, 
    check_if_complex_query, ODR_AVAILABLE
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ResearchStartResponse(BaseModel):
    """Response for starting a deep research task."""
    research_id: str
    status: str
    message: str


@router.post("/research", response_model=ResearchStartResponse)
async def start_research(request: ResearchRequest):
    """Start a deep research task on a topic."""
    if not ODR_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Deep research functionality is not available. Please install open_deep_research package."
        )
    
    try:
        research_id = await research(request)
        return ResearchStartResponse(
            research_id=research_id,
            status="started",
            message="Deep research task started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting deep research: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start deep research: {str(e)}"
        )


@router.get("/research/{research_id}", response_model=ResearchResponse)
async def get_research_status(research_id: str):
    """Get the status of a deep research task."""
    if not ODR_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Deep research functionality is not available. Please install open_deep_research package."
        )
    
    try:
        return get_research(research_id)
    except Exception as e:
        logger.error(f"Error getting research status: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get research status: {str(e)}"
        )


class ComplexQueryCheckRequest(BaseModel):
    """Request to check if a query is complex and requires deep research."""
    question: str


class ComplexQueryCheckResponse(BaseModel):
    """Response indicating if a query is complex."""
    is_complex: bool
    recommended_approach: str


@router.post("/research/check_complexity", response_model=ComplexQueryCheckResponse)
async def check_query_complexity(request: ComplexQueryCheckRequest):
    """Check if a query is complex and requires deep research."""
    is_complex = check_if_complex_query(request.question)
    
    return ComplexQueryCheckResponse(
        is_complex=is_complex,
        recommended_approach="deep_research" if is_complex else "rag"
    )