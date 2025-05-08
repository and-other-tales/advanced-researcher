"""Main entrypoint for the app."""
import asyncio
import os
from typing import Optional, Union
from uuid import UUID
from pathlib import Path

import langsmith
from backend.chain import ChatRequest, answer_chain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langserve import add_routes
from langsmith import Client
from pydantic import BaseModel

from backend.dynamic_routes import router as dynamic_routes
from backend.dataset_routes import router as dataset_routes
from backend.deep_research_routes import router as deep_research_routes
from backend.auto_learn_routes import router as auto_learn_routes

client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


add_routes(
    app,
    answer_chain,
    path="/chat",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)

# Add dynamic ingestion routes
app.include_router(dynamic_routes, prefix="/api")

# Add dataset creation routes
app.include_router(dataset_routes, prefix="/api")

# Add deep research routes
app.include_router(deep_research_routes, prefix="/api")

# Add auto-learning routes
app.include_router(auto_learn_routes, prefix="/api")


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


@app.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


@app.patch("/feedback")
async def update_feedback(body: UpdateFeedbackBody):
    feedback_id = body.feedback_id
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=body.score,
        comment=body.comment,
    )
    return {"result": "patched feedback successfully", "code": 200}


# Uses a thread pool executor to run synchronous functions asynchronously
# This is the best practice for running blocking IO operations in async context
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


class GetTraceBody(BaseModel):
    run_id: UUID


@app.post("/get_trace")
async def get_trace(body: GetTraceBody):
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# Get project root directory
project_root = Path(__file__).resolve().parent.parent
backend_dir = Path(__file__).resolve().parent

# Path to the static files directory
static_path = os.path.join(backend_dir, "static")

# Path to the frontend build directory
frontend_path = os.path.join(project_root, "frontend", "public")

# Mount static files directory if it exists
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Check for the landing page
landing_page = os.path.join(static_path, "index.html")

# Mount the static directory if it exists
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Always serve the frontend directly at the root
@app.get("/")
async def serve_frontend():
    """Serve the frontend directly at the root path."""
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {
            "message": "API server is running but frontend is not built. Run 'cd frontend && yarn build' to build the frontend.",
            "endpoints": {
                "chat": "/chat",
                "api": "/api/*",
                "health": "/health"
            }
        }


if __name__ == "__main__":
    import uvicorn
    import os

    # Get port from environment variable with fallback to 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
