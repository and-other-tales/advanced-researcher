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

# Add a root API endpoint
@app.get("/api")
async def api_root():
    """API root endpoint that returns available endpoints."""
    return {
        "status": "ok",
        "endpoints": {
            "knowledge_bases": "/api/knowledge_bases",
            "datasets": "/api/datasets",
            "research": "/api/research",
            "learning": "/api/learning"
        }
    }

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
frontend_out_path = os.path.join(project_root, "frontend", "out")

# First check if frontend/out exists (production build)
if os.path.exists(frontend_out_path):
    # Mount the static directory for both _next paths using frontend/out
    app.mount("/_next", StaticFiles(directory=os.path.join(frontend_out_path, "_next")), name="next_static")
    # Also mount static assets from frontend build
    for subdir in ["images", "css", "js"]:
        subdir_path = os.path.join(frontend_out_path, subdir)
        if os.path.exists(subdir_path):
            app.mount(f"/{subdir}", StaticFiles(directory=subdir_path), name=f"frontend_{subdir}")
    
# Also mount assets from backend/static directory
if os.path.exists(static_path):
    # Mount the entire static directory
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # Check for Next.js build hash in backend/static
    build_hash_dirs = [d for d in os.listdir(static_path) 
                      if os.path.isdir(os.path.join(static_path, d)) 
                      and d not in ["chunks", "css", "js", "media"]]
    
    if build_hash_dirs:
        build_hash = build_hash_dirs[0]  # Use first found build hash
        app.mount(f"/{build_hash}", StaticFiles(directory=os.path.join(static_path, build_hash)), name="build_hash")
        
    # Mount key Next.js directories
    for next_dir in ["chunks", "css", "media"]:
        next_dir_path = os.path.join(static_path, next_dir)
        if os.path.exists(next_dir_path):
            app.mount(f"/_next/static/{next_dir}", StaticFiles(directory=next_dir_path), name=f"next_{next_dir}")

# Add any Next.js static asset routes
@app.get("/_next/static/{build_id}/_buildManifest.js")
async def serve_build_manifest(build_id: str):
    """Serve the Next.js build manifest."""
    # First check in frontend/out
    if os.path.exists(frontend_out_path):
        manifest_path = os.path.join(frontend_out_path, "_next", "static", build_id, "_buildManifest.js")
        if os.path.exists(manifest_path):
            return FileResponse(manifest_path)
    
    # Then check in backend/static directory
    manifest_path = os.path.join(static_path, build_id, "_buildManifest.js")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path)
    
    # Check in all subdirectories of backend/static
    for subdir in os.listdir(static_path):
        potential_path = os.path.join(static_path, subdir, "_buildManifest.js")
        if os.path.exists(potential_path):
            return FileResponse(potential_path)
            
    return {"error": "Build manifest not found"}

@app.get("/_next/static/{build_id}/_ssgManifest.js")
async def serve_ssg_manifest(build_id: str):
    """Serve the Next.js SSG manifest."""
    # First check in frontend/out
    if os.path.exists(frontend_out_path):
        manifest_path = os.path.join(frontend_out_path, "_next", "static", build_id, "_ssgManifest.js")
        if os.path.exists(manifest_path):
            return FileResponse(manifest_path)
    
    # Then check in backend/static directory
    manifest_path = os.path.join(static_path, build_id, "_ssgManifest.js")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path)
    
    # Check in all subdirectories of backend/static
    for subdir in os.listdir(static_path):
        potential_path = os.path.join(static_path, subdir, "_ssgManifest.js")
        if os.path.exists(potential_path):
            return FileResponse(potential_path)
            
    return {"error": "SSG manifest not found"}

# Catch all route for Next.js static files
@app.get("/_next/{path:path}")
async def serve_next_static(path: str):
    """Serve Next.js static files from either frontend/out or backend/static."""
    # First check in frontend/out
    if os.path.exists(frontend_out_path):
        file_path = os.path.join(frontend_out_path, "_next", path)
        if os.path.exists(file_path) and not os.path.isdir(file_path):
            return FileResponse(file_path)
    
    # For static chunks pattern: _next/static/chunks/...
    if path.startswith("static/chunks/"):
        chunks_path = os.path.join(static_path, "chunks", path.replace("static/chunks/", ""))
        if os.path.exists(chunks_path):
            return FileResponse(chunks_path)
            
    # For static css pattern: _next/static/css/...
    if path.startswith("static/css/"):
        css_path = os.path.join(static_path, "css", path.replace("static/css/", ""))
        if os.path.exists(css_path):
            return FileResponse(css_path)
            
    # For static media pattern: _next/static/media/...
    if path.startswith("static/media/"):
        media_path = os.path.join(static_path, "media", path.replace("static/media/", ""))
        if os.path.exists(media_path):
            return FileResponse(media_path)
    
    return {"error": f"Static file not found: {path}"}

# Always serve the frontend directly at the root
@app.get("/")
async def serve_frontend():
    """Serve the frontend directly at the root path."""
    # First try the frontend/out directory
    if os.path.exists(frontend_out_path):
        index_path = os.path.join(frontend_out_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    
    # Fall back to backend/static directory
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

# Add explicit routes for test files in the static directory
@app.get("/test.html")
async def serve_test_page():
    """Serve the test page for debugging."""
    test_path = os.path.join(static_path, "test.html")
    if os.path.exists(test_path):
        return FileResponse(test_path)
    return {"error": "Test page not found"}
    
@app.get("/debug.html")
async def serve_debug_page():
    """Serve the debug page for troubleshooting."""
    debug_path = os.path.join(static_path, "debug.html")
    if os.path.exists(debug_path):
        return FileResponse(debug_path)
    return {"error": "Debug page not found"}


if __name__ == "__main__":
    import uvicorn
    import os

    # Get port from environment variable with fallback to 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
