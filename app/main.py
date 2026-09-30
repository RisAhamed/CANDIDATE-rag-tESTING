"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import HTMLResponse

from app.api import candidates, search, upload
from app.core.config import settings
from app.core.logging import logger
from app.qdrant.collections import create_collections
from app.qdrant.client import qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting Candidate Search RAG Application")

    # Create Qdrant collections
    try:
        create_collections()
        logger.info("Qdrant collections ready")
    except Exception as e:
        logger.error(f"Failed to create Qdrant collections: {e}")

    # Check Qdrant connection
    if qdrant_client.health_check():
        logger.info("Qdrant connection: OK")
    else:
        logger.warning("Qdrant connection: FAILED")

    yield

    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title="Candidate Search RAG",
    description="AI-powered candidate search with interview-aware ranking",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(search.router)
app.include_router(upload.router)
app.include_router(candidates.router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    qdrant_ok = qdrant_client.health_check()
    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": "connected" if qdrant_ok else "disconnected",
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm": "configured" if settings.GROQ_API_KEY else "not_configured",
    }


@app.get("/api/health")
async def api_health():
    """API health check endpoint."""
    return await health()


@app.post("/api/ingestion/reindex")
async def reindex():
    """Re-index all candidates (placeholder for future implementation)."""
    return {"message": "Re-indexing not yet implemented", "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
