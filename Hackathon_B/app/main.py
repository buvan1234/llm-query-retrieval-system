from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from app.api.endpoints import router
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    description="An intelligent document processing and query system for insurance, legal, HR, and compliance domains.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "LLM-Powered Intelligent Query–Retrieval System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1")
async def api_info():
    """API information endpoint."""
    return {
        "api_name": "LLM-Powered Intelligent Query–Retrieval System API",
        "version": "1.0.0",
        "base_url": "/api/v1",
        "endpoints": {
            "health": "GET /api/v1/health",
            "process_query": "POST /api/v1/hackrx/run",
            "stats": "GET /api/v1/stats",
            "clear_cache": "POST /api/v1/clear-cache",
            "save_index": "POST /api/v1/save-index",
            "load_index": "POST /api/v1/load-index",
            "get_clauses": "GET /api/v1/clauses/{query}",
            "explain_decision": "POST /api/v1/explain"
        },
        "authentication": "Bearer token required for all endpoints except /health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
