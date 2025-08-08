from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
import logging
from app.models import QueryRequest, QueryResponse, HealthCheck
from app.services.retrieval_service import RetrievalService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# Initialize the retrieval service
retrieval_service = RetrievalService()


def verify_team_token(authorization: Optional[str] = Header(None)) -> bool:
    """Verify the team token for authentication."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Extract token from "Bearer <token>" format
        if authorization.startswith("Bearer "):
            token = authorization[7:]  # Remove "Bearer " prefix
        else:
            token = authorization
        
        # Verify against team token
        if token == settings.team_token:
            return True
        else:
            raise HTTPException(status_code=401, detail="Invalid team token")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        message="LLM-Powered Intelligent Query–Retrieval System is running"
    )


@router.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    authorized: bool = Depends(verify_team_token)
):
    """
    Process documents and answer questions.
    
    This endpoint:
    1. Downloads and processes the provided document
    2. Creates embeddings for semantic search
    3. Processes each question to find relevant context
    4. Generates accurate answers using LLM
    5. Returns structured JSON response
    """
    try:
        logger.info(f"Processing query request with {len(request.questions)} questions")
        
        # Validate request
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Process the query request
        response = await retrieval_service.process_query_request(request)
        
        logger.info(f"Successfully processed {len(response.answers)} answers")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/stats")
async def get_processing_stats(authorized: bool = Depends(verify_team_token)):
    """Get processing statistics and performance metrics."""
    try:
        stats = retrieval_service.get_processing_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@router.post("/clear-cache")
async def clear_cache(authorized: bool = Depends(verify_team_token)):
    """Clear the processing cache."""
    try:
        retrieval_service.clear_cache()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@router.post("/save-index")
async def save_index(authorized: bool = Depends(verify_team_token)):
    """Save the current index to disk."""
    try:
        retrieval_service.save_index()
        return {
            "status": "success",
            "message": "Index saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving index: {str(e)}")


@router.post("/load-index")
async def load_index(authorized: bool = Depends(verify_team_token)):
    """Load index from disk."""
    try:
        success = retrieval_service.load_index()
        if success:
            return {
                "status": "success",
                "message": "Index loaded successfully"
            }
        else:
            return {
                "status": "warning",
                "message": "No index found to load"
            }
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading index: {str(e)}")


@router.get("/clauses/{query:path}")
async def get_relevant_clauses(
    query: str,
    top_k: int = 5,
    authorized: bool = Depends(verify_team_token)
):
    """Get relevant policy clauses for a query."""
    try:
        clauses = retrieval_service.get_relevant_clauses(query, top_k)
        return {
            "status": "success",
            "query": query,
            "clauses": clauses,
            "total_found": len(clauses)
        }
    except Exception as e:
        logger.error(f"Error getting clauses: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving clauses: {str(e)}")


@router.post("/explain")
async def explain_decision(
    query: str,
    answer: str,
    authorized: bool = Depends(verify_team_token)
):
    """Generate explanation for a decision/answer."""
    try:
        explanation = retrieval_service.explain_decision(query, answer)
        return {
            "status": "success",
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error explaining decision: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")
