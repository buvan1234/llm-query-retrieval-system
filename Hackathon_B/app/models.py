from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for document query processing."""
    documents: str = Field(..., description="URL or path to the document(s)")
    questions: List[str] = Field(..., description="List of questions to answer")


class QueryResponse(BaseModel):
    """Response model for document query processing."""
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the processing")


class DocumentChunk(BaseModel):
    """Model for document chunks with metadata."""
    content: str
    page_number: Optional[int] = None
    chunk_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Model for search results."""
    content: str
    similarity_score: float
    chunk_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Model for document processing results."""
    document_url: str
    chunks: List[DocumentChunk]
    total_chunks: int
    processing_time: float


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    message: str
    version: str = "1.0.0"
