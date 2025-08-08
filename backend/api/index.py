# api/index.py - Ultra-Fast FastAPI app optimized for Vercel
from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import logging
import os
import json
import io
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import hashlib

# Document processing - optimized imports
import PyPDF2
from docx import Document
import email
from email.policy import default

# Vector search - optimized
import numpy as np
from sentence_transformers import SentenceTransformer

# Perplexity API
import openai

# Setup minimal logging for speed
logging.basicConfig(level=logging.WARNING)  # Reduced logging
logger = logging.getLogger(__name__)

# Initialize FastAPI with optimized settings
app = FastAPI(
    title="HackRx 6.0 Ultra-Fast Intelligence Query System", 
    version="2.0.0",
    docs_url=None,  # Disable docs for speed
    redoc_url=None,  # Disable redoc for speed
)

# Optimized CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only needed methods
    allow_headers=["*"],
)

# Configuration
HACKRX_BEARER_TOKEN = "2bfa46dcffe69615807a8586036130853540f4c0682bcbbf5625c64a5f43e96f"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")
CHUNK_SIZE = 500  # Reduced for faster processing
CHUNK_OVERLAP = 100  # Reduced overlap
MAX_CHUNKS = 5  # Limit chunks for speed

# Thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Initialize Perplexity client with connection pooling
perplexity_client = None
if PERPLEXITY_API_KEY:
    perplexity_client = openai.OpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai",
        timeout=10.0  # Shorter timeout
    )

# Pre-load embedding model in background
embedding_model = None
model_loading = False

def load_model_sync():
    """Load model synchronously in background"""
    global embedding_model, model_loading
    try:
        model_loading = True
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        model_loading = False
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loading = False

# Start loading model immediately
threading.Thread(target=load_model_sync, daemon=True).start()

# Memory cache for processed documents
document_cache: Dict[str, Dict[str, Any]] = {}
CACHE_MAX_SIZE = 10

@lru_cache(maxsize=32)
def get_cached_embedding_model():
    """Cached access to embedding model"""
    global embedding_model
    return embedding_model

# Optimized Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

    class Config:
        # Optimize Pydantic validation
        validate_assignment = False
        use_enum_values = True

class QueryResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    model_status: str

class OptimizedDocumentProcessor:
    @staticmethod
    async def download_document_fast(url: str) -> bytes:
        """Ultra-fast document download"""
        try:
            # Use connection pooling and streaming
            async with httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                follow_redirects=True
            ) as client:
                response = await client.get(url, headers={'User-Agent': 'FastAPI/2.0'})
                response.raise_for_status()
                return response.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)[:100]}")

    @staticmethod
    def extract_text_pdf_fast(content: bytes) -> str:
        """Fast PDF text extraction"""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Limit pages for speed
            max_pages = min(50, len(pdf_reader.pages))
            text_parts = []
            
            for i in range(max_pages):
                page_text = pdf_reader.pages[i].extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
                    
            return "\n".join(text_parts)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)[:50]}")

    @staticmethod
    def extract_text_docx_fast(content: bytes) -> str:
        """Fast DOCX text extraction"""
        try:
            doc = Document(io.BytesIO(content))
            text_parts = [p.text for p in doc.paragraphs[:200] if p.text.strip()]  # Limit paragraphs
            return "\n".join(text_parts)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DOCX processing failed: {str(e)[:50]}")

    @staticmethod
    def chunk_text_fast(text: str) -> List[str]:
        """Ultra-fast text chunking"""
        if not text:
            return []
            
        words = text.split()
        if len(words) <= CHUNK_SIZE:
            return [text]
            
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            if chunk.strip() and len(chunks) < MAX_CHUNKS:
                chunks.append(chunk)
            elif len(chunks) >= MAX_CHUNKS:
                break
                
        return chunks

class FastVectorSearch:
    @staticmethod
    def simple_similarity_search(query: str, chunks: List[str], top_k: int = 3) -> List[Dict]:
        """Fast keyword-based similarity search as fallback"""
        query_words = set(query.lower().split())
        results = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            common_words = query_words.intersection(chunk_words)
            score = len(common_words) / max(len(query_words), 1)
            
            results.append({
                "chunk": chunk,
                "score": score,
                "rank": i + 1
            })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    async def embedding_search(query: str, chunks: List[str], top_k: int = 3) -> List[Dict]:
        """Fast embedding-based search if model is available"""
        model = get_cached_embedding_model()
        if not model:
            return FastVectorSearch.simple_similarity_search(query, chunks, top_k)
        
        try:
            # Quick embedding generation
            query_emb = await asyncio.get_event_loop().run_in_executor(
                thread_pool, lambda: model.encode([query], show_progress_bar=False)
            )
            chunk_embs = await asyncio.get_event_loop().run_in_executor(
                thread_pool, lambda: model.encode(chunks, show_progress_bar=False)
            )
            
            # Fast cosine similarity
            query_norm = query_emb / np.linalg.norm(query_emb)
            chunk_norms = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)
            similarities = np.dot(chunk_norms, query_norm.T).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for i, idx in enumerate(top_indices):
                results.append({
                    "chunk": chunks[idx],
                    "score": float(similarities[idx]),
                    "rank": i + 1
                })
            
            return results
        except Exception:
            return FastVectorSearch.simple_similarity_search(query, chunks, top_k)

class FastPerplexityHandler:
    @staticmethod
    async def generate_answer_fast(question: str, context_chunks: List[Dict]) -> str:
        """Ultra-fast answer generation"""
        if not perplexity_client:
            return "API not configured"
        
        try:
            # Minimal context preparation
            context = "\n".join([chunk['chunk'][:300] for chunk in context_chunks[:2]])  # Limit context
            
            # Streamlined prompt
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely:"
            
            # Fast API call with minimal settings
            response = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: perplexity_client.chat.completions.create(
                    model="sonar-pro",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,  # Shorter responses
                    top_p=0.9
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                return "Rate limit exceeded. Try again later."
            elif "401" in error_msg:
                return "API authentication failed."
            return f"Error: {error_msg[:50]}..."

# Simplified auth
def verify_token_fast(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth")
    
    token = authorization[7:]  # Skip "Bearer "
    if token != HACKRX_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/")
async def root():
    return {"status": "active", "version": "2.0.0 - Ultra Fast"}

@app.get("/health")
async def health():
    model_status = "ready" if embedding_model else ("loading" if model_loading else "not_loaded")
    return HealthResponse(status="healthy", model_status=model_status)

@app.post("/hackrx/run")
async def hackrx_run_fast(
    request: QueryRequest,
    authorization: str = Header(None)
):
    """Ultra-fast main endpoint"""
    verify_token_fast(authorization)
    
    start_time = time.time()
    
    try:
        # Input validation (minimal)
        if not request.documents or not request.questions:
            raise HTTPException(status_code=400, detail="Missing documents or questions")
        
        # Check cache first
        doc_hash = hashlib.md5(request.documents.encode()).hexdigest()
        
        if doc_hash in document_cache:
            chunks = document_cache[doc_hash]["chunks"]
        else:
            # Fast document processing
            content = await OptimizedDocumentProcessor.download_document_fast(request.documents)
            
            # Quick text extraction based on URL
            if '.pdf' in request.documents.lower():
                text = OptimizedDocumentProcessor.extract_text_pdf_fast(content)
            elif '.docx' in request.documents.lower():
                text = OptimizedDocumentProcessor.extract_text_docx_fast(content)
            else:
                text = OptimizedDocumentProcessor.extract_text_pdf_fast(content)  # Default
            
            chunks = OptimizedDocumentProcessor.chunk_text_fast(text)
            
            # Cache result (with size limit)
            if len(document_cache) >= CACHE_MAX_SIZE:
                # Remove oldest entry
                oldest_key = next(iter(document_cache))
                del document_cache[oldest_key]
            
            document_cache[doc_hash] = {"chunks": chunks, "timestamp": time.time()}
        
        # Process questions in parallel
        async def process_question(question: str) -> str:
            # Fast similarity search
            similar_chunks = await FastVectorSearch.embedding_search(question, chunks, top_k=2)
            if not similar_chunks:
                similar_chunks = [{"chunk": chunks[0] if chunks else "", "score": 1.0, "rank": 1}]
            
            # Generate answer
            return await FastPerplexityHandler.generate_answer_fast(question, similar_chunks)
        
        # Process all questions concurrently
        tasks = [process_question(q) for q in request.questions[:10]]  # Limit questions
        answers = await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:100]}")

# Optimized error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return {"error": "Not found"}

@app.exception_handler(500)  
async def server_error(request, exc):
    return {"error": "Server error"}

# For Vercel
handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)