import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging
from app.models import DocumentChunk, SearchResult
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for creating and managing document embeddings."""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.chunks = []
        self.index_file = "faiss_index.pkl"
        self.chunks_file = "chunks.pkl"
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for document chunks."""
        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Created embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise Exception(f"Embedding creation failed: {e}")
    
    def build_faiss_index(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Build FAISS index for efficient similarity search."""
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            self.index.add(embeddings.astype('float32'))
            
            self.chunks = chunks
            logger.info(f"Built FAISS index with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise Exception(f"Index building failed: {e}")
    
    def search_similar_chunks(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search for similar chunks using semantic similarity."""
        if not self.index or not self.chunks:
            raise Exception("Index not built. Please process documents first.")
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            top_k = top_k or settings.top_k_results
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append(SearchResult(
                        content=chunk.content,
                        similarity_score=float(score),
                        chunk_id=chunk.chunk_id,
                        metadata=chunk.metadata
                    ))
            
            # Filter by similarity threshold
            threshold = settings.similarity_threshold
            results = [r for r in results if r.similarity_score >= threshold]
            
            logger.info(f"Found {len(results)} similar chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            raise Exception(f"Search failed: {e}")
    
    def find_policy_clauses(self, query: str) -> List[SearchResult]:
        """Find specific policy clauses related to the query."""
        # Search for similar chunks
        similar_chunks = self.search_similar_chunks(query)
        
        # Filter for chunks with policy clauses
        clause_results = []
        for result in similar_chunks:
            if result.metadata.get("policy_clauses"):
                clause_results.append(result)
        
        return clause_results
    
    def extract_relevant_context(self, query: str, max_chunks: int = 3) -> str:
        """Extract relevant context for answering a query."""
        results = self.search_similar_chunks(query, top_k=max_chunks)
        
        if not results:
            return ""
        
        # Combine relevant chunks
        context_parts = []
        for result in results:
            context_parts.append(f"Relevant content (similarity: {result.similarity_score:.3f}):\n{result.content}")
        
        return "\n\n".join(context_parts)
    
    def save_index(self, directory: str = "."):
        """Save FAISS index and chunks to disk."""
        try:
            if self.index and self.chunks:
                # Save FAISS index
                index_path = os.path.join(directory, self.index_file)
                faiss.write_index(self.index, index_path)
                
                # Save chunks
                chunks_path = os.path.join(directory, self.chunks_file)
                with open(chunks_path, 'wb') as f:
                    pickle.dump(self.chunks, f)
                
                logger.info(f"Saved index and chunks to {directory}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise Exception(f"Index save failed: {e}")
    
    def load_index(self, directory: str = "."):
        """Load FAISS index and chunks from disk."""
        try:
            index_path = os.path.join(directory, self.index_file)
            chunks_path = os.path.join(directory, self.chunks_file)
            
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load chunks
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                logger.info(f"Loaded index and chunks from {directory}")
                return True
            else:
                logger.warning("Index files not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if not self.index:
            return {"status": "no_index"}
        
        return {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__
        }
