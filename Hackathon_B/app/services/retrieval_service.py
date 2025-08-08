import time
import logging
from typing import List, Dict, Any, Optional
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.models import QueryRequest, QueryResponse, ProcessingResult
from app.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Main service that orchestrates the entire retrieval and answer generation process."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.current_document_url = None
        self.processing_cache = {}
    
    async def process_query_request(self, request: QueryRequest) -> QueryResponse:
        """Process a complete query request with document processing and answer generation."""
        start_time = time.time()
        
        try:
            # Step 1: Process document if not already processed
            if request.documents != self.current_document_url:
                logger.info(f"Processing new document: {request.documents}")
                chunks = await self.document_processor.process_document(request.documents)
                
                # Step 2: Create embeddings and build index
                embeddings = self.embedding_service.create_embeddings(chunks)
                self.embedding_service.build_faiss_index(chunks, embeddings)
                
                self.current_document_url = request.documents
                self.processing_cache[request.documents] = {
                    "chunks": chunks,
                    "processing_time": time.time() - start_time
                }
            
            # Step 3: Generate answers for each question
            answers = []
            metadata = {
                "total_questions": len(request.questions),
                "processing_time": time.time() - start_time,
                "document_url": request.documents,
                "index_stats": self.embedding_service.get_index_stats()
            }
            
            for i, question in enumerate(request.questions):
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
                
                # Extract query intent
                intent = self.llm_service.extract_query_intent(question)
                
                # Optimize query for better retrieval
                optimized_query = self.llm_service.optimize_query(question)
                
                # Find relevant context
                relevant_context = self.embedding_service.extract_relevant_context(optimized_query)
                
                # Find specific policy clauses if needed
                policy_clauses = []
                if intent.get("requires_clause_matching", True):
                    clause_results = self.embedding_service.find_policy_clauses(optimized_query)
                    policy_clauses = [result.metadata.get("policy_clauses", []) for result in clause_results]
                    policy_clauses = [clause for sublist in policy_clauses for clause in sublist]
                
                # Generate answer
                answer = self.llm_service.generate_answer(question, relevant_context, policy_clauses)
                
                # Validate answer
                validation = self.llm_service.validate_answer(question, answer, relevant_context)
                
                answers.append(answer)
                
                # Add question-specific metadata
                metadata[f"question_{i+1}"] = {
                    "original_query": question,
                    "optimized_query": optimized_query,
                    "intent": intent,
                    "context_length": len(relevant_context),
                    "policy_clauses_found": len(policy_clauses),
                    "validation": validation
                }
            
            total_time = time.time() - start_time
            logger.info(f"Completed processing {len(request.questions)} questions in {total_time:.2f}s")
            
            return QueryResponse(
                answers=answers,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to process query request: {e}")
            raise Exception(f"Query processing failed: {e}")
    
    async def process_single_question(self, document_url: str, question: str) -> str:
        """Process a single question against a document."""
        try:
            # Process document if needed
            if document_url != self.current_document_url:
                chunks = await self.document_processor.process_document(document_url)
                embeddings = self.embedding_service.create_embeddings(chunks)
                self.embedding_service.build_faiss_index(chunks, embeddings)
                self.current_document_url = document_url
            
            # Optimize query
            optimized_query = self.llm_service.optimize_query(question)
            
            # Get relevant context
            relevant_context = self.embedding_service.extract_relevant_context(optimized_query)
            
            # Find policy clauses
            clause_results = self.embedding_service.find_policy_clauses(optimized_query)
            policy_clauses = []
            for result in clause_results:
                policy_clauses.extend(result.metadata.get("policy_clauses", []))
            
            # Generate answer
            answer = self.llm_service.generate_answer(question, relevant_context, policy_clauses)
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to process single question: {e}")
            raise Exception(f"Question processing failed: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        stats = {
            "current_document": self.current_document_url,
            "index_stats": self.embedding_service.get_index_stats(),
            "token_usage": self.llm_service.get_token_usage(),
            "cache_size": len(self.processing_cache)
        }
        
        if self.current_document_url and self.current_document_url in self.processing_cache:
            cache_data = self.processing_cache[self.current_document_url]
            stats["document_processing_time"] = cache_data["processing_time"]
            stats["total_chunks"] = len(cache_data["chunks"])
        
        return stats
    
    def clear_cache(self):
        """Clear the processing cache."""
        self.processing_cache.clear()
        self.current_document_url = None
        logger.info("Cleared processing cache")
    
    def save_index(self, directory: str = "."):
        """Save the current index to disk."""
        try:
            self.embedding_service.save_index(directory)
            logger.info(f"Saved index to {directory}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise Exception(f"Index save failed: {e}")
    
    def load_index(self, directory: str = ".") -> bool:
        """Load index from disk."""
        try:
            success = self.embedding_service.load_index(directory)
            if success:
                logger.info(f"Loaded index from {directory}")
            return success
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_relevant_clauses(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant policy clauses for a query."""
        try:
            clause_results = self.embedding_service.find_policy_clauses(query)
            clauses = []
            
            for result in clause_results[:top_k]:
                for clause in result.metadata.get("policy_clauses", []):
                    clauses.append({
                        "text": clause.get("text", ""),
                        "similarity_score": result.similarity_score,
                        "chunk_id": result.chunk_id,
                        "type": clause.get("type", "policy_clause")
                    })
            
            return clauses
            
        except Exception as e:
            logger.error(f"Failed to get relevant clauses: {e}")
            return []
    
    def explain_decision(self, query: str, answer: str) -> Dict[str, Any]:
        """Generate explanation for the decision/answer."""
        try:
            # Get relevant context and clauses
            relevant_context = self.embedding_service.extract_relevant_context(query)
            clauses = self.get_relevant_clauses(query)
            
            explanation = {
                "query": query,
                "answer": answer,
                "context_used": relevant_context[:500] + "..." if len(relevant_context) > 500 else relevant_context,
                "relevant_clauses": clauses[:3],  # Top 3 clauses
                "reasoning": f"Based on the policy document analysis, the answer was generated using {len(clauses)} relevant policy clauses and document context."
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain decision: {e}")
            return {
                "query": query,
                "answer": answer,
                "error": "Failed to generate explanation"
            }
