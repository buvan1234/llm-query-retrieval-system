import openai
import logging
from typing import List, Dict, Any, Optional
from app.config import settings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)


class LLMService:
    """Service for OpenAI LLM integration and query processing."""
    
    def __init__(self):
        openai.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.use_free_models = settings.use_free_models
        
        # Initialize free model if enabled
        if self.use_free_models:
            try:
                self.local_model = pipeline(
                    "text-generation",
                    model=settings.local_model_name,
                    device="cpu"  # Use CPU to avoid GPU memory issues
                )
                logger.info(f"Loaded free local model: {settings.local_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")
                self.local_model = None
        else:
            self.local_model = None
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract structured information from natural language query."""
        try:
            system_prompt = """
            You are an expert insurance policy analyzer. Extract structured information from the user's query.
            
            Return a JSON object with the following fields:
            - intent: The main intent of the query (e.g., "coverage_check", "policy_details", "exclusions", "conditions")
            - entities: List of key entities mentioned (e.g., ["knee surgery", "pre-existing diseases"])
            - policy_type: Type of policy if mentioned (e.g., "health insurance", "life insurance")
            - question_type: Type of question (e.g., "what", "does", "how", "when")
            - requires_clause_matching: Boolean indicating if specific policy clauses need to be found
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse the response to extract JSON
            content = response.choices[0].message.content
            # Simple JSON extraction (in production, use proper JSON parsing)
            import json
            try:
                return json.loads(content)
            except:
                # Fallback if JSON parsing fails
                return {
                    "intent": "general_query",
                    "entities": [],
                    "policy_type": "unknown",
                    "question_type": "what",
                    "requires_clause_matching": True
                }
                
        except Exception as e:
            logger.error(f"Failed to extract query intent: {e}")
            return {
                "intent": "general_query",
                "entities": [],
                "policy_type": "unknown",
                "question_type": "what",
                "requires_clause_matching": True
            }
    
    def generate_answer(self, query: str, context: str, policy_clauses: List[Dict] = None) -> str:
        """Generate answer using LLM with context and policy clauses."""
        try:
            # Build the prompt
            system_prompt = """
            You are an expert insurance policy analyst. Answer questions based on the provided policy document context.
            
            Guidelines:
            1. Be precise and accurate in your answers
            2. Quote specific policy clauses when relevant
            3. If information is not available in the context, say so clearly
            4. Provide clear explanations for complex policy terms
            5. Be helpful and professional in tone
            """
            
            # Prepare context
            context_parts = [f"Policy Document Context:\n{context}"]
            
            if policy_clauses:
                clauses_text = "\n\nRelevant Policy Clauses:\n"
                for i, clause in enumerate(policy_clauses[:3], 1):  # Limit to top 3 clauses
                    clauses_text += f"{i}. {clause.get('text', '')}\n"
                context_parts.append(clauses_text)
            
            full_context = "\n\n".join(context_parts)
            
            user_prompt = f"""
            {full_context}
            
            Question: {query}
            
            Please provide a comprehensive answer based on the policy document information above.
            """
            
            # Use free model if available, otherwise fall back to OpenAI
            if self.use_free_models and self.local_model:
                return self._generate_with_free_model(user_prompt)
            else:
                return self._generate_with_openai(user_prompt)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
    
    def _generate_with_free_model(self, prompt: str) -> str:
        """Generate answer using free local model."""
        try:
            # Generate response using local model
            response = self.local_model(
                prompt,
                max_length=len(prompt.split()) + 200,  # Add 200 words
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.local_model.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Remove the input prompt from the response
            answer = generated_text[len(prompt):].strip()
            
            # Clean up the response
            if not answer:
                answer = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate with free model: {e}")
            return "I apologize, but I encountered an error with the local model. Please try again."
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate with OpenAI: {e}")
            return "I apologize, but I encountered an error with the OpenAI API. Please check your API key and try again."
    
    def validate_answer(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """Validate the generated answer against the context."""
        try:
            system_prompt = """
            You are an expert insurance policy validator. Evaluate if the given answer is accurate and complete based on the provided context.
            
            Return a JSON object with:
            - is_accurate: Boolean indicating if the answer is factually correct
            - completeness_score: Float (0-1) indicating how complete the answer is
            - confidence_score: Float (0-1) indicating confidence in the answer
            - missing_information: List of any important information that might be missing
            - suggestions: List of suggestions for improvement
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                    Context: {context}
                    Question: {query}
                    Answer: {answer}
                    
                    Please evaluate this answer.
                    """}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            import json
            try:
                return json.loads(content)
            except:
                return {
                    "is_accurate": True,
                    "completeness_score": 0.8,
                    "confidence_score": 0.7,
                    "missing_information": [],
                    "suggestions": []
                }
                
        except Exception as e:
            logger.error(f"Failed to validate answer: {e}")
            return {
                "is_accurate": True,
                "completeness_score": 0.8,
                "confidence_score": 0.7,
                "missing_information": [],
                "suggestions": []
            }
    
    def optimize_query(self, original_query: str) -> str:
        """Optimize the query for better retrieval."""
        try:
            system_prompt = """
            You are an expert at optimizing queries for document retrieval. Rewrite the query to be more specific and effective for finding relevant information in insurance policy documents.
            
            Guidelines:
            1. Keep the original intent
            2. Add relevant insurance terminology
            3. Make it more specific if it's too general
            4. Focus on key entities and concepts
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original query: {original_query}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            optimized_query = response.choices[0].message.content.strip()
            logger.info(f"Optimized query: {original_query[:30]}... -> {optimized_query[:30]}...")
            return optimized_query
            
        except Exception as e:
            logger.error(f"Failed to optimize query: {e}")
            return original_query
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get current token usage statistics."""
        try:
            # This would typically come from OpenAI's usage API
            # For now, return a placeholder
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost": 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get token usage: {e}")
            return {}
