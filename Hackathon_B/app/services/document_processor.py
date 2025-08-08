import requests
import PyPDF2
import io
import re
from typing import List, Dict, Any, Optional
from docx import Document
import logging
from app.models import DocumentChunk
from app.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing various document types (PDF, DOCX, etc.)."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    async def download_document(self, url: str) -> bytes:
        """Download document from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download document from {url}: {e}")
            raise Exception(f"Failed to download document: {e}")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise Exception(f"Failed to process PDF: {e}")
    
    def extract_text_from_docx(self, docx_content: bytes) -> str:
        """Extract text from DOCX content."""
        try:
            doc = Document(io.BytesIO(docx_content))
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX: {e}")
            raise Exception(f"Failed to process DOCX: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', ' ', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s+([\.\,\;\:\!\?])', r'\1', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, page_number: Optional[int] = None) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunk = DocumentChunk(
                    content=chunk_text,
                    page_number=page_number,
                    chunk_id=f"chunk_{len(chunks)}",
                    metadata={
                        "start_word": i,
                        "end_word": min(i + self.chunk_size, len(words)),
                        "word_count": len(chunk_words)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def extract_policy_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Extract policy clauses and their metadata."""
        clauses = []
        
        # Common policy clause patterns
        clause_patterns = [
            r'(?:Section|Clause|Article)\s+(\d+[\.\d]*)[:\s]+([^\.]+\.)',
            r'(\d+\.\s*[^\.]+\.)',
            r'([A-Z][^\.]+(?:coverage|benefit|exclusion|condition|limit)[^\.]*\.)',
        ]
        
        for pattern in clause_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                clause_text = match.group(0)
                clauses.append({
                    "text": clause_text,
                    "type": "policy_clause",
                    "position": match.start()
                })
        
        return clauses
    
    async def process_document(self, document_url: str) -> List[DocumentChunk]:
        """Process document from URL and return chunks."""
        try:
            # Download document
            content = await self.download_document(document_url)
            
            # Determine document type and extract text
            if document_url.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(content)
            elif document_url.lower().endswith(('.docx', '.doc')):
                text = self.extract_text_from_docx(content)
            else:
                # Try PDF as default
                text = self.extract_text_from_pdf(content)
            
            # Clean text
            text = self.clean_text(text)
            
            # Extract policy clauses
            clauses = self.extract_policy_clauses(text)
            
            # Create chunks
            chunks = self.chunk_text(text)
            
            # Add clause information to metadata
            for chunk in chunks:
                chunk.metadata["policy_clauses"] = [
                    clause for clause in clauses 
                    if clause["text"] in chunk.content
                ]
            
            logger.info(f"Processed document: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process document {document_url}: {e}")
            raise Exception(f"Document processing failed: {e}")
    
    def get_document_summary(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Generate document summary from chunks."""
        total_words = sum(len(chunk.content.split()) for chunk in chunks)
        total_clauses = sum(len(chunk.metadata.get("policy_clauses", [])) for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_words": total_words,
            "total_clauses": total_clauses,
            "average_chunk_size": total_words / len(chunks) if chunks else 0
        }
