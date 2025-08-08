# LLM-Powered Intelligent Query–Retrieval System Documentation

## Overview

This system is designed to process large documents (PDFs, DOCX, emails) and provide intelligent, contextual answers to natural language queries. It's specifically optimized for insurance, legal, HR, and compliance domains.

## System Architecture

```
Input Documents → LLM Parser → Embedding Search → Clause Matching → Logic Evaluation → JSON Output
```

### Core Components

1. **Document Processor** (`app/services/document_processor.py`)
   - Downloads documents from URLs
   - Extracts text from PDF and DOCX files
   - Chunks text into overlapping segments
   - Extracts policy clauses and metadata

2. **Embedding Service** (`app/services/embedding_service.py`)
   - Creates vector embeddings using Sentence Transformers
   - Builds FAISS index for efficient similarity search
   - Performs semantic search and clause matching
   - Manages index persistence

3. **LLM Service** (`app/services/llm_service.py`)
   - Integrates with OpenAI GPT-4
   - Extracts query intent and optimizes queries
   - Generates contextual answers
   - Validates answer accuracy

4. **Retrieval Service** (`app/services/retrieval_service.py`)
   - Orchestrates the entire workflow
   - Manages caching and performance optimization
   - Provides explainable decision reasoning

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `env_example.txt` to `.env` and configure:
```bash
cp env_example.txt .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4)
- `TEAM_TOKEN`: Authentication token (already set)

### 3. Start the Server
```bash
python start_server.py
```

Or directly with uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Base URL: `http://localhost:8000/api/v1`

### Authentication
All endpoints (except `/health`) require authentication:
```
Authorization: Bearer 2bfa46dcffe69615807a8586036130853540f4c0682bcbbf5625c64a5f43e96f
```

### Main Endpoint: `POST /hackrx/run`

**Request Body:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
    ],
    "metadata": {
        "total_questions": 3,
        "processing_time": 15.23,
        "document_url": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
        "index_stats": {
            "total_chunks": 45,
            "index_size": 45,
            "dimension": 384
        }
    }
}
```

### Additional Endpoints

- `GET /health` - Health check (no auth required)
- `GET /stats` - Get processing statistics
- `POST /clear-cache` - Clear processing cache
- `POST /save-index` - Save current index to disk
- `POST /load-index` - Load index from disk
- `GET /clauses/{query}` - Get relevant policy clauses
- `POST /explain` - Generate explanation for decision

## System Features

### 1. Document Processing
- **Supported Formats**: PDF, DOCX
- **Text Extraction**: Clean, structured text extraction
- **Chunking**: Overlapping chunks for context preservation
- **Clause Detection**: Automatic policy clause identification

### 2. Semantic Search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS for fast similarity search
- **Similarity Threshold**: Configurable relevance filtering
- **Context Extraction**: Intelligent context selection

### 3. LLM Integration
- **Model**: OpenAI GPT-4
- **Query Optimization**: Automatic query enhancement
- **Intent Extraction**: Structured query understanding
- **Answer Validation**: Quality and accuracy checking

### 4. Performance Optimization
- **Caching**: Document processing cache
- **Token Efficiency**: Optimized prompt engineering
- **Batch Processing**: Efficient multi-question handling
- **Index Persistence**: Save/load for repeated use

## Evaluation Criteria Met

### ✅ Accuracy
- Precise query understanding through intent extraction
- Advanced clause matching with semantic similarity
- Answer validation and quality checking
- Context-aware response generation

### ✅ Token Efficiency
- Optimized prompts for cost-effectiveness
- Query optimization to reduce token usage
- Efficient context selection
- Batch processing for multiple questions

### ✅ Latency
- FAISS vector search for fast retrieval
- Document processing cache
- Asynchronous processing
- Optimized embedding generation

### ✅ Reusability
- Modular service architecture
- Configurable settings
- Extensible design patterns
- Clear separation of concerns

### ✅ Explainability
- Detailed metadata in responses
- Clause traceability
- Decision reasoning
- Context source identification

## Testing

### Run Test Suite
```bash
python test_system.py
```

### Manual Testing
1. Start the server: `python start_server.py`
2. Access API docs: http://localhost:8000/docs
3. Test with sample data using the Swagger UI

### Sample Test Data
The system includes comprehensive test data with 10 sample questions covering:
- Policy coverage details
- Waiting periods
- Exclusions and conditions
- Benefits and discounts
- Hospital definitions
- Treatment coverage

## Configuration

### Key Settings (`app/config.py`)
- `chunk_size`: 1000 words per chunk
- `chunk_overlap`: 200 words overlap
- `top_k_results`: 5 similar chunks
- `similarity_threshold`: 0.7 minimum similarity
- `max_tokens`: 2000 for LLM responses
- `temperature`: 0.1 for consistent responses

### Environment Variables
- `OPENAI_API_KEY`: Required for LLM functionality
- `PINECONE_API_KEY`: Optional for cloud vector DB
- `DEBUG`: Enable/disable debug mode
- `HOST/PORT`: Server configuration

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   - Set `OPENAI_API_KEY` in environment
   - Check API key validity

2. **Document Download Failures**
   - Verify document URL accessibility
   - Check network connectivity
   - Ensure proper URL encoding

3. **Memory Issues**
   - Reduce `chunk_size` for large documents
   - Clear cache: `POST /clear-cache`
   - Monitor system resources

4. **Slow Performance**
   - Check OpenAI API response times
   - Verify FAISS index optimization
   - Monitor token usage

### Performance Monitoring
- Use `/stats` endpoint for system metrics
- Monitor token usage and costs
- Track processing times per question
- Check cache hit rates

## Security Considerations

- **Authentication**: Bearer token required for all endpoints
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error messages
- **Rate Limiting**: Consider implementing for production
- **API Key Security**: Store securely in environment variables

## Production Deployment

### Recommended Setup
1. **Web Server**: Nginx for reverse proxy
2. **Process Manager**: Gunicorn with uvicorn workers
3. **Environment**: Docker containerization
4. **Monitoring**: Application performance monitoring
5. **Logging**: Structured logging with rotation

### Scaling Considerations
- **Horizontal Scaling**: Multiple server instances
- **Database**: PostgreSQL for persistent storage
- **Caching**: Redis for session management
- **Load Balancing**: Distribute requests across instances

## Future Enhancements

1. **Multi-Document Support**: Process multiple documents simultaneously
2. **Advanced Clustering**: Group similar clauses and concepts
3. **Custom Embeddings**: Domain-specific embedding models
4. **Real-time Updates**: Live document processing
5. **Advanced Analytics**: Query pattern analysis and insights

---

This system provides a robust, scalable solution for intelligent document query processing with excellent accuracy, efficiency, and explainability.
