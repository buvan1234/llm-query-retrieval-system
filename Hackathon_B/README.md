# LLM-Powered Intelligent Query–Retrieval System

An intelligent document processing and query system designed for insurance, legal, HR, and compliance domains. The system can process PDFs, DOCX, and email documents, extract structured information, and provide contextual answers to natural language queries.

## Features

- **Document Processing**: Handle PDF, DOCX, and email documents
- **Semantic Search**: Use embeddings (FAISS/Pinecone) for intelligent retrieval
- **Clause Matching**: Advanced semantic similarity for policy clause matching
- **Explainable Decisions**: Provide clear reasoning for all responses
- **Structured Output**: JSON responses with detailed explanations
- **Token Optimization**: Efficient LLM usage for cost-effectiveness

## System Architecture

```
Input Documents → LLM Parser → Embedding Search → Clause Matching → Logic Evaluation → JSON Output
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Application**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access API Documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Base URL: `http://localhost:8000/api/v1`

### Authentication
```
Authorization: Bearer 2bfa46dcffe69615807a8586036130853540f4c0682bcbbf5625c64a5f43e96f
```

### POST `/hackrx/run`
Process documents and answer questions.

**Request Body:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
    ]
}
```

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── models.py              # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Document parsing
│   │   ├── embedding_service.py   # Vector embeddings
│   │   ├── llm_service.py        # LLM integration
│   │   └── retrieval_service.py  # Search and retrieval
│   └── api/
│       ├── __init__.py
│       └── endpoints.py        # API endpoints
├── requirements.txt
├── .env.example
└── README.md
```

## Evaluation Criteria

- **Accuracy**: Precision of query understanding and clause matching
- **Token Efficiency**: Optimized LLM token usage
- **Latency**: Response speed and real-time performance
- **Reusability**: Code modularity and extensibility
- **Explainability**: Clear decision reasoning and clause traceability

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: Pinecone/FAISS
- **LLM**: OpenAI GPT-4
- **Database**: PostgreSQL (optional)
- **Document Processing**: PyPDF2, python-docx
- **Embeddings**: Sentence Transformers
