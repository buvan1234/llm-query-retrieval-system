import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # LLM Configuration (Free Models)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Free tier model
    use_free_models: bool = os.getenv("USE_FREE_MODELS", "True").lower() == "true"
    local_model_name: str = os.getenv("LOCAL_MODEL", "distilgpt2")  # Smaller, faster free model
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")
    
    # Authentication
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Application Settings
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Team Token for HackRX
    team_token: str = os.getenv("TEAM_TOKEN", "2bfa46dcffe69615807a8586036130853540f4c0682bcbbf5625c64a5f43e96f")
    
    # Vector Search Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # LLM Settings
    max_tokens: int = 2000
    temperature: float = 0.1
    
    class Config:
        env_file = ".env"


settings = Settings()
