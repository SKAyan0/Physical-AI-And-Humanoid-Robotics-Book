from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # API Settings
    api_title: str = "RAG Chatbot API for Robotics Book"
    api_version: str = "1.0.0"
    api_description: str = "API for the Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot"

    # Gemini Settings
    gemini_api_key: str = Field(default="MISSING_GEMINI_API_KEY", alias="GEMINI_API_KEY", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL", description="Google Gemini model to use for chat completions")
    gemini_embedding_model: str = Field(default="text-embedding-004", alias="GEMINI_EMBEDDING_MODEL", description="Google model to use for embeddings")


    # Qdrant Settings
    qdrant_url: str = Field(default="MISSING_QDRANT_URL", alias="QDRANT_URL", description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY", description="Qdrant API key")
    qdrant_collection_name: str = Field(default="book_content", alias="QDRANT_COLLECTION_NAME", description="Qdrant collection name")

    # Database Settings
    database_url: str = Field(default="MISSING_DATABASE_URL", alias="NEON_DATABASE_URL", description="Database URL for Neon Postgres")

    # Application Settings
    debug: bool = Field(default=False, alias="DEBUG", description="Enable debug mode")
    host: str = Field(default="0.0.0.0", alias="HOST", description="Host to bind to")
    port: int = Field(default=8000, alias="PORT", description="Port to bind to")
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS", description="Comma-separated list of allowed origins")

    # RAG Settings
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE", description="Size of content chunks in characters")
    rag_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP", description="Overlap between content chunks in characters")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K", description="Number of chunks to retrieve for each query")
    rag_similarity_threshold: float = Field(default=0.5, alias="RAG_SIMILARITY_THRESHOLD", description="Minimum similarity score for retrieval")
    rag_max_content_length: int = Field(default=10000, alias="RAG_MAX_CONTENT_LENGTH", description="Maximum length of content to process")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_prefix": "",
        "populate_by_name": True  # Allow population by field name or alias
    }

    @property
    def allowed_origins_list(self) -> List[str]:
        """
        Convert the allowed_origins string to a list
        """
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


# Create a global settings instance
settings = Settings()