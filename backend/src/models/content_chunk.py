from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class ContentChunk(BaseModel):
    """
    Model representing a chunk of book content that serves as knowledge base for the chatbot
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the chunk")
    content_id: str = Field(..., min_length=1, description="Foreign key to Book Content")
    chunk_text: str = Field(..., min_length=1, description="The text content of the chunk")
    chunk_metadata: Dict[str, Any] = Field(default={}, description="Metadata about the chunk (module, chapter, etc.)")
    vector_id: str = Field(..., min_length=1, description="ID in Qdrant vector store")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when chunk was created")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_text": "A ROS 2 node is an entity that performs computation in the ROS graph...",
                "chunk_metadata": {
                    "module": "Module 1",
                    "chapter": "Chapter 1.1",
                    "section": "Basic Concepts",
                    "page_reference": "p. 15"
                },
                "vector_id": "vector_12345",
                "created_at": "2025-12-15T10:00:00Z"
            }
        }