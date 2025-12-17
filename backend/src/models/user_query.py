from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class UserQuery(BaseModel):
    """
    Model representing questions from learners that the RAG chatbot processes to provide relevant answers
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the query")
    query_text: str = Field(..., min_length=1, description="The actual question text from the user")
    user_context: Optional[Dict[str, Any]] = Field(default={}, description="Context about the query (selected text, referring page, etc.)")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when query was created")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174002",
                "query_text": "How do I create a ROS 2 node in Python?",
                "user_context": {
                    "selected_text": "A ROS 2 node is an entity that performs computation in the ROS graph.",
                    "referring_page": "/module1/chapter1.1",
                    "timestamp": "2025-12-15T10:30:00Z"
                },
                "created_at": "2025-12-15T10:30:00Z"
            }
        }