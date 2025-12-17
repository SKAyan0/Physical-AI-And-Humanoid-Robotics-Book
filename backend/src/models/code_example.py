from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class CodeExample(BaseModel):
    """
    Model representing executable code samples demonstrating robotics concepts using ROS 2, Isaac, and related technologies
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the code example")
    content_id: str = Field(..., min_length=1, description="Foreign key to Book Content")
    code_snippet: str = Field(..., min_length=1, description="The actual code snippet")
    language: str = Field(..., min_length=1, description="Programming language of the code (python, c++, etc.)")
    module: str = Field(..., pattern=r"^(Module 1|Module 2|Module 3|Module 4)$", description="Module the example belongs to")
    chapter: str = Field(..., min_length=1, description="Chapter within the module")
    description: str = Field(default="", description="Description of what the code example demonstrates")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when code example was created")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174004",
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "code_snippet": "import rclpy\nfrom rclpy.node import Node\n\nclass MinimalPublisher(Node):\n    def __init__(self):\n        super().__init__('minimal_publisher')",
                "language": "python",
                "module": "Module 1",
                "chapter": "Chapter 1.2",
                "description": "Basic ROS 2 publisher node example",
                "created_at": "2025-12-15T10:45:00Z"
            }
        }