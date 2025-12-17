from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class BookContent(BaseModel):
    """
    Model representing structured educational content organized in 4 modules with chapters covering specific robotics concepts
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the content")
    title: str = Field(..., min_length=1, description="Title of the content")
    module: str = Field(..., pattern=r"^(Module 1|Module 2|Module 3|Module 4)$", description="Module the content belongs to")
    chapter: str = Field(..., min_length=1, description="Chapter within the module")
    section: str = Field(default="", description="Section within the chapter")
    content: str = Field(..., min_length=1, description="The actual content in markdown format")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when content was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when content was last updated")
    version: str = Field(default="1.0.0", description="Version of the content")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Introduction to ROS 2 Nodes",
                "module": "Module 1",
                "chapter": "Chapter 1.1",
                "section": "Basic Concepts",
                "content": "# ROS 2 Nodes\n\nA ROS 2 node is an entity that performs computation in the ROS graph...",
                "version": "1.0.0"
            }
        }