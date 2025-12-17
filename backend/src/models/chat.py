"""
Pydantic models for chat messages and sessions.
These models are used for request/response validation and data transfer objects.
"""
from datetime import datetime
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """
    Enum for message roles in chat conversations.
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """
    Pydantic model representing a single chat message.
    """
    role: MessageRole = Field(..., description="The role of the message sender")
    content: str = Field(..., min_length=1, description="The content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was created")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata for the message")


class ChatSession(BaseModel):
    """
    Pydantic model representing a chat session.
    """
    session_id: str = Field(..., min_length=1, description="Unique identifier for the session")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the session was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the session was last updated")
    is_active: bool = Field(default=True, description="Whether the session is currently active")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of messages in the session")


class ChatRequest(BaseModel):
    """
    Pydantic model for chat API requests.
    """
    session_id: Optional[str] = Field(default=None, description="Session ID, if continuing an existing session")
    message: str = Field(..., min_length=1, max_length=10000, description="The user's message")
    context: Optional[str] = Field(default=None, description="Additional context for the query (e.g., selected text)")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata for the request")


class ChatResponse(BaseModel):
    """
    Pydantic model for chat API responses.
    """
    session_id: str = Field(..., description="The session ID")
    response: str = Field(..., description="The assistant's response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the response was generated")
    sources: List[str] = Field(default_factory=list, description="List of sources used in the response")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata for the response")


class SessionCreateRequest(BaseModel):
    """
    Pydantic model for creating a new chat session.
    """
    session_id: Optional[str] = Field(default=None, description="Optional session ID (will be auto-generated if not provided)")
    initial_context: Optional[str] = Field(default=None, description="Optional initial context for the session")


class SessionResponse(BaseModel):
    """
    Pydantic model for session API responses.
    """
    session_id: str = Field(..., description="The session ID")
    created_at: datetime = Field(..., description="When the session was created")
    updated_at: datetime = Field(..., description="When the session was last updated")
    is_active: bool = Field(..., description="Whether the session is currently active")
    message_count: int = Field(..., description="Number of messages in the session")