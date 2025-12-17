"""
Pydantic models for API request/response validation.
These models define the contracts for all API endpoints.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """
    Base model for API responses.
    """
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Human-readable message about the result")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the response was generated")


class APIError(BaseModel):
    """
    Model for API error responses.
    """
    success: bool = Field(default=False, description="Always False for error responses")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Specific error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the error occurred")


class HealthCheckResponse(BaseModel):
    """
    Model for health check endpoint response.
    """
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the check was performed")
    services: Dict[str, str] = Field(..., description="Status of individual services")


class ChatQueryRequest(BaseModel):
    """
    Model for chat query API requests.
    """
    message: str = Field(..., min_length=1, max_length=10000, description="The user's message or question")
    session_id: Optional[str] = Field(default=None, description="Session ID for context (optional)")
    selected_text: Optional[str] = Field(default=None, description="Text selected by user for context")
    context_module: Optional[str] = Field(default=None, description="Module to limit search context")
    context_chapter: Optional[str] = Field(default=None, description="Chapter to limit search context")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response generation")


class ChatQueryResponse(BaseModel):
    """
    Model for chat query API responses.
    """
    response: str = Field(..., description="The AI-generated response")
    session_id: str = Field(..., description="Session ID")
    sources: List[str] = Field(default_factory=list, description="Sources used in the response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the response was generated")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class DocumentIngestRequest(BaseModel):
    """
    Model for document ingestion API requests.
    """
    content: str = Field(..., min_length=1, description="The content to be ingested")
    title: str = Field(..., min_length=1, max_length=500, description="Title of the document")
    module: str = Field(..., min_length=1, description="Module this content belongs to")
    chapter: str = Field(..., min_length=1, description="Chapter this content belongs to")
    section: Optional[str] = Field(default=None, description="Section this content belongs to")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    chunk_size: int = Field(default=512, ge=100, le=2000, description="Size of text chunks")
    overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class DocumentIngestResponse(BaseModel):
    """
    Model for document ingestion API responses.
    """
    document_id: str = Field(..., description="ID of the ingested document")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    processing_time: float = Field(..., description="Time taken to process the document in seconds")
    status: str = Field(..., description="Processing status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the ingestion was completed")


class SearchRequest(BaseModel):
    """
    Model for search API requests.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    module: Optional[str] = Field(default=None, description="Module to limit search")
    chapter: Optional[str] = Field(default=None, description="Chapter to limit search")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score")


class SearchResponse(BaseModel):
    """
    Model for search API responses.
    """
    query: str = Field(..., description="The original search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Time taken for the search in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the search was performed")


class SessionCreateRequest(BaseModel):
    """
    Model for session creation API requests.
    """
    initial_context: Optional[str] = Field(default=None, description="Initial context for the session")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional session metadata")


class SessionResponse(BaseModel):
    """
    Model for session API responses.
    """
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="When the session was created")
    updated_at: datetime = Field(..., description="When the session was last updated")
    is_active: bool = Field(..., description="Whether the session is active")
    message_count: int = Field(..., description="Number of messages in the session")


class SessionListResponse(BaseModel):
    """
    Model for session listing API responses.
    """
    sessions: List[SessionResponse] = Field(..., description="List of sessions")
    total_count: int = Field(..., description="Total number of sessions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the list was generated")