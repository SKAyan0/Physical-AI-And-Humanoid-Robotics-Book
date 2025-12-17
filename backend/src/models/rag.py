"""
Pydantic models for RAG (Retrieval-Augmented Generation) functionality.
These models handle document chunks, metadata, and RAG-specific data structures.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """
    Enum for different types of content in the RAG system.
    """
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    HEADING = "heading"
    PARAGRAPH = "paragraph"


class DocumentChunk(BaseModel):
    """
    Pydantic model representing a chunk of document content for RAG.
    """
    id: Optional[str] = Field(default=None, description="Unique identifier for the chunk")
    content: str = Field(..., min_length=1, description="The content of the chunk")
    content_type: ContentType = Field(default=ContentType.TEXT, description="Type of content in the chunk")
    source_module: str = Field(..., min_length=1, description="Module where this content originated")
    source_chapter: str = Field(..., min_length=1, description="Chapter where this content originated")
    source_section: Optional[str] = Field(default=None, description="Section where this content originated")
    chunk_order: int = Field(..., ge=0, description="Order of this chunk in the original content")
    embedding_vector: Optional[List[float]] = Field(default=None, description="Embedding vector for this chunk")
    vector_id: Optional[str] = Field(default=None, description="ID in the vector database")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the chunk")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the chunk was created")


class RAGQuery(BaseModel):
    """
    Pydantic model for RAG queries.
    """
    query_text: str = Field(..., min_length=1, max_length=10000, description="The query text")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")
    selected_text: Optional[str] = Field(default=None, description="Text selected by the user for context")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters to apply to the search")


class RAGResult(BaseModel):
    """
    Pydantic model for RAG query results.
    """
    query: str = Field(..., description="The original query")
    results: List[DocumentChunk] = Field(..., description="List of relevant document chunks")
    scores: List[float] = Field(..., description="Similarity scores for each result")
    retrieved_count: int = Field(..., description="Number of chunks retrieved")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="When the query was processed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the result")


class DocumentIngestionRequest(BaseModel):
    """
    Pydantic model for document ingestion requests.
    """
    content: str = Field(..., min_length=1, description="The content to be ingested")
    source_module: str = Field(..., min_length=1, description="Module where this content belongs")
    source_chapter: str = Field(..., min_length=1, description="Chapter where this content belongs")
    source_section: Optional[str] = Field(default=None, description="Section where this content belongs")
    chunk_size: int = Field(default=512, ge=100, le=2000, description="Size of text chunks")
    overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class DocumentIngestionResponse(BaseModel):
    """
    Pydantic model for document ingestion responses.
    """
    document_id: str = Field(..., description="ID of the ingested document")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="When the document was processed")
    status: str = Field(..., description="Status of the ingestion process")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the ingestion")