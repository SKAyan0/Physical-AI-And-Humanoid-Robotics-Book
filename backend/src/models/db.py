"""
Database models for the RAG chatbot system using SQLAlchemy.
These models represent the core entities stored in Neon Postgres.
"""
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from src.services.database import Base


class UserSession(Base):
    """
    Model representing a user's chat session.
    """
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationship to chat messages
    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(Base):
    """
    Model representing individual chat messages within a session.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("user_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata_json = Column(Text)  # JSON string for additional metadata

    # Relationship back to session
    session = relationship("UserSession", back_populates="messages")


class BookContent(Base):
    """
    Model representing book content/chapters that can be queried by the RAG system.
    """
    __tablename__ = "book_content"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    module = Column(String, nullable=False)  # Module number or name
    chapter = Column(String, nullable=False)  # Chapter number or name
    content = Column(Text, nullable=False)
    content_hash = Column(String, unique=True, nullable=False)  # For deduplication
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)


class ContentChunk(Base):
    """
    Model representing chunks of book content for RAG indexing.
    """
    __tablename__ = "content_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(Integer, ForeignKey("book_content.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_order = Column(Integer, nullable=False)
    vector_id = Column(String)  # Reference to Qdrant vector ID
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship back to book content
    content = relationship("BookContent")


class UserQueryLog(Base):
    """
    Model for logging user queries for analytics and debugging.
    """
    __tablename__ = "user_query_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("user_sessions.id"))
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    query_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    response_timestamp = Column(DateTime)
    metadata_json = Column(Text)  # JSON string for query context and results
    is_successful = Column(Boolean, default=True)