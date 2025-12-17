import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from src.config import Settings
from src.models.chat import ChatMessage, ChatSession, ChatRequest, ChatResponse
from src.models.db import ChatMessage as DBChatMessage, UserSession as DBUserSession
from src.services.database import get_db_service
from src.services.gemini_client import get_gemini_service
from src.services.qdrant_client import get_qdrant_service


logger = logging.getLogger(__name__)


class ChatService:
    """
    Service class for managing chat sessions and interactions.
    Handles session creation, message history, and chat completions with RAG context.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the chat service with configuration.

        Args:
            settings: Application settings containing configuration
        """
        self.settings = settings
        self.db_service = get_db_service()
        self.gemini_service = get_gemini_service()
        self.qdrant_service = get_qdrant_service()
        # Note: RAG functionality will be implemented in rag_service.py
        # For now, we'll use individual services directly

        if not self.db_service:
            raise RuntimeError("Database service not initialized")
        if not self.gemini_service:
            raise RuntimeError("Gemini service not initialized")
        if not self.qdrant_service:
            raise RuntimeError("Qdrant service not initialized")

    def create_session(self, initial_context: Optional[str] = None) -> str:
        """
        Create a new chat session.

        Args:
            initial_context: Optional initial context for the session

        Returns:
            str: The created session ID
        """
        session_id = str(uuid4())

        # Create database session
        db = next(self.db_service.get_session())
        try:
            db_session = DBUserSession(
                session_id=session_id,
                is_active=True
            )
            db.add(db_session)
            db.commit()

            logger.info(f"Created new chat session: {session_id}")
            return session_id
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create session {session_id}: {e}")
            raise
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get an existing chat session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            ChatSession object or None if not found
        """
        db = next(self.db_service.get_session())
        try:
            db_session = db.query(DBUserSession).filter(DBUserSession.session_id == session_id).first()
            if not db_session:
                return None

            # Get messages for this session
            db_messages = db.query(DBChatMessage).filter(
                DBChatMessage.session_id == db_session.id
            ).order_by(DBChatMessage.timestamp).all()

            # Convert to Pydantic models
            messages = [
                ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata_json
                )
                for msg in db_messages
            ]

            return ChatSession(
                session_id=db_session.session_id,
                created_at=db_session.created_at,
                updated_at=db_session.updated_at,
                is_active=db_session.is_active,
                messages=messages
            )
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
        finally:
            db.close()

    def add_message_to_session(self, session_id: str, message: ChatMessage) -> bool:
        """
        Add a message to an existing session.

        Args:
            session_id: The session ID to add the message to
            message: The message to add

        Returns:
            bool: True if successful
        """
        db = next(self.db_service.get_session())
        try:
            # Get the database session
            db_session = db.query(DBUserSession).filter(DBUserSession.session_id == session_id).first()
            if not db_session:
                logger.error(f"Session {session_id} not found")
                return False

            # Create database message
            db_message = DBChatMessage(
                session_id=db_session.id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
                metadata_json=str(message.metadata) if message.metadata else None
            )

            db.add(db_message)
            db_session.updated_at = datetime.utcnow()
            db.commit()

            logger.info(f"Added message to session {session_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return False
        finally:
            db.close()

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """
        Get the conversation history for a session.

        Args:
            session_id: The session ID to get history for
            limit: Maximum number of messages to return

        Returns:
            List of ChatMessage objects
        """
        db = next(self.db_service.get_session())
        try:
            # Get the database session
            db_session = db.query(DBUserSession).filter(DBUserSession.session_id == session_id).first()
            if not db_session:
                return []

            # Get messages for this session
            db_messages = db.query(DBChatMessage).filter(
                DBChatMessage.session_id == db_session.id
            ).order_by(DBChatMessage.timestamp.desc()).limit(limit).all()

            # Convert to Pydantic models and reverse to get chronological order
            messages = [
                ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata_json
                )
                for msg in reversed(db_messages)  # Reverse to get chronological order
            ]

            return messages
        except Exception as e:
            logger.error(f"Failed to get conversation history for session {session_id}: {e}")
            return []
        finally:
            db.close()

    def process_chat_request(self, chat_request: ChatRequest) -> ChatResponse:
        """
        Process a chat request and generate a response.

        Args:
            chat_request: The chat request containing the user's message

        Returns:
            ChatResponse containing the assistant's response
        """
        # Create or get session
        session_id = chat_request.session_id or self.create_session()
        if not chat_request.session_id:
            # If we created a new session, make sure to save it
            pass

        # Prepare the conversation context
        conversation_history = self.get_conversation_history(session_id, limit=10)

        # Build the conversation history for Gemini chat
        messages = []

        # Add system instruction as the first message (if needed)
        system_message = {
            "role": "user",
            "content": (
                "You are an AI assistant for the Physical AI & Humanoid Robotics Book. "
                "Your purpose is to help readers understand robotics concepts by answering "
                "questions based on the book content. Be helpful, accurate, and reference "
                "the book content when possible."
            )
        }
        messages.append(system_message)

        # Add previous conversation history
        for msg in conversation_history:
            role = "user" if msg.role == "user" else "model"  # Gemini uses 'model' for assistant
            messages.append({
                "role": role,
                "content": msg.content
            })

        # Add the current user message with context
        user_message_content = chat_request.message
        if chat_request.context:
            user_message_content = f"Context from selected text: {chat_request.context}\n\nUser question: {chat_request.message}"

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message_content
        })

        # If we have RAG context, perform a RAG query to get relevant information
        sources = []
        if chat_request.context or len(chat_request.message) > 10:  # Only query if there's substantial content
            try:
                # Placeholder for RAG functionality - this will be implemented in rag_service.py
                # For now, we'll skip RAG context but will implement it when rag_service is ready
                # rag_results = self.rag_service.query(
                #     query_text=chat_request.message,
                #     selected_text=chat_request.context,
                #     top_k=3
                # )

                # TODO: Implement RAG query functionality when rag_service is ready
                # For now, this is a placeholder to show where RAG context would be added
                pass  # Empty try block needs a statement

            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Generate response using Gemini
        try:
            temperature = chat_request.metadata.get("temperature", 0.7) if chat_request.metadata else 0.7
            max_tokens = chat_request.metadata.get("max_tokens", 1000) if chat_request.metadata else 1000

            response = self.gemini_service.chat_completion(
                messages=messages,
                model=self.settings.gemini_model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract the response text
            if hasattr(response, 'text'):
                assistant_response = response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                assistant_response = response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Unexpected response format from Gemini API")

            # Create chat messages for the database
            user_message = ChatMessage(
                role="user",
                content=chat_request.message
            )

            assistant_message = ChatMessage(
                role="assistant",
                content=assistant_response
            )

            # Add messages to session
            self.add_message_to_session(session_id, user_message)
            self.add_message_to_session(session_id, assistant_message)

            return ChatResponse(
                session_id=session_id,
                response=assistant_response,
                timestamp=datetime.utcnow(),
                sources=[],  # Will be populated when RAG functionality is implemented
                metadata={"sources_count": 0}  # Will be updated when RAG functionality is implemented
            )

        except Exception as e:
            logger.error(f"Failed to generate chat response: {e}")
            raise


# Global instance - will be initialized with settings
chat_service: Optional[ChatService] = None


def get_chat_service() -> Optional[ChatService]:
    """
    Get the global chat service instance.

    Returns:
        ChatService: The global instance or None if not initialized
    """
    return chat_service


def init_chat_service(settings: Settings) -> ChatService:
    """
    Initialize the global chat service instance with the provided settings.

    Args:
        settings: Application settings containing configuration

    Returns:
        ChatService: The initialized service instance
    """
    global chat_service
    chat_service = ChatService(settings)
    return chat_service