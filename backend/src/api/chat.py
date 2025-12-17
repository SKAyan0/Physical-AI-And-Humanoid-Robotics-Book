"""
API routes for chat functionality.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status

from src.models.chat import ChatRequest, ChatResponse, SessionCreateRequest, SessionResponse
from src.models.api import APIResponse, APIError
from src.services.chat_service import get_chat_service, init_chat_service
from src.services.database import get_db


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def chat_query(chat_request: ChatRequest):
    """
    Handle a chat query and return a response.

    This endpoint processes a user's message and returns an AI-generated response
    with RAG context when available.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    try:
        response = chat_service.process_chat_request(chat_request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post("/query-by-selection", response_model=ChatResponse)
async def chat_query_by_selection(chat_request: ChatRequest):
    """
    Handle a chat query with selected text context.

    This endpoint processes a user's message along with selected text context
    and returns an AI-generated response based on the selected content.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    # The chat service already handles selected text via the context field
    # This endpoint is essentially the same as /query but with explicit intent
    try:
        response = chat_service.process_chat_request(chat_request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request with selection: {str(e)}"
        )


@router.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new chat session.

    This endpoint creates a new chat session that can be used for maintaining
    conversation context across multiple interactions.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    try:
        session_id = chat_service.create_session(request.initial_context)
        session = chat_service.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )

        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active,
            message_count=len(session.messages)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating session: {str(e)}"
        )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get information about a specific chat session.

    This endpoint retrieves information about an existing chat session.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    try:
        session = chat_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )

        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active,
            message_count=len(session.messages)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving session: {str(e)}"
        )


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 10):
    """
    Get the conversation history for a specific session.

    This endpoint retrieves the message history for an existing chat session.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    try:
        history = chat_service.get_conversation_history(session_id, limit)
        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )

        return {
            "session_id": session_id,
            "history": [msg.dict() for msg in history],
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving session history: {str(e)}"
        )


@router.post("/session/{session_id}/message")
async def add_message_to_session(session_id: str, message: str):
    """
    Add a message to an existing session.

    This endpoint allows adding a message to a specific session without
    triggering a full chat response generation.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    from src.models.chat import ChatMessage, MessageRole
    chat_message = ChatMessage(
        role=MessageRole.USER,
        content=message
    )

    try:
        success = chat_service.add_message_to_session(session_id, chat_message)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )

        return APIResponse(
            success=True,
            message="Message added to session successfully",
            data={"session_id": session_id}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding message to session: {str(e)}"
        )