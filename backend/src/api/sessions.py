"""
API routes for session management functionality.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Optional

from src.models.chat import SessionCreateRequest, SessionResponse
from src.models.api import APIResponse, APIError
from src.services.chat_service import get_chat_service


router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("/", response_model=SessionResponse)
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


@router.get("/{session_id}", response_model=SessionResponse)
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


@router.get("/", response_model=List[SessionResponse])
async def list_sessions(skip: int = 0, limit: int = 20):
    """
    List chat sessions.

    This endpoint retrieves a list of chat sessions with pagination.
    Note: This implementation requires database access to list all sessions.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    # Note: The current chat service doesn't have a method to list all sessions
    # In a full implementation, this would query the database for all sessions
    # For now, we'll return an empty list with a warning
    return []


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a specific chat session.

    This endpoint deletes an existing chat session and all its messages.
    Note: This implementation is a placeholder as the chat service doesn't have delete functionality yet.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    # Note: The current chat service doesn't have a delete method
    # In a full implementation, this would mark the session as inactive or delete it
    # For now, we'll return a success response with a warning
    return APIResponse(
        success=True,
        message="Session deletion functionality not fully implemented yet",
        data={"session_id": session_id}
    )


@router.patch("/{session_id}/activate")
async def activate_session(session_id: str):
    """
    Activate a specific chat session.

    This endpoint marks a session as active.
    Note: This implementation is a placeholder as the chat service doesn't have update functionality yet.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    # Note: The current chat service doesn't have an update method
    # In a full implementation, this would update the session's active status
    return APIResponse(
        success=True,
        message="Session activation functionality not fully implemented yet",
        data={"session_id": session_id}
    )


@router.patch("/{session_id}/deactivate")
async def deactivate_session(session_id: str):
    """
    Deactivate a specific chat session.

    This endpoint marks a session as inactive.
    Note: This implementation is a placeholder as the chat service doesn't have update functionality yet.
    """
    chat_service = get_chat_service()
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )

    # Note: The current chat service doesn't have an update method
    # In a full implementation, this would update the session's active status
    return APIResponse(
        success=True,
        message="Session deactivation functionality not fully implemented yet",
        data={"session_id": session_id}
    )


@router.get("/{session_id}/history")
async def get_session_history(session_id: str, skip: int = 0, limit: int = 50):
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
        history = chat_service.get_conversation_history(session_id, limit=limit)
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found or has no history"
            )

        return {
            "session_id": session_id,
            "history": [msg.dict() for msg in history],
            "count": len(history),
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving session history: {str(e)}"
        )


@router.get("/health")
async def sessions_health():
    """
    Health check for the sessions API.

    This endpoint checks if the session management service is available.
    """
    chat_service = get_chat_service()

    if chat_service:
        return APIResponse(
            success=True,
            message="Sessions API is healthy",
            data={"service_available": True}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized"
        )