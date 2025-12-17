"""
Error handling and logging utilities for the RAG Chatbot API.
"""
import logging
import sys
from typing import Optional
from functools import wraps
from datetime import datetime

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.models.api import APIError


# Set up logging configuration
def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logger instance
    """
    logger = logging.getLogger("rag_chatbot")
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create file handler
    file_handler = logging.FileHandler("rag_chatbot.log")
    file_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logging()


class RAGChatbotError(Exception):
    """
    Base exception class for RAG Chatbot application errors.
    """
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code or "RAG_CHATBOT_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

    def to_dict(self):
        """
        Convert the error to a dictionary representation.
        """
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class QdrantError(RAGChatbotError):
    """
    Exception for Qdrant-related errors.
    """
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "QDRANT_ERROR", details)


class DatabaseError(RAGChatbotError):
    """
    Exception for database-related errors.
    """
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "DATABASE_ERROR", details)


class GeminiError(RAGChatbotError):
    """
    Exception for Google Gemini-related errors.
    """
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "GEMINI_ERROR", details)


class RAGError(RAGChatbotError):
    """
    Exception for RAG-related errors.
    """
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "RAG_ERROR", details)


def handle_exceptions(func):
    """
    Decorator to handle exceptions and log them appropriately.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RAGChatbotError as e:
            logger.error(f"RAG Chatbot Error in {func.__name__}: {e.message}", extra=e.details)
            raise e
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise RAGChatbotError(
                message="An unexpected error occurred",
                error_code="UNEXPECTED_ERROR",
                details={"function": func.__name__, "error": str(e)}
            )
    return wrapper


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for the FastAPI application.
    """
    if isinstance(exc, RAGChatbotError):
        logger.error(f"RAG Chatbot Error: {exc.message}", extra=exc.details)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIError(
                success=False,
                error=exc.message,
                error_code=exc.error_code,
                details=exc.details
            ).dict()
        )
    elif isinstance(exc, HTTPException):
        logger.warning(f"HTTP Error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content=APIError(
                success=False,
                error=str(exc.detail),
                error_code=f"HTTP_{exc.status_code}",
                details={"headers": getattr(exc, 'headers', None)}
            ).dict()
        )
    else:
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIError(
                success=False,
                error="An unexpected server error occurred",
                error_code="SERVER_ERROR",
                details={"error_type": type(exc).__name__}
            ).dict()
        )


def log_api_call(func):
    """
    Decorator to log API calls with timing and parameters.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        logger.info(f"API call started: {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"API call completed: {func.__name__} (duration: {duration:.2f}s)")
            return result
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"API call failed: {func.__name__} (duration: {duration:.2f}s) - {str(e)}")
            raise

    return wrapper


def validate_input(validation_func):
    """
    Decorator to validate input parameters.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                validation_func(*args, **kwargs)
                return await func(*args, **kwargs)
            except ValueError as e:
                logger.warning(f"Input validation failed for {func.__name__}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
        return wrapper
    return decorator


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name of the logger

    Returns:
        Logger instance
    """
    return logging.getLogger(f"rag_chatbot.{name}")


# Export the logger for use in other modules
app_logger = get_logger(__name__)