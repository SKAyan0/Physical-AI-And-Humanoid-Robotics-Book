import logging
import sys
from datetime import datetime
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the application
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific loggers to WARNING to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    """
    return logging.getLogger(name)


# Custom exception classes
class RAGException(Exception):
    """
    Base exception for RAG-related errors
    """
    def __init__(self, message: str, error_code: str = "RAG_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class ContentNotFoundException(RAGException):
    """
    Raised when requested content is not found
    """
    def __init__(self, message: str = "Requested content not found"):
        super().__init__(message, "CONTENT_NOT_FOUND")


class QueryProcessingException(RAGException):
    """
    Raised when there's an error processing a query
    """
    def __init__(self, message: str = "Error processing query"):
        super().__init__(message, "QUERY_PROCESSING_ERROR")


class VectorStoreException(RAGException):
    """
    Raised when there's an error with the vector store
    """
    def __init__(self, message: str = "Vector store error"):
        super().__init__(message, "VECTOR_STORE_ERROR")