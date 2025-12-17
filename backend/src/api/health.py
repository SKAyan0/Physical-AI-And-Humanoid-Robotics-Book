from fastapi import APIRouter
from datetime import datetime
import logging

from src.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    Check if the RAG service is operational
    """
    try:
        # TODO: Add actual health checks for dependencies (Qdrant, Neon, Gemini) in User Story 3
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.api_version,
            "debug": settings.debug
        }

        logger.info("Health check completed successfully")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }