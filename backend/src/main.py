"""
Main entry point for the RAG Chatbot API for Robotics Book
"""
import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings
from src.services.qdrant_client import init_qdrant_service
from src.services.database import init_db_service
from src.services.gemini_client import init_gemini_service
from src.services.document_ingestor import init_document_ingestor_service
from src.services.chat_service import init_chat_service
from src.services.rag_service import init_rag_service
from src.api.chat import router as chat_router
from src.api.documents import router as documents_router
from src.api.sessions import router as sessions_router


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for application startup and shutdown.
    Initializes services on startup and cleans up on shutdown.
    """
    logger.info("Starting up RAG Chatbot API for Robotics Book...")

    # Load settings
    settings = Settings()

    # Initialize services
    try:
        # Initialize Qdrant service
        init_qdrant_service(settings)
        logger.info("Qdrant service initialized")

        # Initialize database service
        init_db_service(settings)
        logger.info("Database service initialized")

        # Initialize Gemini service
        init_gemini_service(settings)
        logger.info("Gemini service initialized")

        # Initialize document ingestor service
        init_document_ingestor_service(settings)
        logger.info("Document ingestor service initialized")

        # Initialize RAG service
        init_rag_service(settings)
        logger.info("RAG service initialized")

        # Initialize chat service
        init_chat_service(settings)
        logger.info("Chat service initialized")

        # Connect to services
        from src.services.qdrant_client import get_qdrant_service
        from src.services.database import get_db_service

        qdrant_service = get_qdrant_service()
        db_service = get_db_service()

        if qdrant_service:
            qdrant_service.connect()
            qdrant_service.ensure_collection_exists()

        if db_service:
            db_service.connect()

        logger.info("All services initialized and connected successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down RAG Chatbot API for Robotics Book...")

    # Close service connections
    qdrant_service = get_qdrant_service()
    if qdrant_service:
        qdrant_service.close()

    db_service = get_db_service()
    if db_service:
        db_service.close()


# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG Chatbot API for Robotics Book",
    description="API for the Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*",
    "https://physical-ai-and-humanoid-robotics-b-orpin.vercel.app/"
    ],  # In production, configure specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(sessions_router)


@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {
        "message": "Welcome to the RAG Chatbot API for Robotics Book",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/chat",
            "/api/v1/documents",
            "/api/v1/sessions"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for the API.
    """
    return {
        "status": "healthy",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    # Load settings
    settings = Settings()

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )