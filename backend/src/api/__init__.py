"""
API routing and middleware structure for the RAG chatbot backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(level="INFO" if not settings.debug else "DEBUG")

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Additional configuration can be added here
)

# Import and include routers after app creation to avoid circular imports
from . import chat, content, health
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(content.router, prefix="/api/v1", tags=["content"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API for Robotics Book", "version": settings.api_version}