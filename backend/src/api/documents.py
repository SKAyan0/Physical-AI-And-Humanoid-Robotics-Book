"""
API routes for document ingestion functionality.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List

from src.models.rag import DocumentIngestionRequest, DocumentIngestionResponse
from src.models.api import APIResponse, APIError
from src.services.document_ingestor import get_document_ingestor_service


router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(request: DocumentIngestionRequest):
    """
    Ingest a document into the RAG system.

    This endpoint processes a document and adds it to the vector store for
    later retrieval during chat queries.
    """
    document_ingestor = get_document_ingestor_service()
    if not document_ingestor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document ingestor service is not initialized"
        )

    try:
        result = document_ingestor.ingest_document(
            content=request.content,
            source_module=request.module,
            source_chapter=request.chapter,
            source_section=request.section,
            chunk_size=request.chunk_size,
            overlap=request.overlap
        )

        if result["success"]:
            return DocumentIngestionResponse(
                document_id=result.get("document_id", "unknown"),
                chunks_created=result["chunks_created"],
                processing_time=result["processing_time"],
                status="completed",
                metadata=result.get("metadata")
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting document: {str(e)}"
        )


@router.post("/ingest-batch")
async def ingest_documents_batch(requests: List[DocumentIngestionRequest]):
    """
    Ingest multiple documents into the RAG system.

    This endpoint processes multiple documents and adds them to the vector store.
    """
    document_ingestor = get_document_ingestor_service()
    if not document_ingestor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document ingestor service is not initialized"
        )

    results = []
    for request in requests:
        try:
            result = document_ingestor.ingest_document(
                content=request.content,
                source_module=request.module,
                source_chapter=request.chapter,
                source_section=request.section,
                chunk_size=request.chunk_size,
                overlap=request.overlap
            )
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "message": f"Error ingesting document: {str(e)}",
                "chunks_created": 0,
                "processing_time": 0
            })

    return APIResponse(
        success=True,
        message=f"Processed {len(requests)} documents",
        data={"results": results}
    )


@router.get("/status/{document_id}")
async def get_ingestion_status(document_id: str):
    """
    Get the status of a document ingestion job.

    This endpoint returns the current status of a document ingestion process.
    """
    # Note: In a real implementation, we would track ingestion jobs
    # For now, we'll return a placeholder response
    return APIResponse(
        success=True,
        message="Status check functionality not fully implemented yet",
        data={
            "document_id": document_id,
            "status": "completed",  # Placeholder
            "progress": 100,  # Placeholder
            "chunks_processed": 0  # Placeholder
        }
    )


@router.get("/health")
async def documents_health():
    """
    Health check for the documents API.

    This endpoint checks if the document ingestion service is available.
    """
    document_ingestor = get_document_ingestor_service()

    if document_ingestor:
        return APIResponse(
            success=True,
            message="Documents API is healthy",
            data={"service_available": True}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document ingestor service is not initialized"
        )