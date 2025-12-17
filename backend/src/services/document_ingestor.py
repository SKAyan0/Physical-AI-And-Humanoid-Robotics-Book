import hashlib
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from src.config import Settings
from src.models.rag import DocumentChunk
from src.services.gemini_client import get_gemini_service
from src.services.qdrant_client import get_qdrant_service


logger = logging.getLogger(__name__)


class DocumentIngestorService:
    """
    Service class for ingesting documents into the RAG system.
    Handles document chunking, embedding generation, and storage in Qdrant.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the document ingestor service with configuration.

        Args:
            settings: Application settings containing configuration
        """
        self.settings = settings
        self.gemini_service = get_gemini_service()
        self.qdrant_service = get_qdrant_service()

        if not self.gemini_service:
            raise RuntimeError("Gemini service not initialized")
        if not self.qdrant_service:
            raise RuntimeError("Qdrant service not initialized")

    def chunk_document(
        self,
        content: str,
        source_module: str,
        source_chapter: str,
        source_section: Optional[str] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[DocumentChunk]:
        """
        Split a document into overlapping chunks.

        Args:
            content: The document content to chunk
            source_module: Module where this content belongs
            source_chapter: Chapter where this content belongs
            source_section: Section where this content belongs
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of DocumentChunk objects
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")

        chunks = []
        start = 0
        chunk_order = 0

        while start < len(content):
            end = start + chunk_size

            # If this is not the last chunk, try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings near the end of the chunk
                chunk_end = content.rfind('.', start, end)
                if chunk_end == -1:
                    chunk_end = content.rfind('!', start, end)
                if chunk_end == -1:
                    chunk_end = content.rfind('?', start, end)
                if chunk_end == -1:
                    chunk_end = end  # Use original end if no sentence boundary found
                else:
                    chunk_end += 1  # Include the punctuation
            else:
                chunk_end = end

            chunk_text = content[start:chunk_end].strip()

            if chunk_text:  # Only add non-empty chunks
                chunk = DocumentChunk(
                    id=str(uuid4()),
                    content=chunk_text,
                    source_module=source_module,
                    source_chapter=source_chapter,
                    source_section=source_section,
                    chunk_order=chunk_order,
                    created_at=datetime.utcnow()
                )
                chunks.append(chunk)

            # Move start position with overlap
            next_start = chunk_end - overlap
            if next_start <= start:  # Avoid infinite loop if overlap is too large
                start = end
            else:
                start = next_start

            chunk_order += 1

        logger.info(f"Document chunked into {len(chunks)} chunks")
        return chunks

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """
        Generate embeddings for document chunks using Google Gemini.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not chunks:
            return []

        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings using Gemini service
        embeddings = self.gemini_service.generate_embeddings(texts, model=self.settings.gemini_embedding_model)

        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding_vector = embedding

        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embeddings

    def store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Store document chunks in Qdrant vector store.

        Args:
            chunks: List of DocumentChunk objects with embeddings

        Returns:
            bool: True if storage was successful
        """
        if not chunks:
            logger.warning("No chunks to store")
            return True

        # Prepare data for Qdrant
        texts = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            if not chunk.embedding_vector:
                logger.error(f"Chunk {chunk.id} has no embedding vector")
                continue

            texts.append(chunk.content)
            embeddings.append(chunk.embedding_vector)
            metadatas.append({
                "source_module": chunk.source_module,
                "source_chapter": chunk.source_chapter,
                "source_section": chunk.source_section,
                "chunk_order": chunk.chunk_order,
                "id": chunk.id
            })

        if not texts:
            logger.error("No chunks with embeddings to store")
            return False

        # Store in Qdrant
        try:
            # Using the Qdrant service's upsert functionality
            points = []
            from qdrant_client.http import models
            import uuid

            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            "metadata": metadata
                        }
                    )
                )

            self.qdrant_service.client.upsert(
                collection_name=self.qdrant_service.collection_name,
                points=points
            )

            # Update chunks with their Qdrant vector IDs
            for chunk, point in zip(chunks, points):
                chunk.vector_id = point.id

            logger.info(f"Stored {len(points)} chunks in Qdrant")
            return True
        except Exception as e:
            logger.error(f"Failed to store chunks in Qdrant: {e}")
            return False

    def ingest_document(
        self,
        content: str,
        source_module: str,
        source_chapter: str,
        source_section: Optional[str] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Complete document ingestion pipeline: chunk, embed, and store.

        Args:
            content: The document content to ingest
            source_module: Module where this content belongs
            source_chapter: Chapter where this content belongs
            source_section: Section where this content belongs
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            Dict with ingestion results
        """
        start_time = time.time()

        try:
            # Step 1: Chunk the document
            chunks = self.chunk_document(
                content=content,
                source_module=source_module,
                source_chapter=source_chapter,
                source_section=source_section,
                chunk_size=chunk_size,
                overlap=overlap
            )

            if not chunks:
                logger.warning("No chunks created from document")
                return {
                    "success": False,
                    "message": "No content to ingest",
                    "chunks_created": 0,
                    "processing_time": time.time() - start_time
                }

            # Step 2: Generate embeddings
            self.generate_embeddings(chunks)

            # Step 3: Store in Qdrant
            storage_success = self.store_chunks(chunks)

            processing_time = time.time() - start_time

            if storage_success:
                logger.info(f"Document ingestion completed successfully in {processing_time:.2f}s")
                return {
                    "success": True,
                    "message": "Document ingested successfully",
                    "chunks_created": len(chunks),
                    "processing_time": processing_time,
                    "document_id": str(uuid4())  # Generate a document ID
                }
            else:
                logger.error("Document ingestion failed during storage phase")
                return {
                    "success": False,
                    "message": "Failed to store document chunks in vector store",
                    "chunks_created": len(chunks),
                    "processing_time": processing_time
                }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "message": f"Document ingestion failed: {str(e)}",
                "chunks_created": 0,
                "processing_time": processing_time
            }


# Global instance - will be initialized with settings
document_ingestor_service: Optional[DocumentIngestorService] = None


def get_document_ingestor_service() -> Optional[DocumentIngestorService]:
    """
    Get the global document ingestor service instance.

    Returns:
        DocumentIngestorService: The global instance or None if not initialized
    """
    return document_ingestor_service


def init_document_ingestor_service(settings: Settings) -> DocumentIngestorService:
    """
    Initialize the global document ingestor service instance with the provided settings.

    Args:
        settings: Application settings containing configuration

    Returns:
        DocumentIngestorService: The initialized service instance
    """
    global document_ingestor_service
    document_ingestor_service = DocumentIngestorService(settings)
    return document_ingestor_service