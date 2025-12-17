import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.config import Settings
from src.models.rag import RAGQuery, RAGResult, DocumentChunk
from src.services.gemini_client import get_gemini_service
from src.services.qdrant_client import get_qdrant_service


logger = logging.getLogger(__name__)


class RAGService:
    """
    Service class for performing RAG (Retrieval-Augmented Generation) queries.
    Handles vector search, result ranking, and context preparation for LLMs.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the RAG service with configuration.

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

    def query(
        self,
        query_text: str,
        selected_text: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """
        Perform a RAG query to find relevant content based on the query text.

        Args:
            query_text: The main query text
            selected_text: Additional context from user selection (optional)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold for results
            filters: Additional filters to apply to the search

        Returns:
            RAGResult containing the query results
        """
        start_time = datetime.utcnow()

        try:
            # Combine query text with selected text if provided
            full_query = query_text
            if selected_text:
                full_query = f"Context: {selected_text}\n\nQuestion: {query_text}"

            # Generate embedding for the query using Gemini's embedding service
            query_embedding = self.gemini_service.generate_embeddings([full_query], model=self.settings.gemini_embedding_model)[0]

            # Prepare search filters for Qdrant
            search_filter = None
            if filters:
                from qdrant_client.http import models
                filter_conditions = []

                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=value)
                            )
                        )
                    elif isinstance(value, list):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchAny(any=value)
                            )
                        )

                if filter_conditions:
                    search_filter = models.Filter(must=filter_conditions)

            # Perform vector search in Qdrant
            search_results = self.qdrant_service.client.search(
                collection_name=self.qdrant_service.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k * 2  # Get more results to apply similarity filter
            )

            # Filter results by minimum similarity and convert to DocumentChunk objects
            filtered_results = []
            scores = []

            for result in search_results:
                if result.score >= min_similarity:
                    # Extract metadata
                    payload = result.payload
                    metadata = payload.get("metadata", {})

                    # Create DocumentChunk from the result
                    chunk = DocumentChunk(
                        id=metadata.get("id", result.id),
                        content=payload.get("text", ""),
                        source_module=metadata.get("source_module", ""),
                        source_chapter=metadata.get("source_chapter", ""),
                        source_section=metadata.get("source_section"),
                        chunk_order=metadata.get("chunk_order", 0),
                        created_at=datetime.utcnow()
                    )

                    filtered_results.append(chunk)
                    scores.append(result.score)

            # Limit to top_k results after filtering
            if len(filtered_results) > top_k:
                filtered_results = filtered_results[:top_k]
                scores = scores[:top_k]

            # Create and return RAG result
            rag_result = RAGResult(
                query=query_text,
                results=filtered_results,
                scores=scores,
                retrieved_count=len(filtered_results),
                processed_at=datetime.utcnow()
            )

            logger.info(f"RAG query completed: {len(filtered_results)} results found in {(rag_result.processed_at - start_time).total_seconds():.2f}s")
            return rag_result

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise

    def query_with_context(
        self,
        query_text: str,
        context_text: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> RAGResult:
        """
        Perform a RAG query with additional context (e.g., from selected text).

        Args:
            query_text: The main query text
            context_text: Additional context to include in the search
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            RAGResult containing the query results
        """
        return self.query(
            query_text=query_text,
            selected_text=context_text,
            top_k=top_k,
            min_similarity=min_similarity
        )

    def get_related_content(
        self,
        content_id: str,
        top_k: int = 5
    ) -> RAGResult:
        """
        Find content related to a specific document chunk.

        Args:
            content_id: ID of the content to find related content for
            top_k: Number of related results to return

        Returns:
            RAGResult containing related content
        """
        # This would require storing content IDs in the vector database
        # For now, we'll do a simple implementation based on semantic similarity
        # to content with the same source_module and source_chapter

        # Get the original content first
        search_results = self.qdrant_service.client.search(
            collection_name=self.qdrant_service.collection_name,
            query_filter=None,  # We'll implement proper filtering later
            limit=1
        )

        if not search_results:
            return RAGResult(
                query=f"Related to content ID {content_id}",
                results=[],
                scores=[],
                retrieved_count=0,
                processed_at=datetime.utcnow()
            )

        # In a real implementation, we would search for similar content
        # based on the content_id, but for now we'll return a basic result
        # This is a placeholder implementation
        return RAGResult(
            query=f"Related to content ID {content_id}",
            results=[],
            scores=[],
            retrieved_count=0,
            processed_at=datetime.utcnow(),
            metadata={"message": "Related content functionality not fully implemented yet"}
        )

    def enrich_query_with_knowledge(
        self,
        query: RAGQuery
    ) -> RAGResult:
        """
        Enrich a query with knowledge from the RAG system.

        Args:
            query: RAGQuery object containing query parameters

        Returns:
            RAGResult with enriched results
        """
        return self.query(
            query_text=query.query_text,
            selected_text=query.selected_text,
            top_k=query.top_k,
            min_similarity=query.min_similarity,
            filters=query.filters
        )


# Global instance - will be initialized with settings
rag_service: Optional[RAGService] = None


def get_rag_service() -> Optional[RAGService]:
    """
    Get the global RAG service instance.

    Returns:
        RAGService: The global instance or None if not initialized
    """
    return rag_service


def init_rag_service(settings: Settings) -> RAGService:
    """
    Initialize the global RAG service instance with the provided settings.

    Args:
        settings: Application settings containing configuration

    Returns:
        RAGService: The initialized service instance
    """
    global rag_service
    rag_service = RAGService(settings)
    return rag_service