import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.config import Settings

logger = logging.getLogger(__name__)


class QdrantClientService:
    """
    Service class for managing Qdrant vector store connections and operations.
    Handles collection management, document ingestion, and vector search operations.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the Qdrant client service with configuration from settings.

        Args:
            settings: Application settings containing Qdrant configuration
        """
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10.0
        )
        self.collection_name = settings.qdrant_collection_name

    def connect(self) -> bool:
        """
        Test the connection to the Qdrant server.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Test connection by getting collections
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant server: {e}")
            return False

    def ensure_collection_exists(
        self,
        vector_size: int = 768,  # Default Gemini embedding size (text-embedding-004)
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Ensure the required collection exists in Qdrant, create if it doesn't.

        Args:
            vector_size: Size of the embedding vectors
            distance: Distance metric for similarity search

        Returns:
            bool: True if collection exists or was created successfully
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if self.collection_name not in collection_names:
                # Create the collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")

            return True
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            return False

    def close(self):
        """
        Close the connection to the Qdrant server.
        """
        if hasattr(self.client, 'close'):
            self.client.close()
            logger.info("Qdrant client connection closed")


# Global instance - will be initialized with settings
qdrant_service: Optional[QdrantClientService] = None


def get_qdrant_service() -> Optional[QdrantClientService]:
    """
    Get the global Qdrant service instance.

    Returns:
        QdrantClientService: The global instance or None if not initialized
    """
    return qdrant_service


def init_qdrant_service(settings: Settings) -> QdrantClientService:
    """
    Initialize the global Qdrant service instance with the provided settings.

    Args:
        settings: Application settings containing Qdrant configuration

    Returns:
        QdrantClientService: The initialized service instance
    """
    global qdrant_service
    qdrant_service = QdrantClientService(settings)
    return qdrant_service