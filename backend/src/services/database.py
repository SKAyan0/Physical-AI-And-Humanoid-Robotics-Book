import logging
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import Settings

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

# Global variables for engine and session
engine: Optional[object] = None
SessionLocal: Optional[object] = None


class DatabaseService:
    """
    Service class for managing Neon Postgres database connections and operations.
    Handles connection initialization, session management, and basic database operations.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the database service with configuration from settings.

        Args:
            settings: Application settings containing database configuration
        """
        self.settings = settings
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self):
        """
        Initialize the SQLAlchemy engine with connection pooling and settings.
        """
        try:
            # Create the engine with connection pooling and other optimizations
            self.engine = create_engine(
                self.settings.database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes
                echo=False           # Set to True for SQL debugging
            )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def connect(self) -> bool:
        """
        Test the connection to the database by creating tables and getting a session.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Create all tables defined in models
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified successfully")

            # Test connection by getting a session
            with self.SessionLocal() as session:
                # This will raise an exception if connection fails
                session.execute("SELECT 1")

            logger.info("Successfully connected to Neon Postgres database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def get_session(self):
        """
        Get a database session for use in operations.

        Yields:
            Session: Database session for operations
        """
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def close(self):
        """
        Close the database engine connection.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine connection closed")


# Global instance - will be initialized with settings
db_service: Optional[DatabaseService] = None


def get_db_service() -> Optional[DatabaseService]:
    """
    Get the global database service instance.

    Returns:
        DatabaseService: The global instance or None if not initialized
    """
    return db_service


def init_db_service(settings: Settings) -> DatabaseService:
    """
    Initialize the global database service instance with the provided settings.

    Args:
        settings: Application settings containing database configuration

    Returns:
        DatabaseService: The initialized service instance
    """
    global db_service
    db_service = DatabaseService(settings)
    return db_service


def get_db():
    """
    Dependency function for FastAPI to get database sessions.

    Yields:
        Session: Database session for API endpoints
    """
    service = get_db_service()
    if service:
        for session in service.get_session():
            yield session
    else:
        raise RuntimeError("Database service not initialized")