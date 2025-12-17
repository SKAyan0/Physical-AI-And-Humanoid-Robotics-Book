import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai import GenerativeModel

from src.config import Settings

logger = logging.getLogger(__name__)


class GeminiClientService:
    """
    Service class for managing Google Gemini API interactions.
    Handles configuration, client initialization, and common AI operations.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the Gemini client service with configuration from settings.

        Args:
            settings: Application settings containing Gemini configuration
        """
        self.settings = settings

        # Configure the Gemini API
        genai.configure(api_key=settings.gemini_api_key)

        # Initialize the generative model
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction="You are a helpful assistant for a Physical AI & Humanoid Robotics Book. Answer questions based on the provided context and book content."
        )

        self.default_model = settings.gemini_model

    def generate_embeddings(self, texts: List[str], model: str = "text-embedding-004") -> List[List[float]]:
        """
        Generate embeddings for the given texts using Google's embedding API.
        Note: Google's embedding API requires a separate client setup.

        Args:
            texts: List of texts to generate embeddings for
            model: Embedding model to use (default: text-embedding-004)

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            # For Google's embedding API, we need to use the embedding module
            import google.generativeai as genai

            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embeddings.append(result['embedding'])

            logger.info(f"Generated embeddings for {len(texts)} texts using model: {model}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Generate a chat completion using Google Gemini's API.

        Args:
            messages: List of messages in the conversation (with 'role' and 'content' keys)
            model: Model to use for completion (uses default if not provided)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            Completion response from Gemini API
        """
        model_name = model or self.default_model

        try:
            # Convert the messages to the format expected by Gemini
            # Gemini expects a different format, so we need to convert from standard chat format
            gemini_history = []
            for msg in messages:
                role = 'model' if msg['role'] == 'assistant' else 'user'
                gemini_history.append({
                    "role": role,
                    "parts": [msg['content']]
                })

            # Create the chat instance
            chat = self.model.start_chat(history=gemini_history[:-1])  # Exclude the last message as it's the new one

            # Get the last user message to send
            last_user_message = messages[-1]['content'] if messages[-1]['role'] == 'user' else messages[-2]['content']

            # Generate response
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            response = chat.send_message(
                last_user_message,
                generation_config=generation_config,
                **kwargs
            )

            logger.info(f"Generated chat completion using model: {model_name}")
            return response
        except Exception as e:
            logger.error(f"Failed to generate chat completion: {e}")
            raise

    def generate_text(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text based on a prompt using Google Gemini's API.

        Args:
            prompt: Input prompt to generate text from
            model: Model to use for generation (uses default if not provided)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            Generated text response from Gemini API
        """
        model_name = model or self.default_model

        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                **kwargs
            )

            logger.info(f"Generated text completion using model: {model_name}")
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate text completion: {e}")
            raise

    def validate_api_key(self) -> bool:
        """
        Validate the Gemini API key by making a simple test request.

        Returns:
            bool: True if API key is valid, False otherwise
        """
        try:
            # Make a simple test request
            response = self.model.generate_content(
                "Test",
                generation_config={
                    "max_output_tokens": 10,
                }
            )

            logger.info("Gemini API key validation successful")
            return True
        except Exception as e:
            logger.error(f"Gemini API key validation failed: {e}")
            return False


# Global instance - will be initialized with settings
gemini_service: Optional[GeminiClientService] = None


def get_gemini_service() -> Optional[GeminiClientService]:
    """
    Get the global Gemini service instance.

    Returns:
        GeminiClientService: The global instance or None if not initialized
    """
    return gemini_service


def init_gemini_service(settings: Settings) -> GeminiClientService:
    """
    Initialize the global Gemini service instance with the provided settings.

    Args:
        settings: Application settings containing Gemini configuration

    Returns:
        GeminiClientService: The initialized service instance
    """
    global gemini_service
    gemini_service = GeminiClientService(settings)
    return gemini_service