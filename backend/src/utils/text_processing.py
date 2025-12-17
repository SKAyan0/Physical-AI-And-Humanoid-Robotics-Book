"""
Utility functions for text processing in the RAG Chatbot API.
"""
import re
from typing import List, Tuple
from urllib.parse import urlparse


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing line breaks.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize line breaks
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing whitespace
    return text.strip()


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.

    Args:
        text: Input text to extract URLs from

    Returns:
        List of URLs found in the text
    """
    url_pattern = re.compile(
        r'http[s]?://'  # http:// or https://
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract code blocks from text with their language specification.

    Args:
        text: Input text to extract code blocks from

    Returns:
        List of tuples containing (language, code) for each code block
    """
    code_block_pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    matches = code_block_pattern.findall(text)
    return [(lang.strip(), code.strip()) for lang, code in matches]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text to truncate
        max_length: Maximum length of the text
        suffix: Suffix to add to truncated text

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Find a good breaking point (space or punctuation)
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    last_punct = truncated.rfind('.')

    # Break at the latest breaking point that's not too close to the beginning
    break_point = max(last_space, last_punct)
    if break_point > max_length - 100:  # Don't break too early
        truncated = truncated[:break_point + 1]

    return truncated + suffix


def split_text_by_sentences(text: str, max_length: int) -> List[str]:
    """
    Split text into chunks by sentences, ensuring each chunk is no longer than max_length.

    Args:
        text: Input text to split
        max_length: Maximum length of each chunk

    Returns:
        List of text chunks
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk + " " + sentence) <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def remove_markdown_formatting(text: str) -> str:
    """
    Remove basic markdown formatting from text.

    Args:
        text: Input text with markdown formatting

    Returns:
        Text with markdown formatting removed
    """
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove bold and italic
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove code blocks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove links but keep the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove images but keep the alt text
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    # Remove blockquotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

    return text.strip()


def extract_headings(text: str) -> List[Tuple[int, str]]:
    """
    Extract headings from markdown text.

    Args:
        text: Input markdown text

    Returns:
        List of tuples containing (level, heading_text)
    """
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    matches = heading_pattern.findall(text)
    return [(len(h[0]), h[1]) for h in matches]


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the number of tokens in text based on the model.
    This is a rough estimation - for precise token counting, use the appropriate tokenizer for the model.

    Args:
        text: Input text to count tokens for
        model: Model name to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    # Rough estimation: 1 token ~ 4 characters for English text
    # This is a simplified approach - in a real application, use proper tokenizers
    if "gpt-4" in model or "4-turbo" in model:
        # GPT-4 models have slightly different tokenization
        return len(text) // 3
    else:
        # GPT-3.5 and similar models
        return len(text) // 4


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text (tabs, newlines, multiple spaces to single space).

    Args:
        text: Input text to normalize

    Returns:
        Text with normalized whitespace
    """
    # Replace tabs and newlines with spaces
    text = re.sub(r'[\t\n\r]+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    return text.strip()


def detect_language(text: str) -> str:
    """
    Detect the language of text (simplified approach).
    This is a basic implementation - for production use, use a proper language detection library.

    Args:
        text: Input text to detect language for

    Returns:
        Detected language code (simplified)
    """
    # This is a very basic implementation
    # In a real application, use langdetect or similar library
    text = text.lower()[:1000]  # Sample first 1000 characters

    # Count common words for different languages
    english_common = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that']
    count_english = sum(1 for word in english_common if f' {word} ' in text)

    # This is a very simplified approach - just return 'en' for now
    # A real implementation would use a proper language detection library
    return 'en'


def sanitize_input(text: str) -> str:
    """
    Sanitize input text to prevent injection attacks.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    # Remove potentially dangerous characters/sequences
    # This is a basic implementation - for production use, use proper sanitization
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)

    return sanitized


def format_for_display(text: str) -> str:
    """
    Format text for display purposes (clean up formatting).

    Args:
        text: Input text to format

    Returns:
        Formatted text
    """
    # Clean up the text
    text = clean_text(text)
    # Ensure proper sentence spacing
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    # Remove extra spaces after punctuation
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simplified approach).
    This is a basic implementation - for production use, use NLP libraries.

    Args:
        text: Input text to extract keywords from
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of extracted keywords
    """
    # Convert to lowercase and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }

    # Count word frequencies
    word_freq = {}
    for word in words:
        if len(word) > 2 and word not in stop_words:  # Only consider words longer than 2 chars
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]