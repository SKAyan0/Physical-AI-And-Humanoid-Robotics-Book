# Data Model: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

## Core Entities

### Book Content Chunk
- **id**: string (UUID)
- **content_id**: string (foreign key to Book Content)
- **chunk_text**: string (the actual content chunk)
- **chunk_metadata**: object (module, chapter, section, page_reference)
- **vector_id**: string (ID in Qdrant vector store)
- **created_at**: datetime
- **updated_at**: datetime

**Validation rules**: All fields required. Vector_id must be unique. Content_id must reference existing Book Content.

### User Session
- **id**: string (UUID)
- **user_id**: string (optional, for anonymous sessions)
- **created_at**: datetime
- **updated_at**: datetime
- **metadata**: object (browser info, device info, etc.)

**Validation rules**: ID required, created_at must be in the past.

### Chat Log
- **id**: string (UUID)
- **session_id**: string (foreign key to User Session)
- **query_text**: string
- **response_text**: string
- **query_timestamp**: datetime
- **response_timestamp**: datetime
- **retrieved_chunks**: array of strings (IDs of content chunks used)
- **confidence_score**: number (0-1)

**Validation rules**: All fields required. Session_id must reference existing session. Confidence_score must be between 0 and 1.

### Book Content
- **id**: string (UUID)
- **title**: string
- **module**: string (Module 1-4)
- **chapter**: string
- **section**: string
- **content**: string (markdown)
- **created_at**: datetime
- **updated_at**: datetime
- **version**: string

**Validation rules**: Title, module, chapter, and content required. Module must be one of "Module 1", "Module 2", "Module 3", "Module 4".

### Code Example
- **id**: string (UUID)
- **content_id**: string (foreign key to Book Content)
- **code_snippet**: string
- **language**: string (python, c++, etc.)
- **module**: string (Module 1-4)
- **chapter**: string
- **description**: string
- **created_at**: datetime

**Validation rules**: All fields required. Language must be a valid programming language.

## Relationships

- Book Content (1) → (∞) Book Content Chunk (one content item can have multiple chunks)
- User Session (1) → (∞) Chat Log (one session can have multiple chat logs)
- Book Content (1) → (∞) Code Example (one content item can have multiple code examples)

## State Transitions

### Content Publication State
- Draft → Indexed (when content is indexed into Qdrant)
- Indexed → Archived (when content is outdated but preserved)

### Session State
- Active → Inactive (when session expires or user ends session)
- Active → Paused (when user leaves without ending session explicitly)

### Chat Log Status
- Processing → Complete (when response is generated successfully)
- Processing → Failed (when there's an error generating response)
- Complete → Rated (when user provides feedback on response quality)