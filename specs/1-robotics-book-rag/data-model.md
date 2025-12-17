# Data Model: Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot

## Entities

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

**Validation rules**: Title, module, chapter, and content are required fields. Module must be one of "Module 1", "Module 2", "Module 3", "Module 4".

### Content Chunk
- **id**: string (UUID)
- **content_id**: string (foreign key to Book Content)
- **chunk_text**: string
- **chunk_metadata**: object (module, chapter, section, page_reference)
- **vector_id**: string (ID in Qdrant)
- **created_at**: datetime

**Validation rules**: All fields are required. Vector_id must be unique.

### User Query
- **id**: string (UUID)
- **query_text**: string
- **user_context**: object (selected_text, referring_page, timestamp)
- **created_at**: datetime

**Validation rules**: Query_text is required.

### RAG Response
- **id**: string (UUID)
- **query_id**: string (foreign key to User Query)
- **response_text**: string
- **retrieved_chunks**: array of strings (chunk IDs)
- **confidence_score**: number (0-1)
- **created_at**: datetime

**Validation rules**: All fields are required. Confidence_score must be between 0 and 1.

### Code Example
- **id**: string (UUID)
- **content_id**: string (foreign key to Book Content)
- **code_snippet**: string
- **language**: string (python, c++, etc.)
- **module**: string (Module 1-4)
- **chapter**: string
- **description**: string
- **created_at**: datetime

**Validation rules**: All fields are required. Language must be a valid programming language.

## Relationships

- Book Content 1 → * Content Chunk (one content item can have multiple chunks)
- User Query 1 → 1 RAG Response (one query generates one response)
- Book Content 1 → * Code Example (one content item can have multiple code examples)

## State Transitions

### Content Publication
- Draft → Published (when content is ready for learners)
- Published → Archived (when content is outdated but preserved for reference)

### Query Processing
- Received → Processing (when query is received by backend)
- Processing → Completed (when response is generated and returned to user)