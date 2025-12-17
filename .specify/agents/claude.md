# Agent Context: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

## Project Overview
You are assisting with the development of an integrated RAG (Retrieval-Augmented Generation) chatbot for a Physical AI & Humanoid Robotics educational platform. The system combines a Docusaurus frontend with a FastAPI backend to provide contextual answers based on book content with zero hallucination.

## Technical Stack
- **Backend**: FastAPI with Python 3.11
- **Vector Store**: Qdrant Cloud (free tier)
- **Metadata DB**: Neon Serverless Postgres
- **AI SDK**: OpenAI Agents/ChatKit SDK
- **Frontend**: Docusaurus with React components
- **Language Models**: OpenAI GPT models for generation, Ada-002 for embeddings

## Architecture
- **Frontend**: Static Docusaurus site hosted on GitHub Pages
- **Backend**: FastAPI service with async support for streaming responses
- **RAG Pipeline**: Content ingestion → Embedding generation → Vector storage → Retrieval → Generation
- **Session Management**: User sessions with conversation history stored in Neon Postgres

## Key Features
1. **Contextual Q&A**: Answer questions based on book content with zero hallucination
2. **Selected Text Queries**: "Ask about selected text" functionality
3. **Cross-Module Navigation**: Connect concepts across different book modules
4. **Session Continuity**: Maintain conversation context across interactions

## Code Standards
- Python: PEP 8 compliance with type hints
- FastAPI: Dependency injection, Pydantic models, async/await patterns
- React: Component-based architecture with hooks
- Docusaurus: Plugin architecture, MDX support

## Content Structure
The book is organized in 4 modules:
1. Module 1: The Robotic Nervous System (ROS 2)
2. Module 2: The Digital Twin (Gazebo & Unity)
3. Module 3: The AI-Robot Brain (NVIDIA Isaac™)
4. Module 4: Vision-Language-Action (VLA) & Capstone

## Critical Requirements
- Zero hallucination in responses (strictly grounded in book content)
- Fast response times (sub-3s for complex queries)
- Proper handling of selected text queries
- Secure API key management via environment variables
- GDPR-compliant data handling

## Files and Directories
- `backend/src/`: FastAPI application code
- `backend/src/models/`: Pydantic models for data validation
- `backend/src/services/`: Business logic and RAG implementation
- `backend/src/api/`: API routes and endpoints
- `docs/`: Docusaurus documentation site
- `docs/src/components/`: Custom React components including chat widget
- `specs/`: Feature specifications and plans

## Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `QDRANT_URL`: Qdrant vector store URL
- `QDRANT_API_KEY`: Qdrant API key (if using cloud)
- `NEON_DATABASE_URL`: Neon Postgres connection string
- `DEBUG`: Debug mode flag

## Common Tasks
- Add new book content: Create markdown file in appropriate module, run indexing script
- Update RAG logic: Modify services in `backend/src/services/rag_service.py`
- Modify chat UI: Update components in `docs/src/components/RagChatWidget/`
- Extend API: Add endpoints in `backend/src/api/chat.py`

## Quality Assurance
- All responses must be grounded in book content (no hallucination)
- Selected text queries must use the specific context provided
- Error handling should be graceful with informative messages
- Performance should maintain sub-second response times for simple queries