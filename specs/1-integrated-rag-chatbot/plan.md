# Implementation Plan: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

**Branch**: `1-integrated-rag-chatbot` | **Date**: 2025-12-15 | **Spec**: [link](../spec.md)
**Input**: Feature specification from `/specs/1-integrated-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature implements an integrated RAG (Retrieval-Augmented Generation) chatbot for the Physical AI & Humanoid Robotics Book. The system will allow readers to ask questions about the book content and receive accurate answers grounded in the indexed material. The chatbot will support "Ask about selected text" functionality, allowing users to highlight content and ask specific questions about it. The backend uses FastAPI with Qdrant for vector storage and Neon Postgres for metadata, ensuring zero hallucination in responses.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend
**Primary Dependencies**: FastAPI, OpenAI Agents/ChatKit SDK, Qdrant, Neon Postgres, Docusaurus, React
**Storage**: Qdrant (vector store), Neon Postgres (metadata)
**Testing**: pytest, Jest for frontend components
**Target Platform**: GitHub Pages (frontend), Cloud platform for backend (Render/Railway free tier)
**Project Type**: Web application with static frontend and dynamic backend
**Performance Goals**: <1 second response time for chat queries, <5 second page load time
**Constraints**: Must fit within free tier limits of Qdrant and Neon; GitHub Pages only supports static content
**Scale/Scope**: Educational platform for students and developers learning robotics concepts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [X] Spec-Driven Development: Following spec → plan → tasks → implementation workflow
- [X] Pedagogical Clarity: Prioritizing educational value and clarity in all content
- [X] Technical Integrity: Using specified tech stack (FastAPI, Neon, Qdrant) with accurate implementations
- [X] Embodied Intelligence: Connecting theoretical concepts to practical robotic applications
- [X] Curriculum Adherence: Following 4-module structure (ROS 2 → Digital Twin → Isaac → VLA)
- [X] Cost-Conscious Architecture: Using free tier services where possible
- [X] RAG Principle: Ensuring zero hallucination in chatbot responses
- [X] UI Selection Feature Principle: Supporting "Ask about selected text" functionality
- [X] Security Principle: Managing API keys securely via .env files

## Project Structure

### Documentation (this feature)

```text
specs/1-integrated-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── user_session.py
│   │   ├── chat_log.py
│   │   ├── book_content_chunk.py
│   │   ├── qdrant_vector_record.py
│   │   └── session_metadata.py
│   ├── services/
│   │   ├── rag_service.py
│   │   ├── ingestion_service.py
│   │   ├── session_service.py
│   │   ├── search_service.py
│   │   ├── qdrant_client.py
│   │   └── database.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── content.py
│   │   └── sessions.py
│   └── main.py
└── tests/

frontend/
└── docs/
    ├── docs/
    │   ├── module1/
    │   ├── module2/
    │   ├── module3/
    │   ├── module4/
    │   └── ...
    ├── src/
    │   └── components/
    │       └── RagChatWidget/
    │           ├── index.js
    │           ├── SelectionListener.js
    │           └── api.js
    ├── docusaurus.config.js
    ├── package.json
    └── ...
```

**Structure Decision**: Web application with separate backend API and Docusaurus frontend. Backend hosts RAG functionality while frontend provides the educational content and chat interface.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| - | - | - |