---
id: 1
title: Create RAG Chatbot Spec
stage: spec
date: 2025-12-15
surface: agent
model: Claude Sonnet 4.5
feature: integrated-rag-chatbot
branch: 1-integrated-rag-chatbot
user: user
command: /sp.specify
labels: [spec, robotics, ai, rag, docusaurus, chatbot]
links:
  spec: specs/1-integrated-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/1-integrated-rag-chatbot/spec.md
  - specs/1-integrated-rag-chatbot/checklists/requirements.md
tests:
  - none
---

## Prompt

```text
Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

Target audience: Readers of the Physical AI book who need contextual help and "Ask about selection" capabilities.
Focus: Implementing a Retrieval-Augmented Generation (RAG) system using a modern AI-native stack.

Success criteria:
- **Functional RAG:** Chatbot successfully retrieves context from Qdrant to answer book-related queries.
- **Context-Awareness:** Implements "Ask based on selected text" feature via ChatKit.
- **Data Persistence:** User session metadata and chat logs stored in Neon Postgres.
- **High Performance:** FastAPI backend ensures low-latency streaming responses using OpenAI Agents SDK.

Constraints:
- **Backend:** FastAPI, Python 3.10+, OpenAI Agents/ChatKit SDKs.
- **Database:** Neon Serverless Postgres (Metadata) and Qdrant Cloud Free Tier (Vector Store).
- **Frontend:** Integrated React components in Docusaurus using @openai/chatkit-react.
- **Security:** API keys managed via .env; CORS configured for the Docusaurus domain.

Specific Features to Build:
- **Ingestion Pipeline:** A script to chunk Docusaurus Markdown files, generate OpenAI embeddings, and upsert to Qdrant.
- **Session API:** FastAPI endpoints to manage ChatKit sessions and handle tool calls.
- **Hybrid Search:** Combine vector search (Qdrant) with metadata filtering (Neon) where applicable.
- **Selection Listener:** A Docusaurus/React hook that captures user text selections and passes them to the ChatKit context.

Not building:
- Support for non-Markdown document types (PDF, Docx).
- User authentication/login (session-based only).
- Multi-modal RAG (text only for this requirement).
- Extensive UI customization beyond the ChatKit standard widget.
```

## Response snapshot

Created comprehensive specification for the Integrated RAG Chatbot feature for the Physical AI & Humanoid Robotics Book. The specification includes three user stories covering contextual learning assistance (P1), book navigation and discovery (P2), and session continuity (P3). It defines 10 functional requirements covering RAG functionality, selected text feature, data persistence, streaming responses, ingestion pipeline, session management, hybrid search, and selection listener. The success criteria include measurable outcomes for accuracy, latency, reliability, and functionality. A quality checklist was created and all items validated as complete.

## Outcome

- ✅ Impact: Complete specification for RAG chatbot with clear requirements
- 🧪 Tests: none
- 📁 Files: specs/1-integrated-rag-chatbot/spec.md and checklists/requirements.md created
- 🔁 Next prompts: Proceed to planning phase with /sp.plan
- 🧠 Reflection: Specification properly abstracts implementation details while focusing on user needs and measurable outcomes

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - All specification requirements met
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin architectural planning with /sp.plan