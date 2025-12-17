---
id: 2
title: Update Const For RAG Chatbot
stage: constitution
date: 2025-12-15
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution, robotics, ai, rag, docusaurus, security]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

```text
Goal: Update the project constitution to include standards for the Integrated RAG Chatbot.

New Standards to Add:
- Backend Architecture: FastAPI for the API layer with streaming support.
- Database Policy: Neon Serverless Postgres for relational metadata; Qdrant for vector storage.
- AI SDKs: Strict use of OpenAI Agents/ChatKit SDKs.
- RAG Principle: No "hallucination"—responses must be grounded in indexed book content.
- Selection Feature: Support for "Ask about selected text" must be a core UI principle.
- Security: All API keys must be stored in .env and never committed to GitHub.

Update the existing constitution to reflect these as the "rules of the game" for all upcoming work.
```

## Response snapshot

Updated the project constitution to include new standards for the Integrated RAG Chatbot. Added specific requirements for backend architecture (FastAPI with streaming), database policy (Neon for metadata, Qdrant for vectors), AI SDKs (OpenAI Agents/ChatKit), RAG principle (no hallucination), UI selection feature (ask about selected text), and security (API keys in .env files). Enhanced the Technical Integrity principle to include FastAPI requirements and added a new NON-NEGOTIABLE Security Principle. The version was bumped from 1.0.0 to 1.1.0 to reflect the addition of new principles and expansion of existing ones.

## Outcome

- ✅ Impact: Constitution updated with RAG chatbot standards and security requirements
- 🧪 Tests: none
- 📁 Files: .specify/memory/constitution.md updated with new principles
- 🔁 Next prompts: Proceed with implementation following new standards
- 🧠 Reflection: Constitution now properly governs RAG chatbot development with clear anti-hallucination and security requirements

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - All new standards incorporated into constitution
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin RAG chatbot implementation following new standards