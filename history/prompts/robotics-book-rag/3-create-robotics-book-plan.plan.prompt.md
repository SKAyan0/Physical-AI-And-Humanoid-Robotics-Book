---
id: 3
title: Create Robotics Book Plan
stage: plan
date: 2025-12-15
surface: agent
model: Claude Sonnet 4.5
feature: robotics-book-rag
branch: 1-robotics-book-rag
user: user
command: /sp.plan
labels: [plan, robotics, ai, rag, docusaurus, architecture]
links:
  spec: specs/1-robotics-book-rag/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/1-robotics-book-rag/plan.md
  - specs/1-robotics-book-rag/research.md
  - specs/1-robotics-book-rag/data-model.md
  - specs/1-robotics-book-rag/contracts/rag-api.yaml
  - specs/1-robotics-book-rag/quickstart.md
tests:
  - none
---

## Prompt

```text
Create: System architecture (Frontend/Backend split), Book Navigation Structure, RAG Pipeline Design, Integration Testing Plan.

Decisions needing documentation:
- **Backend Hosting:** Selection of platform for FastAPI RAG service (e.g., Render/Railway free tier) since GitHub Pages only supports static content.
- **Indexing Strategy:** Determining trigger for re-indexing book content into Qdrant (e.g., GitHub Action trigger vs. manual script).
- **Context Window:** Optimizing chunk sizes for RAG to balance costs (OpenAI) vs. answer quality for technical content.
- **Chat UI:** UX decision on embedding chat widget (floating button vs. sidebar) and "Select-text-to-ask" interaction flow.

Testing strategy:
- **Site Build:** CI/CD checks for Docusaurus build errors.
- **RAG Accuracy:** "Golden set" of questions for each module to verify retrieval relevance.
- **Code Integrity:** Verification that generated ROS 2/Isaac code snippets are syntactically correct.

Technical details:
- **Approach:** Iterative Spec-Driven Development (Schema -> Code -> Content).
- **Phasing:**
    1.  **Foundation:** Setup Docusaurus skeleton + RAG Backend (FastAPI/Neon/Qdrant) structure.
    2.  **Core Content:** Generate Modules 1 & 2 (ROS 2, Digital Twin) with text and code examples.
    3.  **Advanced Content:** Generate Modules 3 & 4 (Isaac, VLA) + Capstone definition.
    4.  **Integration:** Connect Frontend Chat Widget to Backend API; Implement "Ask about selection" feature.
- **Standards:**
    -   Code: PEP 8 (Python), Standard Docusaurus Markdown.
    -   Infrastructure: Free-tier compatible configuration (Neon/Qdrant).
```

## Response snapshot

Completed the implementation planning for the Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot. Created the following artifacts:
- plan.md: System architecture with frontend/backend split, technical context, and project structure
- research.md: Resolved all key decisions including backend hosting (Render.com), indexing strategy (GitHub Actions), context window optimization (512-token chunks), and chat UI design (floating widget)
- data-model.md: Defined key entities including Book Content, Content Chunk, User Query, RAG Response, and Code Example with relationships
- contracts/rag-api.yaml: OpenAPI specification for the RAG chatbot API with endpoints for querying, selection-based queries, and content search
- quickstart.md: Setup instructions for local development, content structure, testing, and deployment

The plan adheres to all constitutional principles including Spec-Driven Development, Pedagogical Clarity, Technical Integrity, Embodied Intelligence, Curriculum Adherence, and Cost-Conscious Architecture.

## Outcome

- ✅ Impact: Complete implementation plan with architecture decisions and API contracts
- 🧪 Tests: none
- 📁 Files: Multiple planning artifacts created in specs/1-robotics-book-rag/
- 🔁 Next prompts: Proceed to task generation with /sp.tasks
- 🧠 Reflection: Plan addresses all architectural decisions and provides clear path for implementation

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - All planning requirements met
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Generate implementation tasks with /sp.tasks