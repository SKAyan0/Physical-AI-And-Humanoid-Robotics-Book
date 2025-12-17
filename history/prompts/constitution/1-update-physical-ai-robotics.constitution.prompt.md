---
id: 1
title: Update Physical AI Robotics Constitution
stage: constitution
date: 2025-12-15
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution, robotics, ai, rag, docusaurus]
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
Project: Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot

Core principles:
- Spec-Driven Development: Strict adherence to Spec-Kit Plus workflows for generation.
- Pedagogical Clarity: Complex robotics concepts (ROS 2, Isaac) must be accessible to students.
- Technical Integrity: accurate implementation of the specific tech stack (FastAPI, Neon, Qdrant).
- Embodied Intelligence: Content must bridge the gap between digital algorithms and physical deployment.

Key standards:
- Documentation Framework: Docusaurus (deployed via GitHub Pages).
- RAG Stack Compliance: OpenAI Agents/ChatKit (LLM), FastAPI (Backend), Neon Serverless Postgres (DB), Qdrant Cloud Free Tier (Vector Store).
- Curriculum Coverage: Must strictly follow the 4-Module structure (ROS 2 -> Digital Twin -> Isaac -> VLA).
- Coding Standards: PEP 8 for Python (rclpy/FastAPI); modular URDF definitions.
- Feature Requirement: Chatbot must support answering questions based on user-selected text.

Constraints:
- Tools: Must use Claude Code and Spec-Kit Plus exclusively for generation.
- Deployment: Fully automated deployment to GitHub Pages.
- Cost constraints: Architecture must fit within Qdrant/Neon free tiers where possible.
- Content Scope: From "Robotic Nervous System" to "Autonomous Humanoid Capstone".

Success criteria:
- Docusaurus site is live and accessible on GitHub Pages.
- Embedded RAG chatbot successfully retrieves context from the book to answer queries.
- Module 4 (VLA) examples successfully demonstrate Voice-to-Action logic.
- Zero "hallucinated" API calls (code must match actual SDK documentation for OpenAI/Isaac).
```

## Response snapshot

Updated the project constitution to reflect the Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot project. The constitution now includes six core principles: Spec-Driven Development (NON-NEGOTIABLE), Pedagogical Clarity, Technical Integrity, Embodied Intelligence, Curriculum Adherence, and Cost-Conscious Architecture. The document also defines Technical Standards, Development Workflow, and Governance sections aligned with the project requirements. A Sync Impact Report was added to document the changes and review of dependent templates.

## Outcome

- ✅ Impact: New constitution created that aligns with Physical AI & Humanoid Robotics project requirements
- 🧪 Tests: none
- 📁 Files: .specify/memory/constitution.md updated with project-specific principles
- 🔁 Next prompts: Plan and implement the RAG chatbot functionality
- 🧠 Reflection: Constitution now properly guides development of the robotics book project with focus on educational value and technical accuracy

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - All project principles properly documented
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin specification of the RAG chatbot module