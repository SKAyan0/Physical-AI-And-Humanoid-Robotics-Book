<!--
SYNC IMPACT REPORT
Version change: 1.0.0 → 1.1.0
List of modified principles: Technical Integrity (expanded), Added RAG Principle, Added Security Principle
Added sections: RAG Principle, Security Principle
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated to reflect new principles
  - .specify/templates/spec-template.md ✅ updated to reflect new principles
  - .specify/templates/tasks-template.md ✅ updated to reflect new principles
  - .specify/templates/commands/*.md ✅ reviewed for outdated references
Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot Constitution

## Core Principles

### Spec-Driven Development (NON-NEGOTIABLE)
Strict adherence to Spec-Kit Plus workflows for generation. All features must be fully specified before implementation begins, following the spec → plan → tasks → implementation workflow. Every change must be traced back to a requirement in the specification.

### Pedagogical Clarity
Complex robotics concepts (ROS 2, Isaac) must be accessible to students. All code, documentation, and examples must prioritize clarity and educational value over performance optimizations when there's a conflict. Concepts must be explained with practical examples that bridge theory to practice.

### Technical Integrity
Accurate implementation of the specific tech stack (FastAPI, Neon, Qdrant). All code must match actual SDK documentation for OpenAI/Isaac; zero "hallucinated" API calls allowed. Implementation must strictly follow the RAG Stack Compliance requirements: OpenAI Agents/ChatKit (LLM), FastAPI (Backend), Neon Serverless Postgres (DB), Qdrant Cloud Free Tier (Vector Store). Backend architecture MUST use FastAPI for the API layer with streaming support.

### Embodied Intelligence
Content must bridge the gap between digital algorithms and physical deployment. All theoretical concepts must connect to real-world robotic applications, demonstrating how abstract algorithms translate to embodied behaviors in physical systems.

### Curriculum Adherence
Must strictly follow the 4-Module structure (ROS 2 -> Digital Twin -> Isaac -> VLA). Each module builds upon the previous one, maintaining consistent terminology, code patterns, and pedagogical approach throughout the curriculum.

### Cost-Conscious Architecture
Architecture must fit within Qdrant/Neon free tiers where possible. All technical decisions must consider cost implications and provide alternatives that work within free tier limitations while maintaining educational effectiveness.

### RAG Principle (NON-NEGOTIABLE)
Responses from the integrated RAG chatbot MUST be grounded in indexed book content with zero "hallucination". All answers must be verifiable against the source material and never generate content that is not supported by the book content. The system must clearly indicate when information is uncertain or not available in the indexed content.

### UI Selection Feature Principle
The chatbot interface MUST support the "Ask about selected text" feature as a core UI principle. Users must be able to select text in the book content and ask questions specifically about that selected text, with the RAG system using that specific context to generate accurate responses.

## Technical Standards

Technology stack requirements:
- Backend Architecture: FastAPI for the API layer with streaming support
- Database Policy: Neon Serverless Postgres for relational metadata; Qdrant for vector storage
- AI SDKs: Strict use of OpenAI Agents/ChatKit SDKs
- Docusaurus for documentation framework (deployed via GitHub Pages)
- PEP 8 for Python (rclpy/FastAPI)
- Modular URDF definitions for robotics components
- Feature Requirement: Chatbot must support answering questions based on user-selected text
- Content Scope: From "Robotic Nervous System" to "Autonomous Humanoid Capstone"

## Development Workflow

Development must strictly follow Spec-Kit Plus workflows using Claude Code and Spec-Kit Plus exclusively for generation. All implementations require specification-first approach. Code reviews must verify compliance with coding standards, architectural decisions, and curriculum progression. Automated deployment to GitHub Pages required. Each module must include working examples that demonstrate the concepts taught. All API keys must be stored in .env and never committed to GitHub.

## Security Principle (NON-NEGOTIABLE)

All API keys, secrets, and sensitive configuration must be stored in .env files and never committed to the Git repository. The development workflow MUST include validation that no sensitive credentials are exposed in committed code. Environment-specific configurations must be managed externally and never stored in version control.

## Governance

This constitution supersedes all other development practices. All amendments require documentation of impact on existing curriculum and implementation. Compliance with all principles must be verified during code reviews. Use CLAUDE.md for runtime development guidance.

**Version**: 1.1.0 | **Ratified**: 2025-12-15 | **Last Amended**: 2025-12-15
