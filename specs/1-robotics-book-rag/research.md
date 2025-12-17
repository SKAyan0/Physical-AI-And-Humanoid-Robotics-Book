# Research: Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot

## Backend Hosting Decision

**Decision**: Deploy FastAPI RAG service to Render.com free tier
**Rationale**: Render offers generous free tier with 100 free build minutes and 100 free hours of container runtime per month. It integrates well with GitHub and provides easy deployment. Alternative Railway also offers free tier but Render has better documentation and community support for Python applications.
**Alternatives considered**:
- Railway: Good free tier but less Python-specific support
- Heroku: Free tier was discontinued
- AWS Lambda: Would require more complex setup for persistent RAG service
- Google Cloud Run: Potential costs could exceed free tier limits

## Indexing Strategy

**Decision**: Use GitHub Actions triggered on content changes to re-index book content into Qdrant
**Rationale**: GitHub Actions provides seamless integration with the content management workflow. The indexing can be triggered automatically when new content is added to the docs directory, ensuring the RAG knowledge base stays current. This approach is cost-effective and leverages existing GitHub infrastructure.
**Alternatives considered**:
- Manual script execution: Prone to human error and inconsistency
- Scheduled indexing: May run unnecessarily when no content changes occurred
- Real-time indexing: More complex to implement and maintain

## Context Window Optimization

**Decision**: Use 512-token chunks for RAG content with 256-token overlap for technical content
**Rationale**: Technical robotics content requires sufficient context for understanding, but smaller chunks reduce OpenAI costs while maintaining answer quality. The 256-token overlap ensures concepts that span chunk boundaries are preserved. This balance provides good quality answers while staying within cost constraints.
**Alternatives considered**:
- Larger chunks (1024+ tokens): Better context but higher costs
- Smaller chunks (256 tokens): Lower costs but may lose important context
- Adaptive chunking: More complex implementation with variable cost benefits

## Chat UI Design

**Decision**: Implement floating chat button that expands to full chat widget with "Select-text-to-ask" functionality
**Rationale**: Floating button provides persistent access without cluttering the interface. When expanded, users can select text in the document and ask questions about it directly. This design balances accessibility with clean user experience.
**Alternatives considered**:
- Sidebar chat: Takes up horizontal space that could be used for content
- Top/bottom bar: Less intuitive for content-focused application
- Modal popup: Interrupts reading flow more than floating widget

## Technical Architecture Patterns

### Docusaurus + Backend API Pattern
**Best practice**: Separate static content (Docusaurus) from dynamic services (FastAPI) to leverage GitHub Pages for static hosting while maintaining full backend functionality
**Implementation**: Docusaurus site deployed to GitHub Pages, FastAPI backend deployed to Render, with CORS configured for communication

### RAG Pipeline Pattern
**Best practice**: Implement a clear separation between indexing (content → vectors) and querying (user query → relevant content → answer) phases
**Implementation**: Dedicated indexing service that processes book content and stores in Qdrant, separate query service that retrieves and synthesizes answers

### Vector Storage Strategy
**Best practice**: Use metadata filtering in Qdrant to enable module-specific searches and maintain content organization
**Implementation**: Each chunk will include metadata about the module, chapter, and section it came from for targeted retrieval

## Testing Strategy Implementation

### Site Build Verification
- GitHub Actions workflow to verify Docusaurus build completes without errors
- Linting checks for markdown content
- Link validation to ensure no broken references

### RAG Accuracy Testing
- "Golden set" of questions and expected answers for each module
- Automated tests that verify retrieval relevance using similarity scores
- Manual validation of answer quality for complex technical queries

### Code Integrity Verification
- Syntax validation for all ROS 2 Python (rclpy) code examples
- Static analysis to verify code follows PEP 8 standards
- Unit tests for any utility functions in the code examples