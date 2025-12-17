# Quickstart: Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot

## Project Overview
This project creates an interactive robotics textbook with an integrated RAG chatbot that answers questions about the content. The system consists of:
- A Docusaurus frontend deployed to GitHub Pages
- A FastAPI backend for RAG functionality
- Qdrant for vector storage
- Neon Postgres for metadata

## Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (for local development)
- OpenAI API key
- Qdrant Cloud account (free tier)
- Neon Postgres account (free tier)

## Local Development Setup

### Backend (RAG Service)
1. Clone the repository
2. Navigate to the backend directory
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your_key_here
   export QDRANT_URL=your_qdrant_url
   export QDRANT_API_KEY=your_qdrant_key
   export NEON_DATABASE_URL=your_neon_url
   ```
7. Run the application: `uvicorn src.main:app --reload`

### Frontend (Docusaurus)
1. Navigate to the frontend directory
2. Install dependencies: `npm install`
3. Run locally: `npm start`

## Content Structure
The book content is organized into 4 modules:
1. Module 1: The Robotic Nervous System (ROS 2)
2. Module 2: The Digital Twin (Gazebo & Unity)
3. Module 3: The AI-Robot Brain (NVIDIA Isaac™)
4. Module 4: Vision-Language-Action (VLA) & Capstone

Each module contains multiple chapters with text content and code examples.

## Adding New Content
1. Add markdown files to the appropriate module directory in `docs/docs/`
2. Update the sidebar configuration in `docusaurus.config.js`
3. Run the indexing script to update the RAG knowledge base:
   ```bash
   python scripts/index_content.py
   ```

## Running Tests
- Backend: `pytest tests/`
- Frontend: `npm test`
- Integration: `pytest tests/integration/`

## Deployment
1. Frontend: Push to main branch to trigger GitHub Pages deployment
2. Backend: Connect Render to your GitHub repository for automatic deployments

## Key Endpoints
- `/api/chat/query` - Submit queries to the RAG system
- `/api/chat/query-by-selection` - Query about selected text
- `/api/content/search` - Search book content
- `/api/health` - Health check endpoint

## Configuration
The system is configured through environment variables and follows the 12-factor app methodology. All configuration is externalized and can be modified without code changes.