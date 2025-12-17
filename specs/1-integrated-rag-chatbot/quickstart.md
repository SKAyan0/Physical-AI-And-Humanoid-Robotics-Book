# Quickstart Guide: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

## Project Overview
This project implements an integrated RAG (Retrieval-Augmented Generation) chatbot for a robotics textbook. The system combines a Docusaurus frontend with a FastAPI backend to provide contextual answers to user questions based on book content, with zero hallucination.

## Architecture
- **Frontend**: Docusaurus site deployed to GitHub Pages
- **Backend**: FastAPI application with Qdrant vector store and Neon Postgres metadata
- **AI**: OpenAI Agents SDK for RAG functionality
- **UI Feature**: "Ask about selected text" functionality

## Prerequisites
- Python 3.11+
- Node.js 18+
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
   export OPENAI_API_KEY=your_openai_api_key
   export QDRANT_URL=your_qdrant_url
   export QDRANT_API_KEY=your_qdrant_api_key  # If using cloud
   export NEON_DATABASE_URL=your_neon_database_url
   ```
7. Run the application: `uvicorn src.main:app --reload`

### Frontend (Docusaurus)
1. Navigate to the docs directory
2. Install dependencies: `npm install`
3. Run locally: `npm start`

## Content Structure
The book content is organized into 4 modules:
1. Module 1: The Robotic Nervous System (ROS 2)
2. Module 2: The Digital Twin (Gazebo & Unity)
3. Module 3: The AI-Robot Brain (NVIDIA Isaac™)
4. Module 4: Vision-Language-Action (VLA) & Capstone

## Ingestion Process
To index new content:
1. Add markdown files to the appropriate module directory
2. Run the indexing script: `python scripts/index_content.py`
3. The script will chunk content, generate embeddings, and store in Qdrant

## Key Endpoints
- `POST /api/v1/chat/query` - Submit queries to the RAG system
- `POST /api/v1/chat/query-by-selection` - Query about selected text
- `POST /api/v1/content/search` - Search book content
- `GET /api/v1/health` - Health check endpoint

## Testing
- Backend: `pytest tests/`
- Frontend: `npm test`
- Integration: `pytest tests/integration/`

## Deployment
1. Frontend: Push to main branch to trigger GitHub Pages deployment
2. Backend: Deploy to Render/Railway using the provided Dockerfile or deployment configs

## Configuration
The system is configured through environment variables following the 12-factor app methodology. All configuration is externalized and can be modified without code changes.