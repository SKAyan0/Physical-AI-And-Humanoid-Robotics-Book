# Physical AI & Humanoid Robotics Book with Integrated RAG Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docusaurus](https://img.shields.io/badge/Docusaurus-3.0-blue)](https://docusaurus.io/)

## 🤖 Overview

This repository contains a comprehensive educational platform for Physical AI & Humanoid Robotics, featuring an interactive textbook with integrated RAG (Retrieval-Augmented Generation) chatbot. The system bridges the gap between digital AI and physical robotics (Embodied Intelligence) through practical implementation of ROS 2, NVIDIA Isaac, and VLA (Vision-Language-Action) models.

## 🎯 Features

- **Interactive Learning Experience**: Engaging textbook with integrated chatbot for immediate assistance
- **4-Module Curriculum**: Comprehensive learning path from ROS 2 fundamentals to autonomous humanoid projects
- **Integrated RAG Chatbot**: Answers questions based on book content with zero hallucination
- **Selection-Based Q&A**: Ask about selected text for contextual learning
- **Technical Depth**: Real-world examples using ROS 2 (rclpy), NVIDIA Isaac Sim, and OpenAI Agents SDK
- **Capstone Project**: Complete implementation guide for autonomous humanoid systems

## 📚 Modules

### Module 1: The Robotic Nervous System (ROS 2)
- Introduction to Middleware: ROS 2 Nodes, Topics, and Services
- Bridging Digital Agents: Controlling Robots with Python (rclpy)
- Defining the Body: URDF for Humanoid Kinematics

### Module 2: The Digital Twin (Gazebo & Unity)
- Physics Simulation: Gravity, Collisions, and Environment in Gazebo
- Visual Fidelity: Rendering and Human-Robot Interaction in Unity
- Sensory Input: Simulating LiDAR, Depth Cameras, and IMUs

### Module 3: The AI-Robot Brain (NVIDIA Isaac™)
- Photorealism & Data: Synthetic Data Generation in Isaac Sim
- Spatial Awareness: Hardware-accelerated VSLAM and Navigation
- Movement Logic: Path Planning for Bipeds using Nav2

### Module 4: Vision-Language-Action (VLA) & Capstone
- Voice Command Interface: Implementing OpenAI Whisper
- Cognitive Planning: LLMs as Action Planners (Natural Language to ROS 2)
- Capstone Project: Building the Autonomous Humanoid (Integration)

## 🛠️ Tech Stack

### Backend (FastAPI)
- **Framework**: FastAPI for high-performance API layer with streaming support
- **Vector Database**: Qdrant for semantic search and RAG functionality
- **Relational Database**: Neon Serverless Postgres for metadata storage
- **AI SDK**: OpenAI Agents/ChatKit SDK for LLM integration
- **Language**: Python 3.11+

### Frontend (Docusaurus)
- **Framework**: Docusaurus for static site generation
- **Deployment**: GitHub Pages
- **Language**: JavaScript/TypeScript
- **Components**: React-based interactive widgets

### Robotics Stack
- **Middleware**: ROS 2 (Robot Operating System 2)
- **Simulation**: Gazebo, Unity, NVIDIA Isaac
- **Programming**: Python (rclpy), C++

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- OpenAI API Key
- Qdrant Cloud Account (Free Tier)
- Neon Serverless Postgres Account (Free Tier)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_openai_api_key
export QDRANT_URL=your_qdrant_url
export QDRANT_API_KEY=your_qdrant_api_key
export NEON_DATABASE_URL=your_neon_database_url

# Run the application
python src/main.py
```

### Frontend Setup
```bash
cd frontend/docs
npm install
npm start
```

## 🔐 Security

- All API keys stored in `.env` files (never committed)
- No "hallucination" principle: RAG responses grounded in indexed content
- Secure credential management with environment variables
- Regular security audits and updates

## 📖 Usage

1. Access the Docusaurus site to browse the interactive textbook
2. Select text in the book content and ask questions using the integrated chatbot
3. Follow the 4-module curriculum to learn robotics concepts progressively
4. Complete the capstone project to build an autonomous humanoid system

## 🤝 Contributing

We welcome contributions to enhance the educational content and technical implementation. Please read our contributing guidelines before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Physical AI & Humanoid Robotics Book** - Bridging the gap between digital AI and physical robotics (Embodied Intelligence)