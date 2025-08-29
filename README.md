# RAG System

A Retrieval-Augmented Generation (RAG) system that processes and queries 10-K financial documents and OpenAI documentation.

## Features

- Document embedding using Nomic embeddings
- Vector storage with Qdrant
- Azure OpenAI integration for response generation
- Support for multiple document types (10-K reports, OpenAI docs)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`:
   ```
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   ```

3. Run the main script:
   ```bash
   python 10k_agent.py
   ```

## Project Structure

- `10k_agent.py` - Main RAG system implementation
- `qdrant_data/` - Vector database storage (excluded from git)
- `certificates/` - SSL certificates (excluded from git)

## Usage

The system supports querying both 10-K financial documents and OpenAI documentation through semantic search and AI-powered responses.
