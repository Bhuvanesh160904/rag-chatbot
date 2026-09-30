# RAG Chatbot

A local, fully containerized Retrieval-Augmented Generation (RAG) chatbot. Ask questions about your own documents and get grounded, cited answers - powered entirely by open-source tools running on your own machine, with zero API costs.

## What it does

Upload text or PDF documents, and ask natural-language questions about them. The system finds the most relevant passages using vector search, then generates an answer using a local LLM - grounded only in your documents, with the source cited. If the answer isn't in your documents, it says so honestly instead of making something up.

## Tech Stack

- Python - core application logic
- FastAPI - REST API layer
- ChromaDB - local vector database
- sentence-transformers - local text embeddings (no API key needed)
- Ollama (llama3.2:1b) - local LLM for answer generation (no API key needed)
- Docker and Docker Compose - containerized, portable deployment
- pytest - automated test suite

## Architecture

Document (.pdf/.txt) -> chunked into overlapping text segments -> embedded into vectors (sentence-transformers) -> stored in ChromaDB

User question -> embedded into a vector -> matched against stored chunks (retrieval) -> relevant chunks and question sent to local LLM (generation) -> grounded answer returned, with source citation

## Running it - Option 1: Docker (recommended)

Make sure Ollama is running on your host machine first:
ollama serve
ollama pull llama3.2:1b

Then build and run the containerized app:
docker compose up --build

API available at http://localhost:8000 - interactive docs at http://localhost:8000/docs

## Running it - Option 2: Local Python

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Ingest your documents (drop .txt/.pdf files into data/documents/ first):
python3 app/ingest.py

Run as a CLI chatbot:
python3 app/rag.py

Or run as a web API:
uvicorn app.main:app --reload

## API Usage

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is RAG in AI?"}'

Response:
{"answer": "RAG is an AI technique that combines information retrieval with text generation.", "sources": ["rag_explainer.txt"]}

## Running Tests

pytest tests/ -v

11 tests covering chunking logic, retrieval accuracy, and API endpoints.

## Project Structure

rag-chatbot/
  app/
    ingest.py - Document ingestion pipeline
    rag.py - Core RAG logic (retrieve and generate)
    main.py - FastAPI web server
  data/
    documents/ - Drop your .txt/.pdf files here
  tests/ - Automated test suite
  Dockerfile
  docker-compose.yml
  requirements.txt

## Why This Project

Built to demonstrate practical, end-to-end AI engineering: local model orchestration, vector search, API design, containerization, and testing - with no reliance on paid APIs.
