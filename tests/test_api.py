"""
Tests for the FastAPI endpoints, using FastAPI's built-in test client
(doesn't require the server to actually be running).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """The health check endpoint should return a 200 status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "RAG Chatbot API is running"


def test_ask_endpoint_returns_valid_structure():
    """The /ask endpoint should return an answer and a list of sources."""
    response = client.post("/ask", json={"question": "How tall is the Eiffel Tower?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_ask_endpoint_rejects_missing_question():
    """Sending a request without a 'question' field should fail validation."""
    response = client.post("/ask", json={})
    assert response.status_code == 422