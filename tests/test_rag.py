"""
Tests for the RAG retrieval logic.
Requires the vector store to already be populated (run ingest.py first).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag import retrieve, build_prompt


def test_retrieve_returns_results():
    chunks, sources = retrieve("How tall is the Eiffel Tower?", top_k=3)
    assert len(chunks) > 0
    assert len(sources) > 0


def test_retrieve_finds_correct_source():
    chunks, sources = retrieve("How tall is the Eiffel Tower?", top_k=1)
    assert "sample.txt" in sources


def test_retrieve_finds_rag_source():
    chunks, sources = retrieve("What is Retrieval-Augmented Generation?", top_k=1)
    assert "rag_explainer.txt" in sources


def test_build_prompt_includes_question_and_context():
    chunks = ["The sky is blue."]
    prompt = build_prompt("What color is the sky?", chunks)
    assert "What color is the sky?" in prompt
    assert "The sky is blue." in prompt