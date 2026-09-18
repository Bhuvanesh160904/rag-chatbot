"""
Tests for the ingestion pipeline: chunking logic.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingest import chunk_text


def test_chunk_text_basic():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_chunk_text_overlap():
    text = "0123456789" * 100
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_empty_string():
    chunks = chunk_text("", chunk_size=500, overlap=50)
    assert chunks == []  
