"""
Ingestion pipeline: reads documents, splits into chunks, 
embeds them, and stores in ChromaDB.
"""
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DOCS_DIR = "data/documents"
VECTORSTORE_DIR = "data/vectorstore"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks so we don't cut sentences awkwardly

# Local embedding model — runs on your machine, no API key needed
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_documents():
    """Read every file in data/documents and return list of (filename, text)."""
    documents = []
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            text = read_pdf_file(filepath)
        elif filename.endswith(".txt"):
            text = read_text_file(filepath)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue
        documents.append((filename, text))
    return documents


def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Found {len(documents)} document(s).")

    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for filename, text in documents:
        chunks = chunk_text(text)
        print(f"  {filename}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_{i}")
            all_metadatas.append({"source": filename, "chunk_index": i})

    if all_chunks:
        collection.add(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadatas
        )
        print(f"Stored {len(all_chunks)} chunks in the vector store.")
    else:
        print("No chunks to store. Add some .txt or .pdf files to data/documents/")


if __name__ == "__main__":
    main()