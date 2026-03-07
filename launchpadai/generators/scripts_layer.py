"""Generate utility scripts."""
from pathlib import Path


def generate_scripts_layer(config: dict, project_path: Path):
    if not config["include_rag"]:
        _write(project_path / "scripts" / "ingest.py", '''"""Document ingestion placeholder.

Enable RAG when creating the project to get a full ingestion pipeline.
"""
print("RAG not enabled. Re-create project with RAG support to use ingestion.")
''')
        return

    _write(project_path / "scripts" / "ingest.py", '''"""Document ingestion script — load documents into the vector store.

Usage:
    python scripts/ingest.py --source data/documents/
    launchpad ingest --source data/documents/
"""
import argparse
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.ingestion.loaders import DocumentLoader
from knowledge.ingestion.chunkers import RecursiveChunker
from models.embeddings.provider import embeddings
from knowledge.vectorstore.client import vectorstore


def ingest(source_dir: str = "data/documents"):
    """Load, chunk, embed, and store documents."""
    source = Path(source_dir)
    if not source.exists():
        print(f"Error: Source directory '{source}' not found.")
        print(f"Create it and add documents: mkdir -p {source}")
        sys.exit(1)

    loader = DocumentLoader()
    chunker = RecursiveChunker()

    print(f"Loading documents from {source}...")
    documents = loader.load_directory(source)
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("No documents found. Add files to data/documents/ and try again.")
        return

    # Chunk all documents
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc["content"])
        for chunk in chunks:
            chunk["metadata"] = {
                "source": doc["source"],
                "filename": doc["filename"],
                "chunk_index": chunk["chunk_index"],
            }
        all_chunks.extend(chunks)

    print(f"  Created {len(all_chunks)} chunks")

    # Embed and store
    print("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in all_chunks]

    # Batch embed
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embs = embeddings.embed_batch(batch)
        all_embeddings.extend(batch_embs)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    print("Storing in vector database...")
    vectorstore.add(texts=texts, embeddings=all_embeddings, metadatas=metadatas, ids=ids)

    print(f"Done! Ingested {len(all_chunks)} chunks from {len(documents)} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents")
    parser.add_argument("--source", "-s", default="data/documents", help="Source directory")
    args = parser.parse_args()
    ingest(args.source)
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
