"""Generate the knowledge/RAG layer — ingestion, chunking, retrieval."""
from pathlib import Path


def generate_knowledge_layer(config: dict, project_path: Path):
    """Generate RAG pipeline files."""
    if not config["include_rag"]:
        return

    base = project_path / "knowledge"
    _write(base / "__init__.py", "")
    _write(base / "ingestion" / "__init__.py", "")
    _write(base / "vectorstore" / "__init__.py", "")
    _write(base / "retrieval" / "__init__.py", "")

    # Chunker
    _write(base / "ingestion" / "chunkers.py", '''"""Document chunking strategies.

Adjust CHUNK_SIZE and CHUNK_OVERLAP in config/settings.py.
"""
from config.settings import settings


class RecursiveChunker:
    """Split text into overlapping chunks by paragraphs, then sentences."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk(self, text: str) -> list[dict]:
        """Split text into chunks with metadata."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at paragraph or sentence boundary
            if end < len(text):
                for sep in ["\\n\\n", "\\n", ". ", " "]:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size * 0.5:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        end = start + len(chunk_text)
                        break

            chunks.append({
                "text": chunk_text.strip(),
                "start_char": start,
                "end_char": end,
                "chunk_index": len(chunks),
            })

            start = end - self.overlap

        return [c for c in chunks if c["text"]]  # Filter empty chunks
''')

    # Loaders
    _write(base / "ingestion" / "loaders.py", '''"""Document loaders — read files from various sources."""
from pathlib import Path


class DocumentLoader:
    """Load documents from files."""

    SUPPORTED = {".txt", ".md", ".pdf", ".html", ".csv", ".json"}

    def load_file(self, path: str | Path) -> dict:
        """Load a single file and return its content with metadata."""
        path = Path(path)
        if path.suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix == ".pdf":
            return self._load_pdf(path)

        content = path.read_text(encoding="utf-8", errors="ignore")
        return {
            "content": content,
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix,
        }

    def load_directory(self, dir_path: str | Path) -> list[dict]:
        """Load all supported files from a directory."""
        dir_path = Path(dir_path)
        documents = []
        for file in sorted(dir_path.rglob("*")):
            if file.suffix in self.SUPPORTED and file.is_file():
                try:
                    documents.append(self.load_file(file))
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
        return documents

    def _load_pdf(self, path: Path) -> dict:
        """Load a PDF file."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text = "\\n".join(page.get_text() for page in doc)
            return {
                "content": text,
                "source": str(path),
                "filename": path.name,
                "file_type": ".pdf",
                "pages": len(doc),
            }
        except ImportError:
            raise ImportError("Install PyMuPDF: pip install pymupdf")
''')

    # Vector store client
    vectorstore_code = _get_vectorstore_client(config)
    _write(base / "vectorstore" / "client.py", vectorstore_code)

    # Retriever
    _write(base / "retrieval" / "retriever.py", '''"""Retrieval logic — search vector store and return relevant chunks."""
from knowledge.vectorstore.client import vectorstore
from models.embeddings.provider import embeddings
from config.settings import settings


class Retriever:
    """Search the vector store for relevant documents."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or settings.TOP_K_RESULTS

    def retrieve(self, query: str) -> list[dict]:
        """Embed the query and search for similar documents."""
        query_embedding = embeddings.embed(query)
        results = vectorstore.search(query_embedding, top_k=self.top_k)
        return results

    def retrieve_with_scores(self, query: str) -> list[dict]:
        """Retrieve documents with similarity scores."""
        query_embedding = embeddings.embed(query)
        results = vectorstore.search(query_embedding, top_k=self.top_k)
        return results

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            text = doc.get("text", doc.get("content", ""))
            context_parts.append(f"[Document {i} — {source}]\\n{text}")

        return "\\n\\n".join(context_parts)


# Singleton
retriever = Retriever()
''')


def _get_vectorstore_client(config: dict) -> str:
    """Generate vector store client code."""
    vdb = config["vector_db"]

    if vdb == "chroma":
        return '''"""ChromaDB vector store client."""
import chromadb
from config.settings import settings


class VectorStore:
    """ChromaDB client — local persistent storage."""

    def __init__(self, collection_name: str = "default"):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR
            if hasattr(settings, "CHROMA_PERSIST_DIR") else "./data/chroma")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict] = None, ids: list[str] = None):
        """Add documents to the collection."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(texts),
            ids=ids,
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        documents = []
        for i in range(len(results["documents"][0])):
            documents.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return documents

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()


# Singleton
vectorstore = VectorStore()
'''

    elif vdb == "pinecone":
        return '''"""Pinecone vector store client."""
from pinecone import Pinecone
import os
from config.settings import settings


class VectorStore:
    """Pinecone managed vector database client."""

    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX", "default"))

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict] = None, ids: list[str] = None):
        """Upsert vectors to Pinecone."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        vectors = []
        for i, (id_, emb) in enumerate(zip(ids, embeddings)):
            meta = (metadatas[i] if metadatas else {})
            meta["text"] = texts[i]
            vectors.append({"id": id_, "values": emb, "metadata": meta})
        self.index.upsert(vectors=vectors)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Query Pinecone for similar vectors."""
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return [
            {"text": m.metadata.get("text", ""), "metadata": m.metadata, "score": m.score}
            for m in results.matches
        ]


# Singleton
vectorstore = VectorStore()
'''

    else:
        return f'''"""Vector store client — {vdb}.

TODO: Configure your vector store connection.
See config/settings.py for connection details.
"""


class VectorStore:
    """Vector store client for {vdb}."""

    def __init__(self):
        # TODO: Initialize your vector store client
        pass

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict] = None, ids: list[str] = None):
        """Add documents to the store."""
        raise NotImplementedError("Configure your vector store in knowledge/vectorstore/client.py")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar documents."""
        raise NotImplementedError("Configure your vector store in knowledge/vectorstore/client.py")


# Singleton
vectorstore = VectorStore()
'''


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
