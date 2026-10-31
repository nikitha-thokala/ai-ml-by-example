"""
Indexing pipeline in 4 steps: 1. Load → 2. Split → 3. Embed → 4. Store.
Each step is traced in LangSmith as its own run under one "index" run.

  python indexing.py
"""
import json
from functools import lru_cache
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langsmith import traceable

import config


# ---------- 1. Load: fetch each source and turn it into a Document (text + metadata) ----------

def notebook_to_text(raw: str) -> str:
    """Jupyter notebook JSON → readable markdown: text cells as-is, code cells in ```python fences."""
    cells = json.loads(raw)["cells"]
    parts = []
    for cell in cells:
        source = "".join(cell["source"]).strip()
        if not source:
            continue
        parts.append(source if cell["cell_type"] == "markdown" else f"```python\n{source}\n```")
    return "\n\n".join(parts)


def _summary(docs: list[Document]) -> dict:
    """What LangSmith shows as a step's output: counts and samples instead of every full text."""
    return {"count": len(docs),
            "sources": sorted({d.metadata["file"] for d in docs}),
            "first": docs[0].page_content[:300] if docs else ""}


@traceable(name="1. Load", run_type="chain", process_outputs=_summary)
def load(sources: list[dict] = config.SOURCES) -> list[Document]:
    docs = []
    for src in sources:
        resp = requests.get(src["url"], timeout=30)
        resp.raise_for_status()
        file = src["url"].rsplit("/", 1)[-1]
        text = notebook_to_text(resp.text) if file.endswith(".ipynb") else resp.text
        docs.append(Document(page_content=text,
                             metadata={"title": src["title"], "source": src["url"], "file": file}))
    return docs


# ---------- 2. Split: cut each document into overlapping chunks small enough to embed and cite ----------

def _splitter_for(file: str) -> RecursiveCharacterTextSplitter:
    # language-aware splitting: prefer to cut at headings (markdown) or at def/class (python)
    language = Language.PYTHON if file.endswith(".py") else Language.MARKDOWN
    return RecursiveCharacterTextSplitter.from_language(
        language, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP, add_start_index=True)


@traceable(name="2. Split", run_type="chain", process_outputs=_summary)
def split(docs: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(_splitter_for(doc.metadata["file"]).split_documents([doc])):
            chunk.id = f"{doc.metadata['file']}#{i}"  # stable, readable ID the agent cites, e.g. README.md#3
            chunk.metadata["chunk_id"] = chunk.id
            chunks.append(chunk)
    return chunks


# ---------- 3. Embed: turn each chunk into a vector, so similar meanings end up close together ----------

class FastEmbedEmbeddings(Embeddings):
    """Local embedding model (ONNX, no GPU or API key). Downloads the model once on first use."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from fastembed import TextEmbedding  # imported here: loading the model takes a few seconds
        self.model_name = model_name
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models embed questions slightly differently from passages; fastembed handles the prefix
        return next(iter(self.model.query_embed(text))).tolist()


@lru_cache
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings()


@traceable(name="3. Embed", run_type="embedding",
           process_inputs=lambda inputs: {"chunks": len(inputs["chunks"])},
           process_outputs=lambda vectors: {"vectors": len(vectors),
                                            "dimensions": len(vectors[0]) if vectors else 0})
def embed(chunks: list[Document], embeddings: Embeddings | None = None) -> list[list[float]]:
    return (embeddings or get_embeddings()).embed_documents([c.page_content for c in chunks])


# ---------- 4. Store: save chunks + vectors in a vector store and write it to disk ----------

@traceable(name="4. Store", run_type="chain",
           process_inputs=lambda inputs: {"chunks": len(inputs["chunks"]), "path": str(inputs["path"])})
def store(chunks: list[Document], vectors: list[list[float]], embeddings: Embeddings | None = None,
          path: Path = config.INDEX_PATH) -> dict:
    vector_store = InMemoryVectorStore(embeddings or get_embeddings())
    # this is what vector_store.add_documents() does inside; done by hand so Embed and Store are separate steps
    for chunk, vector in zip(chunks, vectors, strict=True):
        vector_store.store[chunk.id] = {"id": chunk.id, "vector": vector,
                                        "text": chunk.page_content, "metadata": chunk.metadata}
    vector_store.dump(str(path))
    return {"path": str(path), "chunks": len(chunks)}


@traceable(name="index", run_type="chain", process_outputs=lambda r: r)
def build_index(sources: list[dict] = config.SOURCES, embeddings: Embeddings | None = None,
                path: Path = config.INDEX_PATH, verbose: bool = True) -> dict:
    docs = load(sources)
    if verbose:
        print(f"1. Load   {len(docs)} documents")
        for d in docs:
            print(f"            {d.metadata['file']}: {len(d.page_content):,} characters")
    chunks = split(docs)
    if verbose:
        print(f"2. Split  {len(chunks)} chunks of up to {config.CHUNK_SIZE} characters "
              f"({config.CHUNK_OVERLAP} overlap)")
    vectors = embed(chunks, embeddings)
    if verbose:
        print(f"3. Embed  {len(vectors)} vectors × {len(vectors[0])} dimensions "
              f"({getattr(embeddings or get_embeddings(), 'model_name', 'custom model')})")
    result = store(chunks, vectors, embeddings, path)
    if verbose:
        print(f"4. Store  saved to {result['path']}")
    return {"documents": len(docs), **result}


if __name__ == "__main__":
    build_index()
