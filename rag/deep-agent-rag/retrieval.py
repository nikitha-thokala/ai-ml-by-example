"""Search the index: embed the question, then return the closest chunks. Used by the agent as a tool."""
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langsmith import traceable

import config
from indexing import get_embeddings


@lru_cache
def load_index(path: Path = config.INDEX_PATH) -> InMemoryVectorStore:
    if not Path(path).exists():
        raise FileNotFoundError(f"No index at {path}. Run `python indexing.py` first.")
    return InMemoryVectorStore.load(str(path), get_embeddings())


@traceable(name="retrieve", run_type="retriever")  # LangSmith shows retriever runs as a list of documents
def retrieve(query: str, k: int = config.TOP_K, vector_store: InMemoryVectorStore | None = None) -> list[dict]:
    results = (vector_store or load_index()).similarity_search_with_score(query, k=k)
    return [{"page_content": doc.page_content, "metadata": {**doc.metadata, "score": round(score, 3)}}
            for doc, score in results]


def format_results(results: list[dict]) -> str:
    if not results:
        return "No matching chunks."
    return "\n\n".join(
        f"[{r['metadata']['chunk_id']}] {r['metadata']['title']} (similarity {r['metadata']['score']})\n"
        f"{r['page_content']}"
        for r in results)


@tool
def search_docs(query: str, k: int = config.TOP_K) -> str:
    """Search the indexed LangChain Academy MCP lesson and langchain-mcp-adapters docs.

    Returns the most similar chunks, each labelled with a chunk ID like [README.md#3] to cite.
    Use short, specific queries (e.g. "MultiServerMCPClient stdio transport") and search again with
    different words if the results don't answer the question.

    Args:
        query: What to look for.
        k: How many chunks to return (1-8).
    """
    return format_results(retrieve(query, k=max(1, min(k, 8))))
