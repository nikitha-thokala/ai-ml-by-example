"""Settings for the deep agent RAG tool. Override any of them with environment variables (or a .env file)."""
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # API keys and LangSmith settings; see .env.example

# --- what to index: the MCP lesson from LangChain Academy's "Introduction to LangChain" course ---
ACADEMY = "https://raw.githubusercontent.com/langchain-ai/lca-lc-foundations/main/notebooks/module-2"
SOURCES = [
    {"title": "LangChain Academy - Lesson 2.1: MCP (notebook)", "url": f"{ACADEMY}/2.1_mcp.ipynb"},
    {"title": "LangChain Academy - Lesson 2.1: MCP server", "url": f"{ACADEMY}/resources/2.1_mcp_server.py"},
    # the lesson's MCP server serves this README as a resource, so it is part of the lesson material
    {"title": "langchain-mcp-adapters README",
     "url": "https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/main/README.md"},
]

# --- indexing ---
CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 150   # characters shared by neighbouring chunks, so a sentence cut at a boundary isn't lost
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # runs locally with fastembed: free, no API key
INDEX_PATH = HERE / "index" / "vectors.json"

# --- skills: task-specific instructions the agents load only when a question matches ---
SKILLS_DIR = HERE / "skills"   # skills/main/ for the main agent, skills/researcher/ for the researcher

# --- search and agent ---
TOP_K = 4                                                 # chunks returned per search
AGENT_MODEL = getenv("AGENT_MODEL", "openai:gpt-4o")      # any LangChain "provider:model" string
RECURSION_LIMIT = 60                                      # caps agent steps (and cost) per question
