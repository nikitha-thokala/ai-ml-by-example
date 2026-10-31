"""
Offline tests: fake web responses, fake embeddings and a fake chat model. No API keys, network or cost.

  pytest tests
"""
import json
from types import SimpleNamespace as NS

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agent
import indexing
import retrieval

NOTEBOOK = json.dumps({"cells": [
    {"cell_type": "markdown", "source": ["## Local MCP server"]},
    {"cell_type": "code", "source": ["client = MultiServerMCPClient({...})"]},
    {"cell_type": "code", "source": []},
]})
README = "# LangChain MCP Adapters\n\n" + "\n\n".join(
    f"## Section {i}\n\n" + f"MultiServerMCPClient connects to several servers. " * 20 for i in range(3))
SERVER = "from mcp.server.fastmcp import FastMCP\n\nmcp = FastMCP('mcp_server')\n"
PAGES = {"2.1_mcp.ipynb": NOTEBOOK, "README.md": README, "2.1_mcp_server.py": SERVER}
SOURCES = [{"title": f"title {f}", "url": f"https://example.com/{f}"} for f in PAGES]


@pytest.fixture
def fake_web(monkeypatch):
    def get(url, timeout):
        return NS(text=PAGES[url.rsplit("/", 1)[-1]], raise_for_status=lambda: None)
    monkeypatch.setattr(indexing.requests, "get", get)


@pytest.fixture
def fake_index(fake_web, tmp_path, monkeypatch):
    """Build a real index on disk with fake embeddings, and point search at it."""
    embeddings = DeterministicFakeEmbedding(size=16)
    path = tmp_path / "vectors.json"
    indexing.build_index(SOURCES, embeddings, path, verbose=False)
    store = indexing.InMemoryVectorStore.load(str(path), embeddings)
    monkeypatch.setattr(retrieval, "load_index", lambda: store)
    return store


# ---------- indexing: the 4 steps ----------

def test_notebook_to_text_keeps_markdown_and_fences_code():
    text = indexing.notebook_to_text(NOTEBOOK)
    assert text == "## Local MCP server\n\n```python\nclient = MultiServerMCPClient({...})\n```"


def test_load_reads_every_source_with_metadata(fake_web):
    docs = indexing.load(SOURCES)
    assert [d.metadata["file"] for d in docs] == list(PAGES)
    assert docs[0].page_content.startswith("## Local MCP server")  # notebook was converted from JSON


def test_split_makes_small_chunks_with_citable_ids(fake_web):
    chunks = indexing.split(indexing.load(SOURCES))
    readme = [c for c in chunks if c.metadata["file"] == "README.md"]
    assert len(readme) > 1
    assert all(len(c.page_content) <= indexing.config.CHUNK_SIZE for c in chunks)
    assert [c.id for c in readme] == [f"README.md#{i}" for i in range(len(readme))]
    assert all(c.metadata["chunk_id"] == c.id and "start_index" in c.metadata for c in chunks)


def test_embed_returns_one_vector_per_chunk(fake_web):
    chunks = indexing.split(indexing.load(SOURCES))
    vectors = indexing.embed(chunks, DeterministicFakeEmbedding(size=16))
    assert len(vectors) == len(chunks) and len(vectors[0]) == 16


def test_store_saves_an_index_that_loads_and_searches(fake_index):
    results = retrieval.retrieve("MultiServerMCPClient", k=2)
    assert len(results) == 2
    assert {"chunk_id", "title", "source", "score"} <= set(results[0]["metadata"])


def test_search_tool_labels_chunks_for_citation(fake_index):
    out = retrieval.search_docs.invoke({"query": "servers", "k": 50})  # k is capped at 8
    assert out.count("\n[") + 1 <= 8
    assert out.startswith("[") and "] title " in out


def test_missing_index_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="indexing.py"):
        retrieval.load_index.__wrapped__(tmp_path / "nope.json")


# ---------- the deep agent: plan → search → analyze → synthesize ----------

class FakeModel(GenericFakeChatModel):
    """Scripted chat model; the main agent and the subagent take turns reading from the same script.

    The script is a plain list, not the `messages` iterator: middleware that serializes the model would use up an
    iterator. Each call's system prompt is recorded so tests can check which skills each agent was shown.
    """

    def __init__(self, script):
        super().__init__(messages=iter([]))
        self.__dict__.update(script=list(script), tool_names=[], system_prompts=[])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.system_prompts.append(messages[0].text if messages[0].type == "system" else "")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        self.tool_names.append([getattr(t, "name", None) or t["name"] for t in tools])
        return self


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_, "type": "tool_call"}])


def test_deep_agent_plans_delegates_and_synthesizes(fake_index, capsys):
    script = [
        call("write_todos", {"todos": [{"content": "Find how to use several MCP servers",
                                        "status": "in_progress"}]}, "c1"),           # main: plan
        call("task", {"subagent_type": "researcher",
                      "description": "How do I use several MCP servers?"}, "c2"),     # main: delegate
        call("search_docs", {"query": "MultiServerMCPClient"}, "c3"),                # researcher: search
        AIMessage(content="Findings: use MultiServerMCPClient [README.md#0]"),        # researcher: analyze
        AIMessage(content="Use MultiServerMCPClient with one entry per server [README.md#0]."),  # main: synthesize
    ]
    model = FakeModel(script)
    result = agent.ask("How do I use several MCP servers?", agent.build_agent(model))

    assert result["answer"] == "Use MultiServerMCPClient with one entry per server [README.md#0]."
    assert result["todos"][0]["content"] == "Find how to use several MCP servers"

    tool_results = {m.tool_call_id: m.content for m in result["messages"] if m.type == "tool"}
    assert "use MultiServerMCPClient [README.md#0]" in tool_results["c2"]  # researcher's findings came back

    # the model is bound once per call: the main agent gets planning + delegation, the researcher gets search
    main_tools = model.tool_names[0]
    researcher_tools = next(names for names in model.tool_names if "search_docs" in names)
    assert {"write_todos", "task"} <= set(main_tools) and "search_docs" not in main_tools
    assert "task" not in researcher_tools

    printed = capsys.readouterr().out
    assert "PLAN" in printed and "SEARCH + ANALYZE → researcher" in printed


# ---------- skills: each agent sees only its own skills and loads one when the question matches ----------

def test_skill_files_are_copied_into_the_virtual_file_system():
    files = agent.load_skill_files()
    assert set(files) == {
        "/skills/main/compare-options/SKILL.md",
        "/skills/researcher/find-code-example/SKILL.md",
        "/skills/researcher/explain-concept/SKILL.md",
        "/skills/researcher/troubleshoot/SKILL.md",
    }
    for path, data in files.items():  # frontmatter name must match the folder name, as the skills spec requires
        assert data["content"].startswith(f"---\nname: {path.split('/')[-2]}\ndescription: ")


def test_agents_load_their_own_skills(fake_index, capsys):
    script = [
        call("read_file", {"file_path": "/skills/main/compare-options/SKILL.md", "limit": 1000}, "c1"),  # main
        call("write_todos", {"todos": [{"content": "stdio: how to configure", "status": "in_progress"}]}, "c2"),
        call("task", {"subagent_type": "researcher", "description": "stdio: how to configure"}, "c3"),
        call("read_file", {"file_path": "/skills/researcher/find-code-example/SKILL.md",
                           "limit": 1000}, "c4"),                                          # researcher
        call("search_docs", {"query": "stdio transport example"}, "c5"),
        AIMessage(content="Findings: stdio runs the server as a subprocess [README.md#0]"),
        AIMessage(content="| Transport | Configure |\n| stdio | command + args [README.md#0] |"),
    ]
    model = FakeModel(script)
    result = agent.ask("Compare stdio and streamable HTTP", agent.build_agent(model))

    # the skill's full instructions reached the main agent only after it asked for them
    tool_results = {m.tool_call_id: m.content for m in result["messages"] if m.type == "tool"}
    assert "# Compare options" in tool_results["c1"]
    assert "# Compare options" not in model.system_prompts[0]

    # each agent's system prompt lists only its own skills (name + description: progressive disclosure)
    main_prompt = model.system_prompts[0]
    researcher_prompt = next(p for p in model.system_prompts if "find-code-example" in p)
    assert "compare-options" in main_prompt and "find-code-example" not in main_prompt
    assert "explain-concept" in researcher_prompt and "troubleshoot" in researcher_prompt
    assert "compare-options" not in researcher_prompt

    printed = capsys.readouterr().out
    assert "SKILL → main agent loads compare-options" in printed
    assert "SKILL → researcher loads find-code-example" in printed
    assert "researcher searches: stdio transport example" in printed
