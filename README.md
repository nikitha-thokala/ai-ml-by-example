# AI/ML by Example

Hands-on AI/ML examples with runnable code: AI agents, MCP, RAG, deep agents and LangSmith tracing.
Each example is small, has offline tests, and comes with a README that explains how it works and why
it's built that way.

## Examples

| Example | What you'll learn | Stack |
|---|---|---|
| [Agent loop with MCP tools](agents/agent-loop-mcp) | How a tool-using agent works: the think → act → observe loop, parallel tool calls, tool errors, turn limits, and tools served by an MCP server | OpenAI, MCP |
| [Deep agent RAG](rag/deep-agent-rag) | Retrieval-augmented generation in two phases: indexing (load, split, embed, store) and a deep agent that plans, delegates to researcher subagents, and writes a cited answer. Includes skills (skill-guided retrieval) and LangSmith tracing | deepagents, LangChain, fastembed, LangSmith, OpenAI |

### Suggested order

1. **Agent loop with MCP:** the basic pattern every agent framework builds on, written by hand.
2. **Deep agent RAG:** the same ideas at a higher level, with retrieval, planning, subagents, skills
   and tracing.

## Concepts covered

| Concept | Where |
|---|---|
| Agent loop: tool calls, results, stopping | [agent-loop-mcp](agents/agent-loop-mcp) |
| MCP (Model Context Protocol): servers, tools, stdio transport | [agent-loop-mcp](agents/agent-loop-mcp) |
| RAG: chunking, embeddings, vector search, citations | [deep-agent-rag](rag/deep-agent-rag) |
| Deep agents: planning, subagents, context isolation | [deep-agent-rag](rag/deep-agent-rag) |
| Skills: progressive disclosure, skills vs system prompt vs code | [deep-agent-rag](rag/deep-agent-rag#part-3-skills-skills) |
| Tracing and debugging with LangSmith | [deep-agent-rag](rag/deep-agent-rag#part-4-tracing-with-langsmith) |
| Testing LLM apps offline with fake models | Both |

## Getting started

Each example is self-contained, with its own `requirements.txt`, README and tests. To run one:

1. **Python 3.13** is what the examples are tested with.
2. Create a virtual environment inside the example's folder. For example:

```bash
cd rag/deep-agent-rag
```

```bash
python3.13 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

3. **Run the tests first.** They need no API keys and cost nothing:

```bash
pytest
```

4. Add your API keys as the example's README explains, then run it.

**API keys:** the examples use an OpenAI key, and the RAG example can also use a LangSmith key for
tracing. Keys go in a `.env` file or your shell environment, never in the code. `.env` is in
`.gitignore`, so it isn't committed.

## Repository layout

```
ai-ml-by-example/
├── agents/
│   └── agent-loop-mcp/     # tool-using agent loop with an MCP server
└── rag/
    └── deep-agent-rag/     # RAG with a deep agent, skills and LangSmith tracing
```

New examples go in the folder for their topic, each with its own README, requirements and tests.

## License

MIT. See [LICENSE](LICENSE).
