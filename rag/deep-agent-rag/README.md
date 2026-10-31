# Deep Agent RAG: ask questions about MCP in LangChain

A RAG (retrieval-augmented generation) tool built as a **deep agent**. It indexes the MCP lesson from
[LangChain Academy](https://academy.langchain.com/), then answers questions by **planning**, sending
**researcher** subagents to **search and analyze** the docs, and **synthesizing** one cited answer.
Both agents use **skills**: task-specific instructions they load only when a question matches.
Every step is traced in **LangSmith**.

What a run prints (illustrative: the skills, plan and answer depend on the model):
```
QUESTION: Compare the stdio and streamable HTTP transports

SKILL → main agent loads compare-options

PLAN
  [in_progress] stdio: how to configure, when to use, limitations
  [pending] streamable HTTP: how to configure, when to use, limitations

SEARCH + ANALYZE → researcher: stdio: how to configure, when to use, limitations
SEARCH + ANALYZE → researcher: streamable HTTP: how to configure, when to use, limitations

SKILL → researcher loads find-code-example
  researcher searches: stdio transport example
  ...

SYNTHESIZE → ANSWER
| Transport | How to configure | ... |
```

---

## How it works

The tool has two parts: **indexing** (run once) and **asking** (run for every question).

```
INDEXING (indexing.py)                         ASKING (agent.py)
─────────────────────                          ─────────────────
1. Load   fetch 3 docs from GitHub             Main agent
2. Split  cut them into ~1,000-char chunks       0. SKILL       load a planning skill if one matches
3. Embed  turn each chunk into a vector          1. PLAN        write_todos: break the question
4. Store  save chunks + vectors to                              into sub-questions
          index/vectors.json                     2. DELEGATE    task → researcher (one per sub-question)
                │                                      Researcher subagent
                │                                       SKILL    load a retrieval recipe if one matches
                │                                       SEARCH   search_docs (vector search)
                └──────────── used by ─────────────►    ANALYZE  keep facts, cite chunk IDs
                                                 3. SYNTHESIZE  combine findings into one answer
```

### What gets indexed

The MCP lesson (Lesson 2.1) from LangChain Academy's **Introduction to LangChain** course, from its
repo [`langchain-ai/lca-lc-foundations`](https://github.com/langchain-ai/lca-lc-foundations):

| Document | What it is |
|---|---|
| `2.1_mcp.ipynb` | The lesson notebook: connecting an agent to a local and an online MCP server |
| `2.1_mcp_server.py` | The lesson's MCP server: a tool, a resource and a prompt |
| `README.md` (langchain-mcp-adapters) | The library docs, which the lesson's server serves as a resource |

To index more documents, add them to `SOURCES` in [config.py](config.py) and run `python indexing.py` again.

---

## Part 1: Indexing in 4 steps ([indexing.py](indexing.py))

Each step is its own function, so you can see what it does and inspect it in LangSmith.

### 1. Load
Fetches each source URL and turns it into a LangChain `Document` (text plus metadata: title, URL,
file name). Notebooks are JSON, so `notebook_to_text()` converts them into readable markdown: text
cells as-is, code cells inside python code fences.

### 2. Split
A whole document is too long to search precisely, so it gets cut into **chunks** of up to 1,000
characters. Neighbouring chunks share 150 characters (the **overlap**), so a sentence cut at a
boundary still appears whole in one of them. The splitter is **language-aware**: markdown is cut at
headings first, Python at `def` and `class`. Each chunk gets a readable ID such as `README.md#4`,
which the agent cites in its answer.

### 3. Embed
Each chunk becomes a **vector**: 384 numbers that capture its meaning, so chunks about similar
topics end up close together. The embedding model, `BAAI/bge-small-en-v1.5`, runs **locally** with
`fastembed`: no API key, no cost, and no GPU. It downloads once, on first use.

### 4. Store
Saves every chunk with its vector in a LangChain `InMemoryVectorStore` and writes it to
`index/vectors.json`. Searching later loads this file, so you index once and ask many times.

> `vector_store.add_documents()` normally does Embed and Store in one call. Here they're split into two
> steps so you can see each one.

Example output:
```
1. Load   3 documents
            2.1_mcp.ipynb: 2,190 characters
            2.1_mcp_server.py: 1,743 characters
            README.md: 11,924 characters
2. Split  25 chunks of up to 1000 characters (150 overlap)
3. Embed  25 vectors × 384 dimensions (BAAI/bge-small-en-v1.5)
4. Store  saved to .../index/vectors.json
```

---

## Part 2: The deep agent ([agent.py](agent.py))

### Why a deep agent?

A plain RAG pipeline does one search and one LLM call. That works for simple questions, but a question
like *"How do I use several MCP servers, and which transports are supported?"* has two parts, and each
part may need several searches. A **deep agent** adds what a simple agent lacks:

| Deep agent feature | Where it is here | Why it helps |
|---|---|---|
| **Planning** | `write_todos` tool (`TodoListMiddleware`) | Breaks the question into sub-questions and tracks them |
| **Subagents** | `researcher`, called with the `task` tool | Each sub-question is researched in a **clean context**; only the findings come back, not every raw chunk |
| **Detailed prompts** | `SYSTEM_PROMPT`, `RESEARCHER_PROMPT` | Spell out the plan → search → analyze → synthesize steps and the citation rules |
| **Virtual file system** | Built into `deepagents` | The agent can save notes during long tasks (in memory, not on your disk) |
| **Skills** | `skills/main/`, `skills/researcher/` | Task-specific instructions each agent loads only when a question matches (see Part 3) |

For a simple question, a regular agent or plain RAG is cheaper and faster. Use a deep agent when the
question has several parts or needs several rounds of searching.

### The four phases

1. **Plan.** The main agent calls `write_todos` to break the question into 1 to 4 sub-questions.
2. **Search.** For each sub-question, it calls `task` to start a `researcher` subagent. Independent
   sub-questions run in parallel. The researcher calls `search_docs`, which embeds the query and
   returns the 4 closest chunks, each labelled with its chunk ID and similarity score. It searches
   again with other words if the first results miss.
3. **Analyze.** The researcher keeps only the facts that answer its sub-question and returns them as
   findings with citations, plus any gaps the docs don't cover.
4. **Synthesize.** The main agent combines all the findings into one answer: a direct answer first,
   then details and code, with a citation like `[README.md#4]` on every fact.

The main agent has no search tool on purpose, so all searching goes through researchers.
`RECURSION_LIMIT` (60 steps) caps how long the agent can run and how much it can spend.

---

## Part 3: Skills ([skills/](skills))

A **skill** is a folder with a `SKILL.md` file: a short YAML header (`name`, `description`) and
step-by-step instructions for one kind of task. Skills use **progressive disclosure**:

1. The agent's system prompt lists only each skill's **name and description**, a line or two each.
2. When a question matches a description, the agent calls `read_file` on that `SKILL.md`.
3. Only then do the full instructions enter its context, and it follows them.

So you can add many task-specific procedures without making every prompt longer. Each agent has
its own skills:

| Agent | Skill | Use it for |
|---|---|---|
| Main agent | `compare-options` | "X vs Y" questions: one todo per option with the same criteria, then a comparison table |
| Researcher | `find-code-example` | "How do I..." questions: search for runnable code, its imports and settings |
| Researcher | `explain-concept` | "What is..." questions: find the defining sentence and how it fits with the rest of MCP |
| Researcher | `troubleshoot` | Errors: search the setup and caveats, list likely causes with a check for each |

The researcher's skills are **retrieval recipes**: they tell it what to search for, in what order, and
what to return. This is called *skill-guided retrieval*: the search strategy is written down for each
type of question instead of the model improvising it.

**How the skills get to the agents.** `create_deep_agent(skills=["/skills/main/"])` gives the main agent
its skills, and `"skills": ["/skills/researcher/"]` in the researcher's spec gives the researcher its own.
These paths are in the agent's **virtual file system**, which lives in memory for the length of one run.
`load_skill_files()` in [agent.py](agent.py) copies the `skills/` folder from disk into it at the
start of each question, so the agent can read the skills but can't change your files.

**Add a skill:** create `skills/researcher/<name>/SKILL.md`. The folder name must match `name:` in the
header. Write the description as "use this when...", because that line is all the agent sees when it
decides which skill to load.

### Skills vs system prompt: what goes where

Each agent gets both. Every model call sees its `SYSTEM_PROMPT` plus the list of skill names and
descriptions; a skill's full text is added only after the agent reads it.

**The rule: if it applies to every question, put it in the system prompt. If it applies only to some
kinds of question, make it a skill.** A skill is optional by design, because the model decides whether
to load it. So a rule that must always hold can't live only in a skill.

There's also a third place, code, for things that shouldn't depend on the model following instructions:

| Layer | Holds | Examples in this project |
|---|---|---|
| **Code** | Rules that must always hold and can be enforced | `search_docs` returns at most 8 chunks; `RECURSION_LIMIT` stops the agent; the main agent has no search tool; skills are copied into memory, so the agent can't change your files |
| **System prompt** | The role, the workflow and rules for every question | "Answer about MCP in LangChain"; plan → delegate → synthesize; cite every fact with its chunk ID; never invent APIs; check the skills first |
| **Skills** | Procedures and formats for some kinds of question | `compare-options` for "X vs Y"; `find-code-example` for "how do I..."; `troubleshoot` for errors |

### Lesson from a live run: the prompt has to tell the agent to use its skills

On the first run with gpt-4o, **neither agent loaded a skill**, even though both skill lists were in
their prompts. The answer also made claims the docs don't support. The prompts only said "if a skill
matches, read it". After changing them to say the agent's **first tool call** is `read_file` on the
matching skill, both agents loaded their skills. The comparison question then got a cited table that
says "not detailed" where the docs are silent, instead of filling the gaps.

Two takeaways:
- Listing skills isn't enough; the system prompt has to make checking them part of the workflow.
- Offline tests with a scripted model prove the wiring works, but only a real model shows whether the
  agent actually uses its skills. Check a run in LangSmith: the `read_file` call on a `SKILL.md` should
  come before the agent plans or searches.

---

## Part 4: Tracing with LangSmith

[LangSmith](https://smith.langchain.com) records each run as a tree, so you can see every step, its
inputs and outputs, how long it took, and how many tokens it used.

- **Indexing:** `@traceable` on each step creates an `index` run with four children: `1. Load`,
  `2. Split`, `3. Embed` and `4. Store`. Each shows a short summary (counts, sources, vector
  dimensions) instead of thousands of raw numbers.
- **Asking:** LangChain and LangGraph trace automatically. The `deep-agent-rag` run shows every LLM
  call, the `write_todos` plan, each `task` call with the researcher's own run inside it, and each
  `search_docs` call. The `retrieve` step is traced as a **retriever** run, so LangSmith shows the
  chunks it returned as a list of documents.

Tracing is on when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set. Without them, everything
still runs; it just isn't traced.

---

## Setup and run

### 1. Install
Tested with Python 3.13.
```bash
cd ai-ml-by-example/rag/deep-agent-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your keys
```bash
cp .env.example .env
```
Open `.env` and fill in:
- `OPENAI_API_KEY`: for the agent's LLM (default model `gpt-4o`)
- `LANGSMITH_API_KEY`: from [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys
  (the free plan is enough)

### 3. Build the index
```bash
python indexing.py
```
This runs the 4 steps and writes `index/vectors.json`. Indexing doesn't call the LLM, so it's free.

### 4. Ask a question
```bash
python agent.py
python agent.py "How do I pass custom headers to an MCP server?"
python agent.py --reindex "What does the Academy lesson's MCP server provide?"
```
With no question, it asks the sample question. If there's no index yet, it builds one first.

### 5. See the traces
Open [smith.langchain.com](https://smith.langchain.com) → **Tracing** → project **deep-agent-rag**.
Click a run to expand the tree. It looks roughly like this:
```
deep-agent-rag
├── model call                   → read_file (a matching skill, e.g. compare-options)
├── model call (plan)            → write_todos
├── task → researcher
│   ├── model call               → read_file (a retrieval recipe, e.g. find-code-example)
│   ├── model call               → search_docs
│   │   └── retrieve             (retriever: the 4 chunks it found)
│   └── model call               (findings)
└── model call (synthesize)      → final answer
```

### 6. Run the tests (no keys needed)
```bash
pytest
```
The tests use fake web pages, fake embeddings and a scripted fake LLM, so they're free and work
offline. They check each indexing step, the search tool, the full deep agent flow (plan, delegate
to the researcher, search, and synthesize), and that each agent sees and loads only its own skills.

---

## Configuration ([config.py](config.py))

| Setting | Default | What it controls |
|---|---|---|
| `SOURCES` | The 3 MCP lesson docs | What to index |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Chunk length and shared characters |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (env: `EMBEDDING_MODEL`) |
| `SKILLS_DIR` | `skills/` | Where the `SKILL.md` files are read from |
| `TOP_K` | 4 | Chunks returned per search |
| `AGENT_MODEL` | `openai:gpt-4o` | The LLM, as a LangChain `provider:model` string (env: `AGENT_MODEL`) |
| `RECURSION_LIMIT` | 60 | Maximum agent steps per question |

**Using Claude instead of OpenAI:**
```bash
pip install langchain-anthropic
```
Then set `AGENT_MODEL=anthropic:claude-sonnet-5-5` and `ANTHROPIC_API_KEY` in `.env`.

If you change `EMBEDDING_MODEL`, run `python indexing.py` again: the index and the queries must use
the same model.

---

## Files

| File | What it does |
|---|---|
| [config.py](config.py) | Sources, chunk sizes, models and paths |
| [indexing.py](indexing.py) | The 4 indexing steps: Load, Split, Embed, Store |
| [retrieval.py](retrieval.py) | Loads the index; the `search_docs` tool the researcher uses |
| [agent.py](agent.py) | The deep agent: planner, researcher subagent, skills loading, and the command-line interface |
| [skills/](skills) | `SKILL.md` files: `main/` for the main agent, `researcher/` for the researcher |
| [tests/test_rag.py](tests/test_rag.py) | Offline tests |
| `.env.example` | Template for your API keys |

## Ideas to extend it

- **Better retrieval:** add keyword (BM25) search next to vector search, or rerank the top results.
- **A real vector database:** replace `InMemoryVectorStore` with Pinecone or pgvector for large
  document sets. Only the Store step and `load_index()` change.
- **Evaluation:** build a small set of questions with known answers and score them with LangSmith
  evaluations, to check whether a change to chunk size or prompts actually helps.
- **More sources:** add the LangChain MCP docs or other Academy lessons to `SOURCES`.
