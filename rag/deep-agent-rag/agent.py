"""
Deep agent over the RAG index: Plan → Search → Analyze → Synthesize.

  python agent.py                                          # ask the sample question
  python agent.py "How do I pass headers to an MCP server?"
  python agent.py --reindex "..."                          # rebuild the index first
"""
import argparse
import sys
from os import getenv

from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langchain.agents.middleware import TodoListMiddleware

import config
from indexing import build_index
from retrieval import search_docs

# Where each agent's skills live in the agent's virtual file system (see load_skill_files)
MAIN_SKILLS = "/skills/main/"
RESEARCHER_SKILLS = "/skills/researcher/"

RESEARCHER_PROMPT = """You research ONE question using the search_docs tool.

0. Skill: your FIRST tool call is read_file on the SKILL.md of the skill whose description best matches
   the question (see the Skills list). Then follow its search recipe and reply format.
1. Search: call search_docs. If the results don't answer the question, search again with different
   words. Use at most 4 searches.
2. Analyze: read the chunks and keep only the facts that answer the question.

Reply with:
- Findings: short bullet points, each ending with the chunk ID it came from, e.g. [README.md#3]
- Code: a short snippet from the docs if it helps, with its chunk ID
- Gaps: anything the docs don't say

Use only what the chunks say. Never invent APIs, parameters or behaviour."""

RESEARCHER = {
    "name": "researcher",
    "description": ("Researches ONE focused sub-question in the indexed docs: searches, reads the chunks, "
                    "and returns findings with chunk-ID citations. Give it one specific question."),
    "system_prompt": RESEARCHER_PROMPT,
    "tools": [search_docs],
    "skills": [RESEARCHER_SKILLS],         # retrieval recipes: code examples, concepts, troubleshooting
}

SYSTEM_PROMPT = """You answer questions about MCP (Model Context Protocol) in LangChain, using an index of
LangChain Academy's MCP lesson and the langchain-mcp-adapters docs.

Before you plan, check the Skills list. If a skill's description matches the question, your FIRST tool
call is read_file on its SKILL.md, and you follow it for planning and for the answer's format.

Work in three steps:
1. Plan: call write_todos to break the question into 1-4 focused sub-questions. A simple question needs one.
2. Search and analyze: hand each sub-question to the `researcher` subagent with the task tool.
   Run independent sub-questions in parallel. Don't search yourself.
3. Synthesize: combine the researchers' findings into one answer.
   - Start with a direct 1-2 sentence answer, then the details, then a code example if useful.
   - Cite every fact with its chunk ID, e.g. [README.md#3].
   - If the docs don't cover something, say so instead of guessing.
Mark each todo as completed when it is done."""

SAMPLE_QUESTION = ("How do I use tools from more than one MCP server in a LangChain agent, "
                   "and which transports can I connect with?")

API_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def build_agent(model=config.AGENT_MODEL):
    return create_deep_agent(
        model=model,
        tools=[],                              # the main agent plans and delegates; only researchers search
        system_prompt=SYSTEM_PROMPT,
        middleware=[TodoListMiddleware()],     # adds the write_todos planning tool
        subagents=[RESEARCHER],
        skills=[MAIN_SKILLS],                  # planning recipes, e.g. compare-options
        name="deep-agent-rag",
    )


def load_skill_files(skills_dir=config.SKILLS_DIR) -> dict:
    """Copy skills/ from disk into the agent's virtual file system, where the skills middleware finds them.

    skills/main/compare-options/SKILL.md → /skills/main/compare-options/SKILL.md
    The agent reads them with read_file, but can't change the files on disk.
    """
    return {f"/skills/{path.relative_to(skills_dir).as_posix()}": create_file_data(path.read_text())
            for path in sorted(skills_dir.rglob("*")) if path.is_file()}


def _print_step(message, agent_name: str) -> None:
    """Show what each agent is doing: the skills it loads, its plan, each delegation and each search."""
    for call in getattr(message, "tool_calls", None) or []:
        args = call["args"]
        if call["name"] == "read_file" and args.get("file_path", "").endswith("SKILL.md"):
            print(f"\nSKILL → {agent_name} loads {args['file_path'].split('/')[-2]}")
        elif call["name"] == "write_todos":
            print("\nPLAN")
            for todo in args.get("todos", []):
                print(f"  [{todo.get('status', 'pending')}] {todo.get('content', '')}")
        elif call["name"] == "task":
            print(f"\nSEARCH + ANALYZE → {args.get('subagent_type', '?')}: {args.get('description', '')}")
        elif call["name"] == "search_docs":
            print(f"  {agent_name} searches: {args.get('query', '')}")


def ask(question: str, agent=None, verbose: bool = True) -> dict:
    agent = agent or build_agent()
    run_config = {
        "recursion_limit": config.RECURSION_LIMIT,
        "run_name": "deep-agent-rag",          # the name of the top-level trace in LangSmith
        "tags": ["deep-agent-rag"],
        "metadata": {"question": question, "model": str(config.AGENT_MODEL)},
    }
    inputs = {"messages": [{"role": "user", "content": question}], "files": load_skill_files()}
    # subgraphs=True also streams the researchers' steps; namespace () is the main agent
    state, seen = None, {}
    for namespace, values in agent.stream(inputs, config=run_config, stream_mode="values", subgraphs=True):
        messages = values.get("messages", [])
        if verbose:
            for message in messages[seen.get(namespace, 0):]:
                _print_step(message, "main agent" if not namespace else "researcher")
        seen[namespace] = len(messages)
        if not namespace:
            state = values
    return {"answer": state["messages"][-1].text, "todos": state.get("todos", []),
            "messages": state["messages"]}


def _check_setup(model: str) -> None:
    provider = model.split(":", 1)[0]
    if provider in API_KEYS and not getenv(API_KEYS[provider]):
        sys.exit(f"Set {API_KEYS[provider]} in your environment or .env file (model: {model}).")
    if not config.INDEX_PATH.exists():
        print("No index yet, building it first.\n")
        build_index()
    if getenv("LANGSMITH_TRACING", "").lower() != "true":
        print("(LangSmith tracing is off. Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to see traces.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="your question (default: a sample question)")
    parser.add_argument("--reindex", action="store_true", help="rebuild the index before answering")
    args = parser.parse_args()

    if args.reindex:
        build_index()
    _check_setup(config.AGENT_MODEL)
    question = " ".join(args.question) or SAMPLE_QUESTION
    print(f"QUESTION: {question}")
    result = ask(question)
    print(f"\nSYNTHESIZE → ANSWER\n{result['answer']}")
