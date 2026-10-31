"""
A tool-using agent written as an explicit loop: an OpenAI model picks a tool, we run it on an MCP server,
send the result back, and repeat until the model answers or the turn limit is hit.

  python agent.py                                   # run the sample questions
  python agent.py "Which city on trip T200 is colder?"
"""
import asyncio
import json
import sys
from os import getenv
from pathlib import Path

import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = getenv("OPENAI_MODEL", "gpt-4o")
MAX_TURNS = 10  # each turn is one model call; caps cost if the agent never finishes
SERVER = Path(__file__).with_name("mcp_server.py")

SYSTEM = (
    "You are a travel assistant. Answer using only facts from the tools; never invent trips or weather. "
    "If a tool returns an error, say what is missing instead of guessing."
)


def to_openai_tools(mcp_tools) -> list[dict]:
    """MCP tool definitions → OpenAI function tools (same JSON Schema, different field names)."""
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}
            for t in mcp_tools]


async def call_mcp_tool(session: ClientSession, call) -> dict:
    """Run one tool call on the MCP server and wrap the output as a tool message."""
    try:
        args = json.loads(call.function.arguments or "{}")  # OpenAI sends arguments as a JSON string
        result = await session.call_tool(call.function.name, args)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        is_error = bool(result.is_error)
    except Exception as e:  # bad JSON or a failed tool goes back to the model as an error, not a crash
        text, is_error = f"tool failed: {e}", True
    # tool messages have no error flag, so the error is marked in the text the model reads
    content = f"ERROR: {text}" if is_error else text
    return {"role": "tool", "tool_call_id": call.id, "content": content, "is_error": is_error}


async def agent_loop(client, session: ClientSession, tools: list[dict], question: str,
                     verbose: bool = True) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    tool_calls = []

    for turn in range(1, MAX_TURNS + 1):
        # 1. Think: the model reads the conversation so far and either calls tools or answers
        response = await client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            # 3. Done: "stop" is a normal answer; anything else ("length", "content_filter") is reported as-is
            return {"answer": msg.content or msg.refusal or "", "stop_reason": choice.finish_reason,
                    "turns": turn, "tool_calls": tool_calls}

        # keep the assistant message with its tool_calls, so each tool result can match a tool_call_id
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [c.model_dump() for c in msg.tool_calls]})

        # 2. Act: run every tool the model asked for, then add one tool message per call
        results = await asyncio.gather(*(call_mcp_tool(session, c) for c in msg.tool_calls))
        for c, r in zip(msg.tool_calls, results):
            tool_calls.append({"tool": c.function.name, "input": c.function.arguments,
                               "output": r["content"], "is_error": r.pop("is_error")})
            if verbose:
                print(f"  turn {turn}: {c.function.name}({c.function.arguments}) → {r['content']}")
            messages.append(r)

    return {"answer": f"stopped: no answer within {MAX_TURNS} turns", "stop_reason": "max_turns",
            "turns": MAX_TURNS, "tool_calls": tool_calls}


async def run_agent(question: str, client=None, verbose: bool = True) -> dict:
    """Start the MCP server, load its tools, and run the loop for one question."""
    client = client or openai.AsyncOpenAI()
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(SERVER.parent))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_openai_tools((await session.list_tools()).tools)
            try:
                return await agent_loop(client, session, tools, question, verbose)
            except openai.APIError as e:
                error = e  # raised below: inside the MCP context it would come out wrapped in an ExceptionGroup
    raise error


async def _main(questions: list[str]):
    client = openai.AsyncOpenAI()
    for q in questions:
        print(f"\n{q}")
        try:
            r = await run_agent(q, client)
        except openai.AuthenticationError:
            sys.exit("Set OPENAI_API_KEY first.")
        except openai.RateLimitError:
            sys.exit("Rate limited or out of credits; check your OpenAI billing and try again.")
        except openai.APIStatusError as e:
            sys.exit(f"API error {e.status_code}: {e.message}")
        except openai.APIConnectionError:
            sys.exit("Could not reach the API; check your network.")
        print(f"  → {r['answer']}\n  ({r['turns']} turns, {len(r['tool_calls'])} tool calls, {r['stop_reason']})")


if __name__ == "__main__":
    if not getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "Which city on trip T100 is the warmest today, and what is that in Fahrenheit?",  # needs 3 steps
        "What's the weather on trip T999?",                                               # tool returns an error
    ]
    asyncio.run(_main(questions))
