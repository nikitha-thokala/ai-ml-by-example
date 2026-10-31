# Agent loop with MCP tools

A minimal tool-using agent. An OpenAI model decides which tool to call, the tool runs on an
[MCP](https://modelcontextprotocol.io/) server, the result goes back to the model, and the loop repeats
until the model answers.

## What an agent loop is

A single LLM call answers from what it already knows. An **agent** can act: it calls tools, reads the
results, and decides the next step from what it found. You need a loop when the next step depends on
the previous result. For example:

> "Which city on trip T100 is the warmest today, and what is that in Fahrenheit?"

The agent can't know which cities to check until it reads the trip, and it can't convert a temperature
until it knows which city is warmest:

```
turn 1: get_trip("T100")              → Lisbon, Madrid, Barcelona
turn 2: get_weather × 3 (in parallel) → 24°C, 31°C, 27°C
turn 3: convert_temperature(31)       → 87.8°F
turn 4: answer                        → "Madrid is warmest at 87.8°F"
```

## How the loop works (`agent.py`)

```
messages = [system, question]
repeat up to MAX_TURNS:
    message = model(messages, tools)             # think
    if not message.tool_calls:                   # done
        return message.content
    run every tool call on the MCP server        # act
    append one tool message per call             # observe, then loop
```

Points that matter in real agents:

- **Append the assistant message with its `tool_calls`** before the tool results. Each tool message
  must refer to a `tool_call_id` from that assistant message, or the API rejects the request.
- **Arguments arrive as a JSON string** (`call.function.arguments`). Parse them with `json.loads`, and
  treat invalid JSON as a tool error.
- **Parallel tool calls.** The model can ask for several tools at once (the three `get_weather` calls
  above). They run in parallel with `asyncio.gather`, and each gets its own `role: "tool"` message.
- **Tool errors go back to the model** instead of crashing the loop, so the agent can recover or
  explain what's missing. Tool messages have no error field, so the text starts with `ERROR:`.
- **`MAX_TURNS` caps cost.** Without it, a confused agent could keep calling tools forever.
- **Check `finish_reason`.** `stop` is a normal answer. `length` or `content_filter` are returned as-is
  so the caller can handle them.

## Why MCP

The tools live in a separate MCP server (`mcp_server.py`) instead of inside the agent:

- The agent finds tools at runtime with `list_tools()`, so you can add a tool without changing the loop.
- MCP works with any model. The same server works with OpenAI, Claude, Claude Code or any other MCP
  client, and you can switch models without rewriting the tools.
- Tools are marked `read_only_hint=True`, which tells clients the tools don't change anything.

`to_openai_tools()` converts the MCP tool definitions into OpenAI function tools. Both use JSON Schema,
so the conversion just renames fields (`input_schema` → `function.parameters`).

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
python agent.py
python agent.py "Which city on trip T200 is colder?"
```

The model defaults to `gpt-4o`. Set `OPENAI_MODEL` to try another one, e.g.
`OPENAI_MODEL=gpt-4o-mini python agent.py`.

Run the offline tests (no API key needed):

```bash
pytest test_agent.py
```

The tests replace OpenAI with a fake client that returns scripted responses, but they run the real
MCP server, so they check the whole loop: tool calls, parallel results, errors, bad JSON and the turn
limit.

## Files

| File | What it does |
|---|---|
| `mcp_server.py` | MCP server with 3 read-only tools over sample data |
| `agent.py` | Starts the server over stdio, loads its tools, and runs the agent loop |
| `test_agent.py` | Offline tests with a fake OpenAI client |

## Going further

- **Less code:** the OpenAI Agents SDK (`openai-agents`) runs this same loop for you with
  `Agent(mcp_servers=[MCPServerStdio(...)])` and `Runner.run()`. Writing it by hand, as here, shows
  what the SDK does.
- **Remote MCP servers:** the Responses API can call an MCP server on a public URL itself, with
  `tools=[{"type": "mcp", "server_url": ...}]`.
- **Structured output:** have the final answer match a schema when another program reads it.
- **Guard rails:** allowlist tools, ask a human before any tool that changes data, and log every call.
