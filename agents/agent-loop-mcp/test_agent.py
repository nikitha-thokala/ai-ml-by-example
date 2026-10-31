"""
Offline tests: a fake OpenAI client replays scripted responses, while the MCP server is real.
No API key or network needed.

  pytest test_agent.py
"""
import asyncio
import json
from types import SimpleNamespace as NS

from openai.types.chat import ChatCompletionMessage

import agent


def answer(text, finish_reason="stop"):
    return NS(choices=[NS(finish_reason=finish_reason,
                          message=ChatCompletionMessage(role="assistant", content=text))])


def tool_calls(*calls):
    """calls: (id, tool name, arguments dict or a raw string)"""
    message = ChatCompletionMessage.model_validate({"role": "assistant", "content": None, "tool_calls": [
        {"id": id_, "type": "function",
         "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
        for id_, name, args in calls]})
    return NS(choices=[NS(finish_reason="tool_calls", message=message)])


class FakeClient:
    """Stands in for openai.AsyncOpenAI: returns one scripted response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def run(question, responses):
    client = FakeClient(responses)
    return asyncio.run(agent.run_agent(question, client, verbose=False)), client


def test_loop_runs_tools_until_answer():
    result, client = run("Warmest city on T100 in Fahrenheit?", [
        tool_calls(("c1", "get_trip", {"trip_id": "T100"})),
        tool_calls(("c2", "get_weather", {"city": "Lisbon"}), ("c3", "get_weather", {"city": "Madrid"})),
        tool_calls(("c4", "convert_temperature", {"celsius": 31})),
        answer("Madrid is warmest at 87.8°F."),
    ])
    assert result["answer"] == "Madrid is warmest at 87.8°F."
    assert result["turns"] == 4
    assert result["stop_reason"] == "stop"
    assert [c["tool"] for c in result["tool_calls"]] == ["get_trip", "get_weather", "get_weather",
                                                          "convert_temperature"]
    assert "87.8" in result["tool_calls"][-1]["output"]  # the real MCP server did the conversion

    # the assistant message with its tool_calls comes first, then one tool message per call, matched by id
    sent = client.requests[2]["messages"]
    assert sent[-3]["role"] == "assistant" and len(sent[-3]["tool_calls"]) == 2
    assert [(m["role"], m["tool_call_id"]) for m in sent[-2:]] == [("tool", "c2"), ("tool", "c3")]
    assert "is_error" not in sent[-1]  # only API fields go to OpenAI
    assert {t["function"]["name"] for t in client.requests[0]["tools"]} == {
        "get_trip", "get_weather", "convert_temperature"}


def test_tool_error_is_sent_back_not_raised():
    result, _ = run("Weather on T999?", [
        tool_calls(("c1", "get_trip", {"trip_id": "T999"})),
        answer("I couldn't find trip T999."),
    ])
    assert "not found" in result["tool_calls"][0]["output"]
    assert result["stop_reason"] == "stop"


def test_unknown_tool_is_reported_as_error():
    result, client = run("x", [
        tool_calls(("c1", "delete_trip", {"trip_id": "T100"})),
        answer("I can't do that."),
    ])
    assert result["tool_calls"][0]["is_error"]
    assert client.requests[1]["messages"][-1]["content"].startswith("ERROR:")


def test_bad_json_arguments_are_reported_as_error():
    result, _ = run("x", [
        tool_calls(("c1", "get_weather", '{"city": "Oslo"')),  # truncated JSON
        answer("Sorry, something went wrong."),
    ])
    assert result["tool_calls"][0]["is_error"]


def test_turn_limit_stops_a_looping_agent(monkeypatch):
    monkeypatch.setattr(agent, "MAX_TURNS", 3)
    loop_forever = [tool_calls((f"c{i}", "get_weather", {"city": "Oslo"})) for i in range(3)]
    result, client = run("x", loop_forever)
    assert result["stop_reason"] == "max_turns"
    assert len(client.requests) == 3


def test_non_tool_finish_reason_is_returned():
    result, _ = run("x", [answer("partial answ", finish_reason="length")])
    assert result["stop_reason"] == "length"
    assert result["turns"] == 1
