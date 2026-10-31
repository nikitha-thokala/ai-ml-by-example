---
name: explain-concept
description: Retrieval recipe for "what is..." and "why / how does it work" questions, e.g. what an MCP resource is, what a transport does.
---

# Explain a concept

Use this when the question asks what something is or how it works, rather than for code.

## Search
1. First search: the term itself, e.g. "MCP resource".
2. Second search: the term with "what is" or "used for", to find the defining sentence.
3. If the term only appears inside code, search for the surrounding section heading.
4. Stop after 3 searches.

## Analyze
- Look for the sentence that defines the term. Headings and the first lines of a section usually have it.
- Collect: what it is, what it's for, and how it relates to the other MCP pieces (tools, resources,
  prompts, transports, clients).
- Keep one short example if a chunk has one.

## Reply
- Definition: one sentence, with its chunk ID.
- How it works: 2-3 bullets, each with a chunk ID.
- Example: a short snippet or a one-line description, with its chunk ID, if the docs have one.
- Gaps: anything the docs don't explain.
