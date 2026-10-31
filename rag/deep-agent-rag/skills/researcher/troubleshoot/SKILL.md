---
name: troubleshoot
description: Retrieval recipe for errors and "it doesn't work" questions, e.g. a server that won't connect, tools that don't load, a missing header.
---

# Troubleshoot

Use this when the question describes an error, a failure, or unexpected behaviour.

## Search
1. First search: the exact error text or the failing step, e.g. "tools not loading".
2. Search the setup for that step: the config keys involved, e.g. "transport url command args".
3. Search for requirements and caveats: "requires", "note", "must", "only".
4. Stop after 4 searches.

## Analyze
- List what the docs say must be true for the step to work: the right transport, a running server,
  required config keys, async usage.
- Compare that list with what the question describes, and note each likely mismatch.
- Don't guess causes the docs don't support.

## Reply
- Likely causes: bullets ordered from most to least likely, each with the chunk ID that supports it.
- How to check: one concrete check per cause, from the docs.
- Correct setup: a snippet from the docs that shows the working configuration, with its chunk ID.
- Gaps: causes the docs don't cover, e.g. network or permissions problems.
