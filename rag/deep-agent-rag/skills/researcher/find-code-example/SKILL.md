---
name: find-code-example
description: Retrieval recipe for "how do I..." questions that need working code, e.g. connecting to a server, passing headers, loading tools.
---

# Find a code example

Use this when the question asks how to do something and the answer should include code.

## Search
1. First search: the task in plain words, plus "example", e.g. "pass custom headers example".
2. If the results have prose but no code, search for the class or function names they mention,
   e.g. "MultiServerMCPClient headers".
3. If the code uses an import or setting you haven't seen explained, search for it once more.
4. Stop after 4 searches.

## Analyze
- Prefer chunks that contain a code block. A chunk that only describes the code is weaker evidence.
- Check the example matches the question: the right transport, client and option names.
- Note anything the example needs to run: imports, a running server, environment variables.

## Reply
- Findings: one or two bullets on what the code does, each with its chunk ID.
- Code: the snippet copied exactly from the chunk, with its chunk ID. Don't fill in missing parts.
- Needs: imports, servers or settings the snippet depends on, with chunk IDs.
- Gaps: anything the docs don't show.
