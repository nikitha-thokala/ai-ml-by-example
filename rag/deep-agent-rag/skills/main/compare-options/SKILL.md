---
name: compare-options
description: Plan and answer questions that compare two or more options, e.g. "stdio vs streamable HTTP", "which transport should I use", "difference between X and Y".
---

# Compare options

Use this when the question asks to compare, choose between, or list the differences between options.

## Plan
1. Name the options being compared. If the question says "which ... should I use", list every option
   the docs mention.
2. Pick 2-4 criteria that matter for the question, e.g. how to configure it, when to use it, limitations.
3. Write one todo per option. Each says the same criteria, so the findings line up:
   "Transport stdio: how to configure, when to use, limitations"

## Delegate
- Send one `researcher` task per option, in parallel.
- Put the criteria in every task description, so each researcher looks for the same things.

## Synthesize
1. One-sentence answer first: the main difference, or which option fits which situation.
2. Then a table, one row per option, one column per criterion. Cite every cell, e.g. [README.md#5].
   Write "not covered in the docs" when a researcher found nothing.
3. End with a short "Use X when..., use Y when..." line, only if the docs support it.
