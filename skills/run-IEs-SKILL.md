---
name: run-ies
description: Use this skill when the user asks to run interaction-energy many-body jobs with run-IEs, queue or check QCArchive/QCFractal manybody records, or report completed IE results from a prepared dataframe. This skill handles submit, progress checks, safe polling behavior, and result summarization using src/qcml_mcp/run-IEs/example_manybody.py.
---

# run-IEs

Use this skill to submit and manage many-body interaction-energy (IE) jobs on a local QCArchive/QCFractal server from a correctly formatted dataframe.

## When to use

Trigger this skill when the user asks for `run-IEs` workflows, including:
- Queueing manybody jobs from an existing dataframe.
- Checking whether queued jobs are complete.
- Returning and saving completed manybody IE results.

If the user explicitly asks for `run-IEs`, prefer this skill.

## Required input

- A correctly formatted pandas dataframe expected by `src/qcml_mcp/run-IEs/example_manybody.py`.
- The dataframe must include the columns needed for queueing, including:
  - `Level of Theory`
  - `qcel_dimer`

If the input is not in this format, stop and ask for a corrected dataframe.

## Script and functions to use

Always use functions from:

`src/qcml_mcp/run-IEs/example_manybody.py`

Primary functions:
- `queue_manybodys(df)`
- `check_manybody_progress(df, max_print=5)`
- `retrieve_manybodies(df)`
- `hard_delete_manybodies_with_children(df)`
- `cancel_then_hard_delete_manybodies(df)`

Use delete/cancel helpers only when the user explicitly asks for deletion/cancellation.

## Core workflow

Follow the branch that matches the user request.

### A) User asks to run/submit jobs

1. Load/receive the dataframe.
2. Queue records with `queue_manybodys(df)`.
3. Confirm records are in `waiting` or `running` state via `check_manybody_progress(...)`.
4. Report that jobs were submitted and are in progress.
5. Exit.

Important: Do not call `retrieve_manybodies(df)` in a loop while records are waiting/running.

### B) User asks whether jobs are complete

1. Check status with `check_manybody_progress(df, max_print=...)`.
2. If any records are still `waiting` or `running`:
   - Tell the user jobs are not complete yet.
   - Do not repeatedly poll.
   - Exit.
3. If records are complete:
   - Call `retrieve_manybodies(df)` once to populate results.
   - Continue to reporting and save steps below.

### C) User asks for results

1. Ensure records are complete (status check first).
2. Retrieve once with `retrieve_manybodies(df)`.
3. Summarize results from the output dataframe, including:
   - count of completed jobs
   - count of errored jobs
   - key IE values from `mb_interaction_energy`
4. Save the resulting dataframe to disk (pickle format).
5. Report saved filename/path.

## Polling and timeout guardrails

- Never continuously call `retrieve_manybodies()` while records are `waiting` or `running`.
- Avoid tight polling loops that waste context or time out.
- Prefer single status checks per user request.
- If jobs are still running, report status and stop.

## Output requirements

- Keep responses concise and explicit about current job state.
- On submit requests: confirm queue action and that jobs are waiting/running, then exit.
- On completion checks: if not complete, say so and exit.
- On completed jobs: provide a short results summary and save the output dataframe.
