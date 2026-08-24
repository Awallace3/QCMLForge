---
name: select-lot
description: Use this skill only when the user explicitly says "select-LoT". Run the select-LoT workflow to recommend a level of theory (LoT) for intermolecular interaction energies given a directory of geometries, using the ensemble time and interaction-energy estimators. Always call the provided script to build the dataframe and predictions, then filter by a user-provided walltime budget in seconds and recommend the LoT that is predicted to be most accurate for the largest fraction of systems.
---

# select-LoT

This skill recommends a level of theory (LoT) for intermolecular interaction energy calculations under a walltime budget. Always use the provided script to build predictions, then post-process the dataframe to choose the best LoT for the largest fraction of systems.

## When to use

Only trigger this skill when the user explicitly says `select-LoT`.

## Inputs you must collect

- Path to a directory of geometry files accepted by `src/qcml_mcp/ie_time_esimator_script.py`.
- Compute budget in walltime seconds (not CPU-seconds). Ask for a number of seconds.
- Optional: list of methods and bases. If not provided, use all 10 methods and 6 basis sets from the script (60 LoTs total).
- Optional: whether to use CP-corrected timing (`using_cp`). Default to False unless the user requests CP.

## Required workflow (do not skip)

1. Call the script `src/qcml_mcp/ie_time_esimator_script.py` to build the prediction dataframe:
   - Use its `main(...)` function.
   - Provide the geometry directory path.
   - Pass `methods` and `bases` if the user specifies them; otherwise omit to use defaults.
   - Set `auto_download=True` to avoid TTY prompts when loading pretrained models.

2. Save the resulting dataframe to a `.pkl` file in the current working directory.

3. Convert the timing column to walltime seconds:
   - Column name is `ESTIMATED CPU TIMES (log10(s))`.
   - Convert to seconds via `seconds = 10 ** log10_seconds`.

4. Filter out LoT rows whose predicted walltime exceeds the user’s budget.
   - If no rows remain, report that no LoT fits the budget and stop.

5. Recommend the LoT that is predicted to be most accurate for the largest fraction of systems:
   - Use `ERROR ESTIMATES (kcal/mol)` as the accuracy proxy (closer to zero is better; use absolute error).
   - For each system (group by `id`), find the LoT with the minimum absolute error among the budget-feasible rows.
   - Count how many systems each LoT wins; choose the LoT with the largest count.
   - Tie-breakers, in order: lowest median absolute error across all systems, then lowest median predicted walltime.

6. Prepare a concise report summarizing:
   - The recommended LoT.
   - The number and fraction of systems it wins.
   - The budget used and the number of LoTs that were feasible.
   - A short table (top 5 LoTs) with counts, median error, and median walltime.

## Output requirements

- Always state the recommended LoT clearly.
- Always save the dataframe `.pkl` in the working directory and report its filename.
- Keep the report concise and suitable for a novice quantum chemist.

## Notes on file formats

The geometry files must be in formats accepted by `parse_geoms` in `src/qcml_mcp/ie_time_esimator_script.py` (QCElemental-recognized formats such as xyz/psi4 with charge, multiplicity, and units).
