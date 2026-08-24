---
name: full-agentic-workflow
description: >-
  Use this skill when the user explicitly asks for "full-agentic-workflow".
  This skill runs an end-to-end quantum chemistry pipeline: (1) classify user-provided or budget-feasible levels of theory (LoTs) into high/medium/low predicted-accuracy buckets via select-LoT given a geometry directory, compute budget, and reference interaction energies,
  (2) ask the user which classified LoTs they want to compute unless they already specified the exact set, (3) start or verify a local QCFractal instance, (4) queue manybody computations on the user-selected LoTs and retrieve results,
  (5) extract CP-corrected interaction energies, convert Hartree to kcal/mol, sum Psi4 walltimes from child outputs, and (6) compute signed errors against a user-provided reference when provided.
  The pipeline uses two conda environments: qcml for select-LoT, and p4_qcml for QCFractal operations.
---

# full-agentic-workflow

End-to-end quantum chemistry workflow that chains together three sub-skills:
1. **select-LoT** — predict IE errors and timings, filter by budget, classify LoTs into predicted-accuracy buckets
2. **connect-qcf** — start/verify a local QCFractal instance
3. **run-IEs** — queue and retrieve manybody interaction energies for user-selected LoTs
4. **Post-process** — extract CP-corrected energies, convert to kcal/mol, sum Psi4 walltimes, compute signed errors vs reference when provided

## When to use

Only trigger this skill when the user explicitly says `full-agentic-workflow`.

## Environment requirements

This workflow requires **two conda environments**. You must switch between them at the right phases:

| Phase | Environment | Reason |
|-------|-------------|--------|
| select-LoT (parsing, prediction) | `qcml` | Needs `psi4`, `apnet_pt`, `qcml_mcp` with pretrained models |
| connect-qcf, run-IEs, post-processing | `p4_qcml` | Needs `qcportal`, `qcfractal`, `pandas`, `psi4`, `qcmanybody` |

### Conda activation in subprocesses

`conda activate` does not work in OpenCode's non-interactive shell. Do not use `conda activate <env>` — it will fail with `CondaError: Run 'conda init' before 'conda activate'`. Instead, invoke Python directly using the env's full binary path:

- qcml: `/home/vlita3/miniconda3/envs/qcml/bin/python3`
- p4_qcml: `/home/vlita3/miniconda3/envs/p4_qcml/bin/python3`

Confirm which environment is active by checking `sys.executable`.

### Cross-environment data transfer

The two conda environments use different Python versions (qcml = 3.12, p4_qcml = 3.10) with the same qcelemental version (0.29.0) but different internal module structures. **Do not pickle DataFrames containing qcelemental Molecule objects in one env then load them in the other** — it will fail with `ModuleNotFoundError: No module named 'qcelemental.models.v1'`.

To transfer data across environments, save geometry file paths instead of pickled molecule objects. Both envs can parse the same geometry files directly.

### Import and source-editing guardrails

If `select-LoT` fails while importing `apnet_pt` or pretrained-model dependencies, treat it as an environment problem first. Do not patch core `apnet_pt` model or dataset files just to make this workflow import. In the transcript that motivated this skill, apparent syntax/import failures were resolved by using the user's working environment with locally downloaded model weights, not by editing `src/apnet_pt/`.

Use these rules:
- Prefer the exact Python binary for the phase (`qcml` for `select-LoT`, `p4_qcml` for QCFractal/run-IEs).
- Set `PYTHONPATH=<repo>/src` when importing project modules from ad hoc scripts.
- If importing `qcml_mcp` triggers `ModuleNotFoundError: No module named 'mcp'`, import the specific script by file path with `importlib.util.spec_from_file_location(...)` instead of importing the package root.
- If the `run-IEs` helper path contains a hyphen (`src/qcml_mcp/run-IEs/example_manybody.py`), import it by file path with `importlib.util`; do not use dotted import syntax for that directory.
- If dependencies or model weights are missing, report the blocker and ask the user whether they want to run that phase in their prepared environment or provide the resulting dataframe. Do not install packages or edit source unless the user explicitly approves.

## Inputs you must collect

- **Geometry input**: Path to either a directory of dimer geometry files or a single dimer geometry file accepted by `src/qcml_mcp/ie_time_esimator_script.py`. If the user gives one file, create/use a temporary workflow directory containing that file because `parse_geoms()` expects a directory.
- **Compute budget**: Walltime in seconds (not CPU-seconds).
- **Reference energies**: Required for LoT accuracy bucketing and optional only when the user explicitly wants budget/timing filtering without accuracy classes. Interaction energies may be provided in any format (CSV, text, dataframe, screenshot). The model is responsible for parsing these into a usable format.
- **Optional**: List of QM methods and basis sets. Defaults to all 10 methods and 6 basis sets if not specified.
- **Optional**: Counterpoise correction (`using_cp`). Defaults to True. When True, the IE predictions use CP-corrected models. Note this affects the interaction energies, not the timing model.
- **Working Directory**: Path to an empty directory in which workflow files are to be saved. DO NOT attempt to work in/inspect files outside of this directory, unless they are explicitly mentioned here. 

## Phase 1: select-LoT (qcml environment)

Use the qcml Python binary: `/home/vlita3/miniconda3/envs/qcml/bin/python3`.

### Step 1: Build the prediction dataframe

Call `src/qcml_mcp/ie_time_esimator_script.py`'s `main()` function. To avoid importing the `qcml_mcp` package root, load the script directly when needed:
```python
import importlib.util
from pathlib import Path

script_path = Path(<repo_root>) / "src/qcml_mcp/ie_time_esimator_script.py"
spec = importlib.util.spec_from_file_location("ie_time_esimator_script", script_path)
lot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lot)

df = lot.main(
    geom_path=<geometry_directory>,
    methods=<methods or None>,
    bases=<bases or None>,
    using_cp=<using_cp>,
    auto_download=True,
)
```

If the user supplies explicit LoT strings such as `wB97X-V/aug-cc-pVTZ/CP`, derive `methods`, `bases`, and `using_cp` from those strings and restrict the dataframe to exactly those LoTs. Do not broaden to default methods/bases unless the user asks for the default search.

This returns a dataframe with columns including:
- `id` — system identifier
- `qcel_dimer` — QCElemental Molecule for each dimer
- `qcel_monA`, `qcel_monB` — monomer molecules
- `Level of Theory` — e.g., "HF/cc-pVDZ/unCP"
- `ERROR ESTIMATES (kcal/mol)` — predicted IE error vs CCSD(T)/CBS
- `ESTIMATED CPU TIMES (log10(s))` — predicted CPU time

### Step 2: Save the full dataframe

Before saving, normalize the select-LoT predicted error sign exactly once. Keep the raw model output column unchanged for provenance, and create one flipped column for all downstream comparisons:
```python
df["predicted_ie_error_kcalmol_for_bucketing"] = -df["ERROR ESTIMATES (kcal/mol)"]
```

Use `predicted_ie_error_kcalmol_for_bucketing` for all reference comparisons, percent-error calculations, bucket assignment, and rankings. Do not flip this column again later in the workflow.

Save to `select_lot_df.pkl` in the current working directory.

### Step 3: Filter by budget

Convert timing to walltime seconds:
```python
df["walltime_seconds"] = 10 ** df["ESTIMATED CPU TIMES (log10(s))"]
```

Filter to rows where `walltime_seconds <= <user_budget>`. If no rows remain, report that no LoT fits the budget and wait for user input.

### Step 4: Classify LoTs into predicted-accuracy buckets

Do not reduce the workflow to a single optimal LoT recommendation. Instead, classify the user-provided or budget-feasible LoTs into predicted-accuracy buckets relative to the user-provided reference interaction energies.

First merge the reference interaction energies onto the budget-feasible dataframe by `id`. The reference value must be in kcal/mol. If references are missing for any `id`, report the missing systems and ask the user before assigning accuracy buckets.

For the percent-error calculation, use the normalized flipped select-LoT predicted error column created in Step 2:
```python
df["predicted_abs_percent_error"] = (
    100
    * df["predicted_ie_error_kcalmol_for_bucketing"].abs()
    / df["reference_ie"].abs()
)
```

Use `abs(reference_ie)` in the denominator so percent errors are non-negative for attractive interactions with negative reference energies. If any reference interaction energy is zero or near-zero, do not compute a percent error for that system without asking the user how to handle it.

Assign buckets per row:
- `high_accuracy`: `predicted_abs_percent_error < 2`
- `medium_accuracy`: `2 <= predicted_abs_percent_error < 5`
- `low_accuracy`: `5 <= predicted_abs_percent_error < 10`
- `not_recommended`: `predicted_abs_percent_error >= 10`

Classify each LoT separately for each system. When summarizing across many systems, report how many systems fall into `high_accuracy`, `medium_accuracy`, `low_accuracy`, and `not_recommended` for each LoT, along with the median predicted absolute percent error. Do not collapse mixed performance into one overall bucket label unless the user explicitly asks for one.

Do not recommend LoTs in `not_recommended`. If every feasible LoT is `not_recommended`, report that no LoT meets the predicted accuracy threshold and ask the user whether to expand the method/basis set search, increase the budget, or proceed for diagnostic purposes only.

After presenting the buckets, ask the user which LoTs they want to compute as real QCFractal jobs. Do not assume that all high/medium/low LoTs should be queued. The user may choose all LoTs in one bucket, a subset across buckets, or diagnostic `not_recommended` LoTs. Only skip this prompt if the user already explicitly said to queue a specific set of LoTs or to queue all feasible/requested LoTs.

### Step 5: Build the submission dataframe

The submission dataframe is the user-selected subset from the budget-feasible/classified rows. Do not queue anything until the user has chosen which LoTs to compute, unless they already explicitly provided that choice. If the user selects `not_recommended` LoTs or asks to run all calculations, submit those exact LoTs but clearly label those rows so they are not presented as recommendations.

The dataframe already contains all needed data. However, **do not include qcelemental Molecule objects** when moving between environments — they may not survive the environment switch. Instead:

1. Create a separate `geom_index.csv` mapping system `id` to its original geometry file path in the user-provided directory.
2. Save the submission dataframe with all columns EXCEPT `qcel_dimer`, `qcel_monA`, `qcel_monB` columns (drop the molecule objects).
3. Save both files — they will be recombined in Phase 3.

```python
# Save geometry file index
import pandas as pd, os

geom_dir = <geometry_directory>
geom_index = pd.DataFrame({
    "id": df["id"].unique(),
    "geom_path": [os.path.join(geom_dir, f) for f in os.listdir(geom_dir) if os.path.isfile(os.path.join(geom_dir, f))]
})
# Match file extensions — use the actual file names from the geometry directory
# that correspond to each system id
geom_index.to_csv("geom_index.csv", index=False)

# Drop molecule columns, save submission df
sub_df = df.drop(columns=["qcel_dimer", "qcel_monA", "qcel_monB"], errors="ignore")
sub_df.to_pickle("run_ies_input.pkl", protocol=2)
```

If the user ran the estimator themselves and provides `select_lot_df.pkl`, skip Phase 1 and continue from that dataframe. Do not re-run the estimator in the wrong environment.

## Phase 2: connect-qcf (p4_qcml environment)

Use the p4_qcml Python binary: `/home/vlita3/miniconda3/envs/p4_qcml/bin/python3`.

Follow the `connect-qcf` skill workflow:
1. Ensure `QCF_BASE_FOLDER` is set. If not, ask the user.
2. Check if a QCFractal instance already exists at `$QCF_BASE_FOLDER`.
3. If it exists but is inactive, start server and compute manager in background.
4. If no setup exists, initialize using `src/qcmlforge/qca.py` conventions (port 7777, no security). Ensure the following resource config is used (not the default values):
```
    resources_config={
        "update_frequency": 15,
        "cores_per_worker": 10,
        "max_workers": 1,
        "memory_per_worker": 250,
    },
```
5. Verify with both process checks (`pgrep`) and `qcportal.PortalClient` connectivity.

Do not proceed to Phase 3 until the QCFractal instance is confirmed active and reachable.

## Phase 3: run-IEs (p4_qcml environment)

Still in the `p4_qcml` environment.

### Step 1: Load and attach molecule objects from geometry files

The submission dataframe was saved without molecule objects (they don't survive env switching). Reconstruct them by re-parsing the original geometry files:

```python
import pandas as pd
import qcelemental as qcel

df = pd.read_pickle("run_ies_input.pkl")
geom_index = pd.read_csv("geom_index.csv")

# Parse geometry files to reconstruct molecule objects
def get_qcel_dimer(geom_path):
    with open(geom_path, "r") as f:
        raw = f.read()
    return qcel.models.Molecule.from_data(raw)

mol_map = {}
for _, row in geom_index.iterrows():
    mol_map[row["id"]] = get_qcel_dimer(row["geom_path"])

# Attach molecule objects to the submission dataframe
df["qcel_dimer"] = df["id"].map(mol_map)
```

### Step 2: Queue manybody computations

Before queueing new computations, check whether matching manybody records already exist in the connected QCFractal server for the same molecule/fragments, method, basis, program/specification, and CP/noCP correction choice. QCFractal may deduplicate equivalent submissions and return existing record IDs, but do not rely on accidental duplicate submission as the lookup mechanism.

If matching records already exist:
- Attach the existing QCFractal record IDs to the dataframe.
- Retrieve completed records immediately when they are already `complete`.
- For existing `waiting` or `running` records, reuse those IDs and report their status instead of queueing fresh computations.
- For existing `error` records, report the error and ask whether to reset/retry them; prefer `client.reset_records()` over submitting duplicate jobs when the user wants to retry transient failures.

Only queue fresh manybody computations for rows where no matching record exists or where the user explicitly asks for a new independent computation.

```python
import importlib.util
from pathlib import Path

mod_path = Path(<repo_root>) / "src/qcml_mcp/run-IEs/example_manybody.py"
spec = importlib.util.spec_from_file_location("run_ies_example_manybody", mod_path)
run_ies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_ies)

df = run_ies.queue_manybodys(df)
```

### Step 3: Confirm and exit

Use `check_manybody_progress(df)` to verify records entered `waiting` or `running` state. Note the following:

- How many jobs were submitted
- For each job, the predicted walltime vs the user's budget
- The predicted completion time window based on the estimated timings

Then **stop execution and report to the user**. Do not poll. These jobs take hours or days.

Save the queued dataframe as `run_ies_queued.pkl`. If you had to queue directly via `qcportal` because the helper module is unavailable, preserve the same columns and format used by `run-IEs`: `qcfractal id` as the returned record id/list, `job status`, `mb_interaction_energy`, `mb_wall_time`, and `psi4 output`.

### Step 4: User re-queries for status

When the user asks about progress (e.g., "are my jobs done?", "check status"):

1. Load `run_ies_queued.pkl`.
2. Call `check_manybody_progress(df)` to get current status.
3. For still-running jobs, compute a **predicted completion percentage**:
   ```python
   # elapsed_since_submission = current_time - submission_time
   # predicted_walltime_seconds = 10 ** df["ESTIMATED CPU TIMES (log10(s))"]
   # completion_pct = min(100 * elapsed_since_submission / predicted_walltime, 99)
   ```
4. Report status per job (waiting/running/complete/error) with completion percentages.
5. If any jobs completed early or errored, note that.
6. **Exit**. Do not call `retrieve_manybodies()` until all jobs are complete.

### Step 5: All jobs complete — retrieve results

When status checks show all records are `complete` (or errored):

1. Call `retrieve_manybodies(df)` once.
2. Report count of completed vs errored jobs.
3. Save as `run_ies_results.pkl`.

## Phase 4: Post-processing

Still in the `p4_qcml` environment.

### Step 1: Parse reference energies

The user provides reference interaction energies. The reference must contain at minimum system identifiers (matching the `id` column) and the reference interaction energy in kcal/mol.

Normalize the reference into a Series or DataFrame keyed by system `id`.

### Step 2: Extract the interaction energy

The `retrieve_manybodies()` function in `example_manybody.py` may return `NaN` for current `qcmanybody` schemas. Prefer the explicit result key matching the LoT correction type:

- CP LoTs: `rec.properties["results"]["cp_corrected_interaction_energy"]`
- unCP/noCP LoTs: `rec.properties["results"]["nocp_corrected_interaction_energy"]`

Only use `rec.properties["ret_energy"]` as a documented fallback after checking the correction-specific key. In the motivating session, `ret_energy` matched `cp_corrected_interaction_energy`, but the user expected the CP-corrected key to be used explicitly.

If you need to extract it manually:
```python
client = qcportal.PortalClient("http://localhost:7777", verify=False)
records = client.get_manybodys([qcf_id], include=["clusters", "**"])
rec = records[0]
props = rec.properties
results = props.get("results", {})
ie = results.get("cp_corrected_interaction_energy")
```

Store the raw manybody interaction energy in Hartree as `mb_interaction_energy`. QCArchive/Psi4 energies are atomic units by convention, and the child singlepoint total energies should be Hartree-scale.

### Step 3: Convert manybody energies to kcal/mol
```python
h2kcalmol = 627.509474  # hartree to kcal/mol

df["mb_interaction_energy_kcalmol"] = df["mb_interaction_energy"] * h2kcalmol
```

Do not flip the sign of the QCFractal computed energies — keep them as-is from the server.

### Step 4: Extract and sum actual Psi4 walltimes

`retrieve_manybodies()` may leave `mb_wall_time` as `NaN`. If so, export/read each child singlepoint stdout and parse lines like:

```text
Psi4 wall time for execution: 0:01:10.35
```

Sum the parsed seconds across the child singlepoints for the manybody record and write the total to `mb_wall_time` in seconds. Also write the child stdout/stderr payloads into `psi4 output` and, when useful for inspection, export them to `psi4_outputs/` with a manifest.

### Step 5: Use the already-flipped select-LoT prediction column

The `ERROR ESTIMATES (kcal/mol)` column from select-LoT uses a sign convention that requires flipping before comparing predicted errors to reference interaction energies or computing predicted percent-error buckets. That flip should already have happened once in Phase 1 Step 2 into `predicted_ie_error_kcalmol_for_bucketing`.

Do not mutate `ERROR ESTIMATES (kcal/mol)` in place, and do not flip `predicted_ie_error_kcalmol_for_bucketing` again. If the dataframe was produced by the user and does not contain `predicted_ie_error_kcalmol_for_bucketing`, inspect/ask before creating it so the sign is not normalized twice.

### Step 6: Compute signed errors

Merge the reference energies with the results dataframe on `id`. Compute signed errors:
```python
df["IE_error_kcalmol"] = df["mb_interaction_energy_kcalmol"] - df["reference_ie"]
```
Round the final value so that it has the same number of decimal places as the least precise value in the subtraction. 

Positive error means the predicted binding is weaker than reference (underbinding). Negative means overbinding.

If reporting actual post-computation percent errors, compute them from the retrieved QCFractal interaction energies, not the select-LoT predicted error:
```python
df["actual_abs_percent_error"] = 100 * df["IE_error_kcalmol"].abs() / df["reference_ie"].abs()
```

### Step 7: Save the final dataframe

Save as `full_workflow_results.pkl`. Include at minimum: `id`, `Level of Theory`, `mb_interaction_energy`, `mb_interaction_energy_kcalmol`, `reference_ie` when provided, `IE_error_kcalmol` when computable, `predicted_ie_error_kcalmol_for_bucketing` when available, `predicted_abs_percent_error` when available, `accuracy_bucket` when available, `walltime_seconds` or `ESTIMATED CPU TIMES (log10(s))`, and `mb_wall_time`.

Before declaring the dataframe complete, check that required columns are present and non-null for every row: `qcfractal id`, `job status`, `mb_interaction_energy`, `mb_interaction_energy_kcalmol`, `mb_wall_time`, and `psi4 output`. All QCFractal statuses should be `complete` unless the user explicitly accepts partial/error rows.

## Output requirements

Always provide:

1. **Printed report** including:
   - The Phase 1 LoT accuracy buckets (`high_accuracy`, `medium_accuracy`, `low_accuracy`, `not_recommended`) with counts and median predicted absolute percent error
   - Budget used and how many LoTs were feasible
   - QCFractal connection status
   - Number of manybody jobs submitted (total across all systems × LoTs)
   - On re-query: predicted completion percentages per job
   - After completion: completed vs errored counts
    - Summary statistics of errors vs reference when reference data is available (mean, median, max, min signed error)
     - A short table of per-system predicted percent errors and buckets when reference data is available
     - Final ranking by predicted absolute percent error and actual `mb_wall_time`, excluding `not_recommended` LoTs from recommendations unless the user explicitly asks to see them

2. **Saved files**:
   - `select_lot_df.pkl` — full select-LoT prediction dataframe
   - `run_ies_input.pkl` — submission dataframe (all feasible LoTs)
   - `run_ies_queued.pkl` — queued dataframe with QCFractal IDs
   - `run_ies_results.pkl` — raw results from QCFractal
    - `full_workflow_results.pkl` — final dataframe with Hartree energies, kcal/mol energies, walltimes, and errors when references are available

## Execution model

This workflow is **not** a single continuous execution. It spans multiple user interactions:

1. **First invocation**: Run Phase 1, present LoT buckets, ask the user which LoTs to compute, then stop unless the user already gave an explicit queueing choice. After the user chooses LoTs, run Phase 2 → Phase 3 (submit only, then exit).
2. **Subsequent re-queries**: Run Phase 3 Step 4 (status check, report, exit).
3. **Final invocation**: When jobs are complete, run Phase 3 Step 5 + Phase 4 (retrieve, post-process, report).

Each interaction should save intermediate files so state persists across sessions.

## Error handling

### Environment errors
If a command fails with `ModuleNotFoundError`, `ImportError`, or `conda` errors:
1. Check which environment is active.
2. Verify the environment has the required packages.
3. Report the missing package/model-weight issue and suggest the correct prepared environment.
4. Do not blindly edit source code, especially under `src/apnet_pt/`.
5. If the user offers to run the estimator elsewhere and provide a dataframe, accept that handoff and resume at retrieval/post-processing.

### QCFractal errors
If connection to QCFractal fails:
1. Confirm `QCF_BASE_FOLDER` is set.
2. Check if the server process is running.
3. Check `$QCF_BASE_FOLDER/logs/` for error messages.
4. Report findings to the user.

### Computation errors
If manybody jobs error out:
1. Report the error details from the QCFractal record.
2. Ask the user whether to troubleshoot or proceed with partial results.

## Skill interaction notes

- This skill does NOT modify or replace the three sub-skills. It orchestrates them.
- When calling code from `select-LoT` or `run-IEs` sub-skill scripts, import from the canonical module paths:
  - `src/qcml_mcp/ie_time_esimator_script.py` via direct file import when package-root imports fail
  - `src/qcml_mcp/run-IEs/example_manybody.py` via direct file import because `run-IEs` is not a valid dotted module name
  - `qcmlforge.qca` (for QCFractal setup)
- If source edits become necessary, limit them to workflow/helper code the user asked you to modify. Do not patch unrelated model package files as part of running this workflow.
