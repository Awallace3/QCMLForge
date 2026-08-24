---
name: connect-qcf
description: Set up, verify, start, and reconnect to a local QCArchive QCFractal instance using src/qcmlforge/qca.py conventions. Only use this skill when the user explicitly includes the phrase "connect-qcf" in their request.
---

# connect-qcf

Use this skill to manage a local QCFractal + compute-manager workflow with minimal friction.

Primary goals:
- Detect whether a local instance already exists.
- Confirm service health with both process checks and a real `qcportal` connection test.
- Start services in the background when needed so the terminal remains usable.
- Follow the setup logic and defaults from `src/qcmlforge/qca.py`.

## Required Preconditions

1. `QCF_BASE_FOLDER` must be set in the environment.
2. The active Python environment already has required dependencies (`qcfractal-server`, `qcfractal-compute-manager`, `qcportal`, and script dependencies).

If `QCF_BASE_FOLDER` is missing, this is always the first step:
- Ask the user where they want the QCFractal base folder located.
- Wait for their path before continuing any other checks.
- Then instruct/set:

```bash
export QCF_BASE_FOLDER="/absolute/path/to/qcf"
```

Do not silently choose a default folder.

## Operational Defaults

- Default API endpoint: `http://localhost:7777`
- Run start actions in background and keep logs monitorable.
- Keep output concise unless an error or ambiguity appears.
- When troubleshooting is needed, include concrete checks and next commands.

Execution policy:
- If the user asks to start/stop/verify, perform the actions directly when tooling is available.
- If execution is blocked (missing binaries, missing env var, permission error), report the blocker succinctly and provide exact remediation commands.

## Workflow

Execute these phases in order.

### Phase 0: Ensure base folder path

Before any existence/process/connectivity checks:
1. Read `QCF_BASE_FOLDER`.
2. If unset, ask user for desired path and wait for answer.
3. Set/export `QCF_BASE_FOLDER` from that user-provided path.
4. Continue to Phase 1.

### Phase 1: Existence check

Check whether a QCFractal setup exists under `QCF_BASE_FOLDER`.

Expected artifacts:
- `qcfractal_config.yaml`
- `resources.yml`
- `postgres/` directory

Interpretation:
- If artifacts exist: treat as existing instance setup and continue to active/inactive detection.
- If not: treat as fresh setup and initialize via `src/qcmlforge/qca.py` logic.

### Phase 2: Active/inactive detection (must do both)

An instance is considered active only when both conditions are true:

1. Process-level checks indicate server and compute manager are running.
2. `qcportal.PortalClient` connectivity succeeds.

Use process checks such as:
- `pgrep -af "qcfractal-server"`
- `pgrep -af "qcfractal-compute-manager"`

Use the connectivity check:

```python
import qcportal

client = qcportal.PortalClient(
    "http://localhost:7777",
    verify=False,
)
```

If process checks pass but connection fails, treat as unhealthy and troubleshoot.
If connection passes but a process check is missing, treat as degraded and report clearly.

### Phase 3A: Existing + active

Do not restart by default.
Return concise confirmation with:
- process status summary
- connection success summary
- optional quick validation command snippets

### Phase 3B: Existing + inactive

Start services in background with monitorable logs.

Preferred pattern:
1. Start server in detached session (`screen`, `nohup`, or equivalent).
2. Start compute manager in detached session.
3. Re-run process checks.
4. Re-run `qcportal` connection check.

Recommended detached start pattern with logs:

```bash
mkdir -p "$QCF_BASE_FOLDER/logs"
nohup qcfractal-server --config="$QCF_BASE_FOLDER/qcfractal_config.yaml" start \
  > "$QCF_BASE_FOLDER/logs/qcfractal-server.log" 2>&1 &
nohup qcfractal-compute-manager --config="$QCF_BASE_FOLDER/resources.yml" \
  > "$QCF_BASE_FOLDER/logs/qcfractal-compute-manager.log" 2>&1 &
```

When issuing start commands, use config paths from `QCF_BASE_FOLDER`:

```bash
qcfractal-server --config="$QCF_BASE_FOLDER/qcfractal_config.yaml" start
qcfractal-compute-manager --config="$QCF_BASE_FOLDER/resources.yml"
```

### Phase 3C: No setup exists

Follow `src/qcmlforge/qca.py` setup defaults:
- initialize config if missing
- initialize database if missing
- keep security disabled for this local workflow
- use API port `7777`

Then start server and compute manager in background, monitor logs, and verify with process + `qcportal` checks.

## Monitoring and Reporting

When services are started/restarted, always provide:
- where logs are being written
- how to attach/view detached sessions
- exact health-check commands for quick revalidation

Default response style:
- concise action summary
- key pass/fail checks
- only expand into troubleshooting details when something fails

## Troubleshooting Escalation

If checks fail, diagnose in this order:
1. `QCF_BASE_FOLDER` validity and file presence
2. config path correctness
3. process start failures from logs
4. port/listener availability for `localhost:7777`
5. `qcportal` import/runtime issues

For each failure, give one likely cause and one concrete next command.

## Safety and Scope

- This skill is for local-only QCFractal workflows.
- Dummy local credentials are not treated as sensitive in this workflow.
- Do not introduce remote deployment/security hardening guidance unless user asks.
