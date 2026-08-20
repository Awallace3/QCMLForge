import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

import train_models


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "run.sh"
PUBLIC_VARIABLES = (
    "PYTHON",
    "ITER",
    "MODEL_DIR",
    "AM_MODEL_PATH",
    "ATOM_TYPE_PARAM_MODEL_PATH",
    "DATA_DIR",
    "RANDOM_SEED",
    "N_EPOCHS",
    "LEARNING_RATE",
    "N_RBF",
    "N_NEURON",
    "N_EMBED",
    "SPEC_TYPE_AP",
    "DS_IN_MEMORY",
    "WORLD_SIZE_DDP",
    "TRAIN_OMP_NUM_THREADS",
    "RACKERS_MODEL_OUT",
    "RACKERS_OVERLAP_MODEL_OUT",
)
FORBIDDEN_ARGUMENTS = (
    "--n_params",
    "--dimer_eval_type",
    "--param_start_mean",
    "--param_start_std",
)


def _make_recorder(tmp_path: Path) -> tuple[Path, Path]:
    call_log = tmp_path / "calls.jsonl"
    recorder = tmp_path / "record_python"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "with open(os.environ['CALL_LOG'], 'a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    recorder.chmod(recorder.stat().st_mode | stat.S_IXUSR)
    return recorder, call_log


def _run_script(tmp_path: Path, overrides: dict[str, str]) -> list[list[str]]:
    recorder, call_log = _make_recorder(tmp_path)
    env = os.environ.copy()
    for variable in PUBLIC_VARIABLES:
        env.pop(variable, None)
    env.update({"PYTHON": str(recorder), "CALL_LOG": str(call_log)})
    env.update(overrides)

    # Run from a scratch directory so that the script's relative default
    # MODEL_DIR (and its `mkdir -p`) never touches the repository tree, while
    # the recorded argv still contains the literal default path strings.
    workdir = tmp_path / "workdir"
    workdir.mkdir(exist_ok=True)
    subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=workdir,
        env=env,
        check=True,
    )
    return [json.loads(line) for line in call_log.read_text().splitlines()]


def _expected_command(
    model_type: str,
    output_path: str,
    *,
    atom_model_path: str,
    atom_type_param_model_path: str,
    random_seed: str,
    n_epochs: str,
    n_rbf: str,
    n_neuron: str,
    n_embed: str,
    data_dir: str,
    spec_type_ap: str,
    learning_rate: str,
    ds_in_memory: str | None,
    world_size_ddp: str,
    omp_num_threads: str,
) -> list[str]:
    command = [
        "-u",
        "./train_models.py",
        "--train_apnet",
        model_type,
        "--ap_model_path",
        output_path,
        "--am_model_path",
        atom_model_path,
        "--atom_type_param_model_path",
        atom_type_param_model_path,
        "--random_seed",
        random_seed,
        "--n_epochs",
        n_epochs,
        "--n_rbf",
        n_rbf,
        "--n_neuron",
        n_neuron,
        "--n_embed",
        n_embed,
        "--data_dir",
        data_dir,
        "--spec_type_ap",
        spec_type_ap,
        "--lr",
        learning_rate,
    ]
    if ds_in_memory is not None:
        command.extend(("--ds_in_memory", ds_in_memory))
    command.extend(
        (
            "--world_size_ddp",
            world_size_ddp,
            "--omp_num_threads",
            omp_num_threads,
        )
    )
    return command


def _parse_recorded_calls(calls, monkeypatch):
    parsed_calls = []
    monkeypatch.setattr(
        train_models,
        "train_pairwise_model",
        lambda **kwargs: parsed_calls.append(kwargs),
    )
    monkeypatch.setattr(train_models, "set_all_seeds", lambda _seed: None)

    for call in calls:
        assert call[:2] == ["-u", "./train_models.py"]
        monkeypatch.setattr(sys, "argv", call[1:])
        train_models.main()

    return parsed_calls


def test_run_script_uses_overrides_for_two_sequential_commands(tmp_path):
    values = {
        "ITER": "iter-test",
        "MODEL_DIR": str(tmp_path / "model dir"),
        "AM_MODEL_PATH": str(tmp_path / "atom model.pt"),
        "ATOM_TYPE_PARAM_MODEL_PATH": str(tmp_path / "hfvr vw model.pt"),
        "DATA_DIR": str(tmp_path / "dataset dir"),
        "RANDOM_SEED": "8675309",
        "N_EPOCHS": "37",
        "LEARNING_RATE": "1.25e-4",
        "N_RBF": "17",
        "N_NEURON": "93",
        "N_EMBED": "11",
        "SPEC_TYPE_AP": "test-spec",
        "DS_IN_MEMORY": "False",
        "WORLD_SIZE_DDP": "1",
        "TRAIN_OMP_NUM_THREADS": "19",
        "RACKERS_MODEL_OUT": str(tmp_path / "pure model.pt"),
        "RACKERS_OVERLAP_MODEL_OUT": str(tmp_path / "overlap model.pt"),
    }
    calls = _run_script(tmp_path, values)

    shared = {
        "atom_model_path": values["AM_MODEL_PATH"],
        "atom_type_param_model_path": values["ATOM_TYPE_PARAM_MODEL_PATH"],
        "random_seed": values["RANDOM_SEED"],
        "n_epochs": values["N_EPOCHS"],
        "n_rbf": values["N_RBF"],
        "n_neuron": values["N_NEURON"],
        "n_embed": values["N_EMBED"],
        "data_dir": values["DATA_DIR"],
        "spec_type_ap": values["SPEC_TYPE_AP"],
        "learning_rate": values["LEARNING_RATE"],
        "ds_in_memory": None,
        "world_size_ddp": values["WORLD_SIZE_DDP"],
        "omp_num_threads": values["TRAIN_OMP_NUM_THREADS"],
    }
    assert calls == [
        _expected_command(
            "RackersTholeDampingModel", values["RACKERS_MODEL_OUT"], **shared
        ),
        _expected_command(
            "RackersTholeDampingOverlapModel",
            values["RACKERS_OVERLAP_MODEL_OUT"],
            **shared,
        ),
    ]
    assert all(
        forbidden not in call
        for call in calls
        for forbidden in FORBIDDEN_ARGUMENTS
    )


@pytest.mark.parametrize(
    ("setting", "expected_argument", "expected_parsed"),
    [("fAlSe", None, False), ("tRuE", "True", True)],
)
def test_ds_in_memory_shell_setting_reaches_parser_semantically(
    tmp_path, monkeypatch, setting, expected_argument, expected_parsed
):
    calls = _run_script(
        tmp_path,
        {
            "DS_IN_MEMORY": setting,
            "MODEL_DIR": str(tmp_path / "models"),
        },
    )

    assert len(calls) == 2
    for call in calls:
        if expected_argument is None:
            assert "--ds_in_memory" not in call
        else:
            option_index = call.index("--ds_in_memory")
            assert call[option_index + 1] == expected_argument

    parsed_calls = _parse_recorded_calls(calls, monkeypatch)
    assert [call["apnet_model_type"] for call in parsed_calls] == [
        "RackersTholeDampingModel",
        "RackersTholeDampingOverlapModel",
    ]
    assert [call["ds_in_memory"] for call in parsed_calls] == [
        expected_parsed,
        expected_parsed,
    ]


def test_invalid_world_size_fails_before_invocation(tmp_path):
    recorder, call_log = _make_recorder(tmp_path)
    model_dir = tmp_path / "models"
    env = os.environ.copy()
    for variable in PUBLIC_VARIABLES:
        env.pop(variable, None)
    env.update(
        {
            "PYTHON": str(recorder),
            "CALL_LOG": str(call_log),
            "WORLD_SIZE_DDP": "2",
            "MODEL_DIR": str(model_dir),
        }
    )

    workdir = tmp_path / "workdir"
    workdir.mkdir(exist_ok=True)
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "WORLD_SIZE_DDP must be exactly 1" in result.stderr
    assert not call_log.exists()
    assert not model_dir.exists()


def test_invalid_ds_in_memory_fails_before_invocation(tmp_path):
    recorder, call_log = _make_recorder(tmp_path)
    model_dir = tmp_path / "models"
    env = os.environ.copy()
    for variable in PUBLIC_VARIABLES:
        env.pop(variable, None)
    env.update(
        {
            "PYTHON": str(recorder),
            "CALL_LOG": str(call_log),
            "DS_IN_MEMORY": "sometimes",
            "MODEL_DIR": str(model_dir),
        }
    )

    workdir = tmp_path / "workdir"
    workdir.mkdir(exist_ok=True)
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "DS_IN_MEMORY must be true or false" in result.stderr
    assert not call_log.exists()
    assert not model_dir.exists()


def test_run_script_defaults(tmp_path):
    calls = _run_script(tmp_path, {})

    shared = {
        "atom_model_path": "./models/ap3_saptpbe0/1/am_ap2_1.pt",
        "atom_type_param_model_path": "./models/ap3_saptpbe0/1/atp_hfvr_1.pt",
        "random_seed": "1",
        "n_epochs": "25",
        "n_rbf": "8",
        "n_neuron": "64",
        "n_embed": "8",
        "data_dir": "../qcmlforge/data_dir",
        "spec_type_ap": "2",
        "learning_rate": "5e-5",
        "ds_in_memory": "True",
        "world_size_ddp": "1",
        "omp_num_threads": "16",
    }
    assert calls == [
        _expected_command(
            "RackersTholeDampingModel",
            "./models/ap3_saptpbe0/1/rackers_thole_1.pt",
            **shared,
        ),
        _expected_command(
            "RackersTholeDampingOverlapModel",
            "./models/ap3_saptpbe0/1/rackers_thole_overlap_1.pt",
            **shared,
        ),
    ]
    assert all(
        forbidden not in call
        for call in calls
        for forbidden in FORBIDDEN_ARGUMENTS
    )
    # The default MODEL_DIR is relative, so it must be created under the
    # scratch working directory rather than inside the repository.
    assert (tmp_path / "workdir" / "models" / "ap3_saptpbe0" / "1").is_dir()
