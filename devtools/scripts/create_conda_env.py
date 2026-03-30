import argparse
import importlib
import os
import re
import glob
import shutil
import subprocess as sp
from tempfile import TemporaryDirectory
from contextlib import contextmanager

# YAML imports
try:
    import yaml as yaml_lib  # PyYAML

    loader = yaml_lib.safe_load

    def dumper(data, stream):
        yaml_lib.safe_dump(data, stream, sort_keys=False)
except ImportError:
    try:
        yaml_lib = importlib.import_module("ruamel_yaml")
    except ImportError:
        try:
            # Load Ruamel YAML from the base conda environment
            from importlib import util as import_util

            CONDA_BIN = os.path.dirname(os.environ["CONDA_EXE"])
            ruamel_yaml_path = glob.glob(
                os.path.join(
                    CONDA_BIN,
                    "..",
                    "lib",
                    "python*.*",
                    "site-packages",
                    "ruamel_yaml",
                    "__init__.py",
                )
            )[0]
            spec = import_util.spec_from_file_location("ruamel_yaml", ruamel_yaml_path)
            if spec is None or spec.loader is None:
                raise ImportError("Unable to load ruamel_yaml module specification")
            yaml_lib = import_util.module_from_spec(spec)
            spec.loader.exec_module(yaml_lib)
        except (KeyError, ImportError, IndexError) as err:
            raise ImportError(
                "No YAML parser could be found in this or the conda environment. "
                "Could not find PyYAML or Ruamel YAML in the current environment, "
                "AND could not find Ruamel YAML in the base conda environment through CONDA_EXE path. "
                "Environment not created!"
            ) from err
    yaml = yaml_lib.YAML(typ="safe")
    loader = yaml.load  # typ="safe" avoids odd typing on output
    dumper = yaml.dump


@contextmanager
def temp_cd():
    """Temporary CD Helper"""
    cwd = os.getcwd()
    with TemporaryDirectory() as td:
        try:
            os.chdir(td)
            yield
        finally:
            os.chdir(cwd)


# Args
parser = argparse.ArgumentParser(
    description="Creates a conda environment from file for a given Python version."
)
parser.add_argument(
    "-n", "--name", type=str, help="The name of the created Python environment"
)
parser.add_argument(
    "-p", "--python", type=str, help="The version of the created Python environment"
)
parser.add_argument("conda_file", help="The file for the created Python environment")

args = parser.parse_args()

# Open the base file
with open(args.conda_file, "r") as handle:
    yaml_script = loader(handle.read())

python_replacement_string = "python {}*".format(args.python)

try:
    for dep_index, dep_value in enumerate(yaml_script["dependencies"]):
        if re.match(
            "python([ ><=*]+[0-9.*]*)?$", dep_value
        ):  # Match explicitly 'python' and its formats
            yaml_script["dependencies"].pop(dep_index)
            break  # Making the assumption there is only one Python entry, also avoids need to enumerate in reverse
except (KeyError, TypeError):
    # Case of no dependencies key, or dependencies: None
    yaml_script["dependencies"] = []
finally:
    # Ensure the python version is added in. Even if the code does not need it, we assume the env does
    yaml_script["dependencies"].insert(0, python_replacement_string)

# Figure out conda path
if "CONDA_EXE" in os.environ:
    conda_path = os.environ["CONDA_EXE"]
else:
    conda_path = shutil.which("conda")
if conda_path is None:
    raise RuntimeError(
        "Could not find a conda binary in CONDA_EXE variable or in executable search path"
    )

print("CONDA ENV NAME  {}".format(args.name))
print("PYTHON VERSION  {}".format(args.python))
print("CONDA FILE NAME {}".format(args.conda_file))
print("CONDA PATH      {}".format(conda_path))

# Write to a temp directory which will always be cleaned up
with temp_cd():
    temp_file_name = "temp_script.yaml"
    with open(temp_file_name, "w") as f:
        dumper(yaml_script, f)
    result = sp.run(
        [conda_path, "env", "create", "-n", args.name, "-f", temp_file_name],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Conda environment creation failed with exit code {result.returncode}"
        )
