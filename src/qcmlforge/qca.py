import subprocess
import os
import yaml

def setup_qcarchive_qcfractal(
    QCF_BASE_FOLDER=None,
    reset=False,
    db_config={
        "name": None,
        "enable_security": "false",
        "allow_unauthenticated_read": None,
        "logfile": None,
        "loglevel": None,
        "service_frequency": None,
        "max_active_services": None,
        "heartbeat_frequency": None,
        "log_access": None,
        "database": {
            "base_folder": None,
            "host": None,
            "port": None,
            "database_name": None,
            "username": None,
            "password": None,
            "own": None,
        },
        "api": {
            "host": None,
            "port": None,
            "secret_key": None,
            "jwt_secret_key": None,
        },
    },
    resources_config={
        "update_frequency": 15,
        "cores_per_worker": 10,
        "max_workers": 1,
        "memory_per_worker": 250,
    },
    worker_sh="""#!/usr/bin/bash
conda activate p4_qcml
    """,
    conda_env="p4_qcml",
    start=False,
):
    """
    Updating any keys in the db_config dictionary will update the
    qcfractal_config.yaml file after generation
    """
    if QCF_BASE_FOLDER:
        os.environ["QCF_BASE_FOLDER"] = QCF_BASE_FOLDER
    if QCF_BASE_FOLDER is None:
        QCF_BASE_FOLDER = os.environ.get("QCF_BASE_FOLDER")
    print(QCF_BASE_FOLDER)
    if reset:
        if os.path.exists(QCF_BASE_FOLDER):
            os.remove(f"{QCF_BASE_FOLDER}/qcfractal_config.yaml")
        os.system(f"rm -r {QCF_BASE_FOLDER}/postgres")
    if QCF_BASE_FOLDER is None:
        print(
            "Set QCF_BASE_FOLDER environment variable to the base folder of the QCFractal installation"
        )
        raise ValueError("QCF_BASE_FOLDER environment variable not set")
    if not os.path.exists(QCF_BASE_FOLDER):
        os.makedirs(QCF_BASE_FOLDER)
    if not os.path.exists(f"{QCF_BASE_FOLDER}/qcfractal_config.yaml"):
        res = subprocess.check_output(
            [
                "qcfractal-server",
                f"--config={QCF_BASE_FOLDER}/qcfractal_config.yaml",
                "init-config",
            ]
        )
    config_yaml = yaml.safe_load(open(f"{QCF_BASE_FOLDER}/qcfractal_config.yaml", "r"))
    yaml_has_changed = False
    for key, value in db_config.items():
        if value and not isinstance(value, dict):
            config_yaml[key] = value
            yaml_has_changed = True
        elif value and isinstance(value, dict):
            for k, v in value.items():
                if v:
                    config_yaml[key][k] = v
                    yaml_has_changed = True
    if yaml_has_changed:
        with open(f"{QCF_BASE_FOLDER}/qcfractal_config.yaml", "w") as f:
            yaml.dump(config_yaml, f)
    if not os.path.exists(f"{QCF_BASE_FOLDER}/postgres"):
        print(f"{QCF_BASE_FOLDER}/postgres")
        subprocess.check_output(
            [
                "qcfractal-server",
                f"--config={QCF_BASE_FOLDER}/qcfractal_config.yaml",
                "init-db",
            ]
        )
    res = subprocess.check_output(
        [
            "qcfractal-server",
            f"--config={QCF_BASE_FOLDER}/qcfractal_config.yaml",
            "info",
        ]
    )
    print(res.decode("utf-8"))
    with open(f"{QCF_BASE_FOLDER}/worker.sh", "w") as f:
        f.write("""        """)
    if conda_env:
        conda_env_str = (
            "    environments:\n"
            "      use_manager_environment: False\n"
            "      conda:\n"
            f"        - {conda_env}"
        )
    else:
        conda_env_str = (
            "    environments:\n"
            "      use_manager_environment: True"
        )
    with open(f"{QCF_BASE_FOLDER}/resources.yml", "w") as f:
        f.write(
            f"""
# qcfractal-manager-config.yml
---
cluster: theoryfs           # descriptive name to present to QCFractal server
loglevel: INFO
logfile: qcfractal-manager.log
update_frequency: {resources_config.get('update_frequency', '60.0')}

server:
  fractal_uri: "http://localhost:{db_config["api"]["port"]}"      # e.g. https://qcarchive.molssi.org
  username: null
  password: null
  verify: False

executors:
  cpuqueue:
    type: local
    cores_per_worker: {resources_config.get('cores_per_worker', '8')}
    memory_per_worker: {resources_config.get('memory_per_worker', '8')}
    max_workers: {resources_config.get('max_workers', '4')}
    queue_tags:
      - '*'
{conda_env_str}
    worker_init:
      - source {QCF_BASE_FOLDER}/worker.sh
"""
        )
    if start:
        res = subprocess.check_output(
            [
                "qcfractal-server",
                f"--config={QCF_BASE_FOLDER}/qcfractal_config.yaml",
                "start",
                # "&",
                # "disown",
            ]
        )
        res = subprocess.check_output(
            [
                "qcfractal-compute-manager",
                "--config=$QCF_BASE_FOLDER/resources.yml",
                # "&", 
                # "disown",
            ]
        )
    print("QCFractal setup complete")
    print("To start the server run:")
    print(f"  qcfractal-server --config={QCF_BASE_FOLDER}/qcfractal_config.yaml start")
    print("To start the compute manager run:")
    print(f"  qcfractal-compute-manager --config={QCF_BASE_FOLDER}/resources.yml")
    return


def main():
    setup_qcarchive_qcfractal(
        start=False,
        reset=False,
        db_config={
            "name": None,
            "enable_security": "false",
            "allow_unauthenticated_read": None,
            "logfile": None,
            "loglevel": None,
            "service_frequency": 10,
            "max_active_services": None,
            "heartbeat_frequency": None,
            "log_access": None,
            "database": {
                "base_folder": None,
                "host": None,
                "port": 5432,
                "database_name": "qca",
                "username": None,
                "password": None,
                "own": None,
            },
            "api": {
                "host": None,
                "port": 7777,
                "secret_key": None,
                "jwt_secret_key": None,
            },
        },
    )
    return


if __name__ == "__main__":
    main()
