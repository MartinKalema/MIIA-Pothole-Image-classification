import subprocess
from potholeClassifier.utils.common import read_yaml
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")


def run_commands():
    config = read_yaml(CONFIG_FILE_PATH)
    commands = [
        f"export MLFLOW_TRACKING_URI={config.model_evaluation.mlflow_tracking_uri}",
        f"export MLFLOW_TRACKING_USERNAME={config.model_evaluation.mlflow_tracking_username}",
        f"export MLFLOW_TRACKING_PASSWORD={config.model_evaluation.mlflow_tracking_password}",
    ]

    for command in commands:
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    run_commands()
