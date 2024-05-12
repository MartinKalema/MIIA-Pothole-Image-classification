import subprocess
from potholeClassifier.utils.common import read_yaml
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")


def read_config() -> dict:
    """
    Read the configuration from the YAML file.

    Returns:
        dict: Configuration parameters.
    """
    return read_yaml(CONFIG_FILE_PATH)


def set_mlflow_environment_variables(config: dict) -> None:
    """
    Set MLflow environment variables based on the configuration.

    Args:
        config (dict): Configuration parameters.

    Returns:
        None
    """
    commands = [
        f"export MLFLOW_TRACKING_URI={config.model_evaluation.mlflow_tracking_uri}",
        f"export MLFLOW_TRACKING_USERNAME={config.model_evaluation.mlflow_tracking_username}",
        f"export MLFLOW_TRACKING_PASSWORD={config.model_evaluation.mlflow_tracking_password}",
    ]

    for command in commands:
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")


if __name__ == "__main__":
    config = read_config()
    set_mlflow_environment_variables(config)
