import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s %(filename)s]: %(message)s:')

project_name = "potholeClassifier"


def create_directories(filepath: Path) -> None:
    """
    Create directories if they don't exist.

    Args:
        filepath (Path): Path to the file or directory.
    """
    filedir = filepath.parent
    if filedir != Path(""):
        filedir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")


def create_empty_file(filepath: Path) -> None:
    """
    Create an empty file if it doesn't exist or has zero size.

    Args:
        filepath (Path): Path to the file.
    """
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")


def process_filepath(filepath: Path) -> None:
    """
    Process a file path by creating directories and empty files.

    Args:
        filepath (Path): Path to the file.
    """
    create_directories(filepath)
    create_empty_file(filepath)


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/prepare_base_model.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/utils/_init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_prepare_base_model.py",
    f"src/{project_name}/pipeline/stage_03_model_training.py",
    f"src/{project_name}/pipeline/stage_04_model_evaluation.py",
    f"src/{project_name}/pipeline/stage_05_prediction.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "main.py",
    "dvc.yaml",
    "Dockerfile",
    "logs/logfile.log",
    "research/01_data_ingestion.ipynb",
    "research/02_prepare_base_model.ipynb",
    "research/03_model_training.ipynb",
    "research/04_model_evaluation.ipynb",
    "templates/index.html",
    'lint.py',
    '.env'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    process_filepath(filepath)
