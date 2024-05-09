import os
import subprocess
from potholeClassifier import logger
from colorama import Fore, Style
from dotenv import load_dotenv
from typing import List

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")


def get_all_python_files(directory: str) -> List[str]:
    """
    Get all Python files within a given directory and its subdirectories.

    Args:
        directory (str): The directory to search for Python files.

    Returns:
        List[str]: A list of file paths to Python files.
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def execute_command(command: List[str]) -> subprocess.CompletedProcess:
    """
    Execute a command in the shell.

    Args:
        command (List[str]): The command and its arguments to execute.

    Returns:
        subprocess.CompletedProcess: The completed subprocess.
    """
    try:
        process = subprocess.run(command, capture_output=True, text=True)
        return process
    except FileNotFoundError as e:
        logger.error(f"Command '{command}' not found: {e}")
        raise e


def lint_python_file(file: str) -> None:
    """
    Lint a Python file using autopep8 and flake8.

    Args:
        file (str): Path to the Python file to lint.
    """
    autopep8_process = execute_command(
        ['autopep8', '--in-place', '--aggressive', '--aggressive', file])
    if autopep8_process.returncode != 0:
        logger.error(f"Autopep8 failed to lint file: {file}")
        logger.error(autopep8_process.stdout)
        return

    flake8_process = execute_command(['flake8', file])
    if flake8_process.returncode != 0:
        logger.error(f"Flake8 found errors in file: {file}")
        logger.error(flake8_process.stdout)
        print(f"{Fore.RED}{flake8_process.stdout}{Style.RESET_ALL}")
    else:
        print(f"Linted file: {file}")


def main():
    """
    Main function to lint all Python files in the project directory.
    """
    project_directory = ROOT_DIR
    all_python_files = get_all_python_files(project_directory)
    for file in all_python_files:
        lint_python_file(file)


if __name__ == "__main__":
    main()
