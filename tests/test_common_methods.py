import pytest
from potholeClassifier.utils.common import *
from potholeClassifier.constants import *
from box import ConfigBox


def test_read_yaml_valid_file() -> None:
    """
    Test that the read_yaml function correctly reads a valid YAML file and 
    returns an instance of the ConfigBox class.

    Steps:
    1. Call read_yaml with CONFIG_FILE_PATH.
    2. Assert that the result is an instance of ConfigBox.

    Expected Outcome:
    The function should return an object of type ConfigBox.
    """
    result = read_yaml(CONFIG_FILE_PATH)
    assert isinstance(result, ConfigBox)


def test_read_yaml_nonexistent_file() -> None:
    """
    Test that the read_yaml function raises a FileNotFoundError when provided 
    with a path to a non-existent file.

    Steps:
    1. Use pytest.raises to catch a FileNotFoundError.
    2. Call read_yaml with the EMPTY_FILE_PATH.

    Expected Outcome:
    The function should raise a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        read_yaml(Path(EMPTY_FILE_PATH))
