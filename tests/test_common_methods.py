import pytest
from potholeClassifier.utils.common import *
from potholeClassifier.constants import *
from box import ConfigBox
import os
import tempfile
import shutil
from io import BytesIO
from unittest.mock import patch


def test_read_yaml_valid_file() -> None:
    result = read_yaml(CONFIG_FILE_PATH)
    assert isinstance(result, ConfigBox)


def test_read_yaml_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        read_yaml(Path(EMPTY_FILE_PATH))
