import pytest
from potholeClassifier.utils.common import * 
from potholeClassifier.constants import *
from box import ConfigBox


def test_read_yaml() -> None:
    result = read_yaml(CONFIG_FILE_PATH)
    assert isinstance(result, ConfigBox)