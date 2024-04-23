from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration class for data ingestion process.

    Attributes:
        root_dir (Path): The root directory where data will be stored or processed.
        source_URL (str): The URL from which data will be fetched.
        local_data_file (Path): The local file path where the downloaded data will be stored.
        unzip_dir (Path): The directory where the downloaded data will be extracted or unzipped.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
