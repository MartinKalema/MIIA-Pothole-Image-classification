from potholeClassifier.constants import *
from potholeClassifier.utils.common import read_yaml, create_directories, save_json
from potholeClassifier.entity.config_entity import DataIngestionConfig
import os

class ConfigurationManager:
    """Class for managing configuration files and preparing base models.
    
    This class handles the loading of configuration files and parameters,
    as well as the creation of directories necessary for preparing base models.
    
    Attributes:
        config_filepath (str, optional): The filepath of the configuration file. Defaults to CONFIG_FILE_PATH.
        params_filepath (str, optional): The filepath of the parameters file. Defaults to PARAMS_FILE_PATH.
    """
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        """Initializes the ConfigurationManager.

        Args:
            config_filepath (str, optional): The filepath of the configuration file. Defaults to CONFIG_FILE_PATH.
            params_filepath (str, optional): The filepath of the parameters file. Defaults to PARAMS_FILE_PATH.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Retrieves the data ingestion configuration from the overall configuration.

        Returns:
            DataIngestionConfig: The configuration for data ingestion process.

        Raises:
            ValueError: If any required configuration parameter is missing or invalid.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config