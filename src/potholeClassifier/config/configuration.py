from potholeClassifier.constants import *
from potholeClassifier.utils.common import read_yaml, create_directories, save_json
from potholeClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
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
    
    def get_base_model_config(self) -> PrepareBaseModelConfig:
        """Retrieves the configuration for preparing base models.

        Returns:
            PrepareBaseModelConfig: The configuration for preparing base models.
        """
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_dense_units=self.params.DENSE_UNITS,
            params_conv_1_filters=self.params.CONV_1_FILTERS,
            params_conv_2_filters=self.params.CONV_2_FILTERS,
            params_conv_3_filters=self.params.CONV_3_FILTERS,
            params_conv_4_filters=self.params.CONV_4_FILTERS,
            params_classes=self.params.CLASSES
        )

        return base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves the training configuration parameters and constructs a TrainingConfig object.

        This method extracts the training configuration parameters from the overall configuration and parameters files,
        constructs the path to the training data directory, creates necessary directories, and packages all the parameters
        into a TrainingConfig object.

        Returns:
            TrainingConfig: An instance of TrainingConfig containing the training configuration parameters.

        Raises:
            ValueError: If any required configuration parameter is missing or invalid.
        """
        training = self.config.training
        base_model_config = self.config.prepare_base_model
        params = self.params

        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "dataset/train")

        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(base_model_config.base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE,
            params_is_augmentation=params.AUGMENTATION
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Get evaluation configuration for model evaluation.

        Returns:
            EvaluationConfig: A data class containing evaluation configuration parameters.

        Raises:
            ValueError: If any required configuration parameter is missing.
        """

        evaluation_config = EvaluationConfig(
            path_of_model=self.config.training.trained_model_path,
            training_data=os.path.join(self.config.data_ingestion.unzip_dir, "dataset/train"),
            mlflow_uri=self.config.model_evaluation.mlflow_tracking_uri,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return evaluation_config