from potholeClassifier.config.configuration import ConfigurationManager
from potholeClassifier.components.data_ingestion import DataIngestion
from potholeClassifier import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionPipeline:
    """
    Class representing the data ingestion pipeline.

    This class orchestrates the data ingestion pipeline by initializing the required
    components, downloading files, and extracting data.

    Methods:
        main: Main method to execute the data ingestion pipeline.
    """

    def main(self) -> None:
        """
        Executes the main steps of the data ingestion pipeline.

        Initializes the configuration manager, retrieves data ingestion configuration,
        performs data ingestion by downloading files and extracting data from them.
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(
            f">>>>>> {STAGE_NAME} completed <<<<<<<\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
