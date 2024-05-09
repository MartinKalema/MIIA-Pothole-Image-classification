import os
import zipfile
import gdown
from potholeClassifier import logger
from potholeClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize DataIngestion object with the provided configuration.

        Args:
            config (DataIngestionConfig): Configuration object for data ingestion.
        """
        self.config = config

    def download_file(self) -> str:
        """Fetch data from a URL.

        Returns:
            str: The path of the downloaded file.

        Raises:
            Exception: If an error occurs during the download process.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            logger.info(
                f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, zip_download_dir, quiet=False)

            logger.info(
                f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            return zip_download_dir

        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise e

    def extract_zip_file(self) -> str:
        """Extract a zip file.

            This method extracts the contents of a zip file specified in the configuration
            to the directory specified in the configuration.

            Returns:
                str: The path of the extracted directory.

            Raises:
                Exception: If an error occurs during the extraction process.
            """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extracted zip file into: {unzip_path}")

            return unzip_path

        except Exception as e:
            logger.error(
                f"Error extracting zip file: {self.config.local_data_file}")
            raise e
