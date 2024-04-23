from potholeClassifier import logger
from potholeClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info("*********************************\n")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(
        f">>>>>> {STAGE_NAME} completed <<<<<<<\n**********************************")
except Exception as e:
    logger.exception(e)
    raise e