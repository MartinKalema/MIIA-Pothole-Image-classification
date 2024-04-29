from potholeClassifier import logger
from potholeClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from potholeClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from potholeClassifier.pipeline.stage_03_model_training import TrainingPipeline

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

STAGE_NAME = "Prepare Base Model Stage"

try:
    logger.info("*********************************\n")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>> {STAGE_NAME} completed <<<<<<<\n **********************************")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"

try:
    logger.info("*********************************\n")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>> {STAGE_NAME} completed <<<<<<<\n **********************************")
except Exception as e:
    logger.exception(e)
    raise e