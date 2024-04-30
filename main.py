from potholeClassifier import logger
from potholeClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from potholeClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from potholeClassifier.pipeline.stage_03_model_training import TrainingPipeline
from potholeClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

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
    obj = PrepareBaseModelPipeline()
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


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info("*********************************\n")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(
        f">>>>>> {STAGE_NAME} completed <<<<<<<\n **********************************")
except Exception as e:
    logger.exception(e)
    raise e