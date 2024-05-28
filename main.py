import logging
from potholeClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from potholeClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from potholeClassifier.pipeline.stage_03_model_training import TrainingPipeline
from potholeClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

logger = logging.getLogger('potholeClassifierLogger')


def execute_pipeline_stage(stage_name: str, pipeline_obj: object) -> None:
    """
    Execute a specific pipeline stage.

    Args:
        stage_name (str): The name of the pipeline stage.
        pipeline_obj: The pipeline object to execute.

    Returns:
        None
    """
    try:
        logger.info(f">>>>>> {stage_name} STARTED <<<<<<")
        pipeline_obj.main()
        logger.info(
            f">>>>>> {stage_name} COMPLETED <<<<<<<\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    STAGE_NAME = "DATA INGESTION"
    data_ingestion = DataIngestionPipeline()
    execute_pipeline_stage(STAGE_NAME, data_ingestion)

    STAGE_NAME = "PREPARING BASE MODEL"
    obj = PrepareBaseModelPipeline()
    execute_pipeline_stage(STAGE_NAME, obj)

    STAGE_NAME = "MODEL TRAINING"
    obj = TrainingPipeline()
    execute_pipeline_stage(STAGE_NAME, obj)

    STAGE_NAME = "MODEL EVALUATION"
    obj = EvaluationPipeline()
    execute_pipeline_stage(STAGE_NAME, obj)
