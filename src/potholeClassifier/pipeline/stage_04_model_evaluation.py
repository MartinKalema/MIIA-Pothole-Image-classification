from potholeClassifier.config.configuration import ConfigurationManager
from potholeClassifier.components.model_evaluation import Evaluation
from potholeClassifier import logger

STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(
            f">>>>>> {STAGE_NAME} completed <<<<<<<\n\n**********************************")
    except Exception as e:
        logger.exception(e)
        raise e