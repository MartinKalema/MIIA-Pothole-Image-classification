from potholeClassifier.config.configuration import ConfigurationManager
from potholeClassifier.components.model_evaluation import Evaluation
from potholeClassifier import logger

STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:
    """
    Class representing the pipeline for model evaluation.

    This class orchestrates the model evaluation pipeline by initializing the required
    components and executing the necessary steps.

    Methods:
        main: Main method to execute the model evaluation pipeline.
    """

    def main(self) -> None:
        """
        Executes the main steps of the model evaluation pipeline.

        Initializes the configuration manager, retrieves evaluation configuration,
        performs evaluation of the model, and logs evaluation metrics into MLflow.
        """
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
            f">>>>>> {STAGE_NAME} completed <<<<<<<\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
