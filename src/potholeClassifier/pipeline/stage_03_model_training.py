from potholeClassifier.config.configuration import ConfigurationManager
from potholeClassifier.components.model_training import Training
from potholeClassifier import logger

STAGE_NAME = "Model Training Stage"


class TrainingPipeline:
    """
    Class representing the pipeline for model training.

    This class orchestrates the model training pipeline by initializing the required
    components and executing the necessary steps.

    Methods:
        main: Main method to execute the model training pipeline.
    """

    def main(self) -> None:
        """
        Executes the main steps of the model training pipeline.

        Initializes the configuration manager, retrieves training configuration,
        prepares the training data, trains the model, and saves the trained model.
        """
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> {STAGE_NAME} completed <<<<<<<\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
