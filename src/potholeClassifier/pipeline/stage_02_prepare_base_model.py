from potholeClassifier.config.configuration import ConfigurationManager
from potholeClassifier.components.prepare_base_model import PrepareBaseModel
from potholeClassifier import logger

STAGE_NAME = "Base Model Preparation Stage"


class PrepareBaseModelPipeline:
    """
    Class representing the pipeline for preparing the base model.

    This class orchestrates the preparation of the base model by initializing the required
    components and executing the necessary steps.

    Methods:
        main: Main method to execute the base model preparation pipeline.
    """

    def main(self) -> None:
        """
        Executes the main steps of the base model preparation pipeline.

        Initializes the configuration manager, retrieves base model configuration,
        prepares the base model, and saves it.
        """
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = PrepareBaseModel(config=base_model_config)
        base_model._save_base_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(
            f">>>>>> {STAGE_NAME} completed <<<<<<<\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
