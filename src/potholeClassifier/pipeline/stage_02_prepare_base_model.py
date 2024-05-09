from potholeClassifier.config.configuration import ConfigurationManager
from potholeClassifier.components.prepare_base_model import PrepareBaseModel
from potholeClassifier import logger

STAGE_NAME = "Base Model Preparation Stage"


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
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
            f">>>>>> {STAGE_NAME} completed <<<<<<<\n\n**********************************")
    except Exception as e:
        logger.exception(e)
        raise e
