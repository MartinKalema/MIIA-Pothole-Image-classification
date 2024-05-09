import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras as mlk
from urllib.parse import urlparse
from potholeClassifier.entity.config_entity import EvaluationConfig
from potholeClassifier.utils.common import save_json


class Evaluation:
    """
    Class for evaluating a machine learning model.

    Attributes:
        config (EvaluationConfig): Configuration object containing evaluation parameters.
        model (tf.keras.Model): Loaded machine learning model.
        score (list): Evaluation score of the model.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """
        Initializes the Evaluation object.

        Args:
            config (EvaluationConfig): Configuration object containing evaluation parameters.
        """
        self.config = config

    def valid_generator(self) -> None:
        """
        Prepares data generators for validation.

        Prepares data generators for validation using the specified parameters in the configuration.
        Applies data augmentation techniques if enabled.
        """
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle=True,
            class_mode='categorical',
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads a trained model from the specified path.

        Args:
            path (Path): Path to the trained model file.

        Returns:
            tf.keras.Model: Loaded machine learning model.
        """
        return tf.keras.models.load_model(path)

    def evaluation(self) -> None:
        """
        Performs evaluation of the loaded model using validation data.
        """
        self.model = self.load_model(self.config.path_of_model)
        self.valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self) -> None:
        """
        Saves the evaluation score to a JSON file.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self) -> None:
        """
        Logs evaluation metrics and the model into MLflow.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            if tracking_url_type_store != "file":
                mlk.log_model(
                    self.model,
                    "model",
                    registered_model_name="PotholeClassificationModel")
            else:
                mlk.keras.log_model(self.model, "model")
