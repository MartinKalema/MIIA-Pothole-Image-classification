from potholeClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class PrepareBaseModel:
    """Class for preparing base models.

    This class provides methods preparing a full model and saving it to a specified path.

    Attributes:
        config (PrepareBaseModelConfig): The configuration for preparing base models.
    """

    def __init__(self, config: PrepareBaseModelConfig):
        """Initializes the PrepareBaseModel.

        Args:
            config (PrepareBaseModelConfig): The configuration for preparing base models.
        """
        self.config = config

    @staticmethod
    def _prepare_full_model(conv_1_filters, conv_2_filters, conv_3_filters, conv_4_filters, dense_units, number_of_classes, image_size):
        """Prepares the full model by freezing specified layers and adding additional layers.

        Args:
            number_of_classes (int): The number of classes in the model.
            conv_1_filters: Number of filters on the first convolutional layer
            conv_2_filters: Number of filters on the second convolutional layer
            conv_3_filters: Number of filters on the third convolutional layer
            conv_4_filters: Number of filters on the fourth convolutional layer
            dense_units: Number of neurons in the fully connected dense layer
            image_size: Input shape of the model

        Returns:
            tf.keras.Model: The prepared full model.
        """
        full_model = Sequential([
            Conv2D(conv_1_filters, (3, 3), activation='relu', input_shape=image_size),
            MaxPooling2D((2, 2)),
            Conv2D(conv_2_filters, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(conv_3_filters, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(conv_4_filters, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(dense_units, activation='relu'),
            Dense(number_of_classes, activation='softmax')
        ])

        # Compile the model
        full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        full_model.summary()
        return full_model
    
    def _save_base_model(self):
        self.full_model = self._prepare_full_model(
            number_of_classes=self.config.params_classes,
            dense_units=self.config.params_dense_units,
            conv_1_filters=self.config.params_conv_1_filters,
            conv_2_filters=self.config.params_conv_2_filters,
            conv_3_filters=self.config.params_conv_3_filters,
            conv_4_filters=self.config.params_conv_4_filters,
            image_size=self.config.params_image_size
        )

        self._save_model(path=self.config.base_model_path, model=self.full_model)

    @staticmethod
    def _save_model(path: Path, model: tf.keras.Model):
        """Saves the model to the specified path.

        Args:
            path (Path): The path where the model will be saved.
            model (tf.keras.Model): The model to be saved.
        """
        model.save(path)
