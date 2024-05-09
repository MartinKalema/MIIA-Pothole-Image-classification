import tensorflow as tf
from datetime import datetime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from potholeClassifier.entity.config_entity import TrainingConfig
import numpy as np


class Training:
    """
    Class for training a machine learning model using configured parameters and data.

    This class is responsible for loading the updated base model, preparing data generators for training and validation,
    and training the model using the provided data.

    Attributes:
        config (TrainingConfig): The configuration for training the model.

    Methods:
        get_base_model(): Loads the updated base model for training.
        train_valid_generator(): Prepares data generators for training and validation.
        train(): Trains and saves the best model.

    """

    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training object with the provided configuration.

        Args:
            config (TrainingConfig): The configuration for training the model.
        """
        self.config = config

    def get_base_model(self):
        """
        Loads the updated base model for training.

        This method loads the updated base model from the specified path in the training configuration.
        """
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )

    def train_valid_generator(self):
        """
        Prepares data generators for training and validation.

        This method prepares data generators for training and validation using the specified parameters
        in the training configuration. It applies data augmentation techniques if enabled.
        """
        # Data generator and flow configuration parameters
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
        )

        # Prepare training data generator with or without augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=20,
                height_shift_range=20,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs)

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='training',
            shuffle=True,
            class_mode='categorical',
            **dataflow_kwargs
        )

        self.valid_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle=True,
            class_mode='categorical',
            **dataflow_kwargs
        )

    def train(self):
        """Train the model using the provided training generator and validation data.

        Args:
            callback_list (list): A list of callbacks to be used during training.
        """
        # Calculate steps per epoch and validation steps based on generator
        # samples and batch size
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=1e-5)

        checkpoint = ModelCheckpoint(filepath=self.config.trained_model_path,
                                     verbose=1,
                                     save_best_only=True)

        callbacks = [checkpoint, lr_reducer]

        start = datetime.now()

        self.model.fit_generator(
            generator=self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1)
