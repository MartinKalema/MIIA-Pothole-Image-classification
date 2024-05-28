import streamlit as st
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from potholeClassifier.utils.common import read_yaml
import numpy as np
from pathlib import Path


class Classifier:
    def __init__(self, config_file_path: Path):
        """
        Constructor method for the Classifier class.

        Args:
            config_file_path (Path): Path to the configuration file.
        """
        self.config_file_path = config_file_path
        self.config = read_yaml(self.config_file_path)
        self.class_names = ["No Pothole", "Has Pothole"]
        self.model = None

    def load_best_model(self) -> None:
        """
        Method to load the best trained model from the specified path.
        """
        if self.model is None:
            self.model = load_model(self.config.training.trained_model_path)

    def import_and_predict(self, image_data: Image) -> np.ndarray:
        """
        Method to preprocess the image data and make predictions using the loaded model.

        Args:
            image_data (Image): Input image data.

        Returns:
            np.array: Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        size = (224, 224)
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis, ...]
        prediction = self.model.predict(img_reshape)
        return prediction

    def classify_image(self, image: Image) -> str:
        """
        Method to classify the input image.

        Args:
            image (Image): Input image data.

        Returns:
            str: Classification result text.
        """
        predictions = self.import_and_predict(image)
        score = tf.nn.softmax(predictions[0])
        result_text = "This image most likely belongs to the <b>{}</b> class.".format(
            self.class_names[np.argmax(score)])
        return result_text


class StreamlitApp:
    def __init__(self, classifier: Classifier):
        """
        Constructor method for the StreamlitApp class.

        Args:
            classifier (Classifier): Instance of the Classifier class.
        """
        self.classifier = classifier

    def load_model(self) -> None:
        """
        Method to run the Streamlit application.
        """
        with st.spinner('Model is being loaded..'):
            self.classifier.load_best_model()

    def display_intro(self) -> None:
        """Display the intro text"""

        st.markdown("## Pothole Image Classification", unsafe_allow_html=True)
        st.write("""
        Potholes are fatal and can cause severe damage to vehicles as well as can cause deadly accidents. In South Asian countries, pavement
        distresses are the primary cause due to poor subgrade conditions, lack of subsurface drainage, and excessive rainfalls. This prediction service classifies images to find whether they have potholes or not.
        """)

    def run(self) -> None:
        """
        Run the app
        
        """

        self.load_model()
        self.display_intro()
        file = st.file_uploader(
            "Please upload the image file", type=[
                "jpg", "png"])

        if file is None:
            st.text("File has not been uploaded yet.")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            result_text = self.classifier.classify_image(image)
            st.markdown(result_text, unsafe_allow_html=True)


if __name__ == "__main__":
    CONFIG_FILE_PATH = Path("config/config.yaml")
    classifier = Classifier(CONFIG_FILE_PATH)
    app = StreamlitApp(classifier)
    app.run()
