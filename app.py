import streamlit as st
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from potholeClassifier.utils.common import read_yaml
import numpy as np
from pathlib import Path

class PotholeClassifier:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = read_yaml(self.config_file_path)
        self.class_names = ["No Pothole", "Has Pothole"]
        self.model = None

    def load_best_model(self):
        if self.model is None:
            self.model = load_model(self.config.training.trained_model_path)
        return self.model

    def import_and_predict(self, image_data):
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")

        size = (224, 224)
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis, ...]
        prediction = self.model.predict(img_reshape)
        return prediction

    def classify_image(self, image):
        predictions = self.import_and_predict(image)
        score = tf.nn.softmax(predictions[0])
        result_text = "This image most likely belongs to the <b>{}</b> class.".format(self.class_names[np.argmax(score)])
        return result_text

class StreamlitApp:
    def __init__(self, classifier):
        self.classifier = classifier

    def run(self):
        st.markdown("## Pothole Image Classification", unsafe_allow_html=True)
        
        with st.spinner('Model is being loaded..'):
            self.classifier.load_best_model()

        st.write("""
        Potholes are fatal and can cause severe damage to vehicles as well as can cause deadly accidents. In South Asian countries, pavement
        distresses are the primary cause due to poor subgrade conditions, lack of subsurface drainage, and excessive rainfalls. This prediction service classifies images to find whether they have potholes or not.
        """)

        file = st.file_uploader("Please upload the image file", type=["jpg", "png"])

        if file is None:
            st.text("File has not been uploaded yet.")
        else: 
            image = Image.open(file)
            st.image(image, use_column_width=True)
            result_text = self.classifier.classify_image(image)
            st.markdown(result_text, unsafe_allow_html=True)

if __name__ == "__main__":
    CONFIG_FILE_PATH = Path("config/config.yaml")
    classifier = PotholeClassifier(CONFIG_FILE_PATH)
    app = StreamlitApp(classifier)
    app.run()
