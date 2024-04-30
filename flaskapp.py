from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from potholeClassifier.utils.common import read_yaml
import numpy as np
from pathlib import Path
import io

app = Flask(__name__)

class Classifier:
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
        image = Image.open(io.BytesIO(image_data))
        image = ImageOps.fit(image, size)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis, ...]
        prediction = self.model.predict(img_reshape)
        return prediction

    def classify_image(self, image_data):
        predictions = self.import_and_predict(image_data)
        score = tf.nn.softmax(predictions[0])
        result_text = "This image most likely belongs to the <b>{}</b> class.".format(self.class_names[np.argmax(score)])
        return result_text

classifier = Classifier(Path("config/config.yaml"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    image_file = request.files['image']
    if image_file:
        image_data = image_file.read()
        result_text = classifier.classify_image(image_data)
        return jsonify({'result': result_text})
    else:
        return jsonify({'error': 'No image file provided'})

if __name__ == "__main__":
    app.run(debug=True)
