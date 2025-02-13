import os
import numpy as np
import cv2
import json
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# Load Class Labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Preprocessing Function
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload Route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded!", class_name=None, image_path=None)

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file!", class_name=None, image_path=None)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("static", filename)
            file.save(file_path)

            img = preprocess_image(file)
            prediction = model.predict(img)
            class_idx = np.argmax(prediction)
            class_name = class_labels[str(class_idx)]  # Ensure class_labels is a dictionary with string keys

            return render_template('index.html', class_name=class_name, image_path=file_path, error=None)

    return render_template('index.html', class_name=None, image_path=None, error=None)

if __name__ == '__main__':
    app.run(debug=True)
