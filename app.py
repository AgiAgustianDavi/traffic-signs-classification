import os
import numpy as np
import cv2
import json
import tensorflow as tf
import streamlit as st

# Load Model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# Load Class Labels
with open("class_labels_fix.json", "r") as f:
    class_labels = json.load(f)

# Preprocessing Function
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit Interface
st.title('Traffic Sign Classification')

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess and predict
    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_name = class_labels[str(class_idx)]

    # Display prediction result
    st.write(f"Predicted Class: {class_name}")
