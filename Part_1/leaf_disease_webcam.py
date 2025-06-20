import streamlit as st
import cv2
import numpy as np
import time
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# Step 1: Download model.h5 if not present
# -------------------------
MODEL_URL = "https://huggingface.co/spaces/etahamad/plant-disease-detection/resolve/main/model.h5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model.h5..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")

# -------------------------
# Step 2: Load the trained model
# -------------------------
model = load_model(MODEL_PATH)

#  Full class labels used during training
labels = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Gray Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Orange Huanglongbing (Citrus Greening)",
    "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites Two-Spotted",
    "Tomato Target Spot", "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus", "Tomato Healthy"
]

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0

# -------------------------
# Streamlit UI
# -------------------------
st.title(" Real-Time Leaf Disease Detection ")
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

last_prediction_time = 0
display_text = "Adjust the leaf in front of the camera..."

while run:
    success, frame = camera.read()
    if not success:
        st.warning(" Failed to grab frame from webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()

    if current_time - last_prediction_time >= 10:
        input_img = preprocess_frame(frame_rgb)
        prediction = model.predict(input_img)[0]
        label_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if label_index < len(labels):
            full_label = labels[label_index]
        else:
            full_label = "Unknown"

        # Extract plant and disease
        try:
            parts = full_label.split(" ", 1)
            plant = parts[0]
            disease = parts[1] if len(parts) > 1 else "Healthy"
        except:
            plant, disease = "Unknown", "Unknown"

        display_text = f" Plant: {plant} | Disease: {disease} ({confidence:.2f})"
        last_prediction_time = current_time

    # Show prediction on the frame
    cv2.putText(frame_rgb, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    FRAME_WINDOW.image(frame_rgb, channels="RGB")

camera.release()