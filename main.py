# https://pneumoniapredictions-tgd943nppslyj6scnxeh98.streamlit.app/

import streamlit as st
from keras.models import load_model
from PIL import Image
from utils import classify, set_background
import os
import tensorflow as tf
import numpy as np

# Set background image
backgroundColor = "#00000"

# Set title 
st.title('Pneumonia Predictions')

# Set header 
st.header('Please upload a chest X-ray image')

# Upload file 
files = st.file_uploader('Upload Image', type=['jpeg', 'jpg', 'png'])

# Define model path
model_path = os.path.join("model", "pneumonia_classifier.h5")
labels_path = os.path.join("model", "labels.txt")

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file does not exist at path: {model_path}")

# Try loading the model with error handling
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except ValueError as e:
    print(f"ValueError: {e}")
except OSError as e:
    print(f"OSError: Could not load model due to an OS error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Load class names 
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"The labels file does not exist at path: {labels_path}")

with open(labels_path, 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# Display image and classify if a file is uploaded 
if files is not None:
    # Open and display the image
    image = Image.open(files).convert('RGB')
    st.image(image, use_container_width=True)  # Updated parameter

    # Classify the image with error handling
    try:
        class_name, conf_score = classify(image, model, class_names)
        st.write("## Predicted Class: {}".format(class_name))
        st.write("### Confidence Score: {:.2f}%".format(conf_score * 100))  # Display score as a percentage
    except Exception as e:
        st.error(f"Error during classification: {e}")
