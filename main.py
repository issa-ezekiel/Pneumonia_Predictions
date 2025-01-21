import streamlit as st
from keras.models import load_model
from PIL import Image
from utils import classify, set_background

# Set background image
backgroundColor = "#00000"

# Set title 
st.title('Pneumonia Classification')

# Set header 
st.header('Please upload a chest X-ray image')

# Upload file 
files = st.file_uploader('Upload Image', type=['jpeg', 'jpg', 'png'])

# Load classifier model
model = load_model('utils.py')

# Load class names 
with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# Display image and classify if a file is uploaded 
if files is not None:
    # Open and display the image
    image = Image.open(files).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification results
    st.write("## Predicted Class: {}".format(class_name))
    st.write("### Confidence Score: {:.2f}%".format(conf_score * 100))  # Display score as a percentage
