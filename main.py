import streamlit as st
from keras.models import load_model
from PIL import Image
from utils import classify, set_background

set_background('./bgs/black.jpg')

# Set title 
st.title('Pneumonia Classification')

# Set header 
st.header('Please upload a chest X-ray image')

# Upload file 
files = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/pneumonia_classifier.h5')

# load class names 
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display images 
if files is not None:
    image = Image.open(files).convert('RGB')
    st.image(image, use_column_width=True)

    # classify images 
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### Score: {}".format(conf_score))