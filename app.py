# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:56:06 2024

@author: ayusi
"""

import streamlit as st
import joblib
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Load the trained SVC model
svc_model_path = r"C:\fingerprint\dataset_blood_group\svc_model.pkl"
svc_model = joblib.load(svc_model_path)

# Initialize Img2Vec model
img2vec = Img2Vec()

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(image)
    img_pil = transforms.ToPILImage()(img_tensor)
    img_features = img2vec.get_vec(img_pil)
    return img_features

# Streamlit app layout
st.image(r"C:\fingerprint\image .jpg", use_column_width=True)

st.title('Blood Group Prediction - SVC Model')

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction on the uploaded image
    if st.button('Make Prediction'):
        # Preprocess the image for the model
        preprocessed_image = preprocess_image(image)
        
        # Make prediction using the SVC model
        prediction = svc_model.predict([preprocessed_image])
        st.write('Prediction (SVC):', prediction[0])
