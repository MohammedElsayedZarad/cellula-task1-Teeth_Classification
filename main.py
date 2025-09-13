import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

model_path = r"C:\Users\USER\Desktop\My_World\Learnings\Courses\cellula intern Cv\project 1\oral_model2.keras"
model = tf.keras.models.load_model(model_path, compile=False)

class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0   
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


st.title('Teeth Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_image, caption="Uploaded Image", width=150)

    with col2:
        if st.button('Classify'):
            img_array = preprocess_image(uploaded_image)
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            confidence = np.max(result) * 100

            st.success(f"Prediction: **{class_names[predicted_class]}** ({confidence:.2f}%)")
