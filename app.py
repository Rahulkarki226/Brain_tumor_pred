import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Disable oneDNN optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



model_path = 'brain_tumer_file.h5'
# Load the model
model = load_model(model_path)
labels = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'No_tumour', 3: 'pitutory_tumor'}


def preprocess_image(image):
    image = image.resize((150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def getResult(image):
    predictions = model.predict(image)
    return predictions


# Streamlit UI
st.title('Brain Tumor Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    image = preprocess_image(image)
    predictions = getResult(image)

    # Determine the predicted label
    predicted_label = labels[np.argmax(predictions)]
    st.write(f"Prediction: {predicted_label}")
