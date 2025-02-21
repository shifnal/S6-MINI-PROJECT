import os
import json
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import io

# Set the page config first as required
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")

# Load the pre-trained model only once (caching it to avoid reloading every time)
@st.cache_resource  # Caching the model to avoid repeated loading
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Load class indices only once
@st.cache_resource  # Caching the class indices
def load_class_indices(class_file_path):
    with open(class_file_path, 'r') as f:
        return json.load(f)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
class_file_path = f"{working_dir}/class_indices.json"

model = load_model(model_path)
class_indices = load_class_indices(class_file_path)

# Optimized Image Preprocessing with Caching
@st.cache_data  # Cache the function to optimize the preprocessing
def load_and_preprocess_image(image, target_size=(224, 224)):  
    # Efficient loading and resizing using Pillow
    img = Image.open(image).resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)  # Process image for prediction
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Convert image to base64 for background
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Convert image to base64 for background image in CSS
background_image_base64 = image_to_base64(Image.open("images/background_image.jpg"))
background_image = f'url(data:image/jpeg;base64,{background_image_base64})'

# Add background image using custom HTML and CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: {background_image};
        background-size: cover;
        background-position: center;
        height: 100vh; /* Ensure the background covers the whole screen height */
    }}

    /* Reduce width of the sidebar */
    .css-1d391kg {{
        width: 200px; /* Set the desired width of the sidebar */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation with "Home" and "About" options
menu = st.sidebar.selectbox("Menu", ["Home", "About"])

if menu == "Home":
    st.markdown("<h1 style='color: white;'>Plant Disease Classifier</h1>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            # Resize and display the image with a white outline without gap
            resized_img = image.resize((150, 150))
            img_base64 = image_to_base64(resized_img)
            st.markdown(
                f"""
                <div style="border: 5px solid white; display: inline-block; padding: 0;">
                    <img src="data:image/png;base64,{img_base64}" style="width: 150px; height: 150px;" />
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            # Only classify when the user presses the button
            if st.button('Classify'):
                with st.spinner('Classifying...'):  # Show a spinner while the model classifies
                    # Perform prediction on the uploaded image
                    prediction = predict_image_class(model, uploaded_image, class_indices)
                    st.markdown(f"<p style='color: blue; background-color: white; padding: 10px; border-radius: 5px;'>Prediction: {str(prediction)}</p>", unsafe_allow_html=True)

elif menu == "About":
    # Remove the background image only on the "About" page
    st.markdown(
        """
        <style>
        .stApp {
            background-image: none !important;
            background-color: transparent !important;
        }
        /* Make text on About page white */
        .about-page-text h1, .about-page-text h2, .about-page-text h3, .about-page-text p, .about-page-text div {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
   
    # About page content with white text
    st.markdown("<div class='about-page-text' style='padding: 20px;'>", unsafe_allow_html=True)
    st.markdown("<h1>About the Plant Disease Classifier</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        This application is designed to classify plant diseases based on images uploaded by the user.
        It leverages machine learning models to identify potential diseases from the images of plants.
        The app is built using Streamlit, TensorFlow, and various other libraries to facilitate easy and
        user-friendly classification.
       
        The primary goal is to assist farmers, gardeners, and researchers in diagnosing plant diseases
        and taking prompt action to ensure plant health and productivity.
       
        ### Features:
        - Upload a plant image for disease classification.
        - Get predictions for different plant diseases.
        - Easy to use interface.
       
        ### Developed by:
        Group - 7 Members:
        - Nadhil Farzeen
        - Shifnal Shyju P
        - Tristin Titus
        - Nidhin Joy
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)