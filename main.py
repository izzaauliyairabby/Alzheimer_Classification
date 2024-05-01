import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('path_to_your_model/model.h5')
    return model

# Title
st.title('Alzheimer MRI Image Classification with Streamlit')

# Upload image
uploaded_file = st.file_uploader("Choose an image or scanned MRI..", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)

    # Load the model
    model = load_model()

    # Perform inference
    prediction = model.predict(image)
    
    # Class labels
    class_names = ['Mild Demented', 'Very Mild Demented', 'Moderate Demented', "Non Demented"]  # Replace with your actual class labels
    
    # Display prediction
    st.write(f"Prediction: {class_names[np.argmax(prediction)]}, Confidence: {np.max(prediction)*100:.2f}%")
