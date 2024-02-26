import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from keras.applications import DenseNet121
from keras.applications.imagenet_utils import preprocess_input

# Load the pre-trained DenseNet121 model
model = DenseNet121(weights='imagenet', include_top=True)

# Function to preprocess the image
def preprocess(image):
    """
    Preprocesses an image to prepare it for input into a deep learning model.
    
    Args:
        image: A numpy array representing the image to be preprocessed
        
    Returns:
        A numpy array containing the preprocessed image
    """
    # Resize the image to (224, 224) as DenseNet121 requires this size
    resized = cv2.resize(image, (224, 224))
    
    # Expand dimensions to create a batch dimension
    x = np.expand_dims(resized, axis=0)
    
    # Preprocess the image using the preprocess_input function from DenseNet
    x = preprocess_input(x)
    
    return x

# Define the Streamlit app
def app():
    # Set the title and description of the app
    st.title('Pneumonia Detection')
    st.markdown('Upload a chest X-ray image to check for pneumonia.')

    # Create a file uploader for the X-ray image
    file = st.file_uploader('Upload image here:', type=['jpg', 'jpeg', 'png'])

    # If a file is uploaded
    if file is not None:
        # Read the image file as bytes
        image_bytes = file.read()
        # Convert the bytes to an OpenCV image
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        # Preprocess the image
        preprocessed_image = preprocess(image)
        # Use the model to make a prediction
        prediction = model.predict(preprocessed_image)
        # Decode the prediction and get the predicted class
        decoded_prediction = tf.keras.applications.densenet.decode_predictions(prediction, top=1)[0][0]
        predicted_class, class_name, probability = decoded_prediction
        # Display the prediction result and probability
        st.markdown(f'## Result: **{class_name}** (Probability: {probability:.2%})')

if __name__ == '__main__':
    app()
