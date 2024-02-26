import streamlit as st
import tensorflow as tf
from keras.applications import DenseNet121
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import numpy as np

# Load the pre-trained DenseNet121 model
model = DenseNet121(weights='imagenet', include_top=True)

# Function to preprocess the image
def preprocess(image):
    """
    Preprocesses an image to prepare it for input into a deep learning model.
    
    Args:
        image: A PIL Image object representing the image to be preprocessed
        
    Returns:
        A numpy array containing the preprocessed image
    """
    # Resize the image to (224, 224) as DenseNet121 requires this size
    resized = image.resize((224, 224))
    
    # Convert the image to a numpy array
    x = np.array(resized)
    
    # Expand dimensions to create a batch dimension
    x = np.expand_dims(x, axis=0)
    
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
        # Open the image using Pillow
        image = Image.open(file)
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
