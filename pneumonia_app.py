import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('D:\Engineering\Projects\Pneumonia Prediction\Pneumonia_model.h5')

# Function to preprocess the image
import cv2
import numpy as np

def preprocess(image):
    """
    Preprocesses an image to prepare it for input into a deep learning model.
    
    Args:
        image: A numpy array representing the image to be preprocessed
        
    Returns:
        A numpy array containing the preprocessed image
    """
    # Convert the image to grayscale if it has more than one channel
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize the image to 224x224 (the input size for the model)
    resized = cv2.resize(gray, (224, 224))
    
    # Convert the image to a 3-channel image
    x = np.expand_dims(resized, axis=2)
    x = np.repeat(x, 3, axis=2)
    
    # Normalize the pixel values to be between 0 and 1 and add a batch dimension
    x = np.expand_dims(x / 255.0, axis=0)
    
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
        # Get the predicted class (0 = normal, 1 = pneumonia)
        predicted_class = np.argmax(prediction[0])
        # Display the prediction result
        if predicted_class == 0:
            st.markdown('## Result: **You Have no Symptoms of Pneumonia**')
        else:
            st.markdown('## Result: **You might be Pneumonic,Please consult a doctor**')

if __name__ == '__main__':
    app()
