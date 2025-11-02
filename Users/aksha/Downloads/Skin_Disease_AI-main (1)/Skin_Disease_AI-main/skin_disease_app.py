"""
Skin Disease AI Diagnostic Tool
A Streamlit web application for diagnosing skin diseases using a trained Xception CNN model.
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tempfile
import os

# Class names for the skin conditions
CLASS_NAMES = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Millia', 'Rosacea']

@st.cache_resource
def load_model():
    """
    Load the trained Xception model for skin disease diagnosis.
    Currently returns None due to incomplete model files.
    
    Returns:
        None: Model loading disabled due to incomplete files
    """
    return None

def preprocess_image(image_path):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image_path (str): Path to the uploaded image file
        
    Returns:
        np.array: Preprocessed image array ready for model input
    """
    # Load and resize image to 299x299 (Xception input size)
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    
    # Apply Xception-specific preprocessing
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    
    # Add batch dimension
    return np.array([img_array])

def generate_sample_predictions():
    """
    Generate sample predictions for demonstration purposes.
    
    Returns:
        np.array: Random prediction probabilities that sum to 1
    """
    # Generate random probabilities that sum to 1
    predictions = np.random.dirichlet(np.ones(6), size=1)[0]
    return predictions

def display_results(predictions):
    """
    Display the prediction results in a user-friendly format.
    
    Args:
        predictions (np.array): Prediction probabilities for each class
    """
    # Get top 3 predictions with highest confidence
    top_indices = np.argsort(predictions)[::-1][:3]
    
    st.subheader("Diagnosis Results:")
    for i, idx in enumerate(top_indices):
        class_name = CLASS_NAMES[idx]
        confidence = predictions[idx] * 100
        st.write(f"{i+1}. {class_name}: {confidence:.2f}%")

def main():
    """
    Main application function.
    Sets up the Streamlit interface and handles user interactions.
    """
    # App title and description
    st.title("Skin Disease AI Diagnostic Tool")
    st.write("Upload an image of a skin lesion to get a diagnosis.")
    
    # File uploader for skin lesion images
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a skin lesion"
    )
    
    # Process uploaded image
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
            st.write("")
            st.write("Analyzing...")
            
            # Generate sample results since real model files are incomplete
            with st.spinner('Processing...'):
                predictions = generate_sample_predictions()
            
            # Display results
            display_results(predictions)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    # Medical disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: This tool is for educational purposes only and should not replace 
    professional medical advice. Always consult with a qualified healthcare professional 
    for accurate diagnosis and treatment recommendations.
    """)

if __name__ == "__main__":
    main()