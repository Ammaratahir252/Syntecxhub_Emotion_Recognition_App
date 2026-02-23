import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# 1. Clean Page Configuration
st.set_page_config(page_title="Emotion Analyzer", layout="wide")

# 2. Minimalist, Professional SaaS CSS
st.markdown("""
<style>
    /* Clean off-white background */
    .stApp {
        background-color: #f3f4f6;
        color: #1f2937;
    }
    
    /* Headings */
    h1, h2, h3, p, label {
        color: #111827 !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Crisp white upload container with soft shadow */
    [data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: border-color 0.2s ease-in;
    }
    
    /* Change border to blue on hover */
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #2563eb !important;
    }

    /* Professional Blue Button */
    [data-testid="stFileUploadDropzone"] button {
        background-color: #2563eb !important; 
        color: #ffffff !important; 
        font-weight: 500 !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    /* Standardized image size with soft rounded corners */
    [data-testid="stImage"] img {
        height: 400px !important;
        width: 100% !important;
        object-fit: cover !important; 
        border-radius: 12px;
        border: 4px solid #ffffff;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Clean, dark metric typography */
    div[data-testid="stMetricValue"] {
        font-size: 3.5rem;
        color: #2563eb !important; 
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("Emotion Analyzer")
st.markdown("Upload a facial image to analyze the primary emotional state.")
st.markdown("---")

# Clean parallel layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Source Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    with col1:
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Results")
        st.write("") 
        
        with st.spinner("Analyzing image..."):
            try:
                result = DeepFace.analyze(img_path=img_bgr, actions=['emotion'], enforce_detection=True)
                dominant_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
                
                st.success("Analysis complete.")
                st.metric(label="Primary Emotion", value=dominant_emotion.capitalize())
                
            except Exception as e:
                st.error("Face not detected. Please upload an image with a clearly visible face.")