import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from deepface import DeepFace

# 1. Clean Page Configuration
st.set_page_config(page_title="Emotion Analyzer", layout="wide", initial_sidebar_state="expanded")

# 2. Premium UI/CSS Overrides
st.markdown("""
<style>
    /* Force all processed images to be exactly 400px high and fit perfectly */
    [data-testid="stImage"] img {
        height: 400px !important;
        width: 100% !important;
        object-fit: contain !important;
        background-color: #0a0a0a; 
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    /* Make the sidebar look sleek */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Hide the default Streamlit sidebar block padding to make our custom card flush */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration (THE UPGRADED UI)
with st.sidebar:
    
    # --- Custom Flexbox Profile Card ---
    st.markdown("""
    <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; display: flex; align-items: center; gap: 15px; border: 1px solid #334155; margin-bottom: 25px;">
        <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Nova&backgroundColor=06b6d4" width="45" style="border-radius: 50%; border: 2px solid #06b6d4;">
        <div style="line-height: 1.2;">
            <div style="color: #f8fafc; font-weight: 600; font-size: 15px;">Admin User</div>
            <div style="color: #000000; background-color: #06b6d4; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px;">PRO PLAN</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Navigation Section ---

    st.markdown("<p style='color: #94a3b8; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;'>Main Menu</p>", unsafe_allow_html=True)
    
    # Swapped radio buttons for a cleaner selectbox
    menu_selection = st.selectbox("", ["Emotion Analyzer", "📝 History", "⚙️ Settings"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # --- App Info ---
    st.markdown("<p style='color: #94a3b8; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;'>System Info</p>", unsafe_allow_html=True)
    st.info("**Engine:** DeepFace + OpenCV\n\n**Status:** Online & Ready\n\n**Version:** 2.1.0")


# 4. Cached Analysis Function (EMOTION ONLY & OPTIMIZED)
@st.cache_data
def analyze_image(img_array):
    """Analyzes faces purely for emotion using a fast backend."""
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return DeepFace.analyze(
        img_path=img_bgr, 
        actions=['emotion'], 
        enforce_detection=True,
        detector_backend='opencv'
    )

# --- MAIN APP AREA ---

if menu_selection == "Emotion Analyzer":
    
    st.title("Emotion Analyzer")
    st.markdown("Upload an image to detect faces and extract their primary emotional state.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Source Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((800, 800))
        img_array = np.array(image)
        
        with col2:
            st.subheader("Analysis & Reporting")
            
            with st.spinner("Analyzing emotions..."):
                try:
                    results = analyze_image(img_array)
                    
                    if not isinstance(results, list):
                        results = [results]
                    
                    num_faces = len(results)
                    img_draw = img_array.copy()
                    report_data = []
                    
                    for i, face_data in enumerate(results):
                        dominant_emotion = face_data['dominant_emotion']
                        
                        region = face_data['region']
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 225, 255), 4)
                        
                        if num_faces == 1:
                            label = dominant_emotion.capitalize()
                            subject_name = "Subject"
                            bg_width = 120 
                        else:
                            label = f"Person {i+1}: {dominant_emotion.capitalize()}"
                            subject_name = f"Person {i+1}"
                            bg_width = 200 
                            
                        text_bg_y = y - 30 if y - 30 > 0 else y + 30
                        cv2.rectangle(img_draw, (x, text_bg_y-5), (x + bg_width, text_bg_y + 25), (0, 225, 255), -1)
                        cv2.putText(img_draw, label, (x + 5, text_bg_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        report_data.append({
                            "Subject": subject_name,
                            "Emotion": dominant_emotion.capitalize()
                        })

                    with col1:
                        st.image(img_draw, use_container_width=True)
                        st.caption(f"Detected {num_faces} face(s).")
                    
                    # --- DYNAMIC SUCCESS MESSAGE ADDED HERE ---
                    if num_faces == 1:
                        st.success(f" Analysis complete: Subject is primarily **{results[0]['dominant_emotion'].capitalize()}**.")
                    else:
                        st.success(f" Analysis complete: Successfully processed **{num_faces}** faces.")
                    # ------------------------------------------

                    df_report = pd.DataFrame(report_data)
                    st.dataframe(df_report, use_container_width=True, hide_index=True)
                    
                    st.markdown("### Export Results")
                    
                    result_img_pil = Image.fromarray(img_draw)
                    img_buffer = io.BytesIO()
                    result_img_pil.save(img_buffer, format="JPEG")
                    img_bytes = img_buffer.getvalue()
                    
                    csv_bytes = df_report.to_csv(index=False).encode('utf-8')
                    
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(label=" Download Image", data=img_bytes, file_name="analyzed_image.jpg", mime="image/jpeg")
                    with dl_col2:
                        st.download_button(label=" Download CSV Data", data=csv_bytes, file_name="emotion_report.csv", mime="text/csv")
                    
                except ValueError:
                    with col1:
                        st.image(image, use_container_width=True)
                    st.error("Face not detected. Please upload an image with clearly visible faces.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

else:
    st.title(menu_selection)
    st.info("This module is currently under construction. Please use the 'Emotion Analyzer' from the sidebar menu.")