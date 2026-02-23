# 🎭 Syntecxhub Emotion Recognition App

This project is an AI-powered Facial Emotion Recognition application built with Python and Streamlit. It utilizes the DeepFace library to analyze biometric facial features and predict the primary emotional state of a subject in an uploaded image.

## 🚀 Features
- **Real-Time Analysis:** Instantly processes uploaded images to detect facial landmarks.
- **Emotion Classification:** Categorizes emotions into states such as Happy, Sad, Angry, Surprise, Fear, Disgust, or Neutral.
- **Modern SaaS UI:** Features a sleek, responsive dark-mode interface with custom CSS styling.
- **Error Handling:** Automatically rejects images without clearly visible human faces.

## 🛠️ Technologies Used
- **Python:** Core programming language.
- **Streamlit:** Frontend framework for building the web interface.
- **DeepFace:** Pre-trained deep learning facial recognition framework.
- **OpenCV & NumPy:** Image processing and array manipulation.

## ⚙️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone [https://github.com/Ammaratahir252/Syntecxhub_Emotion_Recognition_App.git](https://github.com/Ammaratahir252/Syntecxhub_Emotion_Recognition_App.git)
2.Navigate to the project directory:
cd Syntecxhub_Emotion_Recognition_App
3.Install the required dependencies:
pip install streamlit deepface opencv-python pillow tf-keras
4.Run the application:
streamlit run app.py

Usage
Launch the app using the command above.
Click the "Browse files" button or drag and drop a clear portrait image into the dropzone.
The AI will scan the image and output the dominant emotion detected.
