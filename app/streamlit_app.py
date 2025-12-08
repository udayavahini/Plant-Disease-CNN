import streamlit as st
import numpy as np
from PIL import Image

MODEL_PATH = "/workspaces/Plant-Disease-CNN/notebooks/src/plant_disease_cnn.h5"

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
# Loading model
model = load_model('/workspaces/Plant-Disease-CNN/notebooks/src/plant_disease_cnn.h5')

# Defining image size 
IMG_SIZE = (128, 128)

# List of classes 
CLASSES = ['Bell Pepper - Bacterial spot', 'Bell Pepper - Healthy', 'Potato - Early blight', 'Potato - Late blight', 'Potato - Healthy', 'Tomato - Bacterial spot', 'Tomato - Early blight', 'Tomato - Late blight', 'Tomato - Leaf Mold', 'Tomato - Septoria leaf spot', 'Tomato - Spider Mites', 'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato -  Mosaic Virus', 'Tomato - Healthy']

st.set_page_config(page_title="Plant Doctor AI", layout="wide", page_icon="🌱")

# Design
st.markdown("""
<style>
body {
    background-image: url('https://plus.unsplash.com/premium_photo-1722899516572-409bf979e5d6?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8Y3JvcHN8ZW58MHx8MHx8fDA%3D');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.stApp {
    background: rgba(19, 28, 15, 0.01);
    min-height: 100vh;
}

.block-container {
    background: rgba(31, 44, 27, 0.5);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 45px;
    box-shadow: 0 6px 48px 0 rgba(0,0,0,0.40);
}

.sidebar .sidebar-content {
    background: rgba(26,39,26,0.60);
    color: #d3e5c2;
    border-radius: 16px;
}

h1, h2, h3 {
    color: #bdd5c6;
    font-weight: 600;
    text-shadow: 0 2px 10px #22382044;
    letter-spacing: 0.01em;
}

a, .stMarkdown a {
    color: #b5e59e;
    text-decoration: none;
}

.st-bb {
    background-color: #283f26 !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌿 Plant Doctor AI")
st.sidebar.markdown("""
Upload a **plant leaf photo** (preferably from a crop field, close up & well-lit).  
The AI will detect health/disease.
""")
st.sidebar.markdown("---")
st.sidebar.info("""
- Built for real-field agriculture  
- Powered by CNN 
- All CNN runs locally  
""")

st.markdown(
    "<h1 style='text-align:center; font-size:2.25em; margin-bottom:4px;'>Plant Disease Detector</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center; color:#9fb99e; margin-bottom:30px; font-weight: 400;'>AI-powered diagnosis for farm crops</h4>",
    unsafe_allow_html=True
)

file = st.file_uploader("Upload a crop leaf image...", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert('RGB')
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(img, caption="Uploaded Leaf", width="content")
    with col2:
        model = load_model(MODEL_PATH)
        img_resized = img.resize(IMG_SIZE)
        arr = img_to_array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        pred_class = CLASSES[idx]
        confidence = float(np.max(preds)) * 100

        st.markdown(
            f"""<div style="background:rgba(30,54,25,0.87);border-radius:12px;padding:20px 25px;box-shadow:0 5px 14px #16281266;">
                <h3 style='color:#b5e59e;margin-bottom:4px;'>Prediction:</h3>
                <div style='font-size:1.2em;color:#e4fadd;padding-bottom:8px;margin-bottom:2px;'><b>{pred_class}</b></div>
                <b style='color:#a5d692;'>Confidence:</b> <span style='font-size:1.1em;'>{confidence:.2f}%</span>
            </div>""", unsafe_allow_html=True)
        st.progress(confidence/100)
        st.info("For better predictions, use leaf images with clear details and minimal background. Created by Udaya Vahini P (MDA - 2482463)")
else:
    st.markdown(
        "<div style='margin:36px 0 0 0;text-align:center;color:#ced;'>&#8593; Upload a crop leaf image to analyze</div>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<small style='color:#869879;'>This tool aids preliminary diagnosis only as this is a project.</small>", 
    unsafe_allow_html=True
)