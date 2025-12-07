import streamlit as st
import numpy as np
from PIL import Image

MODEL_PATH = "/workspaces/Plant-Disease-CNN/notebooks/src/plant_disease_cnn.h5"
# If using a Keras/TensorFlow model:
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
# Load model
model = load_model('/workspaces/Plant-Disease-CNN/notebooks/src/plant_disease_cnn.h5')

# Define image size (as used in your model)
IMG_SIZE = (128, 128)

# List of classes (should match your training order)
CLASSES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'] 

st.set_page_config(page_title="Plant Doctor AI", layout="wide", page_icon="🌱")

# Dark green, nature-themed blurry background
st.markdown("""
<style>
body {
    background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80');
    /* Try this even darker-green image for more forest feel: */
    /* background-image: url('https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=1500&q=80');*/
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
.stApp {
    background: rgba(26,39,26,0.70);
    border-radius: 16px;
    min-height: 100vh;
}
.block-container {
    background: rgba(25,39,25,0.76);
    border-radius: 20px;
    padding: 2rem 2rem 2rem 2rem;
    margin-top: 45px;
    box-shadow: 0 8px 32px 0 rgba(10,30,10,0.40);
}
.sidebar .sidebar-content {
    background: rgba(40,60,40,0.34);
    border-radius: 18px;
    backdrop-filter: blur(4px);
    color: #e0ffe7;
}
h1, h2, h3 {
    color: #BBF7D0;
    text-shadow: 1px 2px 16px #224a1f;
}
a { color: #74fa9c;}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌿 Plant Doctor AI")
st.sidebar.markdown("""
Upload a plant leaf photo (close up and well-lit for best results).  
The AI will classify its health or disease.
""")
st.sidebar.markdown("---")
st.sidebar.info("""
- Powered by CNN & Keras  
- Built for sustainable agriculture 🌱  
""")

# Main header
st.markdown(
    "<h1 style='text-align:center; font-size: 3em; margin-bottom: 4px;'>🌱 Plant Disease Detector</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center; color: #BBF7D0; margin-bottom:36px;'>AI-powered crop health diagnosis</h4>", 
    unsafe_allow_html=True
)

# File Uploader + Main
file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert('RGB')
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(img, caption="Your uploaded leaf", use_column_width=True)
    with col2:
        # Predict (use your model)
        model = load_model(MODEL_PATH)
        img_resized = img.resize(IMG_SIZE)
        arr = img_to_array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        pred_class = CLASSES[idx]
        confidence = float(np.max(preds)) * 100

        st.markdown(
            f"""<div style="background:rgba(48,88,54,0.84);border-radius:13px;padding:22px;box-shadow:0 3px 9px #13291a;">
            <h2 style='color:#9fffcb;'>Prediction: <span style='color:#c7fde7'>{pred_class}</span></h2>
            <b>Confidence:</b> <span style='color:#aceca1; font-size:1.2em;'>{confidence:.2f}%</span>
            </div>""", unsafe_allow_html=True)
        st.progress(confidence/100)
        st.info("For optimal AI performance, use clear, high-resolution images with a single leaf.")
else:
    st.markdown(
        "<div style='margin:36px 0 0 0; text-align:center; color:#74fa9c;'><i>Upload a plant leaf image to get started!</i></div>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<small style='color:#78ac93;'>This demo is for educational purposes. Results are not a substitute for expert plant pathology advice.</small>", 
    unsafe_allow_html=True
)