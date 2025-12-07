import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="wide")

st.sidebar.title("🌱 Plant Disease Detector")
st.sidebar.markdown("AI-powered identification of plant leaf diseases.\n\n*Upload your leaf photo & get instant results!*")
# Load model
model = load_model('/workspaces/Plant-Disease-CNN/notebooks/src/plant_disease_cnn.h5')

# Define image size (as used in your model)
IMG_SIZE = (128, 128)

# List of classes (should match your training order)
CLASSES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'] 

st.title("🌿 Plant Disease Classifier")
st.write("Upload a plant leaf image:")

file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if file:
    img = Image.open(file).convert('RGB')
    st.image(img, use_column_width=True)
    img_resized = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_class = CLASSES[pred_index]
    confidence = 100 * np.max(prediction)
    st.markdown(f"### 🌱 **Prediction:** `{pred_class}`")
    st.progress(float(confidence) / 100)
    st.write(f"**Confidence:** {confidence:.2f}%")