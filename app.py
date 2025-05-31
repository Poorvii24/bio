%%writefile app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("retinopathy_model.h5")

labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def preprocess_image(img):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🧿 Diabetic Retinopathy Detection")
st.write("Upload a retinal image to classify the severity of diabetic retinopathy.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(img)
    st.write(f"Model Input Shape: {processed_image.shape}")
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    st.success(f"Prediction: **{labels[class_idx]}**")
    st.info(f"Confidence: {confidence:.2f}")
