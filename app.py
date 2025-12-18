import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="YOLO Inference")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("YOLO Model Inference")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Inference"):
        results = model(image)
        annotated = results[0].plot()
        st.image(annotated, caption="Prediction", use_column_width=True)
