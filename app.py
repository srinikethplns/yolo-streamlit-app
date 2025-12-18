# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 09:06:39 2025
@author: srinikethplns
"""

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Daimler: Parts Detector",
    layout="centered"
)

st.title("Daimler - Parts Detection")

@st.cache_resource
def load_model():
    # Load your trained YOLO model
    return YOLO("best.pt")  

model = load_model()

st.subheader("📷 Capture Image")

# Camera input or file uploader
image_file = st.camera_input("Take a picture")
if image_file is None:
    image_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    img_np = np.array(image)

    # Run YOLO prediction
    results = model.predict(
        source=img_np,
        conf=0.30,
        iou=0.5,
        verbose=False
    )

    # Annotate the image
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    st.image(
        annotated_img,
        caption="Detection Result",
        use_container_width=True
    )

    # Show detected classes
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detected_classes = [model.names[i] for i in class_ids]

        if detected_classes:
            st.success("Detected Objects: " + ", ".join(set(detected_classes)))
        else:
            st.warning("No objects detected.")
    else:
        st.warning("No objects detected.")
