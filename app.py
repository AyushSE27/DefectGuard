import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AI Vision DefectGuard", layout="wide")
st.title("🛡️ AI Vision DefectGuard - Deep Learning Quality Control")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.sidebar.header("⚙️ Mode Selection")
mode = st.sidebar.radio("Choose Input Mode", ["Image Upload (Cloud Friendly)", "Live Camera (Localhost)"])
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.4)

if mode == "Image Upload (Cloud Friendly)":
    uploaded_file = st.file_uploader("Upload Product Image to Inspect...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        results = model(img_array, conf=conf_threshold)
        annotated_img = results[0].plot()
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Product Image", use_container_width=True)
        with col2:
            st.image(annotated_img, caption="AI Inspection Result", use_container_width=True)
            
        detections = results[0].boxes
        if len(detections) > 0:
            st.error(f"🚨 ALERT: {len(detections)} Anomaly/Defect Detected!")
        else:
            st.success("✅ STATUS: Product Passed Quality Test")

elif mode == "Live Camera (Localhost)":
    st.info("Note: Live camera mode works when running locally on your laptop.")
    run_cam = st.checkbox("Start Live Stream")
    frame_window = st.image([])
    
    cap = cv2.VideoCapture(0)
    while run_cam:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible.")
            break
        frame = cv2.resize(frame, (640, 480))
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)
    cap.release()