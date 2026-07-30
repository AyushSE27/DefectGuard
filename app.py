import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
import time
from ultralytics import YOLO

# Page Setup
st.set_page_config(page_title="AI Vision DefectGuard (YOLOv8)", layout="wide")
st.title("🛡️ AI Vision DefectGuard")
st.markdown("Automated Anomaly & Defect Inspection")

# Load Pre-trained YOLOv8 AI Model
@st.cache_resource
def load_model():
    # Downloads lightweight YOLOv8 nano model (yolov8n.pt) automatically
    return YOLO('yolov8n.pt')

model = load_model()

# Sidebar - Settings
st.sidebar.header("⚙️ AI Control Panel")
run_cam = st.sidebar.checkbox("Start Live AI Inspection", value=False)
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# Metrics Display
m1, m2, m3 = st.columns(3)
total_placeholder = m1.empty()
pass_placeholder = m2.empty()
fail_placeholder = m3.empty()

# Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live AI Vision Stream")
    frame_window = st.image([])

with col2:
    st.subheader("📊 Inspection Analytics")
    chart_placeholder = st.empty()

# Initialize Session State
if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.passed = 0
    st.session_state.failed = 0

# Video Capture Loop
cap = cv2.VideoCapture(0)

while run_cam:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera feed.")
        break

    frame = cv2.resize(frame, (640, 480))
    
    # Run YOLOv8 AI Inference on Frame
    results = model(frame, conf=conf_threshold, verbose=False)
    
    # Process Detection Results
    annotated_frame = results[0].plot()  # Draws YOLO Bounding Boxes & Labels automatically
    detections = results[0].boxes

    # Defect Logic: If objects/anomalies are detected
    if len(detections) > 0:
        defect_detected = True
        status_text = f"ALERT: {len(detections)} OBJECT/DEFECT DETECTED"
        color = (0, 0, 255) # Red
    else:
        defect_detected = False
        status_text = "STATUS: SURFACE CLEAR (PASSED)"
        color = (0, 255, 0) # Green

    # Status Overlay on Video Stream
    cv2.putText(annotated_frame, status_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Convert BGR to RGB for Display
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)

    # Metrics Update
    st.session_state.total += 1
    if defect_detected:
        st.session_state.failed += 1
    else:
        st.session_state.passed += 1

    total_placeholder.metric("Total Frames Analyzed", st.session_state.total)
    pass_placeholder.metric("Passed (Clear)", st.session_state.passed)
    fail_placeholder.metric("Anomalies / Defects", st.session_state.failed)

    # Analytics Pie Chart
    df = pd.DataFrame({
        "Status": ["Passed", "Defective/Detected"],
        "Count": [st.session_state.passed, st.session_state.failed]
    })
    fig = px.pie(df, values="Count", names="Status", color="Status",
                 color_discrete_map={"Passed": "green", "Defective/Detected": "red"}, hole=0.4)
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(0.05)

cap.release()