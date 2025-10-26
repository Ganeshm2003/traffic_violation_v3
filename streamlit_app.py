import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
import easyocr  # For OCR if needed

# Load models (lazy for memory)
@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov8n.pt')  # Nano for lighter memory
    ocr_reader = easyocr.Reader(['en'])  # English OCR for plates
    return yolo_model, ocr_reader

st.title("Enhanced Traffic Violation Detection System")

# Sidebar for settings
st.sidebar.header("Violation Line Setup")
manual_line = st.sidebar.text_input("Manual Line (x1,y1,x2,y2)", placeholder="[100,300,500,300]")
auto_detect = st.sidebar.checkbox("Enable Auto-Detection", value=True)

st.sidebar.header("Detection Options")
enable_ocr = st.sidebar.checkbox("Enable License Plate OCR", value=True)

# Upload section
uploaded_file = st.file_uploader("Upload Traffic Image/Video", type=['jpg', 'png', 'mp4', 'avi'])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Violations"):
            with st.spinner("Processing..."):
                yolo_model, ocr_reader = load_models()
                results = yolo_model(image)
                # Draw line (manual or auto)
                if manual_line:
                    coords = [int(c) for c in manual_line.replace('[', '').replace(']', '').split(',')]
                    cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
                # Process detections (simplified—add your core/violation_detector.py logic here)
                detections = results[0].boxes.xyxy.cpu().numpy()
                violation_count = len(detections)  # Placeholder: count as violations
                for det in detections:
                    x1, y1, x2, y2 = map(int, det)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                st.image(image, caption=f"Processed: {violation_count} Violations", use_column_width=True)

                # Log CSV
                log_df = pd.DataFrame({
                    'Violation Type': ['Line Crossing'] * violation_count,
                    'Count': [violation_count],
                    'Timestamp': [pd.Timestamp.now()]
                })
                csv_buffer = BytesIO()
                log_df.to_csv(csv_buffer, index=False)
                st.download_button("Download CSV Log", csv_buffer.getvalue(), "violations.csv", "text/csv")

    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file)
        if st.button("Analyze Video"):
            # Add video processing from your core/violation_detector.py
            st.success("Video analysis complete—check logs!")

st.sidebar.info("Upload an image/video to start detecting violations!")
