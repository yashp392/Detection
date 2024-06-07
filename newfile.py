import cv2
import numpy as np
from datetime import datetime
import os
import streamlit as st
import time
from imutils.video import VideoStream
import pytz
from playsound import playsound
import threading

# Ensure the audio file is in the correct path
audio_file_path = "alert.mp3"
if not os.path.exists(audio_file_path):
    st.error("Alert sound file not found.")

# Variables for alert cooldown
last_alert_time = 0
alert_cooldown = 10  # 10 seconds cooldown between alerts

# Load YOLOv3 model
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
with open('coco.names', 'r') as f:
    yolo_classes = [line.strip() for line in f.readlines()]

# Get the index of the "person" class label
person_idx = yolo_classes.index("person")

CONFIDENCE_THRESHOLD = 0.5

def perform_yolo_detection(img, threshold):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    output_layers_names = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    person_count = sum(1 for i in indexes.flatten() if class_ids[i] == person_idx) if len(indexes) > 0 else 0

    for i in indexes.flatten() if len(indexes) > 0 else []:
        if class_ids[i] == person_idx:
            x, y, w, h = boxes[i]
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = f"Person: {confidences[i]:.2f}"
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if person_count >= threshold:
        filename, timestamp = save_image(img)
        return img, person_count, filename, timestamp

    return img, person_count, None, None

def save_image(img):
    if not os.path.exists('saved_images'):
        os.makedirs('saved_images')
    utc_now = datetime.utcnow()
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_timezone)
    timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
    filename = f"saved_images/{timestamp}.jpg"
    cv2.imwrite(filename, img)
    return filename, timestamp

def play_alert():
    try:
        playsound(audio_file_path)
    except Exception as e:
        st.error(f"Failed to play alert: {str(e)}")

# Streamlit UI setup
st.set_page_config(layout="wide")
st.image("logo.png", width=250)

col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
rtsp_url = col1.text_input("RTSP URL", value="rtsp://example.com/live")
threshold = col2.number_input("Threshold", min_value=1, max_value=100, value=1, step=1)
start_button = col3.button("Start Detection")
stop_button = col4.button("Stop Detection")

alert_checkbox = st.sidebar.checkbox("Enable Alert", value=True)

if start_button:
    st.session_state.running = True
    st.session_state.rtsp_url = rtsp_url

if stop_button:
    st.session_state.running = False

placeholder = st.empty()
detection_history = []

if 'running' in st.session_state and st.session_state.running:
    video_stream = VideoStream(src=st.session_state.rtsp_url).start()
    last_detection_time = time.time()
    try:
        while st.session_state.running:
            frame = video_stream.read()
            current_time = time.time()
            if current_time - last_detection_time >= 10:
                processed_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)
                last_detection_time = current_time
                if filename:
                    detection_history.insert(0, (filename, datetime.now().strftime('%H:%M:%S'), person_count))
                    detection_history = detection_history[:5]
                    
                    st.sidebar.write(f"Detected {person_count} persons at {detection_history[0][1]}")
                    st.sidebar.image(filename, width=450)
                    
                    if alert_checkbox and person_count >= threshold:
                        st.write("Playing alert")  # Debug statement
                        threading.Thread(target=play_alert).start()
                        last_alert_time = current_time
                
                placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)
            else:
                placeholder.image(frame, caption="Live Stream", use_column_width=True)
            time.sleep(0.1)
    except Exception as e:
        st.error(f"Error in video stream: {e}")
    finally:
        video_stream.stop()

if detection_history:
    st.header("Detection History")
    for i in range(0, len(detection_history), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(detection_history):
                filename, timestamp, person_count = detection_history[i + j]
                with cols[j]:
                    st.image(filename, width=450)
                    st.write(f"{timestamp}\n{person_count} persons")
