# import cv2
# import numpy as np
# from datetime import datetime
# import os
# import streamlit as st
# import time
# from imutils.video import VideoStream
# import pytz
# import pygame  # New: Replaced playsound with pygame

# # New: Initialize pygame mixer and load alert sound
# pygame.mixer.init()
# pygame.mixer.music.load(r"C:\Users\Dell\Desktop\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\alert.mp3")

# # New: Variables for alert cooldown
# last_alert_time = 0
# alert_cooldown = 10  # 10 seconds cooldown between alerts

# # Load YOLOv3 model (Unchanged)
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load class labels (Unchanged)
# with open('coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label (Unchanged)
# person_idx = yolo_classes.index("person")

# def perform_yolo_detection(img, threshold):
#     # This function is completely unchanged
#     height, width, _ = img.shape
#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.8:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
    
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     person_count = sum(1 for i in indexes.flatten() if class_ids[i] == person_idx) if len(indexes) > 0 else 0

#     for i in indexes.flatten() if len(indexes) > 0 else []:
#         if class_ids[i] == person_idx:
#             x, y, w, h = boxes[i]
#             color = (0, 255, 0)  # Green color for the bounding box
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             text = f"Person: {confidences[i]:.2f}"
#             cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     if person_count >= threshold:
#         filename, timestamp = save_image(img)
#         return img, person_count, filename, timestamp

#     return img, person_count, None, None

# def save_image(img):
#     # This function is completely unchanged
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')
#     utc_now = datetime.utcnow()
#     ist_timezone = pytz.timezone("Asia/Kolkata")
#     ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_timezone)
#     timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/{timestamp}.jpg"
#     cv2.imwrite(filename, img)
#     return filename, timestamp

# # Streamlit UI setup (Unchanged)
# st.image("logo.png", width=250)
# col1, col2, col3 = st.columns([1, 1, 1])
# threshold = col1.number_input("Set person count threshold", min_value=1, max_value=100, value=1, step=1)
# alert_checkbox = col1.checkbox("Enable Alert")

# start_button = col2.button("Start Detection")
# stop_button = col3.button("Stop Detection")

# if start_button:
#     st.session_state.running = True

# if stop_button:
#     st.session_state.running = False

# placeholder = st.empty()
# detection_history = []

# if 'running' in st.session_state and st.session_state.running:
#     video_stream = VideoStream(src="rtsp://user:Admin$123@125.22.133.74:600/media/video1").start()
#     last_detection_time = time.time()
#     try:
#         while st.session_state.running:
#             frame = video_stream.read()
#             current_time = time.time()
#             if current_time - last_detection_time >= 5:
#                 processed_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)
#                 last_detection_time = current_time
#                 if filename:
#                     detection_history.append((filename, datetime.now().strftime('%H:%M:%S'), person_count))
#                     st.sidebar.write(f"Detected {person_count} persons at {detection_history[-1][1]}")
#                     st.sidebar.image(filename, width=500)  # Increase the width to show larger images
                    
#                     # New: Check if alert should be played with cooldown
#                     if alert_checkbox and person_count >= threshold:
#                         current_time = time.time()
#                         if current_time - last_alert_time > alert_cooldown:
#                             pygame.mixer.music.play()
#                             last_alert_time = current_time
                
#                 placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)
#             else:
#                 placeholder.image(frame, caption="Live Stream", use_column_width=True)
#             time.sleep(0.1)
#     except Exception as e:
#         st.error(f"Error in video stream: {e}")
#     finally:
#         video_stream.stop()

# # Sidebar for Detection History (Unchanged)
# st.sidebar.header("Detection History")
# for history in detection_history:
#     filename, timestamp, person_count = history
#     col1, col2 = st.sidebar.columns([3, 1])  # Adjust column sizes for better layout
#     col1.image(filename, width=500)  # Increase width as needed to show larger images
#     col2.write(f"{timestamp}\n{person_count} persons")


























































# import cv2
# import numpy as np
# from datetime import datetime
# import os
# import streamlit as st
# import time
# from imutils.video import VideoStream
# import pytz
# import pygame

# # Initialize pygame mixer and load alert sound
# pygame.mixer.init()
# pygame.mixer.music.load("alert.mp3")  # Assuming the file is in the same directory

# # Variables for alert cooldown
# last_alert_time = 0
# alert_cooldown = 10  # 10 seconds cooldown between alerts

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load class labels
# with open('coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# def perform_yolo_detection(img, threshold, margin):
#     height, width, _ = img.shape
#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > margin:  # Using the margin parameter here
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
    
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     person_count = sum(1 for i in indexes.flatten() if class_ids[i] == person_idx) if len(indexes) > 0 else 0

#     for i in indexes.flatten() if len(indexes) > 0 else []:
#         if class_ids[i] == person_idx:
#             x, y, w, h = boxes[i]
#             color = (0, 255, 0)  # Green color for the bounding box
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             text = f"Person: {confidences[i]:.2f}"
#             cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     if person_count >= threshold:
#         filename, timestamp = save_image(img)
#         return img, person_count, filename, timestamp

#     return img, person_count, None, None

# def save_image(img):
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')
#     utc_now = datetime.utcnow()
#     ist_timezone = pytz.timezone("Asia/Kolkata")
#     ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_timezone)
#     timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/{timestamp}.jpg"
#     cv2.imwrite(filename, img)
#     return filename, timestamp

# # Streamlit UI setup
# st.set_page_config(layout="wide")  # Use wide layout for better horizontal space
# st.image("logo.png", width=250)

# # All controls in one horizontal line
# col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
# rtsp_url = col1.text_input("RTSP URL", value="rtsp://user:Admin$123@125.22.133.74:600/media/video1")
# threshold = col2.number_input("Threshold", min_value=1, max_value=100, value=1, step=1)
# margin = col3.slider("Confidence", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
# start_button = col4.button("Start Detection")
# stop_button = col5.button("Stop Detection")

# alert_checkbox = st.sidebar.checkbox("Enable Alert", value=True)

# if start_button:
#     st.session_state.running = True
#     st.session_state.rtsp_url = rtsp_url

# if stop_button:
#     st.session_state.running = False

# placeholder = st.empty()
# detection_history = []

# if 'running' in st.session_state and st.session_state.running:
#     video_stream = VideoStream(src=st.session_state.rtsp_url).start()
#     last_detection_time = time.time()
#     try:
#         while st.session_state.running:
#             frame = video_stream.read()
#             current_time = time.time()
#             if current_time - last_detection_time >= 5:
#                 processed_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold, margin)
#                 last_detection_time = current_time
#                 if filename:
#                     detection_history.append((filename, datetime.now().strftime('%H:%M:%S'), person_count))
#                     st.sidebar.write(f"Detected {person_count} persons at {detection_history[-1][1]}")
#                     st.sidebar.image(filename, width=300)  # Reduced width for sidebar
                    
#                     # Check if alert should be played with cooldown
#                     if alert_checkbox and person_count >= threshold:
#                         current_time = time.time()
#                         if current_time - last_alert_time > alert_cooldown:
#                             pygame.mixer.music.play()
#                             last_alert_time = current_time
                
#                 placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)
#             else:
#                 placeholder.image(frame, caption="Live Stream", use_column_width=True)
#             time.sleep(0.1)
#     except Exception as e:
#         st.error(f"Error in video stream: {e}")
#     finally:
#         video_stream.stop()

# # Detection History in main area, not sidebar
# if detection_history:
#     st.header("Detection History")
#     for i in range(0, len(detection_history), 3):  # Display 3 images per row
#         cols = st.columns(3)
#         for j in range(3):
#             if i + j < len(detection_history):
#                 filename, timestamp, person_count = detection_history[i + j]
#                 with cols[j]:
#                     st.image(filename, width=300)
#                     st.write(f"{timestamp}\n{person_count} persons")









































# import cv2
# import numpy as np
# from datetime import datetime
# import os
# import streamlit as st
# import time
# from imutils.video import VideoStream
# import pytz
# import pygame

# # Initialize pygame mixer and load alert sound
# pygame.mixer.init()
# pygame.mixer.music.load("alert.mp3")  # Assuming the file is in the same directory

# # Variables for alert cooldown
# last_alert_time = 0
# alert_cooldown = 10  # 10 seconds cooldown between alerts

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load class labels
# with open('coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Fixed confidence value
# CONFIDENCE_THRESHOLD = 0.5

# def perform_yolo_detection(img, threshold):
#     height, width, _ = img.shape
#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > CONFIDENCE_THRESHOLD:  # Using the fixed confidence threshold
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
    
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
#     person_count = sum(1 for i in indexes.flatten() if class_ids[i] == person_idx) if len(indexes) > 0 else 0

#     for i in indexes.flatten() if len(indexes) > 0 else []:
#         if class_ids[i] == person_idx:
#             x, y, w, h = boxes[i]
#             color = (0, 255, 0)  # Green color for the bounding box
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             text = f"Person: {confidences[i]:.2f}"
#             cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     if person_count >= threshold:
#         filename, timestamp = save_image(img)
#         return img, person_count, filename, timestamp

#     return img, person_count, None, None

# def save_image(img):
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')
#     utc_now = datetime.utcnow()
#     ist_timezone = pytz.timezone("Asia/Kolkata")
#     ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_timezone)
#     timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/{timestamp}.jpg"
#     cv2.imwrite(filename, img)
#     return filename, timestamp

# # Streamlit UI setup
# st.set_page_config(layout="wide")  # Use wide layout for better horizontal space
# st.image("logo.png", width=250)

# # Hidden RTSP URL input
# rtsp_url = "rtsp://user:Admin$123@125.22.133.74:600/media/video1"

# # Simple UI with just threshold and buttons
# col1, col2, col3 = st.columns([1, 1, 1])
# threshold = col1.number_input("Set person count threshold", min_value=1, max_value=100, value=1, step=1)
# start_button = col2.button("Start Detection")
# stop_button = col3.button("Stop Detection")

# alert_checkbox = st.sidebar.checkbox("Enable Alert", value=True)

# if start_button:
#     st.session_state.running = True
#     st.session_state.detection_history = []  # Reset history on start

# if stop_button:
#     st.session_state.running = False

# placeholder = st.empty()

# if 'running' in st.session_state and st.session_state.running:
#     video_stream = VideoStream(src=rtsp_url).start()
#     last_detection_time = time.time()
#     try:
#         while st.session_state.running:
#             frame = video_stream.read()
#             current_time = time.time()
#             if current_time - last_detection_time >= 5:
#                 processed_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)
#                 last_detection_time = current_time
#                 if filename:
#                     # New: Insert new detection at the beginning of the list
#                     st.session_state.detection_history.insert(0, (filename, datetime.now().strftime('%H:%M:%S'), person_count))
#                     st.sidebar.write(f"Detected {person_count} persons at {st.session_state.detection_history[0][1]}")
#                     st.sidebar.image(filename, width=300)  # Reduced width for sidebar
                    
#                     # Check if alert should be played with cooldown
#                     if alert_checkbox and person_count >= threshold:
#                         current_time = time.time()
#                         if current_time - last_alert_time > alert_cooldown:
#                             pygame.mixer.music.play()
#                             last_alert_time = current_time
                
#                 placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)
#             else:
#                 placeholder.image(frame, caption="Live Stream", use_column_width=True)
#             time.sleep(0.1)
#     except Exception as e:
#         st.error(f"Error in video stream: {e}")
#     finally:
#         video_stream.stop()

# # New: Display Detection History in reverse order (newest first)
# if 'detection_history' in st.session_state and st.session_state.detection_history:
#     st.header("Detection History")
#     for filename, timestamp, person_count in st.session_state.detection_history:
#         with st.expander(f"{timestamp} - {person_count} persons", expanded=False):
#             col1, col2 = st.columns([4, 1])
#             with col1:
#                 st.image(filename, use_column_width=True)
#             with col2:
#                 st.write(f"Time: {timestamp}")
#                 st.write(f"Count: {person_count}")
#                 if st.button("View Full Size", key=filename):
#                     st.image(filename)






































# import cv2
# import numpy as np
# from datetime import datetime
# import os
# import streamlit as st
# import time
# from imutils.video import VideoStream
# import pytz
# import pygame

# os.environ["SDL_AUDIODRIVER"] = "alsa"  # or "pulse" or "dummy" depending on your system

# try:
#     pygame.mixer.init()
#     pygame.mixer.music.load("alert.mp3")  # Assuming the file is in the same directory
#     print("Audio initialized successfully.")
# except pygame.error as e:
#     print(f"Failed to initialize audio: {e}")  # Assuming the file is in the same directory

# # pygame.mixer.init()
# # pygame.mixer.music.load("alert.mp3")

# # Variables for alert cooldown
# last_alert_time = 0
# alert_cooldown = 10  # 10 seconds cooldown between alerts

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# # Load class labels
# with open('coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Fixed confidence value
# CONFIDENCE_THRESHOLD = 0.5

# def perform_yolo_detection(img, threshold):
#     height, width, _ = img.shape
#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > CONFIDENCE_THRESHOLD:  # Using the fixed confidence threshold
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
    
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
#     person_count = sum(1 for i in indexes.flatten() if class_ids[i] == person_idx) if len(indexes) > 0 else 0

#     for i in indexes.flatten() if len(indexes) > 0 else []:
#         if class_ids[i] == person_idx:
#             x, y, w, h = boxes[i]
#             color = (0, 255, 0)  # Green color for the bounding box
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             text = f"Person: {confidences[i]:.2f}"
#             cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     if person_count >= threshold:
#         filename, timestamp = save_image(img)
#         return img, person_count, filename, timestamp

#     return img, person_count, None, None

# def save_image(img):
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')
#     utc_now = datetime.utcnow()
#     ist_timezone = pytz.timezone("Asia/Kolkata")
#     ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_timezone)
#     timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/{timestamp}.jpg"
#     cv2.imwrite(filename, img)
#     return filename, timestamp

# # Streamlit UI setup
# st.set_page_config(layout="wide")  # Use wide layout for better horizontal space
# st.image("logo.png", width=250)

# # Hidden RTSP URL input
# rtsp_url = "rtsp://user:Admin$123@125.22.133.74:600/media/video1"

# # Simple UI with just threshold and buttons
# col1, col2, col3 = st.columns([1, 1, 1])
# threshold = col1.number_input("Set person count threshold", min_value=1, max_value=100, value=1, step=1)
# start_button = col2.button("Start Detection")
# stop_button = col3.button("Stop Detection")

# alert_checkbox = st.sidebar.checkbox("Enable Alert", value=True)

# if start_button:
#     st.session_state.running = True
#     st.session_state.detection_history = []  # Reset history on start

# if stop_button:
#     st.session_state.running = False

# placeholder = st.empty()

# if 'running' in st.session_state and st.session_state.running:
#     video_stream = VideoStream(src=rtsp_url).start()
#     last_detection_time = time.time()
#     try:
#         while st.session_state.running:
#             frame = video_stream.read()
#             current_time = time.time()
#             if current_time - last_detection_time >= 5:
#                 processed_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)
#                 last_detection_time = current_time
#                 if filename:
#                     # Insert new detection at the beginning of the list
#                     st.session_state.detection_history.insert(0, (filename, datetime.now().strftime('%H:%M:%S'), person_count))
#                     # Keep only the latest 5 detections
#                     st.session_state.detection_history = st.session_state.detection_history
                    
#                     st.sidebar.write(f"Detected {person_count} persons at {st.session_state.detection_history[0][1]}")
#                     st.sidebar.image(filename, width=550)  # Reduced width for sidebar

                                            
#                     # Check if alert should be played with cooldown
#                     if alert_checkbox and person_count >= threshold:
#                         current_time = time.time()
#                         if current_time - last_alert_time > alert_cooldown:
#                             pygame.mixer.music.play()
#                             last_alert_time = current_time
                
#                 placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)
#             else:
#                 placeholder.image(frame, caption="Live Stream", use_column_width=True)
#             time.sleep(0.1)
#     except Exception as e:
#         st.error(f"Error in video stream: {e}")
#     finally:
#         video_stream.stop()

# # Display Detection History in descending order with a limit of 5 entries
# if 'detection_history' in st.session_state and st.session_state.detection_history:
#     st.sidebar.header("Detection History")
#     for filename, timestamp, person_count in st.session_state.detection_history:
#             with col1:
#                 st.sidebar.image(filename, use_column_width=True)
#             with col2:
#                 st.sidebar.write(f"Time: {timestamp}")
#                 st.sidebar.write(f"Count: {person_count}")









































import cv2
import numpy as np
from datetime import datetime
import os
import streamlit as st
import time
from imutils.video import VideoStream
import pytz

# Variables for alert cooldown
last_alert_time = 0
alert_cooldown = 10  # 10 seconds cooldown between alerts

# Load YOLOv3 model
yolo_net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Load class labels
with open('coco.names', 'r') as f:
    yolo_classes = [line.strip() for line in f.readlines()]

# Get the index of the "person" class label
person_idx = yolo_classes.index("person")

# Fixed confidence value
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
            if confidence > CONFIDENCE_THRESHOLD:  # Using the fixed confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
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


# Streamlit UI setup
st.set_page_config(layout="wide")  # Use wide layout for better horizontal space
st.image("logo.png", width=250)

# Hidden RTSP URL input
rtsp_url = "rtsp://user:Admin$123@125.22.133.74:600/media/video1"

# Simple UI with just threshold and buttons
col1, col2, col3 = st.columns([1, 1, 1])
threshold = col1.number_input("Set person count threshold", min_value=1, max_value=100, value=1, step=1)
start_button = col2.button("Start Detection")
stop_button = col3.button("Stop Detection")

alert_checkbox = st.sidebar.checkbox("Enable Alert", value=True)

if start_button:
    st.session_state.running = True
    st.session_state.detection_history = []  # Reset history on start
    st.session_state.consecutive_detections = 0  # Counter for consecutive detections

if stop_button:
    st.session_state.running = False

placeholder = st.empty()

if 'running' in st.session_state and st.session_state.running:
    video_stream = VideoStream(src=rtsp_url).start()
    try:
        while st.session_state.running:
            frame = video_stream.read()
            processed_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)
            
            if person_count >= threshold:
                st.session_state.consecutive_detections += 1  # Increment counter for consecutive detections
            else:
                st.session_state.consecutive_detections = 0  # Reset counter if no detection

            # Display images only if detected 6 consecutive times
            if st.session_state.consecutive_detections >= 6:
                if filename:
                    st.sidebar.write(f"Detected {person_count} persons at {timestamp}")
                    st.sidebar.image(filename, width=550)  # Reduced width for sidebar

                    # Check if alert should be played with cooldown
                    if alert_checkbox:
                        current_time = time.time()
                        if current_time - last_alert_time > alert_cooldown:
                            st.audio("alert.mp3", autoplay=True)
                            last_alert_time = current_time

                placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)
                st.session_state.consecutive_detections = 0  # Reset after displaying
            else:
                placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)

            time.sleep(0.1)  # Maintain this only if necessary to reduce CPU load
    except Exception as e:
        st.error(f"Error in video stream: {e}")
    finally:
        video_stream.stop()

# Display Detection History in descending order with a limit of 5 entries
if 'detection_history' in st.session_state and st.session_state.detection_history:
    st.sidebar.header("Detection History")
    for filename, timestamp, person_count in st.session_state.detection_history:
        st.sidebar.image(filename, use_column_width=True)
        st.sidebar.write(f"Time: {timestamp}")
        st.sidebar.write(f"Count: {person_count}")
