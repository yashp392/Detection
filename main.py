# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from imutils.video import VideoStream
# import os
# from datetime import datetime
# import streamlit as st

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("C:\\Yash\\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\\yolov3.weights", "C:\\Yash\\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\\yolov3.cfg")

# # Load class labels
# yolo_classes = []
# with open('C:\\Yash\\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\\coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Function to perform YOLO object detection on a single image
# def perform_yolo_detection(img, threshold):
#     # Resize the image to the desired resolution
#     resized_img = cv2.resize(img, (1080, 720))  # Adjust the resolution (640, 480) as needed

#     height, width, _ = resized_img.shape  # Get the resized image dimensions

#     # Preprocess image for YOLO
#     blob = cv2.dnn.blobFromImage(resized_img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)

#     # Get YOLO output
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     # Parse YOLO output
#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply non-maximum suppression
#     indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

#     # Initialize the person count
#     person_count = 0

#     # Draw bounding boxes and labels on the resized image
#     font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
#     thickness = 3  # Thickness of the rectangle border
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         if class_ids[i] == person_idx:  # Check if the class ID is "person"
#             label = str(yolo_classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = tuple(int(c) for c in colors[i])

#             # Draw outer rectangle using OpenCV (unchanged)
#             cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, thickness)

#             # Calculate text size for dynamic positioning (unchanged)
#             (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

#             # Draw inner rectangle using PIL for rounded corners
#             pil_img = Image.fromarray(resized_img)  # Convert OpenCV image to PIL image
#             draw = ImageDraw.Draw(pil_img)
#             draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
#             resized_img = np.array(pil_img)  # Convert back to OpenCV image

#             # Draw text using OpenCV (unchanged)
#             cv2.putText(resized_img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

#             # Increment the person count
#             person_count += 1

#     # Draw the person count label
#     label_text = f"Person Count: {person_count}"
#     (label_width, label_height), baseline = cv2.getTextSize(label_text, font, 0.7, 2)
#     cv2.rectangle(resized_img, (10, 10), (10 + label_width, 10 + label_height + 10), (0, 0, 0), -1)
#     cv2.putText(resized_img, label_text, (15, 25), font, 0.7, (255, 255, 255), 2)

#     # Save the image if person count is greater than threshold
#     if person_count >= threshold:
#         filename, timestamp = save_image(resized_img)
#         return resized_img, person_count, filename, timestamp

#     return resized_img, person_count, None, None

# # Function to save the image
# def save_image(img):
#     # Create the 'saved_images' directory if it doesn't exist
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')

#     # Generate a unique filename with timestamp for the saved image
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/image_{timestamp}.jpg"

#     # Save the image
#     cv2.imwrite(filename, img)

#     return filename, timestamp

# # Streamlit app
# st.title("Real-Time Object Detection with YOLO")
# st.sidebar.title("Settings")
# threshold = st.sidebar.slider("Person count threshold for saving images", 1, 100, 3)

# start_button = st.sidebar.button("Start Detection", key="start_detection")
# stop_button = st.sidebar.button("Stop Detection", key="stop_detection")

# if start_button:
#     video_stream = VideoStream("rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live").start()
#     saved_images = []

#     person_count_placeholder = st.empty()
#     image_placeholder = st.empty()
#     saved_images_placeholder = st.sidebar.empty()

#     detection_running = True

#     while detection_running:
#         frame = video_stream.read()
#         if frame is None:
#             break

#         # Perform YOLO object detection on the resized frame
#         yolo_detected_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)

#         # Update the person count and the frame in the main area
#         person_count_placeholder.write(f"Person Count: {person_count}")
#         image_placeholder.image(yolo_detected_frame, caption=f"Person Count: {person_count}", use_column_width=True)

#         # Check if the person count exceeds the threshold
#         if filename and timestamp:
#             saved_images.append((filename, timestamp))

#         # Update the saved images in the sidebar
#         saved_images_placeholder.subheader("Saved Images")
#         for filename, timestamp in saved_images:
#             saved_images_placeholder.image(filename, caption=f"Saved at: {timestamp}", use_column_width=True)

#         if stop_button:
#             detection_running = False

#     video_stream.stop()

































































# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from imutils.video import VideoStream
# import os
# from datetime import datetime
# import streamlit as st
# import time

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("C:\\Yash\\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\\yolov3.weights", "C:\\Yash\\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\\yolov3.cfg")

# # Load class labels
# yolo_classes = []
# with open('C:\\Yash\\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\\coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Function to perform YOLO object detection on a single image
# def perform_yolo_detection(img, threshold):
#     # Resize the image to the desired resolution
#     resized_img = cv2.resize(img, (1080, 720))  # Adjust the resolution (640, 480) as needed

#     height, width, _ = resized_img.shape  # Get the resized image dimensions

#     # Preprocess image for YOLO
#     blob = cv2.dnn.blobFromImage(resized_img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)

#     # Get YOLO output
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     # Parse YOLO output
#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply non-maximum suppression
#     indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

#     # Initialize the person count
#     person_count = 0

#     # Draw bounding boxes and labels on the resized image
#     font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
#     thickness = 3  # Thickness of the rectangle border
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         if class_ids[i] == person_idx:  # Check if the class ID is "person"
#             label = str(yolo_classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = tuple(int(c) for c in colors[i])

#             # Draw outer rectangle using OpenCV (unchanged)
#             cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, thickness)

#             # Calculate text size for dynamic positioning (unchanged)
#             (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

#             # Draw inner rectangle using PIL for rounded corners
#             pil_img = Image.fromarray(resized_img)  # Convert OpenCV image to PIL image
#             draw = ImageDraw.Draw(pil_img)
#             draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
#             resized_img = np.array(pil_img)  # Convert back to OpenCV image

#             # Draw text using OpenCV (unchanged)
#             cv2.putText(resized_img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

#             # Increment the person count
#             person_count += 1

#     # Draw the person count label
#     label_text = f"Person Count: {person_count}"
#     (label_width, label_height), baseline = cv2.getTextSize(label_text, font, 0.7, 2)
#     cv2.rectangle(resized_img, (10, 10), (10 + label_width, 10 + label_height + 10), (0, 0, 0), -1)
#     cv2.putText(resized_img, label_text, (15, 25), font, 0.7, (255, 255, 255), 2)

#     # Save the image if person count is greater than threshold
#     if person_count >= threshold:
#         filename, timestamp = save_image(resized_img)
#         return resized_img, person_count, filename, timestamp

#     return resized_img, person_count, None, None

# # Function to save the image
# def save_image(img):
#     # Create the 'saved_images' directory if it doesn't exist
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')

#     # Generate a unique filename with timestamp for the saved image
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/image_{timestamp}.jpg"

#     # Save the image
#     cv2.imwrite(filename, img)

#     return filename, timestamp

# # Streamlit app
# st.title("Real-Time Object Detection")
# st.sidebar.title("Settings")
# threshold = st.sidebar.slider("Person count threshold for saving images", 1, 100, 6)

# start_button = st.sidebar.button("Start Detection", key="start_detection")
# stop_button = st.sidebar.button("Stop Detection", key="stop_detection")

# if start_button:
#     video_stream = VideoStream("rtsp://user:Admin$123@125.22.133.74:600/media/video1").start()
#     saved_images = []

#     person_count_placeholder = st.empty()
#     image_placeholder = st.empty()
#     saved_images_placeholder = st.sidebar.empty()

#     detection_running = True
#     last_saved_time = time.time()

#     while detection_running:
#         frame = video_stream.read()
#         if frame is None:
#             break

#         # Perform YOLO object detection on the resized frame
#         yolo_detected_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)

#         # Update the person count and the frame in the main area
#         person_count_placeholder.write(f"Person Count: {person_count}")
#         image_placeholder.image(yolo_detected_frame, caption=f"Person Count: {person_count}", use_column_width=True)

#         # Check if the person count exceeds the threshold and save every 5 seconds
#         current_time = time.time()
#         if person_count >= threshold and (current_time - last_saved_time) >= 5:
#             filename, timestamp = save_image(yolo_detected_frame)
#             saved_images.append((filename, timestamp))
#             last_saved_time = current_time

#         # Update the saved images in the sidebar
#         saved_images_placeholder.subheader("Saved Images")
#         for filename, timestamp in saved_images:
#             saved_images_placeholder.image(filename, caption=f"Saved at: {timestamp}", use_column_width=True)

#         if stop_button:
#             detection_running = False

#     video_stream.stop()
























































# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from imutils.video import VideoStream
# import os
# from datetime import datetime
# import streamlit as st
# import time

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet(r"C:\Users\Yash\OneDrive\Desktop\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\yolov3.weights", r"C:\Users\Yash\OneDrive\Desktop\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\yolov3.cfg")

# # Load class labels
# yolo_classes = []
# with open(r'C:\Users\Yash\OneDrive\Desktop\Real-time-Object-Detection-with-OpenCV-Using-YOLO-main\coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Function to perform YOLO object detection on a single image
# def perform_yolo_detection(img, threshold):
#     # Resize the image to the desired resolution
#     resized_img = cv2.resize(img, (1080, 720))  # Adjust the resolution (640, 480) as needed

#     height, width, _ = resized_img.shape  # Get the resized image dimensions

#     # Preprocess image for YOLO
#     blob = cv2.dnn.blobFromImage(resized_img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)

#     # Get YOLO output
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     # Parse YOLO output
#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply non-maximum suppression
#     indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

#     # Initialize the person count
#     person_count = 0

#     # Draw bounding boxes and labels on the resized image
#     font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
#     thickness = 3  # Thickness of the rectangle border
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         if class_ids[i] == person_idx:  # Check if the class ID is "person"
#             label = str(yolo_classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = tuple(int(c) for c in colors[i])

#             # Draw outer rectangle using OpenCV (unchanged)
#             cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, thickness)

#             # Calculate text size for dynamic positioning (unchanged)
#             (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

#             # Draw inner rectangle using PIL for rounded corners
#             pil_img = Image.fromarray(resized_img)  # Convert OpenCV image to PIL image
#             draw = ImageDraw.Draw(pil_img)
#             draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
#             resized_img = np.array(pil_img)  # Convert back to OpenCV image

#             # Draw text using OpenCV (unchanged)
#             cv2.putText(resized_img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

#             # Increment the person count
#             person_count += 1

#     # Draw the person count label
#     label_text = f"Person Count: {person_count}"
#     (label_width, label_height), baseline = cv2.getTextSize(label_text, font, 0.7, 2)
#     cv2.rectangle(resized_img, (10, 10), (10 + label_width, 10 + label_height + 10), (0, 0, 0), -1)
#     cv2.putText(resized_img, label_text, (15, 25), font, 0.7, (255, 255, 255), 2)

#     # Save the image if person count is greater than threshold
#     if person_count >= threshold:
#         filename, timestamp = save_image(resized_img)
#         return resized_img, person_count, filename, timestamp

#     return resized_img, person_count, None, None

# # Function to save the image
# def save_image(img):
#     # Create the 'saved_images' directory if it doesn't exist
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')

#     # Generate a unique filename with timestamp for the saved image
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/image_{timestamp}.jpg"

#     # Save the image
#     cv2.imwrite(filename, img)

#     return filename, timestamp

# # Streamlit app
# st.title("Real-Time Object Detection")
# st.sidebar.title("Settings")

# # Replace slider with input box for person count threshold
# threshold = st.sidebar.number_input("Person count threshold for saving images", min_value=1, max_value=100, value=6, step=1)

# start_button = st.sidebar.button("Start Detection", key="start_detection")
# stop_button = st.sidebar.button("Stop Detection", key="stop_detection")

# if start_button:
#     video_stream = VideoStream("rtsp://user:Admin$123@125.22.133.74:600/media/video1").start()
#     saved_images = []
#     detections = []

#     person_count_placeholder = st.empty()
#     image_placeholder = st.empty()
#     saved_images_placeholder = st.sidebar.empty()
#     detections_placeholder = st.sidebar.empty()

#     detection_running = True
#     last_saved_time = time.time()

#     while detection_running:
#         frame = video_stream.read()
#         if frame is None:
#             break

#         # Perform YOLO object detection on the resized frame
#         yolo_detected_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)

#         # Update the person count and the frame in the main area
#         person_count_placeholder.write(f"Person Count: {person_count}")
#         image_placeholder.image(yolo_detected_frame, caption=f"Person Count: {person_count}", use_column_width=True)

#         # Check if the person count exceeds the threshold and save every 5 seconds
#         current_time = time.time()
#         if person_count >= threshold and (current_time - last_saved_time) >= 5:
#             filename, timestamp = save_image(yolo_detected_frame)
#             saved_images.append((filename, timestamp, person_count))
#             last_saved_time = current_time
#             detections.append((person_count, timestamp))

#         if stop_button:
#             detection_running = False

#     video_stream.stop()

# # Display saved images in a single column
# st.subheader("Saved Images")
# for filename, timestamp, count in saved_images:
#     st.text(f"File: {filename} Saved at: {timestamp} Person Count: {count}")































































# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from imutils.video import VideoStream
# import os
# from datetime import datetime
# import streamlit as st
# import time

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load class labels
# yolo_classes = []
# with open("coco.names", 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Function to perform YOLO object detection on a single image
# def perform_yolo_detection(img, threshold):
#     # Resize the image to the desired resolution
#     resized_img = cv2.resize(img, (1080, 720))  # Adjust the resolution (640, 480) as needed

#     height, width, _ = resized_img.shape  # Get the resized image dimensions

#     # Preprocess image for YOLO
#     blob = cv2.dnn.blobFromImage(resized_img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)

#     # Get YOLO output
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     # Parse YOLO output
#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply non-maximum suppression
#     indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

#     # Initialize the person count
#     person_count = 0

#     # Draw bounding boxes and labels on the resized image
#     font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
#     thickness = 3  # Thickness of the rectangle border
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         if class_ids[i] == person_idx:  # Check if the class ID is "person"
#             label = str(yolo_classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = tuple(int(c) for c in colors[i])

#             # Draw outer rectangle using OpenCV (unchanged)
#             cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, thickness)

#             # Calculate text size for dynamic positioning (unchanged)
#             (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

#             # Draw inner rectangle using PIL for rounded corners
#             pil_img = Image.fromarray(resized_img)  # Convert OpenCV image to PIL image
#             draw = ImageDraw.Draw(pil_img)
#             draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
#             resized_img = np.array(pil_img)  # Convert back to OpenCV image

#             # Draw text using OpenCV (unchanged)
#             cv2.putText(resized_img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

#             # Increment the person count
#             person_count += 1

#     # Draw the person count label
#     label_text = f"Person Count: {person_count}"
#     (label_width, label_height), baseline = cv2.getTextSize(label_text, font, 0.7, 2)
#     cv2.rectangle(resized_img, (10, 10), (10 + label_width, 10 + label_height + 10), (0, 0, 0), -1)
#     cv2.putText(resized_img, label_text, (15, 25), font, 0.7, (255, 255, 255), 2)

#     # Save the image if person count is greater than threshold
#     if person_count >= threshold:
#         filename, timestamp = save_image(resized_img)
#         return resized_img, person_count, filename, timestamp

#     return resized_img, person_count, None, None

# # Function to save the image
# def save_image(img):
#     # Create the 'saved_images' directory if it doesn't exist
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')

#     # Generate a unique filename with timestamp for the saved image
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/image_{timestamp}.jpg"

#     # Save the image
#     cv2.imwrite(filename, img)

#     # Return the filename and time part of the timestamp
#     time_only = datetime.now().strftime("%H:%M:%S")
#     return filename, time_only

# st.image("logo.png", width=250)

# # Replace slider with input box for person count threshold and buttons below header
# threshold = st.number_input("Default Person count", min_value=1, max_value=100, value=1, step=1)
# col3, col4 = st.columns(2)
# with col3:
#     start_button = st.button("Start Detection", key="start_detection")
# with col4:
#     stop_button = st.button("Stop Detection", key="stop_detection")

# # Layout columns for the live stream and saved images
# col1, col2 = st.columns(2)

# # Define saved_images variable before the start button condition
# saved_images = []

# if start_button:
#     video_stream = VideoStream("rtsp://user:Admin$123@125.22.133.74:600/media/video1").start()
#     detections = []

#     person_count_placeholder = col1.empty()
#     image_placeholder = col1.empty()
#     saved_images_placeholder = col2.empty()
#     detections_placeholder = col2.empty()

#     detection_running = True
#     last_saved_time = time.time()

#     while detection_running:
#         frame = video_stream.read()
#         if frame is None:
#             break

#         # Perform YOLO object detection on the resized frame
#         yolo_detected_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)

#         # Update the person count and the frame in the main area
#         person_count_placeholder.write(f"Person Count: {person_count}")
#         image_placeholder.image(yolo_detected_frame, use_column_width=True)

#         # Check if the person count exceeds the threshold and save every 5 seconds
#         current_time = time.time()
#         if person_count >= threshold and (current_time - last_saved_time) >= 5:
#             filename, timestamp = save_image(yolo_detected_frame)
#             saved_images.append((filename, timestamp))
#             last_saved_time = current_time
#             detections.append((person_count, timestamp))

#             # Keep only the last 5 images
#             if len(saved_images) > 5:
#                 saved_images = saved_images[-5:]

#             # Update the saved images display
#             saved_images_placeholder.empty()
#             with saved_images_placeholder:
#                 if len(saved_images) > 0:
#                     cols = st.columns(len(saved_images))  # Create columns for each saved image
#                     for col, (filename, timestamp) in zip(cols, saved_images):
#                         col.image(filename, use_column_width=True)
#                         col.caption(f"Time: {timestamp}")

#         if stop_button:
#             detection_running = False

#     video_stream.stop()

# with col2:
#     st.subheader("List Detection")
#     if len(saved_images) > 0:
#         cols = st.columns(len(saved_images))  # Create columns for each saved image
#         for col, (filename, timestamp) in zip(cols, saved_images):
#             col.image(filename, use_column_width=True)
#             col.caption(f"Time: {timestamp}")









































# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from imutils.video import VideoStream
# import os
# from datetime import datetime
# import streamlit as st
# import time
# import pandas as pd

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load class labels
# yolo_classes = []
# with open('coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Function to perform YOLO object detection on a single image
# def perform_yolo_detection(img, threshold):
#     # Resize the image to the desired resolution
#     resized_img = cv2.resize(img, (1080, 720))  # Adjust the resolution (640, 480) as needed

#     height, width, _ = resized_img.shape  # Get the resized image dimensions

#     # Preprocess image for YOLO
#     blob = cv2.dnn.blobFromImage(resized_img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)

#     # Get YOLO output
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     # Parse YOLO output
#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply non-maximum suppression
#     indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

#     # Initialize the person count
#     person_count = 0

#     # Draw bounding boxes and labels on the resized image
#     font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
#     thickness = 3  # Thickness of the rectangle border
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         if class_ids[i] == person_idx:  # Check if the class ID is "person"
#             label = str(yolo_classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = tuple(int(c) for c in colors[i])

#             # Draw outer rectangle using OpenCV (unchanged)
#             cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, thickness)

#             # Calculate text size for dynamic positioning (unchanged)
#             (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

#             # Draw inner rectangle using PIL for rounded corners
#             pil_img = Image.fromarray(resized_img)  # Convert OpenCV image to PIL image
#             draw = ImageDraw.Draw(pil_img)
#             draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
#             resized_img = np.array(pil_img)  # Convert back to OpenCV image

#             # Draw text using OpenCV (unchanged)
#             cv2.putText(resized_img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

#             # Increment the person count
#             person_count += 1

#     # Draw the person count label
#     label_text = f"Person Count: {person_count}"
#     (label_width, label_height), baseline = cv2.getTextSize(label_text, font, 0.7, 2)
#     cv2.rectangle(resized_img, (10, 10), (10 + label_width, 10 + label_height + 10), (0, 0, 0), -1)
#     cv2.putText(resized_img, label_text, (15, 25), font, 0.7, (255, 255, 255), 2)

#     # Save the image if person count is greater than threshold
#     if person_count >= threshold:
#         filename, timestamp = save_image(resized_img)
#         return resized_img, person_count, filename, timestamp

#     return resized_img, person_count, None, None

# # Function to save the image
# def save_image(img):
#     # Create the 'saved_images' directory if it doesn't exist
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')

#     # Generate a unique filename with timestamp for the saved image
#     timestamp = datetime.now().strftime("%H:%M:%S")
#     filename = f"saved_images/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

#     # Save the image
#     cv2.imwrite(filename, img)

#     return filename, timestamp

# # Streamlit app
# st.image("logo.png", width=250)

# # Replace slider with input box for person count threshold and buttons below header
# threshold = st.number_input("Default Person count", min_value=1, max_value=100, value=1, step=1)
# col3, col4 = st.columns(2)
# with col3:
#     start_button = st.button("Start Detection", key="start_detection")
# with col4:
#     stop_button = st.button("Stop Detection", key="stop_detection")

# # Layout columns for the live stream and saved images
# col1, col2 = st.columns(2)

# # Define saved_images variable before the start button condition
# saved_images = []

# # Placeholder for DataFrame
# df_placeholder = col2.empty()

# if start_button:
#     video_stream = VideoStream("rtsp://user:Admin$123@125.22.133.74:600/media/video1").start()
#     detections = []

#     person_count_placeholder = col1.empty()
#     image_placeholder = col1.empty()

#     detection_running = True
#     last_saved_time = time.time()

#     while detection_running:
#         frame = video_stream.read()
#         if frame is None:
#             break

#         # Perform YOLO object detection on the resized frame
#         yolo_detected_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)

#         # Update the person count and the frame in the main area
#         person_count_placeholder.write(f"Person Count: {person_count}")
#         image_placeholder.image(yolo_detected_frame, caption=f"Person Count: {person_count}", use_column_width=True)

#         # Check if the person count exceeds the threshold and save every 5 seconds
#         current_time = time.time()
#         if person_count >= threshold and (current_time - last_saved_time) >= 5:
#             filename, timestamp = save_image(yolo_detected_frame)
#             saved_images.append((filename, timestamp))
#             last_saved_time = current_time
#             detections.append((timestamp))

#             # Keep only the last 5 images
#             if len(saved_images) > 5:
#                 saved_images = saved_images[-5:]

#             # Update the saved images display
#             df = pd.DataFrame(saved_images, columns=["Filename", "Timestamp"])
#             df['Image'] = df['Filename'].apply(lambda x: f'<img src="{x}" width="100">')
#             df = df[["Image", "Timestamp"]]  # Only keep Image and Timestamp columns
#             df_placeholder.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

#         if stop_button:
#             detection_running = False

#     video_stream.stop()

# # Display saved images in the second column as a table
# with col2:
#     if len(saved_images) > 0:
#         df = pd.DataFrame(saved_images, columns=["Filename", "Timestamp"])
#         df['Image'] = df['Filename'].apply(lambda x: f'<img src="{x}" width="100">')
#         df = df[["Image", "Timestamp"]]  # Only keep Image and Timestamp columns
#         st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)






































# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from imutils.video import VideoStream
# import os
# from datetime import datetime
# import streamlit as st
# import time
# import pandas as pd

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load class labels
# yolo_classes = []
# with open('coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# # Function to perform YOLO object detection on a single image
# def perform_yolo_detection(img, threshold):
#     # Resize the image to the desired resolution
#     resized_img = cv2.resize(img, (1080, 720))  # Adjust the resolution (640, 480) as needed

#     height, width, _ = resized_img.shape  # Get the resized image dimensions

#     # Preprocess image for YOLO
#     blob = cv2.dnn.blobFromImage(resized_img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     yolo_net.setInput(blob)

#     # Get YOLO output
#     output_layers_names = yolo_net.getUnconnectedOutLayersNames()
#     layer_outputs = yolo_net.forward(output_layers_names)

#     # Parse YOLO output
#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply non-maximum suppression
#     indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

#     # Initialize the person count
#     person_count = 0

#     # Draw bounding boxes and labels on the resized image
#     font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
#     thickness = 3  # Thickness of the rectangle border
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         if class_ids[i] == person_idx:  # Check if the class ID is "person"
#             label = str(yolo_classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = tuple(int(c) for c in colors[i])

#             # Draw outer rectangle using OpenCV (unchanged)
#             cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, thickness)

#             # Calculate text size for dynamic positioning (unchanged)
#             (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

#             # Draw inner rectangle using PIL for rounded corners
#             pil_img = Image.fromarray(resized_img)  # Convert OpenCV image to PIL image
#             draw = ImageDraw.Draw(pil_img)
#             draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
#             resized_img = np.array(pil_img)  # Convert back to OpenCV image

#             # Draw text using OpenCV (unchanged)
#             cv2.putText(resized_img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

#             # Increment the person count
#             person_count += 1

#     # Draw the person count label
#     label_text = f"Person Count: {person_count}"
#     (label_width, label_height), baseline = cv2.getTextSize(label_text, font, 0.7, 2)
#     cv2.rectangle(resized_img, (10, 10), (10 + label_width, 10 + label_height + 10), (0, 0, 0), -1)
#     cv2.putText(resized_img, label_text, (15, 25), font, 0.7, (255, 255, 255), 2)

#     # Save the image if person count is greater than threshold
#     if person_count >= threshold:
#         filename, timestamp = save_image(resized_img)
#         return resized_img, person_count, filename, timestamp

#     return resized_img, person_count, None, None

# # Function to save the image
# def save_image(img):
#     # Create the 'saved_images' directory if it doesn't exist
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')

#     # Generate a unique filename with timestamp for the saved image
#     timestamp = datetime.now().strftime("%H:%M:%S")
#     filename = f"saved_images/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

#     # Save the image
#     cv2.imwrite(filename, img)

#     return filename, timestamp

# # Streamlit app
# st.image("logo.png", width=250)

# # Replace slider with input box for person count threshold and buttons below header
# threshold = st.number_input("Default Person count", min_value=1, max_value=100, value=1, step=1)
# col3, col4 = st.columns(2)
# with col3:
#     start_button = st.button("Start Detection", key="start_detection")
# with col4:
#     stop_button = st.button("Stop Detection", key="stop_detection")

# # Layout columns for the live stream and saved images
# col1, col2 = st.columns(2)

# # Define saved_images variable before the start button condition
# saved_images = []

# # Placeholder for DataFrame
# df_placeholder = col2.empty()

# if start_button:
#     video_stream = VideoStream("rtsp://user:Admin$123@125.22.133.74:600/media/video1").start()
#     detections = []

#     person_count_placeholder = col1.empty()
#     image_placeholder = col1.empty()

#     detection_running = True
#     last_saved_time = time.time()

#     while detection_running:
#         frame = video_stream.read()
#         if frame is None:
#             break

#         # Perform YOLO object detection on the resized frame
#         yolo_detected_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)

#         # Update the person count and the frame in the main area
#         person_count_placeholder.write(f"Person Count: {person_count}")
#         image_placeholder.image(yolo_detected_frame, caption=f"Person Count: {person_count}", use_column_width=True)

#         # Check if the person count exceeds the threshold and save every 5 seconds
#         current_time = time.time()
#         if person_count >= threshold and (current_time - last_saved_time) >= 5:
#             filename, timestamp = save_image(yolo_detected_frame)
#             saved_images.append((filename, timestamp))
#             last_saved_time = current_time
#             detections.append((timestamp))

#             # Keep only the last 5 images
#             if len(saved_images) > 5:
#                 saved_images = saved_images[-5:]
#             # Update the saved images display
#             df = pd.DataFrame(saved_images, columns=["Filename", "Timestamp"])
#             df['Image'] = df['Filename'].apply(lambda x: f'<img src="{x}" width="100">')
#             df = df[["Image", "Timestamp"]]  # Only keep Image and Timestamp columns
#             df_placeholder.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

#         if stop_button:
#             detection_running = False

#     video_stream.stop()

# # Display saved images in the second column as a table
# if len(saved_images) > 0:
#     df = pd.DataFrame(saved_images, columns=["Filename", "Timestamp"])
#     df['Image'] = df['Filename'].apply(lambda x: f'<img src="{x}" width="100" />')
#     df_html = df.to_html(escape=False, index=False)
#     st.markdown(df_html, unsafe_allow_html=True)

































































# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from imutils.video import VideoStream
# import os
# from datetime import datetime
# import streamlit as st
# import time
# import pandas as pd

# # Load YOLOv3 model
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load class labels
# with open('coco.names', 'r') as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# # Get the index of the "person" class label
# person_idx = yolo_classes.index("person")

# def draw_boxes(img, boxes, confidences, class_ids):
#     for i in range(len(boxes)):
#         if class_ids[i] == person_idx:  # Check if detected class is person
#             x, y, w, h = boxes[i]
#             label = f"Person {round(confidences[i], 2)}"
#             color = (255, 0, 0)  # Red color for person class
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     return img

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
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     img_with_boxes = draw_boxes(img.copy(), boxes, confidences, class_ids)
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     person_count = sum(1 for i in indexes if class_ids[i] == person_idx)

#     if person_count >= threshold:
#         filename, timestamp = save_image(img_with_boxes)
#         return img_with_boxes, person_count, filename, timestamp

#     return img_with_boxes, person_count, None, None

# def save_image(img):
#     if not os.path.exists('saved_images'):
#         os.makedirs('saved_images')
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"saved_images/{timestamp}.jpg"
#     cv2.imwrite(filename, img)
#     return filename, timestamp

# st.image("logo.png", width=250)
# col1, col2, col3 = st.columns([2, 1, 1])
# threshold = col1.number_input("Set person count threshold", min_value=1, max_value=100, value=10, step=1)

# start_button = col2.button("Start Detection")
# stop_button = col3.button("Stop Detection")

# if start_button:
#     st.session_state.running = True

# if stop_button:
#     st.session_state.running = False

# if 'detection_history' not in st.session_state:
#     st.session_state.detection_history = []

# placeholder = st.empty()

# if 'running' in st.session_state and st.session_state.running:
#     video_stream = VideoStream(src="rtsp://user:Admin$123@125.22.133.74:600/media/video1").start()
#     try:
#         while st.session_state.running:
#             frame = video_stream.read()
#             frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)
#             if filename:
#                 placeholder.image(frame, caption=f"Person Count: {person_count}", use_column_width=True)
#                 st.session_state.detection_history.append((filename, datetime.now().strftime('%H:%M:%S'), person_count))
#             time.sleep(1)
#     except Exception as e:
#         st.error(f"Error in video stream: {e}")
#     finally:
#         video_stream.stop()

# if st.session_state.detection_history:
#     st.sidebar.header("Detection History")
#     df = pd.DataFrame(st.session_state.detection_history, columns=['Image', 'Timestamp', 'Person Count'])
#     df['Image'] = df['Image'].apply(lambda x: f'<img src="{x}" width="100" style="border-radius: 10px;">')
#     st.sidebar.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)







































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

# # os.environ["SDL_AUDIODRIVER"] = "alsa"  # or "pulse" or "dummy" depending on your system

# # try:
# #     pygame.mixer.init()
# #     pygame.mixer.music.load("alert.mp3")  # Assuming the file is in the same directory
# #     print("Audio initialized successfully.")
# # except pygame.error as e:
# #     print(f"Failed to initialize audio: {e}")  # Assuming the file is in the same directory
# os.environ['SDL_AUDIODRIVER'] = 'alsa'

# pygame.mixer.init()
# pygame.mixer.music.load("alert.mp3")

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
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

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

if stop_button:
    st.session_state.running = False

placeholder = st.empty()

if 'running' in st.session_state and st.session_state.running:
    video_stream = VideoStream(src=rtsp_url).start()
    last_detection_time = time.time()
    try:
        while st.session_state.running:
            frame = video_stream.read()
            current_time = time.time()
            if current_time - last_detection_time >= 5:
                processed_frame, person_count, filename, timestamp = perform_yolo_detection(frame, threshold)
                last_detection_time = current_time
                if filename:
                    # Insert new detection at the beginning of the list
                    st.session_state.detection_history.insert(0, (filename, datetime.now().strftime('%H:%M:%S'), person_count))
                    # Keep only the latest 5 detections
                    st.session_state.detection_history = st.session_state.detection_history[:5]
                    
                    st.sidebar.write(f"Detected {person_count} persons at {st.session_state.detection_history[0][1]}")
                    st.sidebar.image(filename, width=550)  # Reduced width for sidebar

                    # Check if alert should be played with cooldown
                    if alert_checkbox and person_count >= threshold:
                        current_time = time.time()
                        if current_time - last_alert_time > alert_cooldown:
                            st.audio("alert.mp3",autoplay=True)
                            last_alert_time = current_time
                
                placeholder.image(processed_frame, caption="Live Stream", use_column_width=True)
            else:
                placeholder.image(frame, caption="Live Stream", use_column_width=True)
            time.sleep(0.1)
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
