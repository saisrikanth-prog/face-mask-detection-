import sys
import os
import numpy as np


if 'cv2' in sys.modules:
    del sys.modules['cv2']


import cv2 

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')


if not os.path.exists(cascade_path):
    print(f"[ERROR] '{cascade_path}' not found!")
    print("Please make sure you downloaded 'haarcascade_frontalface_default.xml' into this project folder.")
    exit()


try:
    face_cascade_class = getattr(cv2, 'CascadeClassifier')
    face_cascade = face_cascade_class(cascade_path)
except AttributeError:
    print("[ERROR] Still failing to bind CascadeClassifier. Please double check that you ran:")
    print("python -m pip install --user --force-reinstall opencv-python")
    exit()

if face_cascade.empty():
    print("[ERROR] Could not load face detector XML data.")
    exit()


model_path = os.path.join(current_dir, "mask_detector.h5")
if not os.path.exists(model_path):
    print(f"[ERROR] {model_path} not found! Run train_model.py first.")
    exit()

print("[INFO] Loading trained mask detector model...")
model = load_model(model_path)


print("[INFO] Starting webcam... Press 'q' to exit.")
cap = cv2.VideoCapture(0)

frame_count = 0
faces = []  

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    frame_count += 1
    
    # Optimization 1: Process face detection only every 2nd frame to prevent CPU lag
    if frame_count % 2 == 0:
        # Optimization 2: Downscale frame internally for rapid cascade scanning
        scale_factor = 2 
        small_frame = cv2.resize(frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces on scaled-down frame
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Upscale bounding box coordinates back to full image scale
        faces = [(x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor) 
                 for (x, y, w, h) in detected_faces]

    # Process predictions & draw visuals (runs on every frame utilizing current/cached data)
    for (x, y, w, h) in faces:
        # Prevent edge-of-screen index slice crashes
        y_start, y_end = max(0, y), min(frame.shape[0], y+h)
        x_start, x_end = max(0, x), min(frame.shape[1], x+w)
        
        face_img = frame[y_start:y_end, x_start:x_end]
        if face_img.size == 0:
            continue
            
        # MobileNetV2 preprocessing pipeline
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        
       
        preds = model.predict(face_img, verbose=0)[0]


        mask = preds[0]          # Grabs 'with_mask' from index 0
        withoutMask = preds[1]   # Grabs 'without_mask' from index 1

         
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Optimization 3: Boundary check text coordinates to keep it visible on-screen
        text_y = max(15, y - 10)
        cv2.putText(frame, f"{label}: {max(mask, withoutMask)*100:.2f}%", (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()