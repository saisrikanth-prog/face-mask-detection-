import sys
import os
import numpy as np

# Force Python to clear any bad/corrupted cv2 imports from memory cache
if 'cv2' in sys.modules:
    del sys.modules['cv2']

import cv2 
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. LOAD FACE DETECTOR & MASK MODEL
# ==========================================
cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_path):
    print(f"[ERROR] '{cascade_path}' not found!")
    sys.exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

model_path = os.path.join(current_dir, "mask_detector.h5")
if not os.path.exists(model_path):
    print(f"[ERROR] '{model_path}' not found!")
    sys.exit()

print("[INFO] Loading trained mask detector model...")
model = load_model(model_path)

# ==========================================
# 2. STREAMING FRAME GENERATOR
# ==========================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    faces = []

    try:
        while True:
            ret, frame = cap.read()
            # If camera fails or is disconnected, stop gracefully
            if not ret or frame is None:
                print("[WARNING] Could not read frame from camera.")
                break

            frame_count += 1

            # Detect faces on every 2nd frame
            if frame_count % 2 == 0:
                scale_factor = 2 
                small_frame = cv2.resize(frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                faces = [(x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor) 
                         for (x, y, w, h) in detected_faces]

            # Draw predictions on every frame
            for (x, y, w, h) in faces:
                y_start, y_end = max(0, y), min(frame.shape[0], y + h)
                x_start, x_end = max(0, x), min(frame.shape[1], x + w)
                
                face_img = frame[y_start:y_end, x_start:x_end]
                if face_img.size == 0:
                    continue
                    
                # Preprocessing
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_img, (224, 224))
                face_img = img_to_array(face_img)
                face_img = preprocess_input(face_img)
                face_img = np.expand_dims(face_img, axis=0)

                preds = model.predict(face_img, verbose=0)[0]

                mask = preds[0]          # with_mask
                withoutMask = preds[1]   # without_mask

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                text_y = max(15, y - 10)
                cv2.putText(frame, f"{label}: {max(mask, withoutMask)*100:.2f}%", (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()
        print("[INFO] Camera released.")

# ==========================================
# 3. ROUTES
# ==========================================
@app.get("/")
def index():
    html_file = os.path.join(current_dir, "index.html")
    if not os.path.exists(html_file):
        return HTMLResponse("<h2>Error: 'index.html' not found in project folder!</h2>", status_code=500)
    with open(html_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    # Using port 8001 to avoid address conflicts
    uvicorn.run(app, host="127.0.0.1", port=8001)