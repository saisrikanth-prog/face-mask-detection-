from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
# from tensorflow.keras.models import load_model

app = FastAPI()

# Load your trained model here
# model = load_model("mask_detector.model")

def generate_frames():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # -----------------------------------------------------------
            # PLACE YOUR MODEL INFERENCE & BOUNDING BOX LOGIC HERE
            # -----------------------------------------------------------
            # Example logic structure:
            # 1. Detect faces using OpenCV Haar Cascade
            # 2. Preprocess cropped face to 224x224
            # 3. Predict using model (Index 0: Mask, Index 1: No Mask)
            # 4. Draw Green Box for Mask, Red Box for No Mask
            # -----------------------------------------------------------

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Stream the frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.get("/")
def index():
    # Serves the clean HTML page
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/video_feed")
def video_feed():
    # Video streaming route
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)