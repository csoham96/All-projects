from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from depth_map import estimate_depth

app = FastAPI()

@app.post("/depth_map/")
async def depth_map(file: UploadFile = File(...)):
    # Read video file
    video = cv2.VideoCapture(file.file)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Estimate depth
        depth = estimate_depth(frame)

        # Convert depth map to bytes to send as response
        _, buffer = cv2.imencode('.jpg', depth * 255)
        yield buffer.tobytes()

    video.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)