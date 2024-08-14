import time
import config
import requests
import torch
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error



transform = torch.hub.load("intel-isl/MiDaS", "transforms")
model_type = config.model_type
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()
def estimate_depth_latency(frame):
    start_time = time.time()

    # Preprocess the frame
    input_batch = transform(frame).unsqueeze(0)

    # Move the input and model to GPU for faster computation
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Generate depth map
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Calculate latency
    latency = time.time() - start_time
    

    return latency,prediction
def calculate_rmse(ground_truth, prediction):
    return np.sqrt(mean_squared_error(ground_truth, prediction))

video_path=config.Video_Path
video = cv2.VideoCapture(video_path)

gt_dm_video=cv2.VideoCapture(config.gt_dm_video_path)
RMSE=[]
LATENCY=[]
while True:
        ret, frame = video.read()
        ret1,ground_truth =gt_dm_video.read()
        if not ret:
            break
        latency,prediction=estimate_depth_latency(frame)
        rmse = calculate_rmse(ground_truth, prediction)
        RMSE.append(rmse)
        LATENCY.append(latency)

print("Average Latency:",sum(LATENCY)/len(latency))
print("Average Root mean squared error:",sum(RMSE)/len(RMSE))

