import cv2
import torch
import numpy as np
import timm
import config
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load MiDaS model
model_type = config.model_type
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# Load transforms to preprocess input images
transform = torch.hub.load("intel-isl/MiDaS", "transforms")

def estimate_depth(frame):
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

    # Convert prediction to numpy array and normalize
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    return depth_map
