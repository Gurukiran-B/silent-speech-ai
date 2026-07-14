import numpy as np
import pandas as pd
import os

MAX_HEIGHT = 100.0
MAX_WIDTH = 100.0

# Try to load exact max values from dataset if it exists
try:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "enhanced_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "enhanced_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        MAX_HEIGHT = df["height"].max()
        MAX_WIDTH = df["width"].max()
except:
    pass

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)

def extract_features(landmarks, frame_width, frame_height):
    """
    Extracts the 7 features expected by the KNN model:
    height, width, mar, ratio, area, height_norm, width_norm
    """
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[61]
    right = landmarks[291]

    top_point = np.array([top.x * frame_width, top.y * frame_height])
    bottom_point = np.array([bottom.x * frame_width, bottom.y * frame_height])
    left_point = np.array([left.x * frame_width, left.y * frame_height])
    right_point = np.array([right.x * frame_width, right.y * frame_height])

    height = calculate_distance(top_point, bottom_point)
    width = calculate_distance(left_point, right_point)

    epsilon = 1e-6
    mar = height / (width + epsilon)
    ratio = width / (height + epsilon)
    area = height * width

    height_norm = height / MAX_HEIGHT
    width_norm = width / MAX_WIDTH

    return np.array([[height, width, mar, ratio, area, height_norm, width_norm]])
