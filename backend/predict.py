import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import pandas as pd

from feature_extractor import get_landmarks
from utils import extract_features
from model_loader import get_model, get_scaler, get_label_encoder

# =====================================
# Constants & Thresholds
# =====================================
WINDOW_SIZE = 7
MIN_CONFIDENCE = 60.0
MOVEMENT_THRESHOLD = 1.0 
SILENCE_FRAMES_THRESHOLD = 5

# =====================================
# State Variables
# =====================================
features_window = deque(maxlen=WINDOW_SIZE)
movement_history = deque(maxlen=5)
previous_mouth_height = 0
is_speaking = False
silence_frames = 0

# =====================================
# History & Stats
# =====================================
history = deque(maxlen=20)
total_predictions = 0
confidence_sum = 0

def process_prediction(image_bytes):
    """
    Processes the raw image bytes from the frontend, extracts features,
    makes a prediction using the trained KNN model, and manages state.
    """
    global features_window, movement_history, previous_mouth_height
    global is_speaking, silence_frames, history, total_predictions, confidence_sum
    
    t0 = time.perf_counter()

    # 1. Decode Image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"word": "Error decoding image", "confidence": 0}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    t1 = time.perf_counter()

    # 2. Extract Landmarks
    landmarks = get_landmarks(mp_image)
    t2 = time.perf_counter()
    
    avg_conf = confidence_sum / total_predictions if total_predictions > 0 else 0

    if not landmarks:
        return {
            "word": "No Face Detected",
            "confidence": 0,
            "history": list(history),
            "spoken": False,
            "total_predictions": total_predictions,
            "average_confidence": round(avg_conf, 2)
        }

    h_img, w_img, _ = frame.shape

    # 3. Extract Features
    features = extract_features(landmarks, w_img, h_img)
    t3 = time.perf_counter()
    
    mouth_height = features[0][0]

    # Speech Detection Logic
    if previous_mouth_height > 0:
        movement = abs(mouth_height - previous_mouth_height)
        movement_history.append(movement)
    previous_mouth_height = mouth_height

    total_movement = sum(movement_history)

    if total_movement >= MOVEMENT_THRESHOLD:
        is_speaking = True
        silence_frames = 0
    else:
        if is_speaking:
            silence_frames += 1
            if silence_frames >= SILENCE_FRAMES_THRESHOLD:
                is_speaking = False
                features_window.clear()

    if not is_speaking:
        return {
            "word": "Waiting for speech...",
            "confidence": 0,
            "history": list(history),
            "spoken": False,
            "total_predictions": total_predictions,
            "average_confidence": round(avg_conf, 2)
        }

    # If speaking, accumulate features
    features_window.append(features[0])
    
    # 4. Scaling & Prediction
    model = get_model()
    scaler = get_scaler()
    label_encoder = get_label_encoder()
    
    if not model or not scaler or not label_encoder:
        return {"word": "Model not loaded", "confidence": 0}

    feature_names = getattr(scaler, "feature_names_in_", [f"feature_{i}" for i in range(len(features_window[0]))])
    features_df = pd.DataFrame(list(features_window), columns=feature_names)
    scaled_window = scaler.transform(features_df)
    t4 = time.perf_counter()
    
    # Use predict_proba for average probabilities across window
    try:
        probabilities = model.predict_proba(scaled_window)
        avg_probs = np.mean(probabilities, axis=0)
        max_prob_index = np.argmax(avg_probs)
        confidence = avg_probs[max_prob_index] * 100
        best_label_idx = max_prob_index
    except AttributeError:
        # Fallback if predict_proba is not available
        predictions = model.predict(scaled_window)
        best_label_idx = np.bincount(predictions).argmax()
        confidence = 80.0
        
    t5 = time.perf_counter()

    # 5. Stable Prediction Logic
    best_label = label_encoder.inverse_transform([best_label_idx])[0]

    if confidence < MIN_CONFIDENCE:
        final_word = "Listening..."
    else:
        final_word = best_label

    # 6. History Update
    if final_word not in ["Waiting for speech...", "Listening..."]:
        if len(history) == 0 or history[-1] != final_word:
            history.append(final_word)

    # 7. Stats Update
    total_predictions += 1
    confidence_sum += confidence
    avg_confidence = confidence_sum / total_predictions

    # Profiling Output
    print(f"Decode time: {(t1-t0)*1000:.2f} ms")
    print(f"MediaPipe time: {(t2-t1)*1000:.2f} ms")
    print(f"Feature extraction: {(t3-t2)*1000:.2f} ms")
    print(f"Scaling: {(t4-t3)*1000:.2f} ms")
    print(f"Prediction: {(t5-t4)*1000:.2f} ms")
    print(f"Total time: {(t5-t0)*1000:.2f} ms")

    return {
        "word": final_word,
        "confidence": round(confidence, 2),
        "history": list(history),
        "spoken": False,
        "total_predictions": total_predictions,
        "average_confidence": round(avg_confidence, 2)
    }

def reset_session():
    """Resets the prediction history and statistics."""
    global features_window, movement_history, previous_mouth_height
    global is_speaking, silence_frames, history, total_predictions, confidence_sum
    
    features_window.clear()
    movement_history.clear()
    previous_mouth_height = 0
    is_speaking = False
    silence_frames = 0
    history.clear()
    total_predictions = 0
    confidence_sum = 0
    
    return {"status": "success", "message": "Session reset"}