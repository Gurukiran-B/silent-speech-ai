import mediapipe as mp
import os

_face_landmarker = None

def init_extractor():
    """Initializes the MediaPipe Face Landmarker once."""
    global _face_landmarker
    
    if _face_landmarker is not None:
        return _face_landmarker
        
    try:
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        
        # Look for the task file in the backend/models directory
        model_path = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1
        )
        
        _face_landmarker = FaceLandmarker.create_from_options(options)
        print("MediaPipe Face Landmarker loaded successfully.")
    except Exception as e:
        print(f"Error loading MediaPipe: {e}")
        
    return _face_landmarker

def get_landmarks(mp_image):
    """Detect face landmarks for a given MediaPipe Image."""
    if _face_landmarker is None:
        init_extractor()
        
    if _face_landmarker is None:
        return None
        
    result = _face_landmarker.detect(mp_image)
    
    if not result.face_landmarks:
        return None
        
    return result.face_landmarks[0]
