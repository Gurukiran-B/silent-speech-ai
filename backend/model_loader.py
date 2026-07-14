import joblib
import os

_model = None
_scaler = None
_label_encoder = None

def load_models():
    """Loads the KNN model, scaler, and label encoder once during startup."""
    global _model, _scaler, _label_encoder
    
    if _model is not None:
        return _model, _scaler, _label_encoder
        
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    try:
        _model = joblib.load(os.path.join(models_dir, "model.pkl"))
        _scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        _label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        
    return _model, _scaler, _label_encoder

def get_model():
    """Returns the loaded KNN model."""
    return _model

def get_scaler():
    """Returns the loaded feature scaler."""
    return _scaler

def get_label_encoder():
    """Returns the loaded label encoder."""
    return _label_encoder
