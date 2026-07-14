from flask import Flask, jsonify, request
from flask_cors import CORS
import base64

from model_loader import load_models
from feature_extractor import init_extractor
from predict import process_prediction, reset_session

# =====================================
# Flask Setup
# =====================================
app = Flask(__name__)
CORS(app)

is_processing_request = False

# =====================================
# Startup Initialization
# =====================================
print("Initializing Silent Speech AI Backend...")
load_models()
init_extractor()
print("Initialization Complete.")

# =====================================
# Routes
# =====================================

@app.route("/")
def home():
    """Health check route to verify backend status."""
    return "Silent Speech AI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives base64 image data from frontend, processes it,
    and returns the predicted word and confidence.
    """
    global is_processing_request
    if is_processing_request:
        return jsonify({
            "word": "Processing...",
            "confidence": 0
        })
    is_processing_request = True
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({
                "word": "No Image",
                "confidence": 0
            })

        image_data = data["image"]

        # Clean base64 header if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        # Process prediction using modularized logic
        result = process_prediction(image_bytes)

        return jsonify(result)

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({
            "word": "Error",
            "confidence": 0
        })
    finally:
        is_processing_request = False

@app.route("/reset", methods=["POST"])
def reset():
    """Resets the prediction history for a new session."""
    try:
        result = reset_session()
        return jsonify(result)
    except Exception as e:
        print("Reset Error:", e)
        return jsonify({"status": "error"})

# =====================================
# Run
# =====================================
if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False,
        threaded=True
    )