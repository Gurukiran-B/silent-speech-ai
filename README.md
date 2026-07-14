# Silent Speech AI

## Project Overview
Silent Speech AI is a Version 1 proof-of-concept system that predicts spoken words from silent lip movements. It uses facial landmark detection and a K-Nearest Neighbors (KNN) model to decode silent speech from video frames. This project is a foundational prototype designed to explore the feasibility of visual speech recognition using a custom dataset and classical machine learning techniques.

## Features
- **Real-Time Lip Tracking:** Uses MediaPipe for highly accurate facial landmark detection.
- **Silent Speech Prediction:** Decodes 10 distinct words from lip movements in real time.
- **Custom ML Pipeline:** End-to-end pipeline covering dataset collection, feature engineering, and model training.
- **Interactive Web Interface:** User-friendly frontend to visualize predictions and prediction history.

## Architecture
The system follows a client-server architecture:
1. **Frontend:** Captures webcam video, converts frames to base64, and sends them to the backend via HTTP POST.
2. **Backend (Flask):** Receives frames, extracts facial landmarks using MediaPipe, computes engineered features, and predicts the word using a trained KNN model.
3. **Response:** The backend returns the predicted word and confidence score to the frontend for display.

## Folder Structure
```
Silent-Speech-AI/
├── backend/
│   ├── models/
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   └── face_landmarker.task
│   ├── app.py
│   ├── predict.py
│   ├── feature_extractor.py
│   ├── model_loader.py
│   ├── utils.py
│   ├── feature_engineering.py
│   ├── analyze_dataset.py
│   ├── visualize_dataset.py
│   ├── train_knn.py
│   └── lip_tracking.py
├── frontend/
│   └── index.html
├── dataset/
│   ├── data.csv
│   └── enhanced_data.csv
├── docs/
│   └── images/
├── requirements.txt
├── README.md
└── .gitignore
```

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask, Flask-CORS
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** scikit-learn, pandas, numpy
- **Data Visualization:** matplotlib, seaborn

## Dataset Collection
The project uses a custom-built dataset collected using the `lip_tracking.py` script. The dataset focuses on a **10-word vocabulary** (hello, yes, no, guru, heena, thanks, please, stop, okay, water). We captured multiple samples for each word to train the initial model.

## Feature Engineering
Facial landmarks are processed to extract 7 key features per frame:
- Mouth Height
- Mouth Width
- Mouth Aspect Ratio (MAR)
- Width-to-Height Ratio
- Mouth Area
- Normalized Height
- Normalized Width

## Machine Learning Pipeline
The ML pipeline is intentionally simple as a proof of concept:
1. **Data Preprocessing:** Missing values are handled, and features are standardized using `StandardScaler`.
2. **Model Training:** A K-Nearest Neighbors (KNN) classifier is trained to map the 7 extracted features to the 10-word vocabulary.
3. **Evaluation:** The optimal `K` value is selected based on cross-validation accuracy.

## Real-Time Prediction Pipeline
1. The frontend sends continuous video frames to the backend.
2. The backend extracts the 7 mouth features using MediaPipe.
3. A sliding window mechanism tracks the last few frames to ensure stable predictions.
4. The KNN model outputs a prediction, which is returned to the user if the confidence threshold is met.

## Results
- **Accuracy:** The current KNN proof of concept achieves approximately **50% accuracy**.
- **Performance:** This is an expected baseline for a simple frame-based, non-temporal model trained on a small, custom dataset.

## Current Limitations
- **Frame-Based Prediction:** The model evaluates individual frames without understanding the temporal sequence of lips moving over time.
- **Limited Vocabulary:** The system can only recognize 10 specific words.
- **Custom Dataset:** The dataset is small and lacks the diversity required for generalized, robust performance.
- **KNN Limitations:** KNN is a simplistic model for this complex task and struggles with overlapping feature distributions.

## Future Improvements
To move beyond this Version 1 prototype, future iterations could include:
- Implementing **LSTM**, **CNN**, or **Transformer** models for proper temporal sequence modeling.
- Expanding the dataset significantly to include more words and diverse speakers.
- Improving the feature extraction to capture finer lip and jaw dynamics.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/Silent-Speech-AI.git
   cd Silent-Speech-AI
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask backend:
   ```bash
   python backend/app.py
   ```
2. Open the frontend:
   Open `frontend/index.html` in your web browser.
3. Grant camera permissions and start silent speaking the supported words!

## License
MIT License
