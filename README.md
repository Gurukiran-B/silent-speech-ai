# 🧠 Silent Speech AI

> Convert lip movements into speech using Computer Vision and AI

---

## 🚀 Overview

Silent Speech AI is a real-time system that captures lip movements from a webcam and converts them into meaningful text and speech.

This project explores **silent human-computer interaction**, enabling communication without vocal sound.

---

## 🎯 Features

* 👄 Real-time lip tracking using MediaPipe
* 📊 Mouth feature extraction (width & height)
* 🧠 Word prediction (hello, yes, no)
* 🔊 Text-to-Speech output
* 🎥 Live webcam processing

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* Flask (for backend integration)

---

## 🧩 Project Structure

```
silent-speech-ai/
│
├── backend/
│   ├── lip_tracking.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── dataset/
├── models/
├── outputs/
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/silent-speech-ai.git
cd silent-speech-ai
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
cd backend
python lip_tracking.py
```

* Webcam will open
* System detects lip movement
* Displays mouth features in real-time

---

## 📌 Future Improvements

* 🔥 Deep learning-based lip reading
* 🎭 Emotion detection from facial expressions
* 🌐 Web-based interface for real-time usage
* 📱 Mobile app integration

---

## 🌍 Real-World Applications

* Accessibility for speech-impaired individuals
* Silent communication in restricted environments
* AI-based human-computer interaction
* Video dubbing and animation systems

---

## 👨‍💻 Author

**Gurukiran B**

---

## ⭐ Contribute

Feel free to fork this repo and improve the system.

---

## 📜 License

This project is open-source and free to use.
