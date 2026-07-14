import cv2
import numpy as np
import mediapipe as mp
import csv
import os
import time

print("Silent Speech Dataset Collector")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
CSV_PATH = os.path.join(BASE_DIR, "data.csv")

print("Model:", MODEL_PATH)
print("CSV:", CSV_PATH)

WORDS = {
    ord('h'): "hello",
    ord('y'): "yes",
    ord('n'): "no",
    ord('g'): "guru",
    ord('e'): "heena",
    ord('t'): "thanks",
    ord('p'): "please",
    ord('s'): "stop",
    ord('o'): "okay",
    ord('w'): "water",
}

SAVE_INTERVAL = 0.30

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["height", "width", "label"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("""
Press once to select a word.

H hello
Y yes
N no
G guru
E heena
T thanks
P please
S stop
O okay
W water

ESC Exit
""")

current = None
counts = {}
frame_id = 0
last_save = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, frame_id)
    frame_id += 1

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        H, W = frame.shape[:2]

        top = np.array([lm[13].x * W, lm[13].y * H])
        bottom = np.array([lm[14].x * W, lm[14].y * H])
        left = np.array([lm[61].x * W, lm[61].y * H])
        right = np.array([lm[291].x * W, lm[291].y * H])

        height = np.linalg.norm(top - bottom)
        width = np.linalg.norm(left - right)

        cv2.putText(frame, f"Word : {current if current else 'None'}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Height : {height:.2f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Width : {width:.2f}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if current:
            cv2.putText(frame, f"Samples : {counts.get(current,0)}/200",
                        (20,140), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            now = time.time()
            if now - last_save >= SAVE_INTERVAL:
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([round(height,2), round(width,2), current])
                    f.flush()
                    os.fsync(f.fileno())

                counts[current] = counts.get(current,0) + 1
                last_save = now
                print(f"Saved {counts[current]} -> {current}")

    cv2.imshow("Silent Speech Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key in WORDS:
        current = WORDS[key]
        counts.setdefault(current, 0)
        print("Current word:", current)

cap.release()
cv2.destroyAllWindows()
